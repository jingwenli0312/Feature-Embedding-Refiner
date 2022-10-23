import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from torch.autograd import Variable
from torch.nn import DataParallel
from utils.utils import get_inner_model, torch_load_cpu

from utils import clip_grad_norms
from nets.actor import ACTOR_RECONSTRUCT
from nets.critic_network import Critic
from utils import torch_load_cpu, get_inner_model, move_to, input_feature_encoding
from utils.logger import log_to_tb_train
from agent.utils import validate
from utils.utils import augment_xy_data_by_8_fold
from utils.plots import plot_tour_vrp
import numpy as np
from travelling_saleman_problem import GROUP_STATE, GROUP_ENVIRONMENT
from torch.utils.checkpoint import checkpoint


class A2C:
    def __init__(self, problem_name, opts, problem):

        # figure out the options
        self.opts = opts
        self.problem_name = problem_name
        self.problem = problem

        # figure out the actor
        self.actor = ACTOR_RECONSTRUCT(self.opts).to(opts.device)

        if not opts.eval_only:
            # figure out the critic
            self.critic = Critic(
                embedding_dim=opts.embedding_dim,
                hidden_dim=opts.hidden_dim,
                n_heads=opts.critic_head_num,
                graph_size=opts.graph_size,
            ).to(opts.device)

            # figure out the optimizer
            self.optimizer = optim.Adam(
                [{'params': self.actor.parameters(), 'lr': opts.lr_model}] +
                [{'params': self.critic.parameters(), 'lr': opts.lr_critic}])

            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: opts.lr_decay ** (epoch))

        if opts.use_cuda and torch.cuda.device_count() > 1:
            self.actor = torch.nn.DataParallel(self.actor)
            if not opts.eval_only:
                self.critic = torch.nn.DataParallel(self.critic)
                # self.critic = self.critic.to(opts.device)
                print('not eval_only', opts.device)

        # self.actor = self.actor.to(opts.device)

    def load_POMO(self, load_POMO_PATH):

        # Load data from load_path
        load_data = {}
        if load_POMO_PATH is not None:
            print('  [*] Loading data from {}'.format(load_POMO_PATH))
            load_data = torch_load_cpu(load_POMO_PATH)

        # Overwrite model parameters by parameters to load
        if isinstance(self.actor, DataParallel):
            model_ = get_inner_model(self.actor.module.POMO)
        else:
            model_ = get_inner_model(self.actor.POMO)
        model_.load_state_dict({**model_.state_dict(), **load_data})

    def load(self, load_path):

        assert load_path is not None
        load_data = torch_load_cpu(load_path)
        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})

        if not self.opts.eval_only:
            # load data for critic
            model_critic = get_inner_model(self.critic)
            model_critic.load_state_dict({**model_critic.state_dict(), **load_data.get('critic', {})})
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.opts.device)
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # done
        print(' [*] Loading data from {}'.format(load_path))

    def save(self, epoch):
        print('Saving model and state...')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic': get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    def eval(self):
        self.actor.eval()
        #         self.actor.AM.eval()
        if not self.opts.eval_only: self.critic.eval()

    def train(self):
        self.actor.train()
        #         self.actor.AM.train()
        if not self.opts.eval_only: self.critic.train()

    def rollout(self, problem, batch, opts, epoch, cri, record=False):

        # prepare the features
        batch = move_to(batch, opts.device)

        dist = []
        if problem.NAME == 'tsp':
            if opts.augment:
                batch = augment_xy_data_by_8_fold(batch)
            batch_size = batch.size(0)

            for i, instance in enumerate(batch):
                dist.append(calculate_distance(batch[i]))

        if problem.NAME == 'pdp' or problem.NAME == 'cvrp':
            batch['depot'] = batch['depot'][:, None, :]
            if opts.augment:
                batch['depot'] = augment_xy_data_by_8_fold(batch['depot'])
                batch['loc'] = augment_xy_data_by_8_fold(batch['loc'])

                # data = torch.cat([batch['depot'].reshape(-1, 1, 2), batch['loc']], dim=1)  # bs, graph_size, 2
                # batch['depot'] = (augment_xy_data_by_8_fold(data)[:, 0, :])[:, None, :]
                # batch['loc'] = augment_xy_data_by_8_fold(data)[:, 1:, :]

                batch['demand'] = batch['demand'].repeat(8, 1)
            batch_size = batch['depot'].size(0)

            for i, instance in enumerate(batch['depot']):
                dist.append(calculate_distance(torch.cat([batch['depot'].reshape(-1, 1, 2), batch['loc']], dim=1)[i]))
        dist = torch.stack(dist, 0)  # [bs, gs, gs]

        if isinstance(self.actor, DataParallel):
            POMO = self.actor.module.POMO
        else:
            POMO = self.actor.POMO

        with torch.no_grad():
            if opts.problem == 'tsp':
                POMO_embedding = POMO.encoder(batch).detach()
            elif opts.problem == 'cvrp':
                all_node_xy = torch.cat((batch['depot'], batch['loc']), dim=1)  # (batch, problem+1, 2)
                depot_demand = torch.zeros((batch_size, 1, 1), device=opts.device)
                all_node_demand = torch.cat((depot_demand, batch['demand'].unsqueeze(-1)),
                                            dim=1)  # (batch, problem+1, 1)
                data = torch.cat((all_node_xy, all_node_demand), dim=2)
                POMO_embedding = POMO.encoder(data).detach()
                # plot_tour_vrp(opts.problem, all_node_xy)

            group_s = opts.graph_size
            solu_embed = None

            history_solutions = []
            history_costs = []
            if opts.init_val_met == 'nearest':
                solution = move_to(problem.get_initial_solutions(batch_size, batch, dist, opts.init_val_met), opts.device)
                best_so_far = problem.get_costs(batch, solution, multi_solu=False)
                history_solutions.append(solution)
                history_costs.append(best_so_far)
                history_solutions = torch.stack(history_solutions, 0).transpose(0, 1)  # bs, num_solu, seq_len
                history_costs = torch.stack(history_costs, 0).transpose(0, 1)  # bs, num_solu
                best_so_far = best_so_far[:, None].repeat(1, group_s)

            else:
                # solution = move_to(problem.get_initial_solutions(batch_size, batch, dist, 'nearest'),
                #                    opts.device)
                # best_so_far = problem.get_costs(batch, solution, multi_solu=False)
                # history_solutions.append(solution)
                # history_costs.append(best_so_far)
                # history_solutions = torch.stack(history_solutions, 0).transpose(0, 1)  # bs, num_solu, seq_len
                # history_costs = torch.stack(history_costs, 0).transpose(0, 1)  # bs, num_solu
                # best_so_far = best_so_far[:, None].repeat(1, group_s)
                best_so_far = 1e5 * torch.ones(batch_size, device=opts.device).unsqueeze(-1).repeat(1, group_s)  # bs, gs, 1

                for _ in range(2):
                    solution, reward_step, _ = POMO(batch, POMO_embedding)
                    index = torch.randint(0, group_s, (batch_size, opts.K), device=opts.device)

                    history_solutions = solution.gather(1, index.unsqueeze(-1).expand(batch_size, opts.K,
                                                                                      solution.size(-1)))
                    history_costs = -reward_step.gather(1, index)  # bs, num_solu
                    best_for_now = torch.cat((best_so_far[None, :], (-reward_step)[None, :]), 0).min(0)[0]
                    best_so_far = best_for_now

            obj_history = [best_so_far.mean(-1)]
            best_cost_history = [torch.min(best_so_far, -1)[0]]
            reward = []
            solution_history = []

            for t in tqdm(range(self.opts.T_max), disable=self.opts.no_progress_bar, desc='rollout',
                          bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):

                solution, reward_step, _, _, solu_embed = self.actor(batch, POMO_embedding,
                                                                                 dist, solu_embed,
                                                                                 history_solutions,
                                                                                 history_costs
                                                                                 )

                if opts.K >= opts.graph_size:
                    history_solutions = solution  # bs, gs, tsp_size
                    history_costs = -reward_step
                else:
                    r = np.random.uniform(low=1, high=opts.epoch_end + 1, size=1)
                    if r < cri[epoch]:
                        index = torch.randint(0, group_s, (batch_size, opts.K - 1), device=opts.device)
                        history_solutions = solution.gather(1, index.unsqueeze(-1).expand(batch_size, opts.K - 1,
                                                                                          solution.size(-1)))
                        history_costs = -reward_step.gather(1, index)

                        # add a random solution for diversity
                        rand_solution = move_to(
                            problem.get_initial_solutions(batch_size, batch, dist, opts.init_val_met), opts.device)  # (bs, seq_len)
                        rand_cost = problem.get_costs(batch, rand_solution, multi_solu=False)
                        history_solutions = torch.cat((rand_solution.unsqueeze(1), history_solutions), 1)
                        history_costs = torch.cat((rand_cost.unsqueeze(1), history_costs), 1)
                    else:
                        index = torch.randint(0, group_s, (batch_size, opts.K), device=opts.device)
                        history_solutions = solution.gather(1, index.unsqueeze(-1).expand(batch_size, opts.K,
                                                                                          solution.size(-1)))
                        history_costs = -reward_step.gather(1, index)

                best_for_now = torch.cat((best_so_far[None, :], (-reward_step)[None, :]), 0).min(0)[0]  # (bs, gs)
                rewards = best_so_far - best_for_now  # >= 0
                best_so_far = best_for_now

                # record informations
                reward.append(rewards.mean(dim=1))
                obj_history.append((-reward_step).mean(-1))
                best_cost_history.append(torch.min(best_so_far, -1)[0])

                if record:
                    solution_history.append(solution)

        # action_prob_all = torch.stack(action_prob_all, 0).view(1, -1)  # T, graph_size
        # np.savetxt('action_prob50_ours.csv', action_prob_all.cpu().numpy(), delimiter=',')

        out = (torch.min(best_so_far, -1)[0],  # best_cost: batch_size
               torch.stack(obj_history, 1),  # batch_size, T
               torch.stack(best_cost_history, 1),  # batch_size, T
               torch.stack(reward, 1),  # batch_size, T
               None if not record else torch.stack(solution_history, 1))

        return out

    def start_training(self, problem, val_dataset, tb_logger):
        train(problem, self, val_dataset, tb_logger, self.problem_name)


def train(problem, agent, val_dataset, tb_logger, problem_name):
    opts = agent.opts
    # cri = np.logspace(np.log(opts.epoch_start + 1), np.log(opts.epoch_end -20 + 1), opts.epoch_end -20, base=np.exp(1))
    cri = np.logspace(np.log(opts.epoch_start + 1), np.log(opts.epoch_end + 1), opts.epoch_end, base=np.exp(1))
    # validate(problem, agent, val_dataset, tb_logger, epoch=0)

    # Start the actual training loop
    for epoch in range(opts.epoch_start, opts.epoch_end):
        # Training mode
        print('\n\n')
        print("|", format(f" Training epoch {epoch} ", "*^60"), "|")
        agent.train()
        problem.train()
        opts = agent.opts
        # lr_scheduler
        agent.lr_scheduler.step(epoch)
        print("Training with lr={:.3e} for run {}".format(agent.optimizer.param_groups[0]['lr'], opts.run_name),
              flush=True)
        step = epoch * (opts.epoch_size // opts.batch_size)

        # Generate new training data for each epoch
        training_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size)
        training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size)

        # start training
        pbar = tqdm(total=(opts.epoch_size // opts.batch_size) * (opts.T_train // opts.n_step),
                    disable=opts.no_progress_bar, desc=f'training',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

        for batch_id, batch in enumerate(training_dataloader):
            train_batch(problem,
                        problem_name,
                        agent,
                        epoch,
                        batch_id,
                        step,
                        batch,
                        tb_logger,
                        opts,
                        pbar,
                        cri)
            step += 1
        pbar.close()

        # save new model
        if (not opts.no_saving and opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                epoch == opts.n_epochs - 1: agent.save(epoch)

        # validate the new model
        validate(problem, agent, val_dataset, tb_logger, epoch, cri)


def calculate_distance(data):
    N_data = data.shape[0]
    dists = torch.zeros((N_data, N_data), dtype=torch.float)
    d1 = -2 * torch.mm(data, data.T)
    d2 = torch.sum(torch.pow(data, 2), dim=1)
    d3 = torch.sum(torch.pow(data, 2), dim=1).reshape(1, -1).T
    dists = d1 + d2 + d3
    dists[dists < 0] = 0
    return torch.sqrt(dists)


def train_batch(
        problem,
        problem_name,
        agent,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts,
        pbar,
        cri
):
    # prepare the features
    batch = move_to(batch, opts.device)  # batch_size, graph_size, 2

    dist = []
    if problem.NAME == 'tsp':
        batch_size = batch.size(0)
        for i, instance in enumerate(batch):
            dist.append(calculate_distance(batch[i]))
    if problem.NAME == 'pdp' or problem.NAME == 'cvrp':
        batch_size = batch['depot'].size(0)
        for i, instance in enumerate(batch['depot']):
            dist.append(calculate_distance(torch.cat([batch['depot'].reshape(-1, 1, 2), batch['loc']], dim=1)[i]))
    dist = torch.stack(dist, 0)  # [bs, gs, gs]

    if isinstance(agent.actor, DataParallel):
        POMO = agent.actor.module.POMO
    else:
        POMO = agent.actor.POMO

    with torch.no_grad():
        if opts.problem == 'tsp':
            POMO_embedding = POMO.encoder(batch).detach()
        elif opts.problem == 'cvrp':
            batch['depot'] = batch['depot'][:, None, :]
            all_node_xy = torch.cat((batch['depot'], batch['loc']), dim=1)  # (batch, problem+1, 2)
            depot_demand = torch.zeros((batch_size, 1, 1), device=opts.device)
            all_node_demand = torch.cat((depot_demand, batch['demand'].unsqueeze(-1)), dim=1)  # (batch, problem+1, 1)
            data = torch.cat((all_node_xy, all_node_demand), dim=2)
            POMO_embedding = POMO.encoder(data).detach()

    solu_embed = None
    group_s = opts.graph_size

    # init_solution = move_to(problem.get_initial_solutions(batch.size(0), batch, dist), opts.device)
    # best_so_far = problem.get_costs(batch, init_solution, multi_solu=False).unsqueeze(-1).repeat(1, group_s)# bs, gs, 1
    best_so_far = 1e5 * torch.ones(batch_size, device=opts.device).unsqueeze(-1).repeat(1, group_s)  # bs, gs, 1

    with torch.no_grad():
        # for _ in range(1):
        solution, reward_step, _ = POMO(batch, POMO_embedding)
        index = torch.randint(0, group_s, (batch_size, opts.K), device=opts.device)
        history_solutions = solution.gather(1, index.unsqueeze(-1).expand(batch_size, opts.K,
                                                                          solution.size(-1)))
        history_costs = -reward_step.gather(1, index)  # bs, num_solu
        best_for_now = torch.cat((best_so_far[None, :], (-reward_step)[None, :]), 0).min(0)[0]
        best_so_far = best_for_now

        # POMO_embedding_new = agent.actor(batch, POMO_embedding,
        #                                          dist, solu_embed,
        #                                          history_solutions,
        #                                          history_costs,
        #                                          embedding_only=True
        #                                          )
        #
        # agent.eval()
        # solution, reward_step, _ = POMO(batch, POMO_embedding_new)
        # best_for_now = torch.cat((best_so_far[None, :], (-reward_step)[None, :]), 0).min(0)[0]
        # best_so_far = best_for_now

    agent.train()

    gamma = opts.gamma
    n_step = opts.n_step
    T = opts.T_train
    t = 0

    while t < T:  # T 200, n = 4, 50 gradient steps
        # empty array
        total_cost = 0
        baseline_val = []
        baseline_val_detached = []
        log_likelihood = []
        reward = []
        t_s = t

        while t - t_s < n_step and not (t == T):
            solution, reward_step, group_prob_list, embedding_new, solu_embed = agent.actor(batch,
                                                                                POMO_embedding,
                                                                                dist,
                                                                                solu_embed,
                                                                                history_solutions,
                                                                                history_costs
                                                                                )

            if opts.K >= opts.graph_size:
                history_solutions = solution  # bs, gs, tsp_size
                history_costs = -reward_step
            else:
                # r = np.random.uniform(low=1, high=opts.epoch_end -20 + 1, size=1)
                # if r < cri[epoch -20]:
                r = np.random.uniform(low=1, high=opts.epoch_end + 1, size=1)
                if r < cri[epoch]:
                    index = torch.randint(0, group_s, (batch_size, opts.K - 1), device=opts.device)
                    history_solutions = solution.gather(1, index.unsqueeze(-1).expand(batch_size, opts.K - 1,
                                                                                      solution.size(-1)))
                    history_costs = -reward_step.gather(1, index)

                    # add a random solution for diversity
                    rand_solution = move_to(
                        problem.get_initial_solutions(batch_size, batch, dist, opts.init_val_met), opts.device)  # (bs, seq_len)
                    rand_cost = problem.get_costs(batch, rand_solution, multi_solu=False)

                    history_solutions = torch.cat((rand_solution.unsqueeze(1), history_solutions), 1)
                    history_costs = torch.cat((rand_cost.unsqueeze(1), history_costs), 1)
                else:
                    index = torch.randint(0, group_s, (batch_size, opts.K), device=opts.device)
                    history_solutions = solution.gather(1, index.unsqueeze(-1).expand(batch_size, opts.K,
                                                                                      solution.size(-1)))
                    history_costs = -reward_step.gather(1, index)

            bl_val_detached, bl_val = agent.critic(embedding_new)

            # get estimated value from baseline
            baseline_val_detached.append(bl_val_detached)
            baseline_val.append(bl_val)

            group_log_prob = group_prob_list.log().sum(dim=2)  # (bs, gs)
            log_likelihood.append(group_log_prob)  # (bs, gs)

            best_for_now = torch.cat((best_so_far[None, :], (-reward_step)[None, :]), 0).min(0)[0]  # (bs, gs)
            reward.append((best_so_far - best_for_now))  # bs, gs
            best_so_far = best_for_now

            # state transient
            total_cost = total_cost + ((-reward_step).mean(-1))

            # next
            t = t + 1

        solu_embed = Variable(solu_embed.data)

        # reward
        total_cost = total_cost / (t - t_s)
        reward_reversed = reward[::-1]  # n_step, bs, gs

        Reward = []
        next_return, _ = agent.critic(embedding_new)
        for r in range(len(reward_reversed)):
            if r == 0:
                R = (next_return * gamma).unsqueeze(-1) + reward_reversed[r]
            else:
                R = next_return * gamma + reward_reversed[r]
            Reward.append(R)
            next_return = R

        # torch.Size([T, batch_size])
        Reward = torch.stack(Reward[::-1], 0)  # n_step, bs, gs
        baseline_val = torch.stack(baseline_val, 0).unsqueeze(-1)
        baseline_val_detached = torch.stack(baseline_val_detached, 0).unsqueeze(-1)
        log_likelihood = torch.stack(log_likelihood, 0)  # n_step, bs, gs

        # calculate loss
        criteria = torch.nn.MSELoss()
        baseline_loss = criteria(baseline_val, Reward)
        reinforce_loss = - ((Reward - baseline_val_detached) * log_likelihood).mean()
        loss = baseline_loss + reinforce_loss

        # update gradient step
        agent.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(agent.optimizer.param_groups, opts.max_grad_norm)

        agent.optimizer.step()

        # Logging to tensorboard
        if (not opts.no_tb):
            current_step = int(step * T / n_step + t // n_step)
            if current_step % int(opts.log_step) == 0:
                log_to_tb_train(tb_logger, agent, total_cost, grad_norms, reward, Reward,
                                reinforce_loss, baseline_loss, log_likelihood,
                                current_step)

        pbar.update(1)
