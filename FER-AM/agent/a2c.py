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
from nets.actor_network import Actor, set_decode_type
from nets.critic_network import Critic
from utils import torch_load_cpu, get_inner_model, move_to, input_feature_encoding
from utils.logger import log_to_tb_train
from agent.utils import validate
from utils.utils import augment_xy_data_by_8_fold
# from utils.plots import plot_tour_tsp
import numpy as np
import random
from torch.utils.checkpoint import checkpoint


class A2C:
    def __init__(self, problem_name, opts, problem):

        # figure out the options
        self.opts = opts
        self.problem_name = problem_name
        self.problem = problem

        # figure out the actor
        self.actor = Actor(
            problem=problem,
            embedding_dim=opts.embedding_dim,
            hidden_dim=opts.hidden_dim,
            hidden_edge_dim=opts.hidden_edge_dim,
            alpha=opts.alpha,
            feed_forward_dim=opts.feed_forward_dim,
            n_heads_encoder=opts.encoder_head_num,
            n_heads_decoder=opts.decoder_head_num,
            n_layers=opts.n_encode_layers,
            graph_size=opts.graph_size,
            normalization=opts.normalization,
            device=opts.device,
            opts=opts
        ).to(opts.device)

        if not opts.eval_only:
            # figure out the critic
            self.critic = Critic(
                problem_name=problem_name,
                embedding_dim=opts.embedding_dim,
                hidden_dim=opts.hidden_dim,
                n_heads=opts.critic_head_num,
                n_layers=opts.n_encode_layers,
                normalization=opts.normalization,
                graph_size=opts.graph_size,
                device=opts.device,
            ).to(opts.device)

            # figure out the optimizer
            self.optimizer = optim.Adam(
                [{'params': self.actor.parameters(), 'lr': opts.lr_model}] +
                [{'params': self.critic.parameters(), 'lr': opts.lr_critic}])

            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: opts.lr_decay ** (epoch))

        if opts.use_cuda and torch.cuda.device_count() > 1:
            self.actor = torch.nn.DataParallel(self.actor)
            if not opts.eval_only: self.critic = torch.nn.DataParallel(self.critic)

    def load_AM(self, load_AM_PATH):

        # Load data from load_path
        load_data = {}
        if load_AM_PATH is not None:
            print('  [*] Loading data from {}'.format(load_AM_PATH))
            load_data = torch_load_cpu(load_AM_PATH)

        # Overwrite model parameters by parameters to load

        if isinstance(self.actor, DataParallel):
            model_ = get_inner_model(self.actor.module.AM)
        else:
            model_ = get_inner_model(self.actor.AM)

        model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

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
        set_decode_type(self.actor, "sampling")
        if isinstance(self.actor, DataParallel):
            set_decode_type(self.actor.module.AM, "sampling")
        else:
            set_decode_type(self.actor.AM, "sampling")

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
            if opts.augment:
                batch['depot'] = augment_xy_data_by_8_fold(batch['depot'][:, None, :]).squeeze(1)
                batch['loc'] = augment_xy_data_by_8_fold(batch['loc'])
                batch['demand'] = batch['demand'].repeat(8, 1)
            batch_size = batch['depot'].size(0)
            for i, instance in enumerate(batch['depot']):
                dist.append(calculate_distance(torch.cat([batch['depot'].reshape(-1, 1, 2), batch['loc']], dim=1)[i]))

        dist = torch.stack(dist, 0)  # [bs, gs, gs]

        if isinstance(self.actor, DataParallel):
            AM = self.actor.module.AM
        else:
            AM = self.actor.AM

        AM_embedding, _ = AM.embedder(AM._init_embed(batch))
        solu_embed = None

        history_solutions = []
        history_costs = []
        if opts.init_val_met == 'nearest':
            solution = move_to(problem.get_initial_solutions(batch_size, batch, dist), opts.device)
            best_so_far = problem.get_costs(batch, solution, multi_solu=False)
            # history_solutions = solution.unsqueeze(0)
            # history_costs = best_so_far.unsqueeze(0)
            history_solutions.append(solution)
            history_costs.append(best_so_far)

        else:
            best_so_far = 1e5 * torch.ones(batch_size, device=opts.device)

            with torch.no_grad():
                for _ in range(1):
                    _, solution = AM._inner(batch, AM_embedding)
                    cost = problem.get_costs(batch, solution, multi_solu=False)
                    # history_solutions = solution.unsqueeze(0)
                    # history_costs = cost.unsqueeze(0)
                    history_solutions.append(solution)
                    history_costs.append(cost)
                    best_for_now = torch.cat((best_so_far[None, :], cost[None, :]), 0).min(0)[0]
                    best_so_far = best_for_now
                    AM_embedding_new = self.actor(problem,
                                                   batch,
                                                   dist,
                                                   AM_embedding,
                                                  history_solutions,
                                                  history_costs,
                                                   solu_embed,
                                                   AM_embed_new_only=True
                                                   )

                    self.eval()
                    _, solution = AM._inner(batch, AM_embedding_new)
                    cost = problem.get_costs(batch, solution, multi_solu=False)
                    best_for_now = torch.cat((best_so_far[None, :], cost[None, :]), 0).min(0)[0]
                    best_so_far = best_for_now

        AM_embedding = AM_embedding.detach()
        obj_history = [best_so_far]
        best_cost_history = [best_so_far]
        reward = []
        solution_history = [solution.clone()]

        for t in tqdm(range(self.opts.T_max), disable=self.opts.no_progress_bar, desc='rollout',
                      bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):

            solution, _, cost, _, solu_embed = self.actor(problem,
                                                             batch,
                                                             dist,
                                                             AM_embedding.detach(),
                                                             history_solutions,
                                                             history_costs,
                                                             solu_embed.detach() if t > 0 else None
                                                             )

            # history_solutions = torch.cat((history_solutions, solution.unsqueeze(0)), 0)
            # history_costs = torch.cat((history_costs, cost.unsqueeze(0)), 0)

            # if len(history_solutions) > opts.K:
            #
            #     # greedily retain a solution pool
            #     history_costs = history_costs.transpose(0, 1)
            #     history_solutions = history_solutions.transpose(0, 1)
            #     index = history_costs.sort(1)[1]
            #     history_costs = history_costs.gather(1, index)
            #     history_solutions = history_solutions.gather(1, index.unsqueeze(-1).expand_as(history_solutions))
            #     history_costs = history_costs[:, :-1].transpose(0, 1)
            #     history_solutions = history_solutions[:, :-1].transpose(0, 1)

            # random and diverse pool
            if len(history_solutions) > opts.K - 1:
                # randomly remove a solution
                index = np.random.randint(0, len(history_solutions))
                del history_solutions[index]
                del history_costs[index]

                r = np.random.uniform(low=1, high=opts.epoch_end + 1, size=1)

                if r < cri[epoch]:
                    # remove the random solution from last step
                    del history_solutions[0]
                    del history_costs[0]

                    # add a random solution for diversity
                    rand_solution = move_to(problem.get_initial_solutions(batch_size, batch, dist, opts.init_val_met),
                                            opts.device)
                    # print('random', rand_solution)
                    rand_cost = problem.get_costs(batch, rand_solution, multi_solu=False)
                    history_solutions.insert(0, rand_solution)
                    history_costs.insert(0, rand_cost)
            history_solutions.append(solution)
            history_costs.append(cost)

            best_for_now = torch.cat((best_so_far[None, :], cost[None, :]), 0).min(0)[0]

            rewards = best_so_far - best_for_now  # >= 0
            best_so_far = best_for_now

            # record informations
            reward.append(rewards)
            obj_history.append(cost)
            best_cost_history.append(best_so_far)

            if record:
                solution_history.append(solution)

        out = (best_so_far,  # best_cost: batch_size, 1
               torch.stack(obj_history, 1),  # batch_size, T
               torch.stack(best_cost_history, 1),  # batch_size, T
               torch.stack(reward, 1),  # batch_size, T
               None if not record else torch.stack(solution_history, 1))

        return out

    def start_training(self, problem, val_dataset, tb_logger):
        train(problem, self, val_dataset, tb_logger, self.problem_name)


def train(problem, agent, val_dataset, tb_logger, problem_name):
    opts = agent.opts
    cri = np.logspace(np.log(opts.epoch_start + 1), np.log(opts.epoch_end + 1), opts.epoch_end, base=np.exp(1))

    # validate(problem, agent, val_dataset, tb_logger, epoch, cri)

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

        set_decode_type(agent.actor, "sampling")
        if isinstance(agent.actor, DataParallel):
            set_decode_type(agent.actor.module.AM, "sampling")
        else:
            set_decode_type(agent.actor.AM, "sampling")

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

    # for t == 0:
    if isinstance(agent.actor, DataParallel):
        AM = agent.actor.module.AM
    else:
        AM = agent.actor.AM

    # with torch.no_grad():
    AM_embedding, _ = AM.embedder(AM._init_embed(batch))
    AM_embedding = AM_embedding.detach()
    solu_embed = None

    history_solutions = []
    history_costs = []
    if opts.init_val_met == 'nearest':
        solution = move_to(problem.get_initial_solutions(batch_size, batch, dist), opts.device)
        best_so_far = problem.get_costs(batch, solution, multi_solu=False)
        # history_solutions = solution
        # history_costs = best_so_far
        history_solutions.append(solution)
        history_costs.append(best_so_far)
    else:
        best_so_far = 1e5 * torch.ones(batch_size, device=opts.device)

        with torch.no_grad():
            for _ in range(1):
                _, solution = AM._inner(batch, AM_embedding)
                cost = problem.get_costs(batch, solution, multi_solu=False)
                # history_solutions = solution.unsqueeze(0)
                # history_costs = cost.unsqueeze(0)
                history_solutions.append(solution)
                history_costs.append(cost)
                best_for_now = torch.cat((best_so_far[None, :], cost[None, :]), 0).min(0)[0]
                best_so_far = best_for_now
                AM_embedding_new = agent.actor(problem,
                                               batch,
                                               dist,
                                               AM_embedding,
                                               history_solutions,
                                               history_costs,
                                               solu_embed,
                                               AM_embed_new_only=True
                                               )

                agent.eval()
                _, solution = AM._inner(batch, AM_embedding_new)
                cost = problem.get_costs(batch, solution, multi_solu=False)
                best_for_now = torch.cat((best_so_far[None, :], cost[None, :]), 0).min(0)[0]
                best_so_far = best_for_now

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
            # get model output, new_solution:[bs, 1, gs]
            solution, log_lh, cost, AM_embedding_new, solu_embed = agent.actor(problem,
                                                                               batch,
                                                                               dist,
                                                                               AM_embedding,
                                                                               history_solutions,
                                                                               history_costs,
                                                                               solu_embed
                                                                               )

            # history_solutions = torch.cat((history_solutions, solution.unsqueeze(0)), 0)
            # history_costs = torch.cat((history_costs, cost.unsqueeze(0)), 0)

            # if len(history_solutions) > opts.K:
            #
            #     # greedily retain a solution pool
            #     history_costs = history_costs.transpose(0, 1)
            #     history_solutions = history_solutions.transpose(0, 1)
            #     index = history_costs.sort(1)[1]
            #     history_costs = history_costs.gather(1, index)
            #     history_solutions = history_solutions.gather(1, index.unsqueeze(-1).expand_as(history_solutions))
            #     history_costs = history_costs[:, :-1].transpose(0, 1)
            #     history_solutions = history_solutions[:, :-1].transpose(0, 1)

            if len(history_solutions) > opts.K - 1:
                # randomly remove a solution
                index = np.random.randint(0, len(history_solutions))
                del history_solutions[index]
                del history_costs[index]

                r = np.random.uniform(low=1, high=opts.epoch_end + 1, size=1)

                if r < cri[epoch]:
                    # remove the random solution from last step
                    del history_solutions[0]
                    del history_costs[0]

                    # add a random solution for diversity
                    rand_solution = move_to(problem.get_initial_solutions(batch_size, batch, dist),
                                            opts.device)
                    rand_cost = problem.get_costs(batch, rand_solution, multi_solu=False)
                    history_solutions.insert(0, rand_solution)
                    history_costs.insert(0, rand_cost)
            history_solutions.append(solution)
            history_costs.append(cost)

            bl_val_detached, bl_val = agent.critic(AM_embedding_new)

            # get estimated value from baseline
            baseline_val_detached.append(bl_val_detached)
            baseline_val.append(bl_val)

            log_likelihood.append(log_lh)

            solution = move_to(solution, opts.device)
            best_for_now = torch.cat((best_so_far[None, :], cost[None, :]), 0).min(0)[0]
            # reward_ctrl = torch.clamp(0.001 / ((AM_embedding - AM_embedding_new) ** 2).mean(-1).mean(-1), 0, 0.1)
            reward.append(best_so_far - best_for_now)  # >= 0
            best_so_far = best_for_now

            # state transient
            total_cost = total_cost + cost

            # next
            t = t + 1

        solu_embed = Variable(solu_embed.data)

        # reward
        total_cost = total_cost / (t - t_s)
        reward_reversed = reward[::-1]

        Reward = []
        next_return, _ = agent.critic(AM_embedding_new)
        for r in range(len(reward_reversed)):
            R = next_return * gamma + reward_reversed[r]
            Reward.append(R)
            next_return = R

        # torch.Size([T, batch_size])
        Reward = torch.stack(Reward[::-1], 0)  # n_step, bs

        baseline_val = torch.stack(baseline_val, 0)  # (n_step, bs)
        baseline_val_detached = torch.stack(baseline_val_detached, 0)
        log_likelihood = torch.stack(log_likelihood, 0)  # (n_step, bs)

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
