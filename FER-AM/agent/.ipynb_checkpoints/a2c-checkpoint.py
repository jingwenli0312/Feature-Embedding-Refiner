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
            feed_forward_dim=opts.feed_forward_dim,
            n_heads_encoder=opts.encoder_head_num,
            n_heads_decoder=opts.decoder_head_num,
            n_layers=opts.n_encode_layers,
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

    def rollout(self, problem, batch, opts, record=False):
        set_decode_type(self.actor, "sampling")
        if isinstance(self.actor, DataParallel):
            set_decode_type(self.actor.module.AM, "sampling")
        else:
            set_decode_type(self.actor.AM, "sampling")

        # prepare the features
        batch = move_to(batch, opts.device)
        # batch_feature = input_feature_encoding(batch, self.problem_name)


        # solution = move_to(problem.get_initial_solutions(batch.size(0), batch), self.opts.device)

        # for t == 0:
        if isinstance(self.actor, DataParallel):
            AM = self.actor.module.AM
        else:
            AM = self.actor.AM
                
        AM_embedding, _ = AM.embedder(AM._init_embed(batch))

        _, solution = AM._inner(batch, AM_embedding)
        best_so_far = problem.get_costs(batch, solution)
        AM_embedding = AM_embedding.detach()


        # best_so_far = problem.get_costs(batch, solution)
        best_solution = solution

        obj_history = [best_so_far]
        best_cost_history = [best_so_far]
        reward = []
        solution_history = [solution.clone()]


        h_x = None
        c_x = None

        # AM_embedding, solution -> ours -> imbedding -> AMdecoder -> solution
        # ours_net:
        #     - input : embedding, hiddenstates, 2 * current_solution
        #     - output : next_embedding, next_hiddenstates, solution

        # for t >0
        for t in tqdm(range(self.opts.T_max), disable=self.opts.no_progress_bar, desc='rollout',
                      bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):

            solution, log_lh, cost, h_x, c_x, AM_embedding_new = self.actor(problem,
                                                                       batch,
                                                                       AM_embedding.detach(),
                                                                       solution,
                                                                       best_solution,
                                                                       h_x.detach() if t>0 else None,
                                                                       c_x.detach() if t>0 else None,
                                                                       )

            index = (cost < best_so_far)
            best_solution[index] = solution[index]

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
    # validate(problem, agent, val_dataset, tb_logger, _id=0)

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
                        pbar)
            step += 1
        pbar.close()

        # save new model
        if (not opts.no_saving and opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                epoch == opts.n_epochs - 1: agent.save(epoch)

        # validate the new model
        validate(problem, agent, val_dataset, tb_logger, _id=epoch)


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
        pbar
):
    # prepare the features
    batch = move_to(batch, opts.device)  # batch_size, graph_size, 2
    # batch_feature = input_feature_encoding(batch, problem_name)

    # for t == 0:
    if isinstance(agent.actor, DataParallel):
        AM = agent.actor.module.AM
    else:
        AM = agent.actor.AM
        
    AM_embedding, _ = AM.embedder(AM._init_embed(batch))

    _, solution = AM._inner(batch, AM_embedding)
    best_so_far = problem.get_costs(batch, solution)
    AM_embedding = AM_embedding.detach()

    # solution = move_to(problem.get_initial_solutions(opts.batch_size, batch_feature), opts.device)  # [batch_size, graph_size]
    best_solution = solution
    # best_so_far = problem.get_costs(batch_feature, solution)

    # if opts.warm_up:
    #     for _ in range(opts.warm_up):
    #         agent.eval()
    #         problem.eval(perturb=False)
    #         solution, log_lh, cost, _, _ = agent.actor(problem,
    #                                                        batch_feature,
    #                                                        solution,
    #                                                        best_solution,
    #                                                        h_x, c_x)
    #
    #         index = (cost < best_so_far)
    #         best_solution[index] = solution[index]
    #
    #         best_for_now = torch.cat((best_so_far[None, :], cost[None, :]), 0).min(0)[0]
    #         best_so_far = best_for_now
    #
    # agent.train()
    # params

    gamma = opts.gamma
    n_step = opts.n_step
    T = opts.T_train
    t = 0
    # previous_embed = None
    h_x = None
    c_x = None

    while t < T: # T 200, n = 4, 50 gradient steps
        # empty array
        total_cost = 0
        baseline_val = []
        baseline_val_detached = []
        log_likelihood = []
        reward = []
        t_s = t

        while t - t_s < n_step and not (t == T):
            
            # get model output, new_solution:[bs, 1, gs]
            solution, log_lh, cost, h_x, c_x, AM_embedding_new = agent.actor(problem,
                                                                        batch,
                                                                        AM_embedding,
                                                                        solution,
                                                                        best_solution,
                                                                        h_x, c_x,
                                                                        )

            # print('var', torch.norm(AM_embedding-AM_embedding_new, 2, dim = 2).mean(-1)) 
            bl_val_detached, bl_val = agent.critic(AM_embedding_new)

            # get estimated value from baseline
            baseline_val_detached.append(bl_val_detached)
            baseline_val.append(bl_val)

            log_likelihood.append(log_lh)

            solution = move_to(solution, opts.device)

            index = (cost < best_so_far)
            best_solution[index] = solution[index]

            best_for_now = torch.cat((best_so_far[None, :], cost[None, :]), 0).min(0)[0]
            reward_ctrl = -0.01 * ((AM_embedding-AM_embedding_new) ** 2).mean(-1).mean(-1)
            reward.append(best_so_far - best_for_now + reward_ctrl)  # >= 0
            
            best_so_far = best_for_now

            # state transient
            total_cost = total_cost + cost

            # next
            t = t + 1

        h_x = Variable(h_x.data)
        c_x = Variable(c_x.data)

        # reward
        total_cost = total_cost / (t - t_s)
        reward_reversed = reward[::-1]
        #print('reward', reward)
        Reward = []
        # loc_embed = agent.actor(problem,
        #                           batch_feature,
        #                           solution,
        #                           best_solution,
        #                           h_x.detach(), c_x.detach(),
        #                           loc_embed_only=True
        #                           )
        next_return, _ = agent.critic(AM_embedding_new)
        for r in range(len(reward_reversed)):
            R = next_return * gamma + reward_reversed[r]
            Reward.append(R)
            next_return = R

        # torch.Size([T, batch_size])
        Reward = torch.stack(Reward[::-1], 0)  # n_step, bs
  
        baseline_val = torch.stack(baseline_val, 0)
        baseline_val_detached = torch.stack(baseline_val_detached, 0)
        log_likelihood = torch.stack(log_likelihood, 0)

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
