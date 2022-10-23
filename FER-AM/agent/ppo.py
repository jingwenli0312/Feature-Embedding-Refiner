import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import clip_grad_norms
from nets.actor_network import Actor, set_decode_type
from nets.critic_network import Critic
from utils import torch_load_cpu, get_inner_model, move_to, input_feature_encoding
from utils.logger import log_to_tb_train
from agent.utils import validate
import math
from torch import nn
from torch.nn import DataParallel

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.mask_true = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.mask_true[:]
        

class PPO:
    def __init__(self, problem_name, opts):
        
        # figure out the options
        self.opts = opts
        self.problem_name = problem_name
        
        # figure out the actor
        self.actor = Actor(
            problem_name = problem_name,
            embedding_dim = opts.embedding_dim,
            hidden_dim = opts.hidden_dim,
            feed_forward_dim=opts.feed_forward_dim,
            n_heads_encoder = opts.encoder_head_num,
            n_heads_decoder = opts.decoder_head_num,
            n_layers = opts.n_encode_layers,
            normalization = opts.normalization,
            device = opts.device,
            opts=opts
        ).to(opts.device)
        
        if not opts.eval_only:
            # figure out the critic
            self.critic = Critic(
                    problem_name = problem_name,
                    embedding_dim = opts.embedding_dim,
                    hidden_dim = opts.hidden_dim,
                    n_heads = opts.critic_head_num,
                    n_layers = opts.n_encode_layers,
                    normalization = opts.normalization,
                    device = opts.device
                ).to(opts.device)
                
            # figure out the optimizer
            self.optimizer = optim.Adam(
                [{'params': self.actor.parameters(), 'lr': opts.lr_model}] + 
                [{'params': self.critic.parameters(), 'lr': opts.lr_critic}])
            
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: opts.lr_decay ** epoch)              
        
        
        if opts.use_cuda and torch.cuda.device_count() > 1:
            self.actor = torch.nn.DataParallel(self.actor)
            if not opts.eval_only: self.critic = torch.nn.DataParallel(self.critic)
    
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
        if not self.opts.eval_only: self.critic.eval()
        
    def train(self):
        self.actor.train()
        if not self.opts.eval_only: self.critic.train()
    
    def rollout(self, problem, batch, opts, record=True):
        set_decode_type(self.actor, "greedy")
        
        solution = move_to(problem.get_initial_solutions(batch['loc'].size(0)), self.opts.device)
        
        best_so_far = problem.get_costs(batch, solution)
        
        obj_history = [best_so_far]
        best_cost_history = [best_so_far]
        reward = []
        solution_history = [solution.clone()]

        batch_feature = input_feature_encoding(batch, self.problem_name)
        edge_embed = None

        
        for t in tqdm(range(self.opts.T_max), disable=self.opts.no_progress_bar, desc = 'rollout', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            # prepare the features

            # pass through model
            solution, _, edge_embed = self.actor(problem,
                                                   batch_feature,
                                                   solution,
                                                   edge_embed)

            cost = problem.get_costs(batch, solution)
            best_for_now = torch.cat((best_so_far[None, :], cost[None, :]), 0).min(0)[0]

            rewards = best_so_far - best_for_now  # >= 0
            best_so_far = best_for_now

            
            # record informations
            reward.append(rewards)
            obj_history.append(cost)
            best_cost_history.append(best_so_far)
            
            if record:
                solution_history.append(solution)

        print('best', torch.stack(best_cost_history, 1).size(), torch.stack(best_cost_history, 1))
        out = (best_so_far, # best_cost: batch_size, 1
               torch.stack(obj_history, 1),  # batch_size, T
               torch.stack(best_cost_history, 1),  # batch_size, T
               torch.stack(reward, 1), # batch_size, T
               None if not record else torch.stack(solution_history, 1))
        
        return out
    
    def start_training(self, problem, val_dataset, tb_logger):
        train(problem, self, val_dataset, tb_logger, self.problem_name)


def train(problem, agent, val_dataset, tb_logger, problem_name):
    opts = agent.opts
    set_decode_type(agent.actor, "sampling")
    
    memory = Memory()
    
    # Start the actual training loop
    for epoch in range(opts.epoch_start, opts.epoch_end):
        # Training mode
        print('\n\n')
        print("|",format(f" Training epoch {epoch} ","*^60"),"|")
        agent.train()
        problem.train()
        # lr_scheduler
        agent.lr_scheduler.step(epoch)
        print("Training with lr={:.3e} for run {}".format(agent.optimizer.param_groups[0]['lr'], opts.run_name) , flush=True)
        step = epoch * (opts.epoch_size // opts.batch_size)    
    
        # Generate new training data for each epoch
        training_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size)
        training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size)
    
        # start training    
        pbar = tqdm(total = (opts.K_epochs) * (opts.epoch_size // opts.batch_size) * (opts.T_train // opts.n_step) ,
                    disable = opts.no_progress_bar, desc = f'training',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        for batch_id, batch in enumerate(training_dataloader):
            train_batch(problem,
                      problem_name,
                      agent,
                      memory,
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
        if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                        epoch == opts.epoch_end - 1): agent.save(epoch)
        
        # validate the new model
        validate(problem, agent, val_dataset, tb_logger, _id=epoch)
    
    
def train_batch(
    problem,
    problem_name,
    agent,
    memory,
    epoch,
    batch_id,
    step,
    batch,
    tb_logger,
    opts,
    pbar
    ):

    solution = move_to(problem.get_initial_solutions(opts.batch_size), opts.device)
    batch_size, graph_size = solution.size()

    # prepare the features
    batch = move_to(batch, opts.device)  # batch_size, graph_size, 2

    #update best_so_far
    best_so_far = problem.get_costs(batch, solution)  # [batch_size]


    # params
    gamma = opts.gamma
    n_step = opts.n_step
    T = opts.T_train
    K_epochs = opts.K_epochs
    eps_clip = opts.eps_clip
    t = 0
    
    # data array
    total_cost = 0
    edge_embed = None
    batch_feature = input_feature_encoding(batch, problem_name)

    while t < T:
        
        t_s = t
        
        memory.actions.append(solution)
        
        while t - t_s < n_step and not (t == T):

            # get model output
            solution, log_lh, edge_embed = agent.actor(problem,
                                              batch_feature,
                                              solution,
                                              edge_embed)

            solution = move_to(solution, opts.device)
            cost = problem.get_costs(batch, solution)  # actor

            best_for_now = torch.cat((best_so_far[None, :], cost[None, :]), 0).min(0)[0]

            reward = best_so_far - best_for_now  # >= 0
            reward = torch.clamp(reward, - opts.reward_clip, opts.reward_clip)
            best_so_far = best_for_now

            memory.states.append(edge_embed)
            memory.actions.append(solution)
            memory.logprobs.append(log_lh)
            memory.rewards.append(reward)
            total_cost = total_cost + cost
            # next            
            t = t + 1


        # begin update        ======================= 
        
        # Get discounted R
        Reward = []
        total_cost = total_cost / (t-t_s)
        reward_reversed = memory.rewards[::-1]
        
        # get last value                
        next_return, _ = agent.critic(batch_feature, solution)
        # calculate return
        for r in range(len(reward_reversed)):     
            R = next_return * gamma + reward_reversed[r]
            Reward.append(R)
            next_return = R

        # clip the return:
        Reward = torch.stack(Reward[::-1], 0)  # n_step, bs
#        Reward = torch.clamp(Reward, - opts.reward_clip, opts.reward_clip)\
        
        # convert list to tensor
        old_states = torch.stack(memory.states).detach().view(-1, graph_size, graph_size, opts.embedding_dim)
        old_states = move_to(old_states, opts.device)

        Reward = Reward.view(-1)
        all_actions = torch.stack(memory.actions)
        old_actions = all_actions[1:].view(-1, graph_size)
        old_logprobs = torch.stack(memory.logprobs).detach().view(-1)
        old_soluitons = all_actions[:-1].view(-1, graph_size)
        
        # Optimize ppo policy for K mini-epochs:
        batch_feature_new = {}
        batch_feature_new['loc'] = batch_feature['loc'].unsqueeze(0).expand(n_step, *batch_feature['loc'].size()).reshape(-1, *batch_feature['loc'].size()[-2:])
        batch_feature_new['dist'] = batch_feature['dist'].unsqueeze(0).expand(n_step, *batch_feature['dist'].size()).reshape(-1, *\
                                    batch_feature['dist'].size()[-2:])

        old_value = None
        old_mask_prob = None
        
        for _k in range(K_epochs):
            # Evaluating old actions and values :
            
            # get estimated value from baseline
            bl_val_detached, bl_val = agent.critic(batch_feature_new, old_soluitons)

            # get new action_prob
            _, logprobs, _ = agent.actor(problem,
                                               batch_feature_new,
                                               old_soluitons,
                                               old_states,
                                               fixed_action=old_actions,
                                               require_entropy=True)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss:


            advantages = Reward - bl_val_detached

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages            
            reinforce_loss = -torch.min(surr1, surr2).mean()
            
            # define criteria
            criteria_mse = torch.nn.MSELoss()
            eps_range = eps_clip * opts.eps_range
            
            # define baseline loss
            if old_value is None:
                baseline_loss = criteria_mse(bl_val, Reward)
                old_value = bl_val.detach()
            else:
                vpredclipped = old_value + torch.clamp(bl_val - old_value, -eps_range, eps_range)
                baseline_loss = max(criteria_mse(bl_val, Reward),
                                    criteria_mse(vpredclipped, Reward))

            
            # check K-L divergence
            approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
            
            # calculate loss
            loss = baseline_loss + reinforce_loss
            # print('loss', loss)

            # update gradient step
            agent.optimizer.zero_grad()
            loss.backward()
            
            #Clip gradient norms and get (clipped) gradient norms for logging
            grad_norms = clip_grad_norms(agent.optimizer.param_groups, opts.max_grad_norm)
            # print('grad_norm', grad_norms)
            
            agent.optimizer.step()
    
            # Logging to tensorboard
            if(not opts.no_tb):
                current_step = int(step * T / n_step * K_epochs + t//n_step * K_epochs + _k)
                if current_step % int(opts.log_step) == 0:
                    log_to_tb_train(tb_logger, agent, total_cost, grad_norms, memory.rewards,
                       reinforce_loss, baseline_loss, logprobs, current_step)
                    information = []
                    
            pbar.update(1)     
        
        
        # end update
        memory.clear_memory()
    
            
     
