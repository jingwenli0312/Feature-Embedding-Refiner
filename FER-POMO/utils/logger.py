import torch
import math
import numpy as np
from utils.plots import plot_grad_flow, plot_improve_pg


# too many indicators are wrong

def log_to_screen(time_used, init_value, best_value, reward, costs_history, search_history,
                  batch_size, dataset_size, T):
    # reward, bs, gs, T
    print('\n', '-' * 60)
    print('Avg total reward:'.center(35), '{:<10f} +- {:<10f}'.format(
        reward.sum(1).mean(), torch.std(reward.sum(1)) / math.sqrt(batch_size)))
    print('Avg step reward:'.center(35), '{:<10f} +- {:<10f}'.format(
        reward.mean(), torch.std(reward) / math.sqrt(batch_size)))

    # cost
    print('-' * 60)
    print('Avg init cost:'.center(35), '{:<10f} +- {:<10f}'.format(
        init_value.mean(), torch.std(init_value) / math.sqrt(batch_size)))
    for per in range(20, 100, 20):
        cost_ = costs_history[:, round(T * per / 100)]
        print(f'Avg cost after {per}% steps:'.center(35), '{:<10f} +- {:<10f}'.format(
            cost_.mean(),
            torch.std(cost_) / math.sqrt(batch_size)))
    # best cost
    print('-' * 60)

    for per in range(20, 100, 20):
        cost_ = search_history[:, round(T * per / 100)]
        print(f'Avg best cost after {per}% steps:'.center(35), '{:<10f} +- {:<10f}'.format(
            cost_.mean(),
            torch.std(cost_) / math.sqrt(batch_size)))
    print(f'Avg final best cost:'.center(35), '{:<10f} +- {:<10f}'.format(
        best_value.mean(), torch.std(best_value) / math.sqrt(batch_size)))

    # time
    print('-' * 60)
    print('Avg used time:'.center(35), '{:f}s'.format(
        time_used.mean() / dataset_size))
    print('-' * 60, '\n')


def log_to_tb_val(tb_logger, time_used, init_value, best_value, reward, costs_history, search_history,
                  batch_size, val_size, dataset_size, T, no_figures, epoch):
    # if not no_figures:
    #     tb_logger.log_images('validation/improve_pg', [plot_improve_pg(costs_history)], epoch)
    #     tb_logger.log_images('validation/search_pg', [plot_improve_pg(search_history)], epoch)

    # reward, bs, gs, T
    tb_logger.log_value('validation/avg_time', time_used.mean() / dataset_size, epoch)
    tb_logger.log_value('validation/avg_total_reward', reward.sum(1).mean(), epoch)
    tb_logger.log_value('validation/avg_step_reward', reward.mean(), epoch)

    tb_logger.log_value(f'validation/avg_init_cost', init_value.mean(), epoch)
    tb_logger.log_value(f'validation/avg_best_cost', best_value.mean(), epoch)

    for per in range(20, 100, 20):
        cost_ = costs_history[:, round(T * per / 100)]
        tb_logger.log_value(f'validation/avg_.{per}_cost', cost_.mean(), epoch)

    # if information:
    #     swapped_array = [info['swaped'] for info in information]
    #     swapped = np.array(swapped_array)
    #     tb_logger.log_value(f'validation/avg_swap_skipped', 1 - np.mean(swapped), epoch)
    #
    #     if information[0]['perturbed'] is not None:
    #         perturbed_array = [info['perturbed'] for info in information]
    #         perturbed = np.array(perturbed_array)
    #         tb_logger.log_value(f'validation/avg_perturb_skipped', np.mean(perturbed), epoch)


def log_to_tb_train(tb_logger, agent, avg_cost, grad_norms, reward, Reward,
                    reinforce_loss, baseline_loss, log_likelihood, mini_step):
    tb_logger.log_value('learnrate_pg', agent.optimizer.param_groups[0]['lr'], mini_step)
    # avg_cost = (total_cost).mean().item()
    tb_logger.log_value('train/avg_cost', avg_cost.mean(), mini_step)
    avg_reward = torch.stack(reward, 0).sum(0).mean().item()  # reward: n_step, bs, gs
    max_reward = torch.stack(reward, 0).sum(0).max().item()
    tb_logger.log_value('train/avg_reward', avg_reward, mini_step)
    tb_logger.log_value('train/target_return', Reward.mean().item(), mini_step)
    # tb_logger.log_value('train/init_cost', initial_cost.mean(), mini_step)
    tb_logger.log_value('train/max_reward', max_reward, mini_step)
    grad_norms, grad_norms_clipped = grad_norms
    tb_logger.log_value('loss/actor_loss', reinforce_loss.item(), mini_step)
    tb_logger.log_value('loss/nll', -log_likelihood.mean().item(), mini_step)
    # tb_logger.log_value('loss/mask_loss', mask_loss.item(), mini_step)
    # tb_logger.log_value('train/entropy', entropy.mean().item(), mini_step)
    # tb_logger.log_value('train/approx_kl_divergence', approx_kl_divergence.item(), mini_step)

    tb_logger.log_value('grad/actor', grad_norms[0], mini_step)
    tb_logger.log_value('grad_clipped/actor', grad_norms_clipped[0], mini_step)
    tb_logger.log_value('loss/critic_loss', baseline_loss.item(), mini_step)
    tb_logger.log_value('loss/total_loss', (reinforce_loss + baseline_loss).item(), mini_step)
    tb_logger.log_value('grad/critic', grad_norms[1], mini_step)
    tb_logger.log_value('grad_clipped/critic', grad_norms_clipped[1], mini_step)

    # if not no_figures:
    if mini_step % 10 == 0:
        tb_logger.log_images('grad/actor',[plot_grad_flow(agent.actor)], mini_step)
        tb_logger.log_images('grad/critic',[plot_grad_flow(agent.critic)], mini_step)

    # if information:
    #     swapped_array = [info['swaped'] for info in information]
    #     swapped = np.array(swapped_array)
    #     tb_logger.log_value(f'train/avg_swap_skipped',  1 - np.mean(swapped), mini_step)