# -*- coding: utf-8 -*-

import time
import torch
from tqdm import tqdm
from utils.logger import log_to_screen, log_to_tb_val
from torch.utils.data import DataLoader
from utils import move_to
import numpy as np


def validate(problem, agent, val_dataset, tb_logger, epoch, cri):
    # Validate mode
    print('\nValidating...', flush=True)
    agent.eval()
    problem.eval()
    opts = agent.opts

    init_value = []
    best_value = []
    costs_history = []
    search_history = []
    reward = []
    time_used = []
    information = []
    history_record = []

    for batch in tqdm(DataLoader(val_dataset, batch_size=opts.eval_batch_size),
                      disable=opts.no_progress_bar or opts.val_size == opts.eval_batch_size,
                      desc='validate', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):


        # run the model
        s_time = time.time()
        if problem.NAME == 'tsp':
            batch_size = batch.size(0)
        if problem.NAME == 'pdp' or problem.NAME == 'cvrp':
            batch_size = batch['depot'].size(0)

        # (batch_size, T); [(batch_size), ... T]
        with torch.no_grad():
            batch = move_to(batch, opts.device)  # batch_size, graph_size, 2
            bv, cost_hist, best_hist, r, rec_history = agent.rollout(problem,
                                                                     batch,
                                                                     opts,
                                                                     epoch,
                                                                     cri,
                                                                     record=opts.record)

            if opts.augment:
                bv = bv.reshape(8, batch_size)
                bv = bv.min(dim=0)[0]
                cost_hist = cost_hist.reshape(8, batch_size, -1)
                cost_hist = cost_hist.min(dim=0)[0]
                best_hist = best_hist.reshape(8, batch_size, -1)
                best_hist = best_hist.min(dim=0)[0]
                r = r.reshape(8, batch_size, opts.T_max)
                r = r.max(dim=0)[0]

        init_value.append(cost_hist[:, 0])
        time_used.append(time.time() - s_time)
        best_value.append(bv)
        costs_history.append(cost_hist)
        search_history.append(best_hist)
        reward.append(r)
        if opts.record:
            history_record.append(rec_history)
        # print('best', best_value)

    init_value = torch.cat(init_value, 0)  # val_size
    best_value = torch.cat(best_value, 0)  # val_size
    costs_history = torch.cat(costs_history, 0)  # val_size, T
    search_history = torch.cat(search_history, 0)  # val_size, T
    reward = torch.cat(reward, 0)  # batch_size, T
    time_used = torch.tensor(time_used)
    #print('cost', search_history.size(), search_history)
    #np.savetxt('/home/lijw/Learn_to_Reconstruct_LSTM/outputs/cvrp50_conv.csv', search_history.mean(0).cpu().numpy(), delimiter=',')  # T, converge_curve

    if opts.record:
        history_record = torch.cat(history_record, 0)  # batch_size, T, 3, p_size
        # print('solutions', history_record.size(), history_record)
        # np.savetxt('/home/lijw/Learn_to_Reconstruct_LSTM/outputs/cvrp50_initial_solutions.csv', history_record.reshape(costs_history.size(0), -1).cpu().numpy(), fmt='%i',  delimiter=',')

    # log to screen
    log_to_screen(time_used,
                  init_value,
                  best_value,
                  reward,
                  costs_history,
                  search_history,
                  batch_size=opts.eval_batch_size,
                  dataset_size=len(val_dataset),
                  T=opts.T_max // 2)

    # log to tb
    if (not opts.no_tb):
        log_to_tb_val(tb_logger,
                      time_used,
                      init_value,
                      best_value,
                      reward,
                      costs_history,
                      search_history,
                      batch_size=opts.eval_batch_size,
                      val_size=opts.val_size,
                      dataset_size=len(val_dataset),
                      T=opts.T_max,
                      no_figures=opts.no_figures,
                      epoch=epoch)

