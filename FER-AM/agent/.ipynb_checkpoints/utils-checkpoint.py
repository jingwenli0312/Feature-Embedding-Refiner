# -*- coding: utf-8 -*-

import time
import torch
from tqdm import tqdm
from utils.logger import log_to_screen, log_to_tb_val
from torch.utils.data import DataLoader
from utils import move_to


def validate(problem, agent, val_dataset, tb_logger, _id=None):
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

        # (batch_size, T); [(batch_size), ... T]
        with torch.no_grad():
            batch = move_to(batch, opts.device)  # batch_size, graph_size, 2
            bv, cost_hist, best_hist, r, rec_history = agent.rollout(problem,
                                                                     batch,
                                                                     opts,
                                                                     record=opts.record)

        init_value.append(cost_hist[:, 0])
        time_used.append(time.time() - s_time)
        best_value.append(bv)
        costs_history.append(cost_hist)
        search_history.append(best_hist)
        reward.append(r)
        if opts.record:
            history_record.append(rec_history)

    init_value = torch.cat(init_value, 0)  # batch_size
    best_value = torch.cat(best_value, 0)  # batch_size
    costs_history = torch.cat(costs_history, 0)  # batch_size, T
    search_history = torch.cat(search_history, 0)  # batch_size, T
    reward = torch.cat(reward, 0)  # batch_size, T
    time_used = torch.tensor(time_used)

    if opts.record:
        history_record = torch.cat(history_record, 0)  # batch_size, T, 3, p_size

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
                      epoch=_id)

