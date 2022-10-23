#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 18:47:10 2020

@author: yiningma
"""

from utils.plots import plot_tour
import imageio
import torch
        
def record_vedios(problem, batch, history_solutions, filename = 'ep_vedio', dpi = 30):
    
    history = torch.stack(history_solutions)
    ep_len, batch_size, p_size = history.size()
    
    for batch_i in range(batch_size):
        
        solutions_per_instance = history[:,batch_i]
        
        with imageio.get_writer(f'./outputs/{filename}_{batch_i}.gif', mode='I') as writer:
        
            for tour in solutions_per_instance:
            
                img = plot_tour(problem, 
                            tour, 
                            batch['coordinates'][batch_i], 
                            show = False, 
                            dpi = dpi)
            
                writer.append_data(img)