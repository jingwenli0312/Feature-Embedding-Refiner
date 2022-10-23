#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 20:47:26 2020

@author: yiningma
"""
import torch
import os
from matplotlib import pyplot as plt
import cv2
import io
import numpy as np


def plot_grad_flow(model):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    named_parameters = model.named_parameters()
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is not None:
                ave_grads.append(p.grad.abs().mean())
            else:
                ave_grads.append(0)
    plt.ioff()
    fig = plt.figure(figsize=(8, 6))
    plt.plot(ave_grads, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, dpi=60)
    plt.close(fig)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def plot_improve_pg(history_value):
    plt.ioff()
    fig = plt.figure(figsize=(4, 3))
    plt.plot(history_value.mean(0).cpu())

    plt.xlabel("T")
    plt.ylabel("Cost")
    plt.title("Avg Improvement Progress")
    plt.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, dpi=60)
    plt.close(fig)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def plot_entropy_pg(entropy):
    plt.ioff()
    fig = plt.figure(figsize=(4, 3))
    plt.plot(entropy.mean(0))
    plt.xlabel("T")
    plt.ylabel("Entropy")
    plt.title("Avg Entropy Progress")
    plt.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, dpi=60)
    plt.close(fig)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def plot_tour_tsp(problem, city_tour, coordinates, save=False, dpi=300, show=True):
    if not show: plt.ioff()

    fig = plt.figure(figsize=(8, 6))

    size = len(city_tour)

    # index = torch.cat((
    #     city_tour.view(-1, 1),
    #     city_tour[0].view(-1, 1))
    # ).repeat(1, 2)
    index = city_tour.view(-1, 1).repeat(1, 2)

    xy = torch.gather(coordinates, 0, index)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axis([-0.05, 1.05] * 2)
    plt.plot(xy[:, 0], xy[:, 1], color='black', zorder=1)

    handle = [None]

    for i in range(1, size):
        node = city_tour[i - 1]
        color, label = 'blue', f'{node}'
        handle[0] = plt.scatter(xy[i, 0], xy[i, 1], s=45, c=color, zorder=2)
        plt.annotate(label, (xy[i, 0] - 0.015, xy[i, 1] - 0.06), fontsize=12)

    plt.legend(handle, ['node'], fontsize=12)
    plt.show()
    #    plot show
    if save:
        outfolder = f'../results/figures'
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        plt.savefig(f'../results/figures/{problem.NAME}_{size}.png', dpi=dpi)
        # plt.savefig(f'../results/figures/{problem.NAME}_{size}.eps', dpi=dpi)
        print(f'Plot saved to: ',
              f'../results/figures/{problem.NAME}_{size}.png',
              f'../results/figures/{problem.NAME}_{size}.eps')

    # if not show:
    #     buf = io.BytesIO()
    #     plt.savefig(buf, dpi=dpi)
    #     plt.close(fig)
    #     buf.seek(0)
    #     img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    #     buf.close()
    #     img = cv2.imdecode(img_arr, 1)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     return img
    # else:
    #     plt.show()
    #     return None


def plot_tour_vrp(problem, coordinates, save=True, dpi=300, show=True):
    if not show: plt.ioff()

    fig = plt.figure(figsize=(8, 6))
    city_tour = torch.tensor([0, 5, 37], device=coordinates.device)
    coordinates = coordinates.squeeze(0)
    print(coordinates.size())
    size = len(coordinates)

    index = city_tour.view(-1, 1).repeat(1, 2)
    xy = torch.gather(coordinates, 0, index).cpu()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('font', family='Times New Roman', size=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axis([-0.05, 1.05] * 2)
    plt.plot(xy[:, 0], xy[:, 1], color='black', zorder=1)

    # mark depot
    g = plt.scatter(xy[0, 0], xy[0, 1], marker='H', s=55, c='red', zorder=2)
    plt.annotate(f'{0}', (coordinates[0, 0] - 0.015, coordinates[0, 1] - 0.06), fontsize=8)

    handle = [g, None]
    coordinates = coordinates.cpu()

    for i in range(1, size):
        # node = coordinates[i - 1]
        color, label = 'blue', f'{i}'
        handle[1] = plt.scatter(coordinates[i, 0], coordinates[i, 1], s=45, c=color, zorder=2)
        plt.annotate(label, (coordinates[i, 0] - 0.015, coordinates[i, 1] - 0.06), fontsize=12)

        # if i <= len(xy):
        #     plt.annotate(label, (xy[i-1, 0] - 0.015, xy[i-1, 1] - 0.06), fontsize=12)

    plt.legend(handle, ['depot', 'node'], fontsize=18)

    # plt.show()

    # if save:
    outfolder = f'/home/lijw/Learn_to_reconstruct_POMO/results/figures'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    plt.savefig(f'/home/lijw/Learn_to_reconstruct_POMO/results/figures/{problem}_{size}.png', dpi=dpi)
    # plt.savefig(f'../results/figures/{problem.NAME}_{size}.eps', dpi=dpi)
    print(f'Plot saved to: ',
          f'/home/lijw/Learn_to_reconstruct_POMO/results/figures/{problem}_{size}.png',
          f'/home/lijw/Learn_to_reconstruct_POMO/results/figures/{problem}_{size}.eps')


def plot_tour(problem, city_tour, coordinates, save=False, p='pdp', dpi=300, show=True):
    if not show: plt.ioff()

    fig = plt.figure(figsize=(8, 6))

    size = problem.size

    if p != 'pdp':
        demand = 0
        raise NotImplementedError()

    index = torch.cat((
        torch.tensor(0).view(-1, 1),
        city_tour.view(-1, 1),
        torch.tensor(0).view(-1, 1))
    ).repeat(1, 2)

    xy = torch.gather(coordinates, 0, index)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axis([-0.05, 1.05] * 2)
    plt.plot(xy[:, 0], xy[:, 1], color='black', zorder=1)

    # mark depot
    g = plt.scatter(xy[0, 0], xy[0, 1], marker='H', s=55, c='red', zorder=2)

    handle = [g, None, None]
    for i in range(1, size + 1):
        node = city_tour[i - 1]
        if node <= size // 2:
            if p == 'pdp':
                color, label, marker = 'blue', f'p{node}', '^'
            else:
                color, label, marker = 'blue', f'p{node}({int(demand[city_tour[i] - 1])})', '^'
        else:
            if p == 'pdp':
                color, label, marker = 'orange', f'd{node - (size // 2)}', 's'
            else:
                color, label, marker = 'orange', f'd{node - (size // 2)}({int(demand[city_tour[i] - 1])})', 's'
        handle[int(node <= size // 2) + 1] = plt.scatter(xy[i, 0], xy[i, 1],
                                                         marker=marker, s=45, c=color, zorder=2)
        plt.annotate(label, (xy[i, 0] - 0.015, xy[i, 1] - 0.06), fontsize=12)

    plt.legend(handle, ['depot', 'delivery node', 'pickup node'], fontsize=12)
    #    plot show
    if save:
        outfolder = f'../results/figures'
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        plt.savefig(f'../results/figures/{problem.NAME}_{size}.png', dpi=dpi)
        plt.savefig(f'../results/figures/{problem.NAME}_{size}.eps', dpi=dpi)
        print(f'Plot saved to: ',
              f'../results/figures/{problem.NAME}_{size}.png',
              f'../results/figures/{problem.NAME}_{size}.eps')

    if not show:
        buf = io.BytesIO()
        plt.savefig(buf, dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        plt.show()
        return None


def plot_heatmap(problem, solutions, mask):
    problem.use_real_mask = True
    real_mask = ~problem.get_swap_mask(solutions).bool()
    problem.use_real_mask = False

    import seaborn as sns;
    sns.set()

    bs = real_mask.size(0)

    fig, ax = plt.subplots(bs, 2, figsize=(6, 3 * bs), gridspec_kw={'width_ratios': [1, 1.25]})

    for i in range(bs):
        ax1, ax2 = ax[i]
        sns.heatmap(real_mask[i], ax=ax1, cbar=False)
        sns.heatmap(mask.detach()[i], ax=ax2)
        plt.setp(ax1, ylabel=f'instance {i}')

    ax[0, 1].set_title('Predicted Masks')
    ax[0, 0].set_title('True Masks')

    plt.show()

