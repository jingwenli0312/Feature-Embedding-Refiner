import torch
import math
from torch.nn import DataParallel

def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

def input_feature_encoding(batch, problem_name):
    if problem_name == 'pdp':
        batch_size, size, _ = batch['loc'].size()
        node_feature = torch.cat((batch['depot'].unsqueeze(1), batch['loc']), 1)  # [batch_size, n_cus+1, 2]

        paired_node = torch.cat((batch['loc'][:, - size // 2:, :], batch['loc'][:, :size // 2, :]), dim=1)
        paired_node_feature = torch.cat((batch['depot'].unsqueeze(1), paired_node), 1)

        # [batch_size, n_cus+1, 4] pickup后加delivery坐标，delivery后加pickup坐标
        return torch.cat((node_feature, paired_node_feature), dim=-1)

    elif problem_name == 'tsp':
        # batch_size, size, _ = batch['loc'].size()
        # after_edge = batch['loc'].gather(1, solution.long().unsqueeze(-1).expand_as(batch['loc']))  # [batch_size, size, 2]
        #
        # return torch.cat((batch['loc'], after_edge), -1)  # [batch_size, size, 4]
        return batch

def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape = (batch_s, problem, 2)
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape = (batch, problem, 1)
    mean_x = x.mean(1).unsqueeze(1)
    mean_y = y.mean(1).unsqueeze(1)
    # print('data', xy_data)
    # print(mean_y.size(), mean_y)
    # print(y.size(), y)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)
    dat9 = torch.cat(((x-mean_x) * math.cos(math.radians(45)) + (y-mean_y) * math.sin(math.radians(45)) + mean_x,
                     (y-mean_y) * math.cos(math.radians(45)) - (x-mean_x) * math.sin(math.radians(45)) + mean_y), dim=2)
    dat10 = torch.cat(((x - mean_x) * math.cos(math.radians(90)) + (y - mean_y) * math.sin(math.radians(90)) + mean_x,
                     (y - mean_y) * math.cos(math.radians(90)) - (x - mean_x) * math.sin(math.radians(90)) + mean_y),
                     dim=2)
    data_augmented = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape = (8*batch, problem, 2)

    return data_augmented

def calculate_distance(data):
    N_data = data.shape[0]
    dists = torch.zeros((N_data, N_data), dtype=torch.float)
    d1 = -2 * torch.mm(data, data.T)
    d2 = torch.sum(torch.pow(data, 2), dim=1)
    d3 = torch.sum(torch.pow(data, 2), dim=1).reshape(1, -1).T
    dists = d1 + d2 + d3
    dists[dists < 0] = 0
    return torch.sqrt(dists)