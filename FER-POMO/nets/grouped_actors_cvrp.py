
"""
The MIT License

Copyright (c) 2020 Yeong-Dae Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from torch.nn import DataParallel

# For debugging
from IPython.core.debugger import set_trace

# Hyper Parameters
from HYPER_PARAMS import *
from TORCH_OBJECTS import *
from cvrp import GROUP_STATE, GROUP_ENVIRONMENT
from utils.utils import augment_xy_data_by_8_fold

########################################
# ACTOR
########################################

class ACTOR_CVRP(nn.Module):

    def __init__(self, opts):
        super(ACTOR_CVRP, self).__init__()

        self.opts = opts
        self.box_select_probabilities = None
        # shape = (batch, group, TSP_SIZE)
        self.problem = opts.problem
        self.encoder = Encoder(opts)
        self.node_prob_calculator = Next_Node_Probability_Calculator_for_group(opts)
        self.opts = opts
        self.graph_size = opts.graph_size
        self.PROB_MAX_LENGTH = {
            4: 8,
            20: 38,
            50: 99,
            100: 189,
            105: 200,
            109: 210,
            114: 215,
            119: 220,
            124: 225,
            128: 225,
            133: 230,
            138: 230,
            142: 235,
            147: 240,
            152: 240,
            156: 240,
            161: 245,
            166: 250,
            171: 260,
            175: 260,
            180: 265,
            185: 265,
            189: 270,
            194: 275,
            199: 280
        }

        self.batch_s = None
        self.encoded_graph = None

    def torch_load_cpu(self, load_path):
        return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

    def get_inner_model(self, model):
        return model.module if isinstance(model, DataParallel) else model


    def get_init_hidden(self):
        return self.enc_init_hx

    def forward(self, data, encoded_nodes, eval_only=False):
        depot_xy = data['depot']  # bs, 1, 2
        node_xy = data['loc']  # bs, cvrp_size, 2
        node_demand = data['demand'].unsqueeze(-1)  # bs, cvrp_size, 1

        batch_s = depot_xy.size(0)
        env = GROUP_ENVIRONMENT(depot_xy, node_xy, node_demand, self.opts)
        group_s = self.graph_size

        group_state, reward, done = env.reset(group_size=group_s)
        self.encoded_graph = encoded_nodes.mean(dim=1, keepdim=True)
        self.node_prob_calculator.reset(encoded_nodes)

        first_action = LongTensor(np.zeros((batch_s, group_s)))  # start from node_0-depot
        group_state, reward_step, solution, done = env.step(first_action)
        second_action = LongTensor(np.arange(group_s) + 1)[None, :].expand(batch_s, group_s)
        group_state, reward_step, solution, done = env.step(second_action)

        group_prob_list = torch.zeros((batch_s, group_s, 0), device=self.opts.device)

        while not done:
            action_probs = self.get_action_probabilities(group_state, encoded_nodes)
            # shape = (batch, group, problem+1)
            action_probs_topk, indices = action_probs.topk(3, dim=-1, largest=True)
            action = action_probs_topk.reshape(batch_s * group_s, -1).multinomial(1) \
                .squeeze(dim=1).reshape(batch_s, group_s)
            action = indices.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            # action = action_probs.reshape(batch_s * group_s, -1).multinomial(1) \
            #     .squeeze(dim=1).reshape(batch_s, group_s)
            # action = action_probs.argmax(dim=2)
            # shape = (batch, group)

            action[group_state.finished] = 0  # stay at depot, if you are finished
            group_state, reward_step, solution, done = env.step(action)

            batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
            group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
            chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
            # shape = (batch, group)
            chosen_action_prob[group_state.finished] = 1  # done episode will gain no more probability
            group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)
            # shape = (batch, group, x)

        zero_tensor_prob = torch.ones((group_prob_list.size(0), group_prob_list.size(1),
                                   self.PROB_MAX_LENGTH[self.graph_size] - group_prob_list.size(-1)), dtype=torch.float32,
                                  device=self.opts.device)
        group_prob_list = torch.cat((group_prob_list, zero_tensor_prob), -1)

        return solution, reward_step, group_prob_list

    def get_action_probabilities(self, group_state, encoded_nodes):
        encoded_LAST_NODES = pick_nodes_for_each_group(encoded_nodes, group_state.current_node, self.opts)
        # shape = (batch, group, EMBEDDING_DIM)
        remaining_loaded = group_state.loaded[:, :, None]
        # shape = (batch, group, 1)

        item_select_probabilities = self.node_prob_calculator(self.encoded_graph, encoded_LAST_NODES,
                                                              remaining_loaded, ninf_mask=group_state.ninf_mask)
        # shape = (batch, group, problem+1)

        return item_select_probabilities


########################################
# ACTOR_SUB_NN : ENCODER
########################################

class Encoder(nn.Module):
    def __init__(self, opts):
        super().__init__()

        self.embedding_dim = opts.embedding_dim
        self.encoder_layer_num = opts.n_encode_layers

        self.embedding_depot = nn.Linear(2, self.embedding_dim)
        self.embedding_node = nn.Linear(3, self.embedding_dim)
        self.layers = nn.ModuleList([Encoder_Layer(opts) for _ in range(self.encoder_layer_num)])

    def forward(self, data):
        # data.shape = (batch, problem+1, 3)

        depot_xy = data[:, [0], 0:2]
        # shape = (batch, 1, 2)
        node_xy_demand = data[:, 1:, 0:3]
        # shape = (batch, problem, 3)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape = (batch, 1, EMBEDDING_DIM)
        embedded_node = self.embedding_node(node_xy_demand)
        # shape = (batch, problem, EMBEDDING_DIM)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape = (batch, problem+1, EMBEDDING_DIM)

        for layer in self.layers:
            out = layer(out)

        return out



class Encoder_Layer(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.embedding_dim = opts.embedding_dim
        self.head_dim = opts.encoder_head_num
        self.key_dim = self.embedding_dim // self.head_dim
        self.Wq = nn.Linear(self.embedding_dim, self.head_dim * self.key_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, self.head_dim * self.key_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.head_dim * self.key_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_dim * self.key_dim, self.embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(opts)
        self.feedForward = Feed_Forward_Module(opts)
        self.addAndNormalization2 = Add_And_Normalization_Module(opts)

    def forward(self, input1):
        # input.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        q = reshape_by_heads(self.Wq(input1), head_num=self.head_dim)
        k = reshape_by_heads(self.Wk(input1), head_num=self.head_dim)
        v = reshape_by_heads(self.Wv(input1), head_num=self.head_dim)
        # q shape = (batch_s, HEAD_NUM, TSP_SIZE, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape = (batch_s, TSP_SIZE, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3


########################################
# ACTOR_SUB_NN : Next_Node_Probability_Calculator
########################################

class Next_Node_Probability_Calculator_for_group(nn.Module):
    def __init__(self, opts):
        super().__init__()

        self.embedding_dim = opts.embedding_dim
        self.head_dim = opts.encoder_head_num
        self.key_dim = self.embedding_dim // self.head_dim
        self.logit_clipping = opts.tanh_clipping

        self.Wq = nn.Linear(2*self.embedding_dim+1, self.head_dim * self.key_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, self.head_dim * self.key_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.head_dim * self.key_dim, bias=False)

        self.multi_head_combine = nn.Linear(self.head_dim * self.key_dim, self.embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

    def reset(self, encoded_nodes):
        # encoded_nodes.shape = (batch, problem+1, EMBEDDING_DIM)

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=self.head_dim)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=self.head_dim)
        # shape = (batch, HEAD_NUM, problem+1, KEY_DIM)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape = (batch, EMBEDDING_DIM, problem+1)

    def forward(self, input1, input2, remaining_loaded, ninf_mask=None):
        # input1.shape = (batch, 1, EMBEDDING_DIM)  mean_pooling
        # input2.shape = (batch, group, EMBEDDING_DIM)  last_node_embedding
        # remaining_loaded.shape = (batch, group, 1)
        # ninf_mask.shape = (batch, group, problem+1)

        group_s = input2.size(1)

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((input1.expand(-1, group_s, -1), input2, remaining_loaded), dim=2)
        # shape = (batch, group, 2*EMBEDDING_DIM+1)

        q = reshape_by_heads(self.Wq(input_cat), head_num=self.head_dim)
        # shape = (batch, HEAD_NUM, group, KEY_DIM)

        out_concat = multi_head_attention(q, self.k, self.v, ninf_mask=ninf_mask)
        # shape = (batch, n, HEAD_NUM*KEY_DIM)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape = (batch, n, EMBEDDING_DIM)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape = (batch, n, problem+1)

        score_scaled = score / np.sqrt(self.embedding_dim)
        # shape = (batch_s, group, problem+1)

        score_clipped = self.logit_clipping * torch.tanh(score_scaled)

        if ninf_mask is None:
            score_masked = score_clipped
        else:
            score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape = (batch, group, problem+1)

        return probs

########################################
# NN SUB CLASS / FUNCTIONS
########################################

def pick_nodes_for_each_group(encoded_nodes, node_index_to_pick, opts):
    # encoded_nodes.shape = (batch, problem, EMBEDDING_DIM)
    # node_index_to_pick.shape = (batch, group)

    gathering_index = node_index_to_pick[:, :, None].expand(-1, -1, opts.embedding_dim)
    # shape = (batch, group, EMBEDDING_DIM)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape = (batch, group, EMBEDDING_DIM)

    return picked_nodes


def reshape_by_heads(qkv, head_num):
    # q.shape = (batch, C, head_num*key_dim)

    batch_s = qkv.size(0)
    C = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, C, head_num, -1)
    # shape = (batch, C, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape = (batch, head_num, C, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, ninf_mask=None):
    # q shape = (batch, head_num, n, key_dim)   : n can be either 1 or group
    # k,v shape = (batch, head_num, problem, key_dim)
    # ninf_mask.shape = (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    problem_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape = (batch, head_num, n, problem)

    score_scaled = score / np.sqrt(key_dim)
    if ninf_mask is not None:
        score_scaled = score_scaled + ninf_mask[:, None, :, :].expand(batch_s, head_num, n, problem_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape = (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape = (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape = (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape = (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.embedding_dim = opts.embedding_dim

        self.norm_by_EMB = nn.BatchNorm1d(self.embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape = (batch, problem, EMBEDDING_DIM)
        batch_s = input1.size(0)
        problem_s = input1.size(1)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, self.embedding_dim))

        return normalized.reshape(batch_s, problem_s, self.embedding_dim)


class Feed_Forward_Module(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.embedding_dim = opts.embedding_dim
        self.ff_hidden_dim = opts.feed_forward_dim
        self.W1 = nn.Linear(self.embedding_dim, self.ff_hidden_dim)
        self.W2 = nn.Linear(self.ff_hidden_dim, self.embedding_dim)

    def forward(self, input1):
        # input.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        return self.W2(F.relu(self.W1(input1)))
