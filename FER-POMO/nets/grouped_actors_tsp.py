
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


# Hyper Parameters
from HYPER_PARAMS import *
from TORCH_OBJECTS import *
from travelling_saleman_problem import GROUP_STATE, GROUP_ENVIRONMENT
from utils import torch_load_cpu, get_inner_model
from utils.utils import augment_xy_data_by_8_fold

########################################
# ACTOR
########################################

class ACTOR_TSP(nn.Module):

    def __init__(self, opts):
        super(ACTOR_TSP, self).__init__()
        self.box_select_probabilities = None
        # shape = (batch, group, TSP_SIZE)

        self.encoder = Encoder(opts)
        self.node_prob_calculator = Next_Node_Probability_Calculator_for_group(opts)
        self.opts = opts
        self.graph_size = opts.graph_size

        self.batch_s = None


    def torch_load_cpu(self, load_path):
        return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

    def get_inner_model(self, model):
        return model.module if isinstance(model, DataParallel) else model


    def get_init_hidden(self):
        return self.enc_init_hx

    def forward(self, data, encoded_nodes, eval_only=False):

        batch_s = data.size(0)
        env = GROUP_ENVIRONMENT(data, self.opts)
        group_s = self.graph_size
        first_action = torch.LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s).to(device)

        group_state, reward_step, done = env.reset(group_size=group_s)

        self.node_prob_calculator.reset(encoded_nodes, group_ninf_mask=group_state.ninf_mask)

        group_state, reward_step, solution, done = env.step(first_action)
        group_prob_list = torch.zeros((batch_s, group_s, 0), device=self.opts.device)

        while not done:
            self.update(group_state, encoded_nodes)
            action_probs = self.get_action_probabilities()  # (batch, group, TSP_SIZE)
            action_probs_topk, indices = action_probs.topk(3, dim=-1, largest=True)
            action = action_probs_topk.reshape(batch_s * group_s, -1).multinomial(1) \
                .squeeze(dim=1).reshape(batch_s, group_s)
            action = indices.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            # action = action_probs.reshape(batch_s * group_s, -1).multinomial(1).squeeze(dim=1).reshape(batch_s, group_s)
            # shape = (batch, group)
            group_state, reward_step, solution, done = env.step(action)

            batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
            group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
            chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
            # shape = (batch, group)
            group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)  # bs, gs, tsp_size-1

        return solution, reward_step, group_prob_list

    def update(self, group_state, encoded_nodes):
        encoded_LAST_NODES = pick_nodes_for_each_group(encoded_nodes, group_state.current_node, self.opts)
        # shape = (batch_s, group, EMBEDDING_DIM)

        probs = self.node_prob_calculator(encoded_LAST_NODES)

        # shape = (batch_s, group, TSP_SIZE)
        self.box_select_probabilities = probs

    def get_action_probabilities(self):
        return self.box_select_probabilities


########################################
# ACTOR_SUB_NN : ENCODER
########################################

class Encoder(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.embedding_dim = opts.embedding_dim
        self.encoder_layer_num = opts.n_encode_layers
        self.embedding = nn.Linear(2, self.embedding_dim)
        self.layers = nn.ModuleList([Encoder_Layer(opts) for _ in range(self.encoder_layer_num)])

    def forward(self, data):
        # data.shape = (batch_s, TSP_SIZE, 2)

        embedded_input = self.embedding(data)
        # shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        out = embedded_input
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

        out_concat = multi_head_attention(q, k, v, self.opts)
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
        self.opts = opts
        self.embedding_dim = opts.embedding_dim
        self.head_dim = opts.encoder_head_num
        self.key_dim = self.embedding_dim // self.head_dim
        self.logit_clipping = opts.tanh_clipping

        self.Wq_graph = nn.Linear(self.embedding_dim, self.head_dim * self.key_dim, bias=False)
        self.Wq_first = nn.Linear(self.embedding_dim, self.head_dim * self.key_dim, bias=False)
        self.Wq_last = nn.Linear(self.embedding_dim, self.head_dim * self.key_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, self.head_dim * self.key_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.head_dim * self.key_dim, bias=False)

        self.multi_head_combine = nn.Linear(self.head_dim * self.key_dim, self.embedding_dim)

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_first = None  # saved q2, for multi-head attention
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.group_ninf_mask = None  # reference to ninf_mask owned by state

    def reset(self, encoded_nodes, group_ninf_mask):
        # encoded_nodes.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        encoded_graph = encoded_nodes.mean(dim=1, keepdim=True)
        # shape = (batch_s, 1, EMBEDDING_DIM)
        self.q_graph = reshape_by_heads(self.Wq_graph(encoded_graph), head_num=self.head_dim)
        # shape = (batch_s, HEAD_NUM, 1, KEY_DIM)
        self.q_first = None
        # shape = (batch_s, HEAD_NUM, group, KEY_DIM)
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=self.head_dim)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=self.head_dim)
        # shape = (batch_s, HEAD_NUM, TSP_SIZE, KEY_DIM)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape = (batch_s, EMBEDDING_DIM, TSP_SIZE)
        self.group_ninf_mask = group_ninf_mask
        # shape = (batch_s, group, TSP_SIZE)

    def forward(self, encoded_LAST_NODE):
        # encoded_LAST_NODE.shape = (batch_s, group, EMBEDDING_DIM)

        if self.q_first is None:
            self.q_first = reshape_by_heads(self.Wq_first(encoded_LAST_NODE), head_num=self.head_dim)
        # shape = (batch_s, HEAD_NUM, group, KEY_DIM)

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_LAST_NODE), head_num=self.head_dim)
        # shape = (batch_s, HEAD_NUM, group, KEY_DIM)

        q = self.q_graph + self.q_first + q_last
        # shape = (batch_s, HEAD_NUM, group, KEY_DIM)

        out_concat = multi_head_attention(q, self.k, self.v, self.opts, group_ninf_mask=self.group_ninf_mask)
        # shape = (batch_s, group, HEAD_NUM*KEY_DIM)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape = (batch_s, group, EMBEDDING_DIM)

        #  Single-Head Attention, for probability calculation
        #######################################################      
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape = (batch_s, group, TSP_SIZE)

        score_scaled = score / np.sqrt(self.embedding_dim)
        # shape = (batch_s, group, TSP_SIZE)

        score_clipped = self.logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + self.group_ninf_mask.clone()

        probs = F.softmax(score_masked, dim=2)
        # shape = (batch_s, group, TSP_SIZE)
        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def pick_nodes_for_each_group(encoded_nodes, node_index_to_pick, opts):
    # encoded_nodes.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)
    # node_index_to_pick.shape = (batch_s, group_s)
    batch_s = node_index_to_pick.size(0)
    group_s = node_index_to_pick.size(1)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_s, group_s, opts.embedding_dim)

    # shape = (batch_s, group, EMBEDDING_DIM)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape = (batch_s, group, EMBEDDING_DIM)

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


def multi_head_attention(q, k, v, opts, ninf_mask=None, group_ninf_mask=None):
    # q shape = (batch_s, head_num, n, key_dim)   : n(group) can be either 1 or TSP_SIZE
    # k,v shape = (batch_s, head_num, TSP_SIZE, key_dim)
    # ninf_mask.shape = (batch_s, TSP_SIZE)
    # group_ninf_mask.shape = (batch_s, group, TSP_SIZE)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape = (batch_s, head_num, n, TSP_SIZE)

    score_scaled = score / np.sqrt(key_dim)
    if ninf_mask is not None:
        score_scaled = score_scaled + ninf_mask[:, None, None, :].expand(batch_s, head_num, n, opts.graph_size)
    if group_ninf_mask is not None:
        score_scaled = score_scaled + group_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, opts.graph_size)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape = (batch_s, head_num, n, TSP_SIZE)

    out = torch.matmul(weights, v)
    # shape = (batch_s, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape = (batch_s, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape = (batch_s, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.embedding_dim = opts.embedding_dim
        self.graph_size = opts.graph_size
        self.norm_by_EMB = nn.BatchNorm1d(self.embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        batch_s = input1.size(0)
        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * self.graph_size, self.embedding_dim))

        return normalized.reshape(batch_s, self.graph_size, self.embedding_dim)


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
