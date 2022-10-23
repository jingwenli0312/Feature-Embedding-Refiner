import nets.grouped_actors_tsp as TSP_Module
import nets.grouped_actors_cvrp as CVRP_Module
import torch.nn as nn
import math
import torch

from HYPER_PARAMS import *
from nets.graph_encoder import GAT_Solution
from torch.nn import DataParallel
from utils import torch_load_cpu, get_inner_model


class ACTOR_RECONSTRUCT(nn.Module):

    def __init__(self, opts):
        super(ACTOR_RECONSTRUCT, self).__init__()

        self.box_select_probabilities = None
        # shape = (batch, group, TSP_SIZE)
        self.problem = opts.problem
        self.embedding_dim = opts.embedding_dim
        self.hidden_edge_dim = opts.hidden_edge_dim
        self.head_num = opts.encoder_head_num
        self.alpha = opts.alpha
        self.graph_size = opts.graph_size
        self.opts = opts

        """Trainable initial hidden state"""
        std = 1. / math.sqrt(HEAD_NUM)
        enc_init_hx = torch.nn.Parameter(torch.FloatTensor(EMBEDDING_DIM))
        enc_init_hx.data.uniform_(-std, std)
        self.enc_init_hx = enc_init_hx
        self.gat_solution = GAT_Solution(self.problem,
                                         self.embedding_dim,
                                         self.hidden_edge_dim,
                                         self.head_num,
                                         self.alpha,
                                         self.graph_size)
        self.cell = torch.nn.GRUCell(EMBEDDING_DIM, EMBEDDING_DIM, bias=True)

        if self.problem == 'tsp':
            self.POMO = TSP_Module.ACTOR_TSP(opts)
        elif self.problem == 'cvrp':
            self.POMO = CVRP_Module.ACTOR_CVRP(opts)


    def torch_load_cpu(self, load_path):
        return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

    def get_init_hidden(self):
        return self.enc_init_hx

    def forward(self, data, POMO_embedding, dist, solu_embed, history_solutions, history_costs, embedding_only=False):
        if self.problem == 'tsp':
            batch_s = data.size(0)
            graph_s = self.graph_size
        if self.problem == 'cvrp':
            batch_s = data['depot'].size(0)
            graph_s = self.graph_size + 1

        if solu_embed is None:
            solu_embed = self.get_init_hidden()
            solu_embed = solu_embed[None, :].repeat(batch_s * graph_s, 1)  # (bs*gs, embed_dim)

        # encoded_nodes_new = POMO_embedding
        # [bs, gs, embed_dim]
        # solu_embed = self.gat_solution(POMO_embedding, history_solutions.transpose(0, 1),
        #                                                history_costs.transpose(0, 1), dist, solu_embed)
        #
        # encoded_nodes_new = self.cell(solu_embed.view(-1, self.embedding_dim),
        #                               POMO_embedding.view(-1, self.embedding_dim)).view(batch_s, graph_s, -1)



        solu_embed_elu, solu_embed = self.gat_solution(POMO_embedding, history_solutions.transpose(0, 1),
                                                       history_costs.transpose(0, 1), dist, solu_embed)

        encoded_nodes_new = self.cell(solu_embed_elu.view(-1, self.embedding_dim),
                                           POMO_embedding.view(-1, self.embedding_dim)).view(batch_s, graph_s, -1)

        if embedding_only:
            return encoded_nodes_new

        solution, reward_step, group_prob_list = self.POMO(data, encoded_nodes_new)

        return solution, reward_step, group_prob_list, encoded_nodes_new, solu_embed
