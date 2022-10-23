from torch import nn
from nets.graph_encoder import MultiHeadAttentionLayer, EmbeddingNet_Loc, ValueDecoder, MultiHeadCompat
from nets.actor_network import mySequential
import torch


class Critic(nn.Module):

    def __init__(self,
                 problem_name,
                 embedding_dim,
                 hidden_dim,
                 n_heads,
                 n_layers,
                 normalization,
                 graph_size,
                 device
                 ):

        super(Critic, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.normalization = normalization
        self.allow_partial = problem_name == 'sdvrp'
        self.is_vrp = problem_name == 'cvrp' or problem_name == 'sdvrp'
        self.device = device

        # Problem specific placeholders
        if problem_name == 'pdp':
            self.node_dim = 4  # (x, y)_{i-1}, (x, y)_{i},  (x, y)_{i+1}
        elif self.is_vrp:
            # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
            self.step_context_dim = embedding_dim + 1
            self.node_dim = 3  # x, y, demand / prize

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)

            if self.is_vrp and self.allow_partial:  # Need to include the demand if split delivery allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
        elif problem_name == 'tsp':
            self.node_dim = 2
        else:
            assert False, "Unsupported problem: {}".format(self.problem.NAME)
        self.edge_dim = 1

        # # networks
        # self.embedder = EmbeddingNet_Loc(
        #     self.node_dim,
        #     self.embedding_dim,
        #     self.device)

        # self.encoder = mySequential(*(
        #     GraphEncoder(self.n_heads,
        #                      self.embedding_dim,
        #                      self.hidden_dim,
        #                      self.normalization,
        #                      )
        #     for _ in range(self.n_layers - 1)))  # n_layers - 1

        # self.compater_solution = MultiHeadCompat(2,  # heads for current solution
        #                                          self.embedding_dim,
        #                                          self.embedding_dim)
        # self.compater_best_solution = MultiHeadCompat(2,  # heads for best solution
        #                                               self.embedding_dim,
        #                                               self.embedding_dim)

        self.value_head = ValueDecoder(n_heads=n_heads,
                                       embed_dim=self.embedding_dim,
                                       input_dim=self.hidden_dim,
                                       graph_size=graph_size)

    def forward(self, AM_embedding_new):
        """
        :param inputs: (batch_size, graph_size, input_dim)
        :return:
        """

        # # pass through embedderc
        # loc_embed, _, _ = self.embedder(x, solutions, best_solutions)  # solutions: [bs, seq_len, embed_dim]

        # pos_compatibility = self.compater_solution(pos_embed)
        # best_pos_compatibility = self.compater_solution(best_pos_embed)
        # node_embed = self.encoder(loc_embed)

        # pass through value_head, get estimated value
        baseline_value = self.value_head(AM_embedding_new.detach())

        return baseline_value.detach().squeeze(), \
               baseline_value.squeeze()
