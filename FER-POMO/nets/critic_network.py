from torch import nn
from nets.graph_encoder import ValueDecoder



class Critic(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_heads,
                 graph_size
                 ):

        super(Critic, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.value_head = ValueDecoder(n_heads=n_heads,
                                       embed_dim=self.embedding_dim,
                                       input_dim=self.hidden_dim,
                                       graph_size=graph_size)

    def forward(self, AM_embedding_new):
        """
        :param inputs: (batch_size, graph_size, input_dim)
        :return:
        """

        # pass through value_head, get estimated value
        baseline_value = self.value_head(AM_embedding_new.detach())

        return baseline_value.detach(), baseline_value
