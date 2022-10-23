import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention_AM(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention_AM, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)  # [batch_size * graph_size, input_dim]
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = F.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)
        return out


class MultiHeadAttention_LSTM(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
    ):
        super(MultiHeadAttention_LSTM, self).__init__()

        self.n_heads = n_heads // 2  # heads for location

        self.n_heads_V = n_heads
        if val_dim is None:
            val_dim = embed_dim // self.n_heads
        if key_dim is None:
            key_dim = val_dim


        # self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(1 * self.key_dim)  # See Attention is all you need

        self.W_query_node = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        # self.W_query_pos = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))

        self.W_key_node = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        # self.W_key_pos = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))

        self.W_val_node = nn.Parameter(torch.Tensor(self.n_heads_V, self.input_dim, self.key_dim))
        # self.W_val_pos = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))

        if embed_dim is not None:
            self.W_out_node = nn.Parameter(torch.Tensor(self.n_heads_V, self.key_dim, self.embed_dim))
            # self.W_out_pos = nn.Parameter(torch.Tensor(self.n_heads, self.key_dim, self.embed_dim))
        self.rnn = torch.nn.LSTM(n_heads + 2, n_heads, num_layers=1)

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h_node_in, pos_compatibility, best_pos_compatibility, cost, best_cost, h_x, c_x, mask=None):

        batch_size, graph_size, input_dim = h_node_in.size()

        h_node = h_node_in.contiguous().view(-1, input_dim)  # [batch_size * graph_size, embed_dim]
        # h_pos = h_pos_in.contiguous().view(-1, input_dim)

        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_V = (self.n_heads_V, batch_size, graph_size, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q_node = torch.matmul(h_node, self.W_query_node).view(shp)
        # Q_pos = torch.matmul(h_pos, self.W_query_pos).view(shp)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K_node = torch.matmul(h_node, self.W_key_node).view(shp)
        # K_pos = torch.matmul(h_pos, self.W_key_pos).view(shp)
        V_node = torch.matmul(h_node, self.W_val_node).view(shp_V)
        # V_pos = torch.matmul(h_pos, self.W_val_pos).view(shp)

        # (n_heads, batch_size, graph_size, graph_size)
        node_compatability = self.norm_factor * torch.matmul(Q_node, K_node.transpose(2, 3))
        # (2, batch_size, graph_size, graph_size)
        # pos_compatibility = self.compater_solution(h_pos_in)
        # best_pos_compatibility = self.compater_solution(h_pos_best)
        # pos_compatibility = self.norm_factor * torch.matmul(Q_pos, K_pos.transpose(2, 3))

        # node_compatability = node_compatability + pos_compatibility

        if mask is not None:
            mask = mask.view(1, batch_size, graph_size, graph_size).expand_as(node_compatability)
            node_compatability[mask] = -np.inf
            mask_pos = mask.view(1, batch_size, graph_size, graph_size).expand_as(pos_compatibility)
            pos_compatibility[mask_pos] = -np.inf
            best_pos_compatibility[mask_pos] = -np.inf
        # concat 3 compabitility matrix and reserve n_heads dim to LSTM  (gs*gs, bs, n_heads) 8
        compatibility = torch.cat((node_compatability, pos_compatibility, best_pos_compatibility), 0).reshape(graph_size*graph_size, batch_size, -1)
        # (gs*gs, bs, 10) add cost and best_cost
        compatibility = torch.cat((compatibility, cost[None, :, None].expand(graph_size*graph_size, batch_size, 1), best_cost[None, :, None].expand(graph_size*graph_size, batch_size, 1)), -1)

        node_compatability, (h_t, c_t) = self.rnn(compatibility, (h_x, c_x))  # node: (gs*gs, bs, 8)

        attn1 = F.softmax(node_compatability.reshape(-1, batch_size, graph_size, graph_size), dim=-1)
        if mask is not None:
            attnc = attn1.clone()
            attnc[mask] = 0
            attn1 = attnc
        heads_node = torch.matmul(attn1, V_node)  # [n_heads_V, bs, graph_size, val_size]
        # heads_pos = torch.matmul(attn2, V_pos)  # heads, bs, n, 16
        out_node = torch.mm(
            heads_node.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads_V * self.val_dim),
            self.W_out_node.view(-1, self.embed_dim)
        ).view(batch_size, graph_size, -1)

        return out_node, h_t, c_t


class MultiHeadCompat(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadCompat, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention
        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)  #################   reshape
        qflat = q.contiguous().view(-1, input_dim)  # [batch_size * graph_size, input_dim]

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        return compatibility  # (n_heads, batch_size, n_query, graph_size)


class MLP(torch.nn.Module):
    def __init__(self,
                 input_dim=128,
                 feed_forward_dim=64,
                 embedding_dim=64,
                 output_dim=1
                 ):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, feed_forward_dim)
        self.fc2 = torch.nn.Linear(feed_forward_dim, embedding_dim)
        self.fc3 = torch.nn.Linear(embedding_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=0.03)
        self.ReLU = nn.ReLU()

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, in_):  # [batch_size, graph_size, embed_dim+16]
        result = self.ReLU(self.fc1(in_))
        result = self.dropout(result)
        result = self.ReLU(self.fc2(result))
        result = self.fc3(result).squeeze(-1)  # [batch_size, graph_size]
        return result


class ValueDecoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            input_dim,
            graph_size
    ):
        super(ValueDecoder, self).__init__()
        self.hidden_dim = embed_dim
        self.embedding_dim = embed_dim
        self.n_heads = n_heads

        self.project_graph = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.project_node = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.MLP = MLP(input_dim + n_heads*2, embed_dim, output_dim=1)

    def forward(self, h_em, h_x, c_x):  # [1, bs, 8]
        # get embed feature
        mean_pooling = h_em.mean(1)  # mean Pooling
        graph_feature = self.project_graph(mean_pooling)[:, None, :]
        node_feature = self.project_node(h_em)

        # pass through value_head, get estimated value
        fusion = node_feature + graph_feature.expand_as(node_feature)  # torch.Size([2, 50, 128])

        # h_x = h_x.squeeze(0).unsqueeze(1).repeat(1, fusion.size(1), 1)  # [bs, gs, hidden_dim]
        # c_x = c_x.squeeze(0).unsqueeze(1).repeat(1, fusion.size(1), 1)

        value = self.MLP(torch.cat((fusion.mean(dim=1), h_x.squeeze(0), c_x.squeeze(0)), -1))

        return value

class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalization = normalization

        if not self.normalization == 'layer':
            self.normalizer = normalizer_class(embed_dim, affine=True)

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.normalization == 'layer':
            return (input - input.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(input.var((1, 2)).view(-1, 1, 1) + 1e-05)

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input

class MultiHeadEncoder_LSTM(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
            is_share_QK_base=False,
    ):
        super(MultiHeadEncoder_LSTM, self).__init__()

        self.MHA_sublayer = MultiHeadAttentionsubLayer_LSTM(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

        self.FFandNorm_sublayer = FFandNormsubLayer(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

    def forward(self, input1, input2, input3, rew, best_rew, h_x, c_x):
        out, h_t, c_t = self.MHA_sublayer(input1, input2, input3, rew, best_rew, h_x, c_x)
        return self.FFandNorm_sublayer(out), input2, input3, rew, best_rew, h_t, c_t


class MultiHeadAttentionsubLayer_LSTM(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer'
    ):
        super(MultiHeadAttentionsubLayer_LSTM, self).__init__()

        self.MHA = MultiHeadAttention_LSTM(
            n_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim
        )

        self.Norm = Normalization(embed_dim, normalization)


    def forward(self, input1, input2, input3, rew, best_rew, h_x, c_x):
        out1, h_t, c_t = self.MHA(input1, input2, input3, rew, best_rew, h_x, c_x)
        # Normalization
        return self.Norm(out1 + input1), h_t, c_t





class GraphEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
            is_share_QK_base=False,
    ):
        super(GraphEncoder, self).__init__()

        self.MHA_sublayer = MultiHeadAttentionsubLayer(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

        self.FFandNorm_sublayer = FFandNormsubLayer(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

    def forward(self, input):
        out = self.MHA_sublayer(input)
        return self.FFandNorm_sublayer(out)


class MultiHeadAttentionsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer'
    ):
        super(MultiHeadAttentionsubLayer, self).__init__()

        self.MHA = MultiHeadAttention_AM(
            n_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim
        )

        self.Norm = Normalization(embed_dim, normalization)

    def forward(self, input):
        out = self.MHA(input)
        return self.Norm(out + input)

class FFandNormsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(FFandNormsubLayer, self).__init__()

        self.FF1 = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, embed_dim)
        ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)

        self.FF2 = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, embed_dim)
        ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)

        self.Norm = Normalization(embed_dim, normalization)


    def forward(self, input1):
        # FF and Residual connection
        out1 = self.FF1(input1)
        # Normalization
        return self.Norm(out1 + input1)


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='batch'
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )



class EmbeddingNet(nn.Module):

    def __init__(
            self,
            node_dim,
            embedding_dim,
            device
    ):
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.device = device
        self.embedder = nn.Linear(node_dim, embedding_dim)

    def position_encoding_init(self, n_position, emb_dim):
        ####### need to change for depot
        ''' Init the sinusoid position encoding table '''
        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
            for pos in range(1, n_position + 1)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        return torch.from_numpy(position_enc).type(torch.FloatTensor)

    def position_encoding(self, x, enc_pattern, solutions, best_solutions, embedding_dim, device):
        # batch: batch_size, problem_size, dim
        batch_size, seq_length = solutions.size()

        # expand for every batch
        position_enc = enc_pattern.expand(batch_size, seq_length, embedding_dim).to(device)

        # get index according to the solutions
        # visited_time = torch.argsort(solutions, -1)  # [batch_size, seq_length]
        # get index according to the solutions
        index = [torch.nonzero(solutions.long() == i)[:, 1][:, None].expand(batch_size, embedding_dim)
                 for i in range(seq_length)]
        visited_time = torch.stack(index, 1)  # visit time

        best_index = [torch.nonzero(best_solutions.long() == i)[:, 1][:, None].expand(batch_size, embedding_dim)
                 for i in range(seq_length)]
        best_visited_time = torch.stack(best_index, 1)  # visit time

        return torch.gather(position_enc, 1,  # [bs, gs, embed_dim]
                            visited_time.long().expand(batch_size, seq_length, embedding_dim)).clone(), torch.gather(position_enc, 1,  # [bs, gs, embed_dim]
                 best_visited_time.long().expand(batch_size, seq_length, embedding_dim)).clone()

    def forward(self, x, solutions, best_solutions):
        batch_size, seq_length = solutions.size()
        enc_pattern = self.position_encoding_init(seq_length, self.embedding_dim)

        pos_enc, best_pos_enc = self.position_encoding(x, enc_pattern, solutions, best_solutions, self.embedding_dim, self.device)
        return self.embedder(x), pos_enc, best_pos_enc
