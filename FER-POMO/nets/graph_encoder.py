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

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

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

        # pickup
        self.W1_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W1_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W1_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W2_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W2_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W2_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W3_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W3_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W3_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        # delivery
        self.W4_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W4_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W4_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W5_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W5_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W5_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W6_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W6_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W6_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

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

        hflat = h.contiguous().view(-1, input_dim)  # [batch_size * graph_size, embed_dim]
        qflat = q.contiguous().view(-1, input_dim)  # [batch_size * n_query, embed_dim]

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # pickup -> its delivery attention
        n_pick = (graph_size - 1) // 2
        shp_delivery = (self.n_heads, batch_size, n_pick, -1)
        shp_q_pick = (self.n_heads, batch_size, n_pick, -1)

        # pickup -> all pickups attention
        shp_allpick = (self.n_heads, batch_size, n_pick, -1)
        shp_q_allpick = (self.n_heads, batch_size, n_pick, -1)

        # pickup -> all pickups attention
        shp_alldelivery = (self.n_heads, batch_size, n_pick, -1)
        shp_q_alldelivery = (self.n_heads, batch_size, n_pick, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # pickup -> its delivery
        pick_flat = h[:, 1:n_pick + 1, :].contiguous().view(-1, input_dim)  # [batch_size * n_pick, embed_dim]
        delivery_flat = h[:, n_pick + 1:, :].contiguous().view(-1, input_dim)  # [batch_size * n_pick, embed_dim]

        # pickup -> its delivery attention
        Q_pick = torch.matmul(pick_flat, self.W1_query).view(shp_q_pick)  # (self.n_heads, batch_size, n_pick, key_size)
        K_delivery = torch.matmul(delivery_flat, self.W_key).view(
            shp_delivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_delivery = torch.matmul(delivery_flat, self.W_val).view(
            shp_delivery)  # (n_heads, batch_size, n_pick, key/val_size)

        # pickup -> all pickups attention
        Q_pick_allpick = torch.matmul(pick_flat, self.W2_query).view(
            shp_q_allpick)  # (self.n_heads, batch_size, n_pick, -1)
        K_allpick = torch.matmul(pick_flat, self.W_key).view(
            shp_allpick)  # [self.n_heads, batch_size, n_pick, key_size]
        V_allpick = torch.matmul(pick_flat, self.W_val).view(
            shp_allpick)  # [self.n_heads, batch_size, n_pick, key_size]

        # pickup -> all delivery
        Q_pick_alldelivery = torch.matmul(pick_flat, self.W3_query).view(
            shp_q_alldelivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_alldelivery = torch.matmul(delivery_flat, self.W_key).view(
            shp_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_alldelivery = torch.matmul(delivery_flat, self.W_val).view(
            shp_alldelivery)  # (n_heads, batch_size, n_pick, key/val_size)

        # pickup -> its delivery
        V_additional_delivery = torch.cat([  # [n_heads, batch_size, graph_size, key_size]
            torch.zeros(self.n_heads, batch_size, 1, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device),
            V_delivery,  # [n_heads, batch_size, n_pick, key/val_size]
            torch.zeros(self.n_heads, batch_size, n_pick, self.input_dim // self.n_heads, dtype=V.dtype,
                        device=V.device)
        ], 2)

        # delivery -> its pickup attention
        Q_delivery = torch.matmul(delivery_flat, self.W4_query).view(
            shp_delivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_pick = torch.matmul(pick_flat, self.W_key).view(shp_q_pick)  # (self.n_heads, batch_size, n_pick, -1)
        V_pick = torch.matmul(pick_flat, self.W_val).view(shp_q_pick)  # (n_heads, batch_size, n_pick, key/val_size)

        # delivery -> all delivery attention
        Q_delivery_alldelivery = torch.matmul(delivery_flat, self.W5_query).view(
            shp_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        K_alldelivery2 = torch.matmul(delivery_flat, self.W_key).view(
            shp_alldelivery)  # [self.n_heads, batch_size, n_pick, key_size]
        V_alldelivery2 = torch.matmul(delivery_flat, self.W_val).view(
            shp_alldelivery)  # [self.n_heads, batch_size, n_pick, key_size]

        # delivery -> all pickup
        Q_delivery_allpickup = torch.matmul(delivery_flat, self.W6_query).view(
            shp_alldelivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_allpickup2 = torch.matmul(pick_flat, self.W_key).view(
            shp_q_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_allpickup2 = torch.matmul(pick_flat, self.W_val).view(
            shp_q_alldelivery)  # (n_heads, batch_size, n_pick, key/val_size)

        # delivery -> its pick up
        #        V_additional_pick = torch.cat([  # [n_heads, batch_size, graph_size, key_size]
        #            torch.zeros(self.n_heads, batch_size, 1, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device),
        #            V_delivery2,  # [n_heads, batch_size, n_pick, key/val_size]
        #            torch.zeros(self.n_heads, batch_size, n_pick, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device)
        #            ], 2)
        V_additional_pick = torch.cat([  # [n_heads, batch_size, graph_size, key_size]
            torch.zeros(self.n_heads, batch_size, 1, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device),
            torch.zeros(self.n_heads, batch_size, n_pick, self.input_dim // self.n_heads, dtype=V.dtype,
                        device=V.device),
            V_pick  # [n_heads, batch_size, n_pick, key/val_size]
        ], 2)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        ##Pick up
        # ??pair???attention??
        compatibility_pick_delivery = self.norm_factor * torch.sum(Q_pick * K_delivery,
                                                                   -1)  # element_wise, [n_heads, batch_size, n_pick]
        # [n_heads, batch_size, n_pick, n_pick]
        compatibility_pick_allpick = self.norm_factor * torch.matmul(Q_pick_allpick, K_allpick.transpose(2,
                                                                                                         3))  # [n_heads, batch_size, n_pick, n_pick]

        compatibility_pick_alldelivery = self.norm_factor * torch.matmul(Q_pick_alldelivery, K_alldelivery.transpose(2,
                                                                                                                     3))  # [n_heads, batch_size, n_pick, n_pick]

        ##Delivery
        compatibility_delivery_pick = self.norm_factor * torch.sum(Q_delivery * K_pick,
                                                                   -1)  # element_wise, [n_heads, batch_size, n_pick]

        compatibility_delivery_alldelivery = self.norm_factor * torch.matmul(Q_delivery_alldelivery,
                                                                             K_alldelivery2.transpose(2,
                                                                                                      3))  # [n_heads, batch_size, n_pick, n_pick]

        compatibility_delivery_allpick = self.norm_factor * torch.matmul(Q_delivery_allpickup, K_allpickup2.transpose(2,
                                                                                                                      3))  # [n_heads, batch_size, n_pick, n_pick]

        ##Pick up->
        # compatibility_additional?pickup????delivery????attention(size 1),1:n_pick+1??attention,depot?delivery??
        compatibility_additional_delivery = torch.cat([  # [n_heads, batch_size, graph_size, 1]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, dtype=compatibility.dtype, device=compatibility.device),
            compatibility_pick_delivery,  # [n_heads, batch_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device)
        ], -1).view(self.n_heads, batch_size, graph_size, 1)

        compatibility_additional_allpick = torch.cat([  # [n_heads, batch_size, graph_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device),
            compatibility_pick_allpick,  # [n_heads, batch_size, n_pick, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device)
        ], 2).view(self.n_heads, batch_size, graph_size, n_pick)

        compatibility_additional_alldelivery = torch.cat([  # [n_heads, batch_size, graph_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device),
            compatibility_pick_alldelivery,  # [n_heads, batch_size, n_pick, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device)
        ], 2).view(self.n_heads, batch_size, graph_size, n_pick)
        # [n_heads, batch_size, n_query, graph_size+1+n_pick+n_pick]

        ##Delivery->
        compatibility_additional_pick = torch.cat([  # [n_heads, batch_size, graph_size, 1]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, dtype=compatibility.dtype, device=compatibility.device),
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device),
            compatibility_delivery_pick  # [n_heads, batch_size, n_pick]
        ], -1).view(self.n_heads, batch_size, graph_size, 1)

        compatibility_additional_alldelivery2 = torch.cat([  # [n_heads, batch_size, graph_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device),
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device),
            compatibility_delivery_alldelivery  # [n_heads, batch_size, n_pick, n_pick]
        ], 2).view(self.n_heads, batch_size, graph_size, n_pick)

        compatibility_additional_allpick2 = torch.cat([  # [n_heads, batch_size, graph_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device),
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device),
            compatibility_delivery_allpick  # [n_heads, batch_size, n_pick, n_pick]
        ], 2).view(self.n_heads, batch_size, graph_size, n_pick)

        compatibility = torch.cat([compatibility, compatibility_additional_delivery, compatibility_additional_allpick,
                                   compatibility_additional_alldelivery,
                                   compatibility_additional_pick, compatibility_additional_alldelivery2,
                                   compatibility_additional_allpick2], dim=-1)

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility,
                             dim=-1)  # [n_heads, batch_size, n_query, graph_size+1+n_pick*2] (graph_size include depot)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc
        # heads: [n_heads, batrch_size, n_query, val_size], attn????pick?deliver?attn
        heads = torch.matmul(attn[:, :, :, :graph_size], V)  # V: (self.n_heads, batch_size, graph_size, val_size)

        # heads??pick -> its delivery
        heads = heads + attn[:, :, :, graph_size].view(self.n_heads, batch_size, graph_size,
                                                       1) * V_additional_delivery  # V_addi:[n_heads, batch_size, graph_size, key_size]

        # heads??pick -> otherpick, V_allpick: # [n_heads, batch_size, n_pick, key_size]
        # heads: [n_heads, batch_size, graph_size, key_size]
        heads = heads + torch.matmul(
            attn[:, :, :, graph_size + 1:graph_size + 1 + n_pick].view(self.n_heads, batch_size, graph_size, n_pick),
            V_allpick)

        # V_alldelivery: # (n_heads, batch_size, n_pick, key/val_size)
        heads = heads + torch.matmul(
            attn[:, :, :, graph_size + 1 + n_pick:graph_size + 1 + 2 * n_pick].view(self.n_heads, batch_size,
                                                                                    graph_size, n_pick), V_alldelivery)

        # delivery
        heads = heads + attn[:, :, :, graph_size + 1 + 2 * n_pick].view(self.n_heads, batch_size, graph_size,
                                                                        1) * V_additional_pick

        heads = heads + torch.matmul(
            attn[:, :, :, graph_size + 1 + 2 * n_pick + 1:graph_size + 1 + 3 * n_pick + 1].view(self.n_heads,
                                                                                                batch_size, graph_size,
                                                                                                n_pick), V_alldelivery2)

        heads = heads + torch.matmul(
            attn[:, :, :, graph_size + 1 + 3 * n_pick + 1:].view(self.n_heads, batch_size, graph_size, n_pick),
            V_allpickup2)

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

        self.n_heads = n_heads  # heads for location

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
        self.rnn = torch.nn.LSTM(n_heads // 2, n_heads // 2, num_layers=1)
        self.current_pos_project = nn.Linear(n_heads // 2 + 1, n_heads // 2, bias=False)
        # self.best_pos_project = nn.Linear(n_heads // 4 + 1, n_heads // 4, bias=False)

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

        V_node = torch.matmul(h_node, self.W_val_node).view(shp_V)
        pos_compatibility = torch.cat((pos_compatibility.permute(2, 3, 1, 0).reshape(1, batch_size*graph_size*graph_size, -1),
                                       cost[None, :, None].repeat(1, graph_size*graph_size, 1)), -1)  # [1, bs*gs*gs, 2]
        compatibility = self.current_pos_project(pos_compatibility)

        compatability_after, (h_t, c_t) = self.rnn(compatibility, (h_x, c_x))  # node: (gs*gs, bs, 4)

        # attn1 = F.softmax(torch.cat((node_compatability, compatability.reshape(graph_size, graph_size, batch_size, -1).permute(3, 2, 0, 1),
        #                              ), 0), dim=-1)
        attn1 = F.softmax(torch.cat((compatibility, compatability_after), -1).reshape(graph_size, graph_size, batch_size, -1).permute(3, 2, 0, 1), dim=-1)

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
        # self.dropout = torch.nn.Dropout(p=0.03)
        self.ReLU = nn.ReLU()

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, in_):  # [batch_size, graph_size, embed_dim+16]
        result = self.ReLU(self.fc1(in_))
        # result = self.dropout(result)
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

        self.MLP = MLP(input_dim, embed_dim, output_dim=1)
        # self.MLP_hidden = MLP(n_heads*2, n_heads*4, output_dim=1)

    def forward(self, h_em):  # [1, bs, 4]
        # get embed feature
        mean_pooling = h_em.mean(1)  # mean Pooling
        graph_feature = self.project_graph(mean_pooling)[:, None, :]
        node_feature = self.project_node(h_em)

        # pass through value_head, get estimated value
        fusion = node_feature + graph_feature.expand_as(node_feature)  # torch.Size([2, 50, 128])

        # value = self.MLP(torch.cat((fusion.mean(dim=1), h_x.squeeze(0), c_x.squeeze(0)), -1))
        # value = self.MLP(torch.cat((fusion.mean(dim=1), h_x.squeeze(0).view(fusion.size(0), fusion.size(1), fusion.size(1), -1).mean(1).mean(1),
        #                             c_x.squeeze(0).view(fusion.size(0), fusion.size(1), fusion.size(1), -1).mean(1).mean(1)), -1))
        # return value

        value = self.MLP(fusion.mean(dim=1))  # [bs]
        # return value
        # value_hidden = self.MLP_hidden(torch.cat((h_x.squeeze(0), c_x.squeeze(0)), -1))
        # value_hidden = self.MLP_hidden(torch.cat((h_x.squeeze(0).view(fusion.size(0), fusion.size(1), fusion.size(1), -1).mean(1).mean(1),
        #                             c_x.squeeze(0).view(fusion.size(0), fusion.size(1), fusion.size(1), -1).mean(1).mean(1)), -1))
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
        return self.Norm(out1 + input1), h_t, c_t  # out1 + input1


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

        self.Norm = Normalization(embed_dim, normalization)


    def forward(self, input1):
        # FF and Residual connection
        out1 = self.FF1(input1)
        # Normalization
        return self.Norm(out1 + input1)

######
class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            problem,
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
                ) if problem.NAME == 'pdp' else
                MultiHeadAttention_AM(
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

class EmbeddingNet_Loc(nn.Module):

    def __init__(
            self,
            node_dim,
            embedding_dim,
            device
    ):
        super(EmbeddingNet_Loc, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.device = device
        self.embedder = nn.Linear(node_dim, embedding_dim, bias=False)

    def forward(self, x):
        return self.embedder(x)

class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            problem,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(problem, n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h = self.layers(h)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


class EmbeddingNet_Pos(nn.Module):

    def __init__(
            self,
            problem,
            node_dim,
            embedding_dim,
            device
    ):
        super(EmbeddingNet_Pos, self).__init__()
        self.problem = problem
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.device = device

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
        position_enc = enc_pattern.expand(batch_size, seq_length, embedding_dim)

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
        return pos_enc, best_pos_enc


class GAT_Solution(nn.Module):
    def __init__(self, problem, embed_dim, hidden_edge_dim, n_heads, alpha, graph_size):
        """Dense version of GAT."""
        super(GAT_Solution, self).__init__()

        self.attentions = SolutionAttentionLayer(problem, embed_dim, hidden_edge_dim, n_heads, graph_size, alpha=alpha, concat=True)

    def forward(self, node_embed, solution, costs, dist, solution_embed_old):
        # node_embed = F.dropout(node_embed, self.dropout, training=self.training)
        solu_embed_elu, solu_embed = self.attentions(node_embed, solution, costs, dist, solution_embed_old)  # (gs, embed_dim)

        return solu_embed_elu, solu_embed  # (bs, gs, embed_dim)

class SolutionAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, problem, embedding_dim, hidden_edge_dim, n_heads, graph_size, alpha, concat=False):
        super(SolutionAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.key_dim = embedding_dim // n_heads
        self.concat = concat
        self.embed_dim = embedding_dim
        self.problem = problem
        self.norm_factor = 1 / math.sqrt(self.key_dim)  # See Attention is all you need
        self.sqrt_qkv_dim = self.key_dim ** (1 / 2)
        self.mix1_init = (1 / 2) ** (1 / 2)
        self.mix2_init = (1 / 16) ** (1 / 2)
        self.ms_hidden_dim = 16

        self.Wq = nn.Linear(embedding_dim, n_heads * self.key_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, n_heads * self.key_dim, bias=False)
        mix1_weight = torch.torch.distributions.Uniform(low=-self.mix1_init, high=self.mix1_init).sample(
            (n_heads, 2, self.ms_hidden_dim))
        mix1_bias = torch.torch.distributions.Uniform(low=-self.mix1_init, high=self.mix1_init).sample(
            (n_heads, self.ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        # shape: (head, 2, ms_hidden)
        self.mix1_bias = nn.Parameter(mix1_bias)
        # shape: (head, ms_hidden)

        mix2_weight = torch.torch.distributions.Uniform(low=-self.mix2_init, high=self.mix2_init).sample(
            (n_heads, self.ms_hidden_dim, 1))
        mix2_bias = torch.torch.distributions.Uniform(low=-self.mix2_init, high=self.mix2_init).sample((n_heads, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        # shape: (head, ms_hidden, 1)
        self.mix2_bias = nn.Parameter(mix2_bias)
        # shape: (head, 1)

        self.norm_head = nn.Linear(n_heads, 1, bias=False)
        self.cell = torch.nn.GRUCell(embedding_dim, embedding_dim, bias=True)

    def mix_attn(self, q, k, edge):
        # q shape: (num_solu, bs, n_head, gs, key_dim)
        # k shape: (num_solu, bs, n_head, gs, key_dim)
        # cost_mat.shape: (num_solu, batch, gs)

        num_solu, bs, n_heads, gs, key_dim = q.size()

        dot_product = self.norm_factor * torch.sum(q * k, -1)
        # shape: (num_solu, bs, n_head, gs)

        dot_product_score = dot_product / self.sqrt_qkv_dim
        # shape: (num_solu, bs, n_head, gs)

        edge_score = edge[:, :, None, :].expand(num_solu, bs, n_heads, gs)
        # shape: (num_solu, bs, n_head, gs)

        two_scores = torch.stack((dot_product_score, edge_score), dim=-1)
        # shape: (num_solu, bs, n_head, gs, 2)

        two_scores = two_scores.unsqueeze(4).transpose(2, 3)  # shape: (num_solu, bs, gs, n_head, 1, 2)

        ms1 = torch.matmul(two_scores, self.mix1_weight)  # shape: (num_solu, bs, gs, n_head, 1, ms_hidden)
        # shape: (num_solu, bs, n_head, gs, ms_hidden)

        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]  # shape: (num_solu, bs, gs, n_head, 1, ms_hidden)
        # shape: (num_solu, bs, n_head, gs, ms_hidden)

        ms1_activated = F.relu(ms1)

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)  # shape: (num_solu, bs, gs, n_head, 1, 1)
        # shape: (num_solu, bs, n_head, gs, 1)

        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]  # shape: (num_solu, bs, gs, n_head, 1, 1)
        # shape: (num_solu, bs, n_head, gs, 1)

        ms2 = ms2.transpose(2, 3)  # shape: (num_solu, bs, n_head, gs, 1, 1)

        mixed_scores = ms2.squeeze(-1).squeeze(-1)
        # shape: (num_solu, bs, n_head, gs)

        # weights = mixed_scores.mean(2)
        weights = self.norm_head(mixed_scores.transpose(-1, -2)).squeeze(-1)
        # shape: (num_solu, bs, gs)

        return weights

    def forward(self, node_embed, solutions, costs, dist, solution_embed_old):
        # node_embed: (bs, gs, embed_dim), dist:(bs, gs, gs)

        gs = node_embed.size(1)
        # solutions = torch.stack(solutions, 0)
        num_solu, bs, seq_len = solutions.size()

        next = torch.cat((solutions[:, :, 1:], solutions[:, :, 0].unsqueeze(-1)), -1)
        edge_pair = torch.cat((solutions.unsqueeze(-1), next.unsqueeze(-1)), -1)  # (num_solu, bs, seq_len, 2)

        embed_next = node_embed.unsqueeze(0).repeat(num_solu, 1, 1, 1).gather(2, next.unsqueeze(-1).expand(
            num_solu, bs, -1, self.embed_dim))
        embed_solu = node_embed.unsqueeze(0).repeat(num_solu, 1, 1, 1).gather(2, solutions.unsqueeze(-1).expand(
            num_solu, bs, -1, self.embed_dim))
        # [num_solu, bs, n_head, seq_len, key_dim]
        embed_solu = self.Wq(embed_solu).reshape(num_solu, bs, seq_len, self.n_heads, -1).transpose(2, 3)
        embed_next = self.Wk(embed_next).reshape(num_solu, bs, seq_len, self.n_heads, -1).transpose(2, 3)

        if self.problem == 'tsp' or self.problem == 'pdp':
            # (num_solu, bs, seq_len)
            edge_cost = dist.unsqueeze(0).repeat(num_solu, 1, 1, 1).gather(2, solutions.unsqueeze(-1)
                                                                           .expand(num_solu, bs, gs, gs)).gather(3,
                                                                                                                 next.unsqueeze(
                                                                                                                     -1)).squeeze(
                -1)
        else:
            edge_cost = dist.unsqueeze(0).repeat(num_solu, 1, 1, 1).gather(2, solutions[..., None].expand(
                *solutions.size(), dist.size(-1))).gather(3, next.unsqueeze(-1)).squeeze(-1)

        e = self.mix_attn(embed_solu, embed_next, edge_cost)
        cost = e / costs.unsqueeze(-1).expand_as(e)  # (num_solu, bs, seq_len)
        attention = torch.zeros((bs, gs, gs), device=node_embed.device)
        tmp = torch.zeros((bs, gs, gs), device=node_embed.device)  # n_cus

        for i in range(num_solu):
            tmp.index_put_(
                tuple(torch.cat((torch.arange(bs, device=node_embed.device).view(bs, 1, 1).expand(bs, seq_len, 1),
                                 edge_pair[i]), -1).view(-1, 3).t()), cost[i].view(-1))
            attention += tmp
            tmp = tmp * 0
        del tmp

        # attention = attention + attention.transpose(-1, -2)

        zero_vec = -9e15 * torch.ones_like(attention)
        attention = torch.where(attention != 0.0, attention, zero_vec)
        attention[:, torch.arange(gs), torch.arange(gs)] = -9e15
        attention = F.softmax(attention, dim=-1)  # (bs, gs, gs)

        solution_embed = torch.matmul(attention, node_embed)  # (bs, gs, hidden_edge_dim)
        solution_embed_new = self.cell(solution_embed.view(bs * gs, self.embed_dim),
                                       solution_embed_old.view(bs * gs, self.embed_dim)).view(bs, gs, self.embed_dim)

        if self.concat:
            return F.elu(solution_embed_new), solution_embed_new
        else:
            return solution_embed_new, solution_embed_new
