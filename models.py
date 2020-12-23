"""
Implementing Graph Attention Layers, as described in the paper 'Graph Attention Networks' by Velickovic et al.
(https://arxiv.org/pdf/1710.10903.pdf)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention Layer
    """
    def __init__(self, in_dim, out_dim, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.out_dim = out_dim
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.concat = concat

        self.W = nn.Parameter(torch.zeros((in_dim, out_dim)), requires_grad=True)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, h, A):
        # h has shape (n, in_dim)

        Wh = torch.mm(h, self.W)    # Transformation shared across nodes

        # Creating matrix of concatenations of node features
        attn_input = self._make_attention_input(Wh)

        # Attention scores
        e = self.leakyrelu(self.attn(attn_input))

        # Attention weights
        zero_vec = torch.ones_like(e) * 1e-15
        attention = torch.where(A.detach().cpu() > 0, e.detach().cpu(), zero_vec.detach().cpu()).to('cuda')   # Mask out non-neighbours
        attention = torch.softmax(attention, dim=1)
        attention = torch.dropout(attention, self.dropout, train=self.trainig)

        # Computing new node features using the attention
        h_prime = torch.matmul(attention, Wh)

        print(h_prime.shape)
        # If concat, use ELU. Else, use nothing (author recommendations)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _make_attention_input(self, Wh):
        """ Preparing the attention mechanism.
            Creating matrix with all possible combinations of concatenations of node features:
                h1 || h2,
                ...
                h1 || hN,
                h2 || h1,
                ...
                h1 || hN,
                ...
                hN || h1,
                hN || hN,
        """
        N = len(Wh)
        Wh_blocks_repeating = Wh.repeat_interleave(N, dim=0)   # Left-side of the matrix
        Wh_blocks_alternating = Wh.repeat(N, 1)                # Right-side of the matrix

        combined = torch.cat((Wh_blocks_repeating, Wh_blocks_alternating), dim=1)   # Shape (N*N, 2*num_features)

        return combined.view(N, N, 2 * self.out_dim)


class GAT(nn.Module):
    """
    GAT model consisting of multiple Attention Layers, each with potentially multiple heads
    """
    def __init__(self, node_dim, hid_dim, num_classes, dropout, alpha, num_heads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.gat1 = GraphAttentionLayer(node_dim, hid_dim, dropout, alpha)
        self.gat2 = GraphAttentionLayer(hid_dim, num_classes, dropout, alpha, concat=False)

    def forward(self, x, A):
        x = torch.dropout(x, p=self.dropout, train=self.training)
        x = self.gat1(x, A)
        x = torch.dropout(x, p=self.dropout, train=self.training)
        x = self.gat2(x, A)
        return x
