"""
Graph Attention Network (GAT) for link prediction.

Based on Veličković et al. (2018): "Graph Attention Networks"

Key idea: Learn attention weights for neighbors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATConv(nn.Module):
    """GAT convolution layer with multi-head attention."""

    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # Linear transformations for each head
        self.weight = nn.Parameter(torch.Tensor(heads, in_channels, out_channels))

        # Attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.bias = nn.Parameter(torch.Tensor(heads * out_channels if concat else out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        """Simplified GAT forward pass."""
        num_nodes = x.size(0)
        row, col = edge_index

        # Transform and compute attention for each head separately
        outputs = []

        for h in range(self.heads):
            # Transform
            x_h = x @ self.weight[h]  # [num_nodes, out_channels]

            # Attention
            alpha_src = (x_h * self.att_src[0, h]).sum(dim=-1)  # [num_nodes]
            alpha_dst = (x_h * self.att_dst[0, h]).sum(dim=-1)  # [num_nodes]

            alpha = alpha_src[row] + alpha_dst[col]  # [num_edges]
            alpha = F.leaky_relu(alpha, 0.2)

            # Softmax per node
            alpha_max = torch.zeros(num_nodes, device=x.device)
            alpha_max.scatter_reduce_(0, row, alpha, reduce='amax', include_self=False)
            alpha = torch.exp(alpha - alpha_max[row])

            alpha_sum = torch.zeros(num_nodes, device=x.device)
            alpha_sum.index_add_(0, row, alpha)
            alpha = alpha / alpha_sum[row].clamp(min=1e-16)

            # Dropout
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            # Aggregate
            out_h = torch.zeros(num_nodes, self.out_channels, device=x.device)
            messages = alpha.unsqueeze(-1) * x_h[col]
            out_h.index_add_(0, row, messages)

            outputs.append(out_h)

        # Combine heads
        if self.concat:
            out = torch.cat(outputs, dim=-1)
        else:
            out = torch.stack(outputs).mean(dim=0)

        out = out + self.bias
        return out



class GAT(nn.Module):
    """GAT model for link prediction."""

    def __init__(self, in_channels, hidden_channels, out_channels, heads=4,
                 num_layers=2, dropout=0.6):
        super().__init__()

        self.convs = nn.ModuleList()

        # First layer (multi-head, concatenate)
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout)
            )

        # Output layer (average heads)
        self.convs.append(
            GATConv(hidden_channels * heads if num_layers > 1 else hidden_channels,
                    out_channels, heads=1, concat=False, dropout=dropout)
        )

        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        return torch.sigmoid((src * dst).sum(dim=-1))


def create_gat(num_node_features, hidden_dim=64, embedding_dim=32, heads=4,
               num_layers=2, dropout=0.6):
    return GAT(
        in_channels=num_node_features,
        hidden_channels=hidden_dim,
        out_channels=embedding_dim,
        heads=heads,
        num_layers=num_layers,
        dropout=dropout
    )
