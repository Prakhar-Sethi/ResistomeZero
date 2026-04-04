"""
GraphSAGE for link prediction.

Based on Hamilton et al. (2017): "Inductive Representation Learning on Large Graphs"

Key idea: Sample and aggregate features from neighbors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGEConv(nn.Module):
    """GraphSAGE convolution layer."""

    def __init__(self, in_channels, out_channels, aggregator='mean'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregator = aggregator

        # Transformation for concatenated features
        self.linear = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        row, col = edge_index
        num_nodes = x.size(0)

        # Aggregate neighbor features
        if self.aggregator == 'mean':
            neighbor_sum = torch.zeros_like(x)
            neighbor_sum.index_add_(0, row, x[col])
            deg = torch.bincount(row, minlength=num_nodes).float().clamp(min=1)
            neighbor_agg = neighbor_sum / deg.view(-1, 1)
        else:
            raise NotImplementedError(f"Aggregator {self.aggregator} not implemented")

        # Concatenate self features with aggregated neighbor features
        out = torch.cat([x, neighbor_agg], dim=-1)

        # Transform
        out = self.linear(out)

        return out


class GraphSAGE(nn.Module):
    """GraphSAGE model for link prediction."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        return torch.sigmoid((src * dst).sum(dim=-1))


def create_graphsage(num_node_features, hidden_dim=64, embedding_dim=32,
                     num_layers=2, dropout=0.5):
    return GraphSAGE(
        in_channels=num_node_features,
        hidden_channels=hidden_dim,
        out_channels=embedding_dim,
        num_layers=num_layers,
        dropout=dropout
    )
