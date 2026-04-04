"""
Graph Convolutional Network (GCN) for link prediction.

Based on Kipf & Welling (2017): Semi-Supervised Classification with Graph Convolutional Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNConv(nn.Module):
    """Single GCN layer."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Add self-loops
        num_nodes = x.size(0)
        loop_index = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)

        # Compute degree
        row, col = edge_index
        deg = torch.bincount(row, minlength=num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Normalize
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Message passing (vectorized)
        row, col = edge_index
        out = torch.zeros(num_nodes, x.size(1), dtype=x.dtype, device=x.device)
        out.index_add_(0, col, norm.unsqueeze(1) * x[row])

        # Linear transformation
        out = self.linear(out)

        return out


class GCN(nn.Module):
    """
    GCN model for link prediction.

    Architecture:
        Input -> GCN -> ReLU -> GCN -> ReLU -> GCN -> Output
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Encode nodes into embeddings.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)

        return x

    def decode(self, z, edge_index):
        """
        Decode edge probabilities from node embeddings.

        Uses dot product decoder: p(edge) = sigmoid(z_i^T z_j)

        Args:
            z: Node embeddings [num_nodes, out_channels]
            edge_index: Edge indices to predict [2, num_edges]

        Returns:
            Edge probabilities [num_edges]
        """
        src = z[edge_index[0]]  # [num_edges, out_channels]
        dst = z[edge_index[1]]  # [num_edges, out_channels]

        # Dot product
        logits = (src * dst).sum(dim=-1)  # [num_edges]

        return torch.sigmoid(logits)

    def decode_all(self, z):
        """
        Decode all possible edges (for evaluation).

        Args:
            z: Node embeddings [num_nodes, out_channels]

        Returns:
            Adjacency matrix [num_nodes, num_nodes]
        """
        prob_adj = torch.sigmoid(z @ z.t())
        return prob_adj


def create_gcn(num_node_features, hidden_dim=64, embedding_dim=32, num_layers=3, dropout=0.5):
    """
    Factory function to create GCN model.

    Args:
        num_node_features: Number of input features
        hidden_dim: Hidden layer dimension
        embedding_dim: Output embedding dimension
        num_layers: Number of GCN layers
        dropout: Dropout rate

    Returns:
        GCN model
    """
    return GCN(
        in_channels=num_node_features,
        hidden_channels=hidden_dim,
        out_channels=embedding_dim,
        num_layers=num_layers,
        dropout=dropout
    )
