"""
Relational Graph Convolutional Network (R-GCN) for heterogeneous graphs.

Based on Schlichtkrull et al. (2018): "Modeling Relational Data with Graph Convolutional Networks"

Key idea: Different edge types have different transformation matrices.
This is PERFECT for our heterogeneous antibiotic resistance graph!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNConv(nn.Module):
    """Single R-GCN layer with relation-specific transformations."""

    def __init__(self, in_channels, out_channels, num_relations, num_bases=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        # Basis decomposition to reduce parameters
        if num_bases is None:
            num_bases = min(num_relations, 4)  # Limit bases for efficiency

        self.num_bases = num_bases

        # Basis matrices
        self.bases = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))

        # Coefficients for each relation
        self.coeffs = nn.Parameter(torch.Tensor(num_relations, num_bases))

        # Self-loop transformation
        self.self_loop = nn.Linear(in_channels, out_channels, bias=False)

        # Bias
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coeffs)
        nn.init.xavier_uniform_(self.self_loop.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_indices_by_type):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_indices_by_type: Dict[relation_id -> edge_index [2, num_edges]]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        out = torch.zeros(num_nodes, self.out_channels, device=x.device)

        # Process each relation type
        for rel_id, edge_index in edge_indices_by_type.items():
            if edge_index.size(1) == 0:
                continue

            # Get relation-specific weight matrix (basis decomposition)
            weight = torch.sum(
                self.coeffs[rel_id].view(-1, 1, 1) * self.bases,
                dim=0
            )  # [in_channels, out_channels]

            # Message passing for this relation
            row, col = edge_index
            messages = x[col] @ weight  # [num_edges, out_channels]

            # Aggregate messages (mean aggregation)
            out.index_add_(0, row, messages)

            # Normalize by degree
            deg = torch.bincount(row, minlength=num_nodes).float().clamp(min=1)
            out = out / deg.view(-1, 1)

        # Add self-loop
        out = out + self.self_loop(x)

        # Add bias
        out = out + self.bias

        return out


class RGCN(nn.Module):
    """
    R-GCN model for link prediction on heterogeneous graphs.

    Architecture:
        Input -> R-GCN -> ReLU -> R-GCN -> Output embeddings
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_relations,
                 num_layers=2, num_bases=None, dropout=0.5):
        super().__init__()

        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(
            RGCNConv(in_channels, hidden_channels, num_relations, num_bases)
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases)
            )

        # Output layer
        self.convs.append(
            RGCNConv(hidden_channels, out_channels, num_relations, num_bases)
        )

        self.dropout = dropout

    def forward(self, x, edge_indices_by_type):
        """
        Encode nodes into embeddings.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_indices_by_type: Dict[relation_id -> edge_index]

        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_indices_by_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_indices_by_type)

        return x

    def decode(self, z, edge_index):
        """
        Decode edge probabilities from node embeddings.

        Uses dot product: p(edge) = sigmoid(z_i^T z_j)

        Args:
            z: Node embeddings [num_nodes, out_channels]
            edge_index: Edges to predict [2, num_edges]

        Returns:
            Edge probabilities [num_edges]
        """
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        return torch.sigmoid((src * dst).sum(dim=-1))


def create_rgcn(num_node_features, num_relations, hidden_dim=64, embedding_dim=32,
                num_layers=2, num_bases=4, dropout=0.5):
    """
    Factory function to create R-GCN model.

    Args:
        num_node_features: Input feature dimension
        num_relations: Number of edge types
        hidden_dim: Hidden layer dimension
        embedding_dim: Output embedding dimension
        num_layers: Number of R-GCN layers
        num_bases: Number of basis matrices (for parameter reduction)
        dropout: Dropout rate

    Returns:
        R-GCN model
    """
    return RGCN(
        in_channels=num_node_features,
        hidden_channels=hidden_dim,
        out_channels=embedding_dim,
        num_relations=num_relations,
        num_layers=num_layers,
        num_bases=num_bases,
        dropout=dropout
    )
