"""
Heterogeneous Graph Transformer (HGT) for link prediction.

Based on Hu et al. (2020): "Heterogeneous Graph Transformer"
https://arxiv.org/abs/2003.01332

Used in: VRE hospital outbreak prediction (PLOS Digital Health 2025)
         AmpHGT antimicrobial peptide prediction (BMC Biology 2025)

Key idea: relation-type-specific attention between nodes, allowing the model
to weight messages differently depending on the biological relationship type
(gene→antibiotic vs gene→mechanism vs gene→GO term, etc.).

Implementation notes:
- We use SHARED K/Q/V projections (not per-type) since all nodes share the
  same 8-dim one-hot feature space. Type-specificity is encoded via node-type
  embedding that is concatenated to features before projection.
- Relation-specific matrices are kept small (scalar per head) to avoid
  overparameterisation on this graph.
- Proper Xavier init throughout for training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HGTConv(nn.Module):
    """
    Single HGT layer with shared projections + relation-specific attention scaling.

    For each relation r, the attention score between src s and dst d is:
        alpha(s,r,d) = softmax_over_neighbors( (Q_d · (W_r * K_s)) / sqrt(d_head) )

    where W_r is a relation-specific diagonal scaling (num_heads scalars).
    """

    def __init__(self, in_channels, out_channels, num_node_types, num_edge_types,
                 num_heads=4, dropout=0.2):
        super().__init__()
        assert out_channels % num_heads == 0
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.num_heads     = num_heads
        self.head_dim      = out_channels // num_heads
        self.dropout       = dropout
        self.scale          = math.sqrt(self.head_dim)
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Node-type embedding — concatenated to features so type is encoded
        self.type_emb = nn.Embedding(num_node_types, in_channels)

        # Shared K/Q/V projections
        self.k_lin = nn.Linear(in_channels * 2, out_channels, bias=False)
        self.q_lin = nn.Linear(in_channels * 2, out_channels, bias=False)
        self.v_lin = nn.Linear(in_channels * 2, out_channels, bias=False)

        # Relation-specific attention scale: one scalar per head per relation
        # (lightweight — avoids the d×d per-relation matrices in the original)
        self.rel_scale = nn.Parameter(torch.ones(num_edge_types, num_heads))

        # Output projection + skip connection
        self.out_lin  = nn.Linear(out_channels, out_channels)
        self.skip_lin = nn.Linear(in_channels * 2, out_channels, bias=False)
        self.norm     = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in [self.k_lin, self.q_lin, self.v_lin, self.out_lin, self.skip_lin]:
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        nn.init.normal_(self.type_emb.weight, std=0.1)
        nn.init.ones_(self.rel_scale)

    def forward(self, x, edge_indices_by_type, node_types_tensor):
        """
        x                   : [N, in_channels]
        edge_indices_by_type: {rel_id: [2, E]}
        node_types_tensor   : [N] long tensor of node type ids
        """
        N = x.size(0)

        # Augment features with type embedding
        type_vec = self.type_emb(node_types_tensor)   # [N, in_channels]
        x_aug    = torch.cat([x, type_vec], dim=-1)   # [N, 2*in_channels]

        # Project to K, Q, V
        K = self.k_lin(x_aug).view(N, self.num_heads, self.head_dim)  # [N, H, d]
        Q = self.q_lin(x_aug).view(N, self.num_heads, self.head_dim)
        V = self.v_lin(x_aug).view(N, self.num_heads, self.head_dim)

        # Two-pass softmax attention (per-destination-node normalisation across all relations)
        all_dsts, all_attn_raw, all_vals = [], [], []

        for rel_id, edge_index in edge_indices_by_type.items():
            if edge_index.numel() == 0 or rel_id >= self.num_edge_types:
                continue
            src, dst = edge_index
            scale = self.rel_scale[rel_id]           # [H]
            attn = (Q[dst] * K[src]).sum(-1)         # [E, H]
            attn = attn * scale.unsqueeze(0) / self.scale
            all_dsts.append(dst)
            all_attn_raw.append(attn)
            all_vals.append(V[src])

        if not all_dsts:
            skip = self.skip_lin(x_aug)
            return F.elu(self.norm(skip))

        all_dsts_cat  = torch.cat(all_dsts)          # [total_E]
        all_attn_cat  = torch.cat(all_attn_raw)      # [total_E, H]
        all_vals_cat  = torch.cat(all_vals)          # [total_E, H, d]

        # Numerically stable softmax: subtract global max before exp
        # (global max is valid for softmax stability: softmax(x) = softmax(x - c))
        attn_shifted = all_attn_cat - all_attn_cat.detach().max()
        attn_exp     = torch.exp(attn_shifted)       # [total_E, H]

        # Sum exp per destination node per head
        attn_sum = torch.zeros(N, self.num_heads, device=x.device)
        attn_sum.index_add_(0, all_dsts_cat, attn_exp)

        # Normalize: proper per-node softmax
        attn_norm = attn_exp / (attn_sum[all_dsts_cat] + 1e-16)   # [total_E, H]
        attn_norm = F.dropout(attn_norm, p=self.dropout, training=self.training)

        # Aggregate
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        msg = attn_norm.unsqueeze(-1) * all_vals_cat  # [total_E, H, d]
        out.index_add_(0, all_dsts_cat, msg)

        out = out.view(N, self.out_channels)
        out = self.out_lin(out)

        # Skip connection + LayerNorm
        skip = self.skip_lin(x_aug)
        out  = self.norm(out + skip)

        return F.elu(out)


class HGT(nn.Module):
    """Multi-layer HGT for link prediction."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_node_types, num_edge_types, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.node_type_map = None   # must be set before forward()

        self.convs = nn.ModuleList()
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        for i in range(num_layers):
            self.convs.append(
                HGTConv(dims[i], dims[i + 1], num_node_types, num_edge_types,
                        num_heads, dropout)
            )

    def _node_types_tensor(self, num_nodes, device):
        t = torch.zeros(num_nodes, dtype=torch.long, device=device)
        num_types = self.convs[0].num_node_types
        for nid, typ in self.node_type_map.items():
            if nid < num_nodes and typ < num_types:
                t[nid] = typ
        return t

    def forward(self, x, edge_indices_by_type):
        assert self.node_type_map is not None, "Set model.node_type_map before forward()"
        nt = self._node_types_tensor(x.size(0), x.device)
        for conv in self.convs[:-1]:
            x = conv(x, edge_indices_by_type, nt)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_indices_by_type, nt)
        return x

    def decode(self, z, edge_index):
        return torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1))


def create_hgt(num_node_features, num_node_types, num_edge_types,
               hidden_dim=64, embedding_dim=32, num_heads=4, num_layers=2, dropout=0.2):
    return HGT(
        in_channels=num_node_features,
        hidden_channels=hidden_dim,
        out_channels=embedding_dim,
        num_node_types=num_node_types,
        num_edge_types=num_edge_types,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
