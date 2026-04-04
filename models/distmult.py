"""
DistMult Knowledge Graph Embedding for link prediction.

Based on Yang et al. (2015): "Embedding Entities and Relations for Learning
and Inference in Knowledge Bases"
https://arxiv.org/abs/1412.6575

Used in BRIDGE (bioRxiv 2026) — the best-performing KGE model for AMR
link prediction (97.13% accuracy, 99.07% AUPRC on K. pneumoniae).

Key idea: Each entity and relation gets a learned embedding vector.
Score(h, r, t) = h^T diag(r) t = sum(h * r * t)

IMPORTANT — transductive limitation:
DistMult learns a separate embedding per entity. It CANNOT score edges
involving entities unseen during training. This means it fundamentally
cannot do zero-shot prediction for unseen antibiotics.

We include it as a transductive baseline to demonstrate exactly this
limitation — motivating why inductive GNN approaches are necessary for
zero-shot AMR prediction.

Training protocol:
- Only train/val/test on gene-antibiotic pairs where the antibiotic
  appeared in training (transductive split)
- Compare against GNNs on the same seen-antibiotic subset
- Then show GNNs additionally generalize to unseen antibiotics (zero-shot)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistMult(nn.Module):
    """
    DistMult KGE model.

    Scores a triple (head, relation, tail) as: sum(e_h * w_r * e_t)
    where e_h, w_r, e_t are learned embedding vectors.

    For our graph we use a single relation type (gene_confers_resistance_to_antibiotic)
    since DistMult is evaluated purely on gene-antibiotic link prediction.
    """

    def __init__(self, num_entities, embedding_dim=64, num_relations=1, dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_relations  = num_relations

        self.entity_emb   = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        self.dropout      = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score(self, head_idx, tail_idx, rel_idx=0):
        """
        Compute DistMult score for (head, rel, tail) triples.
        Score = sum(e_h * w_r * e_t)
        """
        rel = torch.tensor(rel_idx, device=head_idx.device) if isinstance(rel_idx, int) else rel_idx

        e_h = F.dropout(self.entity_emb(head_idx), p=self.dropout, training=self.training)
        e_t = F.dropout(self.entity_emb(tail_idx), p=self.dropout, training=self.training)
        w_r = self.relation_emb(rel.expand(head_idx.shape[0]) if rel.dim() == 0 else rel)

        return (e_h * w_r * e_t).sum(dim=-1)

    def forward(self, edge_index, rel_idx=0):
        """Returns sigmoid scores for edges — matches interface expected by training loop."""
        return torch.sigmoid(self.score(edge_index[0], edge_index[1], rel_idx))

    def decode(self, z, edge_index):
        """
        Standard decode interface (z is unused — DistMult uses its own embeddings).
        Kept for compatibility with ranking eval loop.
        """
        return self.forward(edge_index)

    def get_all_entity_embeddings(self):
        return self.entity_emb.weight  # [num_entities, dim]

    def get_relation_embedding(self, rel_idx=0):
        return self.relation_emb(torch.tensor(rel_idx, device=self.entity_emb.weight.device))

    @torch.no_grad()
    def score_all_tails(self, head_idx, tail_indices, rel_idx=0):
        """
        Vectorised DistMult scoring of one head against many tails.
        Returns shape [len(tail_indices)] — no dropout, inference only.
        score(h, r, t_i) = sum(e_h * w_r * e_{t_i})
        """
        dev = self.entity_emb.weight.device
        h = self.entity_emb(torch.tensor([head_idx], device=dev))   # [1, dim]
        r = self.relation_emb(torch.tensor([rel_idx], device=dev))  # [1, dim]
        t = self.entity_emb(tail_indices.to(dev))                   # [N, dim]
        return (h * r * t).sum(-1)                                  # [N]


def create_distmult(num_entities, embedding_dim=64, num_relations=1, dropout=0.2):
    return DistMult(
        num_entities=num_entities,
        embedding_dim=embedding_dim,
        num_relations=num_relations,
        dropout=dropout,
    )
