"""
TransE Knowledge Graph Embedding for link prediction.

Based on Bordes et al. (2013): "Translating Embeddings for Modeling
Multi-Relational Data"
https://arxiv.org/abs/1301.3666

Used in BRIDGE (bioRxiv 2026) and KIDS (Nature Communications 2022)
as a standard KGE baseline for AMR link prediction.

Key idea: Relations are modelled as translations in embedding space.
A triple (h, r, t) is valid if e_h + w_r ≈ e_t.
Score(h, r, t) = -||e_h + w_r - e_t||_2  (higher = more likely)

TransE is a distance-based model — it assumes relations are
anti-symmetric (unlike DistMult) but cannot model symmetric or
1-to-N relations well.

IMPORTANT — same transductive limitation as DistMult and ComplEx:
TransE learns fixed entity embeddings. Without training edges for a
test antibiotic, its embedding is never updated and carries no
resistance signal. We include it to complete the BRIDGE KGE benchmark
(TransE, DistMult, ComplEx) in the zero-shot setting and confirm that
the failure is structural, not model-specific.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    """
    TransE KGE model.

    Score(h, r, t) = -||e_h + w_r - e_t||_2
    Trained with margin-based or BCE loss via sigmoid(score).

    Parameters
    ----------
    num_entities : int
    embedding_dim : int
    num_relations : int
    dropout : float
    norm : int
        1 for L1, 2 for L2 distance (default 2).
    """

    def __init__(self, num_entities, embedding_dim=64,
                 num_relations=1, dropout=0.2, norm=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_relations  = num_relations
        self.dropout        = dropout
        self.norm           = norm

        self.entity_emb   = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        # Normalise relation embeddings to unit sphere (standard TransE init)
        with torch.no_grad():
            self.relation_emb.weight.data = F.normalize(
                self.relation_emb.weight.data, p=2, dim=-1)

    def score(self, head_idx, tail_idx, rel_idx=0):
        """
        Compute TransE score for (head, rel, tail) triples.
        Score = -||e_h + w_r - e_t||_p  (higher = more plausible)
        """
        rel = (torch.tensor(rel_idx, device=head_idx.device)
               if isinstance(rel_idx, int) else rel_idx)
        rel_expand = rel.expand(head_idx.shape[0]) if rel.dim() == 0 else rel

        e_h = F.dropout(self.entity_emb(head_idx), p=self.dropout, training=self.training)
        e_t = F.dropout(self.entity_emb(tail_idx), p=self.dropout, training=self.training)
        w_r = self.relation_emb(rel_expand)

        diff = e_h + w_r - e_t           # [batch, dim]
        return -torch.norm(diff, p=self.norm, dim=-1)  # [batch]

    def forward(self, edge_index, rel_idx=0):
        """Returns sigmoid scores — matches training loop interface."""
        return torch.sigmoid(self.score(edge_index[0], edge_index[1], rel_idx))

    def decode(self, z, edge_index):
        """Standard decode interface (z unused — TransE uses own embeddings)."""
        return self.forward(edge_index)

    @torch.no_grad()
    def score_all_tails(self, head_idx, tail_indices, rel_idx=0):
        """
        Vectorised TransE scoring of one head against many tails.
        Returns shape [len(tail_indices)] — no dropout, inference only.
        Score(h, r, t_i) = -||e_h + w_r - e_{t_i}||_2
        """
        dev = self.entity_emb.weight.device
        e_h = self.entity_emb(torch.tensor([head_idx], device=dev))  # [1, dim]
        w_r = self.relation_emb(torch.tensor([rel_idx], device=dev)) # [1, dim]
        e_t = self.entity_emb(tail_indices.to(dev))                  # [N, dim]

        diff = (e_h + w_r) - e_t          # [N, dim]  (broadcasts)
        return -torch.norm(diff, p=self.norm, dim=-1)  # [N]

    def get_all_entity_embeddings(self):
        return self.entity_emb.weight

    def get_relation_embedding(self, rel_idx=0):
        dev = self.entity_emb.weight.device
        return self.relation_emb(torch.tensor(rel_idx, device=dev))


def create_transe(num_entities, embedding_dim=64, num_relations=1,
                  dropout=0.2, norm=2):
    return TransE(
        num_entities=num_entities,
        embedding_dim=embedding_dim,
        num_relations=num_relations,
        dropout=dropout,
        norm=norm,
    )
