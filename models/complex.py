"""
ComplEx Knowledge Graph Embedding for link prediction.

Based on Trouillon et al. (2016): "Complex Embeddings for Simple Link Prediction"
https://arxiv.org/abs/1606.06357

Used in BRIDGE (bioRxiv 2026) as the best-performing KGE model for AMR
link prediction — outperforming DistMult, TransE, HolE, and RotatE.

Key idea: Entities and relations are embedded as complex vectors.
Score(h, r, t) = Re(< e_h, w_r, conj(e_t) >)
              = sum(Re(h)*Re(r)*Re(t) + Re(h)*Im(r)*Im(t)
                  + Im(h)*Re(r)*Im(t) - Im(h)*Im(r)*Re(t))

ComplEx is a strict generalisation of DistMult:
  - When imaginary parts are zero, ComplEx reduces exactly to DistMult.
  - ComplEx can model asymmetric relations; DistMult cannot.

IMPORTANT — transductive limitation (same as DistMult):
ComplEx learns a separate embedding per entity. It CANNOT score edges
involving entities unseen during training. We include it alongside
DistMult as a stronger transductive KGE baseline to match BRIDGE's
evaluation and demonstrate why inductive GNNs are required for zero-shot
AMR prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplEx(nn.Module):
    """
    ComplEx KGE model.

    Each entity gets a complex embedding (real + imaginary parts).
    Each relation gets a complex embedding (real + imaginary parts).
    Score = Re(<e_h, w_r, conj(e_t)>)

    Parameters
    ----------
    num_entities : int
        Total number of nodes in the graph.
    embedding_dim : int
        Dimension of *each* of the real and imaginary parts.
        Total parameter cost = 2 * num_entities * embedding_dim
                             + 2 * num_relations * embedding_dim
    num_relations : int
        Number of relation types (default 1 for gene-antibiotic only).
    dropout : float
        Dropout applied to entity embeddings during scoring.
    """

    def __init__(self, num_entities, embedding_dim=64, num_relations=1, dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_relations  = num_relations
        self.dropout        = dropout

        # Real and imaginary parts stored as separate embeddings
        self.entity_re  = nn.Embedding(num_entities, embedding_dim)
        self.entity_im  = nn.Embedding(num_entities, embedding_dim)
        self.rel_re     = nn.Embedding(num_relations, embedding_dim)
        self.rel_im     = nn.Embedding(num_relations, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_re.weight)
        nn.init.xavier_uniform_(self.entity_im.weight)
        nn.init.xavier_uniform_(self.rel_re.weight)
        nn.init.xavier_uniform_(self.rel_im.weight)

    def score(self, head_idx, tail_idx, rel_idx=0):
        """
        Compute ComplEx score for (head, rel, tail) triples.
        Score = Re(<e_h, w_r, conj(e_t)>)
              = sum(h_re*r_re*t_re + h_re*r_im*t_im
                  + h_im*r_re*t_im - h_im*r_im*t_re)
        """
        rel = (torch.tensor(rel_idx, device=head_idx.device)
               if isinstance(rel_idx, int) else rel_idx)
        rel_expand = rel.expand(head_idx.shape[0]) if rel.dim() == 0 else rel

        h_re = F.dropout(self.entity_re(head_idx), p=self.dropout, training=self.training)
        h_im = F.dropout(self.entity_im(head_idx), p=self.dropout, training=self.training)
        t_re = F.dropout(self.entity_re(tail_idx), p=self.dropout, training=self.training)
        t_im = F.dropout(self.entity_im(tail_idx), p=self.dropout, training=self.training)
        r_re = self.rel_re(rel_expand)
        r_im = self.rel_im(rel_expand)

        return (
            (h_re * r_re * t_re).sum(-1)
            + (h_re * r_im * t_im).sum(-1)
            + (h_im * r_re * t_im).sum(-1)
            - (h_im * r_im * t_re).sum(-1)
        )

    def forward(self, edge_index, rel_idx=0):
        """Returns sigmoid scores for edges — matches interface expected by training loop."""
        return torch.sigmoid(self.score(edge_index[0], edge_index[1], rel_idx))

    def decode(self, z, edge_index):
        """
        Standard decode interface (z is unused — ComplEx uses its own embeddings).
        Kept for compatibility with ranking eval loop.
        """
        return self.forward(edge_index)

    @torch.no_grad()
    def score_all_tails(self, head_idx, tail_indices, rel_idx=0):
        """
        Vectorised ComplEx scoring of one head against many tails.
        Returns shape [len(tail_indices)] — no dropout, inference only.
        score(h, r, t_i) = Re(<e_h, w_r, conj(e_{t_i})>)
        """
        dev = self.entity_re.weight.device
        h_re = self.entity_re(torch.tensor([head_idx], device=dev))  # [1, dim]
        h_im = self.entity_im(torch.tensor([head_idx], device=dev))
        r_re = self.rel_re(torch.tensor([rel_idx], device=dev))      # [1, dim]
        r_im = self.rel_im(torch.tensor([rel_idx], device=dev))
        t_re = self.entity_re(tail_indices.to(dev))                  # [N, dim]
        t_im = self.entity_im(tail_indices.to(dev))
        return (
            (h_re * r_re * t_re).sum(-1)
            + (h_re * r_im * t_im).sum(-1)
            + (h_im * r_re * t_im).sum(-1)
            - (h_im * r_im * t_re).sum(-1)
        )                                                            # [N]

    def get_all_entity_embeddings(self):
        """
        Return concatenated [real | imag] embeddings for ranking.
        The ranking loop uses dot-product scoring; we override via decode(),
        so this is provided only for interface parity with DistMult.
        """
        return torch.cat([self.entity_re.weight, self.entity_im.weight], dim=-1)

    def get_relation_embedding(self, rel_idx=0):
        dev = self.entity_re.weight.device
        idx = torch.tensor(rel_idx, device=dev)
        return torch.cat([self.rel_re(idx), self.rel_im(idx)], dim=-1)


def create_complex(num_entities, embedding_dim=64, num_relations=1, dropout=0.2):
    return ComplEx(
        num_entities=num_entities,
        embedding_dim=embedding_dim,
        num_relations=num_relations,
        dropout=dropout,
    )
