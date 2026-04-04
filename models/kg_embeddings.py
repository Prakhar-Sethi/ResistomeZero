"""
Knowledge Graph Embedding models for link prediction.

Implements:
- TransE: Translation-based embedding
- DistMult: Bilinear model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data
    Bordes et al. (2013)

    Models relations as translations: h + r ≈ t
    Score: -||h + r - t||
    """

    def __init__(self, num_entities, num_relations, embedding_dim=50, margin=1.0):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin

        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)

        # Relation embeddings
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # Normalize entity embeddings
        with torch.no_grad():
            self.entity_embeddings.weight.div_(
                self.entity_embeddings.weight.norm(dim=1, keepdim=True)
            )

    def forward(self, head, relation, tail):
        """
        Args:
            head: Head entity indices [batch_size]
            relation: Relation indices [batch_size]
            tail: Tail entity indices [batch_size]

        Returns:
            Scores [batch_size]
        """
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        # Translation: h + r - t
        score = -torch.norm(h + r - t, p=2, dim=1)

        return score

    def get_embeddings(self):
        """Get entity embeddings for downstream tasks."""
        return self.entity_embeddings.weight


class DistMult(nn.Module):
    """
    DistMult: Embedding Entities and Relations for Learning and Inference in Knowledge Bases
    Yang et al. (2015)

    Bilinear model: score = <h, r, t> = sum(h * r * t)
    """

    def __init__(self, num_entities, num_relations, embedding_dim=50):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)

        # Relation embeddings (diagonal matrix)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, head, relation, tail):
        """
        Args:
            head: Head entity indices [batch_size]
            relation: Relation indices [batch_size]
            tail: Tail entity indices [batch_size]

        Returns:
            Scores [batch_size]
        """
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        # Bilinear: sum(h * r * t)
        score = (h * r * t).sum(dim=1)

        return score

    def get_embeddings(self):
        """Get entity embeddings for downstream tasks."""
        return self.entity_embeddings.weight


class KGModel(nn.Module):
    """
    Wrapper for KG embedding models to support link prediction task.

    Converts edge-based prediction to triple-based prediction.
    """

    def __init__(self, base_model, num_entities, num_relations, embedding_dim):
        super().__init__()

        if base_model == 'transe':
            self.model = TransE(num_entities, num_relations, embedding_dim)
        elif base_model == 'distmult':
            self.model = DistMult(num_entities, num_relations, embedding_dim)
        else:
            raise ValueError(f"Unknown model: {base_model}")

        self.base_model = base_model

    def forward(self, head, relation, tail):
        return self.model(head, relation, tail)

    def decode(self, z, edge_index, relation_type=0):
        """
        Decode edge probabilities.

        For KG models, z is not used (they have their own embeddings).

        Args:
            z: Unused (for compatibility with GNN interface)
            edge_index: Edges to predict [2, num_edges]
            relation_type: Relation type for these edges

        Returns:
            Edge scores [num_edges]
        """
        head = edge_index[0]
        tail = edge_index[1]
        relation = torch.full_like(head, relation_type)

        scores = self.model(head, relation, tail)

        # Normalize to [0, 1]
        if self.base_model == 'transe':
            # TransE scores are negative distances, convert to probabilities
            scores = torch.sigmoid(scores / 5.0)  # Scale factor
        else:
            # DistMult scores can be large, use sigmoid
            scores = torch.sigmoid(scores)

        return scores

    def get_embeddings(self):
        """Get entity embeddings."""
        return self.model.get_embeddings()


def create_transe(num_entities, num_relations, embedding_dim=50, margin=1.0):
    return KGModel('transe', num_entities, num_relations, embedding_dim)


def create_distmult(num_entities, num_relations, embedding_dim=50):
    return KGModel('distmult', num_entities, num_relations, embedding_dim)
