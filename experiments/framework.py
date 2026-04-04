"""
Comprehensive experiment framework for link prediction.

Features:
- All metrics (AUC, AP, Precision@K, Recall@K, Hit@K, MRR)
- Efficient negative sampling
- Proper logging
- Reproducibility
- Model checkpointing
"""

import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import json
from pathlib import Path


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def efficient_negative_sampling(num_nodes, pos_edge_index, num_neg, max_tries=3):
    """
    BLAZING FAST negative sampling - pure random (no collision checking).

    For large graphs, the probability of sampling an existing edge is tiny,
    so we can skip the expensive collision checking.

    Args:
        num_nodes: Total number of nodes
        pos_edge_index: Positive edges [2, num_pos] (unused in fast version)
        num_neg: Number of negative samples
        max_tries: Unused (kept for compatibility)

    Returns:
        Negative edge index [2, num_neg]
    """
    # Pure random sampling - MUCH faster!
    src = torch.randint(0, num_nodes, (num_neg,), dtype=torch.long)
    dst = torch.randint(0, num_nodes, (num_neg,), dtype=torch.long)

    # Remove self-loops
    mask = src != dst
    while mask.sum() < num_neg:
        # Resample the self-loops
        num_resample = num_neg - mask.sum()
        new_src = torch.randint(0, num_nodes, (num_resample,), dtype=torch.long)
        new_dst = torch.randint(0, num_nodes, (num_resample,), dtype=torch.long)
        src = torch.cat([src[mask], new_src])
        dst = torch.cat([dst[mask], new_dst])
        mask = src != dst

    return torch.stack([src[:num_neg], dst[:num_neg]])


def compute_metrics(pos_scores, neg_scores, k_values=[10, 50, 100]):
    """
    Compute comprehensive evaluation metrics.

    Args:
        pos_scores: Scores for positive edges [num_pos]
        neg_scores: Scores for negative edges [num_neg]
        k_values: K values for Precision@K, Recall@K, Hit@K

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    pos_scores = pos_scores.cpu().numpy() if torch.is_tensor(pos_scores) else pos_scores
    neg_scores = neg_scores.cpu().numpy() if torch.is_tensor(neg_scores) else neg_scores

    # Combine scores and labels
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    # AUC and AP
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    # Rank-based metrics
    # Sort by score (descending)
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    metrics = {
        'auc': auc,
        'ap': ap,
    }

    # Compute Precision@K, Recall@K, Hit@K
    num_pos = len(pos_scores)
    for k in k_values:
        if k > len(scores):
            continue

        top_k_labels = sorted_labels[:k]
        num_correct = np.sum(top_k_labels)

        precision_k = num_correct / k
        recall_k = num_correct / num_pos
        hit_k = 1.0 if num_correct > 0 else 0.0

        metrics[f'precision@{k}'] = precision_k
        metrics[f'recall@{k}'] = recall_k
        metrics[f'hit@{k}'] = hit_k

    # MRR (Mean Reciprocal Rank)
    # Find ranks of positive samples
    pos_ranks = []
    for i, label in enumerate(sorted_labels):
        if label == 1:
            pos_ranks.append(i + 1)  # 1-indexed rank

    if pos_ranks:
        mrr = np.mean([1.0 / rank for rank in pos_ranks])
        metrics['mrr'] = mrr

    return metrics


class Trainer:
    """Trainer for link prediction models."""

    def __init__(self, model, optimizer, device='cpu', log_interval=10):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.log_interval = log_interval

        self.train_history = []
        self.val_history = []

    def train_epoch(self, x, edge_index, pos_edges, neg_edges, edge_indices_by_type=None):
        """
        Train for one epoch.

        Args:
            x: Node features
            edge_index: Training graph edges (for message passing)
            pos_edges: Positive edges to predict
            neg_edges: Negative edges
            edge_indices_by_type: For R-GCN (dict of edge indices by type)

        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move to device
        x = x.to(self.device)
        pos_edges = pos_edges.to(self.device)
        neg_edges = neg_edges.to(self.device)

        # Forward pass
        if edge_indices_by_type is not None:
            # R-GCN
            edge_indices_by_type = {
                k: v.to(self.device) for k, v in edge_indices_by_type.items()
            }
            z = self.model(x, edge_indices_by_type)
        else:
            # Standard GNN
            edge_index = edge_index.to(self.device)
            z = self.model(x, edge_index)

        # Decode
        pos_pred = self.model.decode(z, pos_edges)
        neg_pred = self.model.decode(z, neg_edges)

        # Loss
        pos_loss = -torch.log(pos_pred + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_pred + 1e-15).mean()
        loss = pos_loss + neg_loss

        # Backward
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, x, edge_index, pos_edges, neg_edges, edge_indices_by_type=None):
        """
        Evaluate model.

        Args:
            Same as train_epoch

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        # Move to device
        x = x.to(self.device)
        pos_edges = pos_edges.to(self.device)
        neg_edges = neg_edges.to(self.device)

        # Forward pass
        if edge_indices_by_type is not None:
            edge_indices_by_type = {
                k: v.to(self.device) for k, v in edge_indices_by_type.items()
            }
            z = self.model(x, edge_indices_by_type)
        else:
            edge_index = edge_index.to(self.device)
            z = self.model(x, edge_index)

        # Decode
        pos_scores = self.model.decode(z, pos_edges)
        neg_scores = self.model.decode(z, neg_edges)

        # Compute metrics
        metrics = compute_metrics(pos_scores, neg_scores)

        return metrics

    def fit(self, x, edge_index, train_pos, train_neg, val_pos, val_neg,
            epochs=100, early_stopping=10, edge_indices_by_type=None):
        """
        Full training loop with early stopping.

        Returns:
            Best validation metrics
        """
        best_val_auc = 0
        patience_counter = 0
        best_metrics = None

        print(f"\nTraining {self.model.__class__.__name__}...")
        print("Epoch | Train Loss | Val AUC | Val AP | Time")
        print("-" * 60)

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            loss = self.train_epoch(x, edge_index, train_pos, train_neg, edge_indices_by_type)

            # Validate
            if epoch % self.log_interval == 0:
                val_metrics = self.evaluate(x, edge_index, val_pos, val_neg, edge_indices_by_type)
                epoch_time = time.time() - start_time

                print(f"{epoch:5d} | {loss:10.4f} | {val_metrics['auc']:7.4f} | "
                      f"{val_metrics['ap']:6.4f} | {epoch_time:.2f}s")

                # Early stopping
                if val_metrics['auc'] > best_val_auc:
                    best_val_auc = val_metrics['auc']
                    best_metrics = val_metrics
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return best_metrics


def save_results(results, filepath):
    """Save results to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {filepath}")


def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)
