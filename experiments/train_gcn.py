"""
Quick proof-of-concept: Train GCN on CARD graph for link prediction.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from models.gcn import create_gcn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# LOAD GRAPH
# ============================================================================
print("="*80)
print("GCN LINK PREDICTION - PROOF OF CONCEPT")
print("="*80)

graph_path = Path(__file__).parent.parent / "data" / "graphs" / "card_graph.pt"
print(f"\n[1/5] Loading graph from {graph_path}...")

graph_data = torch.load(graph_path)

num_nodes = graph_data['num_nodes']
node_features = graph_data['node_features']
train_edge_index = graph_data['train_edge_index']
train_pos_edges = graph_data['train_pos_edge_index']
val_pos_edges = graph_data['val_pos_edge_index']
test_pos_edges = graph_data['test_pos_edge_index']

print(f"   Nodes: {num_nodes}")
print(f"   Node features: {node_features.shape}")
print(f"   Train edges (for MP): {train_edge_index.shape[1]}")
print(f"   Train pos edges (to predict): {train_pos_edges.shape[1]}")
print(f"   Val pos edges: {val_pos_edges.shape[1]}")
print(f"   Test pos edges: {test_pos_edges.shape[1]}")

# ============================================================================
# CREATE NEGATIVE SAMPLES
# ============================================================================
print("\n[2/5] Creating negative samples...")

def negative_sampling(num_nodes, pos_edge_index, num_neg_samples):
    """Sample negative edges (non-existing edges) - OPTIMIZED."""
    # Create positive edge set for fast lookup
    pos_edges_set = set()
    for i in range(pos_edge_index.shape[1]):
        src, dst = pos_edge_index[0, i].item(), pos_edge_index[1, i].item()
        pos_edges_set.add((src, dst))
        pos_edges_set.add((dst, src))

    # Generate more samples than needed, then filter
    # This is much faster than while loop
    num_to_sample = num_neg_samples * 3  # Oversample
    src_nodes = np.random.randint(0, num_nodes, num_to_sample)
    dst_nodes = np.random.randint(0, num_nodes, num_to_sample)

    neg_edges = []
    for src, dst in zip(src_nodes, dst_nodes):
        if len(neg_edges) >= num_neg_samples:
            break
        if src != dst and (src, dst) not in pos_edges_set:
            neg_edges.append([src, dst])

    # If we didn't get enough, just return what we have (rare case)
    if len(neg_edges) < num_neg_samples:
        print(f"   Warning: Only found {len(neg_edges)} negative samples instead of {num_neg_samples}")

    return torch.tensor(neg_edges[:num_neg_samples], dtype=torch.long).t()

train_neg_edges = negative_sampling(num_nodes, train_pos_edges, train_pos_edges.shape[1])
val_neg_edges = negative_sampling(num_nodes, val_pos_edges, val_pos_edges.shape[1])
test_neg_edges = negative_sampling(num_nodes, test_pos_edges, test_pos_edges.shape[1])

print(f"   Train neg edges: {train_neg_edges.shape[1]}")
print(f"   Val neg edges: {val_neg_edges.shape[1]}")
print(f"   Test neg edges: {test_neg_edges.shape[1]}")

# ============================================================================
# CREATE MODEL
# ============================================================================
print("\n[3/5] Creating GCN model...")

model = create_gcn(
    num_node_features=node_features.shape[1],
    hidden_dim=64,
    embedding_dim=32,
    num_layers=2,
    dropout=0.3
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f"   Model: {model}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters())}")

# ============================================================================
# TRAINING
# ============================================================================
print("\n[4/5] Training...")

def train_epoch():
    model.train()
    optimizer.zero_grad()

    # Forward pass
    z = model(node_features, train_edge_index)

    # Positive samples
    pos_pred = model.decode(z, train_pos_edges)
    pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))

    # Negative samples
    neg_pred = model.decode(z, train_neg_edges)
    neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))

    # Total loss
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def evaluate(pos_edges, neg_edges):
    model.eval()

    z = model(node_features, train_edge_index)

    # Positive predictions
    pos_pred = model.decode(z, pos_edges).cpu().numpy()

    # Negative predictions
    neg_pred = model.decode(z, neg_edges).cpu().numpy()

    # Combine
    preds = np.concatenate([pos_pred, neg_pred])
    labels = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])

    # Metrics
    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)

    return auc, ap

# Train for a few epochs
num_epochs = 20
best_val_auc = 0

print("\nEpoch | Train Loss | Val AUC | Val AP")
print("-" * 45)

for epoch in range(1, num_epochs + 1):
    loss = train_epoch()

    if epoch % 5 == 0:
        val_auc, val_ap = evaluate(val_pos_edges, val_neg_edges)
        print(f"{epoch:5d} | {loss:10.4f} | {val_auc:7.4f} | {val_ap:6.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc

# ============================================================================
# FINAL EVALUATION
# ============================================================================
print("\n[5/5] Final evaluation on test set...")

test_auc, test_ap = evaluate(test_pos_edges, test_neg_edges)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Test AUC: {test_auc:.4f}")
print(f"Test AP:  {test_ap:.4f}")
print("="*80)

print("\n✅ Proof-of-concept successful!")
print("   - Graph loaded ✓")
print("   - Model created ✓")
print("   - Training works ✓")
print("   - Evaluation works ✓")
print("\nNext steps: Implement more models and comprehensive experiments!")
