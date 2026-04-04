"""
SUPER QUICK proof-of-concept: Just test one epoch to verify pipeline works.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from models.gcn import create_gcn
import numpy as np

print("="*80)
print("GCN LINK PREDICTION - QUICK TEST")
print("="*80)

# Load graph
print("\n[1/4] Loading graph...")
graph_path = Path(__file__).parent.parent / "data" / "graphs" / "card_graph.pt"
graph_data = torch.load(graph_path)

num_nodes = graph_data['num_nodes']
node_features = graph_data['node_features']
train_edge_index = graph_data['train_edge_index']
train_pos_edges = graph_data['train_pos_edge_index']

print(f"   Nodes: {num_nodes}")
print(f"   Train pos edges: {train_pos_edges.shape[1]}")

# Create simple negative samples (just random, not checking if they exist)
print("\n[2/4] Creating negative samples (simple random)...")
num_neg = min(1000, train_pos_edges.shape[1])  # Just 1000 for speed
neg_src = torch.randint(0, num_nodes, (num_neg,))
neg_dst = torch.randint(0, num_nodes, (num_neg,))
train_neg_edges = torch.stack([neg_src, neg_dst])
print(f"   Neg edges: {num_neg}")

# Create model
print("\n[3/4] Creating model and training 1 epoch...")
model = create_gcn(5, 32, 16, 2, 0.3)  # Smaller model for speed
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train one epoch
model.train()
optimizer.zero_grad()

z = model(node_features, train_edge_index)

# Positive samples (use subset for speed)
pos_subset = train_pos_edges[:, :num_neg]
pos_pred = model.decode(z, pos_subset)
pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))

# Negative samples
neg_pred = model.decode(z, train_neg_edges)
neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))

loss = pos_loss + neg_loss
loss.backward()
optimizer.step()

print(f"   Loss after 1 epoch: {loss.item():.4f}")

# Quick evaluation
print("\n[4/4] Quick evaluation...")
model.eval()
with torch.no_grad():
    z = model(node_features, train_edge_index)

    # Eval on subset
    val_pos = graph_data['val_pos_edge_index'][:, :100]
    val_neg_src = torch.randint(0, num_nodes, (100,))
    val_neg_dst = torch.randint(0, num_nodes, (100,))
    val_neg = torch.stack([val_neg_src, val_neg_dst])

    pos_scores = model.decode(z, val_pos).cpu().numpy()
    neg_scores = model.decode(z, val_neg).cpu().numpy()

    print(f"   Avg positive score: {pos_scores.mean():.4f}")
    print(f"   Avg negative score: {neg_scores.mean():.4f}")

print("\n" + "="*80)
print("✅ PIPELINE WORKS!")
print("="*80)
print("\nComponents verified:")
print("  ✓ Graph loading")
print("  ✓ Model creation")
print("  ✓ Forward pass")
print("  ✓ Loss computation")
print("  ✓ Backward pass")
print("  ✓ Evaluation")
print("\nReady to implement full training pipeline!")
print("="*80)
