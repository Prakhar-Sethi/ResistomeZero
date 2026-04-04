"""
Minimal training script to debug issues.
Step-by-step validation of the training pipeline.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from models.gcn import create_gcn

print("Step 1: Loading graph...")
graph_data = torch.load(Path(__file__).parent.parent / "data" / "graphs" / "card_hetero_graph.pt")
print(f"  Loaded: {graph_data['num_nodes']} nodes")

print("\nStep 2: Preparing data...")
node_features = graph_data['node_features']
train_pos = graph_data['train_pos_edges']
edge_index = torch.cat([v for v in graph_data['train_typed_edge_indices'].values()], dim=1)
print(f"  Train edges: {train_pos.shape[1]}")
print(f"  Message passing edges: {edge_index.shape[1]}")

print("\nStep 3: Creating negative samples (fast version)...")
num_nodes = graph_data['num_nodes']
num_neg = min(1000, train_pos.shape[1])
neg_src = torch.randint(0, num_nodes, (num_neg,))
neg_dst = torch.randint(0, num_nodes, (num_neg,))
train_neg = torch.stack([neg_src, neg_dst])
print(f"  Negative samples: {train_neg.shape[1]}")

print("\nStep 4: Creating model...")
model = create_gcn(node_features.shape[1], hidden_dim=32, embedding_dim=16, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")

print("\nStep 5: Training 1 epoch...")
model.train()
optimizer.zero_grad()

# Forward
z = model(node_features, edge_index)
print(f"  Embeddings: {z.shape}")

# Decode (use subset for speed)
pos_subset = train_pos[:, :num_neg]
pos_scores = model.decode(z, pos_subset)
neg_scores = model.decode(z, train_neg)
print(f"  Positive scores: {pos_scores.shape}")
print(f"  Negative scores: {neg_scores.shape}")

# Loss
pos_loss = -torch.log(pos_scores + 1e-15).mean()
neg_loss = -torch.log(1 - neg_scores + 1e-15).mean()
loss = pos_loss + neg_loss
print(f"  Loss: {loss.item():.4f}")

# Backward
loss.backward()
optimizer.step()
print("  Backward pass complete")

print("\nStep 6: Evaluation...")
model.eval()
with torch.no_grad():
    z = model(node_features, edge_index)
    pos_scores = model.decode(z, pos_subset)
    neg_scores = model.decode(z, train_neg)

    print(f"  Avg pos score: {pos_scores.mean().item():.4f}")
    print(f"  Avg neg score: {neg_scores.mean().item():.4f}")

    # Simple accuracy
    correct = (pos_scores > 0.5).sum() + (neg_scores < 0.5).sum()
    total = len(pos_scores) + len(neg_scores)
    acc = correct.item() / total
    print(f"  Accuracy: {acc:.4f}")

print("\n" + "="*60)
print("SUCCESS! Minimal training works.")
print("="*60)
