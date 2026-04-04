"""Quick demo: Train ONE model (GCN) to validate pipeline."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from models.gcn import create_gcn
from experiments.framework import set_seed, efficient_negative_sampling, Trainer

print("QUICK DEMO: Training GCN")
print("="*60)

# Load graph
graph_path = Path(__file__).parent.parent / "data" / "graphs" / "card_hetero_graph.pt"
print(f"Loading graph from {graph_path}...")
graph_data = torch.load(graph_path)

num_nodes = graph_data['num_nodes']
node_features = graph_data['node_features']
train_pos = graph_data['train_pos_edges']
val_pos = graph_data['val_pos_edges']
test_pos = graph_data['test_pos_edges']

# Build train graph
train_edge_index = torch.cat([v for v in graph_data['train_typed_edge_indices'].values()], dim=1)

print(f"  Nodes: {num_nodes}")
print(f"  Train edges: {train_pos.shape[1]}")

# Negative sampling
print("Creating negative samples...")
train_neg = efficient_negative_sampling(num_nodes, train_pos, min(1000, train_pos.shape[1]))
val_neg = efficient_negative_sampling(num_nodes, val_pos, min(500, val_pos.shape[1]))
test_neg = efficient_negative_sampling(num_nodes, test_pos, min(500, test_pos.shape[1]))

print(f"  Train neg: {train_neg.shape[1]}")

# Create model
print("Creating GCN...")
set_seed(42)
model = create_gcn(node_features.shape[1], hidden_dim=32, embedding_dim=16, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
trainer = Trainer(model, optimizer, device='cpu', log_interval=5)

# Train
print("Training...")
best_metrics = trainer.fit(
    node_features, train_edge_index,
    train_pos[:, :1000], train_neg,  # Use subset for speed
    val_pos[:, :500], val_neg,
    epochs=20, early_stopping=5
)

# Test
print("\nEvaluating on test set...")
test_metrics = trainer.evaluate(
    node_features, train_edge_index,
    test_pos[:, :500], test_neg
)

print("\n" + "="*60)
print("RESULTS:")
print(f"  Val AUC:  {best_metrics['auc']:.4f}")
print(f"  Test AUC: {test_metrics['auc']:.4f}")
print(f"  Test AP:  {test_metrics['ap']:.4f}")
print("="*60)
print("\nPipeline validated! Ready to run all models.")
