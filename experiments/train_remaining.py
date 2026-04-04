"""Train GAT and R-GCN to complete the comparison."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from models.gat import create_gat
from models.rgcn import create_rgcn
import time
import json

torch.manual_seed(42)
np.random.seed(42)

# Load existing results
results_path = Path(__file__).parent.parent / "results" / "metrics" / "results.json"
if results_path.exists():
    with open(results_path, 'r') as f:
        results = json.load(f)
    print("Loaded existing results for GCN and GraphSAGE")
else:
    results = {}

# Load graph
print("Loading graph...")
graph_data = torch.load(Path(__file__).parent.parent / "data" / "graphs" / "card_hetero_graph.pt")

num_nodes = graph_data['num_nodes']
node_features = graph_data['node_features']
train_pos = graph_data['train_pos_edges']
val_pos = graph_data['val_pos_edges']
test_pos = graph_data['test_pos_edges']

train_typed = graph_data['train_typed_edge_indices']
train_edge_index = torch.cat([v for v in train_typed.values()], dim=1)

edge_type_to_id = graph_data['edge_type_to_id']
train_edges_by_type = {edge_type_to_id[k]: v for k, v in train_typed.items()}

# Negative samples
def sample_neg(num_neg):
    src = torch.randint(0, num_nodes, (num_neg,))
    dst = torch.randint(0, num_nodes, (num_neg,))
    mask = src != dst
    return torch.stack([src[mask][:num_neg], dst[mask][:num_neg]])

train_neg = sample_neg(min(2000, train_pos.shape[1]))
val_neg = sample_neg(val_pos.shape[1])
test_neg = sample_neg(test_pos.shape[1])

def compute_auc_ap(pos_scores, neg_scores):
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)

def train_model(model, epochs=50, lr=0.01, use_typed=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_auc = 0
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        if use_typed:
            z = model(node_features, train_edges_by_type)
        else:
            z = model(node_features, train_edge_index)

        pos_scores = model.decode(z, train_pos[:, :train_neg.shape[1]])
        neg_scores = model.decode(z, train_neg)

        loss = -torch.log(pos_scores + 1e-15).mean() - torch.log(1 - neg_scores + 1e-15).mean()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                if use_typed:
                    z = model(node_features, train_edges_by_type)
                else:
                    z = model(node_features, train_edge_index)

                val_pos_scores = model.decode(z, val_pos).cpu().numpy()
                val_neg_scores = model.decode(z, val_neg).cpu().numpy()
                val_auc, val_ap = compute_auc_ap(val_pos_scores, val_neg_scores)

                print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience = 0
                else:
                    patience += 1

                if patience >= 3:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    return best_val_auc

def evaluate_model(model, use_typed=False):
    model.eval()
    with torch.no_grad():
        if use_typed:
            z = model(node_features, train_edges_by_type)
        else:
            z = model(node_features, train_edge_index)

        test_pos_scores = model.decode(z, test_pos).cpu().numpy()
        test_neg_scores = model.decode(z, test_neg).cpu().numpy()
        test_auc, test_ap = compute_auc_ap(test_pos_scores, test_neg_scores)

        return {'auc': test_auc, 'ap': test_ap}

print("\n" + "="*80)
print("TRAINING REMAINING MODELS")
print("="*80)

# GAT
print("\n[1/2] GAT")
print("-"*80)
model = create_gat(5, 32, 32, heads=4, num_layers=2)
start = time.time()
val_auc = train_model(model, epochs=50, lr=0.005)
train_time = time.time() - start
test_metrics = evaluate_model(model)
results['GAT'] = {'val_auc': val_auc, 'test_metrics': test_metrics, 'time': train_time}
print(f"GAT - Test AUC: {test_metrics['auc']:.4f}, AP: {test_metrics['ap']:.4f}, Time: {train_time:.1f}s")

# R-GCN
print("\n[2/2] R-GCN")
print("-"*80)
model = create_rgcn(5, 4, 64, 32, 2, num_bases=3)
start = time.time()
val_auc = train_model(model, epochs=50, use_typed=True)
train_time = time.time() - start
test_metrics = evaluate_model(model, use_typed=True)
results['R-GCN'] = {'val_auc': val_auc, 'test_metrics': test_metrics, 'time': train_time}
print(f"R-GCN - Test AUC: {test_metrics['auc']:.4f}, AP: {test_metrics['ap']:.4f}, Time: {train_time:.1f}s")

# Save all results
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("ALL RESULTS")
print("="*80)
print("\nModel      | Val AUC | Test AUC | Test AP | Time (s)")
print("-"*70)
for name in ['GCN', 'GraphSAGE', 'GAT', 'R-GCN']:
    if name in results:
        r = results[name]
        print(f"{name:10s} | {r['val_auc']:7.4f} | {r['test_metrics']['auc']:8.4f} | {r['test_metrics']['ap']:7.4f} | {r['time']:8.1f}")

print(f"\nResults saved to {results_path}")
print("="*80)
