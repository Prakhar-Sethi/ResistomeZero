"""
Simple but complete training script for all models.
No fancy framework - just straightforward training.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from models.gcn import create_gcn
from models.rgcn import create_rgcn
from models.graphsage import create_graphsage
from models.gat import create_gat
import time
import json

# Seed
torch.manual_seed(42)
np.random.seed(42)

# Load graph
print("Loading graph...")
graph_path = Path(__file__).parent.parent / "data" / "graphs" / "card_hetero_graph.pt"
graph_data = torch.load(graph_path)

num_nodes = graph_data['num_nodes']
node_type_map = graph_data['node_type_map']

# ── Rich node features with per-type input projection ──────────────────────
PROJ_DIM = 64  # common projected dimension fed into all GNNs

gene_rich = graph_data.get('gene_rich_features')       # (n_genes, 320) or None
ab_rich   = graph_data.get('antibiotic_rich_features') # (n_ab,    1024) or None

if gene_rich is not None and ab_rich is not None:
    print("Using rich node features (ESM-2 + Morgan fingerprints)")
    # Boundaries from node_type_map
    gene_mask = torch.tensor([node_type_map[i] == 0 for i in range(num_nodes)])
    ab_mask   = torch.tensor([node_type_map[i] == 1 for i in range(num_nodes)])
    other_mask = ~(gene_mask | ab_mask)

    # Learnable per-type projectors
    gene_proj  = nn.Linear(gene_rich.shape[1], PROJ_DIM)
    ab_proj    = nn.Linear(ab_rich.shape[1],   PROJ_DIM)
    other_proj = nn.Linear(graph_data['node_features'].shape[1], PROJ_DIM)  # for drug classes, mechanisms, families, GO terms, KEGG pathways

    # Other node features are one-hot type (5-dim)
    onehot = graph_data['node_features']

    def build_node_features():
        feats = torch.zeros(num_nodes, PROJ_DIM)
        feats[gene_mask]  = gene_proj(gene_rich)
        feats[ab_mask]    = ab_proj(ab_rich)
        feats[other_mask] = other_proj(onehot[other_mask])
        return feats

    all_projectors = nn.ModuleList([gene_proj, ab_proj, other_proj])
    USE_RICH = True
else:
    print("Rich features not found — falling back to one-hot node types")
    USE_RICH = False
    all_projectors = None
    PROJ_DIM = 5

    def build_node_features():
        return graph_data['node_features']

node_features = build_node_features()
train_pos = graph_data['train_pos_edges']
val_pos = graph_data['val_pos_edges']
test_pos = graph_data['test_pos_edges']

# Build edge index
train_typed = graph_data['train_typed_edge_indices']
train_edge_index = torch.cat([v for v in train_typed.values()], dim=1)

# Edge indices by type for R-GCN
edge_type_to_id = graph_data['edge_type_to_id']
train_edges_by_type = {edge_type_to_id[k]: v for k, v in train_typed.items()}

print(f"Nodes: {num_nodes}")
print(f"Train edges: {train_pos.shape[1]}")
print(f"Val edges: {val_pos.shape[1]}")
print(f"Test edges: {test_pos.shape[1]}")

# Fast negative sampling
def sample_neg(num_neg):
    src = torch.randint(0, num_nodes, (num_neg,))
    dst = torch.randint(0, num_nodes, (num_neg,))
    mask = src != dst
    return torch.stack([src[mask][:num_neg], dst[mask][:num_neg]])

print("\nCreating negative samples...")
train_neg = sample_neg(min(2000, train_pos.shape[1]))
val_neg = sample_neg(val_pos.shape[1])
test_neg = sample_neg(test_pos.shape[1])
print(f"Train neg: {train_neg.shape[1]}")

# Compute metrics
def compute_auc_ap(pos_scores, neg_scores):
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap

# Training function
def train_model(model, epochs=50, lr=0.01, use_typed=False):
    params = list(model.parameters())
    if USE_RICH and all_projectors is not None:
        params += list(all_projectors.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    best_val_auc = 0
    patience = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        if USE_RICH and all_projectors is not None:
            all_projectors.train()
        optimizer.zero_grad()

        nf = build_node_features()

        if use_typed:
            z = model(nf, train_edges_by_type)
        else:
            z = model(nf, train_edge_index)

        pos_scores = model.decode(z, train_pos[:, :train_neg.shape[1]])
        neg_scores = model.decode(z, train_neg)

        loss = -torch.log(pos_scores + 1e-15).mean() - torch.log(1 - neg_scores + 1e-15).mean()
        loss.backward()
        optimizer.step()

        # Validate every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            if USE_RICH and all_projectors is not None:
                all_projectors.eval()
            with torch.no_grad():
                nf = build_node_features()
                if use_typed:
                    z = model(nf, train_edges_by_type)
                else:
                    z = model(nf, train_edge_index)

                val_pos_scores = model.decode(z, val_pos).cpu().numpy()
                val_neg_scores = model.decode(z, val_neg).cpu().numpy()

                val_auc, val_ap = compute_auc_ap(val_pos_scores, val_neg_scores)

                print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience = 0
                else:
                    patience += 1

                if patience >= 3:  # Early stop after 3 checks without improvement
                    print(f"  Early stopping at epoch {epoch}")
                    break

    return best_val_auc

# Evaluate function
def evaluate_model(model, use_typed=False):
    model.eval()
    if USE_RICH and all_projectors is not None:
        all_projectors.eval()
    with torch.no_grad():
        nf = build_node_features()
        if use_typed:
            z = model(nf, train_edges_by_type)
        else:
            z = model(nf, train_edge_index)

        test_pos_scores = model.decode(z, test_pos).cpu().numpy()
        test_neg_scores = model.decode(z, test_neg).cpu().numpy()

        test_auc, test_ap = compute_auc_ap(test_pos_scores, test_neg_scores)

        return {'auc': test_auc, 'ap': test_ap}

# Train all models
results = {}

print("\n" + "="*80)
print("TRAINING ALL MODELS")
print("="*80)

# GCN
print("\n[1/4] GCN")
print("-"*80)
model = create_gcn(PROJ_DIM, 64, 32, 2)
start = time.time()
val_auc = train_model(model, epochs=50)
train_time = time.time() - start
test_metrics = evaluate_model(model)
results['GCN'] = {'val_auc': val_auc, 'test_metrics': test_metrics, 'time': train_time}
print(f"GCN - Test AUC: {test_metrics['auc']:.4f}, AP: {test_metrics['ap']:.4f}, Time: {train_time:.1f}s")

# GraphSAGE
print("\n[2/4] GraphSAGE")
print("-"*80)
model = create_graphsage(PROJ_DIM, 64, 32, 2)
start = time.time()
val_auc = train_model(model, epochs=50)
train_time = time.time() - start
test_metrics = evaluate_model(model)
results['GraphSAGE'] = {'val_auc': val_auc, 'test_metrics': test_metrics, 'time': train_time}
print(f"GraphSAGE - Test AUC: {test_metrics['auc']:.4f}, AP: {test_metrics['ap']:.4f}, Time: {train_time:.1f}s")

# GAT
print("\n[3/4] GAT")
print("-"*80)
model = create_gat(PROJ_DIM, 32, 32, heads=4, num_layers=2)
start = time.time()
val_auc = train_model(model, epochs=50, lr=0.005)  # Lower LR for GAT
train_time = time.time() - start
test_metrics = evaluate_model(model)
results['GAT'] = {'val_auc': val_auc, 'test_metrics': test_metrics, 'time': train_time}
print(f"GAT - Test AUC: {test_metrics['auc']:.4f}, AP: {test_metrics['ap']:.4f}, Time: {train_time:.1f}s")

# R-GCN
print("\n[4/4] R-GCN")
print("-"*80)
num_relations = max(graph_data['edge_type_to_id'].values()) + 1
model = create_rgcn(PROJ_DIM, num_relations, 64, 32, 2, num_bases=4)
start = time.time()
val_auc = train_model(model, epochs=50, use_typed=True)
train_time = time.time() - start
test_metrics = evaluate_model(model, use_typed=True)
results['R-GCN'] = {'val_auc': val_auc, 'test_metrics': test_metrics, 'time': train_time}
print(f"R-GCN - Test AUC: {test_metrics['auc']:.4f}, AP: {test_metrics['ap']:.4f}, Time: {train_time:.1f}s")

# Save results
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print("\nModel      | Val AUC | Test AUC | Test AP | Time (s)")
print("-"*70)
for name in ['GCN', 'GraphSAGE', 'GAT', 'R-GCN']:
    r = results[name]
    print(f"{name:10s} | {r['val_auc']:7.4f} | {r['test_metrics']['auc']:8.4f} | {r['test_metrics']['ap']:7.4f} | {r['time']:8.1f}")

# Save to file
results_dir = Path(__file__).parent.parent / "results" / "metrics"
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {results_dir / 'results.json'}")
print("="*80)
