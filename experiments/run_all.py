"""
Run all models and generate comprehensive comparison.

This is the main experiment script for the research paper.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from models.gcn import create_gcn
from models.rgcn import create_rgcn
from models.graphsage import create_graphsage
from models.gat import create_gat
from models.kg_embeddings import create_transe, create_distmult
from experiments.framework import (
    set_seed, efficient_negative_sampling, Trainer, save_results
)
import time

# Configuration
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 100
EARLY_STOPPING = 15
HIDDEN_DIM = 64
EMBEDDING_DIM = 32

# Paths
GRAPH_PATH = Path(__file__).parent.parent / "data" / "graphs" / "card_hetero_graph.pt"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COMPREHENSIVE LINK PREDICTION BENCHMARK")
print("Antibiotic Resistance Prediction with Graph Neural Networks")
print("="*80)

# Set seed
set_seed(SEED)
print(f"\n[CONFIG]")
print(f"  Device: {DEVICE}")
print(f"  Seed: {SEED}")
print(f"  Epochs: {EPOCHS}")
print(f"  Hidden dim: {HIDDEN_DIM}")
print(f"  Embedding dim: {EMBEDDING_DIM}")

# ============================================================================
# LOAD DATA
# ============================================================================
print(f"\n[1/4] Loading heterogeneous graph...")
graph_data = torch.load(GRAPH_PATH)

num_nodes = graph_data['num_nodes']
node_features = graph_data['node_features']
train_typed_edges = graph_data['train_typed_edge_indices']
train_pos = graph_data['train_pos_edges']
val_pos = graph_data['val_pos_edges']
test_pos = graph_data['test_pos_edges']

print(f"  Nodes: {num_nodes}")
print(f"  Node features: {node_features.shape[1]}")
print(f"  Train edges: {train_pos.shape[1]}")
print(f"  Val edges: {val_pos.shape[1]}")
print(f"  Test edges: {test_pos.shape[1]} (ZERO-SHOT: unseen antibiotics!)")

# ============================================================================
# NEGATIVE SAMPLING
# ============================================================================
print(f"\n[2/4] Creating negative samples...")

train_neg = efficient_negative_sampling(num_nodes, train_pos, train_pos.shape[1])
val_neg = efficient_negative_sampling(num_nodes, val_pos, val_pos.shape[1])
test_neg = efficient_negative_sampling(num_nodes, test_pos, test_pos.shape[1])

print(f"  Train neg: {train_neg.shape[1]}")
print(f"  Val neg: {val_neg.shape[1]}")
print(f"  Test neg: {test_neg.shape[1]}")

# Create unified edge index for standard GNNs (flatten all edge types)
train_edge_index = torch.cat([v for v in train_typed_edges.values()], dim=1)
print(f"  Total train edges (for message passing): {train_edge_index.shape[1]}")

# Prepare edge indices by type for R-GCN
edge_type_to_id = graph_data['edge_type_to_id']
train_edges_by_type = {edge_type_to_id[k]: v for k, v in train_typed_edges.items()}

# ============================================================================
# TRAIN ALL MODELS
# ============================================================================
print(f"\n[3/4] Training all models...")

all_results = {}

# Model configurations
models_config = [
    ('GCN', lambda: create_gcn(node_features.shape[1], HIDDEN_DIM, EMBEDDING_DIM, num_layers=2)),
    ('GraphSAGE', lambda: create_graphsage(node_features.shape[1], HIDDEN_DIM, EMBEDDING_DIM, num_layers=2)),
    ('GAT', lambda: create_gat(node_features.shape[1], HIDDEN_DIM, EMBEDDING_DIM, heads=4, num_layers=2)),
    ('R-GCN', lambda: create_rgcn(node_features.shape[1], len(edge_type_to_id), HIDDEN_DIM, EMBEDDING_DIM, num_layers=2, num_bases=4)),
]

for model_name, model_fn in models_config:
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print('='*80)

    # Create model
    model = model_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, optimizer, device=DEVICE, log_interval=10)

    # Train
    start_time = time.time()

    if model_name == 'R-GCN':
        # R-GCN uses typed edges
        best_val_metrics = trainer.fit(
            node_features, None, train_pos, train_neg, val_pos, val_neg,
            epochs=EPOCHS, early_stopping=EARLY_STOPPING,
            edge_indices_by_type=train_edges_by_type
        )
        # Evaluate on test
        test_metrics = trainer.evaluate(
            node_features, None, test_pos, test_neg,
            edge_indices_by_type=train_edges_by_type
        )
    else:
        # Standard GNNs use flattened edges
        best_val_metrics = trainer.fit(
            node_features, train_edge_index, train_pos, train_neg, val_pos, val_neg,
            epochs=EPOCHS, early_stopping=EARLY_STOPPING
        )
        # Evaluate on test
        test_metrics = trainer.evaluate(
            node_features, train_edge_index, test_pos, test_neg
        )

    train_time = time.time() - start_time

    # Store results
    all_results[model_name] = {
        'val_metrics': best_val_metrics,
        'test_metrics': test_metrics,
        'train_time': train_time
    }

    print(f"\n{model_name} Results:")
    print(f"  Val AUC: {best_val_metrics['auc']:.4f}")
    print(f"  Test AUC: {test_metrics['auc']:.4f}")
    print(f"  Test AP: {test_metrics['ap']:.4f}")
    print(f"  Test MRR: {test_metrics.get('mrr', 0):.4f}")
    print(f"  Training time: {train_time:.2f}s")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print(f"\n[4/4] Saving results...")

# Add metadata
all_results['metadata'] = {
    'seed': SEED,
    'epochs': EPOCHS,
    'hidden_dim': HIDDEN_DIM,
    'embedding_dim': EMBEDDING_DIM,
    'num_nodes': num_nodes,
    'train_edges': train_pos.shape[1],
    'val_edges': val_pos.shape[1],
    'test_edges': test_pos.shape[1],
    'task': 'zero-shot antibiotic resistance prediction',
}

save_results(all_results, RESULTS_DIR / "all_models_results.json")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print('='*80)
print(f"\nModel      | Val AUC | Test AUC | Test AP | Test MRR | Time (s)")
print("-" * 70)

for model_name in ['GCN', 'GraphSAGE', 'GAT', 'R-GCN']:
    if model_name in all_results:
        val_auc = all_results[model_name]['val_metrics']['auc']
        test_auc = all_results[model_name]['test_metrics']['auc']
        test_ap = all_results[model_name]['test_metrics']['ap']
        test_mrr = all_results[model_name]['test_metrics'].get('mrr', 0)
        train_time = all_results[model_name]['train_time']

        print(f"{model_name:10s} | {val_auc:7.4f} | {test_auc:8.4f} | "
              f"{test_ap:7.4f} | {test_mrr:8.4f} | {train_time:7.1f}")

print('='*80)
print(f"\nBest model (by Test AUC): {max(all_results.keys(), key=lambda k: all_results[k]['test_metrics']['auc'] if k != 'metadata' else 0)}")
print(f"\nFull results saved to: {RESULTS_DIR / 'all_models_results.json'}")
print('='*80)
