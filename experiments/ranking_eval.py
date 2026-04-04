"""
Full-drug ranking evaluation for zero-shot antibiotic resistance prediction.

For each test (gene, antibiotic) pair, rank ALL 231 antibiotics by model score
and find where the true antibiotic lands.

Metrics reported:
  - AUC, AP   (original, for backward compatibility)
  - MRR       (Mean Reciprocal Rank)
  - Hits@1/3/10
  - Mean Rank
  - p-value vs. random ranking (Wilcoxon signed-rank test)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from models.gcn import create_gcn
from models.rgcn import create_rgcn
from models.graphsage import create_graphsage
from models.gat import create_gat
from models.hgt import create_hgt
from models.distmult import create_distmult
import time
import json

torch.manual_seed(42)
np.random.seed(42)

# ── Load graph ────────────────────────────────────────────────────────────────
print("Loading graph...")
graph_path = Path(__file__).parent.parent / "data" / "graphs" / "card_hetero_graph.pt"
graph_data = torch.load(graph_path, weights_only=False)

num_nodes       = graph_data['num_nodes']
node_features   = graph_data['node_features']
train_pos       = graph_data['train_pos_edges']
val_pos         = graph_data['val_pos_edges']
test_pos        = graph_data['test_pos_edges']

train_typed         = graph_data['train_typed_edge_indices']
train_edge_index    = torch.cat([v for v in train_typed.values()], dim=1)
edge_type_to_id     = graph_data['edge_type_to_id']
train_edges_by_type = {edge_type_to_id[k]: v for k, v in train_typed.items()}

# Metadata for HGT and DistMult
node_type_map   = graph_data['node_type_map']   # {node_idx: type_id}
num_node_types  = max(node_type_map.values()) + 1

# All 231 antibiotic node indices (used for full-drug ranking)
node_to_idx = graph_data['node_to_idx']
all_antibiotic_names   = list(graph_data['antibiotics'].keys())
all_antibiotic_indices = torch.tensor(
    [node_to_idx[('antibiotic', name)] for name in all_antibiotic_names],
    dtype=torch.long
)
num_antibiotics = len(all_antibiotic_indices)

print(f"Nodes: {num_nodes}")
print(f"Train edges: {train_pos.shape[1]}  |  Val: {val_pos.shape[1]}  |  Test: {test_pos.shape[1]}")
print(f"Antibiotics available for ranking: {num_antibiotics}")

# ── Negative sampling (training only) ────────────────────────────────────────
def sample_neg(n):
    src = torch.randint(0, num_nodes, (n,))
    dst = torch.randint(0, num_nodes, (n,))
    mask = src != dst
    return torch.stack([src[mask][:n], dst[mask][:n]])

train_neg = sample_neg(min(2000, train_pos.shape[1]))
val_neg   = sample_neg(val_pos.shape[1])
test_neg  = sample_neg(test_pos.shape[1])

# ── Metric helpers ────────────────────────────────────────────────────────────
def compute_auc_ap(pos_scores, neg_scores):
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


@torch.no_grad()
def compute_ranking_metrics(model, use_typed=False, is_distmult=False):
    """
    For every test (gene, antibiotic) pair, score the gene against ALL 231
    antibiotics and record the rank of the true antibiotic.

    Random baseline MRR = 1/231 ≈ 0.0043.
    """
    model.eval()
    z = get_embeddings(model, use_typed, is_distmult)

    ab_embs = z[all_antibiotic_indices]          # [231, dim]
    # Pre-build a lookup: antibiotic global index → position in all_antibiotic_indices
    ab_idx_to_pos = {idx.item(): pos for pos, idx in enumerate(all_antibiotic_indices)}

    ranks = []
    for i in range(test_pos.shape[1]):
        gene_idx    = test_pos[0, i].item()
        true_ab_idx = test_pos[1, i].item()

        if true_ab_idx not in ab_idx_to_pos:
            continue  # safety check

        gene_emb = z[gene_idx].unsqueeze(0)                        # [1, dim]
        scores   = torch.sigmoid((gene_emb * ab_embs).sum(-1))     # [231]

        true_score = scores[ab_idx_to_pos[true_ab_idx]].item()
        rank = int((scores > true_score).sum().item()) + 1          # 1-indexed
        ranks.append(rank)

    ranks = np.array(ranks, dtype=float)

    # Random-baseline ranks (uniform over 231) for significance test
    rng           = np.random.default_rng(42)
    random_ranks  = rng.uniform(1, num_antibiotics + 1, size=len(ranks))
    _, p_value    = stats.wilcoxon(1.0 / ranks, 1.0 / random_ranks, alternative='greater')

    return {
        'mrr':          float(np.mean(1.0 / ranks)),
        'hits@1':       float(np.mean(ranks <= 1)),
        'hits@3':       float(np.mean(ranks <= 3)),
        'hits@10':      float(np.mean(ranks <= 10)),
        'mean_rank':    float(np.mean(ranks)),
        'p_value':      float(p_value),
        'num_test_pairs':           int(len(ranks)),
        'num_antibiotics_ranked':   int(num_antibiotics),
    }


# ── Helper: get embeddings for any model type ────────────────────────────────
def get_embeddings(model, use_typed=False, is_distmult=False):
    """Return node embeddings z. For DistMult, returns entity embedding matrix."""
    if is_distmult:
        return model.get_all_entity_embeddings()
    elif use_typed:
        return model(node_features, train_edges_by_type)
    else:
        return model(node_features, train_edge_index)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_model(model, epochs=50, lr=0.01, use_typed=False, is_distmult=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_auc = 0
    patience     = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        if is_distmult:
            pos_scores = model(train_pos[:, :train_neg.shape[1]])
            neg_scores = model(train_neg)
        else:
            z = get_embeddings(model, use_typed)
            pos_scores = model.decode(z, train_pos[:, :train_neg.shape[1]])
            neg_scores = model.decode(z, train_neg)

        loss = (-torch.log(pos_scores + 1e-15).mean()
                - torch.log(1 - neg_scores + 1e-15).mean())
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                z2 = get_embeddings(model, use_typed, is_distmult)
                val_auc, _ = compute_auc_ap(
                    model.decode(z2, val_pos).cpu().numpy(),
                    model.decode(z2, val_neg).cpu().numpy()
                )
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}")
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience     = 0
            else:
                patience += 1
            if patience >= 3:
                print(f"  Early stopping at epoch {epoch}")
                break

    return best_val_auc


def evaluate_auc_ap(model, use_typed=False, is_distmult=False):
    model.eval()
    with torch.no_grad():
        z = get_embeddings(model, use_typed, is_distmult)
        auc, ap = compute_auc_ap(
            model.decode(z, test_pos).cpu().numpy(),
            model.decode(z, test_neg).cpu().numpy()
        )
    return {'auc': auc, 'ap': ap}


# ── Run all four models ───────────────────────────────────────────────────────
_in_dim      = node_features.shape[1]
_num_rel     = max(edge_type_to_id.values()) + 1

# (name, factory_fn, epochs, lr, use_typed, is_distmult)
models_config = [
    ('GCN',       lambda: create_gcn(_in_dim, 64, 32, 2),                                          50, 0.01,  False, False),
    ('GraphSAGE', lambda: create_graphsage(_in_dim, 64, 32, 2),                                     50, 0.01,  False, False),
    ('GAT',       lambda: create_gat(_in_dim, 32, 32, heads=4, num_layers=2),                       50, 0.005, False, False),
    ('R-GCN',     lambda: create_rgcn(_in_dim, _num_rel, 64, 32, 2, num_bases=4),                  50, 0.01,  True,  False),
    ('HGT',       lambda: create_hgt(_in_dim, num_node_types, _num_rel, 64, 32, num_heads=4, num_layers=2), 50, 0.005, True,  False),
    ('DistMult',  lambda: create_distmult(num_nodes, embedding_dim=64),                             100, 0.01, False, True),
]

print("\n" + "="*80)
print("TRAINING ALL MODELS  —  FULL-DRUG RANKING EVALUATION")
print("="*80)

results = {}
n_models = len(models_config)
for idx, (name, make_model, epochs, lr, use_typed, is_distmult) in enumerate(models_config, 1):
    print(f"\n[{idx}/{n_models}] {name}")
    print("-"*60)
    model = make_model()

    # HGT needs node_type_map injected before forward()
    if name == 'HGT':
        model.node_type_map = node_type_map

    t0         = time.time()
    val_auc    = train_model(model, epochs=epochs, lr=lr, use_typed=use_typed, is_distmult=is_distmult)
    train_time = time.time() - t0

    test_metrics    = evaluate_auc_ap(model, use_typed=use_typed, is_distmult=is_distmult)
    ranking_metrics = compute_ranking_metrics(model, use_typed=use_typed, is_distmult=is_distmult)

    results[name] = {
        'val_auc':         val_auc,
        'test_metrics':    test_metrics,
        'ranking_metrics': ranking_metrics,
        'time':            train_time,
    }

    rm = ranking_metrics
    print(f"\n  {name} — AUC: {test_metrics['auc']:.4f}  AP: {test_metrics['ap']:.4f}")
    print(f"  MRR: {rm['mrr']:.4f}  Hits@1: {rm['hits@1']:.4f}  "
          f"Hits@3: {rm['hits@3']:.4f}  Hits@10: {rm['hits@10']:.4f}")
    print(f"  Mean Rank: {rm['mean_rank']:.1f} / {rm['num_antibiotics_ranked']}  "
          f"p-value: {rm['p_value']:.2e}  ({rm['num_test_pairs']} test pairs)")
    print(f"  Training time: {train_time:.1f}s")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(f"\n{'Model':<12} {'AUC':>6} {'AP':>6} "
      f"{'MRR':>7} {'H@1':>6} {'H@3':>6} {'H@10':>6} "
      f"{'MeanRk':>7} {'p-val':>9} {'Time':>8}")
print("-"*80)
for name in ['GCN', 'GraphSAGE', 'GAT', 'R-GCN', 'HGT', 'DistMult']:
    r  = results[name]
    tm = r['test_metrics']
    rm = r['ranking_metrics']
    print(f"{name:<12} {tm['auc']:6.4f} {tm['ap']:6.4f} "
          f"{rm['mrr']:7.4f} {rm['hits@1']:6.4f} {rm['hits@3']:6.4f} {rm['hits@10']:6.4f} "
          f"{rm['mean_rank']:7.1f} {rm['p_value']:9.2e} {r['time']:7.1f}s")

print(f"\nRandom baseline MRR = {1/num_antibiotics:.4f}  "
      f"(1/{num_antibiotics} antibiotics)")

# ── Save ──────────────────────────────────────────────────────────────────────
results_dir = Path(__file__).parent.parent / "results" / "metrics"
results_dir.mkdir(parents=True, exist_ok=True)
out_path = results_dir / "results_with_ranking.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to: {out_path}")
print("="*80)
