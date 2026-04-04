"""
Fast ranking evaluation — skips GCN (slow, ~23 min) and uses its existing
AUC/AP from results.json. Trains GraphSAGE, GAT, R-GCN and computes full
ranking metrics for all three, plus loads GCN AUC/AP from prior run.
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

node_to_idx = graph_data['node_to_idx']
all_antibiotic_names   = list(graph_data['antibiotics'].keys())
all_antibiotic_indices = torch.tensor(
    [node_to_idx[('antibiotic', name)] for name in all_antibiotic_names],
    dtype=torch.long
)
num_antibiotics = len(all_antibiotic_indices)
ab_idx_to_pos   = {idx.item(): pos for pos, idx in enumerate(all_antibiotic_indices)}

print(f"Nodes: {num_nodes}  |  Test edges: {test_pos.shape[1]}")
print(f"Antibiotics for ranking: {num_antibiotics}")

# ── Negative sampling ─────────────────────────────────────────────────────────
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
def compute_ranking_metrics(model, use_typed=False):
    model.eval()
    z = model(node_features, train_edges_by_type if use_typed else train_edge_index)
    ab_embs = z[all_antibiotic_indices]   # [231, dim]

    ranks = []
    for i in range(test_pos.shape[1]):
        gene_idx    = test_pos[0, i].item()
        true_ab_idx = test_pos[1, i].item()
        if true_ab_idx not in ab_idx_to_pos:
            continue
        gene_emb   = z[gene_idx].unsqueeze(0)
        scores     = torch.sigmoid((gene_emb * ab_embs).sum(-1))
        true_score = scores[ab_idx_to_pos[true_ab_idx]].item()
        rank       = int((scores > true_score).sum().item()) + 1
        ranks.append(rank)

    ranks = np.array(ranks, dtype=float)
    rng   = np.random.default_rng(42)
    rand_ranks = rng.uniform(1, num_antibiotics + 1, size=len(ranks))
    _, p_value = stats.wilcoxon(1.0 / ranks, 1.0 / rand_ranks, alternative='greater')

    return {
        'mrr':                      float(np.mean(1.0 / ranks)),
        'hits@1':                   float(np.mean(ranks <= 1)),
        'hits@3':                   float(np.mean(ranks <= 3)),
        'hits@10':                  float(np.mean(ranks <= 10)),
        'mean_rank':                float(np.mean(ranks)),
        'p_value':                  float(p_value),
        'num_test_pairs':           int(len(ranks)),
        'num_antibiotics_ranked':   int(num_antibiotics),
    }


def train_model(model, epochs=50, lr=0.01, use_typed=False):
    optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_auc = 0
    patience     = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model(node_features, train_edges_by_type if use_typed else train_edge_index)
        pos_scores = model.decode(z, train_pos[:, :train_neg.shape[1]])
        neg_scores = model.decode(z, train_neg)
        loss = (-torch.log(pos_scores + 1e-15).mean()
                - torch.log(1 - neg_scores + 1e-15).mean())
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                z2 = model(node_features, train_edges_by_type if use_typed else train_edge_index)
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


def evaluate_auc_ap(model, use_typed=False):
    model.eval()
    with torch.no_grad():
        z = model(node_features, train_edges_by_type if use_typed else train_edge_index)
        auc, ap = compute_auc_ap(
            model.decode(z, test_pos).cpu().numpy(),
            model.decode(z, test_neg).cpu().numpy()
        )
    return {'auc': auc, 'ap': ap}


# ── Load existing GCN result ──────────────────────────────────────────────────
prior_path = Path(__file__).parent.parent / "results" / "metrics" / "results.json"
with open(prior_path) as f:
    prior = json.load(f)

results = {
    'GCN': {
        'val_auc':      prior['GCN']['val_auc'],
        'test_metrics': prior['GCN']['test_metrics'],
        'ranking_metrics': None,   # not computed (too slow)
        'time':         prior['GCN']['time'],
        'note':         'Ranking metrics not computed (training ~23 min). AUC/AP from prior run.'
    }
}
print("\nGCN: loaded from prior results (AUC/AP only — ranking skipped to save time)")

# ── Train fast models ─────────────────────────────────────────────────────────
fast_models = [
    ('GraphSAGE', lambda: create_graphsage(5, 64, 32, 2),               50, 0.01,  False),
    ('GAT',       lambda: create_gat(5, 32, 32, heads=4, num_layers=2), 50, 0.005, False),
    ('R-GCN',     lambda: create_rgcn(5, 4, 64, 32, 2, num_bases=3),   50, 0.01,  True),
]

print("\n" + "="*80)
print("TRAINING FAST MODELS WITH FULL-DRUG RANKING EVALUATION")
print("="*80)

for idx, (name, make_model, epochs, lr, use_typed) in enumerate(fast_models, 1):
    print(f"\n[{idx}/3] {name}")
    print("-"*60)
    model      = make_model()
    t0         = time.time()
    val_auc    = train_model(model, epochs=epochs, lr=lr, use_typed=use_typed)
    train_time = time.time() - t0

    test_metrics    = evaluate_auc_ap(model, use_typed=use_typed)
    ranking_metrics = compute_ranking_metrics(model, use_typed=use_typed)

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
    print(f"  Mean Rank: {rm['mean_rank']:.1f} / {num_antibiotics}  "
          f"p-value: {rm['p_value']:.2e}")
    print(f"  Time: {train_time:.1f}s")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(f"\n{'Model':<12} {'AUC':>6} {'AP':>6} "
      f"{'MRR':>7} {'H@1':>6} {'H@3':>6} {'H@10':>6} "
      f"{'MeanRk':>7} {'p-val':>9}")
print("-"*75)
for name in ['GCN', 'GraphSAGE', 'GAT', 'R-GCN']:
    r  = results[name]
    tm = r['test_metrics']
    rm = r['ranking_metrics']
    if rm is None:
        print(f"{name:<12} {tm['auc']:6.4f} {tm['ap']:6.4f}  {'N/A (see note)':>45}")
    else:
        print(f"{name:<12} {tm['auc']:6.4f} {tm['ap']:6.4f} "
              f"{rm['mrr']:7.4f} {rm['hits@1']:6.4f} {rm['hits@3']:6.4f} {rm['hits@10']:6.4f} "
              f"{rm['mean_rank']:7.1f} {rm['p_value']:9.2e}")

print(f"\nRandom baseline MRR = {1/num_antibiotics:.4f} (1/{num_antibiotics} antibiotics)")

# ── Save ──────────────────────────────────────────────────────────────────────
results_dir = Path(__file__).parent.parent / "results" / "metrics"
results_dir.mkdir(parents=True, exist_ok=True)
out_path = results_dir / "results_with_ranking.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to: {out_path}")
print("="*80)
