"""
Multi-seed evaluation — FIXED VERSION.

Fixes applied vs previous version:
  1. Hard negatives: sample only gene x antibiotic pairs (not random node pairs)
  2. Rich features: ESM-2 (genes) + Morgan fingerprints (antibiotics) projected
     to 64-dim via trained linear layers, gradients flowing properly.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict
from models.gcn import create_gcn
from models.rgcn import create_rgcn
from models.graphsage import create_graphsage
from models.gat import create_gat
from models.hgt import create_hgt
from models.distmult import create_distmult
from models.complex import create_complex
from models.transe import create_transe
import time
import json

SEEDS    = [42, 123, 456, 789, 1234]
EPOCHS   = 80
PATIENCE = 4
PROJ_DIM = 64   # projected feature dimension

# ── Load graph ────────────────────────────────────────────────────────────────
print("Loading graph...")
graph_path = Path(__file__).parent.parent / "data" / "graphs" / "card_hetero_graph.pt"
graph_data = torch.load(graph_path, weights_only=False)

num_nodes      = graph_data['num_nodes']
node_features  = graph_data['node_features']          # [N, 8] one-hot type
gene_rich      = graph_data['gene_rich_features']      # [7240, 320] ESM-2
ab_rich        = graph_data['antibiotic_rich_features'] # [231, 1024] Morgan

train_pos      = graph_data['train_pos_edges']
val_pos        = graph_data['val_pos_edges']
test_pos       = graph_data['test_pos_edges']

train_typed         = graph_data['train_typed_edge_indices']
train_edge_index    = torch.cat([v for v in train_typed.values()], dim=1)
edge_type_to_id     = graph_data['edge_type_to_id']
train_edges_by_type = {edge_type_to_id[k]: v for k, v in train_typed.items()}

node_type_map  = graph_data['node_type_map']
num_node_types = max(node_type_map.values()) + 1
node_to_idx    = graph_data['node_to_idx']

# ── Node index sets for correct negative sampling ─────────────────────────────
gene_idx_list  = torch.tensor([i for i, t in node_type_map.items() if t == 0], dtype=torch.long)
ab_idx_list    = torch.tensor([i for i, t in node_type_map.items() if t == 1], dtype=torch.long)

train_ab_names = set(graph_data['train_antibiotics'])
val_ab_names   = set(graph_data['val_antibiotics'])
test_ab_names  = set(graph_data['test_antibiotics'])

train_ab_idx = torch.tensor(
    [node_to_idx[('antibiotic', n)] for n in train_ab_names if ('antibiotic', n) in node_to_idx],
    dtype=torch.long)
val_ab_idx   = torch.tensor(
    [node_to_idx[('antibiotic', n)] for n in val_ab_names   if ('antibiotic', n) in node_to_idx],
    dtype=torch.long)
test_ab_idx  = torch.tensor(
    [node_to_idx[('antibiotic', n)] for n in test_ab_names  if ('antibiotic', n) in node_to_idx],
    dtype=torch.long)

# All antibiotics for ranking
all_ab_names   = list(graph_data['antibiotics'].keys())
all_ab_indices = torch.tensor(
    [node_to_idx[('antibiotic', n)] for n in all_ab_names], dtype=torch.long)
num_antibiotics = len(all_ab_indices)

# Existing positive edges as a set for filtering negatives
pos_edge_set = set(
    zip(train_pos[0].tolist(), train_pos[1].tolist())
)
pos_edge_set.update(zip(val_pos[0].tolist(),  val_pos[1].tolist()))
pos_edge_set.update(zip(test_pos[0].tolist(), test_pos[1].tolist()))

# Per-gene training positives for filtered ranking (excludes known positives from ranking pool)
gene_to_train_pos_abs = defaultdict(set)
for _i in range(train_pos.shape[1]):
    gene_to_train_pos_abs[train_pos[0, _i].item()].add(train_pos[1, _i].item())

# Drug-class co-resistance heuristic: extract antibiotic → drug class mapping
_train_ab_set = set(train_ab_idx.tolist())
ab_to_dc      = defaultdict(set)
dc_to_train_abs = defaultdict(set)
if 'antibiotic_to_class' in train_typed:
    _atc = train_typed['antibiotic_to_class']
    for _i in range(_atc.shape[1]):
        _ab_i, _dc_i = _atc[0, _i].item(), _atc[1, _i].item()
        ab_to_dc[_ab_i].add(_dc_i)
        if _ab_i in _train_ab_set:
            dc_to_train_abs[_dc_i].add(_ab_i)

_in_dim  = node_features.shape[1]   # 8 (one-hot), will be overridden to PROJ_DIM below
_num_rel = max(edge_type_to_id.values()) + 1

print(f"Nodes: {num_nodes}  |  Train: {train_pos.shape[1]}  |  Val: {val_pos.shape[1]}  |  Test: {test_pos.shape[1]}")
print(f"Gene nodes: {len(gene_idx_list)}  |  Train antibiotics: {len(train_ab_idx)}  |  Test antibiotics: {len(test_ab_idx)}")
print(f"Seeds: {SEEDS}")

# ── FIX 1: Hard negative sampling (gene x antibiotic pairs only) ──────────────
def sample_neg_hard(n, seed, ab_pool):
    """
    Sample n negative (gene, antibiotic) pairs.
    ab_pool: which antibiotics to sample from (train or test pool).
    Filters out known positive edges.
    """
    rng  = torch.Generator(); rng.manual_seed(seed)
    negs = []
    attempts = 0
    while len(negs) < n and attempts < n * 20:
        gi = gene_idx_list[torch.randint(len(gene_idx_list), (1,), generator=rng).item()].item()
        ai = ab_pool[torch.randint(len(ab_pool), (1,), generator=rng).item()].item()
        if (gi, ai) not in pos_edge_set:
            negs.append([gi, ai])
        attempts += 1
    if len(negs) < n:
        # Pad with whatever we got if not enough unique negatives
        while len(negs) < n:
            negs.append(negs[-1])
    t = torch.tensor(negs[:n], dtype=torch.long).t()
    return t   # [2, n]


# ── FIX 2: Rich feature projector (trained jointly with model) ────────────────
class FeatureProjector(nn.Module):
    """
    Projects ESM-2 gene features and Morgan antibiotic fingerprints
    to a shared PROJ_DIM space. Other node types use the one-hot type vector.
    Trained end-to-end with the GNN.
    """
    def __init__(self, proj_dim=64):
        super().__init__()
        self.gene_proj  = nn.Linear(gene_rich.shape[1], proj_dim)
        self.ab_proj    = nn.Linear(ab_rich.shape[1],   proj_dim)
        self.other_proj = nn.Linear(node_features.shape[1], proj_dim)

        # Masks (fixed)
        self.register_buffer('gene_mask',  torch.tensor(
            [node_type_map[i] == 0 for i in range(num_nodes)]))
        self.register_buffer('ab_mask',    torch.tensor(
            [node_type_map[i] == 1 for i in range(num_nodes)]))

    def forward(self):
        other_mask = ~(self.gene_mask | self.ab_mask)
        feats = torch.zeros(num_nodes, self.gene_proj.out_features,
                            device=gene_rich.device)
        feats[self.gene_mask]  = self.gene_proj(gene_rich)
        feats[self.ab_mask]    = self.ab_proj(ab_rich)
        feats[other_mask]      = self.other_proj(node_features[other_mask])
        return feats   # [N, PROJ_DIM]


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_auc_ap(pos_scores, neg_scores):
    s = np.concatenate([pos_scores, neg_scores])
    l = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    return roc_auc_score(l, s), average_precision_score(l, s)


@torch.no_grad()
def compute_ranking(model, projector, get_z):
    """Filtered ranking: for each test pair (gene, true_ab), exclude the gene's
    known training positives from the ranking pool so they don't inflate ranks.
    This is the standard 'filtered' MRR used in KGE evaluation (Bordes et al. 2013)."""
    model.eval(); projector.eval()
    nf = projector()
    z  = get_z(model, nf)
    ab_embs    = z[all_ab_indices]                          # [231, dim]
    ab_idx2pos = {idx.item(): pos for pos, idx in enumerate(all_ab_indices)}

    ranks = []
    for i in range(test_pos.shape[1]):
        g_idx, ab_idx = test_pos[0, i].item(), test_pos[1, i].item()
        if ab_idx not in ab_idx2pos: continue
        if hasattr(model, 'score_all_tails'):
            scores = torch.sigmoid(model.score_all_tails(g_idx, all_ab_indices))
        else:
            scores = torch.sigmoid((z[g_idx].unsqueeze(0) * ab_embs).sum(-1))  # [231]
        true_score = scores[ab_idx2pos[ab_idx]].item()

        # Build valid mask: exclude training positives for this gene (except the true AB)
        train_pos_for_gene = gene_to_train_pos_abs.get(g_idx, set())
        valid_mask = torch.ones(num_antibiotics, dtype=torch.bool)
        for tp_ab in train_pos_for_gene:
            if tp_ab in ab_idx2pos:
                valid_mask[ab_idx2pos[tp_ab]] = False
        valid_mask[ab_idx2pos[ab_idx]] = True   # always include the true AB

        rank = int((scores[valid_mask] > true_score).sum()) + 1
        ranks.append(rank)

    ranks = np.array(ranks, dtype=float)
    rand  = np.random.default_rng(42).uniform(1, num_antibiotics + 1, len(ranks))
    _, p  = stats.wilcoxon(1.0 / ranks, 1.0 / rand, alternative='greater')
    # Effect size: rank-biserial correlation r = 1 - 2*mean_rank/N
    effect_size = 1.0 - 2.0 * float(np.mean(ranks)) / num_antibiotics
    return {
        'mrr':         float(np.mean(1.0 / ranks)),
        'hits@1':      float(np.mean(ranks <= 1)),
        'hits@3':      float(np.mean(ranks <= 3)),
        'hits@10':     float(np.mean(ranks <= 10)),
        'mean_rank':   float(np.mean(ranks)),
        'p_value':     float(p),
        'effect_size': effect_size,
    }


def compute_drug_class_heuristic():
    """Drug-class co-resistance heuristic (no training required).
    score(gene g, AB a) = fraction of training ABs sharing a's drug class that g resists.
    Uses filtered ranking like compute_ranking.
    """
    ab_idx2pos_h = {idx.item(): pos for pos, idx in enumerate(all_ab_indices)}
    all_ab_list  = all_ab_indices.tolist()

    ranks = []
    for i in range(test_pos.shape[1]):
        g_idx, ab_idx = test_pos[0, i].item(), test_pos[1, i].item()
        if ab_idx not in ab_idx2pos_h: continue

        gene_train_pos = gene_to_train_pos_abs.get(g_idx, set())

        # Score each AB by drug-class co-resistance
        scores = np.zeros(num_antibiotics)
        for j, cand_ab in enumerate(all_ab_list):
            dc_set = ab_to_dc.get(cand_ab, set())
            if not dc_set: continue
            siblings = set()
            for dc in dc_set:
                siblings |= dc_to_train_abs.get(dc, set())
            siblings.discard(cand_ab)
            if not siblings: continue
            scores[j] = len(gene_train_pos & siblings) / len(siblings)

        true_score = scores[ab_idx2pos_h[ab_idx]]

        # Filtered rank (exclude training positives)
        valid_mask = np.ones(num_antibiotics, dtype=bool)
        for tp_ab in gene_train_pos:
            if tp_ab in ab_idx2pos_h:
                valid_mask[ab_idx2pos_h[tp_ab]] = False
        valid_mask[ab_idx2pos_h[ab_idx]] = True

        rank = int(np.sum(scores[valid_mask] > true_score)) + 1
        ranks.append(rank)

    if not ranks:
        return {'mrr': 0., 'hits@1': 0., 'hits@3': 0., 'hits@10': 0.,
                'mean_rank': float(num_antibiotics), 'p_value': 1., 'effect_size': -1.}

    ranks = np.array(ranks, dtype=float)
    rand  = np.random.default_rng(42).uniform(1, num_antibiotics + 1, len(ranks))
    try:
        _, p = stats.wilcoxon(1.0 / ranks, 1.0 / rand, alternative='greater')
    except Exception:
        p = 1.0
    effect_size = 1.0 - 2.0 * float(np.mean(ranks)) / num_antibiotics
    return {
        'mrr':         float(np.mean(1.0 / ranks)),
        'hits@1':      float(np.mean(ranks <= 1)),
        'hits@3':      float(np.mean(ranks <= 3)),
        'hits@10':     float(np.mean(ranks <= 10)),
        'mean_rank':   float(np.mean(ranks)),
        'p_value':     float(p),
        'effect_size': effect_size,
    }


# ── get_z helper ──────────────────────────────────────────────────────────────
def make_get_z(use_typed, is_distmult):
    def get_z(model, nf):
        if is_distmult: return model.get_all_entity_embeddings()
        elif use_typed:  return model(nf, train_edges_by_type)
        else:            return model(nf, train_edge_index)
    return get_z


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_seed(make_model, epochs, lr, use_typed, is_distmult, seed):
    torch.manual_seed(seed); np.random.seed(seed)

    projector = FeatureProjector(PROJ_DIM)
    model     = make_model(PROJ_DIM)
    get_z     = make_get_z(use_typed, is_distmult)

    # Train projector + model jointly (not DistMult — it has its own embeddings)
    params = list(model.parameters())
    if not is_distmult:
        params += list(projector.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # Pre-sample negatives with correct hard sampling.
    # Val negatives use val_ab_idx (zero-shot val pool) so early stopping is
    # aligned with the zero-shot evaluation objective.
    train_neg = sample_neg_hard(min(2000, train_pos.shape[1]), seed,    train_ab_idx)
    val_neg   = sample_neg_hard(val_pos.shape[1],              seed+1,  val_ab_idx)
    test_neg  = sample_neg_hard(test_pos.shape[1],             seed+2,  test_ab_idx)

    best_val_auc = 0
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train(); projector.train()
        optimizer.zero_grad()

        # Compute features INSIDE loop so gradients flow to projector
        if is_distmult:
            pos_scores = model(train_pos[:, :train_neg.shape[1]])
            neg_scores = model(train_neg)
        else:
            nf = projector()   # [N, PROJ_DIM] — gradient flows here
            z  = get_z(model, nf)
            pos_scores = model.decode(z, train_pos[:, :train_neg.shape[1]])
            neg_scores = model.decode(z, train_neg)

        loss = (-torch.log(pos_scores + 1e-15).mean()
                - torch.log(1 - neg_scores + 1e-15).mean())
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            model.eval(); projector.eval()
            with torch.no_grad():
                nf2 = projector() if not is_distmult else None
                z2  = get_z(model, nf2)
                val_auc, _ = compute_auc_ap(
                    model.decode(z2, val_pos).cpu().numpy(),
                    model.decode(z2, val_neg).cpu().numpy()
                )
            if val_auc > best_val_auc:
                best_val_auc = val_auc; patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
            model.train(); projector.train()

    # Final evaluation
    model.eval(); projector.eval()
    with torch.no_grad():
        nf = projector() if not is_distmult else None
        z  = get_z(model, nf)
        auc, ap = compute_auc_ap(
            model.decode(z, test_pos).cpu().numpy(),
            model.decode(z, test_neg).cpu().numpy()
        )
        ranking = compute_ranking(model, projector, get_z)

    return {'auc': auc, 'ap': ap, **ranking}


# ── Model configs ─────────────────────────────────────────────────────────────
def _make_hgt(d):
    m = create_hgt(d, num_node_types, _num_rel, 64, 32, num_heads=4, num_layers=2)
    m.node_type_map = node_type_map
    return m

MODEL_CONFIGS = [
    ('GCN',       lambda d: create_gcn(d, 64, 32, 2),                           80,  0.01,  False, False),
    ('GraphSAGE', lambda d: create_graphsage(d, 64, 32, 2),                      80,  0.01,  False, False),
    ('GAT',       lambda d: create_gat(d, 32, 32, heads=4, num_layers=2),        80,  0.005, False, False),
    ('R-GCN',     lambda d: create_rgcn(d, _num_rel, 64, 32, 2, num_bases=4),   80,  0.01,  True,  False),
    ('HGT',       _make_hgt,                                                      80,  0.005, True,  False),
    ('DistMult',  lambda d: create_distmult(num_nodes, embedding_dim=64),        150, 0.01,  False, True),
    ('ComplEx',   lambda d: create_complex(num_nodes,  embedding_dim=64),        150, 0.01,  False, True),
    ('TransE',    lambda d: create_transe(num_nodes,   embedding_dim=64),        150, 0.01,  False, True),
]

# ── Drug-class heuristic baseline (no training) ───────────────────────────────
print("\n[DrugClass-Heuristic] Computing drug-class co-resistance baseline (no training)...")
_dc_result = compute_drug_class_heuristic()
print(f"  MRR={_dc_result['mrr']:.4f} | Hits@1={_dc_result['hits@1']:.4f} | "
      f"H@10={_dc_result['hits@10']:.4f} | MnRk={_dc_result['mean_rank']:.1f} | "
      f"p={_dc_result['p_value']:.3f}")

# ── Main loop ─────────────────────────────────────────────────────────────────
all_results = {'DrugClass-Heuristic': {
    'per_seed': [_dc_result],
    'aggregated': {k: {'mean': _dc_result.get(k, 0.), 'std': 0.0}
                   for k in ['auc', 'ap', 'mrr', 'hits@1', 'hits@3', 'hits@10', 'mean_rank']},
}}
# auc/ap not applicable for heuristic
all_results['DrugClass-Heuristic']['aggregated']['auc'] = {'mean': float('nan'), 'std': 0.0}
all_results['DrugClass-Heuristic']['aggregated']['ap']  = {'mean': float('nan'), 'std': 0.0}

for name, make_model, epochs, lr, use_typed, is_distmult in MODEL_CONFIGS:
    print(f"\n{'='*70}")
    print(f"  {name}  ({len(SEEDS)} seeds)")
    print(f"{'='*70}")

    seed_results = []
    for seed in SEEDS:
        t0 = time.time()
        r  = train_one_seed(make_model, epochs, lr, use_typed, is_distmult, seed)
        elapsed = time.time() - t0
        print(f"  seed={seed:4d} | AUC={r['auc']:.4f} | MRR={r['mrr']:.4f} | "
              f"H@1={r['hits@1']:.4f} | MnRk={r['mean_rank']:.1f} | t={elapsed:.1f}s")
        seed_results.append(r)

    keys = ['auc', 'ap', 'mrr', 'hits@1', 'hits@3', 'hits@10', 'mean_rank']
    agg  = {k: {'mean': float(np.mean([r[k] for r in seed_results])),
                'std':  float(np.std( [r[k] for r in seed_results]))} for k in keys}
    all_results[name] = {'per_seed': seed_results, 'aggregated': agg}

    print(f"\n  {name} SUMMARY:")
    print(f"  AUC:       {agg['auc']['mean']:.4f} +/- {agg['auc']['std']:.4f}")
    print(f"  MRR:       {agg['mrr']['mean']:.4f} +/- {agg['mrr']['std']:.4f}")
    print(f"  Hits@1:    {agg['hits@1']['mean']:.4f} +/- {agg['hits@1']['std']:.4f}")
    print(f"  Mean Rank: {agg['mean_rank']['mean']:.1f} +/- {agg['mean_rank']['std']:.1f}")


# ── Final table ───────────────────────────────────────────────────────────────
print(f"\n\n{'='*90}")
print("FINAL RESULTS  (mean +/- std, 5 seeds, hard negatives, rich features)")
print(f"{'='*90}")
print(f"{'Model':<22} {'AUC':>14} {'MRR':>14} {'Hits@1':>14} {'Hits@10':>14} {'Mean Rank':>16}")
print("-"*100)
for name in ['DrugClass-Heuristic', 'GCN', 'GraphSAGE', 'GAT', 'R-GCN', 'HGT', 'DistMult', 'ComplEx', 'TransE']:
    if name not in all_results: continue
    a = all_results[name]['aggregated']
    auc_str = f"{a['auc']['mean']:.4f}+/-{a['auc']['std']:.4f}" if not (a['auc']['mean'] != a['auc']['mean']) else "    n/a       "
    print(f"{name:<22} "
          f"{auc_str}  "
          f"{a['mrr']['mean']:.4f}+/-{a['mrr']['std']:.4f}  "
          f"{a['hits@1']['mean']:.4f}+/-{a['hits@1']['std']:.4f}  "
          f"{a['hits@10']['mean']:.4f}+/-{a['hits@10']['std']:.4f}  "
          f"{a['mean_rank']['mean']:.1f}+/-{a['mean_rank']['std']:.1f}")

print(f"\nRandom baseline MRR = {1/num_antibiotics:.4f}")

out_path = Path(__file__).parent.parent / "results" / "metrics" / "multiseed_results.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to: {out_path}")
