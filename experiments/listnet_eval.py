"""
R-GCN + ListNet Ranking Loss
=============================
Replaces BCE training with a sampled softmax (InfoNCE / ListNet-style) ranking loss.

For each positive pair (gene g, antibiotic a+), sample K hard negatives (g, a1-)...(g, aK-)
and minimize:

  L_rank = -s(g,a+) + logsumexp([s(g,a+), s(g,a1-), ..., s(g,aK-)])

This directly optimizes ranking quality (MRR) rather than binary classification (AUC).
Scores are raw dot products (no sigmoid) — sigmoid is monotone so ranking is identical,
but raw logits are needed for the softmax to be calibrated.

Also tests: ListNet + StructAlign(lambda=0.5), the combined best model.

Results saved to: results/metrics/listnet_results.json
Existing results are NOT modified.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict
from models.rgcn import create_rgcn
import time
import json

SEEDS    = [42, 123, 456, 789, 1234]
EPOCHS   = 80
PATIENCE = 4
PROJ_DIM = 64
K_NEG    = 5    # negatives per positive for ListNet

# ── Load graph ────────────────────────────────────────────────────────────────
print("Loading graph...")
graph_path = Path(__file__).parent.parent / "data" / "graphs" / "card_hetero_graph.pt"
graph_data = torch.load(graph_path, weights_only=False)

num_nodes     = graph_data['num_nodes']
node_features = graph_data['node_features']
gene_rich     = graph_data['gene_rich_features']
ab_rich       = graph_data['antibiotic_rich_features']

train_pos     = graph_data['train_pos_edges']
val_pos       = graph_data['val_pos_edges']
test_pos      = graph_data['test_pos_edges']

train_typed         = graph_data['train_typed_edge_indices']
edge_type_to_id     = graph_data['edge_type_to_id']
train_edges_by_type = {edge_type_to_id[k]: v for k, v in train_typed.items()}

node_type_map = graph_data['node_type_map']
node_to_idx   = graph_data['node_to_idx']

gene_idx_list = torch.tensor([i for i, t in node_type_map.items() if t == 0], dtype=torch.long)

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

all_ab_names   = list(graph_data['antibiotics'].keys())
all_ab_indices = torch.tensor(
    [node_to_idx[('antibiotic', n)] for n in all_ab_names], dtype=torch.long)
num_antibiotics = len(all_ab_indices)

pos_edge_set = set(zip(train_pos[0].tolist(), train_pos[1].tolist()))
pos_edge_set.update(zip(val_pos[0].tolist(), val_pos[1].tolist()))
pos_edge_set.update(zip(test_pos[0].tolist(), test_pos[1].tolist()))

gene_to_train_pos_abs = defaultdict(set)
for _i in range(train_pos.shape[1]):
    gene_to_train_pos_abs[train_pos[0, _i].item()].add(train_pos[1, _i].item())

_num_rel = max(edge_type_to_id.values()) + 1

# ── Pre-compute Tanimoto matrix (for StructAlign variant) ─────────────────────
print("Pre-computing Tanimoto matrix...")
with torch.no_grad():
    fp    = ab_rich.float()
    dot   = fp @ fp.t()
    norms = fp.sum(dim=1, keepdim=True)
    denom = norms + norms.t() - dot
    tanimoto_matrix = (dot / denom.clamp(min=1e-8)).clamp(0., 1.)

print(f"Nodes: {num_nodes}  |  Train: {train_pos.shape[1]}  |  "
      f"Val: {val_pos.shape[1]}  |  Test: {test_pos.shape[1]}")
print(f"Gene nodes: {len(gene_idx_list)}  |  Train ABs: {len(train_ab_idx)}  |  "
      f"Test ABs: {len(test_ab_idx)}  |  K_NEG: {K_NEG}")


# ── Feature projector ─────────────────────────────────────────────────────────
class FeatureProjector(nn.Module):
    def __init__(self, proj_dim=64):
        super().__init__()
        self.gene_proj  = nn.Linear(gene_rich.shape[1], proj_dim)
        self.ab_proj    = nn.Linear(ab_rich.shape[1],   proj_dim)
        self.other_proj = nn.Linear(node_features.shape[1], proj_dim)
        self.register_buffer('gene_mask', torch.tensor(
            [node_type_map[i] == 0 for i in range(num_nodes)]))
        self.register_buffer('ab_mask',   torch.tensor(
            [node_type_map[i] == 1 for i in range(num_nodes)]))

    def forward(self):
        other_mask = ~(self.gene_mask | self.ab_mask)
        feats = torch.zeros(num_nodes, self.gene_proj.out_features,
                            device=gene_rich.device)
        feats[self.gene_mask] = self.gene_proj(gene_rich)
        feats[self.ab_mask]   = self.ab_proj(ab_rich)
        feats[other_mask]     = self.other_proj(node_features[other_mask])
        return feats


# ── Structural alignment loss ─────────────────────────────────────────────────
def struct_align_loss(z, ab_indices, tanimoto_mat):
    ab_embs = z[ab_indices]
    ab_norm = F.normalize(ab_embs, p=2, dim=-1)
    cos_sim  = ab_norm @ ab_norm.t()
    return F.mse_loss(cos_sim, tanimoto_mat.to(z.device))


# ── ListNet negative sampling ─────────────────────────────────────────────────
def sample_neg_per_pos(pos_edges, K, seed):
    """
    Pre-sample K hard negatives per positive pair.
    Returns tensor of shape [N, K] with antibiotic indices.
    """
    rng = torch.Generator(); rng.manual_seed(seed)
    N   = pos_edges.shape[1]
    neg_abs = torch.zeros(N, K, dtype=torch.long)
    for i in range(N):
        g_idx = pos_edges[0, i].item()
        negs  = []
        attempts = 0
        while len(negs) < K and attempts < K * 30:
            ai = train_ab_idx[torch.randint(len(train_ab_idx), (1,), generator=rng)].item()
            if (g_idx, ai) not in pos_edge_set:
                negs.append(ai)
            attempts += 1
        while len(negs) < K:
            negs.append(train_ab_idx[0].item())
        neg_abs[i] = torch.tensor(negs[:K])
    return neg_abs   # [N, K]


def sample_neg_hard(n, seed, ab_pool):
    """Hard negatives for AUC validation (same as multiseed_eval.py)."""
    rng  = torch.Generator(); rng.manual_seed(seed)
    negs = []
    attempts = 0
    while len(negs) < n and attempts < n * 20:
        gi = gene_idx_list[torch.randint(len(gene_idx_list), (1,), generator=rng).item()].item()
        ai = ab_pool[torch.randint(len(ab_pool), (1,), generator=rng).item()].item()
        if (gi, ai) not in pos_edge_set:
            negs.append([gi, ai])
        attempts += 1
    while len(negs) < n:
        negs.append(negs[-1])
    return torch.tensor(negs[:n], dtype=torch.long).t()


# ── Ranking eval ──────────────────────────────────────────────────────────────
def compute_auc_ap(pos_scores, neg_scores):
    s = np.concatenate([pos_scores, neg_scores])
    l = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    return roc_auc_score(l, s), average_precision_score(l, s)


@torch.no_grad()
def compute_ranking(model, projector):
    model.eval(); projector.eval()
    nf = projector()
    z  = model(nf, train_edges_by_type)
    ab_embs    = z[all_ab_indices]
    ab_idx2pos = {idx.item(): pos for pos, idx in enumerate(all_ab_indices)}

    ranks = []
    for i in range(test_pos.shape[1]):
        g_idx, ab_idx = test_pos[0, i].item(), test_pos[1, i].item()
        if ab_idx not in ab_idx2pos: continue
        scores     = (z[g_idx].unsqueeze(0) * ab_embs).sum(-1)   # raw dot product
        true_score = scores[ab_idx2pos[ab_idx]].item()

        train_pos_for_gene = gene_to_train_pos_abs.get(g_idx, set())
        valid_mask = torch.ones(num_antibiotics, dtype=torch.bool)
        for tp_ab in train_pos_for_gene:
            if tp_ab in ab_idx2pos:
                valid_mask[ab_idx2pos[tp_ab]] = False
        valid_mask[ab_idx2pos[ab_idx]] = True

        rank = int((scores[valid_mask] > true_score).sum()) + 1
        ranks.append(rank)

    ranks = np.array(ranks, dtype=float)
    rand  = np.random.default_rng(42).uniform(1, num_antibiotics + 1, len(ranks))
    _, p  = stats.wilcoxon(1.0 / ranks, 1.0 / rand, alternative='greater')
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


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_seed(use_struct_align, seed):
    torch.manual_seed(seed); np.random.seed(seed)

    projector = FeatureProjector(PROJ_DIM)
    model     = create_rgcn(PROJ_DIM, _num_rel, 64, 32, 2, num_bases=4)
    params    = list(model.parameters()) + list(projector.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01)

    # Pre-sample negatives
    # ListNet: K negatives per positive pair, sampled once
    print(f"    Sampling {K_NEG} negatives per positive ({train_pos.shape[1]} pairs)...", end=' ', flush=True)
    train_neg_per_pos = sample_neg_per_pos(train_pos, K_NEG, seed=seed)  # [N, K]
    print("done")

    # AUC validation negatives (same as multiseed_eval.py)
    val_neg  = sample_neg_hard(val_pos.shape[1],  seed+1, val_ab_idx)
    test_neg = sample_neg_hard(test_pos.shape[1], seed+2, test_ab_idx)

    best_val_auc = 0
    patience_ctr = 0

    for epoch in range(1, EPOCHS + 1):
        model.train(); projector.train()
        optimizer.zero_grad()

        nf = projector()
        z  = model(nf, train_edges_by_type)

        # ── ListNet ranking loss ───────────────────────────────────────────────
        N = train_pos.shape[1]
        g_embs  = z[train_pos[0]]                                  # [N, d]
        ap_embs = z[train_pos[1]]                                  # [N, d]
        neg_embs = z[train_neg_per_pos.view(-1)].view(N, K_NEG, -1) # [N, K, d]

        s_pos = (g_embs * ap_embs).sum(-1, keepdim=True)           # [N, 1]
        s_neg = (g_embs.unsqueeze(1) * neg_embs).sum(-1)           # [N, K]

        all_scores = torch.cat([s_pos, s_neg], dim=-1)             # [N, K+1]
        # InfoNCE / sampled softmax: minimize -s_pos + logsumexp(all)
        link_loss = (-all_scores[:, 0] + torch.logsumexp(all_scores, dim=-1)).mean()

        if use_struct_align:
            align_loss = struct_align_loss(z, all_ab_indices, tanimoto_matrix)
            loss = link_loss + 0.5 * align_loss
        else:
            loss = link_loss

        loss.backward()
        optimizer.step()

        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval(); projector.eval()
            with torch.no_grad():
                nf2 = projector()
                z2  = model(nf2, train_edges_by_type)
                # Val AUC: use sigmoid for binary scoring
                val_pos_s = torch.sigmoid((z2[val_pos[0]] * z2[val_pos[1]]).sum(-1))
                val_neg_s = torch.sigmoid((z2[val_neg[0]] * z2[val_neg[1]]).sum(-1))
                val_auc, _ = compute_auc_ap(val_pos_s.cpu().numpy(), val_neg_s.cpu().numpy())
            if val_auc > best_val_auc:
                best_val_auc = val_auc; patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
            model.train(); projector.train()

    model.eval(); projector.eval()
    with torch.no_grad():
        nf = projector()
        z  = model(nf, train_edges_by_type)
        # AUC: sigmoid for binary eval
        pos_s = torch.sigmoid((z[test_pos[0]] * z[test_pos[1]]).sum(-1))
        neg_s = torch.sigmoid((z[test_neg[0]] * z[test_neg[1]]).sum(-1))
        auc, ap = compute_auc_ap(pos_s.cpu().numpy(), neg_s.cpu().numpy())
        ranking  = compute_ranking(model, projector)

    return {'auc': auc, 'ap': ap, **ranking}


# ── Variants to evaluate ──────────────────────────────────────────────────────
VARIANTS = [
    ('R-GCN+ListNet',           False),   # ListNet only
    ('R-GCN+ListNet+SA(0.5)',   True),    # ListNet + StructAlign
]

all_results = {}

for name, use_sa in VARIANTS:
    print(f"\n{'='*70}")
    print(f"  {name}  ({len(SEEDS)} seeds)")
    print(f"{'='*70}")

    seed_results = []
    for seed in SEEDS:
        t0 = time.time()
        r  = train_one_seed(use_sa, seed)
        elapsed = time.time() - t0
        print(f"  seed={seed:4d} | AUC={r['auc']:.4f} | MRR={r['mrr']:.4f} | "
              f"H@1={r['hits@1']:.4f} | H@10={r['hits@10']:.4f} | "
              f"MnRk={r['mean_rank']:.1f} | t={elapsed:.1f}s")
        seed_results.append(r)

    keys = ['auc', 'ap', 'mrr', 'hits@1', 'hits@3', 'hits@10', 'mean_rank']
    agg  = {k: {'mean': float(np.mean([r[k] for r in seed_results])),
                 'std':  float(np.std( [r[k] for r in seed_results]))} for k in keys}
    all_results[name] = {'per_seed': seed_results, 'aggregated': agg}

    print(f"\n  {name} SUMMARY:")
    print(f"  AUC:       {agg['auc']['mean']:.4f} +/- {agg['auc']['std']:.4f}")
    print(f"  MRR:       {agg['mrr']['mean']:.4f} +/- {agg['mrr']['std']:.4f}")
    print(f"  Hits@1:    {agg['hits@1']['mean']:.4f} +/- {agg['hits@1']['std']:.4f}")
    print(f"  Hits@10:   {agg['hits@10']['mean']:.4f} +/- {agg['hits@10']['std']:.4f}")
    print(f"  Mean Rank: {agg['mean_rank']['mean']:.1f} +/- {agg['mean_rank']['std']:.1f}")


# ── Final table ───────────────────────────────────────────────────────────────
print(f"\n\n{'='*90}")
print("LISTNET RANKING LOSS RESULTS  (R-GCN, 5 seeds)")
print("Reference from struct_align_eval.py:")
print("  R-GCN BCE (baseline):      MRR 0.197 +/- 0.050")
print("  R-GCN+SA(0.5) BCE:         MRR 0.286 +/- 0.057")
print(f"{'='*90}")
print(f"{'Model':<32} {'MRR':>12} {'Hits@1':>12} {'Hits@10':>12} {'Mean Rank':>14}")
print("-"*80)
for name, res in all_results.items():
    a = res['aggregated']
    print(f"{name:<32}  "
          f"{a['mrr']['mean']:.4f}+/-{a['mrr']['std']:.4f}  "
          f"{a['hits@1']['mean']:.4f}+/-{a['hits@1']['std']:.4f}  "
          f"{a['hits@10']['mean']:.4f}+/-{a['hits@10']['std']:.4f}  "
          f"{a['mean_rank']['mean']:.1f}+/-{a['mean_rank']['std']:.1f}")

print(f"\nRandom baseline MRR = {1/num_antibiotics:.4f}")

out_path = Path(__file__).parent.parent / "results" / "metrics" / "listnet_results.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to: {out_path}")
