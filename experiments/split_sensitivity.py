"""
Split sensitivity analysis: run R-GCN and DrugClass-Heuristic on 3 additional
random antibiotic splits to verify results are not specific to the original split.

Each split uses the same 161/23/47 train/val/test ratio.
5 model-init seeds per split.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from models.rgcn import create_rgcn
from collections import defaultdict
import json, time

SEEDS    = [42, 123, 456, 789, 1234]
EPOCHS   = 80
PATIENCE = 4
PROJ_DIM = 64

# ── Load full graph (unsplit) ──────────────────────────────────────────────────
print("Loading graph...")
graph_path = Path(__file__).parent.parent / "data" / "graphs" / "card_hetero_graph.pt"
graph_data = torch.load(graph_path, weights_only=False)

num_nodes       = graph_data['num_nodes']
node_features   = graph_data['node_features']
gene_rich       = graph_data['gene_rich_features']
ab_rich         = graph_data['antibiotic_rich_features']
node_type_map   = graph_data['node_type_map']
num_node_types  = max(node_type_map.values()) + 1
node_to_idx     = graph_data['node_to_idx']
edge_type_to_id = graph_data['edge_type_to_id']
all_typed       = graph_data['typed_edge_indices']   # FULL unsplit edge set

_num_rel = max(edge_type_to_id.values()) + 1

# All antibiotics and their indices
all_ab_names   = list(graph_data['antibiotics'].keys())
all_ab_indices = torch.tensor(
    [node_to_idx[('antibiotic', n)] for n in all_ab_names], dtype=torch.long)
num_antibiotics = len(all_ab_names)

# All gene indices
gene_idx_list = torch.tensor([i for i, t in node_type_map.items() if t == 0], dtype=torch.long)

# All resistance edges (gene→antibiotic) from full graph
all_resist_edges = all_typed['gene_to_antibiotic']   # [2, total_resist]
print(f"Total resistance edges: {all_resist_edges.shape[1]}")
print(f"Antibiotics: {num_antibiotics}  |  Genes: {len(gene_idx_list)}")

# Non-resistance structural edges (same for all splits)
STRUCT_ETYPES = ['gene_to_mechanism', 'gene_to_family', 'antibiotic_to_class',
                 'meg_gene_to_class', 'antibiotic_targets_protein',
                 'gene_to_go_term', 'gene_to_kegg_pathway', 'gene_interacts_gene']

# ── Build split ───────────────────────────────────────────────────────────────
def build_split(split_seed, n_train=161, n_val=23, n_test=47):
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(num_antibiotics)
    train_names = [all_ab_names[i] for i in perm[:n_train]]
    val_names   = [all_ab_names[i] for i in perm[n_train:n_train+n_val]]
    test_names  = [all_ab_names[i] for i in perm[n_train+n_val:n_train+n_val+n_test]]

    train_ab_set = set(node_to_idx[('antibiotic', n)] for n in train_names if ('antibiotic', n) in node_to_idx)
    val_ab_set   = set(node_to_idx[('antibiotic', n)] for n in val_names   if ('antibiotic', n) in node_to_idx)
    test_ab_set  = set(node_to_idx[('antibiotic', n)] for n in test_names  if ('antibiotic', n) in node_to_idx)

    # Split resistance edges by antibiotic membership
    src = all_resist_edges[0].tolist()
    dst = all_resist_edges[1].tolist()

    train_edges, val_edges, test_edges = [], [], []
    for g, a in zip(src, dst):
        if a in train_ab_set:   train_edges.append([g, a])
        elif a in val_ab_set:   val_edges.append([g, a])
        elif a in test_ab_set:  test_edges.append([g, a])

    def to_tensor(edges):
        return torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros(2, 0, dtype=torch.long)

    train_pos = to_tensor(train_edges)
    val_pos   = to_tensor(val_edges)
    test_pos  = to_tensor(test_edges)

    # Build train_typed: structural edges + resistance edges for train ABs only
    train_typed = {}
    for etype in STRUCT_ETYPES:
        if etype in all_typed and all_typed[etype].numel() > 0:
            train_typed[edge_type_to_id[etype]] = all_typed[etype]
    train_typed[edge_type_to_id['gene_to_antibiotic']] = train_pos  # only train resist edges

    # Flat edge index for non-typed models
    train_edge_index = torch.cat([v for v in train_typed.values() if v.numel() > 0], dim=1)

    # AB index tensors
    train_ab_idx = torch.tensor(sorted(train_ab_set), dtype=torch.long)
    val_ab_idx   = torch.tensor(sorted(val_ab_set),   dtype=torch.long)
    test_ab_idx  = torch.tensor(sorted(test_ab_set),  dtype=torch.long)

    # Positive edge set for negative filtering
    pos_edge_set = set(zip(train_pos[0].tolist(), train_pos[1].tolist()))
    pos_edge_set.update(zip(val_pos[0].tolist(),  val_pos[1].tolist()))
    pos_edge_set.update(zip(test_pos[0].tolist(), test_pos[1].tolist()))

    # Per-gene training positives for filtered ranking
    gene_to_train_pos = defaultdict(set)
    for i in range(train_pos.shape[1]):
        gene_to_train_pos[train_pos[0, i].item()].add(train_pos[1, i].item())

    # Drug-class structure for heuristic
    _train_ab_set_idx = set(train_ab_idx.tolist())
    ab_to_dc = defaultdict(set)
    dc_to_train_abs = defaultdict(set)
    if 'antibiotic_to_class' in all_typed and all_typed['antibiotic_to_class'].numel() > 0:
        atc = all_typed['antibiotic_to_class']
        for i in range(atc.shape[1]):
            ab_i, dc_i = atc[0, i].item(), atc[1, i].item()
            ab_to_dc[ab_i].add(dc_i)
            if ab_i in _train_ab_set_idx:
                dc_to_train_abs[dc_i].add(ab_i)

    return {
        'train_pos': train_pos, 'val_pos': val_pos, 'test_pos': test_pos,
        'train_typed': train_typed, 'train_edge_index': train_edge_index,
        'train_ab_idx': train_ab_idx, 'val_ab_idx': val_ab_idx, 'test_ab_idx': test_ab_idx,
        'pos_edge_set': pos_edge_set, 'gene_to_train_pos': gene_to_train_pos,
        'ab_to_dc': ab_to_dc, 'dc_to_train_abs': dc_to_train_abs,
        'n_train': len(train_edges), 'n_val': len(val_edges), 'n_test': len(test_edges),
    }

# ── Feature projector ─────────────────────────────────────────────────────────
class FeatureProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.gene_proj  = nn.Linear(gene_rich.shape[1], PROJ_DIM)
        self.ab_proj    = nn.Linear(ab_rich.shape[1],   PROJ_DIM)
        self.other_proj = nn.Linear(node_features.shape[1], PROJ_DIM)
        self.register_buffer('gene_mask', torch.tensor([node_type_map[i] == 0 for i in range(num_nodes)]))
        self.register_buffer('ab_mask',   torch.tensor([node_type_map[i] == 1 for i in range(num_nodes)]))

    def forward(self):
        other_mask = ~(self.gene_mask | self.ab_mask)
        feats = torch.zeros(num_nodes, PROJ_DIM, device=gene_rich.device)
        feats[self.gene_mask]  = self.gene_proj(gene_rich)
        feats[self.ab_mask]    = self.ab_proj(ab_rich)
        feats[other_mask]      = self.other_proj(node_features[other_mask])
        return feats

# ── Hard negative sampling ────────────────────────────────────────────────────
def sample_neg_hard(n, seed, ab_pool, pos_edge_set):
    rng = torch.Generator(); rng.manual_seed(seed)
    negs, attempts = [], 0
    while len(negs) < n and attempts < n * 20:
        gi = gene_idx_list[torch.randint(len(gene_idx_list), (1,), generator=rng).item()].item()
        ai = ab_pool[torch.randint(len(ab_pool), (1,), generator=rng).item()].item()
        if (gi, ai) not in pos_edge_set:
            negs.append([gi, ai])
        attempts += 1
    while len(negs) < n:
        negs.append(negs[-1])
    return torch.tensor(negs[:n], dtype=torch.long).t()

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_auc_ap(pos_scores, neg_scores):
    s = np.concatenate([pos_scores, neg_scores])
    l = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    return roc_auc_score(l, s), average_precision_score(l, s)

def compute_ranking(z, test_pos, gene_to_train_pos):
    ab_idx2pos = {idx.item(): pos for pos, idx in enumerate(all_ab_indices)}
    ab_embs    = z[all_ab_indices]
    ranks = []
    for i in range(test_pos.shape[1]):
        g_idx, ab_idx = test_pos[0, i].item(), test_pos[1, i].item()
        if ab_idx not in ab_idx2pos: continue
        scores     = torch.sigmoid((z[g_idx].unsqueeze(0) * ab_embs).sum(-1))
        true_score = scores[ab_idx2pos[ab_idx]].item()
        train_positives = gene_to_train_pos.get(g_idx, set())
        valid_mask = torch.ones(num_antibiotics, dtype=torch.bool)
        for tp in train_positives:
            if tp in ab_idx2pos:
                valid_mask[ab_idx2pos[tp]] = False
        valid_mask[ab_idx2pos[ab_idx]] = True
        ranks.append(int((scores[valid_mask] > true_score).sum()) + 1)
    ranks = np.array(ranks, dtype=float)
    rand  = np.random.default_rng(42).uniform(1, num_antibiotics + 1, len(ranks))
    _, p  = stats.wilcoxon(1.0 / ranks, 1.0 / rand, alternative='greater')
    return {
        'mrr':        float(np.mean(1.0 / ranks)),
        'hits@1':     float(np.mean(ranks <= 1)),
        'hits@10':    float(np.mean(ranks <= 10)),
        'mean_rank':  float(np.mean(ranks)),
        'p_value':    float(p),
        'effect_size': 1.0 - 2.0 * float(np.mean(ranks)) / num_antibiotics,
    }

# ── Train R-GCN on one seed/split ────────────────────────────────────────────
def train_rgcn(split, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    projector = FeatureProjector()
    model     = create_rgcn(PROJ_DIM, _num_rel, 64, 32, 2, num_bases=4)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(projector.parameters()), lr=0.01)

    train_neg = sample_neg_hard(min(2000, split['train_pos'].shape[1]), seed,   split['train_ab_idx'], split['pos_edge_set'])
    val_neg   = sample_neg_hard(split['val_pos'].shape[1],              seed+1, split['val_ab_idx'],   split['pos_edge_set'])
    test_neg  = sample_neg_hard(split['test_pos'].shape[1],             seed+2, split['test_ab_idx'],  split['pos_edge_set'])

    best_val_auc, patience_ctr = 0, 0

    for epoch in range(1, EPOCHS + 1):
        model.train(); projector.train(); optimizer.zero_grad()
        nf = projector()
        z  = model(nf, split['train_typed'])
        pos_s = model.decode(z, split['train_pos'][:, :train_neg.shape[1]])
        neg_s = model.decode(z, train_neg)
        loss  = (-torch.log(pos_s + 1e-15).mean() - torch.log(1 - neg_s + 1e-15).mean())
        loss.backward(); optimizer.step()

        if epoch % 5 == 0:
            model.eval(); projector.eval()
            with torch.no_grad():
                nf2 = projector()
                z2  = model(nf2, split['train_typed'])
                val_auc, _ = compute_auc_ap(model.decode(z2, split['val_pos']).numpy(),
                                            model.decode(z2, val_neg).numpy())
            if val_auc > best_val_auc: best_val_auc, patience_ctr = val_auc, 0
            else: patience_ctr += 1
            if patience_ctr >= PATIENCE: break
            model.train(); projector.train()

    model.eval(); projector.eval()
    with torch.no_grad():
        nf = projector()
        z  = model(nf, split['train_typed'])
        auc, ap = compute_auc_ap(model.decode(z, split['test_pos']).numpy(),
                                  model.decode(z, test_neg).numpy())
        rk = compute_ranking(z, split['test_pos'], split['gene_to_train_pos'])
    return {'auc': auc, 'ap': ap, **rk}

# ── Drug-class heuristic ─────────────────────────────────────────────────────
def drug_class_heuristic(split):
    ab_idx2pos_h = {idx.item(): pos for pos, idx in enumerate(all_ab_indices)}
    all_ab_list  = all_ab_indices.tolist()
    ranks = []
    for i in range(split['test_pos'].shape[1]):
        g_idx, ab_idx = split['test_pos'][0, i].item(), split['test_pos'][1, i].item()
        if ab_idx not in ab_idx2pos_h: continue
        gene_train_pos = split['gene_to_train_pos'].get(g_idx, set())
        scores = np.zeros(num_antibiotics)
        for j, cand_ab in enumerate(all_ab_list):
            dc_set = split['ab_to_dc'].get(cand_ab, set())
            if not dc_set: continue
            siblings = set()
            for dc in dc_set:
                siblings |= split['dc_to_train_abs'].get(dc, set())
            siblings.discard(cand_ab)
            if not siblings: continue
            scores[j] = len(gene_train_pos & siblings) / len(siblings)
        true_score = scores[ab_idx2pos_h[ab_idx]]
        train_pos_abs = split['gene_to_train_pos'].get(g_idx, set())
        valid_mask = np.ones(num_antibiotics, dtype=bool)
        for tp in train_pos_abs:
            if tp in ab_idx2pos_h: valid_mask[ab_idx2pos_h[tp]] = False
        valid_mask[ab_idx2pos_h[ab_idx]] = True
        ranks.append(int(np.sum(scores[valid_mask] > true_score)) + 1)
    ranks = np.array(ranks, dtype=float)
    rand  = np.random.default_rng(42).uniform(1, num_antibiotics + 1, len(ranks))
    _, p  = stats.wilcoxon(1.0 / ranks, 1.0 / rand, alternative='greater')
    return {
        'mrr':        float(np.mean(1.0 / ranks)),
        'hits@1':     float(np.mean(ranks <= 1)),
        'hits@10':    float(np.mean(ranks <= 10)),
        'mean_rank':  float(np.mean(ranks)),
        'p_value':    float(p),
        'effect_size': 1.0 - 2.0 * float(np.mean(ranks)) / num_antibiotics,
    }

# ── Main: 3 additional splits (seeds 999, 2025, 314159) + original ────────────
# Original split seed = the one baked in the graph (not re-run here, use stored results)
SPLIT_SEEDS = [999, 2025, 314159]

all_results = {}

for split_seed in SPLIT_SEEDS:
    print(f"\n{'='*70}")
    print(f"  SPLIT SEED {split_seed}")
    print(f"{'='*70}")
    split = build_split(split_seed)
    print(f"  Train: {split['n_train']}  Val: {split['n_val']}  Test: {split['n_test']}")

    # DrugClass Heuristic
    dc = drug_class_heuristic(split)
    print(f"  DrugClass-Heuristic  MRR={dc['mrr']:.4f}  Hits@10={dc['hits@10']:.4f}  MnRk={dc['mean_rank']:.1f}")

    # R-GCN over 5 seeds
    seed_results = []
    for seed in SEEDS:
        t0 = time.time()
        r  = train_rgcn(split, seed)
        print(f"  R-GCN seed={seed:4d} | AUC={r['auc']:.4f} | MRR={r['mrr']:.4f} | "
              f"H@10={r['hits@10']:.4f} | MnRk={r['mean_rank']:.1f} | t={time.time()-t0:.1f}s")
        seed_results.append(r)

    keys = ['mrr', 'hits@1', 'hits@10', 'mean_rank', 'auc']
    agg  = {k: {'mean': float(np.mean([s[k] for s in seed_results])),
                'std':  float(np.std( [s[k] for s in seed_results]))} for k in keys}
    all_results[split_seed] = {'rgcn': agg, 'heuristic': dc,
                                'n_test': split['n_test']}
    print(f"  R-GCN MEAN: MRR={agg['mrr']['mean']:.4f}±{agg['mrr']['std']:.4f}  "
          f"H@10={agg['hits@10']['mean']:.4f}±{agg['hits@10']['std']:.4f}  "
          f"MnRk={agg['mean_rank']['mean']:.1f}±{agg['mean_rank']['std']:.1f}")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n\n{'='*80}")
print("SPLIT SENSITIVITY SUMMARY")
print(f"{'='*80}")
print(f"{'Split':>12}  {'Model':<22}  {'MRR':>8}  {'Hits@10':>8}  {'Mean Rank':>10}  {'n_test':>6}")
print("-"*80)

# Include original (from stored results)
orig_rgcn = {'mrr': 0.2253, 'hits@10': 0.3516, 'mean_rank': 52.4}
orig_dc   = {'mrr': 0.2797, 'hits@10': 0.3196, 'mean_rank': 70.4}
print(f"{'original':>12}  {'R-GCN':<22}  {orig_rgcn['mrr']:>8.4f}  {orig_rgcn['hits@10']:>8.4f}  {orig_rgcn['mean_rank']:>10.1f}  {'1042':>6}")
print(f"{'original':>12}  {'DrugClass-Heuristic':<22}  {orig_dc['mrr']:>8.4f}  {orig_dc['hits@10']:>8.4f}  {orig_dc['mean_rank']:>10.1f}  {'1042':>6}")

for split_seed in SPLIT_SEEDS:
    res = all_results[split_seed]
    a   = res['rgcn']
    dc  = res['heuristic']
    n   = res['n_test']
    print(f"{split_seed:>12}  {'R-GCN':<22}  {a['mrr']['mean']:>8.4f}  {a['hits@10']['mean']:>8.4f}  {a['mean_rank']['mean']:>10.1f}  {n:>6}")
    print(f"{split_seed:>12}  {'DrugClass-Heuristic':<22}  {dc['mrr']:>8.4f}  {dc['hits@10']:>8.4f}  {dc['mean_rank']:>10.1f}  {n:>6}")

out = Path(__file__).parent.parent / "results" / "metrics" / "split_sensitivity.json"
with open(out, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to: {out}")
