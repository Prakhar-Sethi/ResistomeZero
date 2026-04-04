"""
Ablation: Original graph vs GO+KEGG enriched graph.

Tests whether adding Gene Ontology and KEGG pathway edges actually helps.
We run all 6 models under both conditions with the same 5 seeds.

'Original' = only CARD edges (gene_to_antibiotic, gene_to_mechanism,
              gene_to_family, antibiotic_to_class, meg_gene_to_class,
              antibiotic_targets_protein)
'Enriched' = Original + gene_to_go_term + gene_to_kegg_pathway
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from models.gcn import create_gcn
from models.rgcn import create_rgcn
from models.graphsage import create_graphsage
from models.gat import create_gat
from models.hgt import create_hgt
from models.distmult import create_distmult
from models.complex import create_complex
from models.transe import create_transe
from collections import defaultdict
import time, json

SEEDS   = [42, 123, 456, 789, 1234]
EPOCHS  = 80
PATIENCE = 4

ORIGINAL_EDGE_TYPES = {
    'gene_to_antibiotic', 'gene_to_mechanism', 'gene_to_family',
    'antibiotic_to_class', 'meg_gene_to_class', 'antibiotic_targets_protein'
}
ENRICHMENT_EDGE_TYPES = {'gene_to_go_term', 'gene_to_kegg_pathway'}

# ── Load graph ────────────────────────────────────────────────────────────────
print("Loading graph...")
graph_path = Path(__file__).parent.parent / "data" / "graphs" / "card_hetero_graph.pt"
graph_data = torch.load(graph_path, weights_only=False)

num_nodes      = graph_data['num_nodes']
node_features  = graph_data['node_features']
gene_rich      = graph_data['gene_rich_features']
ab_rich        = graph_data['antibiotic_rich_features']
train_pos      = graph_data['train_pos_edges']
val_pos        = graph_data['val_pos_edges']
test_pos       = graph_data['test_pos_edges']
node_type_map  = graph_data['node_type_map']
num_node_types = max(node_type_map.values()) + 1
edge_type_to_id = graph_data['edge_type_to_id']
train_typed    = graph_data['train_typed_edge_indices']

node_to_idx    = graph_data['node_to_idx']
all_ab_names   = list(graph_data['antibiotics'].keys())
all_ab_indices = torch.tensor(
    [node_to_idx[('antibiotic', n)] for n in all_ab_names], dtype=torch.long
)
num_antibiotics = len(all_ab_indices)

PROJ_DIM = 64
_num_rel = max(edge_type_to_id.values()) + 1

# Node index sets for hard negative sampling
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

# All positive edges (for negative filtering)
pos_edge_set = set(zip(train_pos[0].tolist(), train_pos[1].tolist()))
pos_edge_set.update(zip(val_pos[0].tolist(),  val_pos[1].tolist()))
pos_edge_set.update(zip(test_pos[0].tolist(), test_pos[1].tolist()))

# Per-gene training positives for filtered ranking
gene_to_train_pos_abs = defaultdict(set)
for _i in range(train_pos.shape[1]):
    gene_to_train_pos_abs[train_pos[0, _i].item()].add(train_pos[1, _i].item())

print(f"Nodes: {num_nodes}  |  Train: {train_pos.shape[1]}  |  Test: {test_pos.shape[1]}")
print(f"Total edge types: {len(edge_type_to_id)}")

# ── Build two edge sets ───────────────────────────────────────────────────────
def build_edge_sets(include_enrichment):
    edges_by_type = {}
    edge_index_list = []
    for etype, edges in train_typed.items():
        if etype in ORIGINAL_EDGE_TYPES or (include_enrichment and etype in ENRICHMENT_EDGE_TYPES):
            rid = edge_type_to_id[etype]
            edges_by_type[rid] = edges
            edge_index_list.append(edges)
    edge_index = torch.cat(edge_index_list, dim=1)
    return edges_by_type, edge_index

orig_edges_by_type, orig_edge_index = build_edge_sets(include_enrichment=False)
enri_edges_by_type, enri_edge_index = build_edge_sets(include_enrichment=True)

print(f"Original edges: {orig_edge_index.shape[1]}")
print(f"Enriched edges: {enri_edge_index.shape[1]}")

# ── Feature projector (matches multiseed_eval protocol) ───────────────────────
class FeatureProjector(nn.Module):
    def __init__(self, proj_dim=PROJ_DIM):
        super().__init__()
        self.gene_proj  = nn.Linear(gene_rich.shape[1], proj_dim)
        self.ab_proj    = nn.Linear(ab_rich.shape[1],   proj_dim)
        self.other_proj = nn.Linear(node_features.shape[1], proj_dim)
        self.register_buffer('gene_mask',  torch.tensor(
            [node_type_map[i] == 0 for i in range(num_nodes)]))
        self.register_buffer('ab_mask',    torch.tensor(
            [node_type_map[i] == 1 for i in range(num_nodes)]))

    def forward(self):
        other_mask = ~(self.gene_mask | self.ab_mask)
        feats = torch.zeros(num_nodes, self.gene_proj.out_features, device=gene_rich.device)
        feats[self.gene_mask]  = self.gene_proj(gene_rich)
        feats[self.ab_mask]    = self.ab_proj(ab_rich)
        feats[other_mask]      = self.other_proj(node_features[other_mask])
        return feats

# ── Hard negative sampling ─────────────────────────────────────────────────────
def sample_neg_hard(n, seed, ab_pool):
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

def compute_auc_ap(pos, neg):
    s = np.concatenate([pos, neg])
    l = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return roc_auc_score(l, s), average_precision_score(l, s)

@torch.no_grad()
def compute_ranking(model, projector, get_z):
    """Filtered ranking: excludes known training positives per gene from ranking pool."""
    model.eval(); projector.eval()
    z = get_z(model, projector())
    ab_embs    = z[all_ab_indices]
    ab_idx2pos = {idx.item(): pos for pos, idx in enumerate(all_ab_indices)}
    ranks = []
    for i in range(test_pos.shape[1]):
        g_idx, ab_idx = test_pos[0, i].item(), test_pos[1, i].item()
        if ab_idx not in ab_idx2pos: continue
        if hasattr(model, 'score_all_tails'):
            scores = torch.sigmoid(model.score_all_tails(g_idx, all_ab_indices))
        else:
            scores = torch.sigmoid((z[g_idx].unsqueeze(0) * ab_embs).sum(-1))
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
    _, p  = stats.wilcoxon(1 / ranks, 1 / rand, alternative='greater')
    return {
        'mrr':       float(np.mean(1 / ranks)),
        'hits@1':    float(np.mean(ranks <= 1)),
        'hits@10':   float(np.mean(ranks <= 10)),
        'mean_rank': float(np.mean(ranks)),
        'p_value':   float(p),
    }

def make_get_z(use_typed, is_distmult, edges_by_type, edge_index):
    def get_z(model, nf):
        if is_distmult: return model.get_all_entity_embeddings()
        elif use_typed:  return model(nf, edges_by_type)
        else:            return model(nf, edge_index)
    return get_z

def run_seed(make_model, epochs, lr, use_typed, is_distmult,
             seed, edges_by_type, edge_index):
    torch.manual_seed(seed); np.random.seed(seed)
    projector = FeatureProjector(PROJ_DIM)
    model     = make_model()
    get_z     = make_get_z(use_typed, is_distmult, edges_by_type, edge_index)

    params = list(model.parameters())
    if not is_distmult:
        params += list(projector.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # Hard negatives: val uses val_ab_idx (zero-shot aligned)
    train_neg = sample_neg_hard(min(2000, train_pos.shape[1]), seed,   train_ab_idx)
    val_neg   = sample_neg_hard(val_pos.shape[1],              seed+1, val_ab_idx)
    test_neg  = sample_neg_hard(test_pos.shape[1],             seed+2, test_ab_idx)

    best_val, pat = 0, 0
    for epoch in range(1, epochs + 1):
        model.train(); projector.train(); optimizer.zero_grad()
        if is_distmult:
            ps = model(train_pos[:, :train_neg.shape[1]])
            ns = model(train_neg)
        else:
            nf = projector()
            z  = get_z(model, nf)
            ps = model.decode(z, train_pos[:, :train_neg.shape[1]])
            ns = model.decode(z, train_neg)
        loss = -torch.log(ps + 1e-15).mean() - torch.log(1 - ns + 1e-15).mean()
        loss.backward(); optimizer.step()
        if epoch % 5 == 0:
            model.eval(); projector.eval()
            with torch.no_grad():
                nf2 = projector() if not is_distmult else None
                z2  = get_z(model, nf2)
                va, _ = compute_auc_ap(
                    model.decode(z2, val_pos).numpy(),
                    model.decode(z2, val_neg).numpy()
                )
            if va > best_val: best_val, pat = va, 0
            else: pat += 1
            if pat >= PATIENCE: break
            model.train(); projector.train()
    model.eval(); projector.eval()
    with torch.no_grad():
        nf = projector() if not is_distmult else None
        z  = get_z(model, nf)
        auc, ap = compute_auc_ap(model.decode(z, test_pos).numpy(),
                                  model.decode(z, test_neg).numpy())
        rk = compute_ranking(model, projector, get_z)
    return {'auc': auc, 'ap': ap, **rk}

def make_hgt():
    m = create_hgt(PROJ_DIM, num_node_types, _num_rel, 64, 32, num_heads=4, num_layers=2)
    m.node_type_map = node_type_map
    return m

MODEL_CONFIGS = [
    ('GCN',       lambda: create_gcn(PROJ_DIM, 64, 32, 2),                        80,  0.01,  False, False),
    ('GraphSAGE', lambda: create_graphsage(PROJ_DIM, 64, 32, 2),                   80,  0.01,  False, False),
    ('GAT',       lambda: create_gat(PROJ_DIM, 32, 32, heads=4, num_layers=2),     80,  0.005, False, False),
    ('R-GCN',     lambda: create_rgcn(PROJ_DIM, _num_rel, 64, 32, 2, num_bases=4), 80,  0.01,  True,  False),
    ('HGT',       make_hgt,                                                         80,  0.005, True,  False),
    ('DistMult',  lambda: create_distmult(num_nodes, embedding_dim=64),             150, 0.01,  False, True),
    ('ComplEx',   lambda: create_complex(num_nodes,  embedding_dim=64),             150, 0.01,  False, True),
    ('TransE',    lambda: create_transe(num_nodes,   embedding_dim=64),             150, 0.01,  False, True),
]

# ── Run ablation ──────────────────────────────────────────────────────────────
ablation_results = {'original': {}, 'enriched': {}}

for condition, edges_by_type, edge_index in [
    ('original', orig_edges_by_type, orig_edge_index),
    ('enriched', enri_edges_by_type, enri_edge_index),
]:
    print(f"\n{'='*70}")
    print(f"  CONDITION: {condition.upper()}")
    print(f"{'='*70}")

    for name, make_model, epochs, lr, use_typed, is_distmult in MODEL_CONFIGS:
        seed_res = []
        for seed in SEEDS:
            r = run_seed(make_model, epochs, lr, use_typed, is_distmult,
                         seed, edges_by_type, edge_index)
            seed_res.append(r)

        keys = ['auc', 'mrr', 'hits@1', 'hits@10', 'mean_rank']
        agg  = {k: {'mean': float(np.mean([r[k] for r in seed_res])),
                    'std':  float(np.std([r[k]  for r in seed_res]))} for k in keys}
        ablation_results[condition][name] = agg
        print(f"  {name:<12} AUC={agg['auc']['mean']:.4f}±{agg['auc']['std']:.4f}  "
              f"MRR={agg['mrr']['mean']:.4f}±{agg['mrr']['std']:.4f}  "
              f"MnRk={agg['mean_rank']['mean']:.1f}±{agg['mean_rank']['std']:.1f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n\n{'='*80}")
print("ABLATION SUMMARY  —  Original vs Enriched (GO + KEGG)")
print(f"{'='*80}")
print(f"{'Model':<12} {'Orig MRR':>12} {'Enri MRR':>12} {'Delta MRR':>10} {'Orig MnRk':>12} {'Enri MnRk':>12}")
print("-"*80)
for name in ['GCN', 'GraphSAGE', 'GAT', 'R-GCN', 'HGT', 'DistMult', 'ComplEx', 'TransE']:
    o = ablation_results['original'][name]
    e = ablation_results['enriched'][name]
    delta = e['mrr']['mean'] - o['mrr']['mean']
    sign  = '+' if delta >= 0 else ''
    print(f"{name:<12} "
          f"{o['mrr']['mean']:>8.4f}±{o['mrr']['std']:.4f}  "
          f"{e['mrr']['mean']:>8.4f}±{e['mrr']['std']:.4f}  "
          f"{sign}{delta:>8.4f}  "
          f"{o['mean_rank']['mean']:>8.1f}±{o['mean_rank']['std']:.1f}  "
          f"{e['mean_rank']['mean']:>8.1f}±{e['mean_rank']['std']:.1f}")

out = Path(__file__).parent.parent / "results" / "metrics" / "ablation_enrichment.json"
with open(out, 'w') as f:
    json.dump(ablation_results, f, indent=2)
print(f"\nSaved to: {out}")
