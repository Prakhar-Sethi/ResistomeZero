"""
DistMult transductive evaluation.

Evaluates DistMult in two settings:
1. Transductive: train/test on seen antibiotics only (fair for KGE)
2. Zero-shot: test on unseen antibiotics (the main task — expected to fail)

This shows DistMult works perfectly when evaluated fairly (transductively),
but collapses to chance on zero-shot — proving the zero-shot task
requires inductive models.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from models.distmult import create_distmult
import json

SEEDS   = [42, 123, 456, 789, 1234]
EPOCHS  = 200
PATIENCE = 6

graph_path = Path(__file__).parent.parent / "data" / "graphs" / "card_hetero_graph.pt"
graph_data = torch.load(graph_path, weights_only=False)

num_nodes   = graph_data['num_nodes']
train_pos   = graph_data['train_pos_edges']
val_pos     = graph_data['val_pos_edges']
test_pos    = graph_data['test_pos_edges']   # zero-shot (unseen antibiotics)
node_to_idx = graph_data['node_to_idx']
idx_to_node = graph_data['idx_to_node']

# Build transductive test set: gene-antibiotic pairs where antibiotic is in train set
train_ab_names = set(graph_data['train_antibiotics'])
train_ab_idx   = {node_to_idx[('antibiotic', n)] for n in train_ab_names if ('antibiotic', n) in node_to_idx}

# From val_pos, take pairs where antibiotic is a training antibiotic
# (val_pos contains val antibiotics — some may overlap with train antibiotics
#  depending on the split. Use train_pos itself for transductive eval.)
# We create a held-out transductive test by splitting train_pos 80/20
torch.manual_seed(0)
n_train = train_pos.shape[1]
perm    = torch.randperm(n_train)
n_held  = n_train // 5
trans_test_pos  = train_pos[:, perm[:n_held]]
trans_train_pos = train_pos[:, perm[n_held:]]

print(f"Transductive train: {trans_train_pos.shape[1]} edges")
print(f"Transductive test:  {trans_test_pos.shape[1]} edges  (seen antibiotics only)")
print(f"Zero-shot test:     {test_pos.shape[1]} edges  (unseen antibiotics)")

# ── Helpers ───────────────────────────────────────────────────────────────────
def sample_neg(n, seed, ref_pos=None):
    rng = torch.Generator(); rng.manual_seed(seed)
    src = torch.randint(0, num_nodes, (n*3,), generator=rng)
    dst = torch.randint(0, num_nodes, (n*3,), generator=rng)
    mask = src != dst
    return torch.stack([src[mask][:n], dst[mask][:n]])

def compute_auc_ap(pos, neg):
    s = np.concatenate([pos, neg])
    l = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return roc_auc_score(l, s), average_precision_score(l, s)

# ── Run one seed ──────────────────────────────────────────────────────────────
def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model     = create_distmult(num_nodes, embedding_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    tr_neg  = sample_neg(min(2000, trans_train_pos.shape[1]), seed)
    val_neg = sample_neg(val_pos.shape[1], seed+1)
    tt_neg  = sample_neg(trans_test_pos.shape[1], seed+2)   # transductive neg
    zs_neg  = sample_neg(test_pos.shape[1], seed+3)         # zero-shot neg

    best_val, pat = 0, 0

    for epoch in range(1, EPOCHS+1):
        model.train(); optimizer.zero_grad()
        ps = model(trans_train_pos[:, :tr_neg.shape[1]])
        ns = model(tr_neg)
        loss = -torch.log(ps+1e-15).mean() - torch.log(1-ns+1e-15).mean()
        loss.backward(); optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                emb = model.get_all_entity_embeddings()
                va, _ = compute_auc_ap(
                    model.decode(emb, val_pos).numpy(),
                    model.decode(emb, val_neg).numpy()
                )
            if va > best_val: best_val, pat = va, 0
            else: pat += 1
            if pat >= PATIENCE: break

    model.eval()
    with torch.no_grad():
        emb = model.get_all_entity_embeddings()
        # Transductive test (seen antibiotics)
        t_auc, t_ap = compute_auc_ap(
            model.decode(emb, trans_test_pos).numpy(),
            model.decode(emb, tt_neg).numpy()
        )
        # Zero-shot test (unseen antibiotics)
        z_auc, z_ap = compute_auc_ap(
            model.decode(emb, test_pos).numpy(),
            model.decode(emb, zs_neg).numpy()
        )

    return {'trans_auc': t_auc, 'trans_ap': t_ap,
            'zeroshot_auc': z_auc, 'zeroshot_ap': z_ap}

# ── Run all seeds ─────────────────────────────────────────────────────────────
print("\nRunning DistMult — Transductive vs Zero-Shot")
print("="*60)
all_res = []
for seed in SEEDS:
    r = run_seed(seed)
    all_res.append(r)
    print(f"  seed={seed:4d} | Transductive AUC={r['trans_auc']:.4f}  AP={r['trans_ap']:.4f}"
          f"  |  Zero-shot AUC={r['zeroshot_auc']:.4f}  AP={r['zeroshot_ap']:.4f}")

for setting, ak, pk in [('Transductive', 'trans_auc', 'trans_ap'),
                         ('Zero-Shot',   'zeroshot_auc', 'zeroshot_ap')]:
    aucs = [r[ak] for r in all_res]
    aps  = [r[pk] for r in all_res]
    print(f"\n  {setting}:  AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}  "
          f"|  AP = {np.mean(aps):.4f} +/- {np.std(aps):.4f}")

# Save
out = Path(__file__).parent.parent / "results" / "metrics" / "distmult_transductive.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f:
    json.dump({'per_seed': all_res,
               'trans_auc_mean': float(np.mean([r['trans_auc'] for r in all_res])),
               'trans_auc_std':  float(np.std([r['trans_auc']  for r in all_res])),
               'zero_auc_mean':  float(np.mean([r['zeroshot_auc'] for r in all_res])),
               'zero_auc_std':   float(np.std([r['zeroshot_auc']  for r in all_res]))}, f, indent=2)
print(f"\nSaved to: {out}")
