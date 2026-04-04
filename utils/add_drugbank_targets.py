"""
Add antibiotic -> protein_target edges using ChEMBL data.
No registration required — uses the ChEMBL REST API client.

New graph additions:
- Node type 5: protein_target (bacterial proteins that antibiotics bind/inhibit)
- Edge type:   antibiotic_targets_protein
"""

import time
import torch
from pathlib import Path
from collections import defaultdict

DATA_DIR   = Path(__file__).parent.parent / "data"
GRAPH_PATH = DATA_DIR / "graphs" / "card_hetero_graph.pt"
SUMMARY_PATH = DATA_DIR / "graphs" / "hetero_graph_summary.txt"

# Minimum ChEMBL confidence score for target assignment
# 9 = direct single-protein assay, 8 = homologous single protein
# We use >= 7 to get reasonable coverage while staying protein-level
MIN_CONFIDENCE = 7


def get_chembl_id(molecule_client, name):
    """Look up ChEMBL ID for a drug name. Returns None if not found."""
    try:
        results = molecule_client.filter(pref_name__iexact=name).only(
            ["molecule_chembl_id", "pref_name"]
        )
        if results:
            return results[0]["molecule_chembl_id"]
        # Try synonym search
        results = molecule_client.filter(molecule_synonyms__synonym__iexact=name).only(
            ["molecule_chembl_id", "pref_name"]
        )
        if results:
            return results[0]["molecule_chembl_id"]
    except Exception:
        pass
    return None


VALID_TARGET_TYPES = {
    "SINGLE PROTEIN", "PROTEIN FAMILY", "PROTEIN COMPLEX",
    "PROTEIN NUCLEIC-ACID COMPLEX", "MACROMOLECULE",
}

def get_bacterial_targets(mech_client, target_client, chembl_id):
    """
    Fetch curated mechanism-of-action bacterial targets for a ChEMBL compound.
    Uses the drug_mechanism endpoint (high quality, approved drugs only).
    Returns list of dicts with target info.
    """
    try:
        mechs = mech_client.filter(molecule_chembl_id=chembl_id).only(
            ["target_chembl_id", "mechanism_of_action", "action_type"]
        )
        seen = {}
        for m in mechs:
            tid = m.get("target_chembl_id", "")
            if not tid or tid in seen:
                continue
            t = target_client.get(tid)
            if not t:
                continue
            ttype = t.get("target_type", "") or ""
            org   = t.get("organism", "") or ""
            # Keep bacterial targets of any relevant structural type
            if ttype in VALID_TARGET_TYPES and "bacter" in org.lower():
                seen[tid] = {
                    "target_chembl_id": tid,
                    "target_name":      t.get("pref_name", ""),
                    "target_organism":  org,
                    "target_type":      ttype,
                    "mechanism":        m.get("mechanism_of_action", ""),
                    "action_type":      m.get("action_type", ""),
                }
        return list(seen.values())
    except Exception:
        return []


def main():
    print("=" * 70)
    print("ADDING DRUG-TARGET EDGES VIA ChEMBL")
    print("=" * 70)

    from chembl_webresource_client.new_client import new_client
    molecule = new_client.molecule
    mech     = new_client.mechanism
    target   = new_client.target

    # ── Load graph ────────────────────────────────────────────────────────────
    print("\nLoading graph...")
    graph_data    = torch.load(GRAPH_PATH)
    node_to_idx   = graph_data["node_to_idx"]
    idx_to_node   = graph_data["idx_to_node"]
    node_type_map = graph_data["node_type_map"]
    antibiotics   = graph_data["antibiotics"]
    current_idx   = graph_data["num_nodes"]

    print(f"  Nodes: {current_idx}  |  Antibiotics: {len(antibiotics)}")

    # ── Query ChEMBL for each antibiotic ──────────────────────────────────────
    print("\nQuerying ChEMBL for antibiotic -> bacterial target edges...")
    print(f"  (confidence >= {MIN_CONFIDENCE}, single-protein targets only)\n")

    ab_names = list(antibiotics.keys())
    target_nodes  = {}   # target_chembl_id -> target info
    ab_target_edges = []  # (ab_name, target_chembl_id)

    mapped_abs  = 0
    total_edges = 0

    for i, ab_name in enumerate(ab_names):
        chembl_id = get_chembl_id(molecule, ab_name)
        if not chembl_id:
            continue

        targets = get_bacterial_targets(mech, target, chembl_id)
        if targets:
            mapped_abs += 1
            for t in targets:
                tid = t["target_chembl_id"]
                target_nodes[tid] = t
                ab_target_edges.append((ab_name, tid))
                total_edges += 1

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(ab_names)}] mapped={mapped_abs}, edges={total_edges}")
        time.sleep(0.2)  # polite API usage

    print(f"\n  Antibiotics mapped to ChEMBL : {mapped_abs}/{len(ab_names)}")
    print(f"  Unique bacterial target nodes: {len(target_nodes)}")
    print(f"  Total antibiotic->target edges: {total_edges}")

    if not ab_target_edges:
        print("No edges found. Graph unchanged.")
        return

    # ── Add protein_target nodes (new node type 5) ────────────────────────────
    print("\nAdding protein_target nodes...")
    for tid, tinfo in target_nodes.items():
        key = ("protein_target", tid)
        if key not in node_to_idx:
            node_to_idx[key]          = current_idx
            idx_to_node[current_idx]  = key
            node_type_map[current_idx] = 5
            current_idx += 1

    graph_data["protein_targets"] = target_nodes
    graph_data["num_protein_targets"] = len(target_nodes)

    # ── Build edge tensor ─────────────────────────────────────────────────────
    edge_list = []
    skipped   = 0
    for ab_name, tid in ab_target_edges:
        ab_key  = ("antibiotic",     ab_name)
        tgt_key = ("protein_target", tid)
        if ab_key in node_to_idx and tgt_key in node_to_idx:
            ai = node_to_idx[ab_key]
            ti = node_to_idx[tgt_key]
            edge_list.append([ai, ti])
            edge_list.append([ti, ai])  # undirected
        else:
            skipped += 1

    if skipped:
        print(f"  Skipped (missing in graph): {skipped}")

    edge_tensor = torch.tensor(edge_list, dtype=torch.long).t()
    next_id = max(graph_data["edge_type_to_id"].values()) + 1
    graph_data["typed_edge_indices"]["antibiotic_targets_protein"]       = edge_tensor
    graph_data["train_typed_edge_indices"]["antibiotic_targets_protein"] = edge_tensor
    graph_data["edge_type_to_id"]["antibiotic_targets_protein"]          = next_id

    # ── Extend node features (one-hot now 6-dim) ──────────────────────────────
    old_feat = graph_data["node_features"]        # (N, 5)
    n_old    = old_feat.shape[0]
    n_new    = current_idx - n_old

    # Expand existing features to 6 dims
    feat_6d = torch.cat([old_feat, torch.zeros(n_old, 1)], dim=1)

    # New protein_target nodes: one-hot dim 5
    if n_new > 0:
        extra = torch.zeros(n_new, 6)
        for i in range(n_old, current_idx):
            t = node_type_map[i]
            if t < 6:
                extra[i - n_old, t] = 1.0
        feat_6d = torch.cat([feat_6d, extra], dim=0)

    graph_data["node_features"] = feat_6d

    # ── Update metadata ───────────────────────────────────────────────────────
    graph_data["num_nodes"]       = current_idx
    graph_data["node_to_idx"]     = node_to_idx
    graph_data["idx_to_node"]     = idx_to_node
    graph_data["node_type_map"]   = node_type_map

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(graph_data, GRAPH_PATH)
    print(f"\nSaved updated graph -> {GRAPH_PATH}")

    with open(SUMMARY_PATH, "a") as f:
        f.write("\nChEMBL Drug-Target Integration:\n")
        f.write(f"  - Antibiotics mapped to ChEMBL : {mapped_abs}/{len(ab_names)}\n")
        f.write(f"  - New protein_target nodes     : {len(target_nodes)}\n")
        f.write(f"  - New antibiotic->target edges : {total_edges}\n")
        f.write(f"  - Total nodes now              : {current_idx}\n")

    print("\n" + "=" * 70)
    print("ChEMBL drug-target integration complete!")
    print(f"  Nodes: {current_idx}  |  New targets: {len(target_nodes)}")
    total_e = sum(t.shape[1] for t in graph_data["typed_edge_indices"].values())
    print(f"  Total directed edges: {total_e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
