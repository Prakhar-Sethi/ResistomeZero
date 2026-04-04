"""
Add KEGG Pathway annotations to the CARD hetero graph.
Used in: HGT-AMR Knowledge Graph (bioRxiv 2025)

Strategy:
1. For each CARD gene name, query the KEGG REST API for matching genes
2. Get the KEGG pathways those genes belong to
3. Add pathway nodes (node type 7) and gene->pathway edges

KEGG REST API (free, no registration):
  https://rest.kegg.jp/find/genes/{query}
  https://rest.kegg.jp/link/pathway/{gene_id}
  https://rest.kegg.jp/get/{pathway_id}
"""

import json
import time
import torch
import urllib.request
from pathlib import Path
from collections import defaultdict

DATA_DIR   = Path(__file__).parent.parent / "data"
CARD_JSON  = DATA_DIR / "raw" / "card" / "card.json"
GRAPH_PATH = DATA_DIR / "graphs" / "card_hetero_graph.pt"
SUMMARY_PATH = DATA_DIR / "graphs" / "hetero_graph_summary.txt"

KEGG_BASE = "https://rest.kegg.jp"


def kegg_get(path, retries=3):
    """Simple KEGG REST API GET with retry."""
    url = f"{KEGG_BASE}/{path}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                return r.read().decode("utf-8")
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return ""


def find_kegg_gene(gene_name):
    """Search KEGG for a gene name. Returns list of (kegg_id, description)."""
    text = kegg_get(f"find/genes/{urllib.request.quote(gene_name)}")
    results = []
    for line in text.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            results.append((parts[0].strip(), parts[1].strip()))
    return results


def get_gene_pathways(kegg_gene_id):
    """Get pathway IDs for a KEGG gene ID."""
    text = kegg_get(f"link/pathway/{kegg_gene_id}")
    pathway_ids = []
    for line in text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            pid = parts[1].strip()
            # Only keep KEGG pathway maps (map/ko/organism-specific)
            if ":map" in pid or ":path" in pid or "path:" in pid:
                pathway_ids.append(pid)
    return pathway_ids


def get_pathway_name(pathway_id):
    """Get the name of a KEGG pathway."""
    # pathway_id like 'path:map00500' or 'ko:map00500'
    clean = pathway_id.split(":")[-1]
    text = kegg_get(f"get/{clean}")
    for line in text.splitlines():
        if line.startswith("NAME"):
            return line.split(None, 1)[-1].strip().rstrip(".")
    return clean


# Priority organisms to search in KEGG
# These are the well-represented ones in CARD and KEGG
PRIORITY_ORGS = ["eco", "pae", "kpn", "abn", "mtv", "sau"]  # E.coli, P.aeruginosa, K.pneumoniae, A.baumannii, M.tb, S.aureus


def find_gene_in_kegg(gene_name):
    """
    Find a gene in KEGG by name. Try organism-specific searches first,
    then fall back to KO (orthology) search.
    Returns (kegg_id, org) or (None, None).
    """
    # Try KO orthology search (most general)
    results = find_kegg_gene(gene_name)
    if not results:
        return None, None

    # Prefer results from priority organisms
    for kegg_id, desc in results:
        org = kegg_id.split(":")[0] if ":" in kegg_id else ""
        if org in PRIORITY_ORGS:
            return kegg_id, org

    # Otherwise take first result
    kegg_id, _ = results[0]
    org = kegg_id.split(":")[0] if ":" in kegg_id else ""
    return kegg_id, org


def main():
    print("=" * 70)
    print("ADDING KEGG PATHWAY ANNOTATIONS")
    print("(Used in: HGT-AMR KG, bioRxiv 2025)")
    print("=" * 70)

    # ── Load CARD gene names ──────────────────────────────────────────────────
    print("\nLoading CARD gene names...")
    with open(CARD_JSON) as f:
        card = json.load(f)
    for k in ["_version", "_comment", "_timestamp"]:
        card.pop(k, None)

    # For KEGG, intrinsic/chromosomal resistance genes match best
    # Focus on genes with short, standard names (not "OXA-45", "KPC-2" variants)
    # Filter: name without digits after a dash, or known gene name patterns
    aro_to_name = {}
    for aro_id, entry in card.items():
        name = entry.get("ARO_name", "")
        # Skip pure variant names (e.g. "OXA-45", "KPC-2") — unlikely in KEGG
        # Keep names like "mexB", "ampC", "gyrA", "rpoB", "ermA"
        base = name.split("-")[0] if "-" in name else name
        if len(base) >= 3 and base.isalpha() and len(base) <= 8:
            aro_to_name[aro_id] = name

    print(f"  CARD genes with searchable names: {len(aro_to_name)}")
    print(f"  (Filtering to standard gene names for KEGG compatibility)")

    # ── Query KEGG for each gene ──────────────────────────────────────────────
    print("\nQuerying KEGG REST API...")
    aro_to_pathways  = defaultdict(set)   # aro_id -> set of pathway_ids
    pathway_names    = {}                 # pathway_id -> name
    searched = 0
    matched  = 0

    for aro_id, gene_name in aro_to_name.items():
        kegg_id, org = find_gene_in_kegg(gene_name)
        searched += 1

        if kegg_id:
            pathways = get_gene_pathways(kegg_id)
            if pathways:
                matched += 1
                for pid in pathways:
                    aro_to_pathways[aro_id].add(pid)
                    if pid not in pathway_names:
                        pname = get_pathway_name(pid)
                        pathway_names[pid] = pname
                        time.sleep(0.1)

        if searched % 50 == 0:
            print(f"  [{searched}/{len(aro_to_name)}] genes matched={matched}, pathways={len(pathway_names)}")
        time.sleep(0.15)  # respect KEGG rate limit (10 req/s max)

    print(f"\n  Genes queried    : {searched}")
    print(f"  Genes matched    : {matched}")
    print(f"  Unique pathways  : {len(pathway_names)}")

    if not pathway_names:
        print("No pathway data found. Graph unchanged.")
        return

    # Sample pathways
    print("\n  Sample pathways found:")
    for pid, pname in list(pathway_names.items())[:8]:
        print(f"    {pid}: {pname}")

    # ── Load graph ────────────────────────────────────────────────────────────
    print("\nLoading graph...")
    graph_data    = torch.load(GRAPH_PATH)
    node_to_idx   = graph_data["node_to_idx"]
    idx_to_node   = graph_data["idx_to_node"]
    node_type_map = graph_data["node_type_map"]
    current_idx   = graph_data["num_nodes"]
    print(f"  Current nodes: {current_idx}")

    # ── Add pathway nodes (node type 7) ───────────────────────────────────────
    new_pathway_count = 0
    for pid in pathway_names:
        key = ("kegg_pathway", pid)
        if key not in node_to_idx:
            node_to_idx[key]           = current_idx
            idx_to_node[current_idx]   = key
            node_type_map[current_idx] = 7
            current_idx += 1
            new_pathway_count += 1

    print(f"  New pathway nodes: {new_pathway_count}")

    # ── Build gene->pathway edges ─────────────────────────────────────────────
    edge_list  = []
    seen_pairs = set()
    skipped    = 0

    for aro_id, pathway_ids in aro_to_pathways.items():
        gene_key = ("gene", aro_id)
        if gene_key not in node_to_idx:
            skipped += 1
            continue
        gi = node_to_idx[gene_key]
        for pid in pathway_ids:
            pkey = ("kegg_pathway", pid)
            if pkey not in node_to_idx:
                continue
            pi   = node_to_idx[pkey]
            pair = (gi, pi)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                edge_list.append([gi, pi])
                edge_list.append([pi, gi])

    print(f"  Gene->pathway edges: {len(edge_list)//2}")

    # ── Add to graph ──────────────────────────────────────────────────────────
    if edge_list:
        edge_tensor = torch.tensor(edge_list, dtype=torch.long).t()
        next_id = max(graph_data["edge_type_to_id"].values()) + 1
        graph_data["typed_edge_indices"]["gene_to_kegg_pathway"]       = edge_tensor
        graph_data["train_typed_edge_indices"]["gene_to_kegg_pathway"] = edge_tensor
        graph_data["edge_type_to_id"]["gene_to_kegg_pathway"]          = next_id

    # ── Extend node features ──────────────────────────────────────────────────
    old_feat   = graph_data["node_features"]
    n_old      = old_feat.shape[0]
    n_types    = old_feat.shape[1]
    n_new      = current_idx - n_old
    needed_dim = max(n_types, 8)

    if n_types < needed_dim:
        old_feat = torch.cat([old_feat, torch.zeros(n_old, needed_dim - n_types)], dim=1)

    if n_new > 0:
        extra = torch.zeros(n_new, needed_dim)
        for i in range(n_old, current_idx):
            t = node_type_map[i]
            if t < needed_dim:
                extra[i - n_old, t] = 1.0
        graph_data["node_features"] = torch.cat([old_feat, extra], dim=0)
    else:
        graph_data["node_features"] = old_feat

    # ── Save ──────────────────────────────────────────────────────────────────
    graph_data["num_nodes"]        = current_idx
    graph_data["num_kegg_pathways"]= new_pathway_count
    graph_data["kegg_pathway_info"]= pathway_names
    graph_data["node_to_idx"]      = node_to_idx
    graph_data["idx_to_node"]      = idx_to_node
    graph_data["node_type_map"]    = node_type_map

    torch.save(graph_data, GRAPH_PATH)
    print(f"\nSaved -> {GRAPH_PATH}")

    with open(SUMMARY_PATH, "a") as f:
        f.write("\nKEGG Pathway Integration:\n")
        f.write(f"  - Genes matched to KEGG   : {matched}/{searched}\n")
        f.write(f"  - Pathway nodes added      : {new_pathway_count}\n")
        f.write(f"  - Gene->pathway edges      : {len(edge_list)//2}\n")
        f.write(f"  - Total nodes now          : {current_idx}\n")

    total_e = sum(t.shape[1] for t in graph_data["typed_edge_indices"].values())
    print("\n" + "=" * 70)
    print("KEGG Pathway integration complete!")
    print(f"  Nodes: {current_idx}  |  Pathways: {new_pathway_count}")
    print(f"  Total directed edges: {total_e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
