"""
Integrate MEGARes v3.0 into the CARD heterogeneous graph.

Strategy:
- Download MEGARes annotations + external header mappings
- Use mappings to identify which MEGARes gene groups overlap with CARD
- Add MEGARes-only gene groups as new gene nodes
- Add new edges: gene -> drug_class, gene -> mechanism, gene -> compound_type
- Preserves existing CARD structure; only appends new nodes/edges
"""

import csv
import io
import json
import urllib.request
import torch
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
GRAPH_PATH = DATA_DIR / "graphs" / "card_hetero_graph.pt"
SUMMARY_PATH = DATA_DIR / "graphs" / "hetero_graph_summary.txt"

ANNOTATIONS_URL = "https://www.meglab.org/downloads/megares_v3.00/megares_annotations_v3.00.csv"
MAPPINGS_URL    = "https://www.meglab.org/downloads/megares_v3.00/megares_to_external_header_mappings_v3.00.csv"


def fetch_csv(url):
    print(f"  Downloading {url.split('/')[-1]} ...")
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read().decode("utf-8")


def parse_annotations(csv_text):
    """Returns list of dicts with keys: header, type, class, mechanism, group."""
    reader = csv.DictReader(io.StringIO(csv_text))
    return list(reader)


def parse_mappings(csv_text):
    """Returns set of MEGARes headers that have a CARD mapping."""
    reader = csv.DictReader(io.StringIO(csv_text))
    card_mapped = set()
    for row in reader:
        db = row.get("Database", "").strip()
        meg_header = row.get("MEGARes_header", "").strip()
        if db == "CARD" and meg_header:
            card_mapped.add(meg_header)
    return card_mapped


def main():
    print("=" * 70)
    print("INTEGRATING MEGARes v3.0 INTO CARD HETERO GRAPH")
    print("=" * 70)

    # ── Download ─────────────────────────────────────────────────────────────
    annotations_csv = fetch_csv(ANNOTATIONS_URL)
    mappings_csv    = fetch_csv(MAPPINGS_URL)

    annotations  = parse_annotations(annotations_csv)
    card_headers = parse_mappings(mappings_csv)

    print(f"\n  MEGARes total entries  : {len(annotations)}")
    print(f"  Entries mapped to CARD : {len(card_headers)}")
    print(f"  MEGARes-only entries   : {len(annotations) - len(card_headers)}")

    # ── Group entries by gene group (skip CARD-overlapping ones) ─────────────
    # A gene group is novel only if NONE of its entries map to CARD
    group_info   = {}   # group -> {type, class, mechanism, headers: [...]}
    for row in annotations:
        grp  = row["group"].strip()
        hdr  = row["header"].strip()
        if grp not in group_info:
            group_info[grp] = {
                "type":      row["type"].strip(),
                "class":     row["class"].strip(),
                "mechanism": row["mechanism"].strip(),
                "headers":   [],
                "has_card":  False,
            }
        group_info[grp]["headers"].append(hdr)
        if hdr in card_headers:
            group_info[grp]["has_card"] = True

    novel_groups = {g: d for g, d in group_info.items() if not d["has_card"]}
    print(f"\n  Total gene groups      : {len(group_info)}")
    print(f"  Groups overlapping CARD: {len(group_info) - len(novel_groups)}")
    print(f"  Novel MEGARes groups   : {len(novel_groups)}")

    # Type breakdown
    type_counts = defaultdict(int)
    for d in novel_groups.values():
        type_counts[d["type"]] += 1
    print("\n  Novel groups by type:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:20s}: {c}")

    # ── Load existing graph ───────────────────────────────────────────────────
    print("\nLoading existing graph...")
    graph_data   = torch.load(GRAPH_PATH)
    node_to_idx  = graph_data["node_to_idx"]
    idx_to_node  = graph_data["idx_to_node"]
    node_type_map = graph_data["node_type_map"]
    current_idx  = graph_data["num_nodes"]

    resistance_genes = graph_data["resistance_genes"]
    drug_classes     = graph_data["drug_classes"]
    mechanisms       = graph_data["mechanisms"]

    print(f"  Existing nodes: {current_idx}")
    print(f"  Existing genes: {len(resistance_genes)}")

    # ── Add new nodes ─────────────────────────────────────────────────────────
    new_genes      = {}   # meg_id -> info dict
    new_classes    = {}   # class_name -> info
    new_mechanisms = {}   # mech_name -> info

    for grp, d in novel_groups.items():
        meg_id   = f"MEG_{grp}"
        class_nm = d["class"]
        mech_nm  = d["mechanism"]
        ctype    = d["type"]

        new_genes[meg_id] = {
            "aro_id":      meg_id,
            "name":        grp,
            "accession":   "",
            "description": f"MEGARes gene group: {grp} ({ctype})",
            "model_type":  "MEGARes",
            "source":      "MEGARes_v3",
            "compound_type": ctype,
        }

        if class_nm not in drug_classes and class_nm not in new_classes:
            new_classes[class_nm] = {"name": class_nm}

        if mech_nm not in mechanisms and mech_nm not in new_mechanisms:
            new_mechanisms[mech_nm] = {"name": mech_nm}

    print(f"\n  New gene nodes to add   : {len(new_genes)}")
    print(f"  New drug class nodes    : {len(new_classes)}")
    print(f"  New mechanism nodes     : {len(new_mechanisms)}")

    # Register new gene nodes (type 0)
    for meg_id, info in new_genes.items():
        node_to_idx[("gene", meg_id)] = current_idx
        idx_to_node[current_idx]      = ("gene", meg_id)
        node_type_map[current_idx]    = 0
        resistance_genes[meg_id]      = info
        current_idx += 1

    # Register new drug class nodes (type 2)
    for class_nm in new_classes:
        node_to_idx[("drug_class", class_nm)] = current_idx
        idx_to_node[current_idx]              = ("drug_class", class_nm)
        node_type_map[current_idx]            = 2
        drug_classes[class_nm]               = {"name": class_nm}
        current_idx += 1

    # Register new mechanism nodes (type 3)
    for mech_nm in new_mechanisms:
        node_to_idx[("mechanism", mech_nm)] = current_idx
        idx_to_node[current_idx]            = ("mechanism", mech_nm)
        node_type_map[current_idx]          = 3
        mechanisms[mech_nm]                 = {"name": mech_nm}
        current_idx += 1

    # ── Build new edges ───────────────────────────────────────────────────────
    new_gene_to_class   = []   # (gene_idx, class_idx)  -- reuse antibiotic_to_class type
    new_gene_to_mech    = []   # (gene_idx, mech_idx)   -- reuse gene_to_mechanism type

    for grp, d in novel_groups.items():
        meg_id   = f"MEG_{grp}"
        class_nm = d["class"]
        mech_nm  = d["mechanism"]

        gene_idx  = node_to_idx[("gene",       meg_id)]
        class_idx = node_to_idx[("drug_class", class_nm)]
        mech_idx  = node_to_idx[("mechanism",  mech_nm)]

        new_gene_to_class.append([gene_idx, class_idx])
        new_gene_to_class.append([class_idx, gene_idx])   # undirected
        new_gene_to_mech.append([gene_idx, mech_idx])
        new_gene_to_mech.append([mech_idx,  gene_idx])    # undirected

    print(f"\n  New gene->class edges   : {len(new_gene_to_class)//2}")
    print(f"  New gene->mechanism edges: {len(new_gene_to_mech)//2}")

    # Append to existing typed edge indices
    def extend_edges(existing_tensor, new_edges):
        if not new_edges:
            return existing_tensor
        new_t = torch.tensor(new_edges, dtype=torch.long).t()
        return torch.cat([existing_tensor, new_t], dim=1)

    # gene_to_mechanism
    graph_data["typed_edge_indices"]["gene_to_mechanism"] = extend_edges(
        graph_data["typed_edge_indices"]["gene_to_mechanism"], new_gene_to_mech
    )
    graph_data["train_typed_edge_indices"]["gene_to_mechanism"] = extend_edges(
        graph_data["train_typed_edge_indices"]["gene_to_mechanism"], new_gene_to_mech
    )

    # MEGARes genes -> drug classes via new edge type "meg_gene_to_class"
    # (separate from antibiotic_to_class which is drug-level; this is gene-level)
    if new_gene_to_class:
        new_class_tensor = torch.tensor(new_gene_to_class, dtype=torch.long).t()
        graph_data["typed_edge_indices"]["meg_gene_to_class"]       = new_class_tensor
        graph_data["train_typed_edge_indices"]["meg_gene_to_class"] = new_class_tensor
        next_id = max(graph_data["edge_type_to_id"].values()) + 1
        graph_data["edge_type_to_id"]["meg_gene_to_class"]          = next_id

    # ── Update node features tensor ───────────────────────────────────────────
    old_features = graph_data["node_features"]        # (old_N, 5) one-hot type
    n_new = current_idx - graph_data["num_nodes"]
    if n_new > 0:
        extra = torch.zeros(n_new, old_features.shape[1])
        for i in range(graph_data["num_nodes"], current_idx):
            t = node_type_map[i]
            if t < extra.shape[1]:
                extra[i - graph_data["num_nodes"], t] = 1.0
        graph_data["node_features"] = torch.cat([old_features, extra], dim=0)

    # ── Update metadata ───────────────────────────────────────────────────────
    graph_data["num_nodes"]       = current_idx
    graph_data["num_genes"]       = len(resistance_genes)
    graph_data["num_drug_classes"]= len(drug_classes)
    graph_data["num_mechanisms"]  = len(mechanisms)
    graph_data["node_to_idx"]     = node_to_idx
    graph_data["idx_to_node"]     = idx_to_node
    graph_data["node_type_map"]   = node_type_map
    graph_data["resistance_genes"]= resistance_genes
    graph_data["drug_classes"]    = drug_classes
    graph_data["mechanisms"]      = mechanisms

    # gene_rich_features: extend with zeros for new gene nodes
    if graph_data.get("gene_rich_features") is not None:
        old_gf = graph_data["gene_rich_features"]
        pad    = torch.zeros(len(new_genes), old_gf.shape[1])
        graph_data["gene_rich_features"] = torch.cat([old_gf, pad], dim=0)

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(graph_data, GRAPH_PATH)
    print(f"\nSaved updated graph -> {GRAPH_PATH}")

    # Append to summary
    with open(SUMMARY_PATH, "a") as f:
        f.write("\nMEGARes v3.0 Integration:\n")
        f.write(f"  - Novel gene groups added : {len(new_genes)}\n")
        f.write(f"  - New drug class nodes    : {len(new_classes)}\n")
        f.write(f"  - New mechanism nodes     : {len(new_mechanisms)}\n")
        f.write(f"  - New gene->class edges   : {len(new_gene_to_class)//2}\n")
        f.write(f"  - New gene->mechanism edges:{len(new_gene_to_mech)//2}\n")
        f.write(f"  - Total nodes now         : {current_idx}\n")

    print("\n" + "=" * 70)
    print("MEGARes integration complete!")
    print(f"  Nodes: {graph_data['num_nodes']}  |  Genes: {graph_data['num_genes']}")
    total_edges = sum(t.shape[1] for t in graph_data["typed_edge_indices"].values())
    print(f"  Total directed edges: {total_edges}")
    print("=" * 70)


if __name__ == "__main__":
    main()
