"""
Add Gene Ontology (GO) annotations to the CARD hetero graph.
Used in: KIDS (Nature Communications, 2022)

Strategy:
1. Collect NCBI protein accessions from CARD genes
2. Batch-map to UniProt via the UniProt ID mapping API
3. Extract GO terms (Biological Process, Molecular Function, Cellular Component)
4. Add GO term nodes (new node type 6) and gene->GO edges
"""

import json
import time
import torch
import urllib.request
import urllib.parse
from pathlib import Path
from collections import defaultdict

DATA_DIR   = Path(__file__).parent.parent / "data"
CARD_JSON  = DATA_DIR / "raw" / "card" / "card.json"
GRAPH_PATH = DATA_DIR / "graphs" / "card_hetero_graph.pt"
SUMMARY_PATH = DATA_DIR / "graphs" / "hetero_graph_summary.txt"

UNIPROT_BASE = "https://rest.uniprot.org"
BATCH_SIZE   = 500   # UniProt recommends <= 500 per batch


# ── Helpers ──────────────────────────────────────────────────────────────────

def post_json(url, data):
    encoded = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=encoded, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def submit_mapping_job(accessions):
    """Submit a batch of NCBI protein accessions -> UniProtKB mapping job."""
    result = post_json(f"{UNIPROT_BASE}/idmapping/run", {
        "ids":  ",".join(accessions),
        "from": "EMBL-GenBank-DDBJ_CDS",
        "to":   "UniProtKB",
    })
    return result["jobId"]


def poll_job(job_id, max_wait=120):
    """Poll until the job is done, return results URL."""
    for _ in range(max_wait // 3):
        time.sleep(3)
        status = get_json(f"{UNIPROT_BASE}/idmapping/status/{job_id}")
        if "results" in status or "redirectURL" in status:
            return status
        if "failedIds" in status and not status.get("results"):
            return status
    raise TimeoutError(f"Job {job_id} did not finish in {max_wait}s")


def extract_go_terms(uniprot_entry):
    """Pull GO cross-references from a UniProt entry dict."""
    go_terms = []
    for xref in uniprot_entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") != "GO":
            continue
        go_id   = xref.get("id", "")          # e.g. GO:0005488
        props   = {p["key"]: p["value"] for p in xref.get("properties", [])}
        go_name = props.get("GoTerm", "")      # e.g. "F:nucleic acid binding"
        aspect  = go_name[0] if go_name else ""  # P/F/C
        if go_id:
            go_terms.append({
                "go_id":   go_id,
                "go_name": go_name[2:] if len(go_name) > 2 else go_name,
                "aspect":  {"P": "biological_process",
                            "F": "molecular_function",
                            "C": "cellular_component"}.get(aspect, "unknown"),
            })
    return go_terms


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ADDING GENE ONTOLOGY ANNOTATIONS")
    print("(Used in: KIDS, Nature Communications 2022)")
    print("=" * 70)

    # ── Collect CARD protein accessions ──────────────────────────────────────
    print("\nLoading CARD protein accessions...")
    with open(CARD_JSON) as f:
        card = json.load(f)
    for k in ["_version", "_comment", "_timestamp"]:
        card.pop(k, None)

    # aro_id -> list of protein accessions
    aro_to_accessions = defaultdict(list)
    for aro_id, entry in card.items():
        seqs = entry.get("model_sequences", {}).get("sequence", {})
        for seq in seqs.values():
            acc = seq.get("protein_sequence", {}).get("accession", "")
            if acc:
                aro_to_accessions[aro_id].append(acc)

    all_accessions = list({acc for accs in aro_to_accessions.values() for acc in accs})
    print(f"  CARD genes with accessions : {len(aro_to_accessions)}")
    print(f"  Unique protein accessions  : {len(all_accessions)}")

    # Build reverse map: accession -> aro_id (first match)
    acc_to_aro = {}
    for aro_id, accs in aro_to_accessions.items():
        for acc in accs:
            if acc not in acc_to_aro:
                acc_to_aro[acc] = aro_id

    # ── Batch UniProt ID mapping ──────────────────────────────────────────────
    print(f"\nMapping accessions to UniProt (batch size={BATCH_SIZE})...")
    batches = [all_accessions[i:i+BATCH_SIZE]
               for i in range(0, len(all_accessions), BATCH_SIZE)]

    uniprot_to_aro   = {}   # uniprot_acc -> aro_id
    aro_to_go_terms  = defaultdict(list)   # aro_id -> list of go dicts
    all_go_info      = {}   # go_id -> {go_name, aspect}

    for bi, batch in enumerate(batches):
        print(f"  Batch {bi+1}/{len(batches)} ({len(batch)} accessions)...", end=" ", flush=True)
        try:
            job_id = submit_mapping_job(batch)
            status = poll_job(job_id)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        results = status.get("results", [])
        print(f"{len(results)} mapped", end=" ")

        # Each result: {from: ncbi_acc, to: {uniprot entry dict}}
        for r in results:
            ncbi_acc = r.get("from", "")
            entry    = r.get("to", {})
            aro_id   = acc_to_aro.get(ncbi_acc)
            if not aro_id:
                continue

            go_terms = extract_go_terms(entry)
            if go_terms:
                aro_to_go_terms[aro_id].extend(go_terms)
                for gt in go_terms:
                    all_go_info[gt["go_id"]] = {
                        "go_name": gt["go_name"],
                        "aspect":  gt["aspect"],
                    }

        print(f"| GO terms so far: {len(all_go_info)}")
        time.sleep(1)

    print(f"\n  Genes with GO annotations  : {len(aro_to_go_terms)}")
    print(f"  Unique GO terms collected  : {len(all_go_info)}")

    if not all_go_info:
        print("No GO terms found. Graph unchanged.")
        return

    # Aspect breakdown
    aspect_counts = defaultdict(int)
    for gt in all_go_info.values():
        aspect_counts[gt["aspect"]] += 1
    for asp, cnt in sorted(aspect_counts.items()):
        print(f"    {asp}: {cnt}")

    # ── Load graph ────────────────────────────────────────────────────────────
    print("\nLoading graph...")
    graph_data    = torch.load(GRAPH_PATH)
    node_to_idx   = graph_data["node_to_idx"]
    idx_to_node   = graph_data["idx_to_node"]
    node_type_map = graph_data["node_type_map"]
    current_idx   = graph_data["num_nodes"]
    print(f"  Current nodes: {current_idx}")

    # ── Add GO term nodes (node type 6) ───────────────────────────────────────
    print("\nAdding GO term nodes...")
    new_go_count = 0
    for go_id, info in all_go_info.items():
        key = ("go_term", go_id)
        if key not in node_to_idx:
            node_to_idx[key]           = current_idx
            idx_to_node[current_idx]   = key
            node_type_map[current_idx] = 6
            current_idx += 1
            new_go_count += 1

    print(f"  New GO term nodes: {new_go_count}")

    # ── Build gene->GO edges ──────────────────────────────────────────────────
    edge_list = []
    skipped   = 0
    seen_pairs = set()

    for aro_id, go_terms in aro_to_go_terms.items():
        gene_key = ("gene", aro_id)
        if gene_key not in node_to_idx:
            skipped += 1
            continue
        gi = node_to_idx[gene_key]
        for gt in go_terms:
            go_key = ("go_term", gt["go_id"])
            if go_key not in node_to_idx:
                continue
            ti = node_to_idx[go_key]
            pair = (gi, ti)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                edge_list.append([gi, ti])
                edge_list.append([ti, gi])  # undirected

    print(f"  Gene->GO edges: {len(edge_list)//2}")
    if skipped:
        print(f"  Skipped (gene not in graph): {skipped}")

    # ── Add edges to graph ────────────────────────────────────────────────────
    edge_tensor = torch.tensor(edge_list, dtype=torch.long).t()
    next_id = max(graph_data["edge_type_to_id"].values()) + 1
    graph_data["typed_edge_indices"]["gene_to_go_term"]       = edge_tensor
    graph_data["train_typed_edge_indices"]["gene_to_go_term"] = edge_tensor
    graph_data["edge_type_to_id"]["gene_to_go_term"]          = next_id

    # ── Extend node features ──────────────────────────────────────────────────
    old_feat = graph_data["node_features"]
    n_old    = old_feat.shape[0]
    n_types  = old_feat.shape[1]
    n_new    = current_idx - n_old

    # Expand feature dim if needed (new type 6)
    needed_dim = 7
    if n_types < needed_dim:
        pad_cols = needed_dim - n_types
        old_feat = torch.cat([old_feat, torch.zeros(n_old, pad_cols)], dim=1)

    if n_new > 0:
        extra = torch.zeros(n_new, needed_dim)
        for i in range(n_old, current_idx):
            t = node_type_map[i]
            if t < needed_dim:
                extra[i - n_old, t] = 1.0
        graph_data["node_features"] = torch.cat([old_feat, extra], dim=0)
    else:
        graph_data["node_features"] = old_feat

    # ── Update metadata ───────────────────────────────────────────────────────
    graph_data["num_nodes"]         = current_idx
    graph_data["num_go_terms"]      = new_go_count
    graph_data["go_term_info"]      = all_go_info
    graph_data["node_to_idx"]       = node_to_idx
    graph_data["idx_to_node"]       = idx_to_node
    graph_data["node_type_map"]     = node_type_map

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(graph_data, GRAPH_PATH)
    print(f"\nSaved -> {GRAPH_PATH}")

    with open(SUMMARY_PATH, "a") as f:
        f.write("\nGene Ontology Integration:\n")
        f.write(f"  - Genes with GO annotations: {len(aro_to_go_terms)}\n")
        f.write(f"  - GO term nodes added       : {new_go_count}\n")
        f.write(f"  - Gene->GO edges            : {len(edge_list)//2}\n")
        f.write(f"  - Total nodes now           : {current_idx}\n")

    total_e = sum(t.shape[1] for t in graph_data["typed_edge_indices"].values())
    print("\n" + "=" * 70)
    print("Gene Ontology integration complete!")
    print(f"  Nodes: {current_idx}  |  GO terms: {new_go_count}")
    print(f"  Total directed edges: {total_e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
