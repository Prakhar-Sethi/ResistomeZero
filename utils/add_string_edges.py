"""
Fetch STRING PPI edges for CARD resistance genes and add to the hetero graph.

v2 — fixed identifier strategy:
  CARD variant names (OXA-48, KPC-2) do not match STRING canonical symbols.
  Instead we extract NCBI protein accession numbers from the CARD JSON
  (e.g. ACT97415.1) and pass those to the STRING API, which resolves them
  reliably to STRING protein IDs.

Strategy:
  1. Parse CARD JSON -> extract (aro_id, protein_accession) per gene per organism
  2. Query STRING /get_string_ids with protein accessions -> STRING IDs
  3. Fetch /network interactions between mapped proteins (score >= 400)
  4. Map back to graph node indices via aro_id
  5. Add / overwrite gene_interacts_gene edge type in the graph
"""

import json
import time
import torch
import requests
from pathlib import Path
from collections import defaultdict

DATA_DIR   = Path(__file__).parent.parent / "data"
CARD_JSON  = DATA_DIR / "raw" / "card" / "card.json"
GRAPH_PATH = DATA_DIR / "graphs" / "card_hetero_graph.pt"

STRING_API = "https://string-db.org/api/json"
MIN_SCORE  = 400   # medium confidence (0–1000)
CALLER     = "amr_gnn_research"

# Top organisms to query — taxon_id: display name
TARGET_ORGANISMS = {
    287:   "Pseudomonas aeruginosa",
    470:   "Acinetobacter baumannii",
    573:   "Klebsiella pneumoniae",
    562:   "Escherichia coli",
    550:   "Enterobacter cloacae",
    546:   "Citrobacter freundii",
    83332: "Mycobacterium tuberculosis H37Rv",
    197:   "Campylobacter jejuni",
    571:   "Klebsiella oxytoca",
    584:   "Proteus mirabilis",
}

CHUNK_SIZE = 200  # STRING API handles up to ~2000 but smaller is safer


def load_card_genes_by_organism():
    """Parse CARD JSON -> {taxon_id: [(aro_id, protein_accession), ...]}"""
    print("Loading CARD data...")
    with open(CARD_JSON) as f:
        card = json.load(f)
    for k in ['_version', '_comment', '_timestamp']:
        card.pop(k, None)

    org_genes = defaultdict(list)
    for aro_id, entry in card.items():
        if not isinstance(entry, dict):
            continue
        seqs = entry.get('model_sequences', {}).get('sequence', {})
        if not seqs:
            continue
        for seq in seqs.values():
            taxon_id = seq.get('NCBI_taxonomy', {}).get('NCBI_taxonomy_id', '')
            try:
                taxon_id = int(taxon_id)
            except (ValueError, TypeError):
                continue
            if taxon_id not in TARGET_ORGANISMS:
                break
            prot_acc = seq.get('protein_sequence', {}).get('accession', '').strip()
            if prot_acc and aro_id:
                org_genes[taxon_id].append((aro_id, prot_acc))
            break  # one sequence per CARD entry

    return org_genes


def map_to_string_ids(accessions, taxon_id):
    """
    Map a list of NCBI protein accessions to STRING IDs.
    Returns list of {queryItem, stringId, preferredName, ...} dicts.
    """
    results = []
    for i in range(0, len(accessions), CHUNK_SIZE):
        chunk = accessions[i:i + CHUNK_SIZE]
        identifiers = "\r".join(chunk)
        try:
            r = requests.post(
                f"{STRING_API}/get_string_ids",
                data={
                    "identifiers":   identifiers,
                    "species":       taxon_id,
                    "limit":         1,
                    "echo_query":    1,
                    "caller_identity": CALLER,
                },
                timeout=60,
            )
            r.raise_for_status()
            results.extend(r.json())
        except Exception as e:
            print(f"    Mapping error (chunk {i//CHUNK_SIZE}): {e}")
        time.sleep(0.5)
    return results


def get_interactions(string_ids, taxon_id):
    """Fetch interactions between a set of STRING IDs."""
    results = []
    for i in range(0, len(string_ids), CHUNK_SIZE):
        chunk = string_ids[i:i + CHUNK_SIZE]
        identifiers = "\r".join(chunk)
        try:
            r = requests.post(
                f"{STRING_API}/network",
                data={
                    "identifiers":    identifiers,
                    "species":        taxon_id,
                    "required_score": MIN_SCORE,
                    "caller_identity": CALLER,
                },
                timeout=120,
            )
            r.raise_for_status()
            results.extend(r.json())
        except Exception as e:
            print(f"    Interaction fetch error (chunk {i//CHUNK_SIZE}): {e}")
        time.sleep(0.5)
    return results


def main():
    print("=" * 70)
    print("ADDING STRING PPI EDGES TO CARD HETERO GRAPH  (v2 — accession IDs)")
    print("=" * 70)

    org_genes = load_card_genes_by_organism()
    for taxon_id, org_name in TARGET_ORGANISMS.items():
        print(f"  {len(org_genes[taxon_id]):4d} genes  taxon={taxon_id}  {org_name}")

    print("\nLoading existing graph...")
    graph_data  = torch.load(GRAPH_PATH, weights_only=False)
    node_to_idx = graph_data['node_to_idx']
    edge_type_to_id = graph_data['edge_type_to_id']

    all_ppi_edges   = set()   # set of (aro_id_a, aro_id_b) tuples (sorted)
    total_mapped    = 0
    total_genes_tried = 0

    for taxon_id, org_name in TARGET_ORGANISMS.items():
        genes = org_genes[taxon_id]
        if not genes:
            continue

        print(f"\n[{org_name}]  ({len(genes)} genes)")
        total_genes_tried += len(genes)

        accessions   = [acc  for _, acc  in genes]
        aro_ids      = [aro  for aro, _  in genes]
        acc_to_aro   = {acc: aro for aro, acc in genes}

        # Step 1: Map protein accessions -> STRING IDs
        print(f"  Mapping {len(accessions)} protein accessions -> STRING...")
        hits = map_to_string_ids(accessions, taxon_id)

        acc_to_string = {}
        for hit in hits:
            query = hit.get('queryItem', '').strip()
            sid   = hit.get('stringId', '').strip()
            if query in acc_to_aro and sid:
                acc_to_string[query] = sid

        matched = len(acc_to_string)
        total_mapped += matched
        print(f"  Mapped: {matched}/{len(accessions)}")

        if matched < 2:
            print("  Too few mapped — skipping interaction fetch.")
            continue

        # Build reverse: STRING ID -> aro_id
        string_to_aro = {sid: acc_to_aro[acc] for acc, sid in acc_to_string.items()}
        string_ids    = list(acc_to_string.values())

        # Step 2: Fetch interactions
        print(f"  Fetching PPI interactions (score >= {MIN_SCORE})...")
        interactions = get_interactions(string_ids, taxon_id)

        new_edges = 0
        for ix in interactions:
            sid_a = ix.get('stringId_A', '').strip()
            sid_b = ix.get('stringId_B', '').strip()
            score = float(ix.get('score', 0))
            if sid_a in string_to_aro and sid_b in string_to_aro and score >= MIN_SCORE / 1000:
                aro_a = string_to_aro[sid_a]
                aro_b = string_to_aro[sid_b]
                edge  = tuple(sorted([aro_a, aro_b]))
                if edge not in all_ppi_edges:
                    all_ppi_edges.add(edge)
                    new_edges += 1

        print(f"  New PPI edges this organism: {new_edges}")

    print(f"\n{'='*70}")
    print(f"TOTAL: {total_mapped}/{total_genes_tried} genes mapped")
    print(f"TOTAL unique PPI edges: {len(all_ppi_edges)}")

    if not all_ppi_edges:
        print("No PPI edges found — graph unchanged.")
        return

    # Convert ARO ID pairs -> node index pairs
    print("\nConverting ARO ID pairs -> node indices...")
    ppi_edge_list = []
    skipped = 0
    for aro_a, aro_b in all_ppi_edges:
        key_a = ('gene', aro_a)
        key_b = ('gene', aro_b)
        if key_a in node_to_idx and key_b in node_to_idx:
            ia = node_to_idx[key_a]
            ib = node_to_idx[key_b]
            ppi_edge_list.append([ia, ib])
            ppi_edge_list.append([ib, ia])  # undirected
        else:
            skipped += 1

    print(f"  Edges converted: {len(ppi_edge_list)//2} unique  |  skipped: {skipped}")

    if not ppi_edge_list:
        print("Could not map any PPI edges to graph nodes — graph unchanged.")
        return

    ppi_tensor = torch.tensor(ppi_edge_list, dtype=torch.long).t()

    # Overwrite / add edge type
    graph_data['typed_edge_indices']['gene_interacts_gene']       = ppi_tensor
    graph_data['train_typed_edge_indices']['gene_interacts_gene'] = ppi_tensor
    if 'gene_interacts_gene' not in edge_type_to_id:
        edge_type_to_id['gene_interacts_gene'] = max(edge_type_to_id.values()) + 1
        graph_data['edge_type_to_id'] = edge_type_to_id

    torch.save(graph_data, GRAPH_PATH)
    print(f"\nSaved updated graph -> {GRAPH_PATH}")

    summary_path = DATA_DIR / "graphs" / "hetero_graph_summary.txt"
    with open(summary_path, 'a') as f:
        f.write(f"\nSTRING PPI Edges (v2 — protein accession mapping):\n")
        f.write(f"  - gene_interacts_gene: {len(ppi_edge_list)//2} unique edges\n")
        f.write(f"  - Organisms queried: {len(TARGET_ORGANISMS)}\n")
        f.write(f"  - Genes mapped to STRING: {total_mapped}/{total_genes_tried}\n")
        f.write(f"  - Min confidence score: {MIN_SCORE}\n")

    print(f"\ngene_interacts_gene: {len(ppi_edge_list)//2} edges  "
          f"({len(ppi_edge_list)} directed)")
    print("Done!")


if __name__ == "__main__":
    main()
