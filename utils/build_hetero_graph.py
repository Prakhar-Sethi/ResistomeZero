"""
Build HETEROGENEOUS graph from CARD with typed edges.

This is the NOVEL contribution - instead of treating all edges the same,
we preserve the semantic relationships in the graph structure.

Edge Types:
1. gene --[confers_resistance_to]--> antibiotic
2. gene --[has_mechanism]--> resistance_mechanism
3. gene --[belongs_to_family]--> gene_family
4. antibiotic --[member_of_class]--> drug_class
"""

import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
CARD_JSON = DATA_DIR / "raw" / "card" / "card.json"
GRAPH_DIR = DATA_DIR / "graphs"
GRAPH_DIR.mkdir(exist_ok=True)

print("="*80)
print("BUILDING HETEROGENEOUS KNOWLEDGE GRAPH FROM CARD")
print("="*80)

# Load CARD
print("\n[1/7] Loading CARD database...")
with open(CARD_JSON, 'r') as f:
    card_data = json.load(f)

for field in ['_version', '_comment', '_timestamp']:
    card_data.pop(field, None)

print(f"   Loaded {len(card_data)} entries")

# ============================================================================
# ENTITY EXTRACTION (same as before)
# ============================================================================
print("\n[2/7] Extracting entities...")

resistance_genes = {}
antibiotics = {}
drug_classes = {}
mechanisms = {}
gene_families = {}

antibiotic_names = set()
drug_class_names = set()
mechanism_names = set()
gene_family_names = set()

for aro_id, entry in card_data.items():
    resistance_genes[aro_id] = {
        'aro_id': aro_id,
        'name': entry.get('ARO_name', ''),
        'accession': entry.get('ARO_accession', ''),
        'description': entry.get('ARO_description', '')[:200],
        'model_type': entry.get('model_type', '')
    }

    aro_categories = entry.get('ARO_category', {})
    for cat_id, cat_info in aro_categories.items():
        cat_class = cat_info.get('category_aro_class_name', '')
        cat_name = cat_info.get('category_aro_name', '')

        if cat_class == 'Antibiotic':
            antibiotic_names.add(cat_name)
        elif cat_class == 'Drug Class':
            drug_class_names.add(cat_name)
        elif cat_class == 'Resistance Mechanism':
            mechanism_names.add(cat_name)
        elif cat_class == 'AMR Gene Family':
            gene_family_names.add(cat_name)

for name in antibiotic_names:
    antibiotics[name] = {'name': name}
for name in drug_class_names:
    drug_classes[name] = {'name': name}
for name in mechanism_names:
    mechanisms[name] = {'name': name}
for name in gene_family_names:
    gene_families[name] = {'name': name}

print(f"   Genes: {len(resistance_genes)}")
print(f"   Antibiotics: {len(antibiotics)}")
print(f"   Drug classes: {len(drug_classes)}")
print(f"   Mechanisms: {len(mechanisms)}")
print(f"   Gene families: {len(gene_families)}")

# ============================================================================
# HETEROGENEOUS RELATIONSHIP EXTRACTION
# ============================================================================
print("\n[3/7] Extracting typed relationships...")

# Store relationships by type
edge_types = {
    'gene_to_antibiotic': [],      # Type 0
    'gene_to_mechanism': [],        # Type 1
    'gene_to_family': [],           # Type 2
    'antibiotic_to_class': [],      # Type 3
}

antibiotic_to_class = defaultdict(set)

for aro_id, entry in card_data.items():
    aro_categories = entry.get('ARO_category', {})

    gene_antibiotics = set()
    gene_mechanisms = set()
    gene_families_for_gene = set()

    for cat_id, cat_info in aro_categories.items():
        cat_class = cat_info.get('category_aro_class_name', '')
        cat_name = cat_info.get('category_aro_name', '')

        if cat_class == 'Antibiotic':
            gene_antibiotics.add(cat_name)
        elif cat_class == 'Drug Class':
            for ab in gene_antibiotics:
                antibiotic_to_class[ab].add(cat_name)
        elif cat_class == 'Resistance Mechanism':
            gene_mechanisms.add(cat_name)
        elif cat_class == 'AMR Gene Family':
            gene_families_for_gene.add(cat_name)

    for ab in gene_antibiotics:
        edge_types['gene_to_antibiotic'].append((aro_id, ab))
    for mech in gene_mechanisms:
        edge_types['gene_to_mechanism'].append((aro_id, mech))
    for fam in gene_families_for_gene:
        edge_types['gene_to_family'].append((aro_id, fam))

for ab, classes in antibiotic_to_class.items():
    for dc in classes:
        edge_types['antibiotic_to_class'].append((ab, dc))

for edge_type, edges in edge_types.items():
    print(f"   {edge_type}: {len(edges)}")

# ============================================================================
# BUILD NODE INDEX (unified across all types)
# ============================================================================
print("\n[4/7] Building unified node index...")

node_to_idx = {}
idx_to_node = {}
node_type_map = {}
current_idx = 0

# Add all nodes with type tracking
for aro_id in resistance_genes:
    node_to_idx[('gene', aro_id)] = current_idx
    idx_to_node[current_idx] = ('gene', aro_id)
    node_type_map[current_idx] = 0  # gene type
    current_idx += 1

for ab_name in antibiotics:
    node_to_idx[('antibiotic', ab_name)] = current_idx
    idx_to_node[current_idx] = ('antibiotic', ab_name)
    node_type_map[current_idx] = 1  # antibiotic type
    current_idx += 1

for dc_name in drug_classes:
    node_to_idx[('drug_class', dc_name)] = current_idx
    idx_to_node[current_idx] = ('drug_class', dc_name)
    node_type_map[current_idx] = 2  # drug_class type
    current_idx += 1

for mech_name in mechanisms:
    node_to_idx[('mechanism', mech_name)] = current_idx
    idx_to_node[current_idx] = ('mechanism', mech_name)
    node_type_map[current_idx] = 3  # mechanism type
    current_idx += 1

for fam_name in gene_families:
    node_to_idx[('gene_family', fam_name)] = current_idx
    idx_to_node[current_idx] = ('gene_family', fam_name)
    node_type_map[current_idx] = 4  # gene_family type
    current_idx += 1

total_nodes = current_idx
print(f"   Total nodes: {total_nodes}")

# ============================================================================
# CREATE TYPED EDGE INDICES
# ============================================================================
print("\n[5/7] Creating typed edge indices...")

# Map edge type names to IDs
edge_type_to_id = {
    'gene_to_antibiotic': 0,
    'gene_to_mechanism': 1,
    'gene_to_family': 2,
    'antibiotic_to_class': 3,
}

# Create edge index tensors for each type (undirected)
typed_edge_indices = {}
edge_type_list = []  # For each edge, store its type

for edge_type_name, edge_list in edge_types.items():
    edges = []
    type_id = edge_type_to_id[edge_type_name]

    for src_key, dst_key in edge_list:
        # Determine src and dst node types
        if edge_type_name == 'gene_to_antibiotic':
            src = node_to_idx[('gene', src_key)]
            dst = node_to_idx[('antibiotic', dst_key)]
        elif edge_type_name == 'gene_to_mechanism':
            src = node_to_idx[('gene', src_key)]
            dst = node_to_idx[('mechanism', dst_key)]
        elif edge_type_name == 'gene_to_family':
            src = node_to_idx[('gene', src_key)]
            dst = node_to_idx[('gene_family', dst_key)]
        elif edge_type_name == 'antibiotic_to_class':
            src = node_to_idx[('antibiotic', src_key)]
            dst = node_to_idx[('drug_class', dst_key)]

        # Add both directions (undirected graph)
        edges.append([src, dst])
        edges.append([dst, src])

    typed_edge_indices[edge_type_name] = torch.tensor(edges, dtype=torch.long).t()
    print(f"   {edge_type_name}: {typed_edge_indices[edge_type_name].shape[1]} directed edges")

# ============================================================================
# CREATE ZERO-SHOT SPLIT (Novel Task!)
# ============================================================================
print("\n[6/7] Creating ZERO-SHOT split (unseen antibiotics)...")

# Split antibiotics into seen/unseen
antibiotic_list = list(antibiotics.keys())
train_antibiotics, test_antibiotics = train_test_split(
    antibiotic_list, test_size=0.2, random_state=42
)
train_antibiotics, val_antibiotics = train_test_split(
    train_antibiotics, test_size=0.125, random_state=42  # 0.125 * 0.8 = 0.1
)

train_ab_set = set(train_antibiotics)
val_ab_set = set(val_antibiotics)
test_ab_set = set(test_antibiotics)

print(f"   Train antibiotics: {len(train_antibiotics)}")
print(f"   Val antibiotics: {len(val_antibiotics)}")
print(f"   Test antibiotics: {len(test_antibiotics)}")

# Split gene-antibiotic edges based on antibiotic
train_edges = []
val_edges = []
test_edges = []

for gene_id, ab_name in edge_types['gene_to_antibiotic']:
    src = node_to_idx[('gene', gene_id)]
    dst = node_to_idx[('antibiotic', ab_name)]

    if ab_name in train_ab_set:
        train_edges.append([src, dst])
    elif ab_name in val_ab_set:
        val_edges.append([src, dst])
    elif ab_name in test_ab_set:
        test_edges.append([src, dst])

train_pos_edges = torch.tensor(train_edges, dtype=torch.long).t()
val_pos_edges = torch.tensor(val_edges, dtype=torch.long).t()
test_pos_edges = torch.tensor(test_edges, dtype=torch.long).t()

print(f"   Train edges: {train_pos_edges.shape[1]}")
print(f"   Val edges: {val_pos_edges.shape[1]}")
print(f"   Test edges: {test_pos_edges.shape[1]}")

# Build training graph (exclude val/test gene-antibiotic edges)
train_graph_edges = {}
for edge_type_name in edge_types.keys():
    if edge_type_name == 'gene_to_antibiotic':
        # Only include train edges
        edges = []
        for i in range(train_pos_edges.shape[1]):
            src = train_pos_edges[0, i].item()
            dst = train_pos_edges[1, i].item()
            edges.append([src, dst])
            edges.append([dst, src])
        train_graph_edges[edge_type_name] = torch.tensor(edges, dtype=torch.long).t()
    else:
        # Include all edges of other types
        train_graph_edges[edge_type_name] = typed_edge_indices[edge_type_name]

# ============================================================================
# SAVE HETEROGENEOUS GRAPH
# ============================================================================
print("\n[7/7] Saving heterogeneous graph...")

# Node type one-hot (fallback / type indicator)
node_type_onehot = torch.zeros(total_nodes, 5)
for idx in range(total_nodes):
    node_type_onehot[idx, node_type_map[idx]] = 1.0

# Load rich per-type features if available
FEATURES_DIR = DATA_DIR / "features"
gene_rich_features = None
antibiotic_rich_features = None

gene_feat_path = FEATURES_DIR / "gene_esm2_features.pt"
ab_feat_path = FEATURES_DIR / "antibiotic_morgan_features.pt"

if gene_feat_path.exists() and ab_feat_path.exists():
    print("   Loading rich node features (ESM-2 + Morgan fingerprints)...")
    gene_feat_data = torch.load(gene_feat_path)
    ab_feat_data = torch.load(ab_feat_path)

    # Build per-type feature tensors aligned to our node ordering
    gene_features_raw = torch.zeros(len(resistance_genes), gene_feat_data['dim'])
    gene_aro_order = gene_feat_data['gene_order']
    for i, aro_id in enumerate(gene_aro_order):
        if aro_id in resistance_genes:
            node_idx_in_gene = list(resistance_genes.keys()).index(aro_id)
            gene_features_raw[node_idx_in_gene] = gene_feat_data['features'][i]

    ab_features_raw = torch.zeros(len(antibiotics), ab_feat_data['dim'])
    ab_name_order = ab_feat_data['antibiotic_order']
    ab_name_to_i = {name: i for i, name in enumerate(ab_name_order)}
    ab_list = list(antibiotics.keys())
    for j, ab_name in enumerate(ab_list):
        if ab_name in ab_name_to_i:
            ab_features_raw[j] = ab_feat_data['features'][ab_name_to_i[ab_name]]

    gene_rich_features = gene_features_raw
    antibiotic_rich_features = ab_features_raw
    print(f"   Gene features: {gene_rich_features.shape} (ESM-2-8M)")
    print(f"   Antibiotic features: {antibiotic_rich_features.shape} (Morgan ECFP4)")
else:
    print("   Rich features not found — using one-hot type encoding only.")
    print("   Run: python utils/compute_node_features.py")

# Unified node features: one-hot type (always present, used as fallback)
node_features = node_type_onehot

graph_data = {
    # Basic info
    'num_nodes': total_nodes,
    'node_features': node_features,  # one-hot type (5-dim), fallback
    'gene_rich_features': gene_rich_features,        # ESM-2 (320-dim) or None
    'antibiotic_rich_features': antibiotic_rich_features,  # Morgan (1024-dim) or None
    'node_to_idx': node_to_idx,
    'idx_to_node': idx_to_node,
    'node_type_map': node_type_map,

    # Heterogeneous edges (full graph)
    'typed_edge_indices': typed_edge_indices,
    'edge_type_to_id': edge_type_to_id,

    # Training graph (excludes val/test gene-antibiotic edges)
    'train_typed_edge_indices': train_graph_edges,

    # Zero-shot splits
    'train_pos_edges': train_pos_edges,
    'val_pos_edges': val_pos_edges,
    'test_pos_edges': test_pos_edges,

    # Antibiotic splits (for zero-shot evaluation)
    'train_antibiotics': train_antibiotics,
    'val_antibiotics': val_antibiotics,
    'test_antibiotics': test_antibiotics,

    # Entity info
    'resistance_genes': resistance_genes,
    'antibiotics': antibiotics,
    'drug_classes': drug_classes,
    'mechanisms': mechanisms,
    'gene_families': gene_families,

    # Metadata
    'num_genes': len(resistance_genes),
    'num_antibiotics': len(antibiotics),
    'num_drug_classes': len(drug_classes),
    'num_mechanisms': len(mechanisms),
    'num_gene_families': len(gene_families),
}

# Save
graph_path = GRAPH_DIR / "card_hetero_graph.pt"
torch.save(graph_data, graph_path)
print(f"   Saved to: {graph_path}")

# Summary
summary_path = GRAPH_DIR / "hetero_graph_summary.txt"
with open(summary_path, 'w') as f:
    f.write("CARD Heterogeneous Knowledge Graph Summary\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total nodes: {total_nodes}\n")
    f.write(f"  - Genes: {len(resistance_genes)}\n")
    f.write(f"  - Antibiotics: {len(antibiotics)}\n")
    f.write(f"  - Drug classes: {len(drug_classes)}\n")
    f.write(f"  - Mechanisms: {len(mechanisms)}\n")
    f.write(f"  - Gene families: {len(gene_families)}\n\n")

    f.write("Edge Types:\n")
    for edge_type, edge_idx in typed_edge_indices.items():
        f.write(f"  - {edge_type}: {edge_idx.shape[1]} edges\n")

    f.write(f"\nZero-Shot Task (Predict resistance to unseen antibiotics):\n")
    f.write(f"  - Train antibiotics: {len(train_antibiotics)}\n")
    f.write(f"  - Val antibiotics: {len(val_antibiotics)}\n")
    f.write(f"  - Test antibiotics: {len(test_antibiotics)}\n")
    f.write(f"  - Train edges: {train_pos_edges.shape[1]}\n")
    f.write(f"  - Val edges: {val_pos_edges.shape[1]}\n")
    f.write(f"  - Test edges: {test_pos_edges.shape[1]}\n")

print(f"   Summary: {summary_path}")

print("\n" + "="*80)
print("HETEROGENEOUS GRAPH COMPLETE!")
print("="*80)
print("\nKEY FEATURES:")
print("  ✓ Typed edges (4 relation types)")
print("  ✓ Zero-shot task (unseen antibiotics)")
print("  ✓ Preserves biological semantics")
print("  ✓ Ready for R-GCN and KG embedding models")
print("="*80)
