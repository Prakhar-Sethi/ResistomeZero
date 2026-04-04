"""
Build graph representations from CARD database for link prediction.

Creates a heterogeneous knowledge graph with:
- Nodes: Resistance genes, antibiotics, drug classes, resistance mechanisms
- Edges: Various relationships (confers resistance, has mechanism, etc.)
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from typing import Dict, List, Tuple, Set
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
CARD_JSON = DATA_DIR / "raw" / "card" / "card.json"
GRAPH_DIR = DATA_DIR / "graphs"
GRAPH_DIR.mkdir(exist_ok=True)

print("="*80)
print("BUILDING KNOWLEDGE GRAPH FROM CARD DATABASE")
print("="*80)

# Load CARD data
print("\n[1/6] Loading CARD database...")
with open(CARD_JSON, 'r') as f:
    card_data = json.load(f)

# Remove metadata
for field in ['_version', '_comment', '_timestamp']:
    card_data.pop(field, None)

print(f"   Loaded {len(card_data)} entries")

# ============================================================================
# ENTITY EXTRACTION
# ============================================================================
print("\n[2/6] Extracting entities...")

# Entity collections
resistance_genes = {}  # ARO_id -> info
antibiotics = {}       # name -> info
drug_classes = {}      # name -> info
mechanisms = {}        # name -> info
gene_families = {}     # name -> info

# Track all unique entities
antibiotic_names = set()
drug_class_names = set()
mechanism_names = set()
gene_family_names = set()

for aro_id, entry in card_data.items():
    # Extract resistance gene info
    resistance_genes[aro_id] = {
        'aro_id': aro_id,
        'name': entry.get('ARO_name', ''),
        'accession': entry.get('ARO_accession', ''),
        'description': entry.get('ARO_description', '')[:200],
        'model_type': entry.get('model_type', '')
    }

    # Extract categories
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

# Create entity dictionaries
for name in antibiotic_names:
    antibiotics[name] = {'name': name, 'type': 'antibiotic'}

for name in drug_class_names:
    drug_classes[name] = {'name': name, 'type': 'drug_class'}

for name in mechanism_names:
    mechanisms[name] = {'name': name, 'type': 'mechanism'}

for name in gene_family_names:
    gene_families[name] = {'name': name, 'type': 'gene_family'}

print(f"   Resistance genes: {len(resistance_genes)}")
print(f"   Antibiotics: {len(antibiotics)}")
print(f"   Drug classes: {len(drug_classes)}")
print(f"   Resistance mechanisms: {len(mechanisms)}")
print(f"   Gene families: {len(gene_families)}")

# ============================================================================
# RELATIONSHIP EXTRACTION
# ============================================================================
print("\n[3/6] Extracting relationships...")

# Relationship collections (head, relation, tail)
relationships = {
    'gene_resists_antibiotic': [],
    'gene_has_mechanism': [],
    'gene_in_family': [],
    'antibiotic_in_class': [],
}

# Map antibiotics to their classes (for antibiotic_in_class edges)
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
            # Track which antibiotics belong to this class
            for ab in gene_antibiotics:
                antibiotic_to_class[ab].add(cat_name)
        elif cat_class == 'Resistance Mechanism':
            gene_mechanisms.add(cat_name)
        elif cat_class == 'AMR Gene Family':
            gene_families_for_gene.add(cat_name)

    # Create relationships
    for ab in gene_antibiotics:
        relationships['gene_resists_antibiotic'].append((aro_id, ab))

    for mech in gene_mechanisms:
        relationships['gene_has_mechanism'].append((aro_id, mech))

    for fam in gene_families_for_gene:
        relationships['gene_in_family'].append((aro_id, fam))

# Add antibiotic-class relationships
for ab, classes in antibiotic_to_class.items():
    for drug_class in classes:
        relationships['antibiotic_in_class'].append((ab, drug_class))

print(f"   Gene-Antibiotic edges: {len(relationships['gene_resists_antibiotic'])}")
print(f"   Gene-Mechanism edges: {len(relationships['gene_has_mechanism'])}")
print(f"   Gene-Family edges: {len(relationships['gene_in_family'])}")
print(f"   Antibiotic-Class edges: {len(relationships['antibiotic_in_class'])}")

# ============================================================================
# BUILD HOMOGENEOUS GRAPH (for standard GNN models)
# ============================================================================
print("\n[4/6] Building homogeneous graph...")

# Create unified node index
node_to_idx = {}
idx_to_node = {}
node_types = {}
current_idx = 0

# Add all nodes
for aro_id in resistance_genes:
    node_to_idx[('gene', aro_id)] = current_idx
    idx_to_node[current_idx] = ('gene', aro_id)
    node_types[current_idx] = 'gene'
    current_idx += 1

for ab_name in antibiotics:
    node_to_idx[('antibiotic', ab_name)] = current_idx
    idx_to_node[current_idx] = ('antibiotic', ab_name)
    node_types[current_idx] = 'antibiotic'
    current_idx += 1

for dc_name in drug_classes:
    node_to_idx[('drug_class', dc_name)] = current_idx
    idx_to_node[current_idx] = ('drug_class', dc_name)
    node_types[current_idx] = 'drug_class'
    current_idx += 1

for mech_name in mechanisms:
    node_to_idx[('mechanism', mech_name)] = current_idx
    idx_to_node[current_idx] = ('mechanism', mech_name)
    node_types[current_idx] = 'mechanism'
    current_idx += 1

for fam_name in gene_families:
    node_to_idx[('gene_family', fam_name)] = current_idx
    idx_to_node[current_idx] = ('gene_family', fam_name)
    node_types[current_idx] = 'gene_family'
    current_idx += 1

total_nodes = current_idx
print(f"   Total nodes: {total_nodes}")

# Create edge index (all edges in both directions for undirected graph)
edge_list = []

for gene_id, ab_name in relationships['gene_resists_antibiotic']:
    src = node_to_idx[('gene', gene_id)]
    dst = node_to_idx[('antibiotic', ab_name)]
    edge_list.append((src, dst))
    edge_list.append((dst, src))  # Undirected

for gene_id, mech_name in relationships['gene_has_mechanism']:
    src = node_to_idx[('gene', gene_id)]
    dst = node_to_idx[('mechanism', mech_name)]
    edge_list.append((src, dst))
    edge_list.append((dst, src))

for gene_id, fam_name in relationships['gene_in_family']:
    src = node_to_idx[('gene', gene_id)]
    dst = node_to_idx[('gene_family', fam_name)]
    edge_list.append((src, dst))
    edge_list.append((dst, src))

for ab_name, dc_name in relationships['antibiotic_in_class']:
    src = node_to_idx[('antibiotic', ab_name)]
    dst = node_to_idx[('drug_class', dc_name)]
    edge_list.append((src, dst))
    edge_list.append((dst, src))

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
print(f"   Total edges: {edge_index.shape[1]}")

# ============================================================================
# CREATE TRAIN/VAL/TEST SPLITS FOR LINK PREDICTION
# ============================================================================
print("\n[5/6] Creating train/val/test splits...")

# We'll predict gene-antibiotic links (the main task)
all_pos_edges = relationships['gene_resists_antibiotic']
print(f"   Positive edges (gene-antibiotic): {len(all_pos_edges)}")

# Convert to indices
pos_edge_index_list = []
for gene_id, ab_name in all_pos_edges:
    src = node_to_idx[('gene', gene_id)]
    dst = node_to_idx[('antibiotic', ab_name)]
    pos_edge_index_list.append([src, dst])

pos_edge_index_np = np.array(pos_edge_index_list).T

# Split: 80% train, 10% val, 10% test
num_edges = pos_edge_index_np.shape[1]
indices = np.arange(num_edges)

train_idx, test_val_idx = train_test_split(indices, test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(test_val_idx, test_size=0.5, random_state=42)

train_pos_edge_index = torch.tensor(pos_edge_index_np[:, train_idx], dtype=torch.long)
val_pos_edge_index = torch.tensor(pos_edge_index_np[:, val_idx], dtype=torch.long)
test_pos_edge_index = torch.tensor(pos_edge_index_np[:, test_idx], dtype=torch.long)

print(f"   Train edges: {train_pos_edge_index.shape[1]}")
print(f"   Val edges: {val_pos_edge_index.shape[1]}")
print(f"   Test edges: {test_pos_edge_index.shape[1]}")

# For message passing during training, include all OTHER edges (not gene-antibiotic)
# This simulates the scenario where we know structure but want to predict specific links
train_edge_index_list = []

# Add all non-gene-antibiotic edges to train graph
for gene_id, mech_name in relationships['gene_has_mechanism']:
    src = node_to_idx[('gene', gene_id)]
    dst = node_to_idx[('mechanism', mech_name)]
    train_edge_index_list.append([src, dst])
    train_edge_index_list.append([dst, src])

for gene_id, fam_name in relationships['gene_in_family']:
    src = node_to_idx[('gene', gene_id)]
    dst = node_to_idx[('gene_family', fam_name)]
    train_edge_index_list.append([src, dst])
    train_edge_index_list.append([dst, src])

for ab_name, dc_name in relationships['antibiotic_in_class']:
    src = node_to_idx[('antibiotic', ab_name)]
    dst = node_to_idx[('drug_class', dc_name)]
    train_edge_index_list.append([src, dst])
    train_edge_index_list.append([dst, src])

# Add training gene-antibiotic edges
for i in range(train_pos_edge_index.shape[1]):
    src = train_pos_edge_index[0, i].item()
    dst = train_pos_edge_index[1, i].item()
    train_edge_index_list.append([src, dst])
    train_edge_index_list.append([dst, src])

train_edge_index_for_mp = torch.tensor(train_edge_index_list, dtype=torch.long).t()
print(f"   Train graph edges (for message passing): {train_edge_index_for_mp.shape[1]}")

# ============================================================================
# SAVE GRAPH DATA
# ============================================================================
print("\n[6/6] Saving graph data...")

# Create node features (one-hot encoding of node types for now)
node_type_to_id = {
    'gene': 0,
    'antibiotic': 1,
    'drug_class': 2,
    'mechanism': 3,
    'gene_family': 4
}

node_features = torch.zeros(total_nodes, 5)  # 5 node types
for idx in range(total_nodes):
    node_type = node_types[idx]
    node_features[idx, node_type_to_id[node_type]] = 1.0

graph_data = {
    'num_nodes': total_nodes,
    'node_features': node_features,
    'node_to_idx': node_to_idx,
    'idx_to_node': idx_to_node,
    'node_types': node_types,

    # Full graph (for visualization/analysis)
    'edge_index': edge_index,

    # Train/val/test splits for link prediction
    'train_pos_edge_index': train_pos_edge_index,
    'val_pos_edge_index': val_pos_edge_index,
    'test_pos_edge_index': test_pos_edge_index,

    # Train graph for message passing (excludes val/test edges)
    'train_edge_index': train_edge_index_for_mp,

    # Metadata
    'num_genes': len(resistance_genes),
    'num_antibiotics': len(antibiotics),
    'num_drug_classes': len(drug_classes),
    'num_mechanisms': len(mechanisms),
    'num_gene_families': len(gene_families),

    # Entity info
    'resistance_genes': resistance_genes,
    'antibiotics': antibiotics,
    'drug_classes': drug_classes,
    'mechanisms': mechanisms,
    'gene_families': gene_families,
}

# Save as PyTorch file
graph_path = GRAPH_DIR / "card_graph.pt"
torch.save(graph_data, graph_path)
print(f"   Saved to: {graph_path}")

# Save summary
summary_path = GRAPH_DIR / "graph_summary.txt"
with open(summary_path, 'w') as f:
    f.write("CARD Knowledge Graph Summary\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total nodes: {total_nodes}\n")
    f.write(f"  - Resistance genes: {len(resistance_genes)}\n")
    f.write(f"  - Antibiotics: {len(antibiotics)}\n")
    f.write(f"  - Drug classes: {len(drug_classes)}\n")
    f.write(f"  - Resistance mechanisms: {len(mechanisms)}\n")
    f.write(f"  - Gene families: {len(gene_families)}\n\n")
    f.write(f"Total edges: {edge_index.shape[1]}\n\n")
    f.write("Link Prediction Task (Gene-Antibiotic)\n")
    f.write(f"  - Train edges: {train_pos_edge_index.shape[1]}\n")
    f.write(f"  - Val edges: {val_pos_edge_index.shape[1]}\n")
    f.write(f"  - Test edges: {test_pos_edge_index.shape[1]}\n")

print(f"   Summary saved to: {summary_path}")

print("\n" + "="*80)
print("GRAPH CONSTRUCTION COMPLETE!")
print("="*80)
print(f"\nLoad the graph with:")
print(f"  graph_data = torch.load('{graph_path}')")
