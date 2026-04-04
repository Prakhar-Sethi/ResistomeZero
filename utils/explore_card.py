"""
Explore CARD database structure to understand what entities and relationships
we can extract for graph construction.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

# Paths
CARD_DIR = Path(__file__).parent.parent / "data" / "raw" / "card"
CARD_JSON = CARD_DIR / "card.json"

print("Loading CARD database...")
with open(CARD_JSON, 'r') as f:
    card_data = json.load(f)

# Remove metadata fields
metadata_fields = ['_version', '_comment', '_timestamp']
for field in metadata_fields:
    card_data.pop(field, None)

print(f"Total entries: {len(card_data)}")
print("\n" + "="*80)

# Sample a few entries to understand structure
sample_ids = list(card_data.keys())[:3]
print(f"\nSample entry IDs: {sample_ids}")
print("\n" + "="*80)

# Examine first entry in detail
first_entry = card_data[sample_ids[0]]
print(f"\nFirst entry structure (ID: {sample_ids[0]}):")
print(f"Top-level keys: {list(first_entry.keys())}")

# Analyze all entries
print("\n" + "="*80)
print("\nAnalyzing all entries...")

# Collect statistics
model_types = Counter()
aro_categories = []
drug_classes = []
resistance_mechanisms = []
organisms = []
genes = []

for aro_id, entry in card_data.items():
    # Model type
    model_type = entry.get('model_type')
    model_types[model_type] += 1

    # ARO category
    aro_cat = entry.get('ARO_category')
    if aro_cat:
        if isinstance(aro_cat, dict):
            for cat_id, cat_info in aro_cat.items():
                cat_name = cat_info.get('category_aro_name', '')
                if cat_name:
                    aro_categories.append(cat_name)

    # Check for other relevant fields
    model_sequences = entry.get('model_sequences')
    if model_sequences:
        for seq_id, seq_info in model_sequences.items():
            # Extract gene/protein names
            if 'protein_sequence' in seq_info:
                gene_name = seq_info.get('NCBI_taxonomy', {}).get('NCBI_taxonomy_name', '')
                if gene_name:
                    organisms.append(gene_name)

# Print statistics
print("\n" + "="*80)
print("\nMODEL TYPES:")
for model_type, count in model_types.most_common():
    print(f"  {model_type}: {count}")

print("\n" + "="*80)
print(f"\nARO CATEGORIES (Top 20):")
category_counts = Counter(aro_categories)
for cat, count in category_counts.most_common(20):
    print(f"  {cat}: {count}")

print("\n" + "="*80)
print(f"\nUnique organisms found: {len(set(organisms))}")
if organisms:
    org_counts = Counter(organisms)
    print("\nTop 10 organisms:")
    for org, count in org_counts.most_common(10):
        print(f"  {org}: {count}")

# Detailed examination of one protein homolog entry
print("\n" + "="*80)
print("\nDetailed example of a protein_homolog_model entry:")
for aro_id, entry in card_data.items():
    if entry.get('model_type') == 'protein homolog model':
        print(f"\nARO ID: {aro_id}")
        print(f"Model name: {entry.get('model_name')}")
        print(f"Model ID: {entry.get('model_id')}")
        print(f"Model type: {entry.get('model_type')}")

        # ARO category
        if 'ARO_category' in entry:
            print("\nARO Categories:")
            for cat_id, cat_info in entry['ARO_category'].items():
                print(f"  - {cat_info.get('category_aro_class_name')}: {cat_info.get('category_aro_name')}")

        # ARO accession
        if 'ARO_accession' in entry:
            print(f"\nARO Accession: {entry['ARO_accession']}")

        # ARO name
        if 'ARO_name' in entry:
            print(f"ARO Name: {entry['ARO_name']}")

        # ARO description
        if 'ARO_description' in entry:
            desc = entry['ARO_description'][:200] + "..." if len(entry['ARO_description']) > 200 else entry['ARO_description']
            print(f"Description: {desc}")

        break

# Save summary
print("\n" + "="*80)
summary_path = CARD_DIR / "exploration_summary.txt"
with open(summary_path, 'w') as f:
    f.write("CARD Database Exploration Summary\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total entries: {len(card_data)}\n\n")

    f.write("Model Types:\n")
    for model_type, count in model_types.most_common():
        f.write(f"  {model_type}: {count}\n")

    f.write(f"\n\nARO Categories (Top 50):\n")
    for cat, count in category_counts.most_common(50):
        f.write(f"  {cat}: {count}\n")

    f.write(f"\n\nUnique organisms: {len(set(organisms))}\n")

print(f"\nSummary saved to {summary_path}")

# Analyze what types of relationships we can extract
print("\n" + "="*80)
print("\nPOTENTIAL GRAPH ENTITIES:")
print("1. ARO Terms (resistance genes/mechanisms)")
print("2. Drug Classes (from ARO categories)")
print("3. Resistance Mechanisms (from ARO categories)")
print("4. Organisms (bacteria)")
print("5. Antibiotics (from categories)")

print("\nPOTENTIAL RELATIONSHIPS:")
print("1. ARO_term -> confers_resistance_to -> Antibiotic")
print("2. ARO_term -> has_mechanism -> Resistance_Mechanism")
print("3. ARO_term -> found_in -> Organism")
print("4. ARO_term -> belongs_to_class -> Drug_Class")
print("5. ARO_term -> part_of_category -> ARO_Category")

print("\n" + "="*80)
print("Exploration complete!")
