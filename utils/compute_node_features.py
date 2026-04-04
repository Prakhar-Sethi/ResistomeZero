"""
Compute rich node features for the AMR knowledge graph.

Gene nodes:       ESM-2 protein language model embeddings (320-dim)
Antibiotic nodes: Morgan molecular fingerprints from PubChem SMILES (1024-dim)
Other nodes:      one-hot type encoding (few nodes, kept simple)

Run this once before rebuilding the graph:
    python utils/compute_node_features.py
"""

import json
import torch
import numpy as np
import requests
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
CARD_JSON = DATA_DIR / "raw" / "card" / "card.json"
FEATURES_DIR = DATA_DIR / "features"
FEATURES_DIR.mkdir(exist_ok=True)

print("Loading CARD database...")
with open(CARD_JSON, 'r') as f:
    card_data = json.load(f)
for field in ['_version', '_comment', '_timestamp']:
    card_data.pop(field, None)
print(f"  {len(card_data)} entries loaded")

# ============================================================================
# PART 1: ESM-2 PROTEIN EMBEDDINGS FOR GENE NODES
# ============================================================================
print("\n[1/2] Computing ESM-2 embeddings for gene nodes...")

gene_feat_path = FEATURES_DIR / "gene_esm2_features.pt"
if gene_feat_path.exists():
    print("  Already computed — skipping.")
    gene_features = torch.load(gene_feat_path)['features']
else:
    import esm
    print("  Loading ESM-2-8M model (320-dim, fast)...")
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()

    gene_order = list(card_data.keys())
    sequences = []
    for aro_id in gene_order:
        entry = card_data[aro_id]
        seq = ""
        for seq_data in entry.get('model_sequences', {}).get('sequence', {}).values():
            seq = seq_data.get('protein_sequence', {}).get('sequence', '')
            if seq:
                break
        sequences.append(seq)

    print(f"  {sum(1 for s in sequences if s)} / {len(sequences)} genes have protein sequences")
    MAX_LEN = 1022
    n_truncated = sum(1 for s in sequences if len(s) > MAX_LEN)
    if n_truncated:
        print(f"  Truncating {n_truncated} sequences longer than {MAX_LEN}aa")
    sequences = [s[:MAX_LEN] for s in sequences]

    BATCH_SIZE = 32
    EMB_DIM = 320
    gene_features = torch.zeros(len(gene_order), EMB_DIM)
    n_batches = (len(gene_order) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Running inference: {len(gene_order)} sequences, {n_batches} batches...")

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(gene_order))
            batch_seqs = sequences[start:end]
            valid = [(i, seq) for i, seq in enumerate(batch_seqs) if seq]
            if valid:
                valid_indices, valid_seqs = zip(*valid)
                data = [(str(i), seq) for i, seq in enumerate(valid_seqs)]
                _, _, tokens = batch_converter(data)
                out = esm_model(tokens, repr_layers=[6], return_contacts=False)
                reps = out["representations"][6]
                for j, (orig_i, seq) in enumerate(zip(valid_indices, valid_seqs)):
                    gene_features[start + orig_i] = reps[j, 1:len(seq) + 1].mean(0)
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                print(f"    Batch {batch_idx+1}/{n_batches} done")

    torch.save({
        'features': gene_features,
        'gene_order': gene_order,
        'dim': EMB_DIM,
        'model': 'esm2_t6_8M_UR50D',
    }, gene_feat_path)
    print(f"  Saved: {gene_feat_path}")

# ============================================================================
# PART 2: MORGAN FINGERPRINTS FOR ANTIBIOTIC NODES
# ============================================================================
print("\n[2/2] Computing Morgan fingerprints for antibiotic nodes...")

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

antibiotic_names = set()
for aro_id, entry in card_data.items():
    for cat_id, cat_info in entry.get('ARO_category', {}).items():
        if cat_info.get('category_aro_class_name') == 'Antibiotic':
            antibiotic_names.add(cat_info.get('category_aro_name', ''))
antibiotic_names = sorted(antibiotic_names)
print(f"  {len(antibiotic_names)} unique antibiotics")

FP_DIM = 1024

def fetch_smiles_pubchem(name):
    # Request all SMILES variants; take whatever PubChem returns
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/property/CanonicalSMILES,IsomericSMILES/JSON"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            props = r.json()['PropertyTable']['Properties'][0]
            # Accept any SMILES key PubChem returns
            for key in ('CanonicalSMILES', 'IsomericSMILES', 'ConnectivitySMILES', 'SMILES'):
                if key in props:
                    return props[key]
    except Exception:
        pass
    return None

def smiles_to_morgan(smiles, n_bits=FP_DIM, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

antibiotic_features = torch.zeros(len(antibiotic_names), FP_DIM)
found = 0
not_found = []

for i, name in enumerate(antibiotic_names):
    smiles = fetch_smiles_pubchem(name)
    if smiles:
        fp = smiles_to_morgan(smiles)
        if fp is not None:
            antibiotic_features[i] = torch.tensor(fp)
            found += 1
        else:
            not_found.append(name)
    else:
        not_found.append(name)

    if (i + 1) % 20 == 0 or i == len(antibiotic_names) - 1:
        print(f"    {i+1}/{len(antibiotic_names)} — found: {found}, missing: {len(not_found)}")

    time.sleep(0.2)

if not_found:
    print(f"\n  Could not fetch SMILES for {len(not_found)} antibiotics (zero vectors used):")
    for n in not_found[:10]:
        print(f"    - {n}")
    if len(not_found) > 10:
        print(f"    ... and {len(not_found)-10} more")

ab_feat_path = FEATURES_DIR / "antibiotic_morgan_features.pt"
torch.save({
    'features': antibiotic_features,
    'antibiotic_order': antibiotic_names,
    'dim': FP_DIM,
    'method': 'Morgan_ECFP4_radius2',
    'not_found': not_found,
}, ab_feat_path)
print(f"  Saved: {ab_feat_path}")

print("\nFeature computation complete.")
print(f"  Gene features:       {gene_features.shape}  (ESM-2-8M embeddings)")
print(f"  Antibiotic features: {antibiotic_features.shape}  (Morgan ECFP4)")
print(f"  Antibiotic coverage: {found}/{len(antibiotic_names)} ({100*found/len(antibiotic_names):.1f}%)")
