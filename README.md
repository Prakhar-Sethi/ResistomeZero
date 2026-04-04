# ResistomeZero

**Zero-Shot Antibiotic Resistance Prediction via Heterogeneous Graph Neural Networks**

> Can a model predict whether a gene confers resistance to an antibiotic it has *never seen during training*? This project says yes — and shows exactly which architectures can do it.

---

## The Problem

Every existing computational AMR (antimicrobial resistance) paper trains and tests on the **same set of antibiotics**. That's not how the real world works. When a new antibiotic enters development, there is zero historical resistance data for it. Prior methods are blind to this scenario.

We define the first **zero-shot benchmark** for antibiotic resistance prediction: 47 test antibiotics with absolutely no resistance edges in the training graph. The model must generalize purely through graph structure and biological features.

---

## What We Built

A heterogeneous knowledge graph from [CARD v4.0.1](https://card.mcmaster.ca/) enriched with Gene Ontology, KEGG pathways, STRING PPI data, and ChEMBL drug-target annotations:

| Component | Count |
|-----------|-------|
| Total nodes | 9,135 |
| Resistance genes | 6,442 |
| Antibiotics | 231 |
| Drug classes | 46 |
| Resistance mechanisms | 8 |
| Gene families | 522 |
| Edge types | 9 |
| Total directed edges | ~53,600 |

**Node features:**
- Genes: [ESM-2](https://github.com/facebookresearch/esm) protein language model embeddings (320-dim)
- Antibiotics: Morgan molecular fingerprints (1,024-dim)
- All nodes: one-hot type encoding (8-dim)

All features projected jointly to 64-dim via trained linear layers.

---

## Zero-Shot Split

```
Train:  161 antibiotics  →  resistance edges visible during training
Val:     23 antibiotics  →  zero resistance edges during training
Test:    47 antibiotics  →  zero resistance edges during training
```

Test antibiotics exist as nodes (reachable via drug class and protein target edges) but carry zero resistance signal.

---

## Models

| Category | Models |
|----------|--------|
| Non-learning baseline | DrugClass-Heuristic |
| Homogeneous GNN | GCN, GraphSAGE |
| Attention GNN | GAT |
| Relation-aware GNN | **R-GCN**, HGT |
| Transductive KGE | DistMult, ComplEx, TransE |

---

## Results

Full-drug ranking evaluation: each test gene is scored against all 231 antibiotics simultaneously using **filtered MRR** (training positives excluded from ranking pool). Hard negatives: gene×antibiotic pairs only.

| Model | AUC | MRR | Hits@1 | Hits@10 | Mean Rank |
|-------|-----|-----|--------|---------|-----------|
| DrugClass-Heuristic† | — | 0.280 | 25.5% | 32.0% | 70.4 |
| **R-GCN** | **0.896 ± 0.016** | **0.243 ± 0.078** | **19.0%** | **35.7%** | **52.8 ± 8.0** |
| GraphSAGE | 0.879 ± 0.010 | 0.020 ± 0.004 | 0.17% | 1.3% | 99.8 ± 3.1 |
| GCN | 0.696 ± 0.056 | 0.015 ± 0.002 | 0.12% | 2.3% | 135.2 ± 5.5 |
| TransE | 0.630 ± 0.010 | 0.013 ± 0.000 | 0% | 0% | 82.3 ± 2.4 |
| GAT | 0.754 ± 0.068 | 0.009 ± 0.001 | 0% | 0% | 140.6 ± 6.1 |
| ComplEx‡ | 0.515 ± 0.043 | 0.009 ± 0.000 | 0% | 0% | 114.0 ± 3.3 |
| DistMult‡ | 0.491 ± 0.030 | 0.009 ± 0.000 | 0% | 0% | 114.0 ± 1.6 |
| HGT | 0.848 ± 0.015 | 0.007 ± 0.002 | 0% | 0% | 156.4 ± 19.9 |

*5 seeds, mean ± std. Random baseline MRR = 0.0043. R-GCN is ~57× better than random (p < 10⁻⁶⁰, r = 0.54).*

† No training required.  ‡ Transductive KGE — cannot generalize to unseen antibiotics by construction.

### Key Findings

**1. Drug class is the dominant zero-shot signal.**
The heuristic — no parameters, no training — achieves MRR 0.280 (65× random). The best indicator of zero-shot resistance is simply an antibiotic's drug class: genes that resist known beta-lactams will likely resist novel ones.

**2. R-GCN is the only GNN that works.**
Only R-GCN is statistically significant across all 5 seeds. It learns per-relation weight matrices for each of the 9 edge types, allowing it to cleanly propagate the drug-class signal through `antibiotic→class` edges. All other GNNs collapse all edge types into a single aggregation, diluting this signal. R-GCN beats the heuristic on mean rank (52.8 vs 70.4) and Hits@10 (35.7% vs 32.0%).

**3. AUC is misleading.**
GraphSAGE: 87.9% AUC, but MRR 0.020 and Hits@10 1.3%. HGT: 84.8% AUC, Hits@10 0%. High AUC only requires distinguishing gene-antibiotic pairs from random node pairs — trivial given type information. Full-drug ranking is the correct metric.

**4. All KGE models fail zero-shot — regardless of architecture.**
DistMult, ComplEx (BRIDGE's best model with 97% transductive accuracy), and TransE all achieve Hits@10 = 0% across all seeds. KGE models learn lookup-table embeddings updated only on training edges. A test antibiotic with zero training edges has a random embedding — making all downstream scores random. Inductive GNNs compute embeddings at inference from graph topology, which is why they can generalize.

---

## Ablation: GO + KEGG Enrichment

| Model | Original MRR | Enriched MRR | Δ |
|-------|-------------|-------------|---|
| R-GCN | 0.205 ± 0.040 | 0.224 ± 0.055 | **+0.019** |
| GraphSAGE | 0.023 ± 0.010 | 0.021 ± 0.012 | −0.002 |
| GCN | 0.016 ± 0.002 | 0.016 ± 0.002 | −0.000 |
| GAT | 0.016 ± 0.009 | 0.009 ± 0.001 | −0.007 |
| HGT | 0.007 ± 0.001 | 0.007 ± 0.000 | 0.000 |
| DistMult / ComplEx / TransE | — | — | 0.000 |

R-GCN uniquely benefits from enrichment because its per-relation matrices treat `gene→GO` and `gene→KEGG` as distinct relation types it can learn from independently. KGE models are invariant — they don't propagate structure at inference.

---

## Installation

```bash
git clone https://github.com/Prakhar-Sethi/ResistomeZero.git
cd ResistomeZero
pip install -r requirements.txt
```

Requires PyTorch ≥ 2.0 and PyTorch Geometric ≥ 2.3.

---

## Reproducing Results

### 1. Download and build the graph

```bash
python utils/download_card.py          # downloads CARD v4.0.1
python utils/build_graph.py            # builds base heterogeneous graph
python utils/compute_node_features.py  # ESM-2 gene features + Morgan fingerprints
python utils/add_go_terms.py           # Gene Ontology enrichment
python utils/add_kegg_pathways.py      # KEGG pathway enrichment
python utils/add_string_edges.py       # STRING PPI edges (MTB)
python utils/add_drugbank_targets.py   # ChEMBL drug-target edges
```

### 2. Run main evaluation (all models, 5 seeds)

```bash
python experiments/multiseed_eval.py
```

Results saved to `results/metrics/multiseed_results.json`.

### 3. Run ablation (original vs enriched graph)

```bash
python experiments/ablation_enrichment.py
```

### 4. Run split sensitivity

```bash
python experiments/split_sensitivity.py
```

### 5. Run transductive vs zero-shot comparison

```bash
python experiments/distmult_transductive.py
```

---

## Project Structure

```
ResistomeZero/
├── models/
│   ├── gcn.py               # Graph Convolutional Network
│   ├── graphsage.py         # GraphSAGE
│   ├── gat.py               # Graph Attention Network
│   ├── rgcn.py              # Relational GCN (best model)
│   ├── hgt.py               # Heterogeneous Graph Transformer
│   ├── distmult.py          # DistMult KGE
│   ├── complex.py           # ComplEx KGE
│   └── transe.py            # TransE KGE
├── experiments/
│   ├── multiseed_eval.py    # Primary: all models × 5 seeds
│   ├── ablation_enrichment.py
│   ├── split_sensitivity.py
│   ├── distmult_transductive.py
│   └── ranking_eval.py
├── utils/
│   ├── download_card.py
│   ├── build_graph.py
│   ├── compute_node_features.py
│   ├── add_go_terms.py
│   ├── add_kegg_pathways.py
│   ├── add_string_edges.py
│   └── add_drugbank_targets.py
├── results/
│   ├── metrics/             # JSON result files (all experiments)
│   ├── tables/              # LaTeX tables
│   └── plots/               # Performance figures
├── paper/
│   └── main.tex             # IEEE conference paper
└── requirements.txt
```

---

## Paper

The full paper (IEEE conference format) is in `paper/main.tex`. It covers:
- Problem formulation and motivation
- Graph construction methodology
- Model descriptions with equations
- Full results, ablation, and split sensitivity analysis
- Discussion and limitations

---

## Citation

If you use this work, please cite:

```bibtex
@article{resistomezero2026,
  title   = {Zero-Shot Antibiotic Resistance Prediction via Heterogeneous Graph Neural Networks},
  year    = {2026}
}
```

---

## License

MIT
