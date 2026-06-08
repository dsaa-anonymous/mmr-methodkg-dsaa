# MethodKG HGT Structural-Only Baselines

This package implements a **graph-only Heterogeneous Graph Transformer (HGT)** baseline for MethodKG.

It is inspired by the official `pyHGT` repository, but uses modern PyTorch Geometric `HGTConv` instead of the original project-specific TensorFlow/PyG-era data pipeline. The original pyHGT repository stores heterogeneous graph features and adjacency by type and implements the core transformer-style heterogeneous convolution in `conv.py`; this package adapts that idea to your MethodKG files and benchmark splits.

## What this package includes

- `build_hgt_methodkg_graph.py` — builds a PyG `HeteroData` graph.
- `run_hgt_structural.py` — trains/evaluates HGT for one target/split.
- `run_hgt_all_splits.py` — loops over the main targets and splits.
- `summarize_hgt_results.py` — combines metrics into one CSV.
- `requirements_hgt.txt` — Python dependencies.

## What model phase this covers

| Phase | Included? |
|---|---:|
| Historical graph features | No |
| node2vec / metapath2vec | No |
| GraphSAGE structural-only | No |
| **HGT structural-only** | **Yes** |
| Text+Graph late fusion | No |
| SciBERT/SPECTER/e5 + GraphSAGE/HGT | No |

This is a **graph-only** model. It does not use title/abstract text embeddings.

## Graph schema

Node types:

- `award`
- `person`
- `institution`
- `program`
- `nsf_org`
- `directorate`
- `state`
- `year`

Edge types include:

- `award --has_pi--> person`
- `award --has_copi--> person`
- `award --at_institution--> institution`
- `award --funded_by_program--> program`
- `program --program_in_org--> nsf_org`
- `nsf_org --org_in_directorate--> directorate`
- `institution --institution_in_state--> state`
- `award --has_year--> year`
- plus reverse edges for message passing.

## Important leakage note

This HGT implementation is a **transductive heterogeneous graph-only baseline**. It uses the unlabeled full MethodKG graph structure. It does not use human labels outside the training split, and it does not use text embeddings. However, because heterogeneous entity nodes connect awards across the full graph, this model should not be described as the strictest temporal-leakage-safe method.

For strict leakage-safe temporal claims, use the historical graph-feature baseline. For this HGT result, use wording such as:

> We include HGT as a transductive heterogeneous graph-only baseline over MethodKG's typed NSF award graph.

## Required input files

Required:

```bash
cleaned_nsf_awards_2000_2025.csv
benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv
```

Recommended:

```bash
award_pi_edges.csv
```

`award_pi_edges.csv` gives better PI/Co-PI edges. Without it, the builder falls back to lead-PI-only edges.

## Install

Use your TIDE PyTorch/CUDA image if possible.

```bash
pip install -r requirements_hgt.txt
```

If PyTorch Geometric install fails, install the wheel matching your PyTorch/CUDA version from the official PyG installation page. For example, the command often looks like:

```bash
pip install torch_geometric
```

or, for some CUDA builds:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-<TORCH_VERSION>+<CUDA>.html
pip install torch_geometric
```

## 1. Build the HGT graph

With `award_pi_edges.csv`:

```bash
python build_hgt_methodkg_graph.py \
  --awards cleaned_nsf_awards_2000_2025.csv \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --award_pi_edges award_pi_edges.csv \
  --outdir hgt_data_v1
```

Without `award_pi_edges.csv`:

```bash
python build_hgt_methodkg_graph.py \
  --awards cleaned_nsf_awards_2000_2025.csv \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --outdir hgt_data_v1
```

For a stricter graph-only baseline without text-length metadata:

```bash
python build_hgt_methodkg_graph.py \
  --awards cleaned_nsf_awards_2000_2025.csv \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --award_pi_edges award_pi_edges.csv \
  --outdir hgt_data_v1_no_textlen \
  --drop_text_length_features
```

Check:

```bash
cat hgt_data_v1/hgt_build_summary.json
cat hgt_data_v1/hgt_benchmark_award_match_report.csv
```

You want:

```text
benchmark_nodes = 2500
benchmark_nodes_expected = 2500
```

## 2. Smoke test

```bash
python run_hgt_structural.py \
  --graph_dir hgt_data_v1 \
  --outdir results/hgt_smoke/integration_random \
  --target target_integration_binary \
  --split_col split_random_cluster_stratified \
  --hidden_channels 64 \
  --num_layers 2 \
  --heads 4 \
  --epochs 30 \
  --patience 8 \
  --device auto \
  --tune_threshold
```

## 3. Main primary run

```bash
python run_hgt_all_splits.py \
  --graph_dir hgt_data_v1 \
  --outdir results/hgt_structural_primary_all \
  --targets target_integration_binary target_design_binary target_mmr_multiclass \
  --splits split_random_cluster_stratified split_temporal_cluster_safe split_cross_program_cluster_safe split_cold_start_pi_cluster_safe split_cold_start_institution_cluster_safe split_edu_to_eng_cluster_safe \
  --hidden_channels 128 \
  --num_layers 2 \
  --heads 4 \
  --dropout 0.35 \
  --lr 0.003 \
  --epochs 300 \
  --patience 40 \
  --device auto \
  --tune_threshold
```

## 4. Larger L40 GPU run

Since you have an NVIDIA L40, this is reasonable after the smoke test:

```bash
python run_hgt_all_splits.py \
  --graph_dir hgt_data_v1 \
  --outdir results/hgt_structural_primary_all_h256 \
  --targets target_integration_binary target_design_binary target_mmr_multiclass \
  --hidden_channels 256 \
  --num_layers 3 \
  --heads 4 \
  --dropout 0.4 \
  --lr 0.002 \
  --epochs 400 \
  --patience 50 \
  --device cuda \
  --use_amp \
  --tune_threshold
```

Start with the smaller run first.

## 5. Summarize

```bash
python summarize_hgt_results.py \
  --results_dir results/hgt_structural_primary_all \
  --output hgt_structural_primary_summary.csv
```

## 6. Zip for review

```bash
zip -r methodkg_hgt_structural_results.zip \
  hgt_data_v1 \
  results/hgt_structural_primary_all \
  hgt_structural_primary_summary.csv \
  -x "*/hgt_model.pt"
```

Upload `methodkg_hgt_structural_results.zip` for interpretation.
