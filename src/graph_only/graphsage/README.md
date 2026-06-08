# MethodKG Graph Phase 3: Structural-only GraphSAGE v2

This package implements **Graph Phase 3** for the MethodKG project: a structural-only GraphSAGE model over an award-level temporal projection graph.

It is inspired by the official GraphSAGE implementation by Hamilton, Ying, and Leskovec, but it uses modern PyTorch Geometric rather than the older TensorFlow/NetworkX pipeline. This is more practical for TIDE GPU and your current CSV files.

## What model this implements

Included:

- Graph Phase 3: **GraphSAGE structural-only**
- Award-level temporal projection graph
- No title/abstract text
- No candidate flags
- No annotation guidance
- No label columns as features
- Same benchmark split columns as your v2 benchmark

Not included here:

- Text+Graph late fusion
- SciBERT/SPECTER/e5 + GraphSAGE
- HGT

Those should come in later phases.

## Why not directly copy the old `williamleif/GraphSAGE` code?

The official repository is important historically, but it targets an older TensorFlow/NetworkX-style input format. Your current MethodKG workflow already uses CSV benchmark files and TIDE GPU access, so a PyTorch Geometric implementation is easier to run, maintain, and extend.

The original GraphSAGE repo expects files like `*-G.json`, `*-id_map.json`, `*-class_map.json`, and optional `*-feats.npy`. This package instead builds a PyG `Data` object directly from your MethodKG CSVs.

## Input files

Required:

```bash
cleaned_nsf_awards_2000_2025.csv
benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv
```

Recommended:

```bash
award_pi_edges.csv
```

The `award_pi_edges.csv` file gives better PI/Co-PI sharing edges. If omitted, the builder falls back to the lead `person_id` in the cleaned award file.

## Installation on TIDE

Use a PyTorch GPU/CUDA image if TIDE provides one. If you need to install manually, install PyTorch and PyTorch Geometric according to the CUDA version in your environment.

A typical CUDA 12.1 example is:

```bash
pip install -r requirements_graphsage.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

If your PyTorch version is not 2.4.0 or CUDA is not 12.1, adjust the `data.pyg.org` wheel URL.

Check GPU:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')
PY
```

Check PyG:

```bash
python - <<'PY'
import torch_geometric
print(torch_geometric.__version__)
PY
```

## Step 1: Build the GraphSAGE award graph

```bash
python build_graphsage_award_graph.py \
  --awards cleaned_nsf_awards_2000_2025.csv \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --award_pi_edges award_pi_edges.csv \
  --outdir graphsage_data_v2
```

This creates:

```text
graphsage_data_v2/graphsage_award_graph.pt
graphsage_data_v2/graphsage_node_index.csv
graphsage_data_v2/graphsage_edge_summary.csv
graphsage_data_v2/graphsage_feature_manifest.csv
graphsage_data_v2/graphsage_build_summary.json
graphsage_data_v2/graphsage_label_maps.json
```


After building, check that the summary reports all benchmark rows:

```bash
cat graphsage_data_v2/graphsage_build_summary.json | grep benchmark_nodes -A2
cat graphsage_data_v2/graphsage_benchmark_award_match_report.csv
```

You want `benchmark_nodes` to equal `benchmark_nodes_expected` and, for your current benchmark, that should be 2,500 unique benchmark awards.

### Edge construction

The graph has one node per award. In v2, the node table is built from the union of the cleaned full award corpus and the labeled benchmark file, so all 2,500 benchmark rows are retained even if some awards were dropped during cleaning. Edges connect older awards to later awards when they share:

- PI / Co-PI
- institution
- program element code
- NSF organization
- directorate

The default edge direction is **older -> newer**, which is a safer default for temporal experiments. You can add `--bidirectional` for an undirected/transductive baseline, but label it clearly if you use it.

## Step 2: Run one smoke test

```bash
python run_graphsage_structural.py \
  --graph_dir graphsage_data_v2 \
  --outdir results/graphsage_smoke/integration_random \
  --target target_integration_binary \
  --split_col split_random_cluster_stratified \
  --epochs 50 \
  --patience 10 \
  --hidden_channels 64 \
  --device auto \
  --tune_threshold
```

## Step 3: Run all primary targets and splits

```bash
python run_graphsage_all_splits.py \
  --graph_dir graphsage_data_v2 \
  --outdir results/graphsage_structural_primary_all \
  --targets target_integration_binary target_design_binary target_mmr_multiclass \
  --splits split_random_cluster_stratified split_temporal_cluster_safe split_cross_program_cluster_safe split_cold_start_pi_cluster_safe split_cold_start_institution_cluster_safe split_edu_to_eng_cluster_safe \
  --hidden_channels 128 \
  --num_layers 2 \
  --dropout 0.35 \
  --lr 0.003 \
  --epochs 300 \
  --patience 40 \
  --device auto \
  --tune_threshold
```

For the L40 GPU, you can try a larger model later:

```bash
python run_graphsage_all_splits.py \
  --graph_dir graphsage_data_v2 \
  --outdir results/graphsage_structural_primary_all_h256 \
  --targets target_integration_binary target_design_binary target_mmr_multiclass \
  --hidden_channels 256 \
  --num_layers 3 \
  --dropout 0.4 \
  --lr 0.002 \
  --epochs 400 \
  --patience 50 \
  --device cuda \
  --tune_threshold \
  --use_amp
```

## Step 4: Summarize results

```bash
python summarize_graphsage_results.py \
  --results_dir results/graphsage_structural_primary_all \
  --output graphsage_structural_primary_summary.csv
```

## Step 5: Zip outputs for review

```bash
zip -r methodkg_graphsage_structural_results.zip \
  graphsage_data_v2 \
  results/graphsage_structural_primary_all \
  graphsage_structural_primary_summary.csv \
  -x "*/graphsage_model.pt"
```

You can omit `graphsage_model.pt` to keep the zip small.

## How to interpret this model

This is a **graph-only structural model**. It should not be expected to beat text-only baselines, because your labels are largely about methodology reporting in the abstract. A useful outcome is:

- better than dummy,
- competitive with historical graph features / node2vec / metapath2vec,
- maybe helpful on integration/design under transfer or cold-start splits.

The main ICDM result will likely come later from **Text+Graph late fusion** or **SciBERT/SPECTER/e5 + GraphSAGE**.
