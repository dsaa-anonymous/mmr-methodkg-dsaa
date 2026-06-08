# MethodKG TG4: SciBERT + HGT

This package implements **TG4: text+heterogeneous graph modeling** for MethodKG.
It uses pretrained text embeddings on award nodes and HGT message passing over the full heterogeneous MethodKG schema.

Recommended paper name:

```text
SciBERT+HGT (TG4)
```

This differs from:

- TG1 late fusion: concatenates text/graph features and trains a classifier.
- TG2/TG3 SimTeG/GraphSAGE: uses text embeddings with homogeneous award-projection GraphSAGE.
- TG4 here: uses text embeddings with typed nodes and typed edges in a heterogeneous HGT.

## Files

```text
create_text_hgt_embeddings.py
build_text_hgt_methodkg_graph.py
run_text_hgt.py
run_text_hgt_all_splits.py
summarize_text_hgt_results.py
requirements_text_hgt.txt
```

## Expected inputs

Required:

```text
cleaned_nsf_awards_2000_2025.csv
benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv
```

Recommended:

```text
award_pi_edges.csv
```

## Installation

Use your TIDE PyTorch/CUDA image if possible. Then:

```bash
pip install -r requirements_text_hgt.txt
```

If PyTorch Geometric does not install cleanly, install the wheel matching your Torch/CUDA version. For example, inspect versions:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
PY
```

Then follow https://pytorch-geometric.readthedocs.io/ for the matching install command.

## Step 1: Create full-corpus SciBERT embeddings

```bash
python create_text_hgt_embeddings.py \
  --awards cleaned_nsf_awards_2000_2025.csv \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --outdir text_hgt_embeddings_scibert_v1 \
  --model_name allenai/scibert_scivocab_uncased \
  --backend transformers \
  --batch_size 32 \
  --device cuda \
  --max_length 512 \
  --normalize
```

Output:

```text
text_hgt_embeddings_scibert_v1/methodkg_simteg_text_embeddings.csv
```

The filename contains `simteg` because the embedding utility is shared with the SimTeG-style package.

## Step 2: Build the text+HGT heterogeneous graph

```bash
python build_text_hgt_methodkg_graph.py \
  --awards cleaned_nsf_awards_2000_2025.csv \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --award_pi_edges award_pi_edges.csv \
  --text_embeddings text_hgt_embeddings_scibert_v1/methodkg_simteg_text_embeddings.csv \
  --outdir text_hgt_data_scibert_v1
```

Check:

```bash
cat text_hgt_data_scibert_v1/text_hgt_build_summary.json
cat text_hgt_data_scibert_v1/text_hgt_benchmark_award_match_report.csv
```

You want `benchmark_nodes = 2500` and `missing_text_embeddings = 0`.

## Step 3: Smoke test

```bash
python run_text_hgt.py \
  --graph_dir text_hgt_data_scibert_v1 \
  --outdir results/text_hgt_smoke/integration_random \
  --target target_integration_binary \
  --split_col split_random_cluster_stratified \
  --hidden_channels 128 \
  --num_layers 2 \
  --heads 4 \
  --dropout 0.35 \
  --lr 0.003 \
  --epochs 50 \
  --patience 10 \
  --device cuda \
  --tune_threshold
```

## Step 4: Main TG4 run

```bash
python run_text_hgt_all_splits.py \
  --graph_dir text_hgt_data_scibert_v1 \
  --outdir results/text_hgt_scibert_primary_all_v1 \
  --targets target_integration_binary target_design_binary target_mmr_multiclass \
  --splits split_random_cluster_stratified split_temporal_cluster_safe split_cross_program_cluster_safe split_cold_start_pi_cluster_safe split_cold_start_institution_cluster_safe split_edu_to_eng_cluster_safe \
  --hidden_channels 128 \
  --num_layers 2 \
  --heads 4 \
  --dropout 0.35 \
  --lr 0.003 \
  --epochs 300 \
  --patience 40 \
  --device cuda \
  --tune_threshold
```

## Optional larger L40 run

```bash
python run_text_hgt_all_splits.py \
  --graph_dir text_hgt_data_scibert_v1 \
  --outdir results/text_hgt_scibert_primary_all_h256_v1 \
  --targets target_integration_binary target_design_binary target_mmr_multiclass \
  --hidden_channels 256 \
  --num_layers 3 \
  --heads 4 \
  --dropout 0.40 \
  --lr 0.002 \
  --epochs 400 \
  --patience 50 \
  --device cuda \
  --use_amp \
  --tune_threshold
```

Start with the default run first.

## Step 5: Summarize

```bash
python summarize_text_hgt_results.py \
  --results_dir results/text_hgt_scibert_primary_all_v1 \
  --output text_hgt_scibert_primary_summary_v1.csv
```

## Step 6: Zip light results for review

```bash
zip -r methodkg_text_hgt_results_v1.zip \
  text_hgt_data_scibert_v1/text_hgt_build_summary.json \
  text_hgt_data_scibert_v1/text_hgt_benchmark_award_match_report.csv \
  text_hgt_data_scibert_v1/text_hgt_edge_summary.csv \
  text_hgt_data_scibert_v1/text_hgt_feature_manifest.csv \
  text_hgt_embeddings_scibert_v1/simteg_text_embedding_summary.json \
  results/text_hgt_scibert_primary_all_v1 \
  text_hgt_scibert_primary_summary_v1.csv \
  -x "*.pt" -x "*.bin" -x "*.pth"
```

Upload `methodkg_text_hgt_results_v1.zip`.

## Methodological note

This is a transductive text+heterogeneous-GNN baseline over the full unlabeled MethodKG graph. Labels are only used for the 2,500 benchmark award nodes. For strict temporal claims, compare with your leakage-safe historical-feature baselines and split-safe evaluation results.
