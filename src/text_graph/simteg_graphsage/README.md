# MethodKG TG2: SimTeG-style GraphSAGE

This package implements a practical SimTeG-style text+graph model for MethodKG.

SimTeG's core idea is a two-stage textual-graph pipeline:

1. Encode textual nodes with a strong language model.
2. Use those text embeddings as node features for a GNN.

For MethodKG, award nodes are NSF awards. The default implementation uses pretrained/frozen SciBERT mean-pooled embeddings for all awards in the full 32K corpus and then trains GraphSAGE on the award-projection graph. This is safer than using split-specific fine-tuned LM embeddings unless you carefully fine-tune only on each split's training set.

## Files

- `create_simteg_text_embeddings.py`: create full-corpus award text embeddings.
- `build_simteg_graphsage_graph.py`: build award-level GraphSAGE data with text embeddings as node features.
- `run_simteg_graphsage.py`: train/evaluate one target and split.
- `run_simteg_all_splits.py`: run all selected targets/splits.
- `summarize_simteg_results.py`: combine metrics files into one CSV.
- `requirements_simteg.txt`: Python requirements.

## Required inputs

Recommended:

```text
cleaned_nsf_awards_2000_2025.csv
benchmark_v2/methodkg_labeled_benchmark_v3_modeling.csv
award_pi_edges.csv
```

The benchmark file supplies labels and splits. The cleaned 32K file supplies the graph corpus. `award_pi_edges.csv` adds Co-PI/collaboration structure.

## Step 1: install

```bash
pip install -r requirements_simteg.txt
```

If PyTorch Geometric is not already installed, use the install command that matches your TIDE CUDA/PyTorch image. Check:

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
import torch_geometric
print(torch_geometric.__version__)
PY
```

## Step 2: create full-corpus SciBERT embeddings

```bash
python create_simteg_text_embeddings.py \
  --awards cleaned_nsf_awards_2000_2025.csv \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --outdir simteg_text_embeddings_scibert_v1 \
  --model_name allenai/scibert_scivocab_uncased \
  --backend transformers \
  --batch_size 32 \
  --device cuda \
  --max_length 512 \
  --normalize
```

This creates:

```text
simteg_text_embeddings_scibert_v1/methodkg_simteg_text_embeddings.csv
```

You can use sentence-transformers instead:

```bash
python create_simteg_text_embeddings.py \
  --awards cleaned_nsf_awards_2000_2025.csv \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --outdir simteg_text_embeddings_minilm_v1 \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --backend sentence_transformers \
  --batch_size 128 \
  --device cuda \
  --normalize
```

## Step 3: build SimTeG GraphSAGE graph

```bash
python build_simteg_graphsage_graph.py \
  --awards cleaned_nsf_awards_2000_2025.csv \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --award_pi_edges award_pi_edges.csv \
  --text_embeddings simteg_text_embeddings_scibert_v1/methodkg_simteg_text_embeddings.csv \
  --outdir simteg_graphsage_data_scibert_v1
```

Check:

```bash
cat simteg_graphsage_data_scibert_v1/simteg_graphsage_build_summary.json
cat simteg_graphsage_data_scibert_v1/simteg_benchmark_award_match_report.csv
```

You want `benchmark_nodes = 2500`.

To use text embeddings only with no structural node metadata:

```bash
python build_simteg_graphsage_graph.py \
  --awards cleaned_nsf_awards_2000_2025.csv \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --award_pi_edges award_pi_edges.csv \
  --text_embeddings simteg_text_embeddings_scibert_v1/methodkg_simteg_text_embeddings.csv \
  --outdir simteg_graphsage_data_scibert_textonly_v1 \
  --no_structural_features
```

## Step 4: smoke test

```bash
python run_simteg_graphsage.py \
  --graph_dir simteg_graphsage_data_scibert_v1 \
  --outdir results/simteg_smoke/integration_random \
  --target target_integration_binary \
  --split_col split_random_cluster_stratified \
  --hidden_channels 128 \
  --num_layers 2 \
  --dropout 0.35 \
  --lr 0.003 \
  --epochs 50 \
  --patience 10 \
  --device cuda \
  --tune_threshold
```

## Step 5: main primary run

```bash
python run_simteg_all_splits.py \
  --graph_dir simteg_graphsage_data_scibert_v1 \
  --outdir results/simteg_graphsage_scibert_primary_all_v1 \
  --targets target_integration_binary target_design_binary target_mmr_multiclass \
  --splits split_random_cluster_stratified split_temporal_cluster_safe split_cross_program_cluster_safe split_cold_start_pi_cluster_safe split_cold_start_institution_cluster_safe split_edu_to_eng_cluster_safe \
  --hidden_channels 128 \
  --num_layers 2 \
  --dropout 0.35 \
  --lr 0.003 \
  --epochs 300 \
  --patience 40 \
  --device cuda \
  --tune_threshold
```

## Step 6: optional larger run

```bash
python run_simteg_all_splits.py \
  --graph_dir simteg_graphsage_data_scibert_v1 \
  --outdir results/simteg_graphsage_scibert_primary_all_h256_v1 \
  --targets target_integration_binary target_design_binary target_mmr_multiclass \
  --hidden_channels 256 \
  --num_layers 3 \
  --dropout 0.4 \
  --lr 0.002 \
  --epochs 400 \
  --patience 50 \
  --device cuda \
  --use_amp \
  --tune_threshold
```

Start with the smaller run.

## Step 7: summarize

```bash
python summarize_simteg_results.py \
  --results_dir results/simteg_graphsage_scibert_primary_all_v1 \
  --output simteg_graphsage_scibert_primary_summary_v1.csv
```

## Step 8: upload light results

```bash
zip -r methodkg_simteg_graphsage_results_v1.zip \
  simteg_graphsage_data_scibert_v1/simteg_graphsage_build_summary.json \
  simteg_graphsage_data_scibert_v1/simteg_benchmark_award_match_report.csv \
  simteg_graphsage_data_scibert_v1/simteg_edge_summary.csv \
  simteg_graphsage_data_scibert_v1/simteg_graphsage_feature_manifest.csv \
  results/simteg_graphsage_scibert_primary_all_v1 \
  simteg_graphsage_scibert_primary_summary_v1.csv \
  simteg_text_embeddings_scibert_v1/simteg_text_embedding_summary.json \
  -x "*.pt" -x "*.bin" -x "*.pth"
```

If the edge summary filename is `simteg_graphsage_edge_summary.csv`, use that instead.

## Paper wording

Use this label:

```text
SciBERT+GraphSAGE (SimTeG-style)
```

This model is not raw text fine-tuning inside the GNN. It is a two-stage model: text embeddings first, GraphSAGE second. It is the right bridge between late fusion and deeper text+graph GNNs.
