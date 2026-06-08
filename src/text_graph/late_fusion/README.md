# MethodKG Text+Graph Late Fusion v1

This package implements **TG1: Late Fusion** for MethodKG.

It trains classifiers on the 2,500-row labeled benchmark by combining:

- frozen text embeddings from title + abstract
- leakage-safe historical graph features
- node2vec award embeddings
- metapath2vec award embeddings
- optional simple metadata

It does **not** use label, target, split, candidate, annotation, guidance, or review-priority columns as model inputs.

## Files

- `create_text_embeddings.py` — creates frozen text embeddings for awards.
- `run_late_fusion_baselines.py` — runs one target/split.
- `run_late_fusion_all_splits.py` — loops over targets and splits.
- `summarize_late_fusion_results.py` — summarizes recursive metrics files.
- `requirements_late_fusion.txt`

## Required inputs

Minimum:

```bash
benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv
```

Recommended feature files from previous phases:

```bash
graph_features_v1/methodkg_graph_only_features.csv
walk_embeddings_v1/node2vec_award_embeddings.csv
walk_embeddings_v1/metapath2vec_award_embeddings.csv
```

## 1. Create text embeddings

Fast MiniLM embeddings:

```bash
python create_text_embeddings.py \
  --input benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --outdir text_embeddings_minilm_v1 \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --backend sentence_transformers \
  --batch_size 128 \
  --device auto \
  --normalize
```

On TIDE GPU you may also try a stronger embedding model, for example:

```bash
python create_text_embeddings.py \
  --input benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --outdir text_embeddings_scibert_mean_v1 \
  --model_name allenai/scibert_scivocab_uncased \
  --backend transformers \
  --batch_size 32 \
  --device cuda \
  --max_length 512 \
  --normalize
```

## 2. Smoke test late fusion

```bash
python run_late_fusion_baselines.py \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --text_embeddings text_embeddings_minilm_v1/methodkg_text_embeddings.csv \
  --graph_features graph_features_v1/methodkg_graph_only_features.csv \
  --node2vec_embeddings walk_embeddings_v1/node2vec_award_embeddings.csv \
  --metapath2vec_embeddings walk_embeddings_v1/metapath2vec_award_embeddings.csv \
  --include_metadata \
  --outdir results/late_fusion_smoke/integration_random \
  --target target_integration_binary \
  --split_col split_random_cluster_stratified \
  --models dummy fusion_lr fusion_svm fusion_mlp fusion_extra_trees \
  --tune_threshold
```

## 3. Run primary targets across all splits

```bash
python run_late_fusion_all_splits.py \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --text_embeddings text_embeddings_minilm_v1/methodkg_text_embeddings.csv \
  --graph_features graph_features_v1/methodkg_graph_only_features.csv \
  --node2vec_embeddings walk_embeddings_v1/node2vec_award_embeddings.csv \
  --metapath2vec_embeddings walk_embeddings_v1/metapath2vec_award_embeddings.csv \
  --include_metadata \
  --outdir results/late_fusion_primary_all \
  --targets target_integration_binary target_design_binary target_mmr_multiclass \
  --splits split_random_cluster_stratified split_temporal_cluster_safe split_cross_program_cluster_safe split_cold_start_pi_cluster_safe split_cold_start_institution_cluster_safe split_edu_to_eng_cluster_safe \
  --models dummy fusion_lr fusion_svm fusion_mlp fusion_extra_trees \
  --tune_threshold
```

## 4. Recommended ablations

Run these feature-group ablations to show what graph context adds beyond text:

Text only embeddings:

```bash
python run_late_fusion_all_splits.py \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --text_embeddings text_embeddings_minilm_v1/methodkg_text_embeddings.csv \
  --outdir results/late_fusion_ablation_text_only \
  --targets target_integration_binary target_design_binary \
  --models fusion_lr fusion_svm fusion_mlp \
  --tune_threshold
```

Text + historical graph features:

```bash
python run_late_fusion_all_splits.py \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --text_embeddings text_embeddings_minilm_v1/methodkg_text_embeddings.csv \
  --graph_features graph_features_v1/methodkg_graph_only_features.csv \
  --outdir results/late_fusion_ablation_text_graphhist \
  --targets target_integration_binary target_design_binary \
  --models fusion_lr fusion_svm fusion_mlp \
  --tune_threshold
```

Text + graph embeddings:

```bash
python run_late_fusion_all_splits.py \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --text_embeddings text_embeddings_minilm_v1/methodkg_text_embeddings.csv \
  --node2vec_embeddings walk_embeddings_v1/node2vec_award_embeddings.csv \
  --metapath2vec_embeddings walk_embeddings_v1/metapath2vec_award_embeddings.csv \
  --outdir results/late_fusion_ablation_text_walkemb \
  --targets target_integration_binary target_design_binary \
  --models fusion_lr fusion_svm fusion_mlp \
  --tune_threshold
```

## 5. Summarize

```bash
python summarize_late_fusion_results.py \
  --results_dir results/late_fusion_primary_all \
  --output late_fusion_primary_summary.csv
```

## 6. Zip results for review

```bash
zip -r methodkg_late_fusion_results.zip \
  results/late_fusion_primary_all \
  late_fusion_primary_summary.csv \
  text_embeddings_minilm_v1/text_embedding_summary.json
```

## Paper framing

This is **TG1: late fusion**:

> We concatenate frozen award-text embeddings, historical graph features, and graph embeddings, then train supervised classifiers on the human-labeled benchmark.

The key comparison is against text-only models:

- If late fusion improves integration/design under temporal, cross-program, EDU→ENG, or cold-start splits, it supports the MethodKG claim that graph context improves robustness beyond text alone.
- If it does not improve, report that graph context is weak/complementary and motivate deeper GNN fusion.
