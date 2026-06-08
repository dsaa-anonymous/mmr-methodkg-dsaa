# MethodKG Text-Only Models v5 Final

This package contains the fixed text-only modeling scripts for MethodKG.

## Files

- `run_text_baselines.py`: runs regex, TF-IDF Logistic Regression, TF-IDF SVM, and optional frozen embedding baselines.
- `run_text_all_splits.py`: loops over recommended MethodKG splits and targets.
- `train_transformer_text.py`: optional transformer/SciBERT fine-tuning.
- `requirements_text_only.txt`: Python dependencies.

## Important fixes in v5

- `run_text_baselines.py` accepts `--seed`.
- Embedding model names are aligned between wrapper and baseline script.
- Internal validation split is fixed for splits with only `train`/`test`, such as `split_edu_to_eng_cluster_safe`.
- The zip extracts into `methodkg_text_only_models_v5_final/` to avoid confusion with earlier versions.

## Recommended input

Use:

```bash
benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv
```

## Smoke test

```bash
python run_text_all_splits.py \
  --input benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --outdir results/text_only_smoke_v5 \
  --targets target_integration_binary \
  --splits split_random_cluster_stratified \
  --classical_models regex tfidf_lr tfidf_svm
```

## Frozen embedding smoke test for the previously failing split

```bash
python run_text_all_splits.py \
  --input benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --outdir results/text_only_embeddings_smoke_v5 \
  --targets target_integration_binary \
  --splits split_edu_to_eng_cluster_safe \
  --classical_models tfidf_lr tfidf_svm \
  --include_embeddings
```

## Main classical run

```bash
python run_text_all_splits.py \
  --input benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --outdir results/text_only_classical_all_v5 \
  --targets target_integration_binary target_design_binary target_mmr_multiclass \
  --splits split_random_cluster_stratified split_temporal_cluster_safe split_cross_program_cluster_safe split_cold_start_pi_cluster_safe split_cold_start_institution_cluster_safe split_edu_to_eng_cluster_safe \
  --classical_models regex tfidf_lr tfidf_svm
```

## Main embedding run

```bash
python run_text_all_splits.py \
  --input benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --outdir results/text_only_embeddings_all_v5 \
  --targets target_integration_binary target_design_binary target_mmr_multiclass \
  --splits split_random_cluster_stratified split_temporal_cluster_safe split_cross_program_cluster_safe split_cold_start_pi_cluster_safe split_cold_start_institution_cluster_safe split_edu_to_eng_cluster_safe \
  --classical_models tfidf_lr tfidf_svm \
  --include_embeddings
```
