# MethodKG node2vec / metapath2vec Baselines

This toolkit implements lightweight graph-embedding baselines for MethodKG:

- **node2vec-style award projection embeddings**
- **metapath2vec-style heterogeneous random-walk embeddings**
- classifiers over the same benchmark splits used in the text-only and graph-feature baselines

These are intended as **graph-only embedding baselines**. They do not use title/abstract text, candidate flags, annotation guidance, or labels as input features.

## Important interpretation note

node2vec and metapath2vec are transductive random-walk embedding methods. In this toolkit, embeddings are trained over the full unlabeled MethodKG graph and then classifiers are trained only on the labeled benchmark rows. This is useful as a graph-embedding baseline, but for strict temporal/inductive claims you should still rely on:

1. leakage-safe historical graph-feature baselines, and
2. later inductive GNNs such as GraphSAGE/HGT on TIDE/GPU.

For the paper, describe these as **transductive graph-embedding baselines** unless you later add split-specific temporal graph construction.

## Files

```text
build_walk_graph_embeddings.py
run_embedding_baselines.py
run_embedding_all_splits.py
summarize_embedding_results.py
requirements_walk_embeddings.txt
README.md
```

## Required inputs

Minimum:

```text
cleaned_nsf_awards_2000_2025.csv
benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv
```

The embedding builder uses the cleaned award table directly. If it contains `co_pi_records`, those are used for Co-PI/person nodes. If not, it still uses lead PI, institution, program, NSF organization, directorate, state, and year fields.

## Install

```bash
pip install -r requirements_walk_embeddings.txt
```

## Step 1: Build node2vec + metapath2vec embeddings

Start with a smoke test configuration:

```bash
python build_walk_graph_embeddings.py \
  --awards cleaned_nsf_awards_2000_2025.csv \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --outdir walk_embeddings_smoke \
  --method both \
  --embedding_dim 32 \
  --num_walks 2 \
  --walk_length 10 \
  --epochs 2 \
  --workers 2
```

If that works, run a stronger embedding build:

```bash
python build_walk_graph_embeddings.py \
  --awards cleaned_nsf_awards_2000_2025.csv \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --outdir walk_embeddings_v1 \
  --method both \
  --embedding_dim 64 \
  --num_walks 8 \
  --walk_length 20 \
  --epochs 5 \
  --workers 2
```

For a stronger but slower run:

```bash
python build_walk_graph_embeddings.py \
  --awards cleaned_nsf_awards_2000_2025.csv \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --outdir walk_embeddings_v1_stronger \
  --method both \
  --embedding_dim 128 \
  --num_walks 10 \
  --walk_length 30 \
  --epochs 8 \
  --workers 4
```

## Step 2: Run classifiers for one smoke test

```bash
python run_embedding_baselines.py \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --embeddings walk_embeddings_v1/node2vec_award_embeddings.csv \
  --outdir results/node2vec_smoke/integration_random \
  --target target_integration_binary \
  --split_col split_random_cluster_stratified \
  --models dummy emb_lr emb_svm emb_rf emb_extra_trees \
  --tune_threshold
```

## Step 3: Run all primary targets and splits

```bash
python run_embedding_all_splits.py \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --embeddings walk_embeddings_v1/node2vec_award_embeddings.csv walk_embeddings_v1/metapath2vec_award_embeddings.csv \
  --outdir results/walk_embedding_primary_all \
  --targets target_integration_binary target_design_binary target_mmr_multiclass \
  --splits split_random_cluster_stratified split_temporal_cluster_safe split_cross_program_cluster_safe split_cold_start_pi_cluster_safe split_cold_start_institution_cluster_safe split_edu_to_eng_cluster_safe \
  --models dummy emb_lr emb_svm emb_rf emb_extra_trees \
  --tune_threshold
```

## Step 4: Summarize results

```bash
python summarize_embedding_results.py \
  --results_dir results/walk_embedding_primary_all \
  --output walk_embedding_primary_summary.csv
```

## Recommended reporting

For the paper, report these models as:

- `G1 node2vec`: homogeneous award-award projection embedding + classifier
- `G2 metapath2vec`: heterogeneous metapath random-walk embedding + classifier

Compare them with:

- dummy baseline
- graph historical-feature baseline
- text-only baselines
- later text+graph models

## Suggested paper phrasing

> We include node2vec and metapath2vec as transductive graph-embedding baselines. node2vec operates on an award-award projection induced by shared investigators, institutions, and programs. metapath2vec operates on the heterogeneous MethodKG schema using award-person-award, award-institution-award, award-program-award, and award-NSF-organization-award metapaths. Embeddings are used as graph-only features for supervised classifiers over the human-labeled benchmark.
