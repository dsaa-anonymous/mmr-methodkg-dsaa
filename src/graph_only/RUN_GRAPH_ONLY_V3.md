# MethodKG graph-only v3 rerun commands

Run these commands from the repository root after copying this `graph_only/` folder into `src/graph_only/`.

The patched scripts auto-detect:

- benchmark: `data/benchmark/methodkg_labeled_benchmark_v3_modeling.csv`, `data/benchmark/benchmark_v3/`, or `data/benchmark/benchmark_v3.zip`
- cleaned awards: `data/processed/methodkg_outputs_v7_clustered_from_cleaned/cleaned_nsf_awards_2000_2025.csv` or the best matching awards CSV under `data/processed/`
- award-PI edges: `data/edges/award_pi_edges.csv` or the best matching edge file when available

They also ignore macOS sidecar files such as `._*.csv` and `__MACOSX` zip entries.

## 0. Optional cleanup for macOS sidecar files

```bash
find data -name '._*' -type f -delete
find data -name '__MACOSX' -type d -prune -exec rm -rf {} +
```

## 1. Historical graph-only features and classical graph baselines

```bash
python src/graph_only/historical_features/build_graph_only_features.py --overwrite

python src/graph_only/historical_features/run_graph_all_splits.py --overwrite
```

Outputs:

- features: `artifacts/features/graph_features_v1/`
- experiments: `experiments/graph_only/historical_features/graph_only_primary_all/`
- rollups: `paper_outputs/summaries/graph_only_historical_features_metrics_summary.csv` and `paper_outputs/tables/graph_only_historical_features_test_metrics.csv`

## 2. Walk embeddings: node2vec / metapath2vec

```bash
python src/graph_only/walk_embeddings/build_walk_graph_embeddings.py --overwrite

python src/graph_only/walk_embeddings/run_embedding_all_splits.py --overwrite
```

Outputs:

- embeddings: `artifacts/features/walk_embeddings_v1/`
- experiments: `experiments/graph_only/node2vec_metapath2vec/walk_embedding_primary_all/`
- rollups: `paper_outputs/summaries/walk_embedding_metrics_summary.csv` and `paper_outputs/tables/walk_embedding_test_metrics.csv`

## 3. GraphSAGE structural graph and runs

```bash
python src/graph_only/graphsage/build_graphsage_award_graph.py --overwrite

python src/graph_only/graphsage/run_graphsage_all_splits.py --overwrite
```

Outputs:

- graph artifact: `artifacts/graphs/graphsage_data_v2/`
- experiments: `experiments/graph_only/graphsage_structural/graphsage_structural_primary_all_v2/`
- rollups: `paper_outputs/summaries/graphsage_structural_metrics_summary.csv` and `paper_outputs/tables/graphsage_structural_test_metrics.csv`

## 4. HGT structural graph and runs

```bash
python src/graph_only/hgt/build_hgt_methodkg_graph.py --overwrite

python src/graph_only/hgt/run_hgt_all_splits.py --overwrite
```

Outputs:

- graph artifact: `artifacts/graphs/hgt_data_v1/`
- experiments: `experiments/graph_only/hgt_structural/hgt_structural_primary_all_h256/`
- rollups: `paper_outputs/summaries/hgt_structural_metrics_summary.csv` and `paper_outputs/tables/hgt_structural_test_metrics.csv`

## Explicit v3 paths, if needed

If auto-discovery still points to the wrong file, pass explicit paths:

```bash
python src/graph_only/historical_features/build_graph_only_features.py \
  --benchmark data/benchmark/methodkg_labeled_benchmark_v3_modeling.csv \
  --awards data/processed/methodkg_outputs_v7_clustered_from_cleaned/cleaned_nsf_awards_2000_2025.csv \
  --award_pi_edges data/edges/award_pi_edges.csv \
  --overwrite
```
