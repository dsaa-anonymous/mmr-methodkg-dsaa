# Text+Graph v3 rerun guide

Run all commands from the repository root.

The patched scripts prefer the v3 benchmark automatically:

- `data/benchmark/methodkg_labeled_benchmark_v3_modeling.csv`
- `data/benchmark/benchmark_v3/methodkg_labeled_benchmark_v3_modeling.csv`
- `data/benchmark/benchmark_v3.zip`

They ignore macOS sidecar files such as `._*.csv`, `.DS_Store`, and `__MACOSX`.

## 1. Late-fusion MiniLM embeddings

```bash
python src/text_graph/late_fusion/create_text_embeddings.py --overwrite
```

Default output:

```text
artifacts/features/text_embeddings_minilm_v1/methodkg_text_embeddings.csv
```

## 2. Late fusion: text + graph features + walk embeddings

This wrapper auto-uses any existing feature files in:

```text
artifacts/features/text_embeddings_minilm_v1/methodkg_text_embeddings.csv
artifacts/features/graph_features_v1/methodkg_graph_only_features.csv
artifacts/features/walk_embeddings_v1/node2vec_award_embeddings.csv
artifacts/features/walk_embeddings_v1/metapath2vec_award_embeddings.csv
```

```bash
python src/text_graph/late_fusion/run_late_fusion_all_splits.py --overwrite --include_metadata
```

Default experiment output:

```text
experiments/text_graph/late_fusion_minilm/primary/
```

Paper rollups:

```text
paper_outputs/summaries/late_fusion_minilm_metrics_summary.csv
paper_outputs/tables/late_fusion_minilm_test_metrics.csv
```

## 3. Metadata-only and text+metadata ablations

Metadata-only:

```bash
python src/text_graph/metadata_ablations/run_metadata_all_splits.py --overwrite
```

Text+metadata with SciBERT embeddings:

```bash
python src/text_graph/metadata_ablations/run_metadata_all_splits.py --overwrite --use_default_text_embeddings
```

Default output:

```text
experiments/text_graph/metadata_only/
experiments/text_graph/text_metadata_scibert_v1/
```

## 4. SimTeG-GraphSAGE

Create SciBERT award text embeddings for all graph nodes:

```bash
python src/text_graph/simteg_graphsage/create_simteg_text_embeddings.py --overwrite
```

Build the SimTeG GraphSAGE graph:

```bash
python src/text_graph/simteg_graphsage/build_simteg_graphsage_graph.py --overwrite
```

Run all benchmark splits:

```bash
python src/text_graph/simteg_graphsage/run_simteg_all_splits.py --overwrite
```

Default outputs:

```text
artifacts/features/simteg_text_embeddings_scibert_v1/
artifacts/graphs/simteg_graphsage_data_scibert_v1/
experiments/text_graph/simteg_graphsage/primary/
paper_outputs/summaries/simteg_graphsage_metrics_summary.csv
paper_outputs/tables/simteg_graphsage_test_metrics.csv
```

## 5. Text-HGT

Create SciBERT award text embeddings for HGT:

```bash
python src/text_graph/text_hgt/create_text_hgt_embeddings.py --overwrite
```

Build the Text-HGT heterogeneous graph:

```bash
python src/text_graph/text_hgt/build_text_hgt_methodkg_graph.py --overwrite
```

Run all benchmark splits:

```bash
python src/text_graph/text_hgt/run_text_hgt_all_splits.py --overwrite
```

Default outputs:

```text
artifacts/features/text_hgt_embeddings_scibert_v1/
artifacts/graphs/text_hgt_data_scibert_v1/
experiments/text_graph/text_hgt/primary/
paper_outputs/summaries/text_hgt_metrics_summary.csv
paper_outputs/tables/text_hgt_test_metrics.csv
```

## Explicit v3 path override

If auto-discovery ever selects the wrong file, pass the v3 benchmark explicitly:

```bash
python src/text_graph/late_fusion/run_late_fusion_all_splits.py \
  --benchmark data/benchmark/methodkg_labeled_benchmark_v3_modeling.csv \
  --overwrite --include_metadata
```
