# MethodKG Lightweight Graph-Only Baselines v1

This toolkit implements **Phase 2 / lightweight graph-only baselines** for MethodKG.
It is designed to run on a Mac CPU before you get TIDE GPU access.

The toolkit does **not** use award title/abstract text, candidate flags, annotation guidance, or labels as input features.
It builds leakage-safe historical graph/context features from the full cleaned NSF corpus and trains simple graph-only classifiers on the 2,500-row benchmark.

## Files

- `build_graph_only_features.py`  
  Builds leakage-safe historical graph features for each labeled award.

- `run_graph_baselines.py`  
  Trains graph-only classifiers for one target and one split.

- `run_graph_all_splits.py`  
  Runs graph-only classifiers across the recommended targets and splits.

- `summarize_graph_results.py`  
  Collects all `metrics.csv` files into one summary CSV.

- `requirements_graph_only.txt`  
  Minimal Python requirements.

## Required input files

Minimum required:

```bash
cleaned_nsf_awards_2000_2025.csv
benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv
```

Recommended optional input:

```bash
award_pi_edges.csv
```

If `award_pi_edges.csv` is available from your cleaning pipeline, use it. It enables Co-PI/team/collaboration features. If you do not provide it, the feature builder falls back to lead-PI-only features.

## Step 1: Install requirements

```bash
pip install -r requirements_graph_only.txt
```

## Step 2: Build graph-only features

With `award_pi_edges.csv`:

```bash
python build_graph_only_features.py \
  --awards cleaned_nsf_awards_2000_2025.csv \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --award_pi_edges award_pi_edges.csv \
  --outdir graph_features_v1
```

Without `award_pi_edges.csv`:

```bash
python build_graph_only_features.py \
  --awards cleaned_nsf_awards_2000_2025.csv \
  --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
  --outdir graph_features_v1
```

This creates:

```text
graph_features_v1/methodkg_graph_only_features.csv
graph_features_v1/methodkg_graph_only_feature_manifest.csv
graph_features_v1/methodkg_graph_only_feature_summary.csv
```

## Step 3: Smoke test one target and split

```bash
python run_graph_baselines.py \
  --features graph_features_v1/methodkg_graph_only_features.csv \
  --outdir results/graph_only_smoke/integration_random \
  --target target_integration_binary \
  --split_col split_random_cluster_stratified \
  --models dummy graph_lr graph_rf graph_extra_trees
```

## Step 4: Run the primary graph-only baselines

```bash
python run_graph_all_splits.py \
  --features graph_features_v1/methodkg_graph_only_features.csv \
  --outdir results/graph_only_primary_all \
  --targets target_integration_binary target_design_binary target_mmr_multiclass \
  --splits split_random_cluster_stratified split_temporal_cluster_safe split_cross_program_cluster_safe split_cold_start_pi_cluster_safe split_cold_start_institution_cluster_safe split_edu_to_eng_cluster_safe \
  --models dummy graph_lr graph_rf graph_extra_trees
```

## Step 5: Optional secondary targets

```bash
python run_graph_all_splits.py \
  --features graph_features_v1/methodkg_graph_only_features.csv \
  --outdir results/graph_only_secondary_all \
  --targets target_mmr_binary target_qual_binary target_quant_binary target_method_signal_binary \
  --splits split_random_cluster_stratified split_temporal_cluster_safe split_cross_program_cluster_safe split_cold_start_pi_cluster_safe split_cold_start_institution_cluster_safe split_edu_to_eng_cluster_safe \
  --models dummy graph_lr graph_rf graph_extra_trees
```

## Step 6: Summarize results

```bash
python summarize_graph_results.py \
  --results_dir results/graph_only_primary_all \
  --output graph_only_primary_summary.csv
```

## What the graph-only features include

The feature builder creates features such as:

- lead PI prior award count
- lead PI prior collaborator count
- team prior award/collaboration summaries
- institution prior award/person/program counts
- program prior award/person/institution counts
- NSF organization/directorate/state prior counts
- cold-start flags for PI, institution, and program
- current team size and Co-PI count
- known static categorical metadata such as program, NSF organization, directorate, institution, state, and award instrument

Historical features use **only awards with `start_year < target_start_year`**. Same-year awards are not used as prior history.

## Important modeling note

These are lightweight graph-only baselines, not full GNNs. They are useful for answering:

> Does historical PI/institution/program/collaboration context contain predictive signal even without award text?

Once you get TIDE access, you can build heavier graph models such as GraphSAGE, R-GCN, or HGT using the same graph schema.
