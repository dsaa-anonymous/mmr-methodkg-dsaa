# MethodKG Metadata Ablations

This package adds two missing ablations from the MethodKG project plan:

1. **Metadata-only**: structured metadata without title/abstract text and without graph features.
2. **Text + metadata**: precomputed text embeddings plus structured metadata, without graph features.

These runs answer whether gains attributed to graph context are actually explained by simpler structured metadata such as year, directorate, program, institution, state, and award instrument.

## Where this belongs in the clean repo

Recommended code location:

```text
MethodKG_Clean/src/text_graph/metadata_ablations/
```

Recommended output locations:

```text
MethodKG_Clean/experiments/metadata_only/structured_metadata_v1/
MethodKG_Clean/experiments/text_graph/text_metadata_scibert_v1/
MethodKG_Clean/paper_outputs/summaries/
```

## Inputs

Required:

```text
MethodKG_Clean/data/benchmark/methodkg_labeled_benchmark_v2_modeling.csv
```

Optional for text+metadata:

```text
MethodKG_Clean/artifacts/features/text_embeddings_scibert_mean_v1/methodkg_text_embeddings.csv
```

No graph-only features, node2vec, metapath2vec, GraphSAGE, or HGT outputs are used by this package.

## Default metadata fields

The script automatically uses whichever of these columns are present:

```text
start_year, AwardInstrument, NSFDirectorate, NSFOrganization,
Program(s), ProgramElementCode(s), primary_program_key,
organization_clean, State, team_size, num_pis, num_copis
```

It excludes leakage-prone columns such as:

```text
label_*, target_*, candidate_*, annotation_*, review_priority,
annotation_guidance, award_amount, title_clean, abstract_clean
```

## Metadata-only command

```bash
python run_metadata_all_splits.py \
  --benchmark MethodKG_Clean/data/benchmark/methodkg_labeled_benchmark_v2_modeling.csv \
  --outdir MethodKG_Clean/experiments/metadata_only/structured_metadata_v1 \
  --targets target_integration_binary target_design_binary target_mmr_multiclass \
  --models dummy metadata_lr metadata_svm metadata_mlp metadata_extra_trees \
  --tune_threshold
```

## Text + metadata command

```bash
python run_metadata_all_splits.py \
  --benchmark MethodKG_Clean/data/benchmark/methodkg_labeled_benchmark_v2_modeling.csv \
  --text_embeddings MethodKG_Clean/artifacts/features/text_embeddings_scibert_mean_v1/methodkg_text_embeddings.csv \
  --outdir MethodKG_Clean/experiments/text_graph/text_metadata_scibert_v1 \
  --targets target_integration_binary target_design_binary target_mmr_multiclass \
  --models dummy metadata_lr metadata_svm metadata_mlp metadata_extra_trees \
  --tune_threshold
```

## Summarize

```bash
python summarize_metadata_results.py \
  --results_dir MethodKG_Clean/experiments/metadata_only/structured_metadata_v1 \
  --output MethodKG_Clean/paper_outputs/summaries/metadata_only_summary_v1.csv

python summarize_metadata_results.py \
  --results_dir MethodKG_Clean/experiments/text_graph/text_metadata_scibert_v1 \
  --output MethodKG_Clean/paper_outputs/summaries/text_metadata_scibert_summary_v1.csv
```

## Paper interpretation

Use these ablations to compare:

```text
Text-only vs Text + metadata
Text + metadata vs Text + graph
Text + graph vs Text + graph + metadata
```

The key reviewer-facing question is whether graph gains are actually due to structured metadata alone.
