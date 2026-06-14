# MMR-MethodKG: A Temporal Text-Graph Benchmark for Mining Mixed-Methods Reporting Signals in NSF STEM & Engineering Education Awards

This repository contains the data-processing, benchmark-construction, training, evaluation, and paper-summary code for **MMR-MethodKG**, a temporal text-graph benchmark for predicting mixed-methods methodology-reporting signals in NSF STEM and engineering education award abstracts.

The repository is organized to support the IEEE ICDM reproducibility checklist. It includes dataset statistics, preprocessing steps, train/validation/test split definitions, training and evaluation code, model-family outputs, and paper-level summary files.

## 1. Repository Structure

```text
MMR-MethodKG/
  artifacts/
    features/
      text_embeddings_minilm_v1/
      text_embeddings_scibert_mean_v1/
      ...

  data/
    raw/
      Full_DSAA_Awards.csv

    processed/
      final_gold_labels_adjudicated.csv
      methodkg_outputs_v7_clustered_from_cleaned/
        cleaned_nsf_awards_2000_2025.csv
        nsf_awards_with_methodology_flags.csv
        annotation_sample_2000_2025.csv
        data_quality_report.csv
        project_text_cluster_report.csv
        annotation_project_text_cluster_report.csv

    edges/
      award_pi_edges.csv
      pi_collaboration_edges.csv
      award_institution_edges.csv
      award_program_edges.csv

    benchmark/
      methodkg_labeled_benchmark_v3_modeling.csv
      methodkg_labeled_benchmark_v3_audit.csv
      methodkg_benchmark_v3_label_quality_report.csv
      methodkg_benchmark_v3_label_issues.csv
      methodkg_benchmark_v3_split_summary.csv
      methodkg_benchmark_v3_split_leakage_report.csv
      methodkg_benchmark_v3_duplicate_cluster_report.csv
      methodkg_benchmark_v3_reliability_report.csv
      methodkg_benchmark_v3_feature_manifest.csv
      methodkg_benchmark_v3_summary.json

    scripts/
      build_methodkg_pipeline.py
      create_methodkg_modeling_benchmark.py

  src/
    text_only/
    metadata_only/
    graph_only/
    text_graph/

  experiments/
    text_only/
    metadata_only/
    graph_only/
    text_graph/

  paper_outputs/
    summaries/
    tables/

  requirements.txt
  README.md
```

## 2. Quick Start for Reviewers

The main released modeling file is:

```text
data/benchmark/methodkg_labeled_benchmark_v3_modeling.csv
```

The main result summaries used for paper interpretation are in:

```text
paper_outputs/summaries/
```

Reviewers who want to verify the reported paper claims should start with:

```text
1. data/benchmark/methodkg_labeled_benchmark_v3_modeling.csv
2. data/benchmark/methodkg_benchmark_v3_split_summary.csv
3. data/benchmark/methodkg_benchmark_v3_split_leakage_report.csv
4. paper_outputs/summaries/
5. experiments/
```

The `paper_outputs/summaries/` folder contains compact model-family summaries. The `experiments/` folders contain full per-task, per-split, and per-model outputs.

## 3. Environment Setup

The code was developed with Python 3.11.

```bash
conda create -n methodkg python=3.11 -y
conda activate methodkg
pip install -r requirements.txt
```

Some transformer and GNN models benefit from GPU acceleration. Classical text, metadata-only, and feature-based graph baselines can run on CPU.

Pretrained models used by the code include:

```text
allenai/scibert_scivocab_uncased
sentence-transformers/all-MiniLM-L6-v2
```

These models are downloaded through the Hugging Face ecosystem unless they are already cached locally.

## 4. Data Pipeline

The data pipeline has three main stages:

```text
raw NSF award data
  -> processed cleaned award files
  -> graph edge files
  -> labeled modeling benchmark
```

The main raw input is expected at:

```text
data/raw/Full_DSAA_Awards.csv
```

The final adjudicated labels are expected at:

```text
data/processed/final_gold_labels_adjudicated.csv
```

### 4.1 Build Processed Award Files and Graph Edges

Run from the repository root:

```bash
python data/scripts/build_methodkg_pipeline.py \
  --input data/raw/Full_DSAA_Awards.csv \
  --overwrite
```

This writes processed award files to:

```text
data/processed/
```

Expected processed outputs:

```text
cleaned_nsf_awards_2000_2025.csv
nsf_awards_with_methodology_flags.csv
annotation_sample_2000_2025.csv
data_quality_report.csv
project_text_cluster_report.csv
annotation_project_text_cluster_report.csv
```

It writes graph edge files to:

```text
data/edges/
```

Expected edge outputs:

```text
award_pi_edges.csv
pi_collaboration_edges.csv
award_institution_edges.csv
award_program_edges.csv
```

If the cleaned awards file already exists, reviewers may rebuild flags, annotation sample, and edges from the cleaned file:

```bash
python data/scripts/build_methodkg_pipeline.py \
  --input data/processed/methodkg_outputs_v7_clustered_from_cleaned/cleaned_nsf_awards_2000_2025.csv \
  --input_is_cleaned \
  --overwrite
```

### 4.2 Build the Benchmark from Final Gold Labels

Run:

```bash
python data/scripts/create_methodkg_modeling_benchmark.py \
  --input data/processed/final_gold_labels_adjudicated.csv \
  --outdir data/benchmark \
  --overwrite
```

This writes:

```text
methodkg_labeled_benchmark_v3_modeling.csv
methodkg_labeled_benchmark_v3_audit.csv
methodkg_benchmark_v3_label_quality_report.csv
methodkg_benchmark_v3_label_issues.csv
methodkg_benchmark_v3_split_summary.csv
methodkg_benchmark_v3_split_leakage_report.csv
methodkg_benchmark_v3_duplicate_cluster_report.csv
methodkg_benchmark_v3_reliability_report.csv
methodkg_benchmark_v3_feature_manifest.csv
methodkg_benchmark_v3_summary.json
```

The main file used by all experiments is:

```text
data/benchmark/methodkg_labeled_benchmark_v3_modeling.csv
```

### 4.3 Reviewer-Safe Benchmark Verification

To rebuild into a temporary folder and compare against the released benchmark files:

```bash
python data/scripts/create_methodkg_modeling_benchmark.py \
  --input data/processed/final_gold_labels_adjudicated.csv \
  --outdir data/_benchmark_rebuild_check \
  --compare_to_dir data/benchmark \
  --overwrite
```

This is the safest command for reviewers because it verifies reproducibility without overwriting the released benchmark.
## 5. Dataset Statistics

The released benchmark contains a full NSF award graph and a 2,500-award human-labeled subset. The graph-level and labeled-benchmark statistics below match the counts reported in the paper.

### 5.1 Graph and Benchmark Size

```text
Awards in METHODKG-FULL:                     32,161
Awards in METHODKG-LABELED:                   2,500
Released edge rows:                         256,182

Unique project clusters in labeled set:       2,330
Duplicate project clusters:                     119
Rows in duplicate clusters:                     289

PI / Co-PI nodes:                            47,812
Institution nodes:                            2,945
Program nodes:                                  453
NSF organization / division nodes:                5
Directorate nodes:                                2
State nodes:                                     56
Award years:                                     26

Award-investigator edges:                    85,110
Award-institution edges:                     32,161
Award-program edges:                         40,564
PI-PI collaboration edges:                   98,347
```

### 5.2 Sampling Strata

The labeled benchmark was constructed using stratified sampling to include positive, negative, ambiguous, and hard-negative cases. The final sampling strata are:

```text
Explicit mixed-methods candidates:              434
Implicit mixed-methods candidates:              433
Design/integration-enriched awards:             369
Quantitative-only hard negatives:               343
Qualitative-only hard negatives:                336
Method-heavy background awards:                 294
Random background awards:                       291
```

These sampling strata describe how the benchmark was constructed. They are not the same as the supervised target-label distributions used for modeling.

Because METHODKG-LABELED is enriched for methodology-relevant cases, label proportions in the labeled subset should not be interpreted as prevalence estimates for the full NSF award corpus.

### 5.3 Supervised Target Distribution

The released modeling file contains the derived target columns used for supervised prediction. The primary binary targets are:

```text
target_integration_binary:
  absent:   1,340   53.60%
  present:  1,160   46.40%

target_design_binary:
  absent:     372   14.88%
  present:  2,128   85.12%
```

The main multiclass target is:

```text
target_mmr_multiclass:
  explicit_mmr:             763   30.52%
  implicit_mmr:             397   15.88%
  no_method_signal:         372   14.88%
  quant_only:               362   14.48%
  multi_method_not_mmr:     338   13.52%
  qual_only:                268   10.72%
```

These labels should be interpreted as methodology-reporting signals in NSF award abstracts, not as verification of the complete methodology ultimately used in the funded projects.

## 6. Evaluation Splits

All model families use the predefined split columns in:

```text
data/benchmark/methodkg_labeled_benchmark_v3_modeling.csv
```

Split columns:

```text
split_random_cluster_stratified
split_temporal_cluster_safe
split_edu_to_eng_cluster_safe
split_cross_program_cluster_safe
split_cold_start_pi_cluster_safe
split_cold_start_institution_cluster_safe
```

Split sizes:

```text
Random cluster-stratified:
  train = 1,750
  validation = 376
  test = 374

Temporal cluster-safe:
  train = 1,257
  validation = 346
  test = 897

EDU -> ENG cluster-safe:
  train = 2,158
  validation = none
  test = 342

Cross-program cluster-safe:
  train = 2,228
  validation = 54
  test = 218

Cold-start PI cluster-safe:
  train = 1,735
  validation = 400
  test = 365

Cold-start institution cluster-safe:
  train = 2,253
  validation = 133
  test = 114
```

Split summaries and leakage checks are stored in:

```text
data/benchmark/methodkg_benchmark_v3_split_summary.csv
data/benchmark/methodkg_benchmark_v3_split_leakage_report.csv
```

The split design reduces leakage from duplicate project-text clusters and supports evaluation under random, temporal, cross-program, EDU-to-ENG, cold-start PI, and cold-start institution generalization settings.


## 7. Model Family Shorthand

The paper uses the following shorthand:

```text
T  = text-only models
G  = graph-only models
TG = text+graph models
```

Additional control families include metadata-only and text+metadata models.
## 8. Code Locations for T, G, and TG Models

The repository separates these model families by input modality.

### 8.1 Text-Only Models: T

Text-only models are stored in:

```text
src/text_only/
```

Main scripts:

```text
src/text_only/run_text_all_splits.py
src/text_only/run_text_baselines.py
src/text_only/train_transformer_text.py
```

Outputs:

```text
experiments/text_only/text_only_all/
paper_outputs/summaries/text_only_all_metrics_summary.csv
paper_outputs/tables/text_only_all_test_metrics.csv
```

This family includes regex baselines, TF-IDF logistic regression, TF-IDF SVM, frozen MiniLM embedding classifiers, and SciBERT fine-tuning.

### 8.2 Graph-Only Models: G

Graph-only models are stored in:

```text
src/graph_only/
```

Graph-only model families include:

```text
historical graph features
node2vec / metapath2vec
GraphSAGE structural
HGT structural
```

Typical outputs:

```text
experiments/graph_only/
paper_outputs/summaries/
paper_outputs/tables/
```

Graph-only models use relational context without award title/abstract text as classifier input.

### 8.3 Text+Graph Models: TG

Text+graph models are stored in:

```text
src/text_graph/
```

The text+graph family is divided into four variants:

```text
TG1 = late fusion
TG2 = SimTeG GraphSAGE with text-only node features
TG3 = Text-HGT
```

#### TG1: Late Fusion

Stored in:

```text
src/text_graph/late_fusion/
```

Main scripts:

```text
src/text_graph/late_fusion/create_text_embeddings.py
src/text_graph/late_fusion/run_late_fusion_all_splits.py
```

Outputs:

```text
experiments/text_graph/late_fusion_scibert/
experiments/text_graph/late_fusion_minilm/
paper_outputs/summaries/late_fusion_scibert_metrics_summary.csv
paper_outputs/summaries/late_fusion_minilm_metrics_summary.csv
```

TG1 concatenates text embeddings with graph-derived features. When `--include_metadata` is used, the same late-fusion pipeline also includes structured award metadata.

#### TG2: SimTeG GraphSAGE with Text-Only Node Features

Stored in:

```text
src/text_graph/simteg_graphsage/
```

Main scripts:

```text
src/text_graph/simteg_graphsage/create_simteg_text_embeddings.py
src/text_graph/simteg_graphsage/build_simteg_graphsage_graph.py
src/text_graph/simteg_graphsage/run_simteg_all_splits.py
src/text_graph/simteg_graphsage/summarize_simteg_results.py
```

Outputs:

```text
artifacts/features/simteg_text_embeddings_scibert_v1/
artifacts/graphs/simteg_graphsage_data_scibert_v1/
experiments/text_graph/simteg_graphsage/
paper_outputs/summaries/simteg_graphsage_metrics_summary.csv
```

TG2 uses SciBERT text embeddings as award-node features during graph construction.


#### TG3: Text-HGT

Stored in:

```text
src/text_graph/text_hgt/
```

Typical outputs:

```text
experiments/text_graph/text_hgt/
paper_outputs/summaries/text_hgt_metrics_summary.csv
```

TG3 uses text-initialized award nodes with heterogeneous graph message passing over the MethodKG schema.

### 8.4 Metadata-Only and Text+Metadata Controls

Metadata-only and text+metadata control models are stored in:

```text
src/metadata_only/
```

Typical outputs:

```text
experiments/metadata_only/
paper_outputs/summaries/
```

These controls test whether contextual gains come from structured award metadata rather than graph structure.

## 9. Running the Main Experiments

All commands should be run from the repository root.

### 9.1 Text-Only Models: T

```bash
python src/text_only/run_text_all_splits.py \
  --input data/benchmark/methodkg_labeled_benchmark_v3_modeling.csv \
  --overwrite \
  --include_embeddings \
  --include_transformer \
  --transformer_splits split_random_cluster_stratified split_temporal_cluster_safe
```

Outputs:

```text
experiments/text_only/text_only_all/
artifacts/features/text_embeddings_minilm_v1/
paper_outputs/summaries/text_only_all_metrics_summary.csv
```

### 9.2 TG1: Late Fusion SciBERT

Create SciBERT text embeddings:

```bash
python src/text_graph/late_fusion/create_text_embeddings.py \
  --embedding_family scibert \
  --overwrite
```

Run TG1 late fusion with SciBERT embeddings:

```bash
python src/text_graph/late_fusion/run_late_fusion_all_splits.py \
  --embedding_family scibert \
  --overwrite
```

Outputs:

```text
artifacts/features/text_embeddings_scibert_mean_v1/
experiments/text_graph/late_fusion_scibert/
paper_outputs/summaries/late_fusion_scibert_metrics_summary.csv
```

### 9.3 TG1: Late Fusion MiniLM

Create MiniLM embeddings:

```bash
python src/text_graph/late_fusion/create_text_embeddings.py \
  --embedding_family minilm \
  --overwrite
```

Run TG1 late fusion with MiniLM embeddings:

```bash
python src/text_graph/late_fusion/run_late_fusion_all_splits.py \
  --embedding_family minilm \
  --overwrite
```

Outputs:

```text
artifacts/features/text_embeddings_minilm_v1/
experiments/text_graph/late_fusion_minilm/
paper_outputs/summaries/late_fusion_minilm_metrics_summary.csv
```

### 9.4 TG2: SimTeG GraphSAGE with Text-Only Node Features

TG2 uses SciBERT text embeddings as award-node features and disables additional structural node features during graph construction.

Step 1: create SciBERT text embeddings.

```bash
python src/text_graph/simteg_graphsage/create_simteg_text_embeddings.py \
  --model_name allenai/scibert_scivocab_uncased \
  --backend transformers \
  --outdir artifacts/features/simteg_text_embeddings_scibert_v1 \
  --overwrite
```

Step 2: build the TG2 graph with text-only node features.

```bash
python src/text_graph/simteg_graphsage/build_simteg_graphsage_graph.py \
  --text_embeddings artifacts/features/simteg_text_embeddings_scibert_v1/methodkg_simteg_text_embeddings.csv \
  --outdir artifacts/graphs/simteg_graphsage_data_scibert_v1 \
  --overwrite
```

Step 3: train and evaluate TG2.

```bash
python src/text_graph/simteg_graphsage/run_simteg_all_splits.py \
  --graph_dir artifacts/graphs/simteg_graphsage_data_scibert_v1 \
  --outdir experiments/text_graph/simteg_graphsage \
  --overwrite
```

Expected outputs:

```text
artifacts/features/simteg_text_embeddings_scibert_v1/
artifacts/graphs/simteg_graphsage_data_scibert_v1/
experiments/text_graph/simteg_graphsage/primary/
paper_outputs/summaries/simteg_graphsage_metrics_summary.csv
```

### 9.5 TG4: Text-HGT

Text-HGT scripts are stored in:

```text
src/text_graph/text_hgt/
```

Run the relevant script with `--help` to inspect arguments:

```bash
python src/text_graph/text_hgt/<script_name>.py --help
```

Outputs:

```text
experiments/text_graph/text_hgt/
paper_outputs/summaries/text_hgt_metrics_summary.csv
paper_outputs/tables/text_hgt_test_metrics.csv
```

### 9.6 Graph-Only Models: G

Graph-only scripts are stored in:

```text
src/graph_only/
```

Run each graph-only script with `--help` to inspect its arguments:

```bash
python src/graph_only/<script_name>.py --help
```

Outputs:

```text
experiments/graph_only/
paper_outputs/summaries/
paper_outputs/tables/
```

### 9.7 Metadata-Only Models

Metadata-only scripts are stored in:

```text
src/metadata_only/
```

Run each metadata-only script with `--help` to inspect its arguments:

```bash
python src/metadata_only/<script_name>.py --help
```

Outputs:

```text
experiments/metadata_only/
paper_outputs/summaries/
paper_outputs/tables/
```

## 10. Reported Metrics

For binary tasks, the code reports:

```text
accuracy
macro-F1
positive-class F1
precision
recall
ROC-AUC
PR-AUC
confusion matrix
```

For the multiclass task, the code reports:

```text
accuracy
macro-F1
weighted-F1
classification report
confusion matrix
```

The paper uses **macro-F1** as the main headline metric because the primary labels are imbalanced. Positive-class F1 and PR-AUC are supporting metrics for binary tasks.

## 11. Output Files

### 11.1 Paper-Level Summaries

The main paper-level result summaries are stored in:

```text
paper_outputs/summaries/
```

Reviewers should inspect this folder first when checking reported results.

Typical files include:

```text
text_only_all_metrics_summary.csv
late_fusion_scibert_metrics_summary.csv
late_fusion_minilm_metrics_summary.csv
simteg_graphsage_metrics_summary.csv
text_hgt_metrics_summary.csv
historical_features_metrics_summary.csv
node2vec_metapath2vec_metrics_summary.csv
graphsage_structural_metrics_summary.csv
hgt_structural_metrics_summary.csv
```

### 11.2 Full Experiment Outputs

Full experiment outputs are stored in:

```text
experiments/text_only/
experiments/metadata_only/
experiments/graph_only/
experiments/text_graph/
```

Common output files include:

```text
metrics_summary.csv
*_metrics.json
*_classification_report.json
*_confusion_matrix.csv
*_predictions.csv
run_config.json
```

Large model files, checkpoints, and caches may also appear in experiment subfolders depending on the model family.

### 11.3 Artifacts

Cached embeddings and derived feature files are stored in:

```text
artifacts/features/
```

Examples:

```text
artifacts/features/text_embeddings_minilm_v1/
artifacts/features/text_embeddings_scibert_mean_v1/
```

## Artifacts Folder

The `artifacts/` folder stores reusable intermediate files generated by the data and model pipelines. These files are not the primary benchmark data and are not the final reported results. Instead, they are cached features used to avoid recomputing expensive embeddings or graph representations during repeated experiments.

The main artifact subfolder is:

```text
artifacts/features/
```

Typical contents include:

```text
artifacts/features/text_embeddings_minilm_v1/
artifacts/features/text_embeddings_scibert_mean_v1/
```

These folders contain cached text embeddings used by text-only and text+graph models.

### MiniLM Embedding Cache

```text
artifacts/features/text_embeddings_minilm_v1/
```

This folder stores MiniLM/Sentence-BERT-style frozen text embeddings for award title and abstract text. These embeddings are used by:

```text
src/text_only/
src/text_graph/late_fusion/
```

They support text-only embedding baselines and TG1 Late Fusion MiniLM.

### SciBERT Embedding Cache

```text
artifacts/features/text_embeddings_scibert_mean_v1/
```

This folder stores SciBERT mean-pooled text embeddings for award title and abstract text. These embeddings are used by:

```text
src/text_graph/late_fusion/
src/text_graph/simteg_graphsage/
src/text_graph/text_hgt/
```

They support TG1 Late Fusion SciBERT, SimTeG-style GraphSAGE, and Text-HGT variants.

### Why Artifacts Are Stored Separately

The artifact files are separated from `data/` because they are derived features, not raw or labeled benchmark data. They are separated from `experiments/` because they can be reused across multiple model families and reruns.

Conceptually:

```text
data/        = raw, processed, edge, and benchmark CSV files
artifacts/   = reusable derived features and embedding caches
experiments/ = model outputs, metrics, predictions, and run configs
paper_outputs/ = compact summaries and tables used for the paper
```


## 12. Main Result Interpretation

The main empirical findings are:

1. Text-only models are the strongest overall baselines.
2. Metadata-only and graph-only models contain useful contextual signal but are insufficient alone.
3. Text+graph models broadly improve over graph-only models, showing that graph structure is most useful when combined with award text.
4. Text+graph models beat text-only only in selected settings.
5. The clearest text+graph improvement is for design prediction under cross-program generalization.
6. TG1 Late Fusion SciBERT is the strongest practical text+graph model.
7. TG4 Text-HGT is useful among text+graph models for selected EDU-to-ENG transfer settings.
8. TG2/TG3 SimTeG-style GraphSAGE is useful mainly for harder multiclass transfer/cold-start settings but does not surpass text-only overall.

The main claims can be checked from:

```text
paper_outputs/summaries/
```

## 13. Reproducibility Checklist Mapping

This section maps repository contents to the IEEE reproducibility checklist.

### 13.1 Mathematical Setting, Algorithms, and Models

The paper defines the supervised prediction setting, model families, input modalities, and temporal graph assumptions.

Code locations:

```text
src/text_only/
src/metadata_only/
src/graph_only/
src/text_graph/
```

### 13.2 Assumptions

The main assumptions are:

```text
1. Labels are methodology-reporting signals in award abstracts, not verified methodology use in completed research outputs.
2. Target-award title, abstract, metadata, and year are available at prediction time.
3. Strict historical graph features use prior awards and prior relationships before the target award year.
4. Some graph-embedding and GNN models use unlabeled graph structure in a semi-transductive representation-learning setting.
5. Validation and test labels are never used for supervised training, threshold selection, early stopping, or model selection.
```

### 13.3 Complexity

This paper is primarily an applied benchmark paper and does not introduce a new theoretical algorithm requiring complexity analysis. For this reason, algorithmic time/space complexity is marked as not applicable in the checklist.

### 13.4 Theoretical Claims

The paper does not make new theoretical claims requiring formal proof. Theoretical-claim checklist items are marked as not applicable.

### 13.5 Dataset Statistics

Dataset statistics are reported in the paper and stored in:

```text
data/benchmark/methodkg_benchmark_v3_summary.json
data/benchmark/methodkg_benchmark_v3_label_quality_report.csv
data/benchmark/methodkg_benchmark_v3_split_summary.csv
data/processed/methodkg_outputs_v7_clustered_from_cleaned/data_quality_report.csv
```

### 13.6 Train/Validation/Test Splits

Split definitions and counts are stored in:

```text
data/benchmark/methodkg_benchmark_v3_split_summary.csv
data/benchmark/methodkg_benchmark_v3_split_leakage_report.csv
```

The split columns are included directly in:

```text
data/benchmark/methodkg_labeled_benchmark_v3_modeling.csv
```

### 13.7 Excluded Data and Preprocessing

Preprocessing is implemented in:

```text
data/scripts/build_methodkg_pipeline.py
```

Benchmark construction is implemented in:

```text
data/scripts/create_methodkg_modeling_benchmark.py
```

Quality reports are written to:

```text
data/processed/methodkg_outputs_v7_clustered_from_cleaned/data_quality_report.csv
data/benchmark/methodkg_benchmark_v3_label_quality_report.csv
data/benchmark/methodkg_benchmark_v3_feature_manifest.csv
```

### 13.8 Dataset Availability

The benchmark files are stored in:

```text
data/benchmark/
```

The graph edge files are stored in:

```text
data/edges/
```

The cleaned processed files are stored in:

```text
data/processed/
```

If the raw NSF file is not included, it can be reconstructed from public NSF award records using award identifiers and metadata fields.

### 13.9 New Data Collection and Annotation

The final adjudicated label file is:

```text
data/processed/final_gold_labels_adjudicated.csv
```

Annotation and adjudication details are described in the paper. Label quality and reliability-related outputs are stored in:

```text
data/benchmark/methodkg_benchmark_v3_label_quality_report.csv
data/benchmark/methodkg_benchmark_v3_reliability_report.csv
```

### 13.10 Dependencies

Dependencies are specified in:

```text
requirements.txt
```

### 13.11 Training Code

Training and evaluation code is stored in:

```text
src/
```

Text-only training:

```text
src/text_only/
```

Graph-only training:

```text
src/graph_only/
```

Text+graph training:

```text
src/text_graph/
```

Metadata-only training:

```text
src/metadata_only/
```

### 13.12 Evaluation Code

Evaluation is integrated into each model-family script and writes metrics to:

```text
experiments/
paper_outputs/summaries/
paper_outputs/tables/
```

### 13.13 Pretrained Models

Pretrained text encoders are loaded through Hugging Face / SentenceTransformers. The main pretrained models are:

```text
allenai/scibert_scivocab_uncased
sentence-transformers/all-MiniLM-L6-v2
```

### 13.14 README with Commands

This README provides the main commands needed to rebuild the benchmark, rerun the core models, and inspect the reported results.

### 13.15 Hyperparameters

Each experiment folder includes a configuration file where available, typically:

```text
run_config.json
```

Default seeds are fixed in the scripts, typically seed `42`. Transformer, graph, and classifier hyperparameters are recorded in the experiment outputs.

### 13.16 Number of Runs

The reported experiments use one run per model/task/split configuration with fixed predefined splits and fixed random seed unless otherwise noted.

### 13.17 Reported Measures

The paper reports macro-F1 as the headline metric and uses positive-class F1 and PR-AUC as supporting metrics for binary tasks.

### 13.18 Central Tendency and Variation

The paper reports performance across multiple predefined evaluation splits rather than repeated random trials. The variation discussed in the paper is across realistic generalization settings: random, temporal, cross-program, EDU-to-ENG, cold-start PI, and cold-start institution.

### 13.19 Runtime and Compute

Classical baselines can be run on CPU. Transformer and graph neural models are faster with a GPU. Exact runtime depends on hardware, local caching of pretrained models, and whether embedding artifacts already exist.

### 13.20 Computing Infrastructure

The experiments were run in a Python TIDE Cluster environment with Python 3.11, and 1 NVIDIA A100 GPU (80 GB VRAM)

## 14. Minimal Reviewer Commands

### 14.1 Verify Benchmark Rebuild

```bash
python data/scripts/create_methodkg_modeling_benchmark.py \
  --input data/processed/final_gold_labels_adjudicated.csv \
  --outdir data/_benchmark_rebuild_check \
  --compare_to_dir data/benchmark \
  --overwrite
```

### 14.2 Run Main Text-Only Baselines

```bash
python src/text_only/run_text_all_splits.py \
  --input data/benchmark/methodkg_labeled_benchmark_v3_modeling.csv \
  --overwrite \
  --include_embeddings
```

### 14.3 Run Main Text+Graph Model

```bash
python src/text_graph/late_fusion/create_text_embeddings.py \
  --embedding_family scibert \
  --overwrite
```

```bash
python src/text_graph/late_fusion/run_late_fusion_all_splits.py \
  --embedding_family scibert \
  --overwrite \
  --include_metadata
```

### 14.4 Inspect Paper-Level Summaries

```bash
ls paper_outputs/summaries
```

## 15. Notes on Exact Reproduction

The released experimental results are tied to:

```text
data/benchmark/methodkg_labeled_benchmark_v3_modeling.csv
```

This file defines the official labels, feature columns, and split assignments used in the paper. Regenerating the benchmark with different split logic can change results. Therefore, the repository includes a comparison mode in the benchmark builder so reviewers can verify that regenerated benchmark outputs match the released benchmark before rerunning experiments.
