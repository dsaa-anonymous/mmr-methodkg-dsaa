from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

V3_BENCHMARK_NAMES = [
    "methodkg_labeled_benchmark_v3_modeling.csv",
    "methodkg_labeled_benchmark_v3.csv",
]
BENCHMARK_NAMES = V3_BENCHMARK_NAMES + [
    "methodkg_labeled_benchmark_v2_modeling.csv",
    "methodkg_labeled_benchmark_v2.csv",
    "methodkg_labeled_benchmark_v1.csv",
]


def is_sidecar_name(name: str) -> bool:
    parts = Path(name).parts
    base = Path(name).name
    return (
        base.startswith("._")
        or base.startswith(".")
        or "__MACOSX" in parts
        or base in {".DS_Store", "Thumbs.db"}
    )


def valid_file(path: Path) -> bool:
    return path.is_file() and not is_sidecar_name(str(path))


def find_repo_root(start: str | Path | None = None) -> Path:
    p = Path(start).expanduser().resolve() if start else Path.cwd().resolve()
    if p.is_file():
        p = p.parent
    for q in [p, *p.parents]:
        if (q / "data").exists() and ((q / "src").exists() or (q / "artifacts").exists() or (q / "experiments").exists()):
            return q
    return Path.cwd().resolve()


def resolve_existing_path(value: str | Path | None, repo_root: Path, *, required: bool = True) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        if required:
            raise FileNotFoundError("Missing required path")
        return None
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    p = p.resolve()
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Path does not exist: {p}")
        return None
    if p.is_file() and is_sidecar_name(str(p)):
        raise ValueError(f"Refusing to use sidecar/hidden file: {p}")
    return p


def resolve_output_path(value: str | Path | None, repo_root: Path, default: str | Path) -> Path:
    p = Path(value).expanduser() if value else Path(default)
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def reset_dir_if_overwrite(outdir: Path, overwrite: bool) -> None:
    if overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)


def _version_score(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    m = re.search(r"v(\d+)", name)
    return (int(m.group(1)) if m else 0, 1 if "modeling" in name else 0, str(path))


def _pick_best(paths: Iterable[Path]) -> Optional[Path]:
    cand = [p for p in paths if valid_file(p)]
    return sorted(cand, key=_version_score, reverse=True)[0] if cand else None


def discover_benchmark(repo_root: Path) -> Path:
    explicit: list[Path] = []
    for name in BENCHMARK_NAMES:
        explicit.extend([
            repo_root / "data" / "benchmark" / name,
            repo_root / "data" / "benchmark" / "benchmark_v3" / name,
            repo_root / "data" / "benchmark" / "benchmark_v2" / name,
            repo_root / "data" / "processed" / "methodkg_outputs_v7_clustered_from_cleaned" / name,
        ])
    explicit.extend([
        repo_root / "data" / "benchmark" / "benchmark_v3.zip",
        repo_root / "data" / "benchmark" / "benchmark_v2.zip",
    ])
    for p in explicit:
        if valid_file(p):
            return p.resolve()
    globs: list[Path] = []
    for r in [repo_root / "data" / "benchmark", repo_root / "data" / "processed"]:
        if r.exists():
            for pat in ["*benchmark*v3*modeling*.csv", "*benchmark*v3*.zip", "*benchmark*v3*.csv", "*benchmark*modeling*.csv", "*benchmark*.zip"]:
                globs.extend(r.rglob(pat))
    best = _pick_best(globs)
    if best:
        return best.resolve()
    raise FileNotFoundError("Could not find MethodKG benchmark v3 under data/benchmark. Pass --benchmark explicitly.")


def discover_awards(repo_root: Path) -> Path:
    explicit = [
        repo_root / "data" / "processed" / "methodkg_outputs_v7_clustered_from_cleaned" / "cleaned_nsf_awards_2000_2025.csv",
        repo_root / "data" / "processed" / "cleaned_nsf_awards_2000_2025.csv",
        repo_root / "data" / "processed" / "methodkg_outputs_v7_clustered_from_cleaned" / "nsf_awards_cleaned.csv",
    ]
    for p in explicit:
        if valid_file(p):
            return p.resolve()
    cand: list[Path] = []
    for r in [repo_root / "data" / "processed" / "methodkg_outputs_v7_clustered_from_cleaned", repo_root / "data" / "processed", repo_root / "data" / "raw"]:
        if r.exists():
            for p in r.rglob("*.csv"):
                n = p.name.lower()
                if valid_file(p) and "award" in n and "benchmark" not in n and "edge" not in n and "pi" not in n and "copi" not in n:
                    cand.append(p)
    best = _pick_best(cand)
    if best:
        return best.resolve()
    raise FileNotFoundError("Could not find cleaned awards CSV. Pass --awards explicitly.")


def discover_award_pi_edges(repo_root: Path) -> Optional[Path]:
    explicit = [
        repo_root / "data" / "edges" / "award_pi_edges.csv",
        repo_root / "data" / "edges" / "award_person_edges.csv",
        repo_root / "data" / "processed" / "methodkg_outputs_v7_clustered_from_cleaned" / "award_pi_edges.csv",
    ]
    for p in explicit:
        if valid_file(p):
            return p.resolve()
    cand: list[Path] = []
    for r in [repo_root / "data" / "edges", repo_root / "data" / "processed" / "methodkg_outputs_v7_clustered_from_cleaned", repo_root / "data" / "processed"]:
        if r.exists():
            for p in r.rglob("*.csv"):
                n = p.name.lower()
                if valid_file(p) and "award" in n and ("pi" in n or "person" in n) and "edge" in n:
                    cand.append(p)
    best = _pick_best(cand)
    return best.resolve() if best else None


def csv_names_in_zip(path: Path) -> list[str]:
    with zipfile.ZipFile(path, "r") as zf:
        return [n for n in zf.namelist() if n.lower().endswith(".csv") and not is_sidecar_name(n)]


def choose_csv_from_zip(path: Path, preferred_name: str | None = None) -> str:
    names = csv_names_in_zip(path)
    if not names:
        raise ValueError(f"No CSV files found inside zip: {path}")
    if preferred_name:
        for n in names:
            if Path(n).name == preferred_name:
                return n
    for preferred in BENCHMARK_NAMES:
        for n in names:
            if Path(n).name == preferred:
                return n
    modeling = [n for n in names if "modeling" in Path(n).name.lower()]
    if modeling:
        return sorted(modeling, key=lambda n: _version_score(Path(n)), reverse=True)[0]
    return sorted(names, key=lambda n: _version_score(Path(n)), reverse=True)[0]


def read_csv_or_zip(path: str | Path, preferred_name: str | None = None, **kwargs) -> pd.DataFrame:
    p = Path(path)
    if is_sidecar_name(str(p)):
        raise ValueError(f"Refusing to read sidecar/hidden file as CSV: {p}")
    kwargs.setdefault("low_memory", False)
    if p.suffix.lower() == ".zip":
        target = choose_csv_from_zip(p, preferred_name=preferred_name)
        with zipfile.ZipFile(p, "r") as zf:
            with zf.open(target) as f:
                return pd.read_csv(f, **kwargs)
    return pd.read_csv(p, **kwargs)


def existing_or_none(path: Path) -> Optional[Path]:
    return path.resolve() if valid_file(path) else None


def default_text_embeddings_minilm(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "features" / "text_embeddings_minilm_v1" / "methodkg_text_embeddings.csv"


def default_text_embeddings_scibert(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "features" / "text_embeddings_scibert_mean_v1" / "methodkg_text_embeddings.csv"


def default_graph_features(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "features" / "graph_features_v1" / "methodkg_graph_only_features.csv"


def default_node2vec_embeddings(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "features" / "walk_embeddings_v1" / "node2vec_award_embeddings.csv"


def default_metapath2vec_embeddings(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "features" / "walk_embeddings_v1" / "metapath2vec_award_embeddings.csv"


def default_simteg_text_embeddings(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "features" / "simteg_text_embeddings_scibert_v1" / "methodkg_simteg_text_embeddings.csv"


def default_text_hgt_embeddings(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "features" / "text_hgt_embeddings_scibert_v1" / "methodkg_simteg_text_embeddings.csv"


def default_simteg_graph_dir(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "graphs" / "simteg_graphsage_data_scibert_v1"


def default_text_hgt_graph_dir(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "graphs" / "text_hgt_data_scibert_v1"


def write_resolved_paths(**items: object) -> None:
    print("Resolved paths:", flush=True)
    for k, v in items.items():
        print(f"  {k}: {v}", flush=True)


def write_metrics_rollup(results_root: Path, paper_outputs: Path, summary_name: str, table_name: str) -> None:
    rows = []
    for pattern in ["metrics.csv", "metrics_all_models.csv"]:
        for path in results_root.rglob(pattern):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            rel = path.parent.relative_to(results_root)
            parts = rel.parts
            df.insert(0, "result_dir", str(rel))
            if len(parts) >= 1 and "target_folder" not in df.columns:
                df.insert(1, "target_folder", parts[0])
            if len(parts) >= 2 and "split_folder" not in df.columns:
                df.insert(2, "split_folder", parts[1])
            if len(parts) >= 3 and "model_family" not in df.columns:
                df.insert(3, "model_family", parts[2])
            rows.append(df)
    if not rows:
        return
    summary = pd.concat(rows, ignore_index=True)
    summaries = paper_outputs / "summaries"
    tables = paper_outputs / "tables"
    summaries.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summaries / summary_name, index=False, encoding="utf-8-sig")
    if "split" in summary.columns:
        mask = summary["split"].astype(str).str.lower().eq("test")
    elif "split_folder" in summary.columns:
        mask = summary["split_folder"].astype(str).str.contains("test", case=False, na=False)
    else:
        mask = pd.Series([True] * len(summary), index=summary.index)
    summary[mask].to_csv(tables / table_name, index=False, encoding="utf-8-sig")
    print("Wrote paper rollups:", summaries / summary_name, "and", tables / table_name)
