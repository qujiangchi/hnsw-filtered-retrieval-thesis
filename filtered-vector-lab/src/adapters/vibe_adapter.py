#!/usr/bin/env python3
"""
VIBE adapter.

VIBE requires Apptainer/Singularity containers by default, but supports --local mode.
Most algorithms need specific datasets downloaded from HuggingFace.
This adapter attempts local mode on a small subset; falls back to placeholder if unavailable.
"""

import os
import sys
import json
import subprocess
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.result_store import merge_all_normalized

VIBE_ROOT = PROJECT_ROOT / "third_party" / "vibe"


def run_vibe_experiment(
    dataset: str = "synthetic-small",
    count: int = 10,
    algorithm: str = "hnswlib",
    local: bool = True,
):
    """Run VIBE experiment."""
    if not VIBE_ROOT.exists():
        raise FileNotFoundError(f"VIBE not found at {VIBE_ROOT}")

    # VIBE's built-in datasets are different from our synthetic-small.
    # We can try running with a VIBE-supported dataset like "glove-25-angular"
    # but that requires downloading the dataset first.
    vibe_datasets = ["glove-25-angular", "sift-128-euclidean", "fashion-mnist-784-euclidean"]
    if dataset not in vibe_datasets:
        print(f"[vibe_adapter] Dataset '{dataset}' not in VIBE built-in list.")
        print(f"[vibe_adapter] Available: {vibe_datasets}")
        print(f"[vibe_adapter] Skipping run; would need dataset download + container/local setup.")
        return None

    cmd = [
        "python3", "run.py",
        "--dataset", dataset,
        "--count", str(count),
        "--algorithm", algorithm,
        "--runs", "3",
    ]
    if local:
        cmd.append("--local")

    print(f"[vibe_adapter] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=VIBE_ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("[vibe_adapter] STDERR:", result.stderr)
        raise RuntimeError(f"VIBE run failed with code {result.returncode}")
    return result


def parse_vibe_results(dataset: str, count: int, algorithm: str = "hnswlib") -> pd.DataFrame:
    """Parse VIBE results from its results directory."""
    # VIBE stores results in hdf5 under results/
    import h5py
    result_dir = VIBE_ROOT / "results" / dataset / str(count)
    if not result_dir.exists():
        return pd.DataFrame()

    files = list(result_dir.glob(f"{algorithm}*.hdf5"))
    if not files:
        return pd.DataFrame()

    rows = []
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            attrs = dict(f.attrs)
            # VIBE stores times, knn, etc.
            best_time = float(attrs.get("best_search_time", 0))
            qps = 1000.0 / best_time if best_time > 0 else 0.0
            rows.append({
                "run_id": pd.NA,
                "project": "VIBE",
                "algorithm": algorithm,
                "dataset": dataset,
                "track": pd.NA,
                "filter_type": pd.NA,
                "filter_width": pd.NA,
                "selectivity": pd.NA,
                "k": count,
                "params_json": json.dumps({"file": str(fpath.name)}, sort_keys=True),
                "recall": pd.to_numeric(attrs.get("recall", pd.NA), errors="coerce"),
                "qps": round(qps, 2),
                "avg_latency_ms": round(best_time * 1000, 3),
                "p50_latency_ms": pd.NA,
                "p95_latency_ms": pd.NA,
                "p99_latency_ms": pd.NA,
                "build_time_s": float(attrs.get("build_time", 0)),
                "index_size_kb": float(attrs.get("index_size", 0)),
                "memory_kb": pd.NA,
                "distcomps": pd.NA,
                "threads": pd.NA,
                "raw_result_path": str(fpath.resolve()),
                "created_at": pd.Timestamp.now(),
            })
    return pd.DataFrame(rows)


def main():
    dataset = "synthetic-small"
    count = 10
    algorithm = "hnswlib"

    error_reason = ""
    try:
        run_vibe_experiment(dataset=dataset, count=count, algorithm=algorithm)
        df = parse_vibe_results(dataset, count, algorithm)
    except Exception as e:
        print(f"[vibe_adapter] Run failed: {e}")
        error_reason = str(e)
        df = None

    if df is None or df.empty:
        df = pd.DataFrame([{
            "run_id": pd.NA,
            "project": "VIBE",
            "algorithm": algorithm,
            "dataset": dataset,
            "track": pd.NA,
            "filter_type": pd.NA,
            "filter_width": pd.NA,
            "selectivity": pd.NA,
            "k": count,
            "params_json": json.dumps({"status": "not_runnable", "reason": error_reason}, sort_keys=True),
            "recall": pd.NA,
            "qps": pd.NA,
            "avg_latency_ms": pd.NA,
            "p50_latency_ms": pd.NA,
            "p95_latency_ms": pd.NA,
            "p99_latency_ms": pd.NA,
            "build_time_s": pd.NA,
            "index_size_kb": pd.NA,
            "memory_kb": pd.NA,
            "distcomps": pd.NA,
            "threads": pd.NA,
            "raw_result_path": pd.NA,
            "created_at": pd.Timestamp.now(),
        }])

    os.makedirs(PROJECT_ROOT / "results" / "normalized", exist_ok=True)
    out_csv = PROJECT_ROOT / "results" / "normalized" / "vibe.csv"
    df.to_csv(out_csv, index=False)
    print(f"[vibe_adapter] Saved: {out_csv}")
    merge_all_normalized()
    print("[vibe_adapter] Merged into all_results.csv")


if __name__ == "__main__":
    main()
