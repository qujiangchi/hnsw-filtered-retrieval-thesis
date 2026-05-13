#!/usr/bin/env python3
"""
ACORN adapter (via SIEVE biganntest filter track).

Note: ACORN's upstream implementation (acorngamma.py) has hardcoded dataset paths
for 'msong' and is not generically runnable on random-filter-s without code changes.
This adapter provides the framework but skips actual execution for unsupported datasets.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.result_store import merge_all_normalized


def run_acorn_experiment(
    dataset: str = "msong",
    count: int = 10,
    algorithm: str = "acorngamma",
    track: str = "filter",
):
    """Run ACORN experiment via biganntest/run.py."""
    sieve_root = PROJECT_ROOT / "third_party" / "SIEVE" / "biganntest"
    if not sieve_root.exists():
        raise FileNotFoundError(f"SIEVE biganntest not found at {sieve_root}")

    # ACORN requires specific datasets; random-filter-s has hardcoded paths
    if dataset not in ("msong", "paper"):
        print(f"[acorn_adapter] WARNING: ACORN upstream has hardcoded paths for 'msong'.")
        print(f"[acorn_adapter] Skipping actual run for dataset={dataset}.")
        return None

    import subprocess
    cmd = [
        "python3", "run.py",
        "--neurips23track", track,
        "--algorithm", algorithm,
        "--dataset", dataset,
        "--count", str(count),
        "--nodocker",
    ]
    env = os.environ.copy()
    pypath = str(sieve_root)
    hnswpath = str(PROJECT_ROOT / "third_party" / "SIEVE" / "hnswtest")
    env["PYTHONPATH"] = f"{pypath}:{hnswpath}"

    print(f"[acorn_adapter] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=sieve_root, env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("[acorn_adapter] STDERR:", result.stderr)
        raise RuntimeError(f"ACORN run failed with code {result.returncode}")
    return result


def parse_acorn_result(dataset: str, count: int, algorithm: str = "acorngamma") -> pd.DataFrame:
    """Parse ACORN HDF5 result into unified schema."""
    import h5py
    result_dir = (
        PROJECT_ROOT / "third_party" / "SIEVE" / "biganntest"
        / "results" / "neurips23" / "filter" / dataset / str(count) / algorithm
    )
    files = list(result_dir.glob("*.hdf5"))
    if not files:
        print(f"[acorn_adapter] No HDF5 result found in {result_dir}")
        return pd.DataFrame()

    hdf5_path = max(files, key=lambda p: p.stat().st_mtime)
    with h5py.File(hdf5_path, "r") as f:
        attrs = dict(f.attrs)

    best_time = float(attrs.get("best_search_time", 0))
    qps = 1000.0 / best_time if best_time > 0 else 0.0

    row = {
        "run_id": pd.NA,
        "project": "ACORN",
        "algorithm": attrs.get("algo", algorithm),
        "dataset": attrs.get("dataset", dataset),
        "track": "filter",
        "filter_type": "predicate",
        "filter_width": pd.NA,
        "selectivity": pd.NA,
        "k": int(attrs.get("count", count)),
        "params_json": json.dumps({"name": attrs.get("name", algorithm)}, sort_keys=True),
        "recall": pd.NA,  # HDF5 does not include recall
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
        "raw_result_path": str(hdf5_path.resolve()),
        "created_at": pd.Timestamp.now(),
    }
    return pd.DataFrame([row])


def main():
    dataset = "random-filter-s"
    count = 10
    algorithm = "acorngamma"

    try:
        run_acorn_experiment(dataset=dataset, count=count, algorithm=algorithm)
        df = parse_acorn_result(dataset, count, algorithm)
    except Exception as e:
        print(f"[acorn_adapter] Run failed: {e}")
        df = None

    if df is None or df.empty:
        df = pd.DataFrame([{
            "run_id": pd.NA,
            "project": "ACORN",
            "algorithm": algorithm,
            "dataset": dataset,
            "track": "filter",
            "filter_type": "predicate",
            "filter_width": pd.NA,
            "selectivity": pd.NA,
            "k": count,
            "params_json": json.dumps({"status": "not_runnable"}, sort_keys=True),
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
    out_csv = PROJECT_ROOT / "results" / "normalized" / "acorn.csv"
    df.to_csv(out_csv, index=False)
    print(f"[acorn_adapter] Saved: {out_csv}")
    merge_all_normalized()
    print("[acorn_adapter] Merged into all_results.csv")


if __name__ == "__main__":
    main()
