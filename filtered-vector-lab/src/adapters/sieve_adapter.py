#!/usr/bin/env python3
"""
SIEVE adapter: runs SIEVE via biganntest/run.py and parses HDF5 results.

Key fixes applied to upstream for Python 3.8 compatibility:
  1. neurips23/streaming/run.py: converted match/case → if/elif/else
  2. neurips23/filter/{sieve,smarthnsw,prefilter,hnswbase}/config.yaml:
     replaced single-quoted JSON strings with double quotes
  3. neurips23/filter/sieve/sieve.py:
     - UnionFind import → networkx.utils.union_find.UnionFind
     - __init__: added .get() defaults for all index_params
  4. Generated random-filter100000-filters.pkl to match dataset metadata
     (original yfcc10M-filters.pkl has 200k filters vs dataset's 5)
"""

import os
import sys
import json
import h5py
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIEVE_ROOT = PROJECT_ROOT / "third_party" / "SIEVE" / "biganntest"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_OUT = PROJECT_ROOT / "results" / "raw" / "sieve"
NORM_OUT = PROJECT_ROOT / "results" / "normalized"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from storage.result_store import merge_all_normalized


def run_sieve_experiment(
    dataset: str = "random-filter-s",
    count: int = 10,
    track: str = "filter",
    algorithm: str = "sieve",
    nodocker: bool = True,
    pythonpath: str = None,
):
    """Run SIEVE experiment via biganntest/run.py."""
    cmd = [
        "python3", "run.py",
        f"--neurips23track", track,
        f"--algorithm", algorithm,
        f"--dataset", dataset,
        f"--count", str(count),
    ]
    if nodocker:
        cmd.append("--nodocker")

    env = os.environ.copy()
    pypath = str(SIEVE_ROOT)
    if pythonpath:
        pypath = f"{pythonpath}:{pypath}"
    # Also need SIEVE's custom hnswlib on PYTHONPATH
    hnswpath = str(PROJECT_ROOT / "third_party" / "SIEVE" / "hnswtest")
    pypath = f"{pypath}:{hnswpath}"
    env["PYTHONPATH"] = pypath

    print(f"[sieve_adapter] Running: {' '.join(cmd)}")
    print(f"[sieve_adapter] PYTHONPATH={pypath}")
    result = subprocess.run(cmd, cwd=SIEVE_ROOT, env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("[sieve_adapter] STDERR:", result.stderr)
        raise RuntimeError(f"SIEVE run failed with code {result.returncode}")
    return result


def find_result_hdf5(dataset: str, count: int, algorithm: str = "sieve") -> Path:
    """Locate the HDF5 result file produced by SIEVE."""
    result_dir = SIEVE_ROOT / "results" / "neurips23" / "filter" / dataset / str(count) / algorithm
    files = list(result_dir.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 result found in {result_dir}")
    # Return the most recently modified file
    return max(files, key=lambda p: p.stat().st_mtime)


def read_ground_truth(dataset: str):
    """Read big-ann-benchmarks format ground-truth file."""
    # Dataset short name in SIEVE: random-filter-s -> random-filter100000
    ds_short = dataset.replace("-s", "") if dataset.endswith("-s") else dataset
    gt_dir = SIEVE_ROOT / "data" / f"{ds_short}100000"
    gt_file = gt_dir / f"gt_100000_1000_50"
    if not gt_file.exists():
        # Try generic pattern
        candidates = list(gt_dir.glob("gt_*"))
        if not candidates:
            raise FileNotFoundError(f"No ground truth file found in {gt_dir}")
        gt_file = candidates[0]

    with open(gt_file, "rb") as f:
        n, d = np.fromfile(f, dtype=np.uint32, count=2)
        I = np.fromfile(f, dtype=np.int32, count=n * d).reshape(n, d)
        D = np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)
    return I, D, n, d


def compute_recall(neighbors: np.ndarray, gt: np.ndarray, k: int = 10) -> float:
    """Compute average recall@k."""
    recalls = []
    n_queries = min(neighbors.shape[0], gt.shape[0])
    for i in range(n_queries):
        gt_set = set(gt[i, :k])
        res_set = set(neighbors[i, :k])
        recalls.append(len(gt_set & res_set) / k)
    return float(np.mean(recalls))


def parse_sieve_result(hdf5_path: Path, dataset: str, count: int) -> pd.DataFrame:
    """Parse SIEVE HDF5 result into unified schema DataFrame."""
    with h5py.File(hdf5_path, "r") as f:
        attrs = dict(f.attrs)
        neighbors = f["neighbors"][:]

    gt, _, n_queries, gt_k = read_ground_truth(dataset)
    recall = compute_recall(neighbors, gt, k=count)

    best_time = float(attrs.get("best_search_time", 0))
    qps = 1000.0 / best_time if best_time > 0 else 0.0

    row = {
        "run_id": pd.NA,
        "project": "SIEVE",
        "algorithm": attrs.get("algo", "sieve"),
        "dataset": attrs.get("dataset", dataset),
        "track": "filter",
        "filter_type": "predicate",
        "filter_width": pd.NA,
        "selectivity": pd.NA,
        "k": int(attrs.get("count", count)),
        "params_json": json.dumps({"name": attrs.get("name", "Sieve")}, sort_keys=True),
        "recall": round(recall, 6),
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

    # Optionally re-run experiment
    # run_sieve_experiment(dataset=dataset, count=count)

    hdf5_path = find_result_hdf5(dataset, count)
    print(f"[sieve_adapter] Parsing result: {hdf5_path}")
    df = parse_sieve_result(hdf5_path, dataset, count)

    os.makedirs(NORM_OUT, exist_ok=True)
    out_csv = NORM_OUT / "sieve_real.csv"
    df.to_csv(out_csv, index=False)
    print(f"[sieve_adapter] Saved normalized CSV: {out_csv}")

    merge_all_normalized()
    print("[sieve_adapter] Merged into all_results.csv")


if __name__ == "__main__":
    main()
