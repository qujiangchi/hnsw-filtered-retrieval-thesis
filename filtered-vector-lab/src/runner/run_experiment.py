#!/usr/bin/env python3
"""
Config-driven experiment runner.

Reads configs/experiments.yaml and dispatches to algorithm adapters.

Usage:
    python src/runner/run_experiment.py --config configs/experiments.yaml
    python src/runner/run_experiment.py --project hnswlib --dataset synthetic-small
"""

import os
import sys
import yaml
import time
import argparse
import itertools
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.result_store import merge_all_normalized


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cartesian_product(params: Dict[str, list]) -> List[Dict[str, Any]]:
    """Generate cartesian product of parameter lists."""
    keys = list(params.keys())
    values = [params[k] if isinstance(params[k], list) else [params[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_hnswlib(config: dict, dataset: str, k: int):
    from src.adapters.hnswlib_adapter import run_hnswlib_experiment

    params = config.get("params", {})
    combos = cartesian_product(params)
    print(f"[runner] hnswlib: {len(combos)} parameter combinations")

    for combo in combos:
        print(f"[runner] hnswlib running: {combo}")
        try:
            run_hnswlib_experiment(
                dataset=dataset,
                M=combo.get("M", 16),
                ef_construction=combo.get("ef_construction", 200),
                ef=combo.get("ef", 50),
                k=k,
                space="l2",
                num_threads=combo.get("num_threads", -1),
            )
        except Exception as e:
            print(f"[runner] hnswlib FAILED: {e}")


def run_range_filtered_ann(config: dict, dataset: str, k: int):
    from src.adapters.range_filtered_ann_adapter import run_range_filtered_ann_experiment

    methods = config.get("methods", ["prefiltering"])
    filter_widths = config.get("filter_width", ["2pow-4"])
    threads = config.get("threads", 4)

    # Current adapter runs all methods internally; we call it once per dataset
    print(f"[runner] RangeFilteredANN: methods={methods}, widths={filter_widths}")
    try:
        # TODO: extend adapter to accept methods/widths explicitly
        run_range_filtered_ann_experiment(
            dataset=dataset,
            threads=threads,
        )
    except Exception as e:
        print(f"[runner] RangeFilteredANN FAILED: {e}")


def run_sieve(config: dict, dataset: str, k: int):
    from src.adapters.sieve_adapter import run_sieve_experiment, find_result_hdf5, parse_sieve_result

    algorithms = config.get("algorithms", ["sieve"])
    track = config.get("track", "filter")
    count = k

    for algo in algorithms:
        print(f"[runner] SIEVE running algorithm={algo}, dataset={dataset}, count={count}")
        try:
            run_sieve_experiment(
                dataset=dataset,
                count=count,
                track=track,
                algorithm=algo,
                nodocker=True,
            )
            hdf5_path = find_result_hdf5(dataset, count, algo)
            df = parse_sieve_result(hdf5_path, dataset, count)

            os.makedirs(PROJECT_ROOT / "results" / "normalized", exist_ok=True)
            out_csv = PROJECT_ROOT / "results" / "normalized" / "sieve_real.csv"
            # If multiple algorithms, append
            if out_csv.exists():
                existing = pd.read_csv(out_csv)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_csv(out_csv, index=False)
            print(f"[runner] SIEVE result saved to {out_csv}")
        except Exception as e:
            print(f"[runner] SIEVE FAILED: {e}")


def dispatch_experiment(exp: dict, dataset: str, k: int):
    project = exp.get("project")
    if not exp.get("enabled", True):
        print(f"[runner] Skipping disabled project: {project}")
        return

    print(f"\n{'='*60}")
    print(f"[runner] Dispatching: {project}")
    print(f"{'='*60}")

    if project == "hnswlib":
        run_hnswlib(exp, dataset, k)
    elif project == "RangeFilteredANN":
        run_range_filtered_ann(exp, dataset, k)
    elif project == "SIEVE":
        run_sieve(exp, dataset, k)
    else:
        print(f"[runner] Unknown project: {project}")


def main():
    parser = argparse.ArgumentParser(description="Run experiments from config")
    parser.add_argument("--config", default="configs/experiments.yaml", help="Experiment config YAML")
    parser.add_argument("--project", help="Run only this project")
    parser.add_argument("--dataset", help="Override dataset")
    parser.add_argument("--k", type=int, help="Override k")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = args.dataset or cfg.get("dataset", "synthetic-small")
    k = args.k or cfg.get("k", 10)

    experiments = cfg.get("experiments", [])
    if not experiments:
        print("[runner] No experiments defined.")
        return

    start_time = time.time()
    for exp in experiments:
        if args.project and exp.get("project") != args.project:
            continue
        dispatch_experiment(exp, dataset, k)

    print(f"\n{'='*60}")
    print("[runner] Merging all normalized results...")
    merge_all_normalized()
    print(f"[runner] Done in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    import pandas as pd  # lazy import for sieve append
    main()
