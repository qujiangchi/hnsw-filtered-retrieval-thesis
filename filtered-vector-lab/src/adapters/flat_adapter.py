#!/usr/bin/env python3
"""
Flat Adapter：暴力精确搜索（Flat / Brute Force）。
作为召回率上限基线。
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.result_store import create_empty_results_df, add_result_row, save_normalized_csv, normalize_dataframe


def run_flat_experiment(
    dataset: str,
    k: int = 10,
    data_dir: str = "data/processed",
    results_dir: str = "results/raw/flat",
    normalized_dir: str = "results/normalized",
):
    dataset_dir = os.path.join(data_dir, dataset)
    base = np.load(os.path.join(dataset_dir, "base.npy"))
    query = np.load(os.path.join(dataset_dir, "query.npy"))
    groundtruth = np.load(os.path.join(dataset_dir, "groundtruth.npy"))

    num_base, dim = base.shape
    num_queries = query.shape[0]
    print(f"[flat] Dataset: {dataset}, base={num_base}, queries={num_queries}, dim={dim}")

    import faiss
    index = faiss.IndexFlatL2(dim)
    t0 = time.time()
    index.add(base)
    build_time = time.time() - t0

    t0 = time.time()
    _, labels = index.search(query, k)
    query_time = time.time() - t0

    qps = num_queries / query_time
    avg_latency_ms = (query_time / num_queries) * 1000.0

    # Recall computation
    recall = np.mean([
        len(set(labels[i, :k]) & set(groundtruth[i, :k])) / k
        for i in range(num_queries)
    ])

    print(f"[flat] Recall@{k}: {recall:.4f}, QPS: {qps:.2f}, Latency: {avg_latency_ms:.2f}ms, Build: {build_time:.2f}s")

    params_json = {"method": "brute_force", "metric": "l2"}
    summary_df = create_empty_results_df()
    summary_df = add_result_row(
        summary_df,
        project="flat",
        algorithm="flat",
        dataset=dataset,
        recall=recall,
        qps=qps,
        k=k,
        params_json=params_json,
        avg_latency_ms=avg_latency_ms,
        build_time_s=build_time,
        threads=1,
    )

    os.makedirs(results_dir, exist_ok=True)
    raw_path = os.path.join(results_dir, f"{dataset}_summary.csv")
    save_normalized_csv(summary_df, raw_path)

    norm_path = os.path.join(normalized_dir, "flat.csv")
    if os.path.exists(norm_path):
        existing = pd.read_csv(norm_path, dtype={"run_id": "string"})
        existing = normalize_dataframe(existing)
        combined = pd.concat([existing, summary_df], ignore_index=True)
    else:
        combined = summary_df
    save_normalized_csv(combined, norm_path)
    return summary_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic-small")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    run_flat_experiment(args.dataset, args.k)


if __name__ == "__main__":
    main()
