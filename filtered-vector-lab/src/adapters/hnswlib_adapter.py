#!/usr/bin/env python3
"""
hnswlib Adapter：基础 HNSW 检索封装。

输入：
    data/processed/<dataset>/base.npy
    data/processed/<dataset>/query.npy
    data/processed/<dataset>/groundtruth.npy

输出：
    results/raw/hnswlib/<dataset>/summary.csv
    results/raw/hnswlib/<dataset>/detail.parquet

用法：
    python src/adapters/hnswlib_adapter.py --dataset synthetic-small --M 16 --ef_construction 200 --ef 50
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# 把项目根目录加入路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.recall import compute_recall_at_k, compute_recall_per_query
from src.storage.result_store import (
    create_empty_results_df,
    add_result_row,
    save_normalized_csv,
    normalize_dataframe,
)


def run_hnswlib_experiment(
    dataset: str,
    M: int = 16,
    ef_construction: int = 200,
    ef: int = 50,
    k: int = 10,
    space: str = "l2",
    num_threads: int = -1,
    data_dir: str = "data/processed",
    results_dir: str = "results/raw/hnswlib",
    normalized_dir: str = "results/normalized",
):
    """运行 hnswlib 实验并保存结果。"""

    dataset_dir = os.path.join(data_dir, dataset)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    # 加载数据
    base_path = os.path.join(dataset_dir, "base.npy")
    query_path = os.path.join(dataset_dir, "query.npy")
    gt_path = os.path.join(dataset_dir, "groundtruth.npy")

    print(f"[hnswlib] Loading base from {base_path}")
    base = np.load(base_path)
    print(f"[hnswlib] Loading query from {query_path}")
    query = np.load(query_path)
    print(f"[hnswlib] Loading groundtruth from {gt_path}")
    groundtruth = np.load(gt_path)

    num_base = base.shape[0]
    num_queries = query.shape[0]
    dim = base.shape[1]

    print(f"[hnswlib] Dataset: {dataset}, base={num_base}, queries={num_queries}, dim={dim}")
    print(f"[hnswlib] Params: M={M}, ef_construction={ef_construction}, ef={ef}, k={k}, space={space}")

    import hnswlib

    # 构建索引
    print("[hnswlib] Building index...")
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=num_base, ef_construction=ef_construction, M=M)
    index.set_num_threads(num_threads)

    t0 = time.time()
    index.add_items(base, np.arange(num_base))
    build_time = time.time() - t0
    print(f"[hnswlib] Build time: {build_time:.4f}s")

    # 查询
    print("[hnswlib] Querying...")
    index.set_ef(ef)

    t0 = time.time()
    labels, distances = index.knn_query(query, k=k)
    query_time = time.time() - t0
    print(f"[hnswlib] Query time: {query_time:.4f}s for {num_queries} queries")

    # 计算指标
    qps = num_queries / query_time
    avg_latency_ms = (query_time / num_queries) * 1000.0
    recall = compute_recall_at_k(groundtruth, labels, k=k)
    recalls_per_query = compute_recall_per_query(groundtruth, labels, k=k)

    print(f"[hnswlib] Recall@{k}: {recall:.4f}")
    print(f"[hnswlib] QPS: {qps:.2f}")
    print(f"[hnswlib] Avg latency: {avg_latency_ms:.4f} ms")

    # 保存 raw detail（parquet）
    raw_dir = os.path.join(results_dir, dataset)
    os.makedirs(raw_dir, exist_ok=True)

    detail_records = []
    for i in range(num_queries):
        for rank in range(k):
            detail_records.append({
                "query_id": i,
                "rank": rank,
                "vector_id": int(labels[i, rank]),
                "distance": float(distances[i, rank]),
                "is_groundtruth": int(labels[i, rank]) in set(groundtruth[i, :k]),
            })

    detail_df = pd.DataFrame(detail_records)
    detail_path = os.path.join(raw_dir, "detail.parquet")
    detail_df.to_parquet(detail_path, index=False)
    print(f"[hnswlib] Saved detail: {detail_path}")

    # 保存 raw summary（csv）
    params_json = {
        "M": M,
        "ef_construction": ef_construction,
        "ef": ef,
        "space": space,
        "num_threads": num_threads,
    }

    summary_df = create_empty_results_df()
    summary_df = add_result_row(
        summary_df,
        project="hnswlib",
        algorithm="hnsw",
        dataset=dataset,
        recall=recall,
        qps=qps,
        k=k,
        params_json=params_json,
        avg_latency_ms=avg_latency_ms,
        build_time_s=build_time,
        threads=num_threads,
        raw_result_path=os.path.abspath(detail_path),
    )

    summary_path = os.path.join(raw_dir, "summary.csv")
    save_normalized_csv(summary_df, summary_path)
    print(f"[hnswlib] Saved summary: {summary_path}")

    # 同时写入 normalized（方便后续合并）
    norm_path = os.path.join(normalized_dir, "hnswlib.csv")
    # 如果已有就追加，否则新建
    if os.path.exists(norm_path):
        existing = pd.read_csv(norm_path, dtype={"run_id": "string"})
        existing = normalize_dataframe(existing)
        combined = pd.concat([existing, summary_df], ignore_index=True)
    else:
        combined = summary_df

    save_normalized_csv(combined, norm_path)
    print(f"[hnswlib] Saved normalized: {norm_path}")

    return summary_df, detail_df


def main():
    parser = argparse.ArgumentParser(description="Run hnswlib baseline experiment")
    parser.add_argument("--dataset", default="synthetic-small", help="Dataset name")
    parser.add_argument("--M", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--ef-construction", type=int, default=200, help="ef_construction")
    parser.add_argument("--ef", type=int, default=50, help="ef (search parameter)")
    parser.add_argument("--k", type=int, default=10, help="Top-k")
    parser.add_argument("--space", default="l2", choices=["l2", "cosine", "ip"], help="Distance space")
    parser.add_argument("--num-threads", type=int, default=-1, help="Number of threads")
    parser.add_argument("--data-dir", default="data/processed", help="Data directory")
    parser.add_argument("--results-dir", default="results/raw/hnswlib", help="Results directory")
    args = parser.parse_args()

    run_hnswlib_experiment(
        dataset=args.dataset,
        M=args.M,
        ef_construction=args.ef_construction,
        ef=args.ef,
        k=args.k,
        space=args.space,
        num_threads=args.num_threads,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
