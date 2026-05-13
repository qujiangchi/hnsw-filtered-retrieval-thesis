#!/usr/bin/env python3
"""
RangeFilteredANN Adapter：用 synthetic-small 数据运行最小实验。

把 synthetic-small 数据转换为 RangeFilteredANN 期望的格式：
    <dataset>.npy
    <dataset>_queries.npy
    <dataset>_filter-values.npy
    <dataset>_queries<filter_width>ranges.npy
    <dataset>_queries<filter_width>gt.npy

运行命令：
    python src/adapters/range_filtered_ann_adapter.py --dataset synthetic-small --threads 4
"""

import os
import sys
import time
import json
import argparse
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 添加 RangeFilteredANN experiments 到路径
RFANN_ROOT = PROJECT_ROOT / "third_party" / "RangeFilteredANN"
sys.path.insert(0, str(RFANN_ROOT / "experiments"))

from src.storage.result_store import create_empty_results_df, add_result_row, save_normalized_csv, normalize_dataframe


def prepare_dataset(dataset_name: str, data_dir: str, output_dir: str):
    """把 synthetic-small 数据转换为 RangeFilteredANN 格式。"""
    src = Path(data_dir) / dataset_name
    dst = Path(output_dir) / dataset_name
    dst.mkdir(parents=True, exist_ok=True)

    base = np.load(src / "base.npy")
    query = np.load(src / "query.npy")
    labels = np.load(src / "labels.npy")
    window_queries = np.load(src / "window_queries.npy")
    groundtruth = np.load(src / "groundtruth.npy")

    np.save(dst / f"{dataset_name}.npy", base)
    np.save(dst / f"{dataset_name}_queries.npy", query)
    np.save(dst / f"{dataset_name}_filter-values.npy", labels)

    # 使用已有的 window_queries 作为 ranges
    # synthetic-small 的 window_queries 是随机生成的固定范围
    num_queries = min(len(query), len(window_queries))
    for fw_name in ["2pow-4", "2pow-8"]:
        # 直接使用 generate_synthetic.py 生成的 window_queries 和 groundtruth
        ranges = window_queries[:num_queries]
        gt_list = groundtruth[:num_queries]
        np.save(dst / f"{dataset_name}_queries{fw_name}ranges.npy", np.array(ranges))
        np.save(dst / f"{dataset_name}_queries{fw_name}gt.npy", np.array(gt_list))

    print(f"[RFANN] Prepared dataset at {dst}")
    return str(dst)


def run_range_filtered_ann_experiment(
    dataset: str = "synthetic-small",
    threads: int = 4,
    data_dir: str = "data/processed",
    results_dir: str = "results/raw/range_filtered_ann",
    normalized_dir: str = "results/normalized",
):
    dataset_dir = prepare_dataset(dataset, data_dir, "/tmp/rfann_datasets")

    os.makedirs(results_dir, exist_ok=True)

    # 设置环境
    os.environ["PARLAY_NUM_THREADS"] = str(threads)

    # 由于 run_our_method.py 的接口较复杂，我们直接调用 wrapper 做一个最小测试
    # 先测试 prefiltering
    print(f"[RFANN] Running prefiltering on {dataset}...")
    try:
        import wrapper as wp
    except ImportError as e:
        print(f"[RFANN] Error importing wrapper: {e}")
        return None

    base = np.load(Path(dataset_dir) / f"{dataset}.npy")
    queries = np.load(Path(dataset_dir) / f"{dataset}_queries.npy")
    labels = np.load(Path(dataset_dir) / f"{dataset}_filter-values.npy")
    n, d = base.shape

    metric = "Euclidian"
    dtype = "float"

    # Prefiltering
    print("[RFANN] Building prefiltering index...")
    t0 = time.time()
    index_constructor = wp.prefilter_index_constructor(metric, dtype)
    build_params = wp.BuildParams(64, 500, 1.175, "/tmp/rfann_index_cache")
    index = index_constructor(base, labels.astype(np.float32), build_params)
    build_time = time.time() - t0
    print(f"[RFANN] Prefiltering build time: {build_time:.3f}s")

    # 查询
    fw_name = "2pow-4"
    ranges = np.load(Path(dataset_dir) / f"{dataset}_queries{fw_name}ranges.npy")
    gt = np.load(Path(dataset_dir) / f"{dataset}_queries{fw_name}gt.npy")
    num_queries = len(ranges)

    print(f"[RFANN] Querying {num_queries} queries with filter_width {fw_name}...")
    query_params = wp.QueryParams(10, 50, 1.35, 10000000, 10000, 1, 10000, None, False)
    filters = [(int(ranges[i][0]), int(ranges[i][1])) for i in range(num_queries)]

    t0 = time.time()
    neighbors, distances = index.batch_search(queries[:num_queries], filters, num_queries, query_params)
    query_time = time.time() - t0
    qps = num_queries / query_time

    # 计算 recall
    recall = 0.0
    for i in range(num_queries):
        gt_set = set(gt[i])
        res_set = set(neighbors[i][:10])
        if len(gt_set) > 0:
            recall += len(gt_set & res_set) / len(gt_set)
    recall /= num_queries

    print(f"[RFANN] Recall@10: {recall:.4f}, QPS: {qps:.2f}, Avg latency: {query_time/num_queries*1000:.4f} ms")

    # 保存结果
    summary_df = create_empty_results_df()
    summary_df = add_result_row(
        summary_df,
        project="RangeFilteredANN",
        algorithm="prefiltering",
        dataset=dataset,
        recall=recall,
        qps=qps,
        k=10,
        params_json={"method": "prefiltering", "filter_width": fw_name},
        avg_latency_ms=(query_time / num_queries) * 1000.0,
        build_time_s=build_time,
        threads=threads,
        filter_type="range",
        filter_width=fw_name,
        selectivity=1.0 / 16,
        raw_result_path=os.path.abspath(os.path.join(results_dir, f"{dataset}_prefiltering.csv")),
    )

    summary_path = os.path.join(results_dir, f"{dataset}_summary.csv")
    save_normalized_csv(summary_df, summary_path)

    # 合并到 normalized
    norm_path = os.path.join(normalized_dir, "range_filtered_ann_real.csv")
    if os.path.exists(norm_path):
        existing = pd.read_csv(norm_path, dtype={"run_id": "string"})
        existing = normalize_dataframe(existing)
        combined = pd.concat([existing, summary_df], ignore_index=True)
    else:
        combined = summary_df
    save_normalized_csv(combined, norm_path)
    print(f"[RFANN] Saved normalized: {norm_path}")

    # 同时更新 all_results.csv
    from src.storage.result_store import merge_all_normalized
    merge_all_normalized()

    return summary_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic-small")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()

    run_range_filtered_ann_experiment(
        dataset=args.dataset,
        threads=args.threads,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
