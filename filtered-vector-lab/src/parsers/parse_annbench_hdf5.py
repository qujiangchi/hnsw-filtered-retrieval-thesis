#!/usr/bin/env python3
"""
ANN-Benchmarks / VIBE 结果解析器。

支持两种输入模式：
1. 直接读取 HDF5 结果文件（results/<dataset>/<count>/<algo>/<args>.hdf5）
2. 读取 data_export.py 已生成的 CSV

对于 HDF5 模式，直接从文件 attrs 和 datasets 中提取：
- attrs: algo, build_time, index_size, best_search_time, count, dataset, distance, name, run_count
- datasets: times, neighbors, distances

输出：
    results/normalized/ann_benchmarks.csv（统一 Schema）

用法：
    # 模式1: 扫描 HDF5 目录
    python src/parsers/parse_annbench_hdf5.py --input results --output results/normalized/ann_benchmarks.csv

    # 模式2: 读取已有 CSV
    python src/parsers/parse_annbench_hdf5.py --csv-input exported.csv --output results/normalized/ann_benchmarks.csv
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def parse_hdf5_file(filepath: str) -> dict:
    """
    解析单个 ANN-Benchmarks HDF5 结果文件。

    Returns
    -------
    dict : 包含统一 Schema 字段的字典，解析失败返回 None
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required to parse ANN-Benchmarks HDF5 files. Install: pip install h5py")

    filepath = Path(filepath)
    if not filepath.exists():
        return None

    try:
        with h5py.File(filepath, "r") as f:
            attrs = dict(f.attrs)
            times = f["times"][:] if "times" in f else np.array([])
            neighbors = f["neighbors"][:] if "neighbors" in f else None
            distances = f["distances"][:] if "distances" in f else None
    except Exception as e:
        print(f"[parse_annbench] Error reading {filepath}: {e}")
        return None

    # 基础字段
    algorithm = attrs.get("algo", attrs.get("algorithm", ""))
    dataset = attrs.get("dataset", "")
    count = int(attrs.get("count", 0))
    name = attrs.get("name", "")
    build_time = float(attrs.get("build_time", -1))
    index_size = float(attrs.get("index_size", 0))
    best_search_time = float(attrs.get("best_search_time", 0))
    run_count = int(attrs.get("run_count", 1))
    distance = attrs.get("distance", "")
    batch_mode = bool(attrs.get("batch_mode", False))
    candidates = attrs.get("candidates", None)

    # 从文件名/路径推断 query arguments
    # 路径: results/<dataset>/<count>/<algo>/<args>.hdf5
    parts = filepath.parts
    params = {"batch_mode": batch_mode, "distance": distance}
    if len(parts) >= 2:
        params["raw_filename"] = filepath.name

    # 计算 latency 分位数（如果 times 存在）
    p50 = p95 = p99 = pd.NA
    if len(times) > 0:
        p50 = float(np.percentile(times, 50.0) * 1000.0)
        p95 = float(np.percentile(times, 95.0) * 1000.0)
        p99 = float(np.percentile(times, 99.0) * 1000.0)

    # QPS
    qps = 1.0 / best_search_time if best_search_time > 0 else pd.NA

    # distcomps（如果存在 additional attrs）
    distcomps = pd.NA
    if "dist_comps" in attrs:
        distcomps = float(attrs["dist_comps"])

    # recall 无法直接从 HDF5 attrs 得到，需要 ground truth distances 对比
    # 这里标记为需要从外部计算
    recall = pd.NA

    # 构建统一 Schema 行
    row = {
        "run_id": pd.NA,
        "project": "ann-benchmarks",
        "algorithm": str(algorithm),
        "dataset": str(dataset),
        "track": pd.NA,
        "filter_type": "none",
        "filter_width": pd.NA,
        "selectivity": pd.NA,
        "k": count,
        "params_json": json.dumps(params, sort_keys=True),
        "recall": recall,
        "qps": qps,
        "avg_latency_ms": best_search_time * 1000.0 if best_search_time > 0 else pd.NA,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "build_time_s": build_time if build_time >= 0 else pd.NA,
        "index_size_kb": index_size,
        "memory_kb": pd.NA,
        "distcomps": distcomps,
        "threads": pd.NA,
        "raw_result_path": str(filepath.resolve()),
        "created_at": pd.Timestamp.now(),
    }
    return row


def scan_hdf5_directory(input_dir: str) -> pd.DataFrame:
    """扫描目录下所有 HDF5 结果文件。"""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"[parse_annbench] Directory not found: {input_dir}")
        return pd.DataFrame()

    files = list(input_dir.rglob("*.hdf5"))
    if not files:
        print(f"[parse_annbench] No HDF5 files found under {input_dir}")
        return pd.DataFrame()

    rows = []
    for f in files:
        print(f"[parse_annbench] Parsing: {f}")
        row = parse_hdf5_file(str(f))
        if row:
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def parse_csv_export(csv_path: str) -> pd.DataFrame:
    """
    读取 ANN-Benchmarks data_export.py 生成的 CSV。
    列名可能包括：algorithm, parameters, count, dataset, k-nn, qps, build, indexsize, p50, p95, p99, distcomps
    """
    df = pd.read_csv(csv_path)
    print(f"[parse_annbench] Loaded CSV with columns: {list(df.columns)}")

    # 标准化列名
    col_map = {}
    for c in df.columns:
        lc = c.strip().lower().replace("-", "_")
        if lc in ["algorithm", "parameters", "count", "dataset", "k_nn", "qps",
                  "build", "indexsize", "p50", "p95", "p99", "distcomps", "epsilon",
                  "largeepsilon", "rel", "candidates", "queriessize", "batch_mode"]:
            col_map[c] = lc
    df = df.rename(columns=col_map)

    # 构建统一 Schema
    result = pd.DataFrame()
    result["run_id"] = [pd.NA] * len(df)
    result["project"] = "ann-benchmarks"
    result["algorithm"] = df.get("algorithm", pd.NA)
    result["dataset"] = df.get("dataset", pd.NA)
    result["track"] = pd.NA
    result["filter_type"] = "none"
    result["filter_width"] = pd.NA
    result["selectivity"] = pd.NA
    result["k"] = pd.to_numeric(df.get("count"), errors="coerce").astype("Int64")
    result["params_json"] = df.get("parameters", pd.NA).apply(
        lambda x: json.dumps({"raw_parameters": str(x)}, sort_keys=True) if pd.notna(x) else "{}"
    )
    result["recall"] = pd.to_numeric(df.get("k_nn"), errors="coerce")
    result["qps"] = pd.to_numeric(df.get("qps"), errors="coerce")
    result["avg_latency_ms"] = pd.NA
    result["p50_latency_ms"] = pd.to_numeric(df.get("p50"), errors="coerce")
    result["p95_latency_ms"] = pd.to_numeric(df.get("p95"), errors="coerce")
    result["p99_latency_ms"] = pd.to_numeric(df.get("p99"), errors="coerce")
    result["build_time_s"] = pd.to_numeric(df.get("build"), errors="coerce")
    result["index_size_kb"] = pd.to_numeric(df.get("indexsize"), errors="coerce")
    result["memory_kb"] = pd.NA
    result["distcomps"] = pd.to_numeric(df.get("distcomps"), errors="coerce")
    result["threads"] = pd.NA
    result["raw_result_path"] = str(Path(csv_path).resolve())
    result["created_at"] = pd.Timestamp.now()

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse ANN-Benchmarks results to unified schema")
    parser.add_argument("--input", help="Input HDF5 directory to scan")
    parser.add_argument("--csv-input", help="Input CSV file from data_export.py")
    parser.add_argument("--output", required=True, help="Output normalized CSV path")
    args = parser.parse_args()

    if args.csv_input:
        df = parse_csv_export(args.csv_input)
    elif args.input:
        df = scan_hdf5_directory(args.input)
    else:
        parser.error("Either --input (HDF5 dir) or --csv-input (CSV file) must be provided")

    if df.empty:
        print("[parse_annbench] No data parsed.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[parse_annbench] Saved normalized CSV: {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
