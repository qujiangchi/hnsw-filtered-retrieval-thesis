#!/usr/bin/env python3
"""
RangeFilteredANN 结果解析器。

输入：
    results/raw/range/<dataset>_results.csv

原始字段（注意：表头只有6列，实际数据有9列）：
    filter_width, method, recall, average_time, qps, threads, build_time, branching_factor, memory

输出：
    results/normalized/range_filtered_ann.csv（统一 Schema）

用法：
    python src/parsers/parse_range_results.py --input results/raw/range --output results/normalized/range_filtered_ann.csv
"""

import os
import re
import json
import argparse
import pandas as pd
from pathlib import Path


def parse_range_csv(filepath: str) -> pd.DataFrame:
    """
    解析单个 RangeFilteredANN 结果 CSV 文件。
    处理表头缺失后三列的问题。
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # 从文件名解析 dataset
    # 例如：sift-128-euclidean_results.csv -> dataset="sift-128-euclidean"
    match = re.match(r"(.+)_results\.csv", filepath.name)
    if match:
        dataset = match.group(1)
    else:
        dataset = filepath.stem

    # 读取 CSV，显式指定9列名（修复表头缺失问题）
    columns = [
        "filter_width", "method", "recall", "average_time", "qps",
        "threads", "build_time", "branching_factor", "memory"
    ]

    df = pd.read_csv(filepath, header=0, names=columns, engine="python")

    # 处理可能的空值和类型
    df["recall"] = pd.to_numeric(df["recall"], errors="coerce")
    df["average_time"] = pd.to_numeric(df["average_time"], errors="coerce")
    df["qps"] = pd.to_numeric(df["qps"], errors="coerce")
    df["threads"] = pd.to_numeric(df["threads"], errors="coerce").astype("Int64")
    df["build_time"] = pd.to_numeric(df["build_time"], errors="coerce")
    df["branching_factor"] = pd.to_numeric(df["branching_factor"], errors="coerce")
    df["memory"] = pd.to_numeric(df["memory"], errors="coerce")

    # 解析 method 列：拆分为 algorithm + params_json
    def parse_method(m):
        if pd.isna(m):
            return "unknown", {}
        parts = str(m).split("_")
        algo = parts[0]
        params = {}
        if len(parts) > 1:
            # 尝试解析参数，格式不固定，如：postfiltering_1.000_2_10_1
            params["raw_method"] = str(m)
            # 简单启发式：如果后面部分看起来像数字，就记录下来
            for i, p in enumerate(parts[1:], start=1):
                try:
                    val = float(p)
                    params[f"param_{i}"] = val
                except ValueError:
                    params[f"param_{i}"] = p
        return algo, params

    parsed = df["method"].apply(parse_method)
    df["algorithm"] = [p[0] for p in parsed]
    df["params_json"] = [json.dumps(p[1], sort_keys=True) for p in parsed]

    # 构建统一 Schema
    result = pd.DataFrame()
    result["run_id"] = [pd.NA] * len(df)  # 由 store 重新生成
    result["project"] = "RangeFilteredANN"
    result["algorithm"] = df["algorithm"]
    result["dataset"] = dataset
    result["track"] = pd.NA
    result["filter_type"] = "range"
    result["filter_width"] = df["filter_width"].astype(str)
    # 尝试从 filter_width 推断 selectivity
    def infer_selectivity(fw):
        if pd.isna(fw):
            return pd.NA
        s = str(fw).strip()
        if s.startswith("2pow-"):
            try:
                exp = int(s.replace("2pow-", ""))
                return 2.0 ** (-exp)
            except ValueError:
                return pd.NA
        return pd.NA
    result["selectivity"] = df["filter_width"].apply(infer_selectivity)
    result["k"] = 10  # RangeFilteredANN 硬编码 TOP_K=10
    result["params_json"] = df["params_json"]
    result["recall"] = df["recall"]
    result["qps"] = df["qps"]
    result["avg_latency_ms"] = df["average_time"] * 1000.0
    result["p50_latency_ms"] = pd.NA
    result["p95_latency_ms"] = pd.NA
    result["p99_latency_ms"] = pd.NA
    result["build_time_s"] = df["build_time"]
    result["index_size_kb"] = pd.NA  # 项目未直接输出磁盘索引大小
    result["memory_kb"] = df["memory"]
    result["distcomps"] = pd.NA
    result["threads"] = df["threads"]
    result["raw_result_path"] = str(filepath.resolve())
    result["created_at"] = pd.Timestamp.now()

    return result


def parse_range_directory(input_dir: str) -> pd.DataFrame:
    """解析目录下所有 *_results.csv 文件。"""
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*_results.csv"))
    if not files:
        print(f"[parse_range] No *_results.csv found in {input_dir}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        print(f"[parse_range] Parsing: {f}")
        dfs.append(parse_range_csv(str(f)))

    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Parse RangeFilteredANN results to unified schema")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", required=True, help="Output normalized CSV path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        df = parse_range_directory(str(input_path))
    else:
        df = parse_range_csv(str(input_path))

    if df.empty:
        print("[parse_range] No data parsed.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[parse_range] Saved normalized CSV: {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
