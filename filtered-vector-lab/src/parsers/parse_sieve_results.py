#!/usr/bin/env python3
"""
SIEVE 结果解析器。

输入：
    SIEVE data_export.py 生成的 CSV

原始字段：
    algorithm, parameters, dataset, count, qps, distcomps, build, indexsize,
    mean_ssd_ios, mean_latency, track, recall/ap

输出：
    results/normalized/sieve.csv（统一 Schema）

用法：
    python src/parsers/parse_sieve_results.py --input results/raw/sieve/res.csv --output results/normalized/sieve.csv
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path


def parse_sieve_csv(filepath: str) -> pd.DataFrame:
    """解析单个 SIEVE 结果 CSV 文件。"""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"[parse_sieve] Loaded {len(df)} rows from {filepath}")
    print(f"[parse_sieve] Columns: {list(df.columns)}")

    # 重命名已知列（处理可能的空格或大小写问题）
    col_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ["algorithm", "parameters", "dataset", "count", "qps",
                  "distcomps", "build", "indexsize", "mean_ssd_ios",
                  "mean_latency", "track", "recall/ap", "recall/ap"]:
            col_map[c] = lc.replace("/", "_")
    df = df.rename(columns=col_map)

    # 确保关键列存在
    for col in ["algorithm", "dataset", "count", "qps", "recall_ap"]:
        if col not in df.columns:
            print(f"[parse_sieve] Warning: expected column '{col}' not found")

    # 类型转换
    df["qps"] = pd.to_numeric(df.get("qps"), errors="coerce")
    df["count"] = pd.to_numeric(df.get("count"), errors="coerce").astype("Int64")
    df["build"] = pd.to_numeric(df.get("build"), errors="coerce")
    df["indexsize"] = pd.to_numeric(df.get("indexsize"), errors="coerce")
    df["distcomps"] = pd.to_numeric(df.get("distcomps"), errors="coerce")
    df["mean_latency"] = pd.to_numeric(df.get("mean_latency"), errors="coerce")

    # 构建统一 Schema
    result = pd.DataFrame()
    result["run_id"] = [pd.NA] * len(df)
    result["project"] = "SIEVE"
    result["algorithm"] = df.get("algorithm", pd.NA)
    result["dataset"] = df.get("dataset", pd.NA)
    result["track"] = df.get("track", pd.NA)
    result["filter_type"] = df.get("track", pd.NA).apply(
        lambda x: "predicate" if str(x).lower() == "filter" else pd.NA
    )
    result["filter_width"] = pd.NA
    result["selectivity"] = pd.NA
    result["k"] = df.get("count", pd.NA)
    result["params_json"] = df.get("parameters", pd.NA).apply(
        lambda x: json.dumps({"raw_parameters": str(x)}, sort_keys=True) if pd.notna(x) else "{}"
    )
    result["recall"] = pd.to_numeric(df.get("recall_ap"), errors="coerce")
    result["qps"] = df["qps"]
    result["avg_latency_ms"] = df.get("mean_latency")
    result["p50_latency_ms"] = pd.NA
    result["p95_latency_ms"] = pd.NA
    result["p99_latency_ms"] = pd.NA
    result["build_time_s"] = df.get("build")
    result["index_size_kb"] = df.get("indexsize")
    result["memory_kb"] = pd.NA
    result["distcomps"] = df.get("distcomps")
    result["threads"] = pd.NA
    result["raw_result_path"] = str(filepath.resolve())
    result["created_at"] = pd.Timestamp.now()

    # 对 SIEVE 特殊说明：mean_latency 和 distcomps 默认可能为 0（未实现 get_additional）
    # 将这些无意义的 0 转为 NULL
    result["avg_latency_ms"] = result["avg_latency_ms"].replace(0, pd.NA)
    result["distcomps"] = result["distcomps"].replace(0, pd.NA)

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse SIEVE results to unified schema")
    parser.add_argument("--input", required=True, help="Input SIEVE CSV file")
    parser.add_argument("--output", required=True, help="Output normalized CSV path")
    args = parser.parse_args()

    df = parse_sieve_csv(args.input)
    if df.empty:
        print("[parse_sieve] No data parsed.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[parse_sieve] Saved normalized CSV: {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
