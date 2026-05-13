#!/usr/bin/env python3
"""
hnswlib 自定义结果解析器。

hnswlib 的 adapter 已经直接输出统一 Schema 的 summary.csv，
所以此 parser 主要是读取并验证格式。

输入：
    results/raw/hnswlib/<dataset>/summary.csv

输出：
    results/normalized/hnswlib.csv

用法：
    python src/parsers/parse_hnswlib_results.py --input results/raw/hnswlib --output results/normalized/hnswlib.csv
"""

import os
import argparse
import pandas as pd
from pathlib import Path


def parse_hnswlib_summary(filepath: str) -> pd.DataFrame:
    """读取 hnswlib summary.csv 并验证 Schema。"""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, dtype={"run_id": "string"})
    print(f"[parse_hnswlib] Loaded {len(df)} rows from {filepath}")
    return df


def parse_hnswlib_directory(input_dir: str) -> pd.DataFrame:
    """解析目录下所有 summary.csv。"""
    input_dir = Path(input_dir)
    files = list(input_dir.rglob("summary.csv"))
    if not files:
        print(f"[parse_hnswlib] No summary.csv found under {input_dir}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        print(f"[parse_hnswlib] Parsing: {f}")
        dfs.append(parse_hnswlib_summary(str(f)))

    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Parse hnswlib results to unified schema")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", required=True, help="Output normalized CSV path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        df = parse_hnswlib_directory(str(input_path))
    else:
        df = parse_hnswlib_summary(str(input_path))

    if df.empty:
        print("[parse_hnswlib] No data parsed.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[parse_hnswlib] Saved normalized CSV: {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
