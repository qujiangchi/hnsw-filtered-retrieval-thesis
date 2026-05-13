#!/usr/bin/env python3
"""
统一结果存储模块。

支持：
- 写入/读取 normalized CSV
- 合并多个项目的 normalized CSV 为 all_results.csv
- 可选 SQLite/DuckDB 持久化
"""

import os
import json
import uuid
import pandas as pd
from datetime import datetime
from pathlib import Path


# 统一 Schema 的 dtype
SCHEMA_DTYPE = {
    "run_id": "string",
    "project": "string",
    "algorithm": "string",
    "dataset": "string",
    "track": "string",
    "filter_type": "string",
    "filter_width": "string",
    "selectivity": "Float64",
    "k": "Int64",
    "params_json": "string",
    "recall": "Float64",
    "qps": "Float64",
    "avg_latency_ms": "Float64",
    "p50_latency_ms": "Float64",
    "p95_latency_ms": "Float64",
    "p99_latency_ms": "Float64",
    "build_time_s": "Float64",
    "index_size_kb": "Float64",
    "memory_kb": "Float64",
    "distcomps": "Float64",
    "threads": "Int64",
    "raw_result_path": "string",
    "created_at": "datetime64[ns]",
}

SCHEMA_COLUMNS = list(SCHEMA_DTYPE.keys())


def generate_run_id() -> str:
    """生成唯一 run_id。"""
    return str(uuid.uuid4())[:8]


def create_empty_results_df() -> pd.DataFrame:
    """创建空的统一 Schema DataFrame。"""
    df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in SCHEMA_DTYPE.items()})
    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    将任意 DataFrame 对齐到统一 Schema。
    - 缺少的列补 NULL
    - 多余的列保留（不删除，但统一列在前）
    - 类型转换
    """
    # 确保所有 Schema 列存在
    for col, dt in SCHEMA_DTYPE.items():
        if col not in df.columns:
            df[col] = pd.NA
        else:
            try:
                if dt == "datetime64[ns]":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                else:
                    df[col] = df[col].astype(dt)
            except (ValueError, TypeError):
                df[col] = pd.NA

    # 重新排列列顺序
    ordered_cols = [c for c in SCHEMA_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in SCHEMA_COLUMNS]
    df = df[ordered_cols + extra_cols]
    return df


def save_normalized_csv(df: pd.DataFrame, output_path: str):
    """保存 normalized DataFrame 为 CSV。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[result_store] Saved normalized CSV: {output_path}")


def load_normalized_csv(path: str) -> pd.DataFrame:
    """加载 normalized CSV 并对齐 Schema。"""
    df = pd.read_csv(path, dtype={"run_id": "string", "project": "string"})
    df = normalize_dataframe(df)
    return df


def merge_all_normalized(
    normalized_dir: str = "results/normalized",
    output_path: str = "results/normalized/all_results.csv",
) -> pd.DataFrame:
    """
    合并 normalized 目录下所有 CSV 为 all_results.csv。
    """
    normalized_dir = Path(normalized_dir)
    if not normalized_dir.exists():
        print(f"[result_store] Normalized dir not found: {normalized_dir}")
        return create_empty_results_df()

    csv_files = sorted(normalized_dir.glob("*.csv"))
    if not csv_files:
        print(f"[result_store] No CSV files found in {normalized_dir}")
        return create_empty_results_df()

    dfs = []
    for f in csv_files:
        if f.name == "all_results.csv":
            continue
        if f.stat().st_size == 0:
            print(f"[result_store] Skipping empty file: {f}")
            continue
        print(f"[result_store] Loading: {f}")
        try:
            df = load_normalized_csv(str(f))
            if df.empty:
                print(f"[result_store] Skipping empty dataframe: {f}")
                continue
            dfs.append(df)
        except Exception as e:
            print(f"[result_store] Error loading {f}: {e}")
            continue

    if not dfs:
        return create_empty_results_df()

    merged = pd.concat(dfs, ignore_index=True)
    merged = normalize_dataframe(merged)
    save_normalized_csv(merged, output_path)
    return merged


def add_result_row(
    df: pd.DataFrame,
    project: str,
    algorithm: str,
    dataset: str,
    recall: float,
    qps: float,
    k: int,
    params_json: dict = None,
    **kwargs,
) -> pd.DataFrame:
    """
    向 DataFrame 添加一条实验结果记录。

    Parameters
    ----------
    df : pd.DataFrame
        现有结果表。
    project, algorithm, dataset, recall, qps, k : 核心字段
    params_json : dict
        算法参数字典，会自动转为 JSON 字符串。
    **kwargs : 其他 Schema 字段。

    Returns
    -------
    pd.DataFrame
        添加记录后的新 DataFrame。
    """
    row = {"run_id": generate_run_id(), "project": project, "algorithm": algorithm,
           "dataset": dataset, "recall": recall, "qps": qps, "k": k,
           "created_at": datetime.now(), "raw_result_path": kwargs.get("raw_result_path", "")}

    if params_json is not None:
        row["params_json"] = json.dumps(params_json, sort_keys=True)
    else:
        row["params_json"] = "{}"

    # 合并其他字段
    for key, val in kwargs.items():
        if key in SCHEMA_DTYPE:
            row[key] = val

    # 确保所有列存在
    for col in SCHEMA_COLUMNS:
        if col not in row:
            row[col] = pd.NA

    new_df = pd.DataFrame([row])
    new_df = normalize_dataframe(new_df)
    df = normalize_dataframe(df)
    return pd.concat([df, new_df], ignore_index=True)
