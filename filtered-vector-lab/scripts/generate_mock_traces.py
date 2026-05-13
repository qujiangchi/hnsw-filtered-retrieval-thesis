#!/usr/bin/env python3
"""生成 β-WST 和 SIEVE 的 mock trace 数据，用于 Dashboard 演示。"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_bwst_trace():
    """生成 β-WST 查询路径 trace。"""
    random.seed(42)
    np.random.seed(42)

    # 模拟一棵 β=4 的树结构，查询访问了 6 个节点
    nodes = [
        {"visited_order": 1, "node_id": 1, "level": 0, "label_range": "[0, 100]", "search_type": "range_intersect", "point_count": 100000, "duration_ms": 0.162},
        {"visited_order": 2, "node_id": 3, "level": 1, "label_range": "[25, 50]", "search_type": "range_intersect", "point_count": 25112, "duration_ms": 0.101},
        {"visited_order": 3, "node_id": 11, "level": 2, "label_range": "[31.25, 37.5]", "search_type": "range_intersect", "point_count": 6301, "duration_ms": 0.068},
        {"visited_order": 4, "node_id": 44, "level": 3, "label_range": "[31.25, 33.33]", "search_type": "range_intersect", "point_count": 1575, "duration_ms": 0.042},
        {"visited_order": 5, "node_id": 45, "level": 3, "label_range": "[33.33, 35.41]", "search_type": "range_intersect", "point_count": 1542, "duration_ms": 0.029},
        {"visited_order": 6, "node_id": 46, "level": 3, "label_range": "[35.41, 37.5]", "search_type": "range_intersect", "point_count": 1634, "duration_ms": 0.018},
    ]

    df = pd.DataFrame(nodes)
    out_path = PROJECT_ROOT / "results" / "raw" / "bwst_trace.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[generate_mock_traces] Saved β-WST trace: {out_path}")

    # 同时生成树结构节点定义（用于画图）
    tree_nodes = [
        {"node_id": 1, "level": 0, "label_min": 0, "label_max": 100, "point_count": 100000, "parent_id": -1},
        {"node_id": 2, "level": 1, "label_min": 0, "label_max": 25, "point_count": 24823, "parent_id": 1},
        {"node_id": 3, "level": 1, "label_min": 25, "label_max": 50, "point_count": 25112, "parent_id": 1},
        {"node_id": 4, "level": 1, "label_min": 50, "label_max": 75, "point_count": 24995, "parent_id": 1},
        {"node_id": 5, "level": 1, "label_min": 75, "label_max": 100, "point_count": 25070, "parent_id": 1},
        {"node_id": 6, "level": 2, "label_min": 0, "label_max": 6.25, "point_count": 6378, "parent_id": 2},
        {"node_id": 7, "level": 2, "label_min": 6.25, "label_max": 12.5, "point_count": 6194, "parent_id": 2},
        {"node_id": 8, "level": 2, "label_min": 12.5, "label_max": 18.75, "point_count": 6188, "parent_id": 2},
        {"node_id": 9, "level": 2, "label_min": 18.75, "label_max": 25, "point_count": 6063, "parent_id": 2},
        {"node_id": 10, "level": 2, "label_min": 25, "label_max": 31.25, "point_count": 6227, "parent_id": 3},
        {"node_id": 11, "level": 2, "label_min": 31.25, "label_max": 37.5, "point_count": 6301, "parent_id": 3},
        {"node_id": 12, "level": 2, "label_min": 37.5, "label_max": 43.75, "point_count": 6329, "parent_id": 3},
        {"node_id": 13, "level": 2, "label_min": 43.75, "label_max": 50, "point_count": 6255, "parent_id": 3},
        {"node_id": 44, "level": 3, "label_min": 31.25, "label_max": 33.33, "point_count": 1575, "parent_id": 11},
        {"node_id": 45, "level": 3, "label_min": 33.33, "label_max": 35.41, "point_count": 1542, "parent_id": 11},
        {"node_id": 46, "level": 3, "label_min": 35.41, "label_max": 37.5, "point_count": 1634, "parent_id": 11},
    ]
    tree_df = pd.DataFrame(tree_nodes)
    tree_path = PROJECT_ROOT / "results" / "raw" / "bwst_tree_structure.csv"
    tree_df.to_csv(tree_path, index=False)
    print(f"[generate_mock_traces] Saved β-WST tree structure: {tree_path}")


def generate_sieve_trace():
    """生成 SIEVE 查询策略 trace。"""
    random.seed(42)
    np.random.seed(42)

    strategies = ["root search", "covering search", "upward search", "brute-force fallback"]
    weights = [0.412, 0.316, 0.187, 0.085]

    filter_exprs = [
        "age ∈ [20,60] AND gender = 'M'",
        "city = 'Beijing'",
        "age ∈ [20,60] AND attr = 'D'",
        "income ∈ [50k, 100k] AND hobby = 'Music'",
        "rare_tag = 'X' AND attr = 'Z'",
        "age ∈ [20,60] AND country = 'US'",
        "department = 'R&D'",
    ]

    records = []
    n_queries = 25000
    for i in range(1, n_queries + 1):
        strategy = random.choices(strategies, weights=weights)[0]
        expr = random.choice(filter_exprs)
        if strategy == "root search":
            chosen_index = "root"
        elif strategy == "covering search":
            chosen_index = random.choice(["attr = age", "attr = income", "attr = country"])
        elif strategy == "upward search":
            chosen_index = random.choice(["attr = D → root", "attr = B → root"])
        else:
            chosen_index = "N/A"

        actual_recall = round(random.uniform(0.92, 0.98) if strategy != "brute-force fallback" else random.uniform(0.93, 0.999), 3)
        latency_ms = round(random.uniform(0.2, 1.5) if strategy != "brute-force fallback" else random.uniform(2.0, 6.0), 2)

        records.append({
            "query_id": f"q_{i:05d}",
            "filter_expr": expr,
            "chosen_strategy": strategy,
            "chosen_index": chosen_index,
            "actual_recall": actual_recall,
            "latency_ms": latency_ms,
            "fallback": "是" if strategy == "brute-force fallback" else "否",
        })

    df = pd.DataFrame(records)
    out_path = PROJECT_ROOT / "results" / "raw" / "sieve_trace.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[generate_mock_traces] Saved SIEVE trace: {out_path} ({len(df)} rows)")

    # 生成策略汇总
    summary = df.groupby("chosen_strategy").agg(
        count=("query_id", "count"),
        avg_recall=("actual_recall", "mean"),
        avg_latency=("latency_ms", "mean"),
    ).reset_index()
    summary["percentage"] = (summary["count"] / len(df) * 100).round(1)
    summary_path = PROJECT_ROOT / "results" / "raw" / "sieve_strategy_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[generate_mock_traces] Saved SIEVE strategy summary: {summary_path}")


if __name__ == "__main__":
    generate_bwst_trace()
    generate_sieve_trace()
