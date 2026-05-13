#!/usr/bin/env python3
"""从已有 normalized 结果生成 experiment_runs.csv 模拟记录。"""

import os
import sys
import json
import uuid
import random
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_experiment_runs():
    all_results = PROJECT_ROOT / "results" / "normalized" / "all_results.csv"
    if not all_results.exists():
        print("[generate_experiment_runs] all_results.csv not found")
        return

    df = pd.read_csv(all_results)
    if df.empty:
        print("[generate_experiment_runs] all_results.csv is empty")
        return

    records = []
    base_time = datetime(2024, 5, 20, 9, 0, 0)

    # 为每个 project-algorithm-dataset 组合生成若干运行记录
    grouped = df.groupby(["project", "algorithm", "dataset"])
    run_idx = 1
    for (project, algo, dataset), sub in grouped:
        n_runs = min(len(sub), random.randint(1, 3))
        for i in range(n_runs):
            row = sub.iloc[i % len(sub)]
            status = random.choices(
                ["成功", "失败", "运行中"],
                weights=[0.75, 0.15, 0.10]
            )[0]
            duration_s = random.randint(60, 600)
            start = base_time + timedelta(hours=run_idx * 2)
            end = start + timedelta(seconds=duration_s) if status != "运行中" else None

            params = json.loads(row.get("params_json", "{}")) if pd.notna(row.get("params_json")) else {}
            records.append({
                "run_id": f"run_20240521_{run_idx:03d}",
                "project": project,
                "algorithm": algo,
                "dataset": dataset,
                "status": status,
                "start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end.strftime("%Y-%m-%d %H:%M:%S") if end else "",
                "duration": f"{duration_s // 60}m {duration_s % 60}s",
                "output_path": f"/results/run_20240521_{run_idx:03d}/",
                "recall": round(row.get("recall", 0), 4) if pd.notna(row.get("recall")) else "",
                "qps": round(row.get("qps", 0), 1) if pd.notna(row.get("qps")) else "",
                "params_json": json.dumps(params),
            })
            run_idx += 1

    # 补充一些额外记录使总数达到 ~128 条（模拟图8）
    extra_needed = 128 - len(records)
    if extra_needed > 0:
        projects = ["SIFT1M-Filter", "GloVe-Filter", "DEEP-Filter", "其他项目"]
        algos = ["HNSW", "Prefiltering", "Postfiltering", "β-WST", "SIEVE"]
        datasets_map = {
            "SIFT1M-Filter": "SIFT1M",
            "GloVe-Filter": "GloVe100K",
            "DEEP-Filter": "DEEP1M",
            "其他项目": "synthetic-small",
        }
        for i in range(extra_needed):
            proj = random.choice(projects)
            algo = random.choice(algos)
            dataset = datasets_map[proj]
            status = random.choices(["成功", "失败", "运行中"], weights=[0.75, 0.15, 0.10])[0]
            duration_s = random.randint(60, 600)
            start = base_time + timedelta(hours=run_idx * 1.5)
            end = start + timedelta(seconds=duration_s) if status != "运行中" else None
            records.append({
                "run_id": f"run_20240521_{run_idx:03d}",
                "project": proj,
                "algorithm": algo,
                "dataset": dataset,
                "status": status,
                "start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end.strftime("%Y-%m-%d %H:%M:%S") if end else "",
                "duration": f"{duration_s // 60}m {duration_s % 60}s",
                "output_path": f"/results/run_20240521_{run_idx:03d}/",
                "recall": round(random.uniform(0.6, 0.99), 3),
                "qps": round(random.uniform(500, 15000), 1),
                "params_json": json.dumps({"M": random.choice([16, 32]), "ef": random.choice([50, 100, 200])}),
            })
            run_idx += 1

    out_df = pd.DataFrame(records)
    out_path = PROJECT_ROOT / "results" / "experiment_runs.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[generate_experiment_runs] Saved {len(out_df)} records to {out_path}")


if __name__ == "__main__":
    generate_experiment_runs()
