#!/usr/bin/env python3
"""
在真实数据集上批量运行完整实验，并生成图表。

用法:
  python3 scripts/run_real_experiments.py [--datasets msmarco nq enron]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASETS = ["msmarco", "nq", "enron"]


def run_cmd(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f" {desc or cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS,
                        help="要运行的数据集")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="跳过数据准备（假设已编码好）")
    parser.add_argument("--skip-figures", action="store_true",
                        help="跳过图表生成")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    # 1. 准备数据（如需要）
    if not args.skip_prepare:
        # 检查数据是否存在
        missing = []
        for ds in args.datasets:
            if not (PROJECT_ROOT / "data" / "real" / ds / "base.npy").exists():
                missing.append(ds)
        if missing:
            print(f"[Prepare] Missing datasets: {missing}. Running download_real_datasets.py...")
            # 根据内存自动调整采样比例
            import psutil
            mem_gb = psutil.virtual_memory().total / (1024**3)
            if mem_gb < 80:
                msmarco_r, nq_r = 0.2, 0.1
            elif mem_gb < 150:
                msmarco_r, nq_r = 0.5, 0.2
            else:
                msmarco_r, nq_r = 1.0, 0.5
            print(f"  Detected {mem_gb:.0f}GB RAM, using ratios: msmarco={msmarco_r}, nq={nq_r}")
            run_cmd(
                f"python3 scripts/download_real_datasets.py "
                f"--msmarco-ratio {msmarco_r} --nq-ratio {nq_r} --enron-ratio 1.0",
                "Downloading & encoding real datasets"
            )

    # 2. 逐个运行实验
    for ds in args.datasets:
        ds_path = PROJECT_ROOT / "data" / "real" / ds
        if not (ds_path / "base.npy").exists():
            print(f"[Skip] {ds}: data not found at {ds_path}")
            continue
        run_cmd(f"python3 scripts/run_thesis_experiments_single_dataset.py {ds}",
                f"Running experiments on {ds}")

    # 3. 生成图表
    if not args.skip_figures:
        run_cmd("python3 scripts/generate_paper_results_and_figures.py",
                "Generating paper figures")

    print("\n" + "=" * 60)
    print(" All done! Check results/ and docs/ for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
