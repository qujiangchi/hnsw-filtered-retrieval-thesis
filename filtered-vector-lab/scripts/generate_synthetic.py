#!/usr/bin/env python3
"""
生成最小合成数据集用于 MVP 验证。

输出文件（保存到 data/processed/synthetic-small/）：
- base.npy: shape=(10000, 128), float32
- query.npy: shape=(1000, 128), float32
- labels.npy: shape=(10000,), int64, 范围 0 到 9999
- window_queries.npy: shape=(1000, 2), int64, 每个 query 一个 [left, right]
- groundtruth.npy: shape=(1000, 10), int64, 暴力搜索计算每个过滤查询的 top-10 真值
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# 把项目根目录加入路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_synthetic_dataset(
    output_dir: str,
    base_count: int = 10000,
    query_count: int = 1000,
    dim: int = 128,
    k: int = 10,
    metric: str = "l2",
    seed: int = 42,
    filter_widths: list = None,
):
    """生成合成数据集并保存。"""
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    print(f"[generate] Output dir: {output_dir}")

    # 1. 生成 base vectors
    print(f"[generate] Generating base vectors: ({base_count}, {dim})")
    base = np.random.randn(base_count, dim).astype(np.float32)
    base_path = os.path.join(output_dir, "base.npy")
    np.save(base_path, base)
    print(f"[generate] Saved: {base_path}")

    # 2. 生成 query vectors
    print(f"[generate] Generating query vectors: ({query_count}, {dim})")
    query = np.random.randn(query_count, dim).astype(np.float32)
    query_path = os.path.join(output_dir, "query.npy")
    np.save(query_path, query)
    print(f"[generate] Saved: {query_path}")

    # 3. 生成 labels（用于 window filter）
    print(f"[generate] Generating labels: ({base_count},)")
    labels = np.arange(base_count, dtype=np.int64)
    np.random.shuffle(labels)  # 随机打乱，模拟非顺序标签
    labels_path = os.path.join(output_dir, "labels.npy")
    np.save(labels_path, labels)
    print(f"[generate] Saved: {labels_path}")

    # 4. 生成 window_queries（每个 query 一个 [left, right] 区间）
    print(f"[generate] Generating window queries: ({query_count}, 2)")
    # 随机生成 left，然后 right = left + width * base_count
    if filter_widths is None:
        filter_widths = [0.01, 0.10, 0.50]

    window_queries = []
    for _ in range(query_count):
        width_ratio = np.random.choice(filter_widths)
        width = max(1, int(width_ratio * base_count))
        left = np.random.randint(0, base_count - width + 1)
        right = left + width - 1
        window_queries.append([left, right])
    window_queries = np.array(window_queries, dtype=np.int64)
    window_queries_path = os.path.join(output_dir, "window_queries.npy")
    np.save(window_queries_path, window_queries)
    print(f"[generate] Saved: {window_queries_path}")

    # 5. 计算 ground truth（暴力搜索）
    print(f"[generate] Computing ground truth with brute-force (k={k}, metric={metric})")
    groundtruth = np.zeros((query_count, k), dtype=np.int64)

    if metric == "l2":
        # 用向量差计算 L2 距离（不平方根，不影响排序）
        for i in range(query_count):
            if i % 100 == 0:
                print(f"[generate] Ground truth progress: {i}/{query_count}")
            diffs = base - query[i]
            dists = np.sum(diffs * diffs, axis=1)
            # 应用 window filter
            left, right = window_queries[i]
            mask = (labels >= left) & (labels <= right)
            # 把不在窗口内的距离设为无穷大
            dists_filtered = np.where(mask, dists, np.inf)
            # 取 top-k（如果窗口内不足 k 个，会用 inf 填充，后续 adapter 需处理）
            topk_idx = np.argsort(dists_filtered)[:k]
            groundtruth[i] = topk_idx
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    gt_path = os.path.join(output_dir, "groundtruth.npy")
    np.save(gt_path, groundtruth)
    print(f"[generate] Saved: {gt_path}")

    # 6. 保存元信息
    meta = {
        "base_count": base_count,
        "query_count": query_count,
        "dim": dim,
        "k": k,
        "metric": metric,
        "seed": seed,
        "filter_widths": filter_widths,
    }
    meta_path = os.path.join(output_dir, "meta.npy")
    np.save(meta_path, meta)
    print(f"[generate] Saved meta: {meta_path}")

    print("[generate] Done.")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for MVP")
    parser.add_argument("--output-dir", default="data/processed/synthetic-small", help="Output directory")
    parser.add_argument("--base-count", type=int, default=10000, help="Number of base vectors")
    parser.add_argument("--query-count", type=int, default=1000, help="Number of query vectors")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension")
    parser.add_argument("--k", type=int, default=10, help="Top-k")
    parser.add_argument("--metric", default="l2", choices=["l2"], help="Distance metric")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_synthetic_dataset(
        output_dir=args.output_dir,
        base_count=args.base_count,
        query_count=args.query_count,
        dim=args.dim,
        k=args.k,
        metric=args.metric,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
