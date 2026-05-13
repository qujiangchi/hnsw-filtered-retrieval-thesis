#!/usr/bin/env python3
"""
生成毕业论文实验所需的合成数据集。
模拟 MS MARCO、Natural Questions 和 Enron Email 的规模和维度。
"""

import os
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_dataset(name, num_base, num_query, dim, seed=42):
    """生成标准合成数据集并保存为npy。"""
    np.random.seed(seed)
    data_dir = PROJECT_ROOT / "data" / "processed" / name
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{name}] Generating {num_base} base vectors (dim={dim})...")
    base = np.random.randn(num_base, dim).astype(np.float32)
    # L2 normalize for cosine-like behavior
    base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-10

    print(f"[{name}] Generating {num_query} query vectors...")
    query = np.random.randn(num_query, dim).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True) + 1e-10

    # Ground truth: brute force exact search
    print(f"[{name}] Computing ground truth (brute force, k=100)...")
    gt = compute_groundtruth(base, query, k=100)

    # Labels for filtering experiments (SIEVE-style)
    # Simulate structured attributes
    print(f"[{name}] Generating filter labels...")
    labels = generate_filter_labels(num_base, name)

    # Meta
    meta = {
        "name": name,
        "num_base": int(num_base),
        "num_query": int(num_query),
        "dim": int(dim),
        "seed": int(seed),
    }

    np.save(data_dir / "base.npy", base)
    np.save(data_dir / "query.npy", query)
    np.save(data_dir / "groundtruth.npy", gt)
    np.save(data_dir / "labels.npy", labels)
    np.save(data_dir / "meta.npy", meta)

    print(f"[{name}] Saved to {data_dir}")
    print(f"[{name}] base: {base.nbytes/1024/1024:.1f} MB, query: {query.nbytes/1024/1024:.1f} MB")
    return data_dir


def compute_groundtruth(base, query, k=100):
    """使用Faiss暴力计算精确最近邻。"""
    import faiss
    num_base = base.shape[0]
    dim = base.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(base)
    _, gt = index.search(query, k)
    return gt.astype(np.int32)


def generate_filter_labels(num_base, dataset_name):
    """
    生成模拟的过滤标签属性。
    返回 shape=(num_base,) 的结构化数组。
    """
    if "enron" in dataset_name.lower():
        # Enron-like: year (1999-2002), domain (0/1), category (0/1/2)
        year = np.random.randint(1999, 2003, size=num_base).astype(np.int16)
        domain = np.random.randint(0, 2, size=num_base).astype(np.int8)
        category = np.random.randint(0, 3, size=num_base).astype(np.int8)
    else:
        # Generic: 3 discrete attributes with varying cardinalities
        year = np.random.randint(2018, 2024, size=num_base).astype(np.int16)
        domain = np.random.randint(0, 5, size=num_base).astype(np.int8)
        category = np.random.randint(0, 10, size=num_base).astype(np.int8)

    dtype = np.dtype([
        ("year", np.int16),
        ("domain", np.int8),
        ("category", np.int8),
    ])
    labels = np.empty(num_base, dtype=dtype)
    labels["year"] = year
    labels["domain"] = domain
    labels["category"] = category
    return labels


def main():
    datasets = [
        ("synthetic-msmarco", 50_000, 5_000, 768),     # 模拟 MS MARCO (缩小~20x)
        ("synthetic-nq", 50_000, 5_000, 768),          # 模拟 Natural Questions (缩小~40x)
        ("synthetic-enron", 30_000, 3_000, 768),       # 模拟 Enron Email (缩小~15x)
    ]
    for name, n_base, n_query, dim in datasets:
        generate_dataset(name, n_base, n_query, dim, seed=42)
        print()


if __name__ == "__main__":
    main()
