#!/usr/bin/env python3
"""
召回率计算模块。

支持：
- 标准 recall@k：|ground_truth ∩ result[:k]| / k
- ANN-Benchmarks 风格阈值 recall（基于距离阈值）
"""

import numpy as np


def compute_recall_at_k(
    groundtruth: np.ndarray,
    results: np.ndarray,
    k: int = None,
) -> float:
    """
    计算标准 recall@k。

    Parameters
    ----------
    groundtruth : np.ndarray, shape=(n_queries, k_gt)
        每条查询的真实 top-k 邻居索引。
    results : np.ndarray, shape=(n_queries, k_res)
        算法返回的邻居索引。
    k : int, optional
        计算 recall@k 时的 k。默认取 min(groundtruth.shape[1], results.shape[1])。

    Returns
    -------
    float
        平均 recall@k，范围 [0, 1]。
    """
    if k is None:
        k = min(groundtruth.shape[1], results.shape[1])

    n_queries = groundtruth.shape[0]
    total_recall = 0.0

    for i in range(n_queries):
        gt_set = set(groundtruth[i, :k])
        res_set = set(results[i, :k])
        if len(gt_set) == 0:
            continue
        total_recall += len(gt_set & res_set) / len(gt_set)

    return total_recall / n_queries


def compute_recall_per_query(
    groundtruth: np.ndarray,
    results: np.ndarray,
    k: int = None,
) -> np.ndarray:
    """
    计算每条查询的 recall@k。

    Returns
    -------
    np.ndarray, shape=(n_queries,)
        每条查询的 recall 值。
    """
    if k is None:
        k = min(groundtruth.shape[1], results.shape[1])

    n_queries = groundtruth.shape[0]
    recalls = np.zeros(n_queries)

    for i in range(n_queries):
        gt_set = set(groundtruth[i, :k])
        res_set = set(results[i, :k])
        if len(gt_set) == 0:
            recalls[i] = 0.0
        else:
            recalls[i] = len(gt_set & res_set) / len(gt_set)

    return recalls


def compute_recall_with_threshold(
    groundtruth_distances: np.ndarray,
    result_distances: np.ndarray,
    k: int,
    epsilon: float = 1e-3,
) -> tuple:
    """
    ANN-Benchmarks 风格的阈值 recall。

    对每条查询，取 groundtruth 中第 k 个最近邻的距离作为阈值，
    统计算法返回的前 k 个结果中有多少个距离 <= 阈值。

    Parameters
    ----------
    groundtruth_distances : np.ndarray, shape=(n_queries, k_gt)
        每条查询的真实邻居距离（已排序）。
    result_distances : np.ndarray, shape=(n_queries, k_res)
        算法返回的邻居距离。
    k : int
        top-k 值。
    epsilon : float
        阈值容差。

    Returns
    -------
    mean_recall : float
    std_recall : float
    recalls_per_query : np.ndarray
    """
    n_queries = len(result_distances)
    recalls = np.zeros(n_queries)

    for i in range(n_queries):
        threshold = groundtruth_distances[i, k - 1] + epsilon
        actual = np.sum(result_distances[i, :k] <= threshold)
        recalls[i] = actual / k

    return np.mean(recalls), np.std(recalls), recalls
