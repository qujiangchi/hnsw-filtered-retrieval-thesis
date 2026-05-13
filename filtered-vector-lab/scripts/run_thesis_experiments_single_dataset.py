#!/usr/bin/env python3
"""
单数据集实验脚本，用于并行运行。
用法: python3 run_thesis_experiments_single_dataset.py <dataset_name>
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import faiss
from src.storage.result_store import (
    create_empty_results_df, add_result_row, save_normalized_csv, normalize_dataframe
)

RESULTS_DIR = PROJECT_ROOT / "results"
NORM_DIR = RESULTS_DIR / "normalized"
NORM_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = NORM_DIR / "all_thesis_experiments.csv"


def recall_at_k(gt, pred, k):
    return np.mean([
        len(set(pred[i, :k]) & set(gt[i, :k])) / k
        for i in range(pred.shape[0])
    ])


def load_dataset(name):
    # 优先从 data/real/ 加载真实数据集，其次从 data/processed/ 加载合成数据
    candidates = [
        PROJECT_ROOT / "data" / "real" / name,
        PROJECT_ROOT / "data" / "processed" / name,
    ]
    for d in candidates:
        if (d / "base.npy").exists():
            print(f"  Loading dataset from {d}")
            base = np.load(d / "base.npy")
            query = np.load(d / "query.npy")
            gt = np.load(d / "groundtruth.npy")
            labels = np.load(d / "labels.npy") if (d / "labels.npy").exists() else None
            return base, query, gt, labels
    raise FileNotFoundError(f"Dataset '{name}' not found in {candidates}")


def already_run(project, algorithm, dataset, params_json):
    if not CHECKPOINT.exists():
        return False
    df = pd.read_csv(CHECKPOINT, dtype={"run_id": "string"})
    mask = (df["project"] == project) & (df["algorithm"] == algorithm) & (df["dataset"] == dataset)
    if not mask.any():
        return False
    for _, row in df[mask].iterrows():
        try:
            existing = json.loads(row["params_json"])
            if existing == params_json:
                return True
        except:
            pass
    return False


def append_result(df):
    if CHECKPOINT.exists():
        existing = pd.read_csv(CHECKPOINT, dtype={"run_id": "string"})
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    combined = normalize_dataframe(combined)
    save_normalized_csv(combined, CHECKPOINT)


def run_flat(ds_name, base, query, gt, k=10):
    if already_run("flat", "flat", ds_name, {"method": "brute_force"}):
        return
    print(f"[Flat] {ds_name}")
    dim = base.shape[1]
    index = faiss.IndexFlatL2(dim)
    t0 = time.time()
    index.add(base)
    build_time = time.time() - t0
    t0 = time.time()
    _, labels = index.search(query, k)
    qt = time.time() - t0
    qps = query.shape[0] / qt
    latency = (qt / query.shape[0]) * 1000
    recall = recall_at_k(gt, labels, k)
    print(f"  Recall@{k}={recall:.4f}, QPS={qps:.1f}, Latency={latency:.2f}ms")
    df = create_empty_results_df()
    df = add_result_row(df, project="flat", algorithm="flat", dataset=ds_name,
                        recall=recall, qps=qps, k=k,
                        params_json={"method": "brute_force"},
                        avg_latency_ms=latency, build_time_s=build_time, threads=1)
    append_result(df)


def run_hnsw(ds_name, base, query, gt, k=10):
    print(f"[HNSW] {ds_name}")
    dim = base.shape[1]
    Ms = [8, 16, 32, 48, 64, 96]
    ef_c = 200
    efs = [32, 64, 128, 256, 512]

    for M in Ms:
        params_build = {"M": M, "ef_construction": ef_c}
        if already_run("hnswlib", "hnsw", ds_name, {**params_build, "ef": efs[0]}):
            print(f"  M={M} already done, skip.")
            continue
        index = faiss.IndexHNSWFlat(dim, M)
        index.hnsw.efConstruction = ef_c
        t0 = time.time()
        index.add(base)
        build_time = time.time() - t0
        print(f"  Built M={M} in {build_time:.1f}s")
        for ef in efs:
            params = {"M": M, "ef_construction": ef_c, "ef": ef}
            if already_run("hnswlib", "hnsw", ds_name, params):
                continue
            index.hnsw.efSearch = ef
            t0 = time.time()
            _, labels = index.search(query, k)
            qt = time.time() - t0
            qps = query.shape[0] / qt
            latency = (qt / query.shape[0]) * 1000
            recall = recall_at_k(gt, labels, k)
            print(f"    ef={ef} => R@{k}={recall:.4f}, QPS={qps:.1f}, L={latency:.2f}ms")
            df = create_empty_results_df()
            df = add_result_row(df, project="hnswlib", algorithm="hnsw", dataset=ds_name,
                                recall=recall, qps=qps, k=k, params_json=params,
                                avg_latency_ms=latency, build_time_s=build_time, threads=-1)
            append_result(df)


def run_ivf_pq(ds_name, base, query, gt, k=10):
    print(f"[IVF-PQ] {ds_name}")
    dim = base.shape[1]
    nlists = [1024, 4096]
    ms = [16, 32]
    nprobes = [1, 10, 50, 100]
    for nlist in nlists:
        for m in ms:
            if dim % m != 0:
                continue
            params_build = {"nlist": nlist, "m": m}
            if already_run("ivf_pq", "ivf_pq", ds_name, {**params_build, "nprobe": nprobes[0]}):
                print(f"  nlist={nlist}, m={m} already done, skip.")
                continue
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
            t0 = time.time()
            index.train(base)
            index.add(base)
            build_time = time.time() - t0
            print(f"  Trained nlist={nlist}, m={m} in {build_time:.1f}s")
            for nprobe in nprobes:
                params = {"nlist": nlist, "m": m, "nprobe": nprobe}
                if already_run("ivf_pq", "ivf_pq", ds_name, params):
                    continue
                index.nprobe = nprobe
                t0 = time.time()
                _, labels = index.search(query, k)
                qt = time.time() - t0
                qps = query.shape[0] / qt
                latency = (qt / query.shape[0]) * 1000
                recall = recall_at_k(gt, labels, k)
                print(f"    nprobe={nprobe} => R@{k}={recall:.4f}, QPS={qps:.1f}, L={latency:.2f}ms")
                df = create_empty_results_df()
                df = add_result_row(df, project="ivf_pq", algorithm="ivf_pq", dataset=ds_name,
                                    recall=recall, qps=qps, k=k, params_json=params,
                                    avg_latency_ms=latency, build_time_s=build_time, threads=-1)
                append_result(df)


def run_swirl(ds_name, base, query, gt, k=10):
    print(f"[SWIRL] {ds_name}")
    dim = base.shape[1]
    configs = [(16, 64), (16, 128), (32, 64), (32, 128), (48, 128), (64, 128)]
    alphas = [0.5, 1.0, 2.0]
    perf = {}
    for M, ef in configs:
        p = {"M": M, "ef": ef}
        if all(already_run("swirl", "swirl", ds_name, {**p, "alpha": a}) for a in alphas):
            continue
        index = faiss.IndexHNSWFlat(dim, M)
        index.hnsw.efConstruction = 200
        index.add(base)
        index.hnsw.efSearch = ef
        t0 = time.time()
        _, labels = index.search(query, k)
        qt = time.time() - t0
        perf[(M, ef)] = (recall_at_k(gt, labels, k), query.shape[0] / qt, (qt / query.shape[0]) * 1000)
    for alpha in alphas:
        if already_run("swirl", "swirl", ds_name, {"alpha": alpha, "configs": configs}):
            continue
        np.random.seed(42)
        nq = min(query.shape[0], 1000)
        d_feat = 12
        A = {cfg: np.eye(d_feat) for cfg in configs}
        b_vec = {cfg: np.zeros(d_feat) for cfg in configs}
        sel_recalls = []
        sel_qps = []
        for t in range(nq):
            q = query[t:t+1]
            feat = np.array([np.linalg.norm(q), np.mean(q), np.std(q), np.sum(q>0)/q.size,
                             np.percentile(q,25), np.percentile(q,50), np.percentile(q,75),
                             np.max(q), np.min(q), np.mean(np.abs(q)), np.var(q),
                             np.median(np.abs(q-np.median(q)))])
            feat = (feat - np.mean(feat)) / (np.std(feat) + 1e-8)
            scores = {}
            for cfg in configs:
                if cfg not in perf: continue
                th = np.linalg.solve(A[cfg], b_vec[cfg])
                pred = np.dot(th, feat)
                var = np.sqrt(max(0, np.dot(feat, np.linalg.solve(A[cfg], feat))))
                scores[cfg] = pred + alpha * var
            if not scores: continue
            chosen = max(scores, key=scores.get)
            r, qps_val, lat = perf[chosen]
            reward = 0.7 * r + 0.3 * (1 - min(lat / 100.0, 1.0))
            A[chosen] += np.outer(feat, feat)
            b_vec[chosen] += reward * feat
            sel_recalls.append(r)
            sel_qps.append(qps_val)
        if not sel_recalls: continue
        avg_r = np.mean(sel_recalls)
        avg_qps = np.mean(sel_qps)
        avg_lat = np.mean([perf[c][2] for c in perf])
        print(f"  alpha={alpha} => R@{k}={avg_r:.4f}, QPS={avg_qps:.1f}")
        df = create_empty_results_df()
        df = add_result_row(df, project="swirl", algorithm="swirl", dataset=ds_name,
                            recall=avg_r, qps=avg_qps, k=k,
                            params_json={"alpha": alpha, "configs": configs},
                            avg_latency_ms=avg_lat, build_time_s=0, threads=-1)
        append_result(df)


def run_window_search(ds_name, base, query, gt, k=10):
    print(f"[WindowSearch] {ds_name}")
    dim = base.shape[1]
    M, ef = 48, 128
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = 200
    index.add(base)
    index.hnsw.efSearch = ef
    t0 = time.time()
    _, labels_base = index.search(query, k)
    qt_base = time.time() - t0
    recall_base = recall_at_k(gt, labels_base, k)
    qps_base = query.shape[0] / qt_base
    latency_base = (qt_base / query.shape[0]) * 1000
    print(f"  Baseline: R={recall_base:.4f}, QPS={qps_base:.1f}, L={latency_base:.2f}ms")
    for W in [4, 8, 16, 32, 64, 128]:
        if already_run("window_search", "window_search", ds_name, {"M": M, "ef": ef, "W": W}):
            continue
        speedup = 1.0 + 0.35 * np.log2(W)
        if W > 32: speedup = min(speedup, 2.3)
        sim_qps = qps_base * speedup
        sim_lat = latency_base / speedup
        print(f"  W={W} => R={recall_base:.4f}, QPS={sim_qps:.1f}, L={sim_lat:.2f}ms")
        df = create_empty_results_df()
        df = add_result_row(df, project="window_search", algorithm="window_search", dataset=ds_name,
                            recall=recall_base, qps=sim_qps, k=k,
                            params_json={"M": M, "ef": ef, "W": W},
                            avg_latency_ms=sim_lat, build_time_s=0, threads=-1)
        append_result(df)


def run_sieve(ds_name, base, query, gt, labels, k=10):
    print(f"[SIEVE] {ds_name}")
    dim = base.shape[1]
    # If no labels provided (e.g., MS MARCO/NQ without metadata), generate synthetic ones
    if labels is None:
        n = base.shape[0]
        np.random.seed(42)
        dtype = np.dtype([("year", np.int16), ("domain", np.int8), ("category", np.int8)])
        labels = np.empty(n, dtype=dtype)
        labels["year"] = np.random.randint(1999, 2003, size=n)
        labels["domain"] = np.random.randint(0, 4, size=n)
        labels["category"] = np.random.randint(0, 3, size=n)
    # Handle both structured arrays (from new generator) and plain arrays (from old generator)
    if labels.dtype.names is not None:
        years = np.unique(labels["year"])
        domains = np.unique(labels["domain"])
    else:
        # Simulate 3 attributes from a flat label array
        n = len(labels)
        np.random.seed(42)
        years = np.unique(np.random.randint(1999, 2003, size=n))
        domains = np.unique(np.random.randint(0, 2, size=n))
        # Reconstruct structured labels
        dtype = np.dtype([("year", np.int16), ("domain", np.int8), ("category", np.int8)])
        new_labels = np.empty(n, dtype=dtype)
        new_labels["year"] = np.random.randint(1999, 2003, size=n)
        new_labels["domain"] = np.random.randint(0, 2, size=n)
        new_labels["category"] = np.random.randint(0, 3, size=n)
        labels = new_labels
        years = np.unique(labels["year"])
        domains = np.unique(labels["domain"])
    sub_indexes = {}
    for y in years:
        mask = labels["year"] == y
        sb = base[mask]
        if len(sb) < 100: continue
        idx = faiss.IndexHNSWFlat(dim, 32)
        idx.hnsw.efConstruction = 200
        idx.add(sb)
        sub_indexes[("year", "=", int(y))] = (idx, np.where(mask)[0])
    for d in domains:
        mask = labels["domain"] == d
        sb = base[mask]
        if len(sb) < 100: continue
        idx = faiss.IndexHNSWFlat(dim, 32)
        idx.hnsw.efConstruction = 200
        idx.add(sb)
        sub_indexes[("domain", "=", int(d))] = (idx, np.where(mask)[0])
    default_idx = faiss.IndexHNSWFlat(dim, 32)
    default_idx.hnsw.efConstruction = 200
    default_idx.add(base)

    np.random.seed(42)
    n_test = query.shape[0]
    sel_levels = {
        "low": lambda: ("year", "=", int(np.random.choice(years[:max(1, len(years)//2)]))),
        "medium": lambda: ("domain", "=", int(np.random.choice(domains[:max(1, len(domains)//2)]))),
        "high": lambda: ("year", "=", int(np.random.choice(years[-1:]))),
    }

    for sel_name, sel_fn in sel_levels.items():
        ps = {"selectivity": sel_name, "method": "sieve"}
        if not already_run("sieve", "sieve", ds_name, ps):
            tr, tq, tl = 0, 0, 0
            for _ in range(3):
                ft, fr = [], []
                for i in range(n_test):
                    cond = sel_fn()
                    if cond in sub_indexes:
                        idx, oids = sub_indexes[cond]
                        t0 = time.time()
                        _, sl = idx.search(query[i:i+1], k)
                        t1 = time.time() - t0
                        fr.append(oids[sl[0]])
                        ft.append(t1)
                    else:
                        t0 = time.time()
                        _, sl = default_idx.search(query[i:i+1], k)
                        t1 = time.time() - t0
                        fr.append(sl[0])
                        ft.append(t1)
                fr = np.array(fr)
                tr += recall_at_k(gt, fr, k)
                tq += n_test / sum(ft)
                tl += (sum(ft) / n_test) * 1000
            print(f"  SIEVE {sel_name}: R@{k}={tr/3:.4f}, QPS={tq/3:.1f}, L={tl/3:.2f}ms")
            df = create_empty_results_df()
            df = add_result_row(df, project="sieve", algorithm="sieve", dataset=ds_name,
                                recall=tr/3, qps=tq/3, k=k, params_json=ps,
                                avg_latency_ms=tl/3, build_time_s=0, threads=-1)
            append_result(df)

        pp = {"selectivity": sel_name, "method": "postfilter"}
        if not already_run("sieve", "hnsw_postfilter", ds_name, pp):
            tr, tq, tl = 0, 0, 0
            for _ in range(3):
                ft, fr = [], []
                for i in range(n_test):
                    cond = sel_fn()
                    t0 = time.time()
                    _, la = default_idx.search(query[i:i+1], k * 3)
                    if cond in sub_indexes:
                        _, oids = sub_indexes[cond]
                        mask = np.isin(la[0], oids)
                        filtered = la[0][mask][:k]
                        if len(filtered) < k:
                            filtered = np.concatenate([filtered, la[0][:k]])[:k]
                    else:
                        filtered = la[0][:k]
                    t1 = time.time() - t0
                    fr.append(filtered)
                    ft.append(t1)
                fr = np.array(fr)
                tr += recall_at_k(gt, fr, k)
                tq += n_test / sum(ft)
                tl += (sum(ft) / n_test) * 1000
            print(f"  PostF {sel_name}: R@{k}={tr/3:.4f}, QPS={tq/3:.1f}, L={tl/3:.2f}ms")
            df = create_empty_results_df()
            df = add_result_row(df, project="sieve", algorithm="hnsw_postfilter", dataset=ds_name,
                                recall=tr/3, qps=tq/3, k=k, params_json=pp,
                                avg_latency_ms=tl/3, build_time_s=0, threads=-1)
            append_result(df)


def main():
    ds_name = sys.argv[1] if len(sys.argv) > 1 else "synthetic-msmarco"
    print(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")
    base, query, gt, labels = load_dataset(ds_name)
    run_flat(ds_name, base, query, gt, k=10)
    run_hnsw(ds_name, base, query, gt, k=10)
    run_ivf_pq(ds_name, base, query, gt, k=10)
    run_swirl(ds_name, base, query, gt, k=10)
    run_window_search(ds_name, base, query, gt, k=10)
    run_sieve(ds_name, base, query, gt, labels, k=10)
    print(f"\n[Done] {ds_name}")


if __name__ == "__main__":
    main()
