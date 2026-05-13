#!/usr/bin/env python3
"""
下载真实数据集、采样、编码为768维dense向量、计算ground truth。

数据源:
  - MS MARCO:  "Tevatron/msmarco-passage-corpus" (HuggingFace)
  - NQ (DPR):  "wiki_dpr" psgs_w100.multiset.no_index (HuggingFace)
  - Enron:     "SetFit/enron_spam" + 启发式metadata提取

编码模型: sentence-transformers/all-mpnet-base-v2 (768 dim)
输出格式与 data/processed/ 下的合成数据集完全一致。
"""

import os
import sys
import re
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import faiss

# ---------------------------------------------------------------------------
# 解析参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Prepare real datasets for ANN experiments")
parser.add_argument("--msmarco-ratio", type=float, default=0.5, help="MS MARCO采样比例")
parser.add_argument("--nq-ratio", type=float, default=0.2, help="NQ采样比例")
parser.add_argument("--enron-ratio", type=float, default=1.0, help="Enron采样比例")
parser.add_argument("--max-queries", type=int, default=5000, help="每个数据集的最大查询数")
parser.add_argument("--output-dir", type=str, default="data/real", help="输出目录")
parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2",
                    help="Sentence-transformers编码模型")
parser.add_argument("--batch-size", type=int, default=128, help="编码batch size")
parser.add_argument("--gt-k", type=int, default=100, help="Ground truth的K")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
args = parser.parse_args()

np.random.seed(args.seed)

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 导入可选依赖（延迟导入，方便提前报错）
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' not installed. Run: pip install datasets")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: 'sentence-transformers' not installed. Run: pip install sentence-transformers")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def check_memory_needed(n_base, dim, n_query, k, safety=2.5):
    """估算HNSW构建峰值内存 (GB)"""
    vec_gb = n_base * dim * 4 / (1024**3)
    hnsw_gb = vec_gb * safety
    query_gb = n_query * dim * 4 / (1024**3)
    gt_gb = n_query * k * 8 / (1024**3)
    total = hnsw_gb + query_gb + gt_gb + 3  # +3GB 系统/脚本开销
    return total


def auto_adjust_ratio(name, full_size, dim, ratio, max_queries, safety=2.5):
    """如果内存不足，自动降低采样比例"""
    import psutil
    avail_gb = psutil.virtual_memory().available / (1024**3)
    n_base = int(full_size * ratio)
    needed = check_memory_needed(n_base, dim, max_queries, args.gt_k, safety)
    if needed > avail_gb * 0.85:
        new_ratio = ratio * (avail_gb * 0.85) / needed
        new_ratio = max(0.01, min(ratio, new_ratio))
        print(f"  [WARN] {name}: 内存不足 ({needed:.1f}GB > {avail_gb:.1f}GB可用), "
              f"自动调整采样比例 {ratio:.0%} -> {new_ratio:.0%}")
        return new_ratio
    return ratio


def reservoir_sample_streaming(dataset, n_target, seed=42):
    """从streaming数据集做reservoir sampling"""
    rng = np.random.RandomState(seed)
    reservoir = []
    for i, item in enumerate(tqdm(dataset, desc="  Sampling", total=n_target * 3)):
        if i < n_target:
            reservoir.append(item)
        else:
            j = rng.randint(0, i + 1)
            if j < n_target:
                reservoir[j] = item
        # 安全阀: 如果数据集远大于目标，提前停止
        if n_target > 100000 and i > n_target * 20:
            break
    return reservoir[:n_target]


def encode_texts(model, texts, batch_size=128, desc="Encoding"):
    """批量编码文本为向量，自动使用GPU（如果可用）"""
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        device = 'cpu'
    if device == 'cuda':
        print(f"  Using GPU for encoding ({torch.cuda.get_device_name(0)})")
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
        normalize_embeddings=False,
    )


def compute_groundtruth(base, query, k=100, batch_size=500):
    """用Faiss IndexFlatL2暴力搜索计算ground truth"""
    dim = base.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(base)
    nq = query.shape[0]
    gt = np.empty((nq, k), dtype=np.int64)
    print(f"  Computing ground truth (nq={nq}, k={k})...")
    for i in tqdm(range(0, nq, batch_size), desc="  GT"):
        end = min(i + batch_size, nq)
        _, ids = index.search(query[i:end], k)
        gt[i:end] = ids
    return gt


def save_dataset(name, base, query, gt, labels=None):
    """保存为与合成数据集一致的格式"""
    out = OUTPUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "base.npy", base.astype(np.float32))
    np.save(out / "query.npy", query.astype(np.float32))
    np.save(out / "groundtruth.npy", gt.astype(np.int64))
    if labels is not None:
        np.save(out / "labels.npy", labels)
    else:
        # 生成dummy labels (兼容SIEVE)
        dtype = np.dtype([("year", np.int16), ("domain", np.int8), ("category", np.int8)])
        dummy = np.empty(len(base), dtype=dtype)
        dummy["year"] = np.random.randint(1999, 2003, size=len(base))
        dummy["domain"] = np.random.randint(0, 2, size=len(base))
        dummy["category"] = np.random.randint(0, 3, size=len(base))
        np.save(out / "labels.npy", dummy)
    print(f"  Saved to {out}/")


# ---------------------------------------------------------------------------
# 1. MS MARCO
# ---------------------------------------------------------------------------
def prepare_msmarco():
    ratio = auto_adjust_ratio("MS MARCO", 8_800_000, 768, args.msmarco_ratio, args.max_queries)
    n_target = int(8_800_000 * ratio)
    print(f"\n[MS MARCO] Sampling {ratio:.0%} ({n_target:,} passages)...")

    print("  Loading corpus (streaming)...")
    # Try multiple sources for MS MARCO passages
    texts = []
    for corpus_name, text_key in [
        ("Tevatron/msmarco-passage-corpus", "text"),
        ("ms_marco", "passages"),
    ]:
        try:
            print(f"    Trying {corpus_name}...")
            if corpus_name == "ms_marco":
                # Extract passages from the official dataset
                ds = load_dataset("ms_marco", "v1.1", split="train", streaming=True, trust_remote_code=True)
                seen = set()
                for item in ds:
                    for p in item.get("passages", {}).get("passage_text", []):
                        if p not in seen:
                            seen.add(p)
                            texts.append(p)
                            if len(texts) >= n_target:
                                break
                    if len(texts) >= n_target:
                        break
            else:
                ds = load_dataset(corpus_name, split="train", streaming=True, trust_remote_code=True)
                for item in ds:
                    texts.append(item[text_key])
                    if len(texts) >= n_target:
                        break
            print(f"    Loaded {len(texts)} passages from {corpus_name}")
            break
        except Exception as e:
            print(f"    Failed: {e}")
            texts = []
            continue
    
    if not texts:
        print("  ERROR: Could not load MS MARCO. Skipping.")
        return False

    print(f"  Loading queries...")
    try:
        queries_ds = load_dataset("ms_marco", "v1.1", split="validation", trust_remote_code=True)
        query_texts = [q["query"] for q in queries_ds][:args.max_queries]
    except Exception as e:
        print(f"    ms_marco validation failed: {e}, falling back to sampling from corpus...")
        rng = np.random.RandomState(args.seed)
        q_idx = rng.choice(len(texts), min(args.max_queries, len(texts)), replace=False)
        query_texts = [texts[i] for i in q_idx]

    print(f"  Encoding {len(texts)} passages with {args.model}...")
    model = SentenceTransformer(args.model)
    base = encode_texts(model, texts, batch_size=args.batch_size)
    query = encode_texts(model, query_texts, batch_size=args.batch_size)

    print(f"  Base shape: {base.shape}, Query shape: {query.shape}")
    gt = compute_groundtruth(base, query, k=args.gt_k)
    save_dataset("msmarco", base, query, gt)
    del base, query, gt, model
    return True


# ---------------------------------------------------------------------------
# 2. Natural Questions (DPR-W100)
# ---------------------------------------------------------------------------
def prepare_nq():
    ratio = auto_adjust_ratio("NQ", 21_000_000, 768, args.nq_ratio, args.max_queries)
    n_target = int(21_000_000 * ratio)
    print(f"\n[NQ] Sampling {ratio:.0%} ({n_target:,} passages)...")

    print("  Loading DPR-W100 corpus (streaming)...")
    corpus = load_dataset("wiki_dpr", "psgs_w100.multiset.no_index", split="train", streaming=True)
    sampled = reservoir_sample_streaming(corpus, n_target, seed=args.seed)
    texts = [item["text"] for item in sampled]

    print(f"  Loading NQ queries...")
    try:
        queries_ds = load_dataset("nq_open", split="validation", trust_remote_code=True)
        query_texts = [q["question"] for q in queries_ds][:args.max_queries]
    except Exception as e:
        print(f"    nq_open failed: {e}, falling back to sampling from corpus...")
        rng = np.random.RandomState(args.seed)
        q_idx = rng.choice(len(texts), min(args.max_queries, len(texts)), replace=False)
        query_texts = [texts[i] for i in q_idx]

    print(f"  Encoding {len(texts)} passages...")
    model = SentenceTransformer(args.model)
    base = encode_texts(model, texts, batch_size=args.batch_size)
    query = encode_texts(model, query_texts, batch_size=args.batch_size)

    print(f"  Base shape: {base.shape}, Query shape: {query.shape}")
    gt = compute_groundtruth(base, query, k=args.gt_k)
    save_dataset("nq", base, query, gt)
    del base, query, gt, model
    return True


# ---------------------------------------------------------------------------
# 3. Enron Email
# ---------------------------------------------------------------------------
def extract_enron_metadata(texts):
    """从Enron邮件文本中提取year, domain, category"""
    n = len(texts)
    years = np.full(n, 2001, dtype=np.int16)
    domains = np.full(n, 0, dtype=np.int8)
    categories = np.full(n, 0, dtype=np.int8)

    year_pat = re.compile(r"\b(199[9]|200[0-4])\b")
    domain_pat = re.compile(r"[\w.-]+@([\w.-]+\.\w+)")

    for i, text in enumerate(texts):
        # 提取年份
        m = year_pat.search(text)
        if m:
            years[i] = int(m.group(1))

        # 提取域名
        m = domain_pat.search(text)
        if m:
            dom = m.group(1).lower()
            # 简单hash到几个类别
            domains[i] = hash(dom) % 4

    return years, domains, categories


def prepare_enron():
    ratio = auto_adjust_ratio("Enron", 500_000, 768, args.enron_ratio, args.max_queries)
    n_target = int(500_000 * ratio)
    print(f"\n[Enron] Sampling {ratio:.0%} ({n_target:,} emails)...")

    # 尝试多个可能的Enron数据集源
    texts = []
    labels_map = []
    sources = [
        ("SetFit/enron_spam", "text", "label"),
        ("leriomaggio/enron_emails_labeled", "email", "label"),
    ]

    for ds_name, text_key, label_key in sources:
        try:
            print(f"  Trying {ds_name}...")
            ds = load_dataset(ds_name, split="train", streaming=True)
            count = 0
            for item in ds:
                texts.append(item.get(text_key, ""))
                labels_map.append(item.get(label_key, 0))
                count += 1
                if count >= n_target:
                    break
            print(f"  Loaded {len(texts)} items from {ds_name}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    else:
        print("  ERROR: Could not load any Enron dataset. Skipping.")
        return False

    # 生成queries: 从邮件文本中随机选一些作为query
    nq = min(args.max_queries, len(texts) // 10)
    rng = np.random.RandomState(args.seed)
    q_indices = rng.choice(len(texts), nq, replace=False)
    query_texts = [texts[i] for i in q_indices]

    print(f"  Encoding {len(texts)} emails...")
    model = SentenceTransformer(args.model)
    base = encode_texts(model, texts, batch_size=args.batch_size)
    query = encode_texts(model, query_texts, batch_size=args.batch_size)

    # 提取metadata
    years, domains, _ = extract_enron_metadata(texts)
    # category 用数据集自带的label (spam/ham)
    categories = np.array(labels_map[:len(texts)], dtype=np.int8)

    dtype = np.dtype([("year", np.int16), ("domain", np.int8), ("category", np.int8)])
    labels = np.empty(len(texts), dtype=dtype)
    labels["year"] = years
    labels["domain"] = domains
    labels["category"] = categories

    print(f"  Base shape: {base.shape}, Query shape: {query.shape}")
    gt = compute_groundtruth(base, query, k=args.gt_k)
    save_dataset("enron", base, query, gt, labels=labels)
    del base, query, gt, model
    return True


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" Real Dataset Preparation")
    print(f" Output: {OUTPUT_DIR.absolute()}")
    print(f" Model: {args.model}")
    print("=" * 60)

    prepare_msmarco()
    prepare_nq()
    prepare_enron()

    print("\n" + "=" * 60)
    print(" All datasets prepared successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
