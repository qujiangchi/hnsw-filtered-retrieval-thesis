# GPU 加速方案指南

## 一、GPU 到底能加速什么

| 阶段 | CPU 20核 | GPU (T4/V100) | 加速比 | HNSW支持GPU？ |
|------|---------|--------------|--------|--------------|
| 文本编码 (440万条) | ~50 min | ~10 min (T4) / ~5 min (V100) | **5-10×** | N/A |
| Ground Truth 计算 | ~15 min | ~0.5 min | **30×** | N/A |
| Flat 暴力搜索 | ~15 min | ~1 min | **15×** | N/A |
| **HNSW 索引构建** | ~30 min | ~30 min | **1×** | ❌ 不支持 |
| **HNSW 查询搜索** | ~5 min | ~5 min | **1×** | ❌ 不支持 |
| IVF-PQ 训练+搜索 | ~30 min | ~10 min | **3×** | 部分支持 |

**结论**：GPU 的价值主要在**编码**和**Ground Truth**两个阶段，能节省约 **1-1.5小时**。
HNSW（论文核心算法）**不支持 GPU**，所以 GPU 不能加速索引构建和 HNSW 查询。

---

## 二、GPU 云实例推荐

### 方案1：国内GPU平台（性价比最高）⭐⭐⭐⭐⭐

| 平台 | 实例 | CPU | 内存 | GPU | 价格 | 6h成本 |
|------|------|-----|------|-----|------|--------|
| **AutoDL** | RTX 4090 + 128GB | 16核 | **128GB** | RTX 4090 | **2.5元/h** | **15元** |
| AutoDL | RTX 3090 + 64GB | 12核 | 64GB | RTX 3090 | 1.5元/h | 9元 |
| 恒源云 | V100-32G + 64GB | 8核 | 64GB | V100 | 2.0元/h | 12元 |

**首选 AutoDL RTX 4090 + 128GB**：
- 128GB 内存足够跑完整 MS MARCO
- RTX 4090 编码速度约 15000 doc/s
- 总价 15元，比纯 CPU 方案（7元）只贵 8元，但快 1.5小时

**操作步骤**：
1. 访问 https://www.autodl.com 注册
2. 选地区 → GPU容器实例 → 筛选 **内存≥128GB**
3. 镜像选 **PyTorch 2.x + Ubuntu 22.04**
4. 创建后通过 JupyterLab 或 SSH 登录
5. 运行：
```bash
git clone https://github.com/你的用户名/filtered-vector-lab.git
cd filtered-vector-lab
bash scripts/cloudlab_setup.sh
```

### 方案2：分步方案（最省钱）⭐⭐⭐⭐

如果不想花15元，可以**拆成两步**：

```
Step 1: 租便宜GPU实例 1小时（只编码）
        → AutoDL RTX 3090 (1.5元/h) × 1h = 1.5元
        → 编码完保存向量到对象存储/本地

Step 2: 租阿里云抢占式CPU (7元) 跑实验
        → HNSW构建、查询实验全部在CPU大内存上完成

总计: ~8.5元，时间 ≈ 纯CPU方案
```

**适合**：预算极紧，不介意多花操作时间。

### 方案3：阿里云GPU（贵，不推荐）

| 实例 | 配置 | 价格 | 6h成本 |
|------|------|------|--------|
| gn6i-c16g1.4xlarge | 16核64GB + T4 | 12元/h | 72元 |
| gn7-c16g1.8xlarge | 64核256GB + A10 | 60元/h | 360元 |

阿里云GPU实例**内存普遍偏小**（64GB），跑完整MS MARCO不够。大内存+GPU的非常贵。

---

## 三、修改脚本支持 GPU

如果使用 GPU 实例，需要把 `faiss-cpu` 换成 `faiss-gpu`，并让编码脚本使用 CUDA。

### 修改 cloudlab_setup.sh

在 `pip install faiss-cpu` 那一行，改为检测GPU：

```bash
# 检测是否有 NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU detected, installing faiss-gpu..."
    pip install faiss-gpu==1.8.0 -q
else
    echo "  No GPU, installing faiss-cpu..."
    pip install faiss-cpu==1.8.0 -q
fi
```

### 修改 download_real_datasets.py 使用 GPU 编码

在 `encode_texts` 函数中，如果有GPU，自动使用：

```python
def encode_texts(model, texts, batch_size=128, desc="Encoding"):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,  # 自动选择 GPU
    )
```

这样脚本在 GPU 实例上编码速度自动提升 5-10 倍。

---

## 四、综合对比

| 方案 | 总价 | 总时间 | 操作复杂度 | 推荐场景 |
|------|------|--------|-----------|---------|
| 阿里云CPU抢占式 | **7元** | 6h | 低 | 预算最低，不赶时间 |
| AutoDL 4090+128GB | **15元** | 4h | 低 | 首选，性价比最高 |
| 分步(GPU编码+CPU实验) | **8.5元** | 6h | 高 | 预算紧且愿意折腾 |
| 阿里云GPU+大内存 | 360元 | 4h | 低 | 土豪随意 |

---

## 五、论文中如何写GPU环境

```
实验在配备 NVIDIA RTX 4090 GPU (24GB显存) 和 128GB CPU内存的
云服务器上进行。GPU主要用于加速文本编码（sentence-transformers
all-mpnet-base-v2模型）和Faiss精确最近邻计算（IndexFlatL2），
HNSW索引构建因算法限制仍使用CPU完成。
```

---

*最后更新: 2026-05-13*
