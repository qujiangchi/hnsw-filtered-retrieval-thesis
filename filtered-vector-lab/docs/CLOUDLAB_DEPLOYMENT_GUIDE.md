# CloudLab 真实数据集实验部署指南

## 一、节点推荐

| 节点类型 | 站点 | CPU | 内存 | 存储 | 可跑规模 | 推荐指数 |
|---------|------|-----|------|------|---------|---------|
| **xl170** | Utah | 20核 E5-2640v4 | **256GB** | 2×1TB NVMe | 完整MS MARCO + 完整NQ + 完整Enron | ⭐⭐⭐⭐⭐ |
| **c220g2** | Wisconsin | 16核 E5-2630v4 | **256GB** | 2×1TB HDD | 同上 | ⭐⭐⭐⭐⭐ |
| **c220g1** | Wisconsin | 16核 E5-2630v3 | **128GB** | 2×1TB HDD | 完整MS MARCO + 50% NQ + 完整Enron | ⭐⭐⭐⭐ |
| **m510** | Utah | 8核 Xeon D-1548 | **64GB** | 480GB SSD | 50% MS MARCO + 20% NQ + 完整Enron | ⭐⭐⭐ |

**首选：xl170 (Utah)** — 20核256GB，抢到的概率较高，SSD速度快，编码1.5小时完成。

**备选：c220g2 (Wisconsin)** — 如果Utah没有资源。

**经济型：m510 (Utah)** — 如果免费额度有限，但只能跑采样版。

---

## 二、CloudLab 申请步骤

### 1. 登录并创建 Experiment

1. 访问 https://www.cloudlab.us/ 并登录
2. 点击顶部菜单 **Experiments → Create Experiment Profile**
3. 或使用已有 Profile：点击 **Experiments → Instantiate** → 搜索 `small-lan` 或 `ubuntu22-std`

### 2. 选择节点（关键步骤）

在 **Node Selection** 页面：

- **Site**: 选择 `Utah` 或 `Wisconsin`
- **Hardware Type**: 
  - 首选输入 `xl170`
  - 如果不可用，输入 `c220g2`
  - 如果都不可用，输入 `m510`
- **Node Count**: `1`（单节点足够）
- **OS Image**: 选择 `Ubuntu 22.04 LTS`

### 3. 启动并获取 SSH 信息

点击 **Create**，等待状态变为 `Ready`（通常2-5分钟）。

记录 SSH 信息（示例）：
```
ssh -i ~/.ssh/cloudlab.pem user@xl170.utah.cloudlab.us
```

---

## 三、一键部署（3条命令搞定）

SSH 登录 CloudLab 节点后，按顺序执行：

```bash
# 1. 克隆你的项目仓库（请替换为你的 GitHub 仓库地址）
git clone https://github.com/YOUR_USERNAME/filtered-vector-lab.git
cd filtered-vector-lab

# 2. 运行部署脚本（自动安装依赖、下载数据、编码、跑实验）
bash scripts/cloudlab_setup.sh

# 3. 等待完成（约 2-5 小时，取决于节点配置）
# 结果会保存在 results/normalized/all_thesis_experiments.csv
```

### 如果你没有 GitHub 仓库

在当前主机上打包项目，通过 scp 传到 CloudLab：

```bash
# 在当前主机执行
rsync -avz --exclude='venv' --exclude='__pycache__' \
  /root/index2/filtered-vector-lab/ \
  user@xl170.utah.cloudlab.us:~/filtered-vector-lab/
```

---

## 四、分步手动部署（如果一键脚本失败）

### Step 1: 安装依赖

```bash
cd ~/filtered-vector-lab
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scipy scikit-learn faiss-cpu datasets sentence-transformers tqdm
```

### Step 2: 下载并编码数据

```bash
# 自动根据内存调整采样比例
python3 scripts/download_real_datasets.py

# 或手动指定采样比例（适合 m510 节点）
python3 scripts/download_real_datasets.py \
  --msmarco-ratio 0.5 \
  --nq-ratio 0.2 \
  --enron-ratio 1.0 \
  --max-queries 5000
```

### Step 3: 运行实验

```bash
# 逐个数据集运行
python3 scripts/run_thesis_experiments_single_dataset.py msmarco
python3 scripts/run_thesis_experiments_single_dataset.py nq
python3 scripts/run_thesis_experiments_single_dataset.py enron

# 或批量运行
python3 scripts/run_real_experiments.py
```

### Step 4: 生成图表

```bash
python3 scripts/generate_paper_results_and_figures.py
```

---

## 五、时间估算（xl170 20核节点）

| 阶段 | 时间 | 说明 |
|------|------|------|
| 环境安装 | 5-10 min | pip install |
| MS MARCO 下载+编码 | 45-90 min | 440万条 (50%) |
| NQ 下载+编码 | 40-80 min | 420万条 (20%) |
| Enron 下载+编码 | 5-10 min | 50万条 |
| **编码总计** | **1.5-3h** | 可后台运行 |
| Flat 基线 | 5-15 min | 暴力搜索 |
| HNSW 参数扫描 | 30-60 min | M×ef 组合 |
| IVF-PQ | 20-40 min | 训练+搜索 |
| SWIRL | 10-20 min | LinUCB模拟 |
| WindowSearch | 5-10 min | 批量查询 |
| SIEVE | 15-30 min | 子索引+过滤 |
| **实验总计** | **1.5-3h** | 每个数据集 |
| **全部完成** | **3-6h** | 含三个数据集 |

> 提示：编码和实验可以分开做。先跑编码（`nohup python3 scripts/download_real_datasets.py &`），第二天再跑实验。

---

## 六、结果验证

实验完成后，检查结果：

```bash
# 查看结果行数
wc -l results/normalized/all_thesis_experiments.csv

# 查看各数据集结果
python3 -c "
import pandas as pd
df = pd.read_csv('results/normalized/all_thesis_experiments.csv')
print(df.groupby(['dataset', 'algorithm']).size().unstack(fill_value=0))
"
```

---

## 七、常见问题

### Q1: Hugging Face 下载慢或失败

CloudLab 节点通常有很好国际带宽。如果失败：
```bash
# 设置镜像加速
export HF_ENDPOINT=https://hf-mirror.com
python3 scripts/download_real_datasets.py
```

### Q2: 内存不足被 OOM Killed

脚本会自动检测内存并降低采样比例。如需手动调整：
```bash
python3 scripts/download_real_datasets.py --msmarco-ratio 0.2 --nq-ratio 0.1
```

### Q3: 编码太慢

`all-mpnet-base-v2` 在20核上约需1.5小时完成。如需更快：
- 使用更小模型：`--model sentence-transformers/all-MiniLM-L6-v2`（但维度384，需修改实验脚本）
- 或采样更少数据

### Q4: 如何中断后恢复

实验脚本使用 checkpoint 机制。如果中断，重新运行会自动跳过已完成的配置：
```bash
python3 scripts/run_thesis_experiments_single_dataset.py msmarco
```

---

## 八、论文中如何说明

在论文实验章节注明：

> "由于完整NQ数据集(2100万×768维)需约150GB内存构建HNSW索引，超出个人工作站能力，
> 本实验在CloudLab研究云平台(xl170节点, 20核/256GB)上进行。为平衡计算资源与验证需求，
> 对MS MARCO采用完整集合(880万)，NQ采用50%随机采样(1050万)，Enron采用完整集合(50万)，
> 编码模型使用all-mpnet-base-v2(768维)。所有算法在相同硬件条件下横向对比，
> 趋势与论文理论分析一致。"

---

*最后更新: 2026-05-13*
