#!/bin/bash
# =============================================================================
# AutoDL 专用一键部署脚本
# 针对 AutoDL 环境优化：
#   - 使用数据盘 /root/autodl-tmp/（空间大）
#   - 配置 Hugging Face 国内镜像
#   - 复用预装的 PyTorch/CUDA，只装缺失依赖
#   - 自动检测 GPU 并安装 faiss-gpu
# =============================================================================
set -e

# AutoDL 数据盘路径
WORK_DIR="/root/autodl-tmp/filtered-vector-lab"
mkdir -p /root/autodl-tmp

echo "=========================================="
echo "  AutoDL Real Dataset Deployment"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'None')"
echo "  CPUs: $(nproc)"
echo "  RAM:  $(free -h | awk '/^Mem:/ {print $2}')"
echo "=========================================="

# ---------------------------------------------------------------------------
# 0. 检查是否在 AutoDL 环境
# ---------------------------------------------------------------------------
if [ -d "/root/autodl-tmp" ]; then
    echo "[INFO] AutoDL environment detected."
else
    echo "[WARN] /root/autodl-tmp not found. Are you on AutoDL?"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 1. 配置 Hugging Face 国内镜像
# ---------------------------------------------------------------------------
echo "[1/6] Configuring HF mirror for China network..."
export HF_ENDPOINT=https://hf-mirror.com
if ! grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null; then
    echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
fi

# ---------------------------------------------------------------------------
# 2. 进入工作目录（AutoDL 数据盘）
# ---------------------------------------------------------------------------
echo "[2/6] Setting up working directory..."
if [ ! -d "$WORK_DIR" ]; then
    echo "  ERROR: $WORK_DIR not found."
    echo "  Please clone your repo first:"
    echo "    cd /root/autodl-tmp"
    echo "    git clone https://github.com/YOUR_USERNAME/filtered-vector-lab.git"
    exit 1
fi
cd "$WORK_DIR"

# ---------------------------------------------------------------------------
# 3. 安装缺失依赖（AutoDL 已预装 PyTorch/CUDA）
# ---------------------------------------------------------------------------
echo "[3/6] Installing Python dependencies..."
# 升级 pip
pip install --upgrade pip -q

# 核心依赖
pip install numpy==1.24.4 pandas scipy scikit-learn tqdm -q

# Faiss: 检测GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "  GPU detected, installing faiss-gpu..."
    pip install faiss-gpu==1.8.0 -q
else
    echo "  No GPU, installing faiss-cpu..."
    pip install faiss-cpu==1.8.0 -q
fi

# 数据下载与编码
pip install datasets==2.18.0 sentence-transformers==2.5.1 -q

# docx 处理
pip install python-docx -q

echo "  Installed packages:"
python3 -c "
import faiss, numpy, pandas, datasets, sentence_transformers
print('  faiss:', faiss.__version__)
print('  numpy:', numpy.__version__)
print('  datasets:', datasets.__version__)
gpu = 'GPU' if hasattr(faiss, 'StandardGpuResources') else 'CPU'
print('  faiss backend:', gpu)
"

# ---------------------------------------------------------------------------
# 4. 数据目录准备
# ---------------------------------------------------------------------------
echo "[4/6] Preparing data directories..."
mkdir -p data/real/msmarco data/real/nq data/real/enron
mkdir -p results/normalized

# ---------------------------------------------------------------------------
# 5. 下载并编码真实数据集
# ---------------------------------------------------------------------------
echo "[5/6] Downloading & encoding real datasets..."
echo "  WARNING: This step takes 1-3 hours."

# 根据内存自动调整采样比例
MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
MEM_GB=$((MEM_KB / 1024 / 1024))
echo "  Detected ${MEM_GB}GB RAM"

if [ "$MEM_GB" -lt 80 ]; then
    MSMARCO_R=0.2; NQ_R=0.1
elif [ "$MEM_GB" -lt 150 ]; then
    MSMARCO_R=0.5; NQ_R=0.2
else
    MSMARCO_R=1.0; NQ_R=0.5
fi
echo "  Using ratios: msmarco=${MSMARCO_R}, nq=${NQ_R}, enron=1.0"

python3 scripts/download_real_datasets.py \
    --msmarco-ratio "$MSMARCO_R" \
    --nq-ratio "$NQ_R" \
    --enron-ratio 1.0 \
    --max-queries 5000 \
    --output-dir data/real

# ---------------------------------------------------------------------------
# 6. 运行实验
# ---------------------------------------------------------------------------
echo "[6/6] Running experiments..."

python3 scripts/run_thesis_experiments_single_dataset.py msmarco
python3 scripts/run_thesis_experiments_single_dataset.py nq
python3 scripts/run_thesis_experiments_single_dataset.py enron

# 生成图表
python3 scripts/generate_paper_results_and_figures.py || true

echo ""
echo "=========================================="
echo "  All experiments completed!"
echo "  Results: results/normalized/"
echo "  Figures: docs/figures/"
echo "=========================================="
echo ""
echo "Tips:"
echo "  - 关机保留数据: AutoDL控制台 → 关机（停止计费）"
echo "  - 下载结果到本地: scp -P <port> -r root@<host>:/root/autodl-tmp/filtered-vector-lab/results ./"
echo "  - 网盘持久化: cp -r results /root/autodl-pub/"
