#!/bin/bash
# =============================================================================
# CloudLab 一键部署脚本
# 在新申请的 CloudLab 节点上执行，自动完成环境搭建、数据下载、实验运行
# 
# 用法:
#   1. SSH 进入 CloudLab 节点
#   2. git clone <你的项目仓库>
#   3. cd filtered-vector-lab/scripts
#   4. bash cloudlab_setup.sh
# =============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "  CloudLab Real Dataset Deployment"
echo "  Node: $(hostname)"
echo "  CPUs: $(nproc)"
echo "  RAM:  $(free -h | awk '/^Mem:/ {print $2}')"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. 系统依赖
# ---------------------------------------------------------------------------
echo "[1/5] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-pip python3-venv wget curl build-essential

# ---------------------------------------------------------------------------
# 2. Python 虚拟环境
# ---------------------------------------------------------------------------
echo "[2/5] Setting up Python venv..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q

# 核心依赖
pip install numpy==1.24.4 pandas scipy scikit-learn tqdm -q

# 检测GPU，自动选择 faiss-gpu 或 faiss-cpu
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "  NVIDIA GPU detected, installing faiss-gpu..."
    pip install faiss-gpu==1.8.0 -q
else
    echo "  No GPU detected, installing faiss-cpu..."
    pip install faiss-cpu==1.8.0 -q
fi

# 数据下载与编码
pip install datasets==2.18.0 sentence-transformers==2.5.1 -q

# docx 处理（用于后续论文图表更新）
pip install python-docx -q

echo "  Installed packages:"
python3 -c "import faiss, numpy, pandas, datasets, sentence_transformers; print('  OK')"

# ---------------------------------------------------------------------------
# 3. 数据目录准备
# ---------------------------------------------------------------------------
echo "[3/5] Preparing data directories..."
mkdir -p data/real/msmarco data/real/nq data/real/enron
mkdir -p results/normalized

# ---------------------------------------------------------------------------
# 4. 下载并编码真实数据集
#    默认采样: MS MARCO 50%, NQ 20%, Enron 100%
#    如需调整，修改下面的环境变量后重新运行
# ---------------------------------------------------------------------------
echo "[4/5] Downloading & encoding real datasets..."
echo "  WARNING: This step takes 1-3 hours on a 20-core node."
echo "  Sampling ratios: MSMARCO=${MSMARCO_RATIO:-0.5}, NQ=${NQ_RATIO:-0.2}, ENRON=${ENRON_RATIO:-1.0}"

MSMARCO_RATIO="${MSMARCO_RATIO:-0.5}"
NQ_RATIO="${NQ_RATIO:-0.2}"
ENRON_RATIO="${ENRON_RATIO:-1.0}"
MAX_QUERIES="${MAX_QUERIES:-5000}"

python3 scripts/download_real_datasets.py \
    --msmarco-ratio "$MSMARCO_RATIO" \
    --nq-ratio "$NQ_RATIO" \
    --enron-ratio "$ENRON_RATIO" \
    --max-queries "$MAX_QUERIES" \
    --output-dir data/real

# ---------------------------------------------------------------------------
# 5. 运行实验
# ---------------------------------------------------------------------------
echo "[5/5] Running experiments..."

# 5.1 MS MARCO
python3 scripts/run_thesis_experiments_single_dataset.py msmarco

# 5.2 NQ
python3 scripts/run_thesis_experiments_single_dataset.py nq

# 5.3 Enron
python3 scripts/run_thesis_experiments_single_dataset.py enron

# 5.4 生成图表（可选，如果论文图表需要更新）
# python3 scripts/generate_paper_results_and_figures.py

echo ""
echo "=========================================="
echo "  All experiments completed!"
echo "  Results: results/normalized/"
echo "=========================================="
