#!/bin/bash
set -e

echo "=========================================="
echo "Filtered Vector Lab - Environment Setup"
echo "=========================================="

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

echo "Python version: $(python3 --version)"

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# 升级 pip
echo "Upgrading pip..."
pip install --upgrade pip

# 安装核心依赖
echo "Installing core dependencies..."
pip install numpy pandas h5py pyarrow plotly streamlit pyyaml pytest

# 安装 hnswlib
echo "Installing hnswlib..."
pip install third_party/hnswlib

# 提示其他项目需手动安装
echo ""
echo "=========================================="
echo "Core dependencies installed."
echo ""
echo "Optional: Install other third-party projects manually:"
echo "  RangeFilteredANN: cd third_party/RangeFilteredANN && pip install ."
echo "  SIEVE: cd third_party/SIEVE/hnswtest && pip install ."
echo "  ANN-Benchmarks: cd third_party/ann-benchmarks && pip install ."
echo "=========================================="
