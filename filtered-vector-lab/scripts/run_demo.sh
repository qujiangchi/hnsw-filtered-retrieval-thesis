#!/bin/bash
set -e

echo "=========================================="
echo "Filtered Vector Lab - Demo Run"
echo "=========================================="

# 检查虚拟环境
if [ -d "venv" ]; then
    echo "[demo] Activating virtual environment..."
    source venv/bin/activate
else
    echo "[demo] No venv found, using system python3"
fi

# 1. 生成合成数据
echo ""
echo "[demo] Step 1: Generating synthetic-small dataset..."
python scripts/generate_synthetic.py --output-dir data/processed/synthetic-small

# 2. 跑 hnswlib baseline（多组参数）
echo ""
echo "[demo] Step 2: Running hnswlib baseline experiments..."
python src/adapters/hnswlib_adapter.py --dataset synthetic-small --M 16 --ef-construction 100 --ef 20
python src/adapters/hnswlib_adapter.py --dataset synthetic-small --M 16 --ef-construction 100 --ef 50
python src/adapters/hnswlib_adapter.py --dataset synthetic-small --M 16 --ef-construction 200 --ef 50
python src/adapters/hnswlib_adapter.py --dataset synthetic-small --M 32 --ef-construction 200 --ef 100

# 3. 合并 normalized results
echo ""
echo "[demo] Step 3: Merging normalized results..."
python -c "
import sys
sys.path.insert(0, '.')
from src.storage.result_store import merge_all_normalized
merge_all_normalized()
print('[demo] Merge complete: results/normalized/all_results.csv')
"

# 4. 启动 Dashboard
echo ""
echo "[demo] Step 4: Starting Streamlit Dashboard..."
echo "[demo] Dashboard will be available at http://localhost:8501"
streamlit run dashboard/app.py

echo ""
echo "[demo] Done."
