# Filtered Vector Lab

基于 HNSW 的过滤向量检索优化方法研究与实现 —— 统一实验平台。

## 项目目标

集成 hnswlib、RangeFilteredANN、SIEVE、ANN-Benchmarks/VIBE 的实验输出，统一解析成标准实验结果表，并用 Streamlit 可视化 recall-QPS、filter_width-QPS、build time、memory/index size、单查询 top-k 解释。

## 核心原则

- **不要一次性做完整系统**：每次只完成一个可验证的小任务
- **先 MVP，后完善**：先用 Python + Streamlit + Pandas + Plotly 跑通，再考虑复杂前端
- **严格依据真实输出**：所有 Schema 和 Parser 必须基于各项目实际生成的 CSV/HDF5，不猜字段
- **主线固定**：HNSW 作为基础 ANN 检索底座；RangeFilteredANN 做数值窗口过滤；SIEVE 做复杂谓词过滤和索引集合选择；ANN-Benchmarks/VIBE 做统一评测参考

## 系统边界（MVP）

**必须集成：**
1. hnswlib：基础 HNSW 检索（baseline）
2. RangeFilteredANN：窗口过滤向量检索（β-WST + ANN）
3. SIEVE：谓词过滤向量检索（索引集合选择）
4. ANN-Benchmarks / VIBE：通用 benchmark 输出参考与格式
5. Streamlit Dashboard：统一可视化

**作为扩展/加分（不阻塞主线）：**
- SWIRL / BALANCE：RL 索引选择思想对比
- index_selection_evaluation：数据库二级索引选择
- TPC-H / TPC-DS：数据库 workload

## 目录结构

```
filtered-vector-lab/
├── README.md
├── configs/                  # 实验配置
│   ├── datasets.yaml
│   ├── experiments.yaml
│   └── algorithms.yaml
├── third_party/              # 第三方源码（只读，不改核心算法）
│   ├── hnswlib/
│   ├── RangeFilteredANN/
│   ├── SIEVE/
│   ├── ann-benchmarks/
│   └── vibe/
├── data/                     # 数据集
│   ├── raw/
│   ├── processed/
│   └── groundtruth/
├── results/                  # 实验结果
│   ├── raw/                  # 各项目原始输出
│   ├── normalized/           # 统一 Schema 后的 CSV
│   └── figures/              # 导出图表
├── src/                      # 核心源码
│   ├── adapters/             # 各项目实验封装
│   │   ├── hnswlib_adapter.py
│   │   ├── range_filtered_ann_adapter.py
│   │   ├── sieve_adapter.py
│   │   └── ann_benchmarks_adapter.py
│   ├── parsers/              # 结果解析器
│   │   ├── parse_range_results.py
│   │   ├── parse_sieve_results.py
│   │   ├── parse_annbench_hdf5.py
│   │   └── parse_hnswlib_results.py
│   ├── metrics/              # 指标计算
│   │   ├── recall.py
│   │   ├── latency.py
│   │   └── pareto.py
│   ├── runner/               # 实验调度
│   │   └── run_experiment.py
│   └── storage/              # 结果存储
│       └── result_store.py
├── dashboard/                # Streamlit 可视化
│   └── app.py
├── scripts/                  # 一键脚本
│   ├── setup_env.sh
│   ├── generate_synthetic.py
│   ├── run_mvp.sh
│   └── run_demo.sh
└── docs/                     # 文档
    ├── development_log.md
    ├── experiment_design.md
    ├── result_schema.md
    └── thesis_notes.md
```

## 快速开始

### 1. 环境准备

```bash
bash scripts/setup_env.sh
```

### 2. 生成最小合成数据

```bash
python scripts/generate_synthetic.py
```

### 3. 跑 hnswlib baseline

```bash
python src/adapters/hnswlib_adapter.py --dataset synthetic-small --M 16 --ef 50
```

### 4. 启动 Dashboard

```bash
streamlit run dashboard/app.py
```

### 5. 一键演示

```bash
bash scripts/run_demo.sh
```

## 统一实验结果 Schema

详见 [docs/result_schema.md](docs/result_schema.md)。

核心字段包括：
- `run_id`, `project`, `algorithm`, `dataset`
- `filter_type`, `filter_width`, `selectivity`, `k`
- `params_json`, `recall`, `qps`
- `avg_latency_ms`, `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms`
- `build_time_s`, `index_size_kb`, `memory_kb`
- `distcomps`, `threads`, `raw_result_path`, `created_at`

## 当前阶段

- [x] 阶段 0：确定毕业设计边界
- [x] 阶段 1：建立项目骨架
- [ ] 阶段 2：盘点各开源项目输出（进行中，已解析 hnswlib/RangeFilteredANN/SIEVE/ANN-Benchmarks/VIBE）
- [ ] 阶段 3：先跑通最小实验
- [ ] 阶段 4：设计统一实验结果 Schema
- [ ] 阶段 5：写结果解析器
- [ ] 阶段 6：实现统一实验调度器
- [ ] 阶段 7：实现核心可视化 Dashboard
- [ ] 阶段 8：补充单查询解释和 trace
- [ ] 阶段 9：整理实验结论和论文图表
- [ ] 阶段 10：部署、演示、答辩材料

## 许可证

各 third_party 子项目保留其原始许可证。本实验平台代码仅用于毕业设计研究。
