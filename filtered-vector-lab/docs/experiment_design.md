# 实验设计文档

## 1. 我解决什么问题？

用三句话说清：
1. **问题**：高维向量近似最近邻（ANN）检索在带过滤条件（数值区间、复杂谓词）时，现有方法在 recall、QPS、构建成本之间存在显著 trade-off，缺乏统一实验平台系统对比。
2. **集成算法**：以 HNSW 为基座，集成 RangeFilteredANN（窗口过滤）、SIEVE（谓词过滤与索引集合选择）、ANN-Benchmarks（通用评测框架）。
3. **可视化证明**：通过 recall-QPS 帕累托曲线、filter width / selectivity 敏感性图、build time / memory 对比图、单查询 top-k 解释页，证明不同过滤场景下各算法的优劣。

## 2. 主线与扩展

### 主线（必须完成）

| 模块 | 输入 | 输出 | 可视化指标 | 验收标准 |
|------|------|------|------------|----------|
| **hnswlib** | base vectors, query vectors, M, ef_construction, ef | labels, distances, query_time | recall@k, QPS, build_time | 能在 synthetic-small 上跑出 summary.csv 和 detail.parquet |
| **RangeFilteredANN** | dataset, filter_width, method | results/*_results.csv | recall-QPS, filter_width-QPS, build_time, memory | 能解析其 9 列数据（修复表头问题），统一 Schema |
| **SIEVE** | dataset, algorithm, track | data_export.py CSV | recall-QPS, build_time, index_size | 能解析 recall/ap, qps, build, indexsize 字段 |
| **ANN-Benchmarks/VIBE** | dataset, algorithm definition | HDF5 / CSV | recall-QPS, p50/p95/p99 latency | 能读取 HDF5 attrs 并导出统一 CSV |
| **Dashboard** | normalized/all_results.csv | Streamlit 页面 | 总览、recall-QPS、filter敏感性、build/memory、单查询解释 | 能启动并展示真实数据图 |

### 扩展（加分项，不阻塞）

| 模块 | 作用 | 为何不放主线 |
|------|------|--------------|
| SWIRL | RL 索引选择 | 训练成本高，推理速度慢，解决的是数据库二级索引选择而非过滤向量检索 |
| index_selection_evaluation | 数据库索引选择 | 关系型数据库索引，与向量检索场景不同 |
| BALANCE | workload-aware 索引 | 同样偏向数据库索引管理 |

## 3. 数据集规划

### MVP 阶段（synthetic-small）

```yaml
synthetic-small:
  base: 10000 vectors x 128 dim, float32
  query: 1000 vectors x 128 dim, float32
  labels: 0-9999 int, 用于 window filter
  filter_widths: ["1%", "10%", "50%"]
  k: 10
  metric: l2
```

### 后续可选真实数据集

| 数据集 | 维度 | 规模 | 来源 |
|--------|------|------|------|
| SIFT | 128 | 1M | ANN-Benchmarks |
| GloVe | 100 | 1.2M | ANN-Benchmarks |
| random-filter-s | 100 | 小 | SIEVE 内置 |
| yfcc-10M | 192 | 10M | SIEVE / BigANN |

## 4. 实验流程

```
生成/准备数据
    ↓
配置实验 (configs/experiments.yaml)
    ↓
调度器调用各 adapter 运行实验
    ↓
保存原始结果 (results/raw/)
    ↓
Parser 解析为统一 Schema (results/normalized/)
    ↓
合并为 all_results.csv
    ↓
Streamlit Dashboard 可视化
    ↓
导出论文图表 (results/figures/)
```

## 5. 各项目已知问题与应对

### RangeFilteredANN
- **问题 1**：`run_our_method.py` 表头只写 6 列，实际数据写 9 列，导致 pandas 读取后三列无列名
  - **应对**：parser 中显式指定 `names=` 或读取后重命名 `Unnamed: 6/7/8`
- **问题 2**：README 写的参数 `--all`，源码实际是 `--all_methods`
  - **应对**：adapter 中使用正确参数 `--all_methods`
- **问题 3**：`method` 列混了算法名和参数（如 `optimized-postfiltering_1.000_2_10_1`）
  - **应对**：parser 中拆分为 `algorithm` + `params_json`
- **问题 4**：对抗数据集的 `filter_width` 为空字符串
  - **应对**：parser 中做容错，空值填 NULL

### SIEVE
- **问题 1**：`sieve.py` 的 `__str__` 永远返回 `"Sieve"`，parameters 列无区分度
  - **应对**：短期内 parser 中从其他信息推断参数；如需细化，后续修改 `__str__`
- **问题 2**：`mean_latency`, `distcomps`, `mean_ssd_ios` 默认恒为 0（未实现 `get_additional`）
  - **应对**：统一 Schema 中这些字段填 NULL 或 0，并在文档中标注

### ANN-Benchmarks / VIBE
- **问题 1**：HDF5 格式需要专门读取
  - **应对**：`parse_annbench_hdf5.py` 用 `h5py` 读取 attrs 和 datasets
- **问题 2**：VIBE 的 recall 存储在 HDF5 group 中，且 `export_results.py` 输出 Parquet
  - **应对**：支持读取 VIBE 的 `summary.parquet` 或直接从 HDF5 解析

### hnswlib
- **问题**：无内置 benchmark 输出格式
  - **应对**：自己写 adapter，封装 `add_items` + `knn_query` + `time.time()` 计时

## 6. 论文图表规划

| 图/表 | 内容 | 预计章节 |
|-------|------|----------|
| 图 1 | 系统架构图 | 第 3 章 |
| 图 2 | 统一实验流程图 | 第 3 章 |
| 图 3 | Recall-QPS 帕累托曲线 | 第 4 章 |
| 图 4 | Filter Width 对 QPS 的影响 | 第 4 章 |
| 图 5 | Build Time 对比 | 第 4 章 |
| 图 6 | Index Size / Memory 对比 | 第 4 章 |
| 图 7 | 单查询 Top-k 解释示例 | 第 4 章 |
| 图 8 | SIEVE 查询策略占比（如有 trace） | 第 4 章 |
| 图 9 | β-WST 静态结构或查询 trace（如有） | 第 4 章 |
| 表 1 | 集成项目与输出字段对应关系 | 第 3 章 |
| 表 2 | 实验数据集说明 | 第 4 章 |
| 表 3 | 算法参数设置 | 第 4 章 |
| 表 4 | 不同算法在 recall ≥ 0.9 下的最佳 QPS | 第 4 章 |
| 表 5 | 构建成本与内存成本对比 | 第 4 章 |
