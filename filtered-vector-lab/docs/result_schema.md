# 统一实验结果 Schema

## 设计原则

1. **兼容所有项目**：必须能容纳 hnswlib、RangeFilteredANN、SIEVE、ANN-Benchmarks/VIBE 的输出
2. **不创造无法得到的字段**：如果某项目不输出某字段，填 NULL，不虚构
3. **单位统一**：时间统一为秒或毫秒，大小统一为 KB
4. **JSON 化复杂参数**：`params_json` 用 JSON 字符串存储变长参数，避免列爆炸

## 字段定义

| 字段名 | 类型 | 是否必填 | 来源 | 说明 |
|--------|------|----------|------|------|
| `run_id` | STRING | 是 | 系统生成 | UUID，每次实验唯一 |
| `project` | STRING | 是 | 系统标记 | `hnswlib`, `RangeFilteredANN`, `SIEVE`, `ann-benchmarks`, `vibe` |
| `algorithm` | STRING | 是 | 各项目输出 | 算法名称，如 `hnsw`, `prefiltering`, `sieve`, `faiss` |
| `dataset` | STRING | 是 | 各项目输出 / 配置 | 数据集名称，如 `synthetic-small`, `sift-128-euclidean` |
| `track` | STRING | 否 | SIEVE / BigANN | 评测赛道，如 `filter`, `range` |
| `filter_type` | STRING | 否 | 推断 | 过滤类型：`none`, `range`, `predicate` |
| `filter_width` | STRING | 否 | RangeFilteredANN | 原始字符串，如 `2pow-16`, `1%`, `10%` |
| `selectivity` | FLOAT | 否 | 推断 / 配置 | 过滤选择性，如 0.01, 0.1, 0.5 |
| `k` | INT | 是 | 配置 / 输出 | top-k 值，如 10 |
| `params_json` | STRING | 是 | 解析生成 | JSON 字符串，存储所有算法参数 |
| `recall` | FLOAT | 是 | 各项目输出 | 平均 recall@k，范围 [0, 1] |
| `qps` | FLOAT | 是 | 各项目输出 | Queries Per Second |
| `avg_latency_ms` | FLOAT | 否 | 计算 | 平均单次查询延迟（毫秒） |
| `p50_latency_ms` | FLOAT | 否 | ANN-Benchmarks | P50 延迟（毫秒） |
| `p95_latency_ms` | FLOAT | 否 | ANN-Benchmarks | P95 延迟（毫秒） |
| `p99_latency_ms` | FLOAT | 否 | ANN-Benchmarks | P99 延迟（毫秒） |
| `build_time_s` | FLOAT | 否 | 各项目输出 | 索引构建时间（秒） |
| `index_size_kb` | FLOAT | 否 | 各项目输出 | 索引大小（KB） |
| `memory_kb` | FLOAT | 否 | RangeFilteredANN | 峰值内存增量（KB） |
| `distcomps` | FLOAT | 否 | SIEVE / ANN-Benchmarks | 距离计算次数 |
| `threads` | INT | 否 | 配置 | 实验线程数 |
| `raw_result_path` | STRING | 是 | 系统记录 | 原始结果文件绝对路径 |
| `created_at` | DATETIME | 是 | 系统生成 | 记录创建时间 |

## 扩展字段（SWIRL / 数据库索引选择）

以下字段仅在需要集成 SWIRL 或 index_selection_evaluation 时使用：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `workload_cost` | FLOAT | workload 总成本 |
| `algorithm_runtime_s` | FLOAT | 算法运行时间 |
| `cost_requests` | INT | 成本模型请求次数 |
| `cache_hits` | INT | 缓存命中次数 |
| `budget_mb` | FLOAT | 索引预算（MB） |
| `selected_indexes_json` | STRING | 选中的索引集合 JSON |

## Pandas dtype

```python
import pandas as pd

SCHEMA_DTYPE = {
    "run_id": "string",
    "project": "string",
    "algorithm": "string",
    "dataset": "string",
    "track": "string",
    "filter_type": "string",
    "filter_width": "string",
    "selectivity": "Float64",
    "k": "Int64",
    "params_json": "string",
    "recall": "Float64",
    "qps": "Float64",
    "avg_latency_ms": "Float64",
    "p50_latency_ms": "Float64",
    "p95_latency_ms": "Float64",
    "p99_latency_ms": "Float64",
    "build_time_s": "Float64",
    "index_size_kb": "Float64",
    "memory_kb": "Float64",
    "distcomps": "Float64",
    "threads": "Int64",
    "raw_result_path": "string",
    "created_at": "datetime64[ns]",
}
```

## SQLite 建表 SQL

```sql
CREATE TABLE IF NOT EXISTS experiment_results (
    run_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    dataset TEXT NOT NULL,
    track TEXT,
    filter_type TEXT,
    filter_width TEXT,
    selectivity REAL,
    k INTEGER NOT NULL,
    params_json TEXT NOT NULL,
    recall REAL,
    qps REAL,
    avg_latency_ms REAL,
    p50_latency_ms REAL,
    p95_latency_ms REAL,
    p99_latency_ms REAL,
    build_time_s REAL,
    index_size_kb REAL,
    memory_kb REAL,
    distcomps REAL,
    threads INTEGER,
    raw_result_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_project ON experiment_results(project);
CREATE INDEX idx_dataset ON experiment_results(dataset);
CREATE INDEX idx_algorithm ON experiment_results(algorithm);
CREATE INDEX idx_filter_type ON experiment_results(filter_type);
```

## 各项目字段映射

### hnswlib（自定义 adapter 输出）

| 统一字段 | hnswlib 来源 |
|----------|-------------|
| `project` | 固定 `"hnswlib"` |
| `algorithm` | 固定 `"hnsw"` |
| `k` | 配置参数 |
| `params_json` | `{"M": 16, "ef_construction": 200, "ef": 50, "space": "l2"}` |
| `recall` | 手动计算：与 ground truth 交集 / k |
| `qps` | `num_queries / total_query_time` |
| `avg_latency_ms` | `total_query_time / num_queries * 1000` |
| `build_time_s` | `time.time()` 围绕 `add_items` |
| `index_size_kb` | 暂无法直接得到，填 NULL |
| `memory_kb` | 暂无法直接得到，填 NULL |
| `threads` | 配置参数 |

### RangeFilteredANN

| 统一字段 | RangeFilteredANN 来源 | 备注 |
|----------|----------------------|------|
| `project` | 固定 `"RangeFilteredANN"` |
| `algorithm` | `method` 列第一个下划线前前缀 | 如 `vamana-tree`, `prefiltering` |
| `dataset` | 从文件名解析 | `results/<dataset>_results.csv` |
| `filter_type` | 固定 `"range"` |
| `filter_width` | `filter_width` 列 | 如 `2pow-16` |
| `k` | 固定 `10`（源码硬编码 TOP_K） | 后续如需支持其他 k，需改源码 |
| `params_json` | `method` 列下划线后参数 + `branching_factor` | JSON 化 |
| `recall` | `recall` 列 |
| `qps` | `qps` 列 |
| `avg_latency_ms` | `average_time * 1000` |
| `build_time_s` | 第 7 列（无名） | 表头缺失，需按位置读取 |
| `index_size_kb` | 暂无法直接得到，填 NULL |
| `memory_kb` | 第 9 列（无名） | 表头缺失，需按位置读取 |
| `threads` | `threads` 列 |

### SIEVE

| 统一字段 | SIEVE 来源 | 备注 |
|----------|-----------|------|
| `project` | 固定 `"SIEVE"` |
| `algorithm` | `algorithm` 列 |
| `dataset` | `dataset` 列 |
| `track` | `track` 列 |
| `filter_type` | 根据 track 推断：`filter` -> `predicate` | |
| `k` | `count` 列 |
| `params_json` | `parameters` 列 | SIEVE 目前永远为 `"Sieve"`，需后续改进 |
| `recall` | `recall/ap` 列 |
| `qps` | `qps` 列 |
| `avg_latency_ms` | `mean_latency` 列 | 默认恒为 0，需后续改进 |
| `build_time_s` | `build` 列 |
| `index_size_kb` | `indexsize` 列 |
| `distcomps` | `distcomps` 列 | 默认恒为 0 |
| `threads` | 暂无法直接得到，填 NULL |

### ANN-Benchmarks

| 统一字段 | ANN-Benchmarks 来源 | 备注 |
|----------|--------------------|------|
| `project` | 固定 `"ann-benchmarks"` |
| `algorithm` | `algorithm` 列 |
| `dataset` | `dataset` 列 |
| `k` | `count` 列 |
| `params_json` | `parameters` 列 |
| `recall` | `k-nn` 列 | data_export.py 导出后列名为 `k-nn` |
| `qps` | `qps` 列 |
| `avg_latency_ms` | `best_search_time * 1000` | 从 HDF5 attrs 计算 |
| `p50_latency_ms` | `p50` 列 * 1000 | 如存在 |
| `p95_latency_ms` | `p95` 列 * 1000 | 如存在 |
| `p99_latency_ms` | `p99` 列 * 1000 | 如存在 |
| `build_time_s` | `build` 列 | |
| `index_size_kb` | `indexsize` 列 | |
| `distcomps` | `distcomps` 列 | 如存在 |
| `threads` | 从 `params_json` 推断 | |

### VIBE

| 统一字段 | VIBE 来源 | 备注 |
|----------|----------|------|
| `project` | 固定 `"vibe"` |
| `algorithm` | `algorithm` 列 / HDF5 `algo` attr |
| `dataset` | `dataset` 列 / HDF5 `dataset` attr |
| `k` | `k` 列 / HDF5 `count` attr |
| `params_json` | `params` 列 |
| `recall` | `recall` 列 / HDF5 `recalls` mean |
| `qps` | `qps` 列 / `1 / best_search_time` |
| `build_time_s` | `build_time` 列 / HDF5 attr |
| `index_size_kb` | `index_size` 列 / HDF5 attr |

## 缺失字段处理规则

1. **某项目不输出的字段**：统一填 `NULL`（pandas 中为 `pd.NA`，SQLite 中为 `NULL`）
2. **默认值为 0 但无实际意义的字段**（如 SIEVE 的 `mean_latency`）：在统一 Schema 中填 `NULL`，并加 `note` 列说明原因（可选）
3. **可从其他字段推导的字段**：在 parser 中计算，如 `avg_latency_ms = average_time * 1000`
4. **需要从文件名解析的字段**：如 RangeFilteredANN 的 `dataset`，在 parser 中通过正则从路径提取
