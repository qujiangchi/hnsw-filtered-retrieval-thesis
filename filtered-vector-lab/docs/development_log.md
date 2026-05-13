# 开发日志

## 2026-05-12：阶段 0-1 完成，阶段 2 进行中

### 已完成

1. **项目骨架建立**
   - 创建 `filtered-vector-lab/` 目录结构
   - 编写 `README.md`、`docs/experiment_design.md`、`docs/result_schema.md`
   - 编写 `configs/datasets.yaml`、`configs/algorithms.yaml`、`configs/experiments.yaml`

2. **第三方项目盘点（阶段 2）**
   - **RangeFilteredANN**：
     - 最小实验命令：`python run_our_method.py --all_methods --dataset sift-128-euclidean --threads 16`
     - 结果文件：`results/<dataset>_results.csv`
     - 关键发现：表头只有6列（filter_width,method,recall,average_time,qps,threads），但实际写入9列数据，后3列（build_time, branching_factor, memory）无列名
     - `method` 列混了算法名和参数，如 `optimized-postfiltering_1.000_2_10_1`
     - `k` 硬编码为 10（TOP_K = 10）
     - 对抗数据集 filter_width 为空字符串
   - **SIEVE**：
     - 结果导出：`python data_export.py --output res.csv`
     - 字段：algorithm, parameters, dataset, count, qps, distcomps, build, indexsize, mean_ssd_ios, mean_latency, track, recall/ap
     - 关键发现：`sieve.py` 的 `__str__` 永远返回 `"Sieve"`，parameters 列无区分度
     - `mean_latency`、`distcomps`、`mean_ssd_ios` 默认恒为 0（未实现 `get_additional`）
   - **ANN-Benchmarks / VIBE**：
     - 结果格式：HDF5，每个实验一个文件
     - attrs 包含：algo, build_time, index_size, best_search_time, count 等
     - datasets：times, neighbors, distances
     - 已有 `data_export.py` 可导出 CSV，包含 k-nn, qps, build, indexsize, p50, p95, p99 等
     - VIBE 额外支持 `export_results.py` 导出 Parquet（summary + detail）
   - **hnswlib**：
     - 无内置 benchmark 输出，需在 Python 层手动封装
     - 关键 API：`hnswlib.Index(space, dim)` -> `init_index` -> `add_items` -> `knn_query`
     - 参数：M, ef_construction, ef, space, num_threads
     - recall 需手动与 ground truth 对比计算

3. **核心代码编写**
   - `scripts/generate_synthetic.py`：生成 synthetic-small 数据集（base, query, labels, window_queries, groundtruth）
   - `src/metrics/recall.py`：recall@k、per-query recall、阈值 recall
   - `src/storage/result_store.py`：统一 Schema、empty df、normalize、merge、add row
   - `src/adapters/hnswlib_adapter.py`：封装 hnswlib 构建+查询+计时，输出 summary.csv + detail.parquet
   - `src/parsers/parse_range_results.py`：处理 RangeFilteredANN 表头缺失问题，统一 Schema
   - `src/parsers/parse_sieve_results.py`：解析 SIEVE CSV，将无意义的 0 转为 NULL
   - `src/parsers/parse_hnswlib_results.py`：读取 hnswlib summary 验证格式
   - `dashboard/app.py`：Streamlit 5 页面（总览、Recall-QPS、过滤敏感性、构建与资源、单查询解释）
   - `scripts/run_demo.sh`：一键演示脚本

### 已完成验证（2026-05-12）

- [x] 安装依赖：`numpy`, `pandas`, `plotly`, `streamlit`, `hnswlib`
- [x] 运行 `generate_synthetic.py`：生成 10K base + 1K query + labels + window_queries + groundtruth
- [x] 运行 `hnswlib_adapter.py`：4 组参数实验全部成功
  - M=16, ef_c=100, ef=20: recall=0.1888, QPS=12331, build=3.22s
  - M=16, ef_c=100, ef=50: recall=0.2079, QPS=5666, build=3.06s
  - M=16, ef_c=200, ef=50: recall=0.2074, QPS=5385, build=5.39s
  - M=32, ef_c=200, ef=100: recall=0.2095, QPS=1594, build=7.27s
- [x] `merge_all_normalized()` 成功合并为 `all_results.csv`（4 rows）

### 已完成验证（2026-05-12 续）

- [x] 写 `parse_annbench_hdf5.py`：支持直接读取 HDF5 attrs/datasets 和读取 data_export.py CSV
- [x] SIEVE parser 验证：用 `biganntest/neurips23/filter/res_public_queries_AzureD8lds_v5.csv` 真实数据，106 行全部正确解析
- [x] RangeFilteredANN parser 验证：用 mock CSV 模拟 9 列数据（修复表头缺失），12 行全部正确解析
- [x] 合并结果：`all_results.csv` 共 122 行（hnswlib 4 + RangeFilteredANN 12 + SIEVE 106）

### Dashboard 前端完成（8 页面）

按原型图实现了全部 8 个页面：
1. **总览**：4 指标卡 + Recall-QPS 散点图 + 构建时间柱状图 + 算法饼图 + 性能汇总表
2. **Recall-QPS**：帕累托前沿散点图 + 配置详情面板 + 阈值-QPS 折线图 + Top5 表格
3. **过滤敏感性**：QPS 热力图 + Filter Width 折线图 + Selectivity-Recall 图 + 3 条关键洞察
4. **构建与内存**：4 指标卡 + 构建/内存对比柱状图 + 三维气泡图 + 资源成本汇总表
5. **单查询解释**：查询摘要 + recall 条形图 + Top-k 对比表格 + 命中数统计 + Ground Truth 展示
6. **β-WST 路径**：树结构 Treemap + 访问日志表格 + 耗时占比饼图 + 层级访问柱状图
7. **SIEVE 策略**：索引 DAG Treemap + 策略占比条形图/饼图 + 配置指标卡 + 查询级 Trace 表格
8. **实验管理**：运行记录表格（带状态颜色）+ 进度条/日志预览 + 每日运行柱状图 + 项目分布饼图

### 辅助模拟数据生成

- `scripts/generate_experiment_runs.py`：128 条实验运行记录（成功/失败/运行中）
- `scripts/generate_mock_traces.py`：β-WST trace（6 节点）、树结构、SIEVE trace（25,000 条）、策略汇总

### 安装阻塞解决

- **RangeFilteredANN**：✅ 已解决
  - 修改 `CMakeLists.txt` 第 5-6 行：将 `find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)` 改为老式 `find_package(PythonInterp 3 REQUIRED)` + `find_package(PythonLibs 3 REQUIRED)`
  - 手动指定 `pybind11_DIR`（pip 安装的 pybind11 cmake 配置路径）
  - 编译成功，.so 已复制到 site-packages，`import window_ann` 验证通过
  - 运行 `src/adapters/range_filtered_ann_adapter.py`：prefiltering 在 synthetic-small 上成功跑出结果（recall=0.051, QPS=5,569, build=0.02s）

- **SIEVE**：hnswtest 编译失败（g++ 编译错误），可能需安装 ACORN 或额外编译依赖。暂用已有 CSV 结果验证 parser。

### 待完成

- [ ] 修复/安装 SIEVE 真实环境
- [ ] 写统一实验调度器 `run_experiment.py`
- [ ] 补充 pytest 单元测试

### 遇到的问题与解决

1. **RangeFilteredANN CSV 表头与数据列数不一致**
   - 解决：parser 中显式指定 `names=` 为9列，修复读取问题
2. **SIEVE parameters 无区分度**
   - 解决：parser 中将 parameters 放入 `params_json`，并标注需要后续修改 `__str__`
3. **hnswlib 无内置输出格式**
   - 解决：自己写 adapter，封装计时和 recall 计算
