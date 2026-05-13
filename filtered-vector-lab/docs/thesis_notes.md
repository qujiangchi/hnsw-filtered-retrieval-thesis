# 论文笔记

## 题目
基于 HNSW 的过滤向量检索优化方法研究与实现

## 核心问题（3句话）
1. **问题**：高维向量近似最近邻（ANN）检索在带过滤条件（数值区间、复杂谓词）时，现有方法在 recall、QPS、构建成本之间存在显著 trade-off，缺乏统一实验平台系统对比。
2. **集成算法**：以 HNSW 为基座，集成 RangeFilteredANN（窗口过滤）、SIEVE（谓词过滤与索引集合选择）、ANN-Benchmarks（通用评测框架）。
3. **可视化证明**：通过 recall-QPS 帕累托曲线、filter width / selectivity 敏感性图、build time / memory 对比图、单查询 top-k 解释页，证明不同过滤场景下各算法的优劣。

## 系统架构

```
┌─────────────────────────────────────────────┐
│          Filtered Vector Lab                │
│  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Datasets   │  │  Experiment Runner  │  │
│  │ synthetic/  │  │  (config-driven)    │  │
│  │ real-world  │  └──────────┬──────────┘  │
│  └──────┬──────┘             │             │
│         │                    ▼             │
│         │         ┌─────────────────────┐  │
│         │         │  Adapters           │  │
│         │         │  hnswlib / Range /  │  │
│         │         │  SIEVE / ANN-Bench  │  │
│         │         └──────────┬──────────┘  │
│         │                    │             │
│         │         ┌──────────┴──────────┐  │
│         │         │  Parsers            │  │
│         │         │  -> Unified Schema  │  │
│         │         └──────────┬──────────┘  │
│         │                    │             │
│         ▼                    ▼             │
│  ┌─────────────────────────────────────┐  │
│  │  Streamlit Dashboard                │  │
│  │  - Overview / Recall-QPS / Filter   │  │
│  │  - Build&Memory / Query Explain     │  │
│  └─────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## 实验设计

### 数据集
| 名称 | 规模 | 维度 | 用途 |
|------|------|------|------|
| synthetic-small | 10K | 128 | MVP 验证 |
| synthetic-medium | 100K | 128 | 扩展实验 |
| SIFT-1M | 1M | 128 | 标准 benchmark |

### 评测指标
| 指标 | 说明 | 优先级 |
|------|------|--------|
| Recall@k | 召回率 | P0 |
| QPS | 每秒查询数 | P0 |
| Avg Latency | 平均延迟 | P1 |
| P50/P95/P99 Latency | 延迟分位数 | P1 |
| Build Time | 索引构建时间 | P1 |
| Index Size | 索引磁盘大小 | P1 |
| Memory | 峰值内存 | P2 |
| Distcomps | 距离计算次数 | P2 |

## 图表规划

### 图
1. 系统架构图（第3章）
2. 统一实验流程图（第3章）
3. Recall-QPS 帕累托曲线（第4章）
4. Filter Width 对 QPS 的影响（第4章）
5. Build Time 对比（第4章）
6. Index Size / Memory 对比（第4章）
7. 单查询 Top-k 解释示例（第4章）
8. SIEVE 查询策略占比（如有 trace，第4章）
9. β-WST 静态结构或查询 trace（如有，第4章）

### 表
1. 集成项目与输出字段对应关系（第3章）
2. 实验数据集说明（第4章）
3. 算法参数设置（第4章）
4. 不同算法在 recall ≥ 0.9 下的最佳 QPS（第4章）
5. 构建成本与内存成本对比（第4章）

## 写作计划

| 周次 | 任务 |
|------|------|
| 1-2 | 建立项目骨架，跑通最小实验 |
| 3-4 | 集成 RangeFilteredANN 和 SIEVE，写解析器 |
| 5-6 | 完善 Dashboard，补充 trace 和单查询解释 |
| 7 | 大规模实验，整理图表 |
| 8 | 写论文，准备答辩 |

## 关键术语对照

| 中文 | 英文 |
|------|------|
| 过滤向量检索 | Filtered Vector Search / Filtered ANN |
| 近似最近邻 | Approximate Nearest Neighbor (ANN) |
| 窗口过滤 | Window Search / Range Filter |
| 谓词过滤 | Predicate Filter |
| 预过滤 | Pre-filtering |
| 后过滤 | Post-filtering |
| 索引集合选择 | Index Set Selection |
| 召回率 | Recall |
| 查询吞吐量 | QPS (Queries Per Second) |
| 构建时间 | Build Time |
