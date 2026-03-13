# Phase 2 Analysis (Strict & Fair)

> 数据来源：`results/strict_fair_test_graph_analysis_20260306_224220.json`  
> 代码版本：`phase2_graph_analysis/extract_and_analyze_graph.py`（revised）  
> 生成时间：2026-03-06 22:42（文件名时间戳）

## 1. 实验目标

在统一且可审计的口径下，对比 `prefill` 与 `decode` 的图级成本差异，回答：

1. decode 是否更偏 memory-bound（从图成本角度）？
2. cache 选择链路（topk/sort/gather）是否进一步强化 decode 的访存压力？
3. 在不同 kernel 切分策略下，prefill/decode 的结构性差异是什么？

## 2. 严格性与公平性修正

相对旧版 phase2，本版修正如下：

1. 删除硬编码时间与速度结论：不再使用固定 `0.376ms`、`15us`、`3x-5x` 等假设。
2. 同口径对照：prefill/decode 在同一 scope 下使用完全一致的算子 schema 和边界策略。
3. 透明估算：每个算子输出 `estimated_flops`、`estimated_read_bytes`、`estimated_write_bytes`、`AI`。
4. 多策略边界：统一报告三种策略：
   - `reduce_barrier`
   - `reduce_or_irregular_barrier`
   - `strict_decode_barrier`
5. 明确范围：
   - `attention_only`：kernel 级 H2O attention 主路径
   - `attention_plus_cache`：在 attention 基础上叠加 topk/sort/gather/cat 的 cache 选择链路

## 3. 实验配置

| 参数 | 值 |
|---|---:|
| `batch_size` | 1 |
| `q_len_prefill` | 4096 |
| `q_len_decode` | 1 |
| `kv_len` | 4096 |
| `head_num` | 32 |
| `kv_head_num` | 32 |
| `head_dim` | 128 |
| `cache_budget` | 512 |
| `dtype` | float16 |
| `compute_intensity_threshold` | 8.0 FLOPs/Byte |

## 4. 结果

### 4.1 Scope = `attention_only`

| 指标 | Prefill | Decode | Ratio (Prefill/Decode) |
|---|---:|---:|---:|
| Total Ops | 17 | 17 | 1.00x |
| Estimated FLOPs | 2.792e11 | 6.803e7 | 4103.89x |
| Estimated RW Bytes (GB) | 18.5630 | 0.1925 | 96.44x |
| Global AI (FLOPs/Byte) | 14.0072 | 0.3292 | 42.55x |

Kernel count（Prefill / Decode）：

1. `reduce_barrier`: 5 / 5
2. `reduce_or_irregular_barrier`: 5 / 5
3. `strict_decode_barrier`: 5 / 5

### 4.2 Scope = `attention_plus_cache`

| 指标 | Prefill | Decode | Ratio (Prefill/Decode) |
|---|---:|---:|---:|
| Total Ops | 24 | 24 | 1.00x |
| Estimated FLOPs | 2.792e11 | 6.936e7 | 4025.38x |
| Estimated RW Bytes (GB) | 18.6263 | 0.2558 | 72.81x |
| Global AI (FLOPs/Byte) | 13.9596 | 0.2525 | 55.29x |

Kernel count（Prefill / Decode）：

1. `reduce_barrier`: 7 / 7
2. `reduce_or_irregular_barrier`: 11 / 11
3. `strict_decode_barrier`: 11 / 11

## 5. 结论（客观事实 vs 分析判断）

### 5.1 客观事实（可观测）

1. 两个 scope 下，prefill 与 decode 的算子数是相同的（17/24），结构上可公平对照。
2. decode 的全局算术强度明显低于 prefill：
   - `attention_only`: `0.3292` vs `14.0072`
   - `attention_plus_cache`: `0.2525` vs `13.9596`
3. 引入 cache 选择链路后，decode 的 AI 进一步下降（`0.3292 -> 0.2525`），且 kernel 边界在保守策略下从 7 增加到 11。

### 5.2 分析判断（解释/结论）

1. 在图级成本层面，decode 比 prefill 更强烈地表现为 memory-leaning（低 AI），与 phase1 的带宽上升趋势一致。
2. topk/sort/gather 相关链路会强化 decode 的不规则访存属性，增加“保守切分策略”下的碎片化风险。
3. 仅依赖“跨 reduce 的激进融合”并不能自动保证公平收益，仍需在 phase3 用真实 kernel 做闭环验证。

## 6. 与 Phase 1 的关系

1. Phase 1（实测）给出运行现象：decode 带宽利用率随长度上升，TFLOPS 低。
2. Phase 2（图分析）给出结构解释：decode 全局 AI 低，且 cache 选择链路进一步拉低 AI、提高潜在切分复杂度。
3. 两阶段结论相互支撑：`现象（Phase1） -> 结构原因（Phase2）`。

## 7. 局限与下一步

1. 本阶段仍是图成本估算，不是端到端时延测量。
2. `topk/sort` FLOPs 使用启发式代理，适合比较趋势，不等价于硬件级真实执行成本。
3. 下一步建议：
   - 在 phase3 以相同 shape 组合进行 kernel 实测；
   - 输出 `bytes/token`、HBM 利用率、kernel 启动占比，验证 phase2 推断。
