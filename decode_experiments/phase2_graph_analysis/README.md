# Phase 2 Revised: Strict & Fair Graph Analysis

## 目标

将 Phase 2 从“手工结论展示”修正为“可审计的图成本分析实验”，用于公平比较 `prefill` 与 `decode` 在相同图结构下的计算/访存特性差异。

## 关键修正

1. 不再使用硬编码的端到端时间（如固定 `0.376ms`）和固定 launch 开销（如 `15us`）。
2. 不再给出无数据支撑的固定 speedup 结论（如“3x-5x”）。
3. 明确分析范围（scope）：
   - `attention_only`：对应 kernel 级 H2O attention 路径。
   - `attention_plus_cache`：在 attention 基础上追加 model 级 cache 选择路径（topk/sort/gather/cat）。
4. 同一 scope 下，prefill 与 decode 使用完全一致的算子 schema 与边界策略，保证比较公平。
5. 输出完整假设与公式结果，方便复核。

## 运行方式

```bash
cd /data2/ldz/FlashTensor-AE
python decode_experiments/phase2_graph_analysis/extract_and_analyze_graph.py \
  --scope both \
  --q_len_prefill 4096 \
  --q_len_decode 1 \
  --kv_len 4096 \
  --dtype float16 \
  --tag strict_fair
```

可选参数：

- `--scope {attention_only,attention_plus_cache,both}`
- `--cache_budget`（默认 `512`）
- `--compute_intensity_threshold`（默认 `8.0`）
- `--print_ops`（打印逐算子估计）

## 输出

输出到：

`decode_experiments/phase2_graph_analysis/results/*.json`

JSON 包含：

1. 配置与公平性规则（`methodology.fairness_rules`）。
2. 每个 scope 的 prefill/decode 总体统计：
   - `estimated_total_flops`
   - `estimated_total_rw_bytes`
   - `global_arithmetic_intensity`
3. 三种 kernel 边界策略下的 kernel 计数与覆盖校验：
   - `reduce_barrier`
   - `reduce_or_irregular_barrier`
   - `strict_decode_barrier`

## 注意

1. 该阶段是图级成本分析，不是内核微基准。
2. `topk/sort` 的 FLOPs 为启发式代理值，已在结果里显式说明。
3. 若要得出真实性能结论，需要在 Phase 3/Phase 4 做 kernel 实测闭环。
