# Decode Experiments 简要汇报

## 1. 汇报目标

本汇报用于回答两个问题：

1. `decode_experiments` 到目前为止做了什么、结论是什么。
2. FlashTensor 在 decode 阶段哪些可行，哪些暂时不可行。

---

## 2. 实验路线与核心结论

### Phase 1：Baseline 实测（prefill vs decode）

做了什么：

- 对 `attn/h2o/roco` 在 `N={2048,4096,8192}` 下做 prefill 与 decode 对比。
- 统计时间、TFLOPS、带宽利用率、显存峰值增量。

关键结果：

- `N=4096` 时，prefill 约 `23.9~33.5 ms`，decode 约 `0.21~0.23 ms`，时间比约 `114x~143x`。
- decode 带宽利用率随长度增长明显：约 `15%~18% (N=2048)` 上升到 `45%~52% (N=8192)`。
- prefill 在 `N=8192` 全部 OOM（3090 24GB）。

结论：

- prefill 主要受计算/容量压力影响。
- decode 更明显是 memory-bound，优化重点应放在访存路径。

### Phase 2：图级成本分析（strict & fair）

做了什么：

- 在统一 schema 和公平边界下，对比 prefill/decode 的 FLOPs、RW Bytes、AI（算术强度）。
- 分析 `attention_only` 与 `attention_plus_cache` 两个 scope。

关键结果：

- `attention_only`：AI `14.0072 (prefill)` vs `0.3292 (decode)`。
- `attention_plus_cache`：decode AI 进一步降到 `0.2525`。
- 加入 cache 选择链（topk/sort/gather）后，保守策略下 kernel 边界数从 `7` 增到 `11`（decode 更碎片化）。

结论：

- 图级证据与 Phase1 一致：decode 的低 AI 与访存主导特征非常明确。
- cache 选择链会进一步强化 decode 的不规则访存问题。

### Phase 3：手写 fused decode kernel 验证

做了什么：

- 实现 Triton fused kernel（attention + score path）并与 baseline 对比。

关键结果（speedup，>1 才算加速）：

- `kv=2048`: `0.77x`
- `kv=4096`: `0.40x`
- `kv=8192`: `0.32x`

结论：

- 当前 fused 方案整体慢于 baseline。
- 说明 decode 场景下“只做融合”不一定有效，错误融合甚至会放大 IO 负担。

---

## 3. FlashTensor 用于 Decode：可行与不可行

### 3.1 可行（有条件成立）

1. **前端编译能力可迁移**：IR 翻译、图规约、基础分区能力可复用。
2. **受约束 decode 可做**：长度桶化（bucket）、相对稳定 batch、连续 KV 下，仍可通过多版本 + dispatch 获取收益。
3. **目标应改为降 bytes/token**：以 IO 优化为主，而不是单纯追求 FLOPS。

### 3.2 不可行（当前不能直接套用）

1. **在线 decode 全动态场景**（continuous batching + 动态 `B/T`）不适合直接套用静态假设。
2. **PagedAttention 不规则访存**会削弱静态 coalescing/tiling 推断，导致优化更保守、收益不稳定。
3. **仅靠激进融合不可行**：在 memory-bound 路径，融合不当会导致更慢（Phase3 已验证）。

---

## 4. 最终结论（一句话）

FlashTensor 对 decode 不是“完全不可用”，而是“**部分可用且强依赖约束条件**”：  
在受控场景可迁移；在真实在线 decode（动态 + paged KV）下，需要编译策略、运行时调度与 KV 管理协同改造后才有稳定收益。
