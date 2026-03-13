# FlashTensor 迁移到 Decode 阶段：终版问题分析

> 目标：给出可执行结论，回答三个核心问题：  
> 1) FlashTensor 在 decode 阶段哪些部分可以适用；  
> 2) 哪些部分不好适用；  
> 3) 遇到的关键困难是什么。  
> 写法约定：每节都区分 **客观事实（可观测）** 与 **分析判断（解释/结论）**。

## 0. 最终结论（先看这一节）

### 客观事实

1. 现有 FlashTensor 编译链（`compile.py -> fission -> Connected partition -> optimize/profile/codegen`）在仓库内已被验证可用于规则张量图优化。
2. `decode_experiments/phase1_baseline` 的实测显示：decode 随上下文变长，带宽利用率明显上升，而 TFLOPS 明显低于 prefill。
3. 在线 decode 的真实约束通常包含动态 batch、动态长度、PagedAttention（分页 KV）。

### 分析判断

1. FlashTensor **在 decode 阶段“部分适用”**，但需要约束条件（桶化长度、有限 batch 策略、连续 KV 或弱分页）。
2. 不适用的核心不是“算子不能表示”，而是“静态优化假设与在线 decode 的动态访存现实冲突”。
3. 迁移路线应分阶段推进：先做受约束 decode，再扩展到 paged-KV 感知优化与运行时协同。

---

## 1. 推理链路与动态性来源（为什么 decode 更难）

### 1.1 客观事实

1. Prefill：一次处理整段 prompt，单次调用里 `q_len/kv_len` 固定。
2. Decode：逐 token 执行，典型 `q_len=1`，`kv_len` 随 step 增长。
3. 在线服务常用 continuous batching，请求会动态加入/退出，导致 `B` 与有效 `kv_len` 持续变化。
4. 若使用 PagedAttention，`KV[t]` 访问通常变成“查表 + 间接寻址”。

### 1.2 分析判断

1. “prefill 静态”只在单次调用成立，不等于线上全局静态。
2. decode 的难点是三重动态叠加：`B` 动态、`T` 动态、KV 地址布局动态。
3. 编译优化越依赖静态循环边界与连续访存，在线 decode 就越容易触发退化路径。

---

## 2. 证据基线（来自本仓库 Phase1）

### 2.1 客观事实

1. 数据来源：`decode_experiments/phase1_baseline/ANALYSIS.md`（RTX 3090, float16）。
2. 在 `N=4096` 时：
   - prefill 约 `23.9~33.5 ms`（attn/h2o/roco）
   - decode 约 `0.21~0.23 ms`
3. decode 带宽利用率（lower bound）随长度上升明显：
   - `N=2048` 约 `14.67%~17.82%`
   - `N=8192` 约 `45.09%~52.40%`
4. prefill 在 `N=8192` 发生 OOM（24GB 3090）。

### 2.2 分析判断

1. decode 单步计算量小、访存占比高，呈现 memory-bound 趋势。
2. prefill 更容易吃到算子融合与计算优化红利；decode 更依赖减少 bytes/token 与提升访存效率。
3. 因此迁移 FlashTensor 到 decode 时，不能只追求“更多融合”，要优先面向访存路径设计。

---

## 三、当前方法（FlashTensor）是否能适用？——严格回答三个核心问题

### 3.1 问题 1：自回归动态形状 vs 编译器静态假设

> 待回答：  
> “动态形状是阻碍 -> 方法只能用于静态形状 -> prefill 阶段形状是静态。  
> prefill 阶段所有参数都是静态的吗？”

#### 3.1.1 客观事实

1. 模型结构参数（`n_heads/head_dim/hidden_size/layer_num`）在推理时固定不变。
2. 单次 prefill 调用里，当前请求的 `q_len=kv_len=S` 在该次执行期间固定。
3. 在线服务中，不同请求的 prompt 长度 `S` 分布离散且动态变化，batch `B` 也会随 continuous batching 动态变化。
4. decode 中 `q_len` 常为 1，但 `kv_len=T` 会随 step 增长，并与请求进出共同造成 shape 持续变化。

#### 3.1.2 严格论证

1. 命题 A（错误命题）：“prefill 所有参数静态”。
2. 前提 A1：若“所有参数静态”成立，则需要对任意请求都满足固定 `B,S,T`。
3. 前提 A2：在线服务客观上存在可观测的请求长度波动和 batch 波动（A1 不满足）。
4. 推导 A3：命题 A 在“系统全局视角”不成立；仅在“单次调用局部视角”近似成立。
5. 推导 A4：FlashTensor 不能直接假设 decode 静态化，而应采用“离散静态化”（bucket + 多版本 + dispatch）。

#### 3.1.3 结论（针对问题 1）

1. **prefill 不是“所有参数静态”**：仅模型超参静态、单次调用长度局部固定。
2. 线上整体依然是动态 shape 问题，decode 动态性更强、更频繁。

### 3.2 问题 2：算力优化（Fusion）vs 访存瓶颈（IO）

> 待回答：  
> “这里的 IO 是从 storage 到 memory，还是从 memory 到 register/shared memory？”

#### 3.2.1 客观事实

1. LLM 推理热路径中，通常不包含高频磁盘读写；权重已驻留 GPU 后，主开销在设备内存层级搬运。
2. decode 每步需读取历史 KV cache，随着 `kv_len` 增长，HBM 读流量增长显著。
3. Phase1 实验显示 decode 带宽利用率随长度显著上升（`~15%-18%` 到 `~45%-52%`），而 TFLOPS 仍低。

#### 3.2.2 严格论证

1. 命题 B：“这里 IO 主要是 storage IO（磁盘到内存）”。
2. 前提 B1：若命题 B 成立，应观测到热路径被磁盘吞吐/延迟主导。
3. 前提 B2：本实验与典型在线推理中，瓶颈现象与 GPU 带宽趋势一致，而非磁盘访问趋势（B1 不满足）。
4. 推导 B3：这里的 IO 应解释为 GPU 内存层级 IO，核心是 `HBM <-> SM(register/shared/L2)`。
5. 推导 B4：decode 优化优先级是减少 bytes/token、提升访存连续性与 cache 复用，而不是单纯堆 FLOPS。

#### 3.2.3 结论（针对问题 2）

1. 文中 IO 指的是**GPU 内部访存 IO**，不是磁盘 storage IO。
2. decode 主要受 memory-bandwidth 和访存效率约束，fusion 必须以“降访存”为目标才有效。

### 3.3 问题 3：连续内存假设 vs PagedAttention（离散内存）

> 待回答：  
> “为什么内存不连续会导致编译器无法解析？”

#### 3.3.1 客观事实

1. 连续 KV 场景下，地址可写成仿射形式（`base + t * stride`），便于编译期做静态推断。
2. PagedAttention 下，访问 `KV[t]` 需要先查 page table，再做间接寻址，线程内地址可能离散。
3. 静态优化（coalescing 向量化、预取、tile 复用、别名分析）普遍依赖规则访存模式。

#### 3.3.2 严格论证

1. 命题 C（过度表述）：“内存不连续会导致编译器无法解析”。
2. 前提 C1：编译器仍可表达不连续访问（IR 可表示 gather/indirect load），因此“无法解析”并不准确。
3. 前提 C2：但在不连续访问下，编译器难以稳定证明连续性和固定 stride，导致高质量静态优化条件不满足。
4. 推导 C3：正确说法应为“可编译但更保守”，性能更依赖运行时分布，稳定性更差。
5. 推导 C4：若不引入 page-aware 调度与访存建模，FlashTensor 在在线 decode 的优化上限会被限制。

#### 3.3.3 结论（针对问题 3）

1. 不是“编译器不能解析”，而是“**难做高质量静态优化**”。
2. 不连续内存访问会直接削弱 coalescing、tiling 和预取收益，导致性能波动和退化。

### 3.4 三问合并后的总判断

#### 客观事实

1. 这三问分别对应 decode 的三类真实约束：动态 shape、memory-bound、paged-KV 不规则访存。
2. 三类约束在在线推理里会同时存在，并非彼此独立。

#### 分析判断

1. FlashTensor 不是“不能用于 decode”，而是“只能在受约束场景直接复用，在线场景需系统级改造”。
2. 改造重点应从“单 kernel 更快”升级为“编译策略 + 运行时分派 + KV 管理协同”。

---

## 4. FlashTensor 能力拆解：哪些能迁移，哪些会卡

### 4.1 IR 翻译与图规约（ONNX -> Asuka, simplify/fission）

### 客观事实

1. 现有流程能把规则算子图翻译到 IR，并做 fission/simplify。
2. decode 单步算子图本身可表示，语义上可进入编译流程。

### 分析判断

1. 这部分 **可迁移**，属于通用编译前端能力。
2. 难点不在“能否表示 decode 图”，而在后续 schedule 与访存优化是否还成立。

### 4.2 子图分区与融合（Connected partition + optimize）

### 客观事实

1. 现有分区/融合策略在规则图上能产生性能更优 kernel 组合。
2. decode 图通常更碎，且 step 粒度更小，运行时 launch 频繁。

### 分析判断

1. 这部分 **有条件可迁移**：在受约束 decode（固定桶 + 连续 KV）下仍有收益。
2. 在线 decode 下，分区空间与运行时动态交互复杂，收益波动大，易被内存带宽上限吞噬。

### 4.3 静态调度（tiling/dynamic_for）与代码生成

### 客观事实

1. 高效 tiling/coalescing 推断通常依赖规则 stride 和可预测循环边界。
2. PagedAttention 会引入 block table 间接寻址与 gather 型访问。

### 分析判断

1. 这部分是 **主要卡点**：不是“不能编译”，而是“只能保守优化”。
2. 对 paged KV，如果没有 page-aware 调度模型，静态 codegen 的收益会明显不稳定。

### 4.4 profile 驱动选择与 runtime dispatch

### 客观事实

1. FlashTensor 已有 profile + codegen + 选择机制。
2. decode 每 token 都会触发调度决策，且在线 shape 分布长尾明显。

### 分析判断

1. 这部分 **可迁移但必须重构策略**：需要 bucketing + 多版本编译 + 轻量 dispatch。
2. 若版本数失控，编译成本、缓存占用、运行时选择开销都会放大。

---

## 5. 迁移决策矩阵（终版）

| FlashTensor 组件 | 受约束 decode（离线/桶化/连续 KV） | 真实在线 decode（continuous batching + paged KV） | 结论 | 关键困难 |
|---|---|---|---|---|
| IR 翻译与图规约 | 可直接复用 | 基本可复用 | 可适用 | 不是主瓶颈 |
| 子图分区与融合 | 有收益 | 收益不稳定 | 部分适用 | memory-bound 下融合红利受限 |
| 静态 tiling/coalescing 调度 | 可做 | 易退化 | 不好直接适用 | 非连续访问破坏规则访存假设 |
| profile + 版本选择 | 可做 | 必须升级 | 部分适用 | shape/batch 动态导致版本爆炸 |
| 连续张量假设下的 kernel 模板 | 可做 | 与 paged KV 冲突 | 不好适用 | 间接寻址 + gather/scatter |
| 端到端运行时协同（调度 + KV 管理） | 简化场景可控 | 现实场景复杂 | 当前薄弱 | 编译器与服务运行时边界未打通 |

---

## 6. 三个核心困难（精确归因）

### 6.1 动态形状困难：`B/T` 持续变化

### 客观事实

1. decode 的 `kv_len` 每步增长，请求随时进出导致 batch 改变。
2. 编译期难以固定循环边界、tile 规模和寄存器预算。

### 分析判断

1. 需要用“离散静态化”替代“完全静态化”：shape bucket + padding + 多版本编译 + dispatch。
2. 长尾形状必须保留 fallback，否则稳定性与覆盖率不可接受。

### 6.2 IO 困难：decode 常是 HBM 带宽受限

### 客观事实

1. 这里的 IO 主要是 GPU 内部内存层级 IO（HBM <-> SM/L2/shared/register）。
2. decode 带宽利用率随上下文增长明显上升（Phase1 已观测）。

### 分析判断

1. 仅做“算子融合提 FLOPS”并不足够，关键是减少访存字节数和提升访问效率。
2. 迁移评估应重点加入 `bytes/token`、HBM 利用率、cache/locality 指标。

### 6.3 PagedAttention 困难：不规则地址访问

### 客观事实

1. paged KV 访问包含 page table 查找和间接寻址。
2. 地址不连续会降低 warp 合并访问（coalescing）质量，增加 gather 开销。

### 分析判断

1. 不是“编译器无法解析”，而是“静态推断能力变弱、优化更保守”。
2. 若不把 page table 语义纳入 IR/调度模型，decode 优化上限会被访存模式锁死。

---

## 7. 最终回答：哪些可适用、哪些不好适用、困难是什么

### 7.1 可以适用（优先迁移）

### 客观事实

1. IR 翻译、图规约、基础分区框架对 decode 图仍然有效。
2. 受约束场景下（桶化长度 + 连续 KV）已有可行优化空间。

### 分析判断

1. 可优先复用“前端编译链 + 多版本 kernel 生成”。
2. 这是低风险切入点，能快速建立 decode 版本基线。

### 7.2 不好适用（需要改造）

### 客观事实

1. 现有高效调度多依赖连续内存和相对稳定 shape。
2. 在线 decode 常见 paged KV 与 continuous batching。

### 分析判断

1. “连续张量假设下的静态调度/融合策略”不适合直接搬到在线 decode。
2. 不改运行时协同（dispatch + KV 管理）时，端到端收益会被系统开销抵消。

### 7.3 核心困难（必须直面）

### 客观事实

1. 动态形状导致版本管理复杂。
2. memory-bound 限制融合收益上限。
3. paged KV 破坏规则访存模型。

### 分析判断

1. 困难是“编译优化与在线系统约束的错位”，而非单一 kernel 技术问题。
2. 解决路径必须是“编译 + 运行时 + KV 内存管理”联合设计。

---

## 8. 可执行迁移路线（建议作为后续实验计划）

### 阶段 A：受约束 decode MVP

### 客观事实

1. 固定 `batch policy`、限制长度桶（如 1k/2k/4k/8k）、连续 KV 能显著降低动态复杂度。

### 分析判断

1. 目标是先验证 FlashTensor 在 decode 的“可迁移上限”。
2. 验收指标：`latency/token`、`bytes/token`、HBM 利用率、正确性一致性。

### 阶段 B：多版本编译 + 轻量 runtime dispatch

### 客观事实

1. 当前框架已有 profile/codegen 基础，可扩展成桶化版本池。

### 分析判断

1. 重点是控制版本数和分派开销，避免“编译时间/缓存成本 > 性能收益”。

### 阶段 C：PagedAttention 感知优化

### 客观事实

1. 真实在线场景需要分页 KV 管理。

### 分析判断

1. 需要在 IR 与调度阶段显式建模 `page_id/offset`，设计 page-aware load/store 与 tile 策略。
2. 验收指标应增加 page locality、gather 代价、不同请求分布下的性能稳定性。

---

## 9. 一句话版结论

FlashTensor 在 decode 阶段的可迁移部分主要是“编译前端与受约束融合能力”；不好直接适用的是“依赖连续内存和静态形状的调度优化”；真正困难是动态 shape、HBM 带宽瓶颈、以及 PagedAttention 带来的不规则访存这三者叠加。
