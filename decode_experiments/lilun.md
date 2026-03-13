# FlashTensor 理论储备：LLM 推理、Workload 与 Bottleneck

> 参考论文：**FlashTensor: Optimizing Tensor Programs by Leveraging Fine-grained Tensor Property**（PPoPP 2025）

## 1. LLM 推理过程：Prefill 和 Decode 的全过程

### 1.1 端到端推理主流程

1. 输入文本经过 tokenizer，转成 token IDs。
2. token IDs 进入 embedding，得到隐藏状态。
3. 执行 **Prefill**：一次性处理整段 prompt，建立初始 KV Cache，并得到第一个待生成 token 的 logits。
4. 执行 **Decode**：自回归循环，每步只处理 1 个新 token，直到遇到 EOS 或达到最大生成长度。
5. 输出生成文本。

### 1.2 Prefill 阶段全过程（一次吃完整个 Prompt）

设 prompt 长度为 `N`，隐藏维度为 `D`。

1. 对 `N` 个 token 并行做 embedding + position encoding。
2. 对每一层 Transformer（共 `L` 层）执行：
   - Norm（如 RMSNorm）
   - 线性投影得到 `Q, K, V`
   - 计算注意力分数 `QK^T / sqrt(d)`（因果 mask）
   - Softmax
   - `Attention * V`
   - 输出投影 + 残差
   - MLP（FFN）+ 残差
3. 每层把该层所有 token 的 `K/V` 写入 KV Cache。
4. 顶层 hidden state 经过 LM Head，得到 logits，采样出第一个生成 token。

**计算特征**：
- 核心是大矩阵并行（接近 GEMM 形态）
- Dense Attention 复杂度主项约为 `O(N^2)`
- 高吞吐、适合大 kernel 融合与算力利用

### 1.3 Decode 阶段全过程（自回归逐 token）

设当前已生成历史长度为 `t`。

1. 取最新 1 个 token 做 embedding + position encoding。
2. 对每一层 Transformer 执行：
   - 只为当前 token 计算 `q_t, k_t, v_t`
   - 将 `k_t, v_t` 追加到 KV Cache
   - `q_t` 与历史 `K_{1:t}` 做注意力（1 行对长序列）
   - Softmax 后与 `V_{1:t}` 加权求和
   - 输出投影 + MLP + 残差
3. LM Head 计算下一个 token logits，采样得到新 token。
4. 重复步骤 1-3，直到停止条件触发。

**计算特征**：
- 单步更像 GEMV / 向量-矩阵访存
- 单步复杂度约 `O(t)`，但每步计算量小、访存占比高
- 延迟敏感（TTFT/TBT）

---

## 2. Prefill 与 Decode 的差异

| 维度 | Prefill | Decode |
|---|---|---|
| 处理对象 | 整段 prompt（`N` 个 token） | 每步 1 个 token |
| 并行性 | token 维高度并行 | 时间维串行（自回归） |
| 主计算形态 | 大矩阵计算（GEMM-like） | 小计算 + 长 KV 读取（GEMV-like） |
| 复杂度主项 | Attention `O(N^2)` | 单步 `O(t)` |
| KV Cache 行为 | 一次性初始化大量 KV | 逐步追加/更新 KV |
| 性能目标 | 提高吞吐、压缩总算时 | 降低每 token 延迟 |
| 常见风险 | 长上下文显存容量和计算爆炸 | 带宽墙、kernel 启动开销、动态形状开销 |

---

## 3. 不同 Workload 介绍

### 3.1 按阶段划分

1. **Prefill-heavy workload**
   - 典型：长 prompt、短输出（如长文摘要、RAG 大上下文注入）
   - 特征：Prefill 时间占主导

2. **Decode-heavy workload**
   - 典型：短 prompt、长输出（如开放式对话、代码续写）
   - 特征：Decode 步数多，逐 token 延迟累积

3. **Mixed workload**
   - 典型：在线服务并发场景（请求长度分布离散）
   - 特征：系统要同时兼顾 TTFT 和 TPS

### 3.2 按注意力机制划分

1. **Dense Attention**
   - 规则、连续内存访问，Prefill 下计算密集。

2. **稀疏/淘汰类注意力（如 H2O、RoCo、SnapKV）**
   - Decode 中常出现 `TopK / Gather / Scatter / Evict` 等操作
   - 更容易产生算子碎片化和访存碎片化
   - 对动态形状和 KV 管理能力要求更高

### 3.3 按批处理与调度划分

1. **单请求低 batch**：更偏 latency-bound（尤其 Decode）
2. **高并发连续 batching**：吞吐提升，但调度与内存管理更复杂

---

## 4. 不同阶段的 Bottleneck

### 4.1 Prefill 阶段 Bottleneck

1. **长上下文的 `O(N^2)` 注意力开销**：计算和中间激活迅速膨胀。
2. **显存容量压力**：大 `N` 时容易接近显存上限，导致性能崩塌。
3. **内核调度与数据复用不足**：如果融合不充分，会放大中间张量读写。

**与 FlashTensor 的关系**：
- FlashTensor 的细粒度张量属性分析和非凸融合，核心价值在于 Prefill 场景下减少中间张量、提升算力利用率。

### 4.2 Decode 阶段 Bottleneck

1. **HBM 带宽瓶颈（memory-bound）**：每步都要读历史 KV，计算反而较少。
2. **Kernel 启动开销占比高**：算子小而碎时，launch overhead 可占明显比例。
3. **动态形状与 KV 管理开销**：长度逐步增长，若有淘汰策略还会引入更多动态访问。
4. **非连续/分页 KV 访问成本**：真实服务中常见页式 KV 管理，地址间接访问更重。

### 4.3 结合本仓库实验观察（RTX 3090, H2O）

- Phase1 报告显示：Decode 阶段 TFLOPS 很低，但带宽利用率随长度上升（说明偏带宽瓶颈）。
- 长上下文 Prefill 出现明显性能下跌与显存压力问题（说明存在容量与访存管理瓶颈）。

---

## 5. 小结

1. LLM 推理本质上由两种物理特性不同的阶段构成：**Prefill（并行、算力密集）** 与 **Decode（串行、带宽敏感）**。
2. FlashTensor 的优势更匹配 Prefill：通过编译与融合降低中间开销、提升高强度张量计算效率。
3. Decode 优化重点通常不在“纯计算融合”，而在 KV 访存路径、动态形状支持、kernel 碎片治理和调度系统协同。
