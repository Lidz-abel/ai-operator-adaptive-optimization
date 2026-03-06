# Phase 1: 基线复现与瓶颈确诊 (RTX 3090)

## 🎯 实验目标

定量证明在带有复杂淘汰逻辑的 Decode 阶段，算子碎片化（Kernel fragmentation）和访存墙（Memory wall）导致了严重的性能退化。

## 🔬 核心假设

### Hypothesis 1: Decode 阶段算子碎片化严重
**预期**: H2O 在 Decode 阶段会被切分成 3-5 个 Kernel，而 Prefill 阶段只需 1-2 个

### Hypothesis 2: Kernel 启动开销占比高
**预期**: Decode 阶段的 Kernel 启动开销 > 30%，Prefill 阶段 < 5%

### Hypothesis 3: 内存带宽利用率低
**预期**: Decode 阶段带宽利用率 < 40%，Prefill 阶段 > 70%

### Hypothesis 4: 相同 KV Cache 被重复读取
**预期**: H2O Decode 阶段 KV Cache 被读取 3-4 次，理想情况只需 1 次

## 📋 实验配置

### 模型
- **标准模型**: Llama-2-7b Attention
- **变体模型**: H2O (带 KV Cache 淘汰机制)

### 测试场景
基于 RTX 3090 (24GB) 的显存限制：

| 场景 | Context Length | Batch Size | 模式 | 预期显存 | 状态 |
|------|----------------|------------|------|----------|------|
| 1 | 2048 | 1 | Prefill | ~17 GB | ✅ 已验证 |
| 2 | 2048 | 1 | Decode | ~17 GB | 🔄 待测试 |
| 3 | 4096 | 1 | Prefill | ~18 GB | 🔄 待测试 |
| 4 | 4096 | 1 | Decode | ~18 GB | 🔄 待测试 |
| 5 | 8192 | 1 | Prefill | ~21 GB | ⚠️ 接近上限 |
| 6 | 8192 | 1 | Decode | ~21 GB | ⚠️ 接近上限 |

### 基线系统
1. **torch**: PyTorch 原生实现
2. **dynamo**: torch.compile (TorchInductor)

## 🚀 快速开始

### 1. 环境准备

```bash
# 确保在项目根目录
cd /data2/ldz/FlashTensor-AE

# 激活环境
source decode_experiments/decode_env.sh

# 进入 Phase 1 目录
cd decode_experiments/phase1_baseline
