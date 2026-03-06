# FlashTensor Decode 阶段扩展实验 (RTX 3090)

##  研究目标

评估并实现 FlashTensor 在 Decode 阶段的应用，特别是针对 H2O、RoCo 等注意力变体模型。

**核心问题**: 在标准大模型中，Decode 阶段确实不需要 FlashTensor（因为张量太小）；但在 H2O、RoCo 等注意力变体中，Decode 阶段引入了极其复杂的 KV Cache 动态淘汰和更新机制，导致严重的访存碎片化和算子碎片化，而这正是 FlashTensor 核心技术（非凸融合）的潜在用武之地。

##  核心

### 1. 瓶颈类型的根本转移
- **Prefill 阶段**: 显存容量瓶颈 (N×N 大张量)
  - FlashTensor 通过"代数变换"缩小张量体积
- **Decode 阶段**: 访存带宽 + Kernel 启动开销瓶颈
  - 每次只处理 1 个 Token，计算极快
  - H2O 拆分出 `Softmax → Reduce → TopK → Gather` 4个小算子
  - Kernel 启动耗时可能比计算本身还长

### 2. 非凸融合在 Decode 阶段的真正价值
- FlashTensor 的核心技术：打破传统编译器的凸融合限制
- 在 Decode 阶段，张量体积缩减可能失效
- 但非凸融合能把细碎小算子打包成一个 Kernel
- 实现数据在寄存器或共享内存中的复用

### 3. 动态张量带来的挑战
- KV 淘汰机制（如 H2O）引入动态形状（Dynamic Shape）
- FlashTensor 目前基于静态张量属性分析
- 需要评估在动态稀疏结构下的适用性

##  环境配置

### 硬件
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **计算能力**: 8.6
- **内存带宽**: ~936 GB/s

### 软件
- Python 3.10.13
- PyTorch 2.2.2+cu121
- CUDA 12.1

###  RTX 3090 显存限制

| 配置项 | A100/H100 | RTX 3090 |
|--------|-----------|----------|
| Max Context | 64K | 16K |
| Recommended Context | 32K, 64K | 2K, 4K, 8K, 16K |
| Max Batch Size | 32 | 8 |
| Recommended Batch | 1, 8, 16, 32 | 1, 2, 4, 8 |

##  快速开始

### 1. 激活环境

```bash
cd /data2/ldz/FlashTensor-AE
source decode_experiments/decode_env.sh
