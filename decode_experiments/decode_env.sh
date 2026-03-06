#!/bin/bash
# Decode 实验专用环境配置 - RTX 3090 版本

PROJECT_DIR=$(dirname $(dirname $(realpath "${BASH_SOURCE[0]}")))
echo "project_dir: $PROJECT_DIR"

# 检测 GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    echo "🖥️  检测到 GPU: $GPU_NAME ($GPU_MEMORY MB)"
else
    echo "❌ 未检测到 NVIDIA GPU"
    exit 1
fi

# 清理现有环境
if command -v deactivate &>/dev/null; then
    deactivate
    echo "已退出之前的虚拟环境"
fi

# 激活虚拟环境
if [ -d "${PROJECT_DIR}/baseline_venv" ]; then
    source ${PROJECT_DIR}/baseline_venv/bin/activate
    echo "✅ 激活 baseline_venv"
elif [ -d "${PROJECT_DIR}/venv" ]; then
    source ${PROJECT_DIR}/venv/bin/activate
    echo "✅ 激活 venv"
else
    echo "⚠️  未找到虚拟环境，使用系统 Python"
fi

# Decode 实验特定配置
export DECODE_EXP_DIR="${PROJECT_DIR}/decode_experiments"
export DECODE_RESULTS_DIR="${DECODE_EXP_DIR}/phase1_baseline/results"

# RTX 3090 特定配置
export GPU_TYPE="RTX3090"
export GPU_MEMORY_GB=24
export CUDA_VISIBLE_DEVICES=0

# CUDA 性能配置
# CUDA_LAUNCH_BLOCKING=0: 异步模式，获得真实性能（默认值，可以不设置）
# CUDA_LAUNCH_BLOCKING=1: 同步模式，用于调试
export CUDA_LAUNCH_BLOCKING=0

# PyTorch CUDA 内存分配器配置
# max_split_size_mb: 限制内存块最大分割大小，防止碎片化
# 512MB 是适合 24GB 显存的经验值
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# ========== 基于原项目硬编码的配置 ==========

# Batch Size（原项目硬编码，不可修改）
export BATCH_SIZE=1

# Context Length 配置（唯一可变参数）
export RECOMMENDED_CONTEXT_LENGTHS="2048,4096,8192"  # 基于 RTX 3090 实测
export MAX_SAFE_CONTEXT_LENGTH=8192  # 保守估计，应该安全
export AGGRESSIVE_CONTEXT_LENGTH=16384  # 需要测试，可能 OOM

# 模型架构（Llama-2-7b 固定值）
export HEAD_NUM=32
export HEAD_DIM=128
export KV_HEAD_NUM=32
export NUM_LAYERS=32

# H2O 特定参数（原项目硬编码）
export CACHE_BUDGET=512

# 数据类型
export DTYPE="float16"

# 性能测试参数（来自原项目）
export KERNEL_RUN=10
export KERNEL_WARMUP=100
export E2E_RUN=50
export E2E_WARMUP=50

echo ""
echo "=" * 70
echo "✅ 环境配置完成"
echo "=" * 70
echo ""
echo "📊 硬件信息:"
echo "   GPU: $GPU_TYPE ($GPU_MEMORY_GB GB)"
echo "   Python: $(which python)"
echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "   CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'Unknown')"
echo "   CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
echo ""
echo "📁 实验配置:"
echo "   实验目录: $DECODE_EXP_DIR"
echo "   结果目录: $DECODE_RESULTS_DIR"
echo ""
echo "🔧 模型配置 (基于 Llama-2-7b):"
echo "   Batch Size: $BATCH_SIZE (固定)"
echo "   Num Layers: $NUM_LAYERS"
echo "   Num Heads: $HEAD_NUM"
echo "   Head Dim: $HEAD_DIM"
echo "   KV Heads: $KV_HEAD_NUM"
echo "   Data Type: $DTYPE"
echo ""
echo "⚠️  RTX 3090 显存限制 (24GB):"
echo "   推荐 Context Length: $RECOMMENDED_CONTEXT_LENGTHS"
echo "   最大安全 Context: $MAX_SAFE_CONTEXT_LENGTH"
echo "   激进配置 Context: $AGGRESSIVE_CONTEXT_LENGTH (需测试)"
echo "   Batch Size: $BATCH_SIZE (原项目不支持修改)"
echo ""
echo "🚀 CUDA 性能配置:"
echo "   CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING (0=异步, 1=同步)"
echo "   PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo ""
