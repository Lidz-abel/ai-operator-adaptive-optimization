#!/bin/bash
set -e

# === 配置 ===
PROJECT_DIR=$(readlink -f .)
LOG_DIR=${PROJECT_DIR}/repro_logs
mkdir -p ${LOG_DIR}

# 硬件与模型
DEVICE="rtx3090"
MODEL="h2o"          # 论文主要用 H2O 展示扩展性
SEQLENS=(1024 2048 4096 8192) # 测试序列

# 设置通用环境变量
export PROJECT_DIR=${PROJECT_DIR}
export PYTHONPATH=${PROJECT_DIR}:${PYTHONPATH}
export HF_HOME="/data2/ldz/hf_cache"
export CUDA_VISIBLE_DEVICES=2

echo "=============================================="
echo "Start Goal 3: Scalability Test on ${MODEL}"
echo "=============================================="

# --- 1. 运行 FlashTensor (our) ---
echo ">>> Phase 1: Running FlashTensor (our)..."
# 激活环境
source ${PROJECT_DIR}/our_venv/bin/activate

for seq in "${SEQLENS[@]}"; do
    echo "Running [our] seqlen=${seq}..."
    # 调用 run_kernel.py，注意我们修改了 --seqlen 参数
    # 使用 tee 同时输出到屏幕和文件
    python run_kernel.py -m ${MODEL} -s our --seqlen ${seq} --no-check 2>&1 | tee ${LOG_DIR}/goal3.${DEVICE}.${MODEL}.our.seq${seq}.log
done
# 退出环境
deactivate

echo "----------------------------------------------"

# --- 2. 运行 PyTorch Baseline (torch) ---
echo ">>> Phase 2: Running PyTorch Baseline (torch)..."
# 激活环境
source ${PROJECT_DIR}/baseline_venv/bin/activate

for seq in "${SEQLENS[@]}"; do
    echo "Running [torch] seqlen=${seq}..."
    # 8192 长度下 PyTorch 可能会 OOM，我们加个允许失败的逻辑
    set +e 
    python run_kernel.py -m ${MODEL} -s torch --seqlen ${seq} --no-check 2>&1 | tee ${LOG_DIR}/goal3.${DEVICE}.${MODEL}.torch.seq${seq}.log
    set -e
done
# 退出环境
deactivate

echo "=============================================="
echo "Goal 3 Testing Finished!"
echo "Logs are saved in ${LOG_DIR}"
