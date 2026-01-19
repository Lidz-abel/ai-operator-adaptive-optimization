#!/bin/bash
set -e

PROJECT_DIR=$(readlink -f .)
LOG_DIR=${PROJECT_DIR}/repro_logs

echo "=============================================="
echo "Start Reproduction: Figure 12 (RTX 3090)"
echo "Logs will be saved to: $LOG_DIR"
echo "=============================================="

# 清理旧日志
rm -rf ${LOG_DIR}
mkdir -p ${LOG_DIR}

# === 选取的典型模型 ===
# 覆盖：标准Attention, 稀疏Attention, 变体
MODELS=("attn" "h2o" "gemma2")

# === 选取的系统 ===
# torch: Baseline
# our: FlashTensor
SYSTEMS=("torch" "our")

# === 1. 运行 Kernel 实验 ===
echo "##############################################"
echo "Phase 1: Running Kernel Experiments"
echo "##############################################"

for sys in "${SYSTEMS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "[Kernel] Running $model on $sys..."
        ./run_single_repro.sh kernel $model $sys
        echo "----------------------------------------------"
    done
done

# === 2. 运行 End-to-End 实验 ===
echo "##############################################"
echo "Phase 2: Running End-to-End Experiments"
echo "##############################################"

for sys in "${SYSTEMS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "[E2E] Running $model on $sys..."
        ./run_single_repro.sh e2e $model $sys
        echo "----------------------------------------------"
    done
done

echo "=============================================="
echo "All experiments finished!"
echo "Now run: python plot_repro.py"
echo "=============================================="