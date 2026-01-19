#!/bin/bash
set -e

# 参数检查
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <exp: kernel|e2e> <model> <sys: torch|our>"
  exit 1
fi

EXP=$1
MODEL=$2
SYS=$3

# === 路径配置 ===
PROJECT_DIR=$(readlink -f .)
LOG_DIR=${PROJECT_DIR}/repro_logs
mkdir -p ${LOG_DIR}

# 硬件标识
DEVICE="rtx3090"

echo ">>> [Single Run] Exp=$EXP, Device=$DEVICE, Model=$MODEL, System=$SYS"

# === 环境切换逻辑 ===
# 注意：这里需要指向我们之前配好的 venv 的 activate 脚本
if [[ "$SYS" == "our" ]]; then
    source ${PROJECT_DIR}/our_venv/bin/activate
else
    source ${PROJECT_DIR}/baseline_venv/bin/activate
fi

# 设置通用环境变量
export PROJECT_DIR=${PROJECT_DIR}
export PYTHONPATH=${PROJECT_DIR}:${PYTHONPATH}
export HF_HOME="/data2/ldz/hf_cache"
export CUDA_VISIBLE_DEVICES=0

# === 执行命令 ===
if [[ "$EXP" == "kernel" ]]; then
    # 运行 Kernel 实验
    # 注意：这里假设 run_kernel.py 已经在根目录 (根据之前的 setup)
    python run_kernel.py -m $MODEL -s $SYS --no-check 2>&1 | tee ${LOG_DIR}/kernel.${DEVICE}.${MODEL}.${SYS}.log

elif [[ "$EXP" == "e2e" ]]; then
    # 运行 E2E 实验
    # -p amax 对应 weight_zoo.json 里的配置
    python run_e2e.py -p amax -m $MODEL -s $SYS 2>&1 | tee ${LOG_DIR}/e2e.${DEVICE}.${MODEL}.${SYS}.log

else
    echo "Invalid experiment type: $EXP"
    exit -1
fi

# 退出环境，保持清洁
deactivate