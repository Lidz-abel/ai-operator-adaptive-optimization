#!/bin/bash
set -e

# === 1. 修改参数检查，现在接收 4 个参数 ===
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <exp: kernel|e2e> <model> <sys: torch|our> <fullgraph: full|nofull>"
  exit 1
fi

EXP=$1
MODEL=$2
SYS=$3
FULLGRAPH_FLAG=$4   # 应为 "full" 或 "nofull"

# === 根据 flag 生成 fullgraph 选项 ===
if [[ "$FULLGRAPH_FLAG" == "full" ]]; then
    FULLGRAPH_OPT="--fullgraph"
elif [[ "$FULLGRAPH_FLAG" == "nofull" ]]; then
    FULLGRAPH_OPT="--no-fullgraph"
else
    echo "Error: fullgraph parameter must be 'full' or 'nofull'"
    exit 1
fi

# === 路径配置 ===
PROJECT_DIR=$(readlink -f .)
LOG_DIR=${PROJECT_DIR}/repro_logs
mkdir -p ${LOG_DIR}

# 硬件标识
DEVICE="rtx3090"

echo ">>> [Single Run] Exp=$EXP, Device=$DEVICE, Model=$MODEL, System=$SYS, Fullgraph=$FULLGRAPH_FLAG"

# === 环境切换逻辑 ===
if [[ "$SYS" == "our" ]]; then
    source ${PROJECT_DIR}/our_venv/bin/activate
else
    source ${PROJECT_DIR}/baseline_venv/bin/activate
fi

# 设置通用环境变量
export PROJECT_DIR=${PROJECT_DIR}
export PYTHONPATH=${PROJECT_DIR}:${PYTHONPATH}
export HF_HOME="/data2/ldz/hf_cache"
export CUDA_VISIBLE_DEVICES=2

# === 执行命令 ===
# 日志文件名中加入 fullgraph 标识
if [[ "$EXP" == "kernel" ]]; then
    # 运行 Kernel 实验
    CMD="python run_kernel.py -m $MODEL -s $SYS $FULLGRAPH_OPT --seqlen 2048 --no-check"
    LOGFILE="${LOG_DIR}/kernel.${DEVICE}.${MODEL}.${SYS}.${FULLGRAPH_FLAG}.log"
    echo "Executing: $CMD"
    $CMD 2>&1 | tee $LOGFILE

elif [[ "$EXP" == "e2e" ]]; then
    # 运行 E2E 实验
    CMD="python run_e2e.py -p amax -m $MODEL -s $SYS $FULLGRAPH_OPT --seqlen 2048"
    LOGFILE="${LOG_DIR}/e2e.${DEVICE}.${MODEL}.${SYS}.${FULLGRAPH_FLAG}.log"
    echo "Executing: $CMD"
    $CMD 2>&1 | tee $LOGFILE

else
    echo "Invalid experiment type: $EXP"
    exit 1
fi

# 退出环境，保持清洁
deactivate