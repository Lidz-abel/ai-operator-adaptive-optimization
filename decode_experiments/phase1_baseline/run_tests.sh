#!/bin/bash
# Phase 1: 批量运行 Prefill vs Decode 对比测试

set -euo pipefail  # 遇到错误立即退出；管道中任一命令失败都视为失败

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Phase 1: Prefill vs Decode Comparison${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查环境
if [ ! -f "../../baseline_venv/bin/activate" ]; then
    echo -e "${RED}Error: baseline_venv not found${NC}"
    echo "Please run setup first"
    exit 1
fi

# 激活虚拟环境
source ../../baseline_venv/bin/activate

# 检查 GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found${NC}"
    exit 1
fi

echo -e "${YELLOW}GPU Info:${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 配置参数（对齐 run_kernel/run_e2e 风格）
MODELS=("h2o" "attn" "roco")
SYSTEM="torch"
CONTEXT_LENGTHS="2048 4096 8192"
WARMUP=10
RUNS=50
REPEATS=3
DTYPE="float16"
MIN_FREE_MEM_MB=20000
DEVICE_INDEX="${DEVICE_INDEX:-auto}"  # 可手动导出 DEVICE_INDEX=3 覆盖
OOM_POLICY="${OOM_POLICY:-skip}"      # 可选 raise/skip

if [ "${DEVICE_INDEX}" = "auto" ]; then
    GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -nr | head -n 1)
    DEVICE_INDEX=$(echo "${GPU_INFO}" | cut -d',' -f1 | tr -d ' ')
    DEVICE_FREE_MB=$(echo "${GPU_INFO}" | cut -d',' -f2 | tr -d ' ')
else
    DEVICE_FREE_MB=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F',' -v idx="${DEVICE_INDEX}" '$1 ~ idx {gsub(/ /, "", $2); print $2; exit}')
fi

if [ -z "${DEVICE_FREE_MB}" ]; then
    echo -e "${RED}Error: failed to query free memory for GPU ${DEVICE_INDEX}${NC}"
    exit 1
fi

if [ "${DEVICE_FREE_MB}" -lt "${MIN_FREE_MEM_MB}" ]; then
    echo -e "${RED}Error: selected GPU ${DEVICE_INDEX} free memory ${DEVICE_FREE_MB} MiB < ${MIN_FREE_MEM_MB} MiB${NC}"
    echo "Please free GPU memory or set DEVICE_INDEX to another GPU."
    exit 1
fi

# 创建结果目录
RESULTS_DIR="results"
mkdir -p ${RESULTS_DIR}

# 记录开始时间
START_TIME=$(date +%s)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "\n${GREEN}Test Configuration:${NC}"
echo "  Models: ${MODELS[@]}"
echo "  System: ${SYSTEM}"
echo "  Context Lengths: ${CONTEXT_LENGTHS}"
echo "  Warmup: ${WARMUP}"
echo "  Runs: ${RUNS}"
echo "  Repeats: ${REPEATS}"
echo "  DType: ${DTYPE}"
echo "  Device: cuda:${DEVICE_INDEX} (free ${DEVICE_FREE_MB} MiB)"
echo "  OOM Policy: ${OOM_POLICY}"
echo "  Results Dir: ${RESULTS_DIR}"
echo ""

# 运行测试
for MODEL in "${MODELS[@]}"; do
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Testing Model: ${MODEL}${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    OUTPUT_FILE="${RESULTS_DIR}/prefill_vs_decode_${MODEL}_${TIMESTAMP}.json"
    LOG_FILE="${RESULTS_DIR}/prefill_vs_decode_${MODEL}_${TIMESTAMP}.log"
    
    echo "Running: python test_prefill_vs_decode.py -m ${MODEL} -s ${SYSTEM} -c ${CONTEXT_LENGTHS} --warmup ${WARMUP} --runs ${RUNS} --repeats ${REPEATS} --dtype ${DTYPE} --device cuda:${DEVICE_INDEX} --oom_policy ${OOM_POLICY}"
    
    if python test_prefill_vs_decode.py \
        --model ${MODEL} \
        --system ${SYSTEM} \
        --context_lengths ${CONTEXT_LENGTHS} \
        --warmup ${WARMUP} \
        --runs ${RUNS} \
        --repeats ${REPEATS} \
        --dtype ${DTYPE} \
        --device cuda:${DEVICE_INDEX} \
        --oom_policy ${OOM_POLICY} \
        --output ${OUTPUT_FILE} \
        2>&1 | tee ${LOG_FILE}; then
        echo -e "${GREEN}✓ ${MODEL} test completed${NC}"

        echo "Plotting ${MODEL}..."
        if [ ! -f "${OUTPUT_FILE}" ]; then
            echo -e "${RED}Error: result file not found: ${OUTPUT_FILE}${NC}" | tee -a ${LOG_FILE}
            exit 1
        fi
        python plot_results.py --input ${OUTPUT_FILE} --output_dir plots 2>&1 | tee -a ${LOG_FILE}
    else
        echo -e "${RED}✗ ${MODEL} test failed${NC}"
        exit 1
    fi
done

# 记录结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All Tests Completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Total time: ${ELAPSED} seconds"
echo "Results saved in: ${RESULTS_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Re-plot one result: python plot_results.py --input <result.json> --output_dir plots"
echo "  2. Inspect raw JSON in: ${RESULTS_DIR}"
