#!/bin/bash
# Phase 1: 批量运行 Prefill vs Decode 对比测试

set -e  # 遇到错误立即退出

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

# 配置参数
MODELS=("h2o" "attn" "roco")
CONTEXT_LENGTHS="2048 4096 8192"
WARMUP=10
RUNS=50

# 创建结果目录
RESULTS_DIR="results"
mkdir -p ${RESULTS_DIR}

# 记录开始时间
START_TIME=$(date +%s)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "\n${GREEN}Test Configuration:${NC}"
echo "  Models: ${MODELS[@]}"
echo "  Context Lengths: ${CONTEXT_LENGTHS}"
echo "  Warmup: ${WARMUP}"
echo "  Runs: ${RUNS}"
echo "  Results Dir: ${RESULTS_DIR}"
echo ""

# 运行测试
for MODEL in "${MODELS[@]}"; do
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Testing Model: ${MODEL}${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    OUTPUT_FILE="${RESULTS_DIR}/prefill_vs_decode_${MODEL}_${TIMESTAMP}.json"
    LOG_FILE="${RESULTS_DIR}/prefill_vs_decode_${MODEL}_${TIMESTAMP}.log"
    
    echo "Running: python test_prefill_vs_decode.py -m ${MODEL} -c ${CONTEXT_LENGTHS} --warmup ${WARMUP} --runs ${RUNS}"
    
    if python test_prefill_vs_decode.py \
        --model ${MODEL} \
        --context_lengths ${CONTEXT_LENGTHS} \
        --warmup ${WARMUP} \
        --runs ${RUNS} \
        --output ${OUTPUT_FILE} \
        2>&1 | tee ${LOG_FILE}; then
        echo -e "${GREEN}✓ ${MODEL} test completed${NC}"
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
echo "  1. Visualize results: python plot_results.py"
echo "  2. Compare with source code: python compare_with_source.py"
