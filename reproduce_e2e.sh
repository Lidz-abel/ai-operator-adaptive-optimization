#用来复现端到端的脚本
#!/bin/bash
#设置空闲显卡
export GPU_ID=2
#模型设置HuggingFace上的模型ID
MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#可修改如下
#MODEL_PATH="meta-llama/Llama-2-7b-hf"
# [实验参数]
SEQLEN=2048           # 序列长度 (论文标准是 4096)
LAYER_NUM=32          # 模拟的层数 (标准 Llama-2-7b 是 32 层)
PLATFORM_NAME="amax"  # 平台代号 (不要改，对应下面的 json)
BENCH_MODEL="h2o"     # 想要测试的 Attention 变体 (h2o 或 vanilla)

if [ ! -f "run_e2e.py" ]; then
    echo "  错误: 请将此脚本放在 FlashTensor-AE 项目根目录下运行！"
    echo "   (即和 run_e2e.py 同一级目录)"
    exit 1
fi

echo "========================================================"
echo "   初始化配置..."
echo "   - 使用显卡: GPU ${GPU_ID}"
echo "   - 测试模型: ${MODEL_PATH}"
echo "   - 序列长度: ${SEQLEN}"
echo "========================================================"

# --- 2. 自动生成配置文件 (weight_zoo.json) ---

cat > weight_zoo.json <<EOF
{
    "yes": "/dummy/path",
    "qiyuan": "/dummy/path",
    "fuse0": "/dummy/path",
    "${PLATFORM_NAME}": "${MODEL_PATH}"
}
EOF

echo " 配置文件已更新。"

# --- 3. 开始运行实验 ---

export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo ""
echo "########################################################"
echo "阶段一: 运行 PyTorch Baseline (基准)"
echo "########################################################"
echo "正在启动..."

python run_e2e.py \
  --model ${BENCH_MODEL} \
  --system torch \
  --seqlen ${SEQLEN} \
  --platform ${PLATFORM_NAME} \
  --layer_num ${LAYER_NUM}

echo ""
echo "########################################################"
echo " 阶段二: 运行 FlashTensor (Ours)"
echo "########################################################"
echo "正在启动..."

python run_e2e.py \
  --model ${BENCH_MODEL} \
  --system our \
  --seqlen ${SEQLEN} \
  --platform ${PLATFORM_NAME} \
  --layer_num ${LAYER_NUM}

echo ""
echo "========================================================"
echo "实验结束！"
echo "请对比上方两个阶段输出的 'Mean inference time'。"
echo "========================================================"

