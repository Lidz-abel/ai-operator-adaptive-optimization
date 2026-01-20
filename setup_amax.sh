#!/bin/bash
set -e

# 获取项目路径
PROJECT_DIR=$(readlink -f .)
HOST=$(hostname)
# 获取 Conda 环境中的 Python 3.10 路径 (确保版本匹配)
PYTHON_310="/home/ldz/miniconda3/envs/flashtensor/bin/python"

echo "Current Hostname: $HOST"
echo "Project Directory: $PROJECT_DIR"
echo "Using Python Binary: $PYTHON_310"

# ==========================================
# 0. 磁盘空间急救配置
# ==========================================
export BUILD_TMP_DIR="${PROJECT_DIR}/tmp_build_cache"
mkdir -p $BUILD_TMP_DIR
export TMPDIR=$BUILD_TMP_DIR
export TEMP=$BUILD_TMP_DIR
export TMP=$BUILD_TMP_DIR

# ==========================================
# 1. 编译器环境配置
# ==========================================
if [ -n "$CONDA_PREFIX" ]; then
    export PATH=$CONDA_PREFIX/bin:$PATH
fi

# 检查 nvcc
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found."
    exit 1
fi

export CC=$(which gcc)
export CXX=$(which g++)
# 设置 RTX 3090 架构标志
export TORCH_CUDA_ARCH_LIST="8.6"

# 之前找到的 MLIR 确切路径
export MLIR_DIR=/home/ldz/miniconda3/envs/flashtensor/lib/cmake/mlir
export LLVM_DIR=/home/ldz/miniconda3/envs/flashtensor/lib/cmake/llvm

echo "Using Compiler: $CC / $CXX"

# ==========================================
# 2. 构建 Baseline 环境 (TVM)
# ==========================================
echo ">>> [1/2] Creating 'baseline_venv' with Python 3.10..."

if [ -d "baseline_venv" ]; then
  rm -rf baseline_venv
fi

# 显式使用 3.10 创建
$PYTHON_310 -m venv baseline_venv
source baseline_venv/bin/activate

echo "Installing Dependencies (No Cache)..."
pip install --upgrade pip --no-cache-dir
pip install packaging==24.2 wheel==0.45.0 --no-cache-dir
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

if [ -f "${PROJECT_DIR}/requirements-pedantic.txt" ]; then
    pip install -r ${PROJECT_DIR}/requirements-pedantic.txt --extra-index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
fi

echo "Installing flash-attn from wheel..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.1/flash_attn-2.6.1+cu123torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-cache-dir

echo "Applying Patches..."
chmod +x ${PROJECT_DIR}/script/patch_pybind11.sh
chmod +x ${PROJECT_DIR}/script/patch_onnxsim.sh
${PROJECT_DIR}/script/patch_pybind11.sh
${PROJECT_DIR}/script/patch_onnxsim.sh

echo "Installing Project..."
TMPDIR=$BUILD_TMP_DIR pip install -v -e ${PROJECT_DIR} --no-build-isolation --no-cache-dir

echo "Building TVM..."
chmod +x ${PROJECT_DIR}/script/build_tvm.sh
${PROJECT_DIR}/script/build_tvm.sh

deactivate


# ==========================================
# 3. 构建 FlashTensor 环境 (Our Method)
# ==========================================
echo ">>> [2/2] Creating 'our_venv' with Python 3.10..."

if [ -d "our_venv" ]; then
  rm -rf our_venv
fi

# 显式使用 3.10 创建
$PYTHON_310 -m venv our_venv
source our_venv/bin/activate

echo "Installing Dependencies (No Cache)..."
pip install --upgrade pip --no-cache-dir
pip install packaging==24.2 wheel==0.45.0 --no-cache-dir
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

if [ -f "${PROJECT_DIR}/requirements-pedantic.txt" ]; then
    pip install -r ${PROJECT_DIR}/requirements-pedantic.txt --extra-index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
fi

echo "Installing flash-attn from wheel..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.1/flash_attn-2.6.1+cu123torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-cache-dir

# 关键：安装 asuka 编译所需的构建工具
echo "Installing build tools for Asuka..."
pip install pybind11 py-build-cmake nanobind --no-cache-dir

echo "Applying Patches..."
${PROJECT_DIR}/script/patch_pybind11.sh
${PROJECT_DIR}/script/patch_onnxsim.sh

echo "Installing Project..."
TMPDIR=$BUILD_TMP_DIR pip install -v -e ${PROJECT_DIR} --no-build-isolation --no-cache-dir

echo "Building FlashTensor (Asuka Core)..."
chmod +x ${PROJECT_DIR}/script/build_our.sh
# build_our.sh 内部会调用 3rd/asuka 的编译，我们在外面已经导出了 MLIR_DIR
${PROJECT_DIR}/script/build_our.sh

deactivate

# ==========================================
# 4. 清理与完成
# ==========================================
rm -rf $BUILD_TMP_DIR
echo "Cleaned up temp files."
echo "=========================================="
echo "SUCCESS: Environment ready with Python 3.10!"
echo "source our_venv/bin/activate"
echo "=========================================="