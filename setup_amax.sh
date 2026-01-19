#!/bin/bash
set -e

# 获取项目路径
PROJECT_DIR=$(readlink -f .)
HOST=$(hostname)

echo "Current Hostname: $HOST"
echo "Project Directory: $PROJECT_DIR"

# ==========================================
# 0. 磁盘空间急救配置 (关键)
# ==========================================

# 在当前大硬盘目录下创建临时文件夹
export BUILD_TMP_DIR="${PROJECT_DIR}/tmp_build_cache"
mkdir -p $BUILD_TMP_DIR

echo "!!! Setting temporary directory to: $BUILD_TMP_DIR !!!"
export TMPDIR=$BUILD_TMP_DIR
export TEMP=$BUILD_TMP_DIR
export TMP=$BUILD_TMP_DIR

# ==========================================
# 1. 编译器环境配置
# ==========================================

# 优先使用 Conda 环境中的 LLVM
if [ -n "$CONDA_PREFIX" ]; then
    export PATH=$CONDA_PREFIX/bin:$PATH
fi

# 检查 nvcc
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install cuda-toolkit in conda."
    exit 1
fi

export CC=$(which gcc)
export CXX=$(which g++)
export TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0"

# 帮助 CMake 找到 Conda 里的 MLIR
if [ -n "$CONDA_PREFIX" ]; then
    export LLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm
    export MLIR_DIR=$CONDA_PREFIX/lib/cmake/mlir
fi

echo "Using Compiler: $CC / $CXX"

# ==========================================
# 2. 构建 Baseline 环境
# ==========================================

echo ">>> [1/2] Creating 'baseline_venv'..."

if [ -d "baseline_venv" ]; then
  rm -rf baseline_venv
fi

python -m venv baseline_venv
source baseline_venv/bin/activate

echo "Installing Dependencies (No Cache)..."
pip install --upgrade pip --no-cache-dir

# 安装基础包
pip install packaging==24.2 wheel==0.45.0 --no-cache-dir

# 安装 PyTorch (指定 source 且不缓存)
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
# 必须设置 TMPDIR 防止编译时爆 /tmp
TMPDIR=$BUILD_TMP_DIR pip install -v -e ${PROJECT_DIR} --no-build-isolation --no-cache-dir

echo "Building TVM (Baseline)..."
chmod +x ${PROJECT_DIR}/script/build_tvm.sh
${PROJECT_DIR}/script/build_tvm.sh

deactivate
echo ">>> 'baseline_venv' setup complete."


# ==========================================
# 3. 构建 FlashTensor 环境
# ==========================================

echo ">>> [2/2] Creating 'our_venv'..."

if [ -d "our_venv" ]; then
  rm -rf our_venv
fi

python -m venv our_venv
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


echo "Applying Patches..."
${PROJECT_DIR}/script/patch_pybind11.sh
${PROJECT_DIR}/script/patch_onnxsim.sh

echo "Installing Project..."
TMPDIR=$BUILD_TMP_DIR pip install -v -e ${PROJECT_DIR} --no-build-isolation --no-cache-dir

echo "Building FlashTensor..."
chmod +x ${PROJECT_DIR}/script/build_our.sh
${PROJECT_DIR}/script/build_our.sh

deactivate
echo ">>> 'our_venv' setup complete."

# 清理临时文件以释放空间
rm -rf $BUILD_TMP_DIR
echo "Cleaned up temp files."