#!/bin/bash
set -x

PROJECT_DIR=$(dirname $(dirname $(readlink -f $0)))

# 进入源码目录
cd ${PROJECT_DIR}/3rd/asuka

# 安装 FlashTensor 核心 (添加 --no-cache-dir)
pip install -e . -v --no-build-isolation --no-cache-dir 2>&1 | tee build.log

pip uninstall triton -y 
pip uninstall triton-nightly -y
pip install -U --no-cache-dir --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly==3.0.0.post20240708181524
