#!/bin/bash
set -x

PROJECT_DIR=$(dirname $(dirname $(readlink -f $0)))

cd ${PROJECT_DIR}/3rd/tvm

rm -rf build
mkdir build
cp ${PROJECT_DIR}/script/tvm_config.cmake build/config.cmake

cd build
# 使用 Ninja 编译
cmake .. -G Ninja
ninja

pip install xgboost==2.0.0 --no-cache-dir

cd ${PROJECT_DIR}/3rd/tvm/python
python setup.py install