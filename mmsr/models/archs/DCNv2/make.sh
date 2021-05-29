#!/usr/bin/env bash

# You may need to modify the following paths before compiling.

CUDA_HOME=/mnt/lustre/wangxintao/cuda90 \
CUDNN_INCLUDE_DIR=/mnt/lustre/wangxintao/cuda90/include \
CUDNN_LIB_DIR=/mnt/lustre/wangxintao/cuda90/lib64 \
python setup.py build develop
