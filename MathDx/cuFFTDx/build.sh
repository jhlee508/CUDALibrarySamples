#!/bin/bash

mkdir -p build && cd build

cmake -DCUFFTDX_CUDA_ARCHITECTURES=80 -Dmathdx_ROOT=/home/jaehwan/lib/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06 ..
make -j

# Run
# ctest