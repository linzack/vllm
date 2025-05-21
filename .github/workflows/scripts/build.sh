#!/bin/bash
set -eux

python_executable=python$1
cuda_home=/usr/local/cuda-$2

# Update paths
PATH=${cuda_home}/bin:$PATH
LD_LIBRARY_PATH=${cuda_home}/lib64:$LD_LIBRARY_PATH

# Install requirements
<<<<<<< HEAD
$python_executable -m pip install -r requirements-build.txt -r requirements-cuda.txt
=======
$python_executable -m pip install -r requirements/build.txt -r requirements/cuda.txt
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

# Limit the number of parallel jobs to avoid OOM
export MAX_JOBS=1
# Make sure release wheels are built for the following architectures
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"
export VLLM_FA_CMAKE_GPU_ARCHES="80-real;90-real"

bash tools/check_repo.sh

# Build
$python_executable setup.py bdist_wheel --dist-dir=dist
