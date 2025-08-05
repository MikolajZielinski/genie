#!/bin/bash

set -e  # Exit on error

# === Find GPU Compute Capability ===
echo "🔍 Detecting GPU Compute Capability..."
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)
CUDA_ARCH="sm_${CC/./}"
echo "Detected CUDA_ARCH=$CUDA_ARCH"

# === Find CUDA Include Path ===
echo "🔍 Detecting CUDA include path..."
CUDA_INCLUDE=$(dirname $(dirname $(which nvcc)))/include
echo "Detected CUDA include path: $CUDA_INCLUDE"

# === Find Thrust Include Path ===
echo "🔍 Detecting Thrust include path..."
CUDA_ROOT=$(dirname $(dirname $(which nvcc)))
THRUST_INCLUDE="$CUDA_ROOT/include"
# Check if thrust headers exist in CUDA include, otherwise try targets path
if [ ! -d "$THRUST_INCLUDE/thrust" ]; then
    THRUST_INCLUDE="$CUDA_ROOT/targets/x86_64-linux/include"
fi
echo "Detected Thrust include path: $THRUST_INCLUDE"

LIBCUDA_PATH=$(find /usr -name 'libcuda.so*' 2>/dev/null | head -n 1)

if [ -n "$LIBCUDA_PATH" ]; then
  CUDA_LIB_DIR=$(dirname "$LIBCUDA_PATH")
  echo "Found libcuda.so at $CUDA_LIB_DIR"
else
  echo "libcuda.so not found in system paths!"
  # fallback or error handling here
  CUDA_LIB_DIR=""
fi

OPTIX_INCLUDE="NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64/include"

# Set Python path for includes and Torch
PYTHON_BIN=python3

# Safe fallback (no cpp_extension required)
PYTORCH_DIR=$($PYTHON_BIN -c "import torch, os; print(os.path.join(torch.__path__[0], 'include'))")
PYTORCH_API_DIR="$PYTORCH_DIR/torch/csrc/api/include"

PYTHON_SITE_PACKAGES=$($PYTHON_BIN -c "import site; print(site.getsitepackages()[0])")
TORCH_LIB_DIR="$PYTHON_SITE_PACKAGES/torch/lib"
PYBIND11_INCLUDES=$($PYTHON_BIN -m pybind11 --includes)
PYTHON_EXT_SUFFIX=$($PYTHON_BIN-config --extension-suffix)

# === Auto-detect ABI Flag ===
echo "🔍 Detecting PyTorch ABI flag..."
TORCH_CXX11_ABI=$($PYTHON_BIN -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
if [ "$TORCH_CXX11_ABI" = "1" ]; then
    ABI_FLAG="-D_GLIBCXX_USE_CXX11_ABI=1"
    echo "Detected ABI: CXX11 ABI enabled"
else
    ABI_FLAG="-D_GLIBCXX_USE_CXX11_ABI=0"
    echo "Detected ABI: CXX11 ABI disabled (pre-CXX11)"
fi
CXX_STD="-std=c++17"

# === Build Directories ===
BUILD_DIR="build"
mkdir -p $BUILD_DIR

# === 1. Compile OptiX Shader to PTX ===
echo "📦 Compiling shaders.cu to PTX..."
nvcc -ptx -arch=${CUDA_ARCH} -o ${BUILD_DIR}/shaders.ptx shaders.cu \
  -I${OPTIX_INCLUDE} -I${CUDA_INCLUDE} ${CXX_STD}

# === 2. Compile CUDA Source ===
echo "🔧 Compiling KNN.cu..."
nvcc -Xcompiler -fPIC -c KNN.cu -o ${BUILD_DIR}/KNN.o \
  --gpu-architecture=compute_86 --gpu-code=${CUDA_ARCH} \
  -I${OPTIX_INCLUDE} -I${CUDA_INCLUDE} ${ABI_FLAG} ${CXX_STD}

# === 3. Compile and link shared Python extension ===
echo "🔗 Compiling bindings.cpp to shared object..."
g++ -shared -fPIC bindings.cpp ${BUILD_DIR}/KNN.o -o optix_knn.so \
  ${CXX_STD} ${ABI_FLAG} ${PYBIND11_INCLUDES} \
  -I${CUDA_INCLUDE} \
  -I${OPTIX_INCLUDE} \
  -I${THRUST_INCLUDE} \
  -I${PYTORCH_DIR} \
  -I${PYTORCH_API_DIR} \
  -L${TORCH_LIB_DIR} \
  -L${CUDA_LIB_DIR} \
  -ltorch -ltorch_cpu -ltorch_python -lc10 -lcuda \
  -Wl,-rpath=${TORCH_LIB_DIR}

echo "✅ Build complete. Output: build/optix_knn.so"
