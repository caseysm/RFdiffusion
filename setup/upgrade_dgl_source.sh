#!/bin/bash
# DGL Source Build Script for RFdiffusion
# Upgrades DGL from 1.x to 2.4+ by building from source

set -e  # Exit on error

echo "Working: RFdiffusion DGL Source Upgrade Script"
echo "========================================="

# Check if we're in a conda environment
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "ERROR: Error: Please activate a conda environment first"
    echo "   conda activate SE3nv-flexible"
    exit 1
fi

echo "Current: Current environment: $CONDA_DEFAULT_ENV"

# Check prerequisites
echo "Checking: Checking prerequisites..."

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: Error: CUDA toolkit not found. Please install CUDA toolkit 12.x"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9.]+')
echo "SUCCESS: CUDA version: $CUDA_VERSION"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "ERROR: Error: CMake not found. Installing..."
    conda install cmake -y
fi

CMAKE_VERSION=$(cmake --version | head -n1 | grep -oP '[0-9.]+')
echo "SUCCESS: CMake version: $CMAKE_VERSION"

# Check current DGL version
echo "Checking: Current DGL version:"
python -c "import dgl; print(f'DGL {dgl.__version__}')" || echo "DGL not installed"

# Confirm upgrade
echo ""
echo "=> This will:"
echo "   1. Uninstall current DGL"
echo "   2. Clone DGL 2.4.x from GitHub"
echo "   3. Build and install from source"
echo "   4. Test RFdiffusion compatibility"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "ERROR: Aborted by user"
    exit 1
fi

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo "Directory: Working directory: $TEMP_DIR"
cd "$TEMP_DIR"

# Uninstall current DGL
echo "Removing: Uninstalling current DGL..."
pip uninstall dgl -y || echo "DGL not installed via pip"

# Clone DGL repository
echo "Downloading: Cloning DGL repository..."
git clone --recursive https://github.com/dmlc/dgl.git
cd dgl

# Checkout specific version
DGL_VERSION="v2.4.0"
echo "Working: Checking out DGL $DGL_VERSION..."
git checkout $DGL_VERSION
git submodule sync
git submodule update --init --recursive

# Set build environment variables
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Detect PyTorch version and set compatible CUDA version
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
echo "Checking: Detected PyTorch version: $PYTORCH_VERSION"

# Set CUDA toolkit version based on PyTorch
if [[ "$PYTORCH_VERSION" =~ ^2\.7 ]]; then
    echo "Target: PyTorch 2.7 detected - using CUDA 12.8 optimizations"
    CUDA_TOOLKIT_VERSION="12.8"
elif [[ "$PYTORCH_VERSION" =~ ^2\.[4-6] ]]; then
    echo "Target: PyTorch 2.4-2.6 detected - using CUDA 12.4 optimizations"
    CUDA_TOOLKIT_VERSION="12.4"
else
    echo "Target: PyTorch 2.3 or older detected - using CUDA 12.1 optimizations"
    CUDA_TOOLKIT_VERSION="12.1"
fi

# Determine CUDA architecture
CUDA_ARCHS="7.0;7.5;8.0;8.6;8.9;9.0"  # Common architectures
if command -v nvidia-smi &> /dev/null; then
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1 | tr -d ' ')
    if [[ ! -z "$GPU_ARCH" ]]; then
        CUDA_ARCHS="$GPU_ARCH"
        echo "Target: Detected GPU compute capability: $GPU_ARCH"
    fi
fi

# Set additional build flags for PyTorch 2.7 + CUDA 12.8
if [[ "$PYTORCH_VERSION" =~ ^2\.7 ]]; then
    echo "Working: Setting PyTorch 2.7 + CUDA 12.8 optimizations..."
    export USE_CUDA=ON
    export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
    export TORCH_CUDA_ARCH_LIST="$CUDA_ARCHS"
    export DGL_LIBRARY_PATH="$CUDA_HOME/lib64"
fi

echo "Building: Building DGL from source..."
echo "   This may take 10-30 minutes depending on your system..."

# Build with CUDA support
mkdir -p build
cd build

cmake .. \
    -DUSE_CUDA=ON \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DUSE_NCCL=ON \
    -DBUILD_TYPE=Release \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)

# Install DGL
echo "Installing: Installing DGL..."
cd ../python
python setup.py install

# Test installation
echo "Testing: Testing DGL installation..."
python -c "
import dgl
print(f'SUCCESS: DGL version: {dgl.__version__}')

import torch
print(f'SUCCESS: PyTorch version: {torch.__version__}')
print(f'SUCCESS: CUDA available: {torch.cuda.is_available()}')

# Test basic DGL functionality
x = torch.tensor([0, 1, 2])
y = torch.tensor([1, 2, 0])
g = dgl.graph((x, y), num_nodes=3)
print(f'SUCCESS: Basic DGL graph creation: OK')

# Test RFdiffusion integration
try:
    import rfdiffusion.util_module
    print('SUCCESS: RFdiffusion DGL integration: OK')
except Exception as e:
    print(f'WARNING: RFdiffusion test failed: {e}')
"

# Cleanup
echo "Cleaning: Cleaning up..."
cd /
rm -rf "$TEMP_DIR"

echo ""
echo "SUCCESS: DGL upgrade complete!"
echo "   You can now use DGL 2.4+ features without torchdata dependencies"
echo "   Original DGL 1.x can be restored with:"
echo "   pip install 'dgl>=1.1.0,<2.0.0' -f https://data.dgl.ai/wheels/cu121/repo.html"