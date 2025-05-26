#!/bin/bash
# PyTorch 2.7 + CUDA 12.8 + DGL 2.4+ Setup Script for RFdiffusion
# Complete installation of bleeding-edge environment

set -e  # Exit on error

echo "🚀 RFdiffusion PyTorch 2.7 + CUDA 12.8 + DGL 2.4+ Setup"
echo "========================================================"

# Check if we're in the right directory
if [[ ! -f "setup.py" ]] || [[ ! -d "env" ]]; then
    echo "❌ Error: Please run this script from the RFdiffusion root directory"
    exit 1
fi

# Step 1: Create the environment
echo "📦 Step 1: Creating PyTorch 2.7 + CUDA 12.8 environment..."
if conda env list | grep -q "SE3nv-pytorch27-cuda128"; then
    echo "⚠️  Environment SE3nv-pytorch27-cuda128 already exists. Remove it? (y/N)"
    read -p "" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove --name SE3nv-pytorch27-cuda128 -y
    else
        echo "❌ Aborted by user"
        exit 1
    fi
fi

conda env create -f env/SE3nv-pytorch27-cuda128.yml
echo "✅ Environment created successfully"

# Step 2: Activate environment
echo "🔄 Step 2: Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SE3nv-pytorch27-cuda128

# Step 3: Verify PyTorch 2.7 installation
echo "🔍 Step 3: Verifying PyTorch 2.7 installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    if torch.cuda.device_count() > 0:
        print(f'GPU name: {torch.cuda.get_device_name(0)}')

# Verify we have PyTorch 2.7+
major, minor = torch.__version__.split('.')[:2]
if int(major) >= 2 and int(minor) >= 7:
    print('✅ PyTorch 2.7+ confirmed')
else:
    print('❌ PyTorch 2.7+ not detected')
    exit(1)
"

# Step 4: Install SE3-Transformer
echo "🔧 Step 4: Installing SE3-Transformer with flexible requirements..."
cd env/SE3Transformer
pip install --no-cache-dir -r requirements-flexible.txt
python setup.py install
cd ../..

# Step 5: Install PyTorch 2.7 with CUDA 12.8 (if not already installed)
echo "📦 Step 5: Installing PyTorch 2.7 with CUDA 12.8..."
# Check if we need to upgrade PyTorch
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if [[ ! "$PYTORCH_VERSION" =~ ^2\.7 ]]; then
    echo "🔄 Upgrading to PyTorch 2.7 with CUDA 12.8..."
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
else
    echo "✅ PyTorch 2.7 already installed"
fi

# Install RFdiffusion with bleeding-edge extras
echo "📦 Installing RFdiffusion with bleeding-edge configuration..."
pip install -e .[bleeding-edge]

# Step 6: Build DGL 2.4+ from source
echo "🔨 Step 6: Building DGL 2.4+ from source..."
echo "   This will take 10-30 minutes depending on your system..."
bash scripts/upgrade_dgl_source.sh

# Step 7: Final verification
echo "🧪 Step 7: Final verification..."
python -c "
import warnings
warnings.filterwarnings('ignore')

print('=== Final Verification ===')

# Check PyTorch
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')

# Check DGL
import dgl
print(f'✅ DGL: {dgl.__version__}')

# Test basic DGL functionality
x = torch.tensor([0, 1, 2])
y = torch.tensor([1, 2, 0])
g = dgl.graph((x, y), num_nodes=3)
print(f'✅ DGL graph creation: OK')

# Check NumPy
import numpy as np
print(f'✅ NumPy: {np.__version__}')

# Test RFdiffusion integration
try:
    import rfdiffusion.util_module
    print('✅ RFdiffusion DGL integration: OK')
    
    # Test inference imports
    from scripts.run_inference import main
    print('✅ RFdiffusion inference: OK')
    
except Exception as e:
    print(f'❌ RFdiffusion test failed: {e}')
    exit(1)

print('\\n🎉 SUCCESS: PyTorch 2.7 + CUDA 12.8 + DGL 2.4+ setup complete!')
print('\\nVersions installed:')
print(f'  - PyTorch: {torch.__version__}')
print(f'  - DGL: {dgl.__version__}')
print(f'  - NumPy: {np.__version__}')
print(f'  - CUDA: {torch.version.cuda}')
"

echo ""
echo "🎯 Setup Complete!"
echo "=================="
echo "Your bleeding-edge RFdiffusion environment is ready with:"
echo "  - PyTorch 2.7+ with CUDA 12.8 support"
echo "  - DGL 2.4+ built from source (no torchdata issues)"
echo "  - NumPy 2.x support"
echo "  - Full RFdiffusion compatibility"
echo ""
echo "To activate this environment in the future:"
echo "  conda activate SE3nv-pytorch27-cuda128"
echo ""
echo "To test RFdiffusion:"
echo "  python scripts/run_inference.py --help"