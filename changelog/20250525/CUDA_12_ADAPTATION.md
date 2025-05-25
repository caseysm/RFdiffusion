# CUDA 12 Environment Adaptation

This document details the changes made to adapt RFdiffusion for CUDA 12.x compatibility (tested with CUDA 12.9).

## Overview

The original RFdiffusion environment was configured for CUDA 11.1 with older package versions. This adaptation updates the environment to work with CUDA 12.x while maintaining compatibility with the core RFdiffusion functionality.

## System Configuration

- **CUDA Version**: 12.x (tested with 12.9, Driver Version: 576.52)
- **GPU**: NVIDIA GeForce RTX 4090
- **Target Python Version**: 3.11 (updated from 3.9)
- **PyTorch Version**: 2.3.0 (stable with DGL 1.1.3)

## Changes Made

### 1. Conda Environment (`env/SE3nv.yml`)

**Original Configuration:**
```yaml
name: SE3nv
dependencies:
  - python=3.9
  - pytorch=1.9
  - torchaudio
  - torchvision
  - cudatoolkit=11.1
  - dgl-cuda11.1
  - pip:
    - hydra-core
    - pyrsistent
```

**Updated Configuration:**
```yaml
name: SE3nv
dependencies:
  - python=3.11
  - nvidia::cuda-toolkit=12.9  # CUDA 12.x toolkit from nvidia channel
  - pip:
    - torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
    - torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
    - torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
    - dgl==1.1.3 -f https://data.dgl.ai/wheels/cu121/repo.html
    - hydra-core>=1.3.0,<2.0.0
    - pyrsistent>=0.20.0
```

**Key Changes:**
- **Python**: 3.9 → 3.11 (for better PyTorch 2.x compatibility)
- **PyTorch**: 1.9 → 2.3.0 (stable version with DGL 1.1.3 compatibility)
- **CUDA Toolkit**: 11.1 → 12.x (using CUDA 12.1 wheels, forward compatible with 12.9)
- **DGL**: cuda11.1 → 1.1.3 (avoids torchdata dependency issues)
- **Installation Method**: Using NVIDIA's conda channel for CUDA toolkit and PyTorch's CUDA 12.1 wheels

### 2. SE3Transformer Requirements (`env/SE3Transformer/requirements.txt`)

**Original Requirements:**
```
e3nn==0.3.3
wandb==0.12.0
pynvml==11.0.0
git+https://github.com/NVIDIA/dllogger#egg=dllogger
decorator==5.1.0
```

**Updated Requirements:**
```
e3nn>=0.5.1
wandb>=0.17.0
pynvml>=11.5.0
git+https://github.com/NVIDIA/dllogger#egg=dllogger
decorator>=5.1.1
```

**Key Changes:**
- **e3nn**: 0.3.3 → ≥0.5.1 (PyTorch 2.x compatibility)
- **wandb**: 0.12.0 → ≥0.17.0 (recent features and bug fixes)
- **pynvml**: 11.0.0 → ≥11.5.0 (better NVIDIA driver compatibility)
- **decorator**: 5.1.0 → ≥5.1.1 (minor updates)

## Compatibility Strategy

### CUDA 12.9 with PyTorch 2.7 Approach
Using the latest available components for optimal performance:
- **CUDA 12.9 toolkit** from NVIDIA's conda channel
- **PyTorch 2.7.0** with CUDA 12.8 support (latest available, forward compatible with 12.9)
- **DGL 2.2.1** which supports CUDA 12.1 (forward compatible)

### Version Selection Rationale
1. **PyTorch 2.7.0**: Latest release with CUDA 12.8 support, includes Blackwell GPU architecture support
2. **CUDA 12.9**: Actual toolkit version matching your system
3. **Python 3.11**: Optimal for PyTorch 2.x performance and compatibility
4. **DGL 2.2.1**: Latest version with CUDA 12 support
5. **Forward compatibility**: PyTorch 2.7's CUDA 12.8 should work with CUDA 12.9 runtime

## Installation Instructions

### 1. Create the Environment
```bash
conda env create -f env/SE3nv_cuda-12.9.yml
conda activate SE3nv_cuda129
```

### 2. Install SE3-Transformer
```bash
cd env/SE3Transformer
pip install --no-cache-dir -r requirements_cuda-12.9.txt
python setup.py install
cd ../..
```

### 3. Install RFdiffusion
```bash
pip install -e .
```

### 4. Download Models
```bash
bash scripts/download_models.sh models/
```

## Potential Issues and Solutions

### 1. CUDA Version Mismatch
**Issue**: PyTorch compiled for CUDA 12.8 running on CUDA 12.9
**Solution**: Forward compatibility should handle this automatically, as CUDA maintains compatibility across minor versions

### 2. DGL Compatibility
**Issue**: DGL 2.2.1 officially supports CUDA 12.1
**Solution**: Modern DGL versions generally work with newer CUDA through runtime compatibility

### 3. SE3-Transformer Compilation
**Issue**: May need compilation against newer CUDA
**Solution**: The updated e3nn version should handle PyTorch 2.x compatibility

### 4. PyTorch 2.7 Breaking Changes
**Issue**: PyTorch 2.7 may have API changes from 1.9
**Solution**: The updated dependencies (e3nn ≥0.5.1) should be compatible with PyTorch 2.x

## Testing Recommendations

After installation, test the environment with:

```bash
# Test PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Test basic RFdiffusion functionality
python scripts/run_inference.py 'contigmap.contigs=[50-50]' inference.output_prefix=test_cuda129 inference.num_designs=1 inference.final_step=2
```

## Fallback Strategy

If the CUDA 12.9 adaptation encounters issues:

1. **Use CUDA 12.8**: Install CUDA 12.8 to match PyTorch 2.7's official support
2. **PyTorch Nightly**: Try PyTorch nightly builds for potential CUDA 12.9 support
3. **Build from Source**: Compile PyTorch and dependencies from source for CUDA 12.9
4. **Use Previous Version**: Fall back to the original CUDA 12.6 approach

## Benefits of This Configuration

- **Latest PyTorch**: Access to PyTorch 2.7 features including Blackwell GPU support
- **Actual CUDA 12.9**: Uses your system's native CUDA toolkit
- **Performance**: Should provide optimal performance on RTX 4090
- **Future-Proof**: Ready for upcoming PyTorch releases with official CUDA 12.9 support

## Notes

- PyTorch 2.7 officially supports CUDA 12.8; 12.9 compatibility through forward compatibility
- Monitor PyTorch and DGL release notes for official CUDA 12.9 support
- Performance testing recommended to ensure optimal utilization of CUDA 12.9 features
- This configuration takes advantage of the latest available packages while maintaining compatibility