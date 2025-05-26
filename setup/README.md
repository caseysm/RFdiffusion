# RFdiffusion Setup Scripts

This directory contains the integrated setup system for RFdiffusion with automatic hardware detection and environment optimization.

## Quick Start

```bash
# Complete one-command setup (from project root)
bash setup.sh

# Or from this directory
bash setup_rfdiffusion.sh
```

## Setup Modes

- **`--full`** (default): Complete setup with models and verification tests
- **`--models`**: Setup with models but skip tests
- **`--minimal`**: Environment only, no models or tests

## Options

- **`--no-models`**: Skip model download
- **`--no-tests`**: Skip verification tests  
- **`--force`**: Force reinstall existing environment

## How It Works

1. **Hardware Detection** (`detect_system_config.py`)
   - Detects GPU architecture and compute capability
   - Identifies CUDA toolkit version
   - Uses compatibility matrix for optimal recommendations

2. **Environment Mapping** (`setup_rfdiffusion.sh`)
   - Maps detection results to existing environment files:
     - `bleeding-edge` → `SE3nv-pytorch27-cuda128.yml`
     - `pytorch27-cuda121` → `SE3nv-flexible.yml`
     - `stable/legacy` → `SE3nv.yml`
     - `cpu-only` → `SE3nv-cpu.yml`

3. **Integrated Installation**
   - Creates conda environment using proven configurations
   - Installs SE3-Transformer from source
   - Builds DGL from source when needed (bleeding-edge)
   - Installs RFdiffusion package
   - Downloads model weights
   - Runs verification tests

## Architecture Support

- **Blackwell** (RTX 50xx, B-series): CUDA 12.8+ required
- **Hopper** (H100, H200): CUDA 11.8+ supported
- **Ada Lovelace** (RTX 40xx): CUDA 11.8+ supported
- **Ampere** (RTX 30xx, A100): CUDA 11.0+ supported
- **Older GPUs**: Automatic fallback configurations
- **CPU-only**: Full support for systems without compatible GPUs

## Files

- **`setup_rfdiffusion.sh`**: Main integrated setup script
- **`detect_system_config.py`**: Hardware detection and analysis
- **`cuda_compatibility_matrix.py`**: Comprehensive compatibility matrix
- **`show_cuda_matrix.py`**: Visual compatibility display
- **`download_models.sh`**: Model weight download script
- **`upgrade_dgl_source.sh`**: DGL source compilation for bleeding-edge setups

## Troubleshooting

### Detection Issues
```bash
# Manual detection
python detect_system_config.py

# Show compatibility matrix
python show_cuda_matrix.py --system
```

### Environment Issues
```bash
# Force reinstall
bash setup_rfdiffusion.sh --force

# Minimal install only
bash setup_rfdiffusion.sh --minimal
```

### Manual Fallback
If auto-detection fails, you can manually create environments:
```bash
# For modern hardware
conda env create -f ../../env/SE3nv-flexible.yml

# For stable compatibility  
conda env create -f ../../env/SE3nv.yml

# For CPU-only
conda env create -f ../../env/SE3nv-cpu.yml
```

## Legacy Scripts

The following scripts are preserved for advanced users:
- **`setup_auto_detect.sh`**: Original auto-detection setup
- **`setup_pytorch27_cuda128.sh`**: Specific bleeding-edge setup

For most users, the new integrated `setup_rfdiffusion.sh` script is recommended.