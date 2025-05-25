# Legacy Environment Configurations

This directory contains legacy environment configurations for older CUDA versions.

## Available Configurations:

### CUDA 11.1 (`cuda-11.1/`)
- **Environment**: `SE3nv.yml` 
- **Requirements**: `SE3Transformer/requirements.txt`
- **Setup**: `setup.py`
- **PyTorch**: 1.9
- **Python**: 3.9
- **DGL**: < 2.0.0

## Usage:

To use a legacy environment:

```bash
# CUDA 11.1
conda env create -f env/legacy/cuda-11.1/SE3nv.yml
conda activate SE3nv
cd env/SE3Transformer
pip install --no-cache-dir -r ../../legacy/cuda-11.1/SE3Transformer/requirements.txt
python setup.py install
cd ../..
cp env/legacy/cuda-11.1/setup.py .
pip install -e .
```

## Note:

For new installations, use the default files in the root `env/` directory, which provide CUDA 12.x compatibility with modern package versions.