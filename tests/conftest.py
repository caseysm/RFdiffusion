#!/usr/bin/env python3
"""
pytest configuration for auto-detection tests
"""

import pytest
import sys
import os
from pathlib import Path

def pytest_configure(config):
    """Configure pytest environment"""
    # Add scripts setup directory to path for setup tests
    scripts_setup_dir = Path(__file__).parent.parent / "scripts" / "setup"
    sys.path.insert(0, str(scripts_setup_dir))
    
    # Add scripts inference directory to path for inference tests
    scripts_inference_dir = Path(__file__).parent.parent / "scripts" / "inference"
    sys.path.insert(0, str(scripts_inference_dir))
    
    # Add tests directory to path  
    tests_dir = Path(__file__).parent
    sys.path.insert(0, str(tests_dir))

@pytest.fixture
def mock_nvidia_smi_rtx4090():
    """Fixture for mocking RTX 4090 nvidia-smi output"""
    return "NVIDIA GeForce RTX 4090, 8.9, 24576, 535.86.05"

@pytest.fixture  
def mock_nvidia_smi_a100():
    """Fixture for mocking A100 nvidia-smi output"""
    return "NVIDIA A100-SXM4-80GB, 8.0, 81920, 525.85.12"

@pytest.fixture
def mock_nvidia_smi_rtx3080():
    """Fixture for mocking RTX 3080 nvidia-smi output"""
    return "NVIDIA GeForce RTX 3080, 8.6, 10240, 470.86"

@pytest.fixture
def mock_cuda_128():
    """Fixture for mocking CUDA 12.8 output"""
    return {
        'nvcc': "Cuda compilation tools, release 12.8, V12.8.128",
        'nvidia_smi': "CUDA Version: 12.8"
    }

@pytest.fixture
def mock_cuda_121():
    """Fixture for mocking CUDA 12.1 output"""
    return {
        'nvcc': "Cuda compilation tools, release 12.1, V12.1.66", 
        'nvidia_smi': "CUDA Version: 12.1"
    }

@pytest.fixture
def mock_cuda_116():
    """Fixture for mocking CUDA 11.6 output"""
    return {
        'nvcc': "Cuda compilation tools, release 11.6, V11.6.124",
        'nvidia_smi': "CUDA Version: 11.6"
    }