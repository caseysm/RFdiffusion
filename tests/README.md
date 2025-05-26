# Tests Directory

This directory contains comprehensive tests for the RFdiffusion project, organized by functionality.

## Directory Structure

```
tests/
├── README.md                     # This documentation
├── conftest.py                   # Pytest configuration and fixtures
├── requirements-test.txt         # Testing dependencies
├── run_detection_tests.py        # Test runner for setup tests
├── setup/                        # Setup and environment detection tests
│   ├── __init__.py
│   ├── test_detect_system_config.py
│   ├── test_cuda_compatibility_matrix.py
│   ├── test_show_cuda_matrix.py
│   └── test_detection_integration.py
└── inference/                    # RFdiffusion inference tests
    ├── __init__.py
    └── test_diffusion.py
```

## Test Categories

### Setup Tests (`tests/setup/`)

Tests for system detection, environment setup, and CUDA compatibility:

- **test_detect_system_config.py**: Tests the automatic system detection functionality
  - GPU detection via nvidia-smi
  - CUDA toolkit detection
  - PyTorch version recommendations
  - Memory warnings and compatibility checks
  - Configuration export functionality

- **test_cuda_compatibility_matrix.py**: Tests the CUDA compatibility matrix
  - Compatibility level determination
  - Architecture mapping (Hopper, Ada Lovelace, Ampere, etc.)
  - Version-specific recommendations
  - Edge case handling for unknown hardware

- **test_show_cuda_matrix.py**: Tests the matrix display CLI tool
  - Command-line argument parsing
  - System detection integration
  - Matrix visualization options
  - Error handling for detection failures

- **test_detection_integration.py**: Integration tests for complete workflows
  - End-to-end detection scenarios
  - Multiple GPU handling
  - Real-world hardware combinations
  - Configuration export integration

### Inference Tests (`tests/inference/`)

Tests for RFdiffusion model inference and generation:

- **test_diffusion.py**: Core diffusion model tests
  - Example command execution in deterministic mode
  - Output validation against reference structures
  - RMSD comparison for reproducibility
  - Integration with example scripts

## Running Tests

### Prerequisites

Install testing dependencies:
```bash
pip install -r requirements-test.txt
```

### Running All Setup Tests

```bash
# Using the test runner
python tests/run_detection_tests.py

# Using pytest
python -m pytest tests/setup/ -v

# Using unittest
python -m unittest discover tests/setup -v
```

### Running Individual Test Files

```bash
# Specific test module
python -m pytest tests/setup/test_detect_system_config.py -v

# Specific test class
python -m pytest tests/setup/test_detect_system_config.py::TestSystemDetector -v

# Specific test method
python -m pytest tests/setup/test_detect_system_config.py::TestSystemDetector::test_detect_nvidia_gpu_success -v
```

### Running Inference Tests

```bash
# Diffusion tests (requires GPU and models)
python -m pytest tests/inference/test_diffusion.py -v

# Run with specific hardware requirements
python -m pytest tests/inference/ -v -m "not slow"
```

### Coverage Reports

```bash
# Generate coverage report
python -m pytest tests/ --cov=rfdiffusion --cov=scripts --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Test Configuration

### Environment Variables

Tests can be configured with environment variables:

```bash
export RFDIFFUSION_TEST_GPU=0          # Skip GPU tests
export RFDIFFUSION_TEST_TIMEOUT=300    # Test timeout in seconds
export RFDIFFUSION_MODELS_PATH=/path   # Model checkpoint directory
```

### Pytest Configuration

Key pytest settings in `conftest.py`:
- Automatic path setup for script imports
- Mock fixtures for common hardware configurations
- Shared test utilities and helpers

### Mock Data

Tests use comprehensive mocking for:
- nvidia-smi output for various GPU models
- CUDA toolkit version detection
- Conda package listings
- System command outputs

## Hardware Coverage

Tests cover a wide range of hardware configurations:

### Modern GPUs
- **H100**: Hopper architecture, compute capability 9.0
- **RTX 4090**: Ada Lovelace, compute capability 8.9
- **A100**: Ampere, compute capability 8.0
- **RTX 3080**: Ampere, compute capability 8.6

### Legacy GPUs  
- **V100**: Volta, compute capability 7.0
- **RTX 2080**: Turing, compute capability 7.5
- **GTX 1080**: Pascal, compute capability 6.1
- **GTX 980**: Maxwell, compute capability 5.2

### CUDA Versions
- CUDA 12.8 (bleeding-edge)
- CUDA 12.1 (modern)
- CUDA 11.6 (stable)
- CUDA 11.3 (legacy)
- CUDA 10.0 (minimum)

## Expected Test Results

### Setup Tests
- **~50 test methods** across 4 test files
- **Runtime**: ~10-30 seconds (mocked, no actual GPU calls)
- **Coverage**: System detection, compatibility matrix, CLI tools

### Inference Tests
- **~10-15 test methods** for diffusion functionality
- **Runtime**: ~5-10 minutes (requires GPU and model weights)
- **Coverage**: Example script execution, output validation

## Troubleshooting

### Import Errors
If tests fail with import errors:
```bash
# Ensure you're in the project root
cd /path/to/RFdiffusion

# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"

# Install in development mode
pip install -e .
```

### GPU Tests
For GPU-dependent tests:
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
nvcc --version

# Download required models
bash setup/download_models.sh models/
```

### Mock Failures
If mocking fails in setup tests:
```bash
# Check subprocess mock compatibility
python -c "from unittest.mock import patch; print('Mock available')"

# Verify test isolation
python -m pytest tests/setup/test_detect_system_config.py::TestSystemDetector::test_detect_nvidia_gpu_success -v -s
```

## Contributing

When adding new tests:

1. **Setup tests**: Place in `tests/setup/` for environment, detection, or compatibility functionality
2. **Inference tests**: Place in `tests/inference/` for RFdiffusion model functionality
3. **Follow naming**: Use `test_*.py` pattern for test files
4. **Add documentation**: Update this README for new test categories
5. **Include mocks**: Use comprehensive mocking for system dependencies
6. **Test isolation**: Ensure tests don't depend on specific hardware
7. **Coverage**: Aim for >90% coverage on new functionality

### Test Template

```python
#!/usr/bin/env python3
"""
Tests for new_functionality.py
"""

import unittest
from unittest.mock import patch, Mock
import sys
from pathlib import Path

# Add appropriate scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent / "scripts" / "setup"  # or "inference"
sys.path.insert(0, str(scripts_dir))

from new_functionality import NewClass

class TestNewClass(unittest.TestCase):
    
    def setUp(self):
        self.instance = NewClass()
    
    def test_basic_functionality(self):
        """Test basic functionality with clear description"""
        result = self.instance.method()
        self.assertEqual(result, expected_value)
    
    @patch('subprocess.run')
    def test_with_mocking(self, mock_run):
        """Test with mocked external dependencies"""
        mock_run.return_value.stdout = "expected output"
        mock_run.return_value.returncode = 0
        
        result = self.instance.method_with_subprocess()
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
```

## Related Documentation

- [CLAUDE.md](../CLAUDE.md): Main project instructions and setup commands
- [setup/README.md](../setup/README.md): Setup script organization and usage
- [inference/](../inference/): Inference script organization
- [examples/](../examples/): Example usage scripts tested by inference tests