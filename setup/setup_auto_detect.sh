#!/bin/bash
# RFdiffusion Auto-Detection Setup Script
# Automatically detects system configuration and creates optimal environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_DIR="$PROJECT_ROOT/env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 RFdiffusion Auto-Detection Setup${NC}"
echo "=================================================="

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check for required tools
check_requirements() {
    print_info "Checking requirements..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    if ! command -v conda &> /dev/null; then
        print_error "Conda is required but not installed"
        exit 1
    fi
    
    print_status "Requirements check passed"
}

# Run system detection
detect_system() {
    print_info "Running system detection..."
    
    cd "$PROJECT_ROOT"
    python3 "$SCRIPT_DIR/detect_system_config.py"
    
    if [ ! -f "system_config.json" ]; then
        print_error "System detection failed - config file not generated"
        exit 1
    fi
    
    print_status "System detection completed"
}

# Parse detection results
parse_config() {
    if ! command -v jq &> /dev/null; then
        print_warning "jq not found - falling back to grep parsing"
        CONFIG_PROFILE=$(grep -o '"config": "[^"]*"' system_config.json | cut -d'"' -f4)
        PYTORCH_VERSION=$(grep -o '"pytorch_version": "[^"]*"' system_config.json | cut -d'"' -f4)
        CUDA_VERSION=$(grep -o '"cuda_version": "[^"]*"' system_config.json | cut -d'"' -f4)
        DGL_VERSION=$(grep -o '"dgl_version": "[^"]*"' system_config.json | cut -d'"' -f4)
        INSTALL_URL=$(grep -o '"install_url": "[^"]*"' system_config.json | cut -d'"' -f4)
        ERROR_MSG=$(grep -o '"error": "[^"]*"' system_config.json | cut -d'"' -f4)
    else
        CONFIG_PROFILE=$(jq -r '.detection_results.recommendations.config' system_config.json)
        PYTORCH_VERSION=$(jq -r '.detection_results.recommendations.pytorch_version' system_config.json)
        CUDA_VERSION=$(jq -r '.detection_results.recommendations.cuda_version' system_config.json)
        DGL_VERSION=$(jq -r '.detection_results.recommendations.dgl_version' system_config.json)
        INSTALL_URL=$(jq -r '.detection_results.recommendations.install_url' system_config.json)
        ERROR_MSG=$(jq -r '.detection_results.recommendations.error' system_config.json)
    fi
    
    # Check for incompatible hardware
    if [[ "$CONFIG_PROFILE" == "incompatible" ]]; then
        print_error "Hardware incompatibility detected!"
        print_error "$ERROR_MSG"
        echo ""
        print_info "Possible solutions:"
        echo "  1. Upgrade CUDA toolkit to minimum required version"
        echo "  2. Use CPU-only mode (much slower)"
        echo "  3. Use a different GPU with compatible CUDA support"
        exit 1
    fi
    
    print_info "Detected configuration: $CONFIG_PROFILE"
    print_info "PyTorch: $PYTORCH_VERSION, CUDA: $CUDA_VERSION, DGL: $DGL_VERSION"
}

# Generate environment file based on detection
generate_environment() {
    ENV_NAME="SE3nv-auto"
    ENV_FILE="$ENV_DIR/${ENV_NAME}.yml"
    
    print_info "Generating environment file: $ENV_FILE"
    
    # Create environment directory if it doesn't exist
    mkdir -p "$ENV_DIR"
    
    cat > "$ENV_FILE" << EOF
# Auto-generated environment for RFdiffusion
# Configuration: $CONFIG_PROFILE
# Generated on: $(date)
name: $ENV_NAME
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults

dependencies:
  - python=3.11
EOF

    # Add CUDA toolkit based on version
    if [[ "$CUDA_VERSION" == "12.8" ]]; then
        cat >> "$ENV_FILE" << EOF
  - nvidia::cuda-toolkit=12.8
  - nvidia::cuda-runtime=12.8
EOF
    elif [[ "$CUDA_VERSION" == "12.1" ]]; then
        cat >> "$ENV_FILE" << EOF
  - nvidia::cuda-toolkit=12.1
  - nvidia::cuda-runtime=12.1
EOF
    elif [[ "$CUDA_VERSION" == "11.6" ]]; then
        cat >> "$ENV_FILE" << EOF
  - nvidia::cuda-toolkit=11.6
  - nvidia::cuda-runtime=11.6
EOF
    elif [[ "$CUDA_VERSION" == "11.3" ]]; then
        cat >> "$ENV_FILE" << EOF
  - nvidia::cuda-toolkit=11.3
  - nvidia::cuda-runtime=11.3
EOF
    fi
    
    # Add build tools if DGL needs source compilation
    if [[ "$DGL_VERSION" == "2.4+" ]] || [[ "$CONFIG_PROFILE" == "bleeding-edge" ]]; then
        cat >> "$ENV_FILE" << EOF
  - cmake>=3.18
  - make
  - cxx-compiler
  - git
EOF
    fi
    
    # Add common dependencies
    cat >> "$ENV_FILE" << EOF
  - numpy
  - scipy
  - matplotlib
  - jupyterlab
  - pip
  
  - pip:
EOF

    # Add PyTorch based on configuration
    if [[ "$CONFIG_PROFILE" == "cpu-only" ]]; then
        cat >> "$ENV_FILE" << EOF
    - torch==$PYTORCH_VERSION --index-url https://download.pytorch.org/whl/cpu
    - torchvision --index-url https://download.pytorch.org/whl/cpu
    - torchaudio --index-url https://download.pytorch.org/whl/cpu
EOF
    else
        CUDA_SUFFIX=$(echo "$CUDA_VERSION" | sed 's/\.//')
        cat >> "$ENV_FILE" << EOF
    - torch==$PYTORCH_VERSION --index-url ${INSTALL_URL}
    - torchvision --index-url ${INSTALL_URL}
    - torchaudio==$PYTORCH_VERSION --index-url ${INSTALL_URL}
EOF
    fi
    
    # Add DGL
    if [[ "$DGL_VERSION" == "2.4+" ]]; then
        cat >> "$ENV_FILE" << EOF
    # DGL 2.4+ will be built from source in post-install
EOF
    else
        cat >> "$ENV_FILE" << EOF
    - dgl==$DGL_VERSION
EOF
    fi
    
    # Add other dependencies
    cat >> "$ENV_FILE" << EOF
    - hydra-core
    - omegaconf
    - biotite
    - py3Dmol
    - modelcif
    - fair-esm
    - opt_einsum
    - icecream
    - biopython
    - pdbfixer
    - openmm
    - e3nn
    - wandb
    - pytest
EOF

    print_status "Environment file generated"
}

# Create conda environment
create_environment() {
    ENV_NAME="SE3nv-auto"
    ENV_FILE="$ENV_DIR/${ENV_NAME}.yml"
    
    print_info "Creating conda environment: $ENV_NAME"
    
    # Remove existing environment if it exists
    if conda env list | grep -q "^$ENV_NAME "; then
        print_warning "Environment $ENV_NAME already exists - removing..."
        conda env remove -n "$ENV_NAME" -y
    fi
    
    # Create new environment
    conda env create -f "$ENV_FILE"
    print_status "Environment created successfully"
}

# Post-installation setup
post_install() {
    ENV_NAME="SE3nv-auto"
    
    print_info "Running post-installation setup..."
    
    # Activate environment for post-install steps
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    
    # Install SE3-Transformer
    if [ -d "$PROJECT_ROOT/env/SE3Transformer" ]; then
        print_info "Installing SE3-Transformer..."
        cd "$PROJECT_ROOT/env/SE3Transformer"
        pip install --no-cache-dir -r requirements.txt
        python setup.py install
        cd "$PROJECT_ROOT"
    else
        print_warning "SE3Transformer directory not found - may need manual installation"
    fi
    
    # Build DGL from source if needed
    if [[ "$DGL_VERSION" == "2.4+" ]] || [[ "$CONFIG_PROFILE" == "bleeding-edge" ]]; then
        print_info "Building DGL from source..."
        bash "$SCRIPT_DIR/upgrade_dgl_source.sh"
    fi
    
    # Install RFdiffusion
    print_info "Installing RFdiffusion..."
    if [[ "$CONFIG_PROFILE" == "bleeding-edge" ]]; then
        pip install -e ".[bleeding-edge]"
    elif [[ "$CONFIG_PROFILE" == *"cuda128"* ]]; then
        pip install -e ".[cuda128-dgl24]"
    else
        pip install -e .
    fi
    
    print_status "Post-installation completed"
}

# Download models
download_models() {
    print_info "Downloading model weights..."
    
    if [ ! -d "$PROJECT_ROOT/models" ]; then
        mkdir -p "$PROJECT_ROOT/models"
    fi
    
    bash "$PROJECT_ROOT/scripts/download_models.sh" "$PROJECT_ROOT/models/"
    print_status "Model download completed"
}

# Run tests
run_tests() {
    ENV_NAME="SE3nv-auto"
    
    print_info "Running verification tests..."
    
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    
    cd "$PROJECT_ROOT"
    
    # Test basic functionality
    print_info "Testing basic imports..."
    python -c "
import torch
import dgl
import rfdiffusion
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    print(f'CUDA version: {torch.version.cuda}')
print(f'DGL: {dgl.__version__}')
print('✅ All imports successful')
"
    
    # Run quick inference test if we have models
    if [ -f "$PROJECT_ROOT/models/Base_ckpt.pt" ]; then
        print_info "Running quick inference test..."
        python scripts/run_inference.py \
            'contigmap.contigs=[50-50]' \
            inference.output_prefix=test_auto_setup \
            inference.num_designs=1 \
            diffuser.T=2 \
            --config-path=../config/inference \
            --config-name=base
        
        if [ -f "test_auto_setup_0.pdb" ]; then
            print_status "Inference test passed"
            rm -f test_auto_setup_*
        else
            print_warning "Inference test produced no output"
        fi
    else
        print_warning "No models found - skipping inference test"
    fi
}

# Print summary
print_summary() {
    ENV_NAME="SE3nv-auto"
    
    echo ""
    echo "=================================================="
    print_status "RFdiffusion Auto-Setup Complete!"
    echo "=================================================="
    echo ""
    print_info "Configuration Summary:"
    echo "  Profile: $CONFIG_PROFILE"
    echo "  PyTorch: $PYTORCH_VERSION"
    echo "  CUDA: $CUDA_VERSION"
    echo "  DGL: $DGL_VERSION"
    echo "  Environment: $ENV_NAME"
    echo ""
    print_info "To activate your environment:"
    echo "  conda activate $ENV_NAME"
    echo ""
    print_info "To run RFdiffusion:"
    echo "  python scripts/run_inference.py 'contigmap.contigs=[100-100]' inference.output_prefix=output"
    echo ""
    
    # Clean up
    rm -f system_config.json
}

# Main execution
main() {
    check_requirements
    detect_system
    parse_config
    generate_environment
    create_environment
    post_install
    
    # Optional steps
    read -p "Download model weights? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        download_models
    fi
    
    read -p "Run verification tests? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_tests
    fi
    
    print_summary
}

# Handle command line arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        --detect-only)
            check_requirements
            detect_system
            parse_config
            echo "Detection complete. Results in system_config.json"
            ;;
        --help|-h)
            echo "RFdiffusion Auto-Detection Setup"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --detect-only    Only run system detection"
            echo "  --help, -h       Show this help message"
            echo ""
            echo "This script automatically detects your GPU and CUDA configuration"
            echo "and creates an optimized conda environment for RFdiffusion."
            ;;
        *)
            main
            ;;
    esac
fi