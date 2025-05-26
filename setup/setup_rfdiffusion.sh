#!/bin/bash
# RFdiffusion Environment Setup Script
# Creates conda environment and installs SE3-Transformer based on provided parameters

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Parameters (will be set by command line args)
ENV_FILE=""
ENV_NAME=""
CONFIG_PROFILE=""
PYTORCH_VERSION=""
CUDA_VERSION=""
DGL_VERSION=""
BUILD_DGL_SOURCE=false
FORCE_REINSTALL=false
SETUP_MODE="full"
IS_FALLBACK=false

print_step() {
    echo -e "${PURPLE}[ENV-SETUP]${NC} $1"
}

print_status() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING:  $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_info() {
    echo -e "${BLUE}INFO:  $1${NC}"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env-file)
                ENV_FILE="$2"
                shift 2
                ;;
            --env-name)
                ENV_NAME="$2"
                shift 2
                ;;
            --config-profile)
                CONFIG_PROFILE="$2"
                shift 2
                ;;
            --pytorch-version)
                PYTORCH_VERSION="$2"
                shift 2
                ;;
            --cuda-version)
                CUDA_VERSION="$2"
                shift 2
                ;;
            --dgl-version)
                DGL_VERSION="$2"
                shift 2
                ;;
            --build-dgl-source)
                BUILD_DGL_SOURCE=true
                shift
                ;;
            --force)
                FORCE_REINSTALL=true
                shift
                ;;
            --mode)
                SETUP_MODE="$2"
                shift 2
                ;;
            --fallback)
                IS_FALLBACK=true
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

print_usage() {
    echo "RFdiffusion Environment Setup Script"
    echo ""
    echo "Usage: $0 --env-file FILE --env-name NAME [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --env-file FILE       Path to conda environment YAML file"
    echo "  --env-name NAME       Name for the conda environment"
    echo ""
    echo "Optional:"
    echo "  --config-profile STR  Configuration profile name"
    echo "  --pytorch-version STR PyTorch version being installed"
    echo "  --cuda-version STR    CUDA version being used"
    echo "  --dgl-version STR     DGL version being installed"
    echo "  --build-dgl-source    Build DGL from source (for bleeding-edge)"
    echo "  --force               Force reinstall existing environment"
    echo "  --mode MODE           Setup mode: full|models|minimal"
    echo "  --fallback            Indicates this is a fallback configuration"
    echo ""
    echo "This script is typically called by setup.sh with auto-detected parameters."
}

# Validate required parameters
validate_params() {
    if [[ -z "$ENV_FILE" ]]; then
        print_error "Environment file not specified (--env-file required)"
        exit 1
    fi
    
    if [[ -z "$ENV_NAME" ]]; then
        print_error "Environment name not specified (--env-name required)"
        exit 1
    fi
    
    if [ ! -f "$PROJECT_ROOT/$ENV_FILE" ]; then
        print_error "Environment file not found: $PROJECT_ROOT/$ENV_FILE"
        exit 1
    fi
}

# Display configuration
show_configuration() {
    print_step "Environment Setup Configuration"
    
    echo "  Environment file: $ENV_FILE"
    echo "  Environment name: $ENV_NAME"
    echo "  Configuration: ${CONFIG_PROFILE:-unknown}"
    
    if [[ -n "$PYTORCH_VERSION" ]]; then
        echo "  PyTorch: $PYTORCH_VERSION"
    fi
    if [[ -n "$CUDA_VERSION" ]]; then
        echo "  CUDA: $CUDA_VERSION"
    fi
    if [[ -n "$DGL_VERSION" ]]; then
        echo "  DGL: $DGL_VERSION"
    fi
    
    echo "  Mode: $SETUP_MODE"
    
    if [[ "$BUILD_DGL_SOURCE" == "true" ]]; then
        echo "  DGL: Build from source"
    fi
    
    if [[ "$IS_FALLBACK" == "true" ]]; then
        echo "  Note: Using fallback configuration"
    fi
    
    echo ""
}

# Create conda environment
create_environment() {
    print_step "Creating conda environment: $ENV_NAME"
    
    # Check if environment already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        if [[ "$FORCE_REINSTALL" == "true" ]]; then
            print_warning "Environment $ENV_NAME already exists - removing for reinstall..."
            conda env remove -n "$ENV_NAME" -y
        else
            print_warning "Environment $ENV_NAME already exists"
            read -p "Remove and recreate? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                conda env remove -n "$ENV_NAME" -y
            else
                print_info "Using existing environment"
                return 0
            fi
        fi
    fi
    
    print_info "Creating environment from $ENV_FILE"
    
    # Create environment
    if conda env create -f "$PROJECT_ROOT/$ENV_FILE"; then
        print_status "Environment $ENV_NAME created successfully"
    else
        print_error "Failed to create environment"
        exit 1
    fi
}

# Install SE3-Transformer
install_se3_transformer() {
    print_step "Installing SE3-Transformer"
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    
    if [ -d "$PROJECT_ROOT/env/SE3Transformer" ]; then
        print_info "Installing SE3-Transformer from source"
        cd "$PROJECT_ROOT/env/SE3Transformer"
        
        # Use appropriate requirements file based on environment
        if [[ "$ENV_NAME" == *"flexible"* ]] || [[ "$ENV_NAME" == *"pytorch27"* ]]; then
            REQUIREMENTS_FILE="requirements-flexible.txt"
        else
            REQUIREMENTS_FILE="requirements.txt"
        fi
        
        if [ -f "$REQUIREMENTS_FILE" ]; then
            print_info "Installing requirements from $REQUIREMENTS_FILE"
            pip install --no-cache-dir -r "$REQUIREMENTS_FILE"
        else
            print_warning "Requirements file $REQUIREMENTS_FILE not found, using default"
            pip install --no-cache-dir -r requirements.txt
        fi
        
        python setup.py install
        cd "$PROJECT_ROOT"
        print_status "SE3-Transformer installed"
    else
        print_warning "SE3Transformer directory not found at $PROJECT_ROOT/env/SE3Transformer"
        print_info "You may need to install SE3-Transformer manually"
    fi
}

# Build DGL from source if needed
build_dgl_source() {
    if [[ "$BUILD_DGL_SOURCE" != "true" ]]; then
        return 0
    fi
    
    print_step "Building DGL from source"
    
    # Ensure we're in the right environment
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    
    if [ -f "$SCRIPT_DIR/upgrade_dgl_source.sh" ]; then
        print_info "Building DGL from source (this may take 10-15 minutes)"
        bash "$SCRIPT_DIR/upgrade_dgl_source.sh"
        print_status "DGL built from source"
    else
        print_warning "DGL source build script not found"
        print_info "Skipping DGL source build"
    fi
}

# Download models if requested
download_models() {
    if [[ "$SETUP_MODE" == "minimal" ]]; then
        print_info "Skipping model download (minimal mode)"
        return 0
    fi
    
    print_step "Downloading model weights"
    
    if [ ! -d "$PROJECT_ROOT/models" ]; then
        mkdir -p "$PROJECT_ROOT/models"
    fi
    
    if [ -f "$SCRIPT_DIR/download_models.sh" ]; then
        print_info "Downloading models (this may take 5-10 minutes)"
        if bash "$SCRIPT_DIR/download_models.sh" "$PROJECT_ROOT/models/"; then
            print_status "Model weights downloaded"
        else
            print_warning "Model download failed - you can download manually later"
        fi
    else
        print_warning "Model download script not found"
        print_info "You can download models manually later"
    fi
}

# Run basic verification
run_verification() {
    print_step "Running basic verification"
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    
    # Test basic imports
    print_info "Testing basic imports..."
    python -c "
try:
    import torch
    print(f'SUCCESS: PyTorch: {torch.__version__}')
    print(f'SUCCESS: CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'SUCCESS: CUDA devices: {torch.cuda.device_count()}')
        print(f'SUCCESS: CUDA version: {torch.version.cuda}')
except ImportError as e:
    print(f'ERROR: PyTorch import failed: {e}')
    exit(1)

try:
    import dgl
    print(f'SUCCESS: DGL: {dgl.__version__}')
except ImportError as e:
    print(f'ERROR: DGL import failed: {e}')
    exit(1)

try:
    import e3nn
    print(f'SUCCESS: E3NN: {e3nn.__version__}')
except ImportError as e:
    print(f'WARNING:  E3NN import failed: {e}')

print('SUCCESS: Environment verification completed')
" || {
        print_error "Environment verification failed"
        return 1
    }
    
    print_status "Environment verification passed"
}

# Print summary
print_summary() {
    echo ""
    print_status "Environment setup completed successfully!"
    echo ""
    print_info "Environment details:"
    echo "  Name: $ENV_NAME"
    echo "  Configuration: ${CONFIG_PROFILE:-unknown}"
    echo "  Mode: $SETUP_MODE"
    echo ""
    print_info "To use this environment:"
    echo "  conda activate $ENV_NAME"
    echo ""
}

# Main execution
main() {
    validate_params
    show_configuration
    create_environment
    install_se3_transformer
    build_dgl_source
    download_models
    run_verification
    print_summary
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_args "$@"
    main
fi