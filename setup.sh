#!/bin/bash
# RFdiffusion Setup - Main orchestrator
# Detects platform → gets setup commands → executes installation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Default options
SKIP_DETECTION=false
SETUP_MODE="full"
FORCE_REINSTALL=false

print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE} RFdiffusion Setup${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                SETUP_MODE="full"
                shift
                ;;
            --minimal)
                SETUP_MODE="minimal"
                shift
                ;;
            --models)
                SETUP_MODE="models"
                shift
                ;;
            --force)
                FORCE_REINSTALL=true
                shift
                ;;
            --skip-detection)
                SKIP_DETECTION=true
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
    echo "RFdiffusion Setup"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Setup Modes:"
    echo "  --full      Complete setup: environment + packages + models + tests (default)"
    echo "  --models    Setup with models but skip tests"
    echo "  --minimal   Environment and packages only"
    echo ""
    echo "Options:"
    echo "  --force           Force reinstall existing environment"
    echo "  --skip-detection  Skip hardware detection, use stable config"
    echo "  --help, -h        Show this help message"
    echo ""
    echo "This script:"
    echo "  1. Detects your hardware and recommends optimal configuration"
    echo "  2. Creates conda environment using appropriate config"
    echo "  3. Installs RFdiffusion package and dependencies"
    echo "  4. Downloads models and runs tests (unless minimal mode)"
}

# Check basic requirements
check_requirements() {
    print_step "Checking requirements"
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    if ! command -v conda &> /dev/null; then
        print_error "Conda is required but not installed"
        print_info "Please install Anaconda or Miniconda: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    print_status "Requirements check passed"
}

# Run detection and get setup commands
detect_and_get_commands() {
    if [[ "$SKIP_DETECTION" == "true" ]]; then
        print_step "Skipping detection, using stable configuration"
        
        # Use stable defaults
        SETUP_COMMANDS=(
            "--env-file" "env/SE3nv.yml"
            "--env-name" "SE3nv"
            "--config-profile" "stable"
            "--mode" "$SETUP_MODE"
        )
        
        if [[ "$FORCE_REINSTALL" == "true" ]]; then
            SETUP_COMMANDS+=("--force")
        fi
        
        print_info "Using: SE3nv environment (stable configuration)"
        return 0
    fi
    
    print_step "Detecting system configuration"
    
    cd "$PROJECT_ROOT"
    
    # Run detection and export results
    if ! python3 setup/detect_system_config.py --export-config > /dev/null 2>&1; then
        print_error "System detection failed, falling back to stable configuration"
        SETUP_COMMANDS=(
            "--env-file" "env/SE3nv.yml"
            "--env-name" "SE3nv"
            "--config-profile" "stable"
            "--fallback" "true"
            "--mode" "$SETUP_MODE"
        )
        if [[ "$FORCE_REINSTALL" == "true" ]]; then
            SETUP_COMMANDS+=("--force")
        fi
        return 0
    fi
    
    if [ ! -f "system_config.json" ]; then
        print_error "Detection completed but no config file generated, using stable fallback"
        SETUP_COMMANDS=(
            "--env-file" "env/SE3nv.yml"
            "--env-name" "SE3nv"
            "--config-profile" "stable"
            "--fallback" "true"
            "--mode" "$SETUP_MODE"
        )
        if [[ "$FORCE_REINSTALL" == "true" ]]; then
            SETUP_COMMANDS+=("--force")
        fi
        return 0
    fi
    
    # Parse detection results and convert to setup commands
    parse_detection_results
    
    print_status "System detection completed"
}

# Parse detection results into setup commands
parse_detection_results() {
    # Extract config using grep (no jq dependency)
    CONFIG_PROFILE=$(grep -o '"config": "[^"]*"' system_config.json | cut -d'"' -f4)
    PYTORCH_VERSION=$(grep -o '"pytorch_version": "[^"]*"' system_config.json | cut -d'"' -f4)
    CUDA_VERSION=$(grep -o '"cuda_version": "[^"]*"' system_config.json | cut -d'"' -f4)
    DGL_VERSION=$(grep -o '"dgl_version": "[^"]*"' system_config.json | cut -d'"' -f4)
    REASON=$(grep -o '"reason": "[^"]*"' system_config.json | cut -d'"' -f4)
    
    # Check for errors
    ERROR_MSG=$(grep -o '"error": "[^"]*"' system_config.json 2>/dev/null | cut -d'"' -f4 || echo "")
    
    if [[ "$CONFIG_PROFILE" == "incompatible" ]] || [[ -n "$ERROR_MSG" ]]; then
        print_error "Hardware incompatibility detected!"
        print_error "${ERROR_MSG:-$REASON}"
        echo ""
        print_info "Possible solutions:"
        echo "  1. Upgrade CUDA toolkit to minimum required version"
        echo "  2. Use stable fallback: $0 --skip-detection"
        echo "  3. Use a different GPU with compatible CUDA support"
        exit 1
    fi
    
    # Map config profile to environment files and setup commands
    case "$CONFIG_PROFILE" in
        "bleeding-edge")
            ENV_FILE="env/SE3nv-pytorch27-cuda128.yml"
            ENV_NAME="SE3nv-pytorch27-cuda128"
            NEEDS_DGL_SOURCE="true"
            ;;
        "pytorch27-cuda121")
            ENV_FILE="env/SE3nv-flexible.yml"
            ENV_NAME="SE3nv-flexible"
            NEEDS_DGL_SOURCE="false"
            ;;
        "stable")
            ENV_FILE="env/SE3nv.yml"
            ENV_NAME="SE3nv"
            NEEDS_DGL_SOURCE="false"
            ;;
        "legacy")
            ENV_FILE="env/SE3nv.yml"
            ENV_NAME="SE3nv"
            NEEDS_DGL_SOURCE="false"
            ;;
        "cpu-only")
            ENV_FILE="env/SE3nv-cpu.yml"
            ENV_NAME="SE3nv-cpu"
            NEEDS_DGL_SOURCE="false"
            ;;
        *)
            print_error "Unknown config profile '$CONFIG_PROFILE', using stable fallback"
            ENV_FILE="env/SE3nv.yml"
            ENV_NAME="SE3nv"
            NEEDS_DGL_SOURCE="false"
            CONFIG_PROFILE="stable"
            ;;
    esac
    
    # Build setup commands array
    SETUP_COMMANDS=(
        "--env-file" "$ENV_FILE"
        "--env-name" "$ENV_NAME"
        "--config-profile" "$CONFIG_PROFILE"
        "--pytorch-version" "$PYTORCH_VERSION"
        "--cuda-version" "$CUDA_VERSION"
        "--dgl-version" "$DGL_VERSION"
    )
    
    if [[ "$NEEDS_DGL_SOURCE" == "true" ]]; then
        SETUP_COMMANDS+=("--build-dgl-source")
    fi
    
    if [[ "$FORCE_REINSTALL" == "true" ]]; then
        SETUP_COMMANDS+=("--force")
    fi
    
    # Add setup mode
    SETUP_COMMANDS+=("--mode" "$SETUP_MODE")
    
    print_info "Detected configuration: $CONFIG_PROFILE"
    print_info "Environment: $ENV_NAME"
    print_info "PyTorch: $PYTORCH_VERSION, CUDA: $CUDA_VERSION, DGL: $DGL_VERSION"
    print_info "Reason: $REASON"
}

# Execute the setup using our commands
execute_setup() {
    print_step "Executing environment setup"
    
    print_info "Setup commands: ${SETUP_COMMANDS[*]}"
    
    # Call the detailed setup script with our computed parameters
    if bash "$PROJECT_ROOT/setup/setup_rfdiffusion.sh" "${SETUP_COMMANDS[@]}"; then
        print_status "Environment setup completed"
    else
        print_error "Environment setup failed"
        exit 1
    fi
}

# Install RFdiffusion package in the created environment
install_rfdiffusion_package() {
    print_step "Installing RFdiffusion package"
    
    # Extract environment name from our commands
    ENV_NAME=""
    for i in "${!SETUP_COMMANDS[@]}"; do
        if [[ "${SETUP_COMMANDS[$i]}" == "--env-name" ]]; then
            ENV_NAME="${SETUP_COMMANDS[$((i+1))]}"
            break
        fi
    done
    
    if [[ -z "$ENV_NAME" ]]; then
        print_error "Could not determine environment name"
        exit 1
    fi
    
    # Activate environment and install package
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    
    print_info "Installing RFdiffusion package in environment: $ENV_NAME"
    
    if python setup.py install; then
        print_status "RFdiffusion package installed"
    else
        print_error "RFdiffusion package installation failed"
        exit 1
    fi
}

# Run tests in the environment
run_tests() {
    if [[ "$SETUP_MODE" == "minimal" ]]; then
        print_info "Skipping tests (minimal mode)"
        return 0
    fi
    
    print_step "Running verification tests"
    
    # Environment should already be activated
    cd "$PROJECT_ROOT"
    
    # Run basic import test
    print_info "Testing imports..."
    python -c "
import torch
import dgl  
import rfdiffusion
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA devices: {torch.cuda.device_count()}')
print(f'✅ DGL: {dgl.__version__}')
print('✅ All imports successful')
" || {
        print_error "Import test failed"
        return 1
    }
    
    # Run pytest if available and not minimal mode
    if [[ "$SETUP_MODE" == "full" ]] && command -v pytest &> /dev/null; then
        print_info "Running pytest suite..."
        python -m pytest tests/ -v --tb=short || {
            print_error "Some tests failed, but installation appears complete"
            return 0  # Don't fail setup for test failures
        }
    fi
    
    print_status "Verification tests completed"
}

# Print final summary
print_summary() {
    echo ""
    echo -e "${GREEN}================================================================${NC}"
    echo -e "${GREEN} RFdiffusion Setup Complete! 🎉${NC}"
    echo -e "${GREEN}================================================================${NC}"
    echo ""
    
    # Extract environment name for summary
    ENV_NAME=""
    for i in "${!SETUP_COMMANDS[@]}"; do
        if [[ "${SETUP_COMMANDS[$i]}" == "--env-name" ]]; then
            ENV_NAME="${SETUP_COMMANDS[$((i+1))]}"
            break
        fi
    done
    
    print_info "Next steps:"
    echo -e "${YELLOW}  # Activate environment${NC}"
    echo "  conda activate ${ENV_NAME:-SE3nv}"
    echo ""
    echo -e "${YELLOW}  # Run RFdiffusion${NC}"
    echo "  python inference/run_inference.py 'contigmap.contigs=[100-100]' inference.output_prefix=my_design"
    echo ""
    
    print_info "For help and examples:"
    echo "  - Check examples/ directory for usage examples"
    echo "  - Read README.md for detailed documentation"
    echo "  - Visit https://github.com/RosettaCommons/RFdiffusion"
    
    # Clean up detection results
    rm -f system_config.json
}

# Main execution
main() {
    print_header
    
    check_requirements
    detect_and_get_commands
    execute_setup
    install_rfdiffusion_package
    run_tests
    print_summary
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_args "$@"
    main
fi