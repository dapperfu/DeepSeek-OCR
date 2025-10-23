#!/bin/bash
# DeepSeek-OCR Setup Script
# Automated installation script for vanilla Python virtual environment
# Migrated from Conda to venv

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.12.9"
VENV_DIR="venv"
VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl"
VLLM_WHEEL_FILE="vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl"

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.12.9 is available
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION_CHECK=$(python3 --version | cut -d' ' -f2)
        if [[ "$PYTHON_VERSION_CHECK" == "3.12"* ]]; then
            PYTHON_CMD="python3"
        else
            print_error "Python 3.12.x is required, found $PYTHON_VERSION_CHECK"
            exit 1
        fi
    else
        print_error "Python 3.12.x not found. Please install Python 3.12.9"
        exit 1
    fi
    
    print_success "Found Python: $($PYTHON_CMD --version)"
}

# Check if CUDA is available
check_cuda() {
    print_status "Checking CUDA installation..."
    
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/')
        print_success "Found CUDA: $CUDA_VERSION"
        
        if [[ "$CUDA_VERSION" != "11.8" ]]; then
            print_warning "CUDA 11.8 is recommended, found $CUDA_VERSION"
        fi
    else
        print_warning "nvidia-smi not found. CUDA may not be properly installed"
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf "$VENV_DIR"
    fi
    
    $PYTHON_CMD -m venv "$VENV_DIR"
    print_success "Virtual environment created at ./$VENV_DIR"
}

# Install base dependencies
install_base_deps() {
    print_status "Installing base dependencies..."
    
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r requirements.txt
    
    print_success "Base dependencies installed"
}

# Install PyTorch with CUDA support
install_torch() {
    print_status "Installing PyTorch 2.6.0 with CUDA 11.8 support..."
    
    "$VENV_DIR/bin/pip" install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu118
    
    print_success "PyTorch installed"
}

# Download and install vLLM
install_vllm() {
    print_status "Installing vLLM 0.8.5..."
    
    if [ ! -f "$VLLM_WHEEL_FILE" ]; then
        print_status "Downloading vLLM wheel..."
        if command -v wget &> /dev/null; then
            wget "$VLLM_WHEEL_URL"
        elif command -v curl &> /dev/null; then
            curl -L -O "$VLLM_WHEEL_URL"
        else
            print_error "Neither wget nor curl found. Please install one of them or download the wheel manually"
            exit 1
        fi
    fi
    
    "$VENV_DIR/bin/pip" install "$VLLM_WHEEL_FILE"
    print_success "vLLM installed"
}

# Install flash-attention
install_flash_attn() {
    print_status "Installing flash-attention 2.7.3..."
    
    "$VENV_DIR/bin/pip" install flash-attn==2.7.3 --no-build-isolation
    
    print_success "Flash-attention installed"
}

# Install development dependencies (optional)
install_dev_deps() {
    if [ -f "requirements-dev.txt" ]; then
        print_status "Installing development dependencies..."
        "$VENV_DIR/bin/pip" install -r requirements-dev.txt
        print_success "Development dependencies installed"
    fi
}

# Main installation function
main() {
    echo "=========================================="
    echo "DeepSeek-OCR Setup Script"
    echo "Migrating from Conda to vanilla Python"
    echo "=========================================="
    echo ""
    
    # Pre-installation checks
    check_python
    check_cuda
    
    # Installation steps
    create_venv
    install_base_deps
    install_torch
    install_vllm
    install_flash_attn
    
    # Optional development dependencies
    if [ "$1" = "--dev" ]; then
        install_dev_deps
    fi
    
    # Cleanup
    if [ -f "$VLLM_WHEEL_FILE" ]; then
        print_status "Cleaning up downloaded wheel..."
        rm "$VLLM_WHEEL_FILE"
    fi
    
    echo ""
    echo "=========================================="
    print_success "Installation complete!"
    echo "=========================================="
    echo ""
    echo "To activate the environment:"
    echo "  source $VENV_DIR/bin/activate"
    echo ""
    echo "To run DeepSeek-OCR:"
    echo "  cd DeepSeek-OCR-master/DeepSeek-OCR-vllm"
    echo "  python run_dpsk_ocr_image.py"
    echo ""
    echo "For development setup, run:"
    echo "  ./setup.sh --dev"
    echo ""
}

# Handle command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [--dev]"
    echo ""
    echo "Options:"
    echo "  --dev    Also install development dependencies"
    echo "  --help   Show this help message"
    echo ""
    exit 0
fi

# Run main function
main "$@"
