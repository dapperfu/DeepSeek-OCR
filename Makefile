# DeepSeek-OCR Makefile
# Migrated from Conda to vanilla Python virtual environment

.PHONY: help venv install-base install-torch install-vllm install-flash-attn install-all clean test lint

# Default target
help:
	@echo "DeepSeek-OCR Build System"
	@echo "========================"
	@echo ""
	@echo "Available targets:"
	@echo "  venv              Create Python virtual environment"
	@echo "  install-base      Install base dependencies from requirements.txt"
	@echo "  install-torch     Install PyTorch 2.6.0 with CUDA 11.8 support"
	@echo "  install-vllm      Download and install vLLM 0.8.5 wheel"
	@echo "  install-flash-attn Install flash-attention 2.7.3"
	@echo "  install-all       Complete installation chain"
	@echo "  install-dev       Install development dependencies"
	@echo "  clean             Remove venv and temporary files"
	@echo "  test              Run tests"
	@echo "  lint              Run type checking with mypy"
	@echo ""
	@echo "Quick start: make install-all"

# Virtual environment setup
venv:
	@echo "Creating Python virtual environment..."
	python3.12 -m venv venv
	@echo "Virtual environment created at ./venv"
	@echo "Activate with: source venv/bin/activate"

# Install base dependencies
install-base: venv
	@echo "Installing base dependencies..."
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt
	@echo "Base dependencies installed"

# Install PyTorch with CUDA support
install-torch: venv
	@echo "Installing PyTorch 2.6.0 with CUDA 11.8 support..."
	venv/bin/pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
	@echo "PyTorch installed"

# Download and install vLLM wheel
install-vllm: venv
	@echo "Installing vLLM 0.8.5..."
	@if [ ! -f vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl ]; then \
		echo "Downloading vLLM wheel..."; \
		wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl; \
	fi
	venv/bin/pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
	@echo "vLLM installed"

# Install flash-attention
install-flash-attn: venv
	@echo "Installing flash-attention 2.7.3..."
	venv/bin/pip install flash-attn==2.7.3 --no-build-isolation
	@echo "Flash-attention installed"

# Complete installation
install-all: install-base install-torch install-vllm install-flash-attn
	@echo "=========================================="
	@echo "Installation complete!"
	@echo "=========================================="
	@echo "To activate the environment:"
	@echo "  source venv/bin/activate"
	@echo ""
	@echo "To run DeepSeek-OCR:"
	@echo "  cd DeepSeek-OCR-master/DeepSeek-OCR-vllm"
	@echo "  python run_dpsk_ocr_image.py"
	@echo ""

# Install development dependencies
install-dev: venv
	@echo "Installing development dependencies..."
	venv/bin/pip install -r requirements-dev.txt
	@echo "Development dependencies installed"

# Clean up
clean:
	@echo "Cleaning up..."
	rm -rf venv
	rm -f vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
	@echo "Cleanup complete"

# Run tests
test: venv
	@echo "Running tests..."
	venv/bin/pytest

# Run type checking
lint: venv
	@echo "Running type checking with mypy..."
	venv/bin/mypy DeepSeek-OCR-master/DeepSeek-OCR-vllm/ DeepSeek-OCR-master/DeepSeek-OCR-hf/
