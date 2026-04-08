#!/bin/bash

# Setup script for Kaggle notebooks - pip-based installation
# Usage in Kaggle cell:
#   %cd /kaggle/input
#   !bash setup_kaggle.sh
# Or copy script into cell and run directly

set -e

# Configuration
PROJECT_DIR="."
PYTHON_VERSION=$(python --version | awk '{print $2}')

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Helper functions
print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if running on Kaggle
check_kaggle() {
    if [ -d "/kaggle/input" ]; then
        print_success "Running on Kaggle detected"
        return 0
    else
        print_info "Not running on Kaggle. This script is optimized for Kaggle."
        print_info "If running locally, use: bash setup.sh"
        return 1
    fi
}

# Check Python
check_python() {
    print_success "Python $PYTHON_VERSION found"
    print_info "Kaggle kernel: $(python -c 'import sys; print(sys.version.split()[0])')"
}

# Upgrade pip
upgrade_pip() {
    print_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel --quiet
    print_success "pip upgraded"
}

# Install base dependencies
install_base_deps() {
    print_info "Installing base dependencies..."
    
    # Uninstall old versions
    pip uninstall -y torch torchvision torchaudio transformers --quiet || true
    
    # Install PyTorch 2.2.2 with CUDA 12.1 support (for L40S GPU)
    print_info "Installing PyTorch 2.2.2 with CUDA 12.1 support..."
    pip install --upgrade torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cu121 --quiet
    print_success "PyTorch 2.2.2 installed with CUDA 12.1"
    
    # Install other core packages
    pip install --upgrade \
        pydantic \
        pyyaml \
        pandas \
        numpy \
        tqdm \
        --quiet
    
    # Install specific transformers version for Vintern
    print_info "Installing transformers==4.38.2 (required for Vintern)..."
    pip install --upgrade transformers==4.38.2 --quiet
    
    # Install vision libraries
    print_info "Installing timm and einops..."
    pip install --upgrade timm einops --quiet
    
    print_success "Base dependencies installed"
}

# Install optional dependencies
install_optional_deps() {
    print_info "Installing optional dependencies..."
    
    # Optional packages that might not be available
    pip install --upgrade \
        kagglehub \
        opencv-python \
        Pillow \
        scikit-learn \
        --quiet 2>/dev/null || print_info "Some optional packages may already be installed"
    
    # Note: transformers handle in base_deps due to version requirement
    print_success "Optional dependencies installed"
}

# Check GPU
check_gpu() {
    print_header "GPU Status"
    
    gpu_status=$(python -c "import torch; print('Available' if torch.cuda.is_available() else 'Not available')" 2>/dev/null)
    print_info "CUDA status: $gpu_status"
    
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
    then
        gpu_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        print_success "GPU count: $gpu_count"
        print_success "GPU: $gpu_name"
    else
        print_info "Using CPU (slower, GPU recommended for Kaggle)"
    fi
}

# Check Kaggle datasets
check_kaggle_datasets() {
    print_header "Checking Kaggle Datasets"
    
    if [ -d "/kaggle/input" ]; then
        print_info "Available datasets in /kaggle/input/:"
        ls -1 /kaggle/input/ | head -10
        echo "..."
    fi
}

# Test installation
test_installation() {
    print_header "Testing Installation"
    
    python << 'PYTEST'
import sys

print("Checking imports...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers: {e}")
    sys.exit(1)

try:
    import pandas
    print(f"✓ Pandas {pandas.__version__}")
except ImportError as e:
    print(f"✗ Pandas: {e}")
    sys.exit(1)

try:
    import yaml
    print(f"✓ PyYAML {yaml.__version__}")
except ImportError as e:
    print(f"✗ PyYAML: {e}")
    sys.exit(1)

try:
    import pydantic
    print(f"✓ Pydantic {pydantic.__version__}")
except ImportError as e:
    print(f"✗ Pydantic: {e}")
    sys.exit(1)

print("\n✓ All core packages installed successfully")
PYTEST
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
    else
        print_error "Installation test failed"
        exit 1
    fi
}

# Mount VLM project from /kaggle/input
setup_project_link() {
    print_header "Setting Up Project Access"
    
    if [ -d "/kaggle/input" ]; then
        print_info "Kaggle input datasets available at: /kaggle/input/"
        print_info "Your code should reference: /kaggle/input/{dataset_name}"
    fi
}

# Show usage
show_usage() {
    print_header "Setup Complete! 🎉"
    echo ""
    echo "Python version: $PYTHON_VERSION"
    echo "Project directory: $PROJECT_DIR"
    echo ""
    echo "Data Loading (auto-detects Kaggle environment):"
    echo "  from utils.data_loader_helper import load_ablation_data"
    echo "  train_samples, val_samples = load_ablation_data(max_samples=1000)"
    echo ""
    echo "Or specific to Kaggle:"
    echo "  from src.data.training_data_provider import create_training_provider"
    echo "  provider = create_training_provider()"
    echo "  train, val = provider.get_train_val_split()"
    echo ""
    echo "Running Experiments:"
    echo "  python scripts/exp1_better_mlp.py"
    echo "  python scripts/exp2_multi_token.py"
    echo "  python scripts/training_runner.py"
    echo ""
    echo "📖 See README.md for full documentation"
    echo ""
}

# Main flow
main() {
    clear
    print_header "VLM-Benchmark Kaggle Setup"
    echo "Using pip-based installation (optimized for Kaggle notebooks)"
    echo ""
    
    # Step 1: Check environment
    print_header "Step 1: Checking Environment"
    check_kaggle || true
    check_python
    
    # Step 2: Upgrade pip
    print_header "Step 2: Upgrading pip"
    upgrade_pip
    
    # Step 3: Install dependencies
    print_header "Step 3: Installing Dependencies"
    install_base_deps
    install_optional_deps
    
    # Step 4: Check GPU
    check_gpu
    
    # Step 5: Check datasets
    check_kaggle_datasets
    
    # Step 6: Setup project link
    setup_project_link
    
    # Step 7: Test installation
    test_installation
    
    # Step 8: Show usage
    show_usage
}

# Run main
main