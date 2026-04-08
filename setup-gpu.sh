#!/bin/bash

# Fast setup script using UV - optimized for GPU training
# Usage: bash setup-gpu.sh
# Requirements: curl, Python 3.11+

set -e

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_VERSION="3.11"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
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

print_command() {
    echo -e "${CYAN}➜ $1${NC}"
}

# Check Python version
check_python() {
    print_info "Checking Python version..."
    
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.11+"
        exit 1
    fi
    
    python_version=$(python --version 2>&1 | awk '{print $2}')
    print_success "Python $python_version found"
}

# Install UV if not present
install_uv() {
    if command -v uv &> /dev/null; then
        uv_version=$(uv --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1)
        print_success "UV found: $uv_version"
        return 0
    fi
    
    print_info "UV not found. Installing UV..."
    print_command "curl -LsSf https://astral.sh/uv/install.sh | sh"
    
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add UV to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if command -v uv &> /dev/null; then
        print_success "UV installed successfully"
    else
        print_error "Failed to install UV"
        exit 1
    fi
}

# Create Python venv and sync dependencies
setup_venv() {
    print_info "Setting up virtual environment with UV..."
    
    # Clean old venv if exists
    if [ -d "$VENV_DIR" ]; then
        print_info "Removing existing venv..."
        rm -rf "$VENV_DIR"
    fi
    
    # Create and sync with uv
    print_command "uv venv .venv --python $PYTHON_VERSION"
    uv venv .venv --python $PYTHON_VERSION
    
    print_success "Virtual environment created"
    
    # Upgrade pip
    print_info "Upgrading pip, setuptools, wheel..."
    "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel --quiet
    
    # Install PyTorch with CUDA 12.1 support (for L40S GPU) BEFORE uv sync
    print_info "Installing PyTorch 2.2.2 with CUDA 12.1 support (for L40S GPU)..."
    "$VENV_DIR/bin/pip" install --upgrade torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cu121 --quiet
    print_success "PyTorch 2.2.2 installed with CUDA 12.1"
    
    # Sync dependencies from pyproject.toml
    print_info "Installing remaining dependencies with UV (this may take a minute)..."
    print_command "uv sync"
    
    uv sync --quiet
    
    print_success "Dependencies installed successfully"
}

# Create project directories
create_dirs() {
    print_info "Creating project directories..."
    
    mkdir -p "$PROJECT_DIR/data/raw/images"
    mkdir -p "$PROJECT_DIR/data/raw/texts"
    mkdir -p "$PROJECT_DIR/checkpoints"
    mkdir -p "$PROJECT_DIR/logs"
    mkdir -p "$PROJECT_DIR/outputs"
    mkdir -p "$PROJECT_DIR/metrics"
    
    print_success "Directories created"
}

# Check GPU/CUDA
check_gpu() {
    print_header "GPU/CUDA Status"
    
    # Source venv
    source "$VENV_DIR/bin/activate"
    
    if python -c "import torch" 2>/dev/null; then
        cuda_available=$(python -c "import torch; print(torch.cuda.is_available())")
        
        if [ "$cuda_available" = "True" ]; then
            gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
            gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
            cuda_version=$(python -c "import torch; print(torch.version.cuda)")
            
            print_success "CUDA is available"
            print_success "CUDA Version: $cuda_version"
            print_success "GPU count: $gpu_count"
            print_success "GPU name: $gpu_name"
            
            # Show GPU memory
            gpu_memory=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')" 2>/dev/null)
            print_success "GPU Memory: $gpu_memory"
        else
            print_info "✗ CUDA not available - training will use CPU (slower)"
            print_info "  Ensure NVIDIA drivers and CUDA toolkit are installed"
        fi
    else
        print_error "PyTorch not installed yet"
    fi
}

# Check transformers version
check_transformers() {
    print_header "Transformers Version Check"
    
    source "$VENV_DIR/bin/activate"
    
    transformers_version=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
    
    if [ "$transformers_version" = "4.38.2" ]; then
        print_success "Transformers version: $transformers_version (CORRECT)"
    else
        print_error "Transformers version: $transformers_version (should be 4.38.2)"
        print_info "Fixing transformers version..."
        pip install transformers==4.38.2 --upgrade --quiet
        print_success "Transformers downgraded to 4.38.2"
    fi
}

# Test installation
test_installation() {
    print_header "Testing Installation"
    
    source "$VENV_DIR/bin/activate"
    
    test_script='
import sys
import torch
import transformers

tests = [
    ("PyTorch", torch.__version__),
    ("Transformers", transformers.__version__),
    ("CUDA Available", torch.cuda.is_available()),
]

print()
for name, value in tests:
    print(f"  ✓ {name}: {value}")

# Check project structure
from pathlib import Path
project = Path(".")
required = ["src", "data", "checkpoints", "scripts", "configs"]
print()
for d in required:
    exists = (project / d).exists()
    status = "✓" if exists else "✗"
    print(f"  {status} Directory {d}/ {'exists' if exists else 'missing'}")

print()
'
    
    if python -c "$test_script"; then
        print_success "All tests passed"
    else
        print_error "Some tests failed"
        return 1
    fi
}

# Download data from Kaggle
download_data() {
    print_header "Downloading Dataset"
    
    source "$VENV_DIR/bin/activate"
    
    # Check if kagglehub is available
    if ! python -c "import kagglehub" 2>/dev/null; then
        print_error "kagglehub not installed. Attempting to install..."
        pip install --quiet kagglehub
    fi
    
    # Check if data already exists
    if [ -f "$PROJECT_DIR/data/raw/texts/vintern.json" ] && [ "$(ls -A "$PROJECT_DIR/data/raw/images/" 2>/dev/null | wc -l)" -gt 0 ]; then
        print_info "✓ Data already exists, skipping download"
        return 0
    fi
    
    # Run Python data download script
    print_info "Downloading dataset from Kaggle..."
    print_command "python -m src.data.download_data"
    
    if python -m src.data.download_data; then
        print_success "Dataset downloaded successfully"
    else
        print_error "Failed to download dataset"
        print_info "You can download manually later by running: python -m src.data.download_data"
        print_info "Or download from: https://www.kaggle.com/datasets/vintern"
        return 1
    fi
}

# Create activation script
create_activation_script() {
    activation_script="$PROJECT_DIR/activate.sh"
    cat > "$activation_script" << 'ACTIVATE_EOF'
#!/bin/bash

# Quick activation script for UV venv
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found at $VENV_DIR"
    echo "Run: bash setup-gpu.sh"
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ VLM-Bridge environment activated"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Available commands:"
echo "  • Train single experiment: python scripts/exp1_better_mlp.py"
echo "  • Run ablation study:     python scripts/training_runner.py"
echo "  • View training logs:     tail -f logs/*.log"
echo ""
ACTIVATE_EOF
    chmod +x "$activation_script"
    print_success "Activation script created: activate.sh"
}

# Main setup
main() {
    clear
    print_header "VLM-Benchmark GPU Setup (UV-based)"
    echo "Project directory: $PROJECT_DIR"
    echo "Python version:    $PYTHON_VERSION"
    echo ""
    
    # Step 1: Check Python
    print_header "Step 1: Checking Python"
    check_python
    
    # Step 2: Install UV
    print_header "Step 2: Installing/Checking UV"
    install_uv
    
    # Step 3: Setup venv
    print_header "Step 3: Setting Up Virtual Environment"
    setup_venv
    
    # Step 4: Create directories
    print_header "Step 4: Creating Project Directories"
    create_dirs
    
    # Step 5: Download data
    print_header "Step 5: Downloading Data"
    download_data
    
    # Step 6: Check GPU
    check_gpu
    
    # Step 7: Verify transformers version
    check_transformers
    
    # Step 8: Test installation
    cd "$PROJECT_DIR"
    test_installation
    
    # Step 9: Create activation script
    print_header "Step 6: Creating Helper Scripts"
    create_activation_script
    
    # Final summary
    print_header "Setup Complete! 🎉"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Activate the environment:"
    echo "     ${CYAN}source activate.sh${NC}"
    echo ""
    echo "  2. Run ablation study:"
    echo "     ${CYAN}python scripts/training_runner.py${NC}"
    echo ""
    echo "  3. Or train single experiment:"
    echo "     ${CYAN}python scripts/exp1_better_mlp.py${NC}"
    echo ""
    echo "📊 Training will save results to:"
    echo "     ${CYAN}outputs/training/results.json${NC}"
    echo ""
    echo "📖 For more info, see README.md"
    echo ""
}

# Run main setup
main
