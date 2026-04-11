#!/bin/bash

# Setup script for VLM-Benchmark Bridge Fine-tuning Framework
# Usage: bash setup.sh
# Optional environment name: bash setup.sh my_env

set -e  # Exit on error

# Configuration
CONDA_ENV_NAME=${1:-"vlm-bridge"}
PYTHON_VERSION="3.11"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if conda is installed
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found. Please install Anaconda or Miniconda first."
        echo "Download from: https://www.anaconda.com/download"
        exit 1
    fi
    print_success "Conda found"
}

# Check Python version
check_python() {
    python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_success "Python $python_version found"
}

# Create conda environment
create_env() {
    print_info "Creating conda environment: $CONDA_ENV_NAME"
    
    # Check if environment already exists
    if conda env list | grep -q "^$CONDA_ENV_NAME"; then
        print_info "Environment '$CONDA_ENV_NAME' already exists"
        read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n "$CONDA_ENV_NAME" --yes
            conda create -n "$CONDA_ENV_NAME" python=$PYTHON_VERSION -y
        else
            print_info "Using existing environment"
        fi
    else
        conda create -n "$CONDA_ENV_NAME" python=$PYTHON_VERSION -y
        print_success "Environment '$CONDA_ENV_NAME' created"
    fi
}

# Activate environment
activate_env() {
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV_NAME"
    print_success "Environment activated: $CONDA_ENV_NAME"
}

# Install dependencies
install_deps() {
    print_info "Installing dependencies from requirements.txt..."
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch with CUDA 12.1 support (for L40S GPU)
    print_info "Installing PyTorch 2.2.2 with CUDA 12.1 support..."
    pip uninstall -y torch torchvision torchaudio || true
    pip install --upgrade torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cu121
    print_success "PyTorch 2.2.2 installed with CUDA 12.1"
    
    # Install specific transformer version for Vintern compatibility
    print_info "Installing transformers==4.38.2 (required for Vintern support)..."
    pip uninstall -y transformers || true
    pip install transformers==4.38.2
    
    # Install vision/attention libraries
    print_info "Installing vision and attention libraries..."
    pip install timm einops
    
    # Install from requirements.txt
    if [ -f "$PROJECT_DIR/requirements.txt" ]; then
        pip install -r "$PROJECT_DIR/requirements.txt"
        print_success "Dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Check GPU/CUDA
check_gpu() {
    print_header "GPU/CUDA Status"
    
    if python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
        if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            gpu_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
            gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
            print_success "CUDA is available"
            print_success "GPU count: $gpu_count"
            print_success "GPU name: $gpu_name"
        else
            print_info "CUDA not available - training will use CPU (slower)"
        fi
    else
        print_info "PyTorch not yet installed, check after environment setup"
    fi
}

# Create project directories
create_dirs() {
    print_info "Creating project directories..."
    
    mkdir -p "$PROJECT_DIR/data/raw/images"
    mkdir -p "$PROJECT_DIR/data/raw/texts"
    mkdir -p "$PROJECT_DIR/checkpoints"
    mkdir -p "$PROJECT_DIR/logs"
    mkdir -p "$PROJECT_DIR/outputs"
    
    print_success "Directories created"
}

# Download models (optional)
download_models() {
    print_header "Model Setup"
    
    read -p "Download Vintern-1B model on first run? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Models will be automatically downloaded on first training run"
        print_info "This requires ~25GB disk space for Vintern-1B"
    fi
}

# Download data from Kaggle
download_data() {
    print_header "Downloading Dataset"
    
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
    
    if python -m src.data.download_data; then
        print_success "Dataset downloaded successfully"
    else
        print_error "Failed to download dataset"
        print_info "You can download manually later by running: python -m src.data.download_data"
        print_info "Or download from: https://www.kaggle.com/datasets/vintern"
        return 1
    fi
}

# Test installation
test_installation() {
    print_header "Testing Installation"
    
    python -c "
import torch
import transformers
from pathlib import Path

print('✓ PyTorch:', torch.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ CUDA available:', torch.cuda.is_available())

# Check project structure
from pathlib import Path
project = Path('.')
required = ['src', 'data', 'checkpoints', 'scripts']
for d in required:
    if (project / d).exists():
        print(f'✓ Directory {d}/ exists')
    else:
        print(f'✗ Directory {d}/ missing')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
    else
        print_error "Installation test failed"
        exit 1
    fi
}

# Generate activation script
create_activation_script() {
    activation_script="$PROJECT_DIR/activate_vlm.sh"
    cat > "$activation_script" << 'EOF'
#!/bin/bash

# Quick activation script for VLM-Benchmark environment
CONDA_ENV_NAME=${1:-"vlm-bridge"}

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ VLM-Bridge environment activated: $CONDA_ENV_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Available commands:"
echo "  - python scripts/base_exp.py              : Run base experiment (residual bridge)"
echo "  - python scripts/exp1_residual_bridge.py  : Experiment 1 - Residual Bridge"
echo "  - python scripts/exp2_multi_token.py      : Experiment 2 - MultiToken"
echo "  - python scripts/exp3_tile_attention.py : Experiment 3 - Tile Attention"
echo "  - python scripts/exp4_mini_qformer.py     : Experiment 4 - MiniQFormer"
echo "  - python scripts/exp5_qformer.py          : Experiment 5 - QFormer"
echo "  - python scripts/exp6_gated_fusion.py     : Experiment 6 - Gated Fusion"
echo "  - bash scripts/run_all_experiments.py     : Run all 5 experiments"
echo ""
EOF
    chmod +x "$activation_script"
    print_success "Activation script created: activate_vlm.sh"
}

# Main setup flow
main() {
    clear
    print_header "VLM-Benchmark Setup"
    echo "Environment name: $CONDA_ENV_NAME"
    echo "Project directory: $PROJECT_DIR"
    echo ""
    
    # Step 1: Check conda
    print_header "Step 1: Checking Prerequisites"
    check_conda
    
    # Step 2: Create environment
    print_header "Step 2: Setting Up Conda Environment"
    create_env
    activate_env
    check_python
    
    # Step 3: Install dependencies
    print_header "Step 3: Installing Dependencies"
    install_deps
    
    # Step 4: Create directories
    print_header "Step 4: Creating Project Directories"
    create_dirs
    
    # Step 5: Download data
    print_header "Step 5: Downloading Data"
    download_data
    
    # Step 6: Check GPU
    check_gpu
    
    # Step 7: Model setup
    download_models
    
    # Step 8: Test installation
    cd "$PROJECT_DIR"
    test_installation
    
    # Step 9: Create activation script
    print_header "Step 5: Creating Helper Scripts"
    create_activation_script
    
    # Final summary
    print_header "Setup Complete! 🎉"
    echo ""
    echo "Next steps:"
    echo "  1. Activate the environment:"
    echo "     source activate_vlm.sh"
    echo ""
    echo "  2. Run a single experiment:"
    echo "     python scripts/base_exp.py"
    echo ""
    echo "  3. Or run all experiments:"
    echo "     python scripts/run_all_experiments.py"
    echo ""
    echo "📖 For detailed instructions, see README.md"
    echo ""
}

# Run main setup
main
