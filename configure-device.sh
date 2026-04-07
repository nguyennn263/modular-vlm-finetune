#!/bin/bash

# Quick device configuration helper
# Shows current GPU and applies optimal configuration

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_device_info() {
    print_header "Device Information"
    
    python3 << 'EOF'
import torch
import os

print("GPU Detection:")
if torch.cuda.is_available():
    print(f"  • Device: CUDA")
    print(f"  • GPU Name: {torch.cuda.get_device_name(0)}")
    
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  • Total Memory: {total_mem:.1f} GB")
    
    # Detect which GPU tier
    print("\nDetected GPU Tier:")
    if total_mem < 20:
        print(f"  → T4 / RTX 2080 (16GB) - Memory constrained")
        print(f"  → Recommended: batch_size=2, grad_acc=4")
    elif total_mem < 30:
        print(f"  → RTX 3090 / RTX 4080 (24GB) - Mid-range")
        print(f"  → Recommended: batch_size=4, grad_acc=2")
    elif total_mem < 46:
        print(f"  → A100 40GB - High-end")
        print(f"  → Recommended: batch_size=12, grad_acc=1")
    else:
        print(f"  → L40 / H100 (45GB+) - Production")
        print(f"  → Recommended: batch_size=16, grad_acc=1")
else:
    print("  • Device: CPU (training will be VERY slow)")

EOF
}

auto_configure() {
    print_header "Auto-Configuration"
    
    python3 << 'EOF'
import sys
sys.path.insert(0, '.')

from utils.device_detector import auto_configure_training

config = auto_configure_training()

EOF
}

manual_config() {
    print_header "Manual Configuration"
    
    echo -e "${CYAN}Available profiles:${NC}"
    echo "  1. T4 (16GB)    - batch=2,  grad_acc=4"
    echo "  2. RTX 3090     - batch=4,  grad_acc=2"
    echo "  3. A100 (40GB)  - batch=12, grad_acc=1"
    echo "  4. L40 (45GB+)  - batch=16, grad_acc=1"
    echo ""
    
    read -p "Select profile (1-4) or 'auto' for auto-detect: " choice
    
    case $choice in
        1)
            print_success "Selected: T4 (16GB) profile"
            echo "Update configs/ablation_config.yaml or use environment variable:"
            echo "  DEVICE_PROFILE=t4_16gb python scripts/training_runner.py"
            ;;
        2)
            print_success "Selected: RTX 3090 profile"
            echo "Update configs/ablation_config.yaml or use environment variable:"
            echo "  DEVICE_PROFILE=rtx3090_24gb python scripts/training_runner.py"
            ;;
        3)
            print_success "Selected: A100 40GB profile"
            echo "Update configs/ablation_config.yaml or use environment variable:"
            echo "  DEVICE_PROFILE=a100_40gb python scripts/training_runner.py"
            ;;
        4)
            print_success "Selected: L40 45GB+ profile"
            echo "Update configs/ablation_config.yaml or use environment variable:"
            echo "  DEVICE_PROFILE=l40_45gb python scripts/training_runner.py"
            ;;
        auto)
            auto_configure
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
}

main() {
    echo ""
    
    # Show device info
    print_device_info
    
    echo ""
    
    # Ask what to do
    read -p "Configure automatically? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        auto_configure
    else
        manual_config
    fi
    
    echo ""
    echo -e "${GREEN}To run training with auto-optimization:${NC}"
    echo "  python scripts/training_runner.py"
    echo ""
}

main
