# 🇻🇳 VLM-Benchmark: Bridge Module Fine-Tuning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Train only bridge modules that convert vision embeddings to language model embeddings**

[Quick Start](#-quick-start) • [Architecture](#-architecture) • [5 Experiments](#-5-bridge-experiments) • [Installation](#-installation) • [Training Comparison](#-training-experiments--comparison)

</div>

---

## 📋 Overview

**VLM-Benchmark** is a specialized fine-tuning framework that trains **only bridge modules** to convert vision embeddings into language model embeddings while keeping both the vision model and language model **completely frozen**.

This approach enables:
- ✅ **Memory efficient** - Only bridge parameters trained
- ✅ **Fast convergence** - Fewer parameters to optimize  
- ✅ **Modular** - Easy to swap bridge architectures
- ✅ **Production ready** - Frozen base models guarantee consistency

### Core Idea

```
Frozen Vision Model    Trainable Bridge    Frozen Language Model
    (4096 dims)       [5 architectures]        (896 dims)
        ↓                    ↓                      ↓
   InternVL-Chat  →  BetterMLP/QFormer  →  Qwen2
```

---

## ✨ Key Features

| Feature | Details |
|---------|---------|
| **5 Bridge Architectures** | BetterMLP, MultiTokenMLP, AttentionBridge, MiniQFormer, QFormer |
| **Frozen Models** | Vision & LLM frozen, only bridge trained |
| **Automatic Device Detection** | CUDA → MPS → CPU fallback |
| **Checkpoint Management** | Save/resume training with early stopping |
| **Minimal Dependencies** | PyTorch, Transformers only |

---

## 🧪 5 Bridge Experiments

### Quick Reference

```bash
# Exp 1: BetterMLP (Baseline)
python scripts/exp1_better_mlp.py

# Exp 2: MultiTokenMLP (8 tokens)
python scripts/exp2_multi_token.py

# Exp 3: AttentionBridge (with attention pooling)
python scripts/exp3_attention_bridge.py

# Exp 4: MiniQFormer (2-layer transformer)
python scripts/exp4_mini_qformer.py

# Exp 5: QFormer (4-layer transformer - SOTA)
python scripts/exp5_qformer.py
```

### Architecture Comparison

#### 1️⃣ BetterMLP (Baseline with Skip Connection)
```
Input (4096) → LayerNorm → Linear(4096→2048)+GELU → Linear(2048→896)
                                  ↓
            Skip connection: Linear(4096→896)
                                  ↓
                          Output (896)
```
✅ Simple, fast  
❌ Limited capacity

#### 2️⃣ MultiTokenMLP  
```
Input (4096) → Linear(4096 → 896×8) → Reshape(1,8,896)
```
✅ More parameters  
✅ Multiple tokens

#### 3️⃣ AttentionBridge
```
Vision patches (num_patches, 1024)
     ↓
Learnable queries (8, 896)
     ↓
Multi-head cross-attention
     ↓
Output (8, 896)
```
✅ Interpretable attention  
✅ Learns to focus on important patches

#### 4️⃣ MiniQFormer (2 layers)
```
Vision patches → Learnable queries (8) → 2×Transformer blocks → Output (8, 896)
```
✅ Transformer-based  
✅ Lightweight (2 layers)

#### 5️⃣ QFormer (4 layers - SOTA)
```
Vision patches → Learnable queries (16) → 4×Transformer blocks → Output (16, 896)  
```
✅ Most expressive  
⚠️ Slowest training

---

## 💻 Installation & Setup

### ⚠️ IMPORTANT: Transformers Version

**You MUST use transformers==4.38.2** for Vintern-1B v3_5 compatibility!
- ❌ Version 4.35.2 or older: **WILL NOT WORK** (missing Qwen2Config)
- ❌ Version 4.40.0+: **May have compatibility issues**
- ✅ Version 4.38.2: **Exact tested version**

### ⭐ RECOMMENDED: Using UV + GPU (Fastest)

```bash
# Clone project
cd /path/to/modular-vlm-finetune

# Run GPU-optimized UV setup (installs UV if needed)
bash setup-gpu.sh

# Activate environment
source activate.sh
```

**Why this approach?**
- ✅ 10-100x faster than pip (UV is extremely fast)
- ✅ Automatic dependency resolution and caching
- ✅ GPU/CUDA detection built-in
- ✅ Version locking with `uv.lock` (reproducible)
- ✅ Single command setup

### Alternative: Using UV Manually

```bash
# Install uv one-time (if not done by setup-gpu.sh)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup project
cd /path/to/modular-vlm-finetune

# Create venv and sync dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

### Alternative: Using Conda Setup

```bash
# Navigate to project directory
cd /path/to/modular-vlm-finetune

# Run setup script (creates conda environment + installs dependencies)
bash setup.sh

# Or specify custom environment name
bash setup.sh my_vlm_env
```

**What the setup script does:**
- ✅ Creates conda environment with Python 3.11
- ✅ Uninstalls old transformers version (if present)
- ✅ Installs transformers==4.38.2 (required for Vintern)
- ✅ Installs timm and einops (vision libraries)
- ✅ Installs all other dependencies from `requirements.txt`
- ✅ Creates project directories (data/, checkpoints/, logs/)
- ✅ Checks GPU/CUDA availability
- ✅ Tests installation

### Step: Activate Environment

```bash
# If using UV setup
source activate.sh

# If using conda
conda activate vlm-bridge
```

### Kaggle Notebook Setup

**Run this in first Kaggle cell:**

```python
!pip uninstall -y transformers
!pip install transformers==4.38.2
!pip install timm einops
!pip install -r /kaggle/input/datasets/.../requirements.txt
```

### Manual Setup (if needed)

If scripts don't work, install manually:

```bash
# Create environment with conda
conda create -n vlm-bridge python=3.11 -y
conda activate vlm-bridge

# Install PyTorch (with CUDA 12.1 support)
conda install pytorch::pytorch pytorch::torchvision pytorch::torchaudio -c pytorch -y

# Install specific transformers version
pip uninstall -y transformers
pip install transformers==4.38.2

# Install vision libraries
pip install timm einops

# Install other dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/raw/{images,texts} checkpoints logs outputs
```

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 16GB | 32GB+ |
| **GPU VRAM** | 12GB | 24GB+ |
| **Disk** | 50GB | 100GB+ |
| **Python** | 3.9+ | 3.11+ |
| **PyTorch** | 2.0 | 2.1+ |
| **Transformers** | 4.38.2 | 4.38.2 (DO NOT CHANGE) |

---

## 🚀 Quick Start: Training

### Option 1: Train Single Experiment (Recommended for First Run)

```bash
# Activate environment
source activate_vlm.sh

# Train BetterMLP (fastest, ~30 min per epoch)
python scripts/exp1_better_mlp.py
```

**Training will:**
1. ✅ Auto-detect device (CUDA → MPS → CPU)
2. ✅ Load Vintern-1B model (auto-download if needed)
3. ✅ Freeze vision & LLM models
4. ✅ Train only bridge module
5. ✅ Save best model checkpoint automatically
6. ✅ Display training progress with loss/metrics

**View results:**
```bash
# Best model saved at:
./checkpoints/better_mlp/best_model.pt

# Training logs:
cat logs/data_loader/log_*.txt
```

### Option 2: Train All 5 Experiments Sequentially

```bash
# Activate environment
source activate_vlm.sh

# Run all 5 bridges in sequence
python scripts/run_all_experiments.py
```

**What happens:**
```
Exp1: BetterMLP         → ~30 min/epoch × 10 = 5 hours
Exp2: MultiToken        → ~35 min/epoch × 10 = 6 hours  
Exp3: AttentionBridge   → ~40 min/epoch × 10 = 7 hours
Exp4: MiniQFormer       → ~45 min/epoch × 10 = 8 hours
Exp5: QFormer           → ~50 min/epoch × 10 = 8 hours
                                    ────────────────────
Total time: ~34 hours (on GPU)
```

**Results saved to:**
```
checkpoints/
├── better_mlp/
│   ├── best_model.pt
│   ├── step_*.pt        (intermediate checkpoints)
│   └── logs/
├── multi_token/
├── attention_bridge/
├── mini_qformer/
└── qformer/
    └── best_model.pt    (most powerful model)
```

### Option 3: Train Specific Experiment

```bash
# Train individual experiments
python scripts/exp1_better_mlp.py       # Baseline (fastest)
python scripts/exp2_multi_token.py      # Multi-token approach
python scripts/exp3_attention_bridge.py # Attention-based
python scripts/exp4_mini_qformer.py     # Light transformer
python scripts/exp5_qformer.py          # Full transformer (best)
```

### Option 4: Custom Training with Python API

```python
from transformers import AutoModel
from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from src.data.loaders import load_datasets

# Load base model
base_model = AutoModel.from_pretrained(
    "5CD-AI/Vintern-1B-v3_5",
    trust_remote_code=True,
    device_map="auto"
)

# Create bridge model
model = create_finetune_model(
    base_model,
    bridge_type='qformer',
    bridge_config={'num_queries': 16}
)

# Load data with train/val/test split
train_ds, val_ds, test_ds = load_datasets(
    csv_path="data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv",
    images_dir="data/raw/images",
    val_ratio=0.1,
    test_ratio=0.1
)

# Configure training
config = TrainConfig(
    output_dir="checkpoints/my_experiment",
    num_epochs=10,
    batch_size=8,
    learning_rate=2e-4,
    eval_steps=100,
    early_stopping=True,
    patience=3
)

# Train
trainer = BridgeTrainer(
    model, 
    train_ds, 
    val_ds, 
    config,
    test_dataset=test_ds  # Optional
)
trainer.train()

# Evaluate on test set
if test_ds is not None:
    metrics = trainer.evaluate(test_ds)
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Perplexity: {metrics['perplexity']:.4f}")
```

---

## ⚡ GPU Optimization & Device Configuration

**Automatic GPU Detection**: The training system automatically detects your GPU and applies optimal parameters. No manual configuration needed!

### Device Profiles

The system supports 4 GPU tiers with pre-optimized parameters:

| GPU | VRAM | Batch | Grad Accum | Seq Len | Flash Attn | Speed |
|-----|------|-------|-----------|---------|-----------|-------|
| **T4 / RTX 2080** | 16 GB | 2 | 4 | 256 | ❌ | 1.0x |
| **RTX 3090** | 24 GB | 4 | 2 | 384 | ❌ | 1.5x |
| **A100 / RTX 4090** | 40 GB | 12 | 1 | 512 | ✅ | 3.0x |
| **L40 / H100** | 45 GB+ | 16 | 1 | 512 | ✅ | **3.5x** |

### Automatic Configuration (Default)

When you run any experiment, the system automatically:
1. ✅ Detects your GPU and available VRAM
2. ✅ Selects the optimal profile
3. ✅ Tunes batch_size, gradient_accumulation_steps, sequence_length, and flash_attention
4. ✅ Logs which device profile was selected

**Example output:**
```
[INFO] Detected GPU: NVIDIA L40 (Tesla L40)
[INFO] Available VRAM: 45 GB
[INFO] Selected profile: l40_45gb
[INFO] Auto-configured:
  - batch_size: 16
  - gradient_accumulation_steps: 1
  - max_seq_length: 512
  - use_flash_attn: True
```

### Manual GPU Configuration

If you need to override the automatic selection, use the interactive configuration helper:

```bash
# Show current GPU and select profile manually
bash configure-device.sh
```

**Menu options:**
```
=== GPU Configuration ===
Current GPU: NVIDIA L40 (Tesla L40)
Detected VRAM: 45 GB
Auto-detected tier: l40_45gb

1) Auto-configure (recommended)
2) T4 / RTX 2080 (16 GB)
3) RTX 3090 (24 GB)
4) A100 / RTX 4090 (40 GB)
5) L40 / H100 (45 GB+)
6) Exit

Select option: _
```

The script will show you the command to use with your selected profile:
```bash
DEVICE_PROFILE=l40_45gb python scripts/exp1_better_mlp.py
```

### Environment Variable Override

Force a specific GPU profile without auto-detection:

```bash
# Use T4 profile (useful for debugging on high-end GPU)
DEVICE_PROFILE=t4_16gb python scripts/exp1_better_mlp.py

# Use L40 profile (maximum speed)
DEVICE_PROFILE=l40_45gb python scripts/run_all_experiments.py

# Force A100 profile
DEVICE_PROFILE=a100_40gb python scripts/exp5_qformer.py
```

### Disabling Auto-Optimization

To disable automatic GPU optimization and use your custom config:

```python
from scripts.base_experiment import BaseExperiment
from src.training import TrainConfig

config = TrainConfig(
    batch_size=2,
    gradient_accumulation_steps=4,
    # ... other custom settings
)

# Set auto_optimize=False to skip device detection
experiment = BaseExperiment(config, auto_optimize=False)
experiment.run()
```

### Speed Improvements

**Training speed on different GPUs** (using Exp1: BetterMLP):

- **T4 (Kaggle)**: ~30 min/epoch (baseline, memory-constrained)
- **RTX 3090**: ~20 min/epoch (1.5x faster)
- **A100**: ~10 min/epoch (3.0x faster)
- **L40 (Production)**: ~8.5 min/epoch (3.5x faster) 🚀

**Full training time estimates** (Exp1-5, sequential, 10 epochs each):

| GPU | Total Time | Cost (Kaggle) |
|-----|-----------|---------------|
| T4 | ~34 hours | FREE |
| L40 | ~10 hours | ~$3-5 |

> **Switching between T4 and L40**: Just run `bash configure-device.sh` and select the profile. Your config will auto-adapt!

### Understanding Effective Batch Size

All profiles maintain similar **effective batch sizes** for training stability:

| Profile | Batch | Accum | Effective |
|---------|-------|-------|-----------|
| T4 | 2 | 4 | **8** |
| RTX 3090 | 4 | 2 | **8** |
| A100 | 12 | 1 | **12** |
| L40 | 16 | 1 | **16** |

Higher effective batch size on L40 = faster convergence + better gradient estimates.

---

## 📊 Training Monitoring

### During Training

Each epoch shows:
```
Epoch 5/10: 95%|███████▊| 56/59 [23:45<01:15, loss=2.1234]
Step 500: train_loss=2.1234, val_loss=2.0987
✓ New best: val_loss=2.0987
```

### After Training

Check results:
```bash
# View best model checkpoint
ls -lh checkpoints/better_mlp/best_model.pt

# View training logs
cat logs/data_loader/log_*.txt

# View all checkpoints
ls -lh checkpoints/better_mlp/
```

---

## 🧪 Training Experiments & Comparison

**Compare 6 training configurations:** 5 bridge architectures + 1 linear baseline (Exp 6).

### Quick Start: Xem danh sách experiments (Dry Run)

```bash
source activate_vlm.sh
python scripts/training_runner.py --dry-run
```

Kết quả: Hiện danh sách 6 training cases:
```
======================================================================
  DRY RUN: bridge_training_comparison
  Total experiments: 6
======================================================================
  [ 1] Exp 1: BetterMLP            — better_mlp
  [ 2] Exp 2: MultiToken           — multi_token
  [ 3] Exp 3: AttentionBridge      — attention_bridge
  [ 4] Exp 4: MiniQFormer          — mini_qformer
  [ 5] Exp 5: QFormer              — qformer
  [ 6] Exp 6: Linear Baseline (Linear(1024→896)) — linear_baseline
```

### Run All Training Cases

```bash
# Tự động chạy tất cả 6 cases
python scripts/training_runner.py
```

Tự động:
- ✅ Chạy tất cả 5 bridge experiments (exp1-5)
- ✅ Chạy "Linear Baseline" (Exp 6) với minimal Linear bridge
- ✅ Lưu progress trong `outputs/training/progress.json`
- ✅ Bỏ qua experiments đã hoàn thành (auto-resume)

**Thời gian:** ~34 giờ trên GPU (như `run_all_experiments.py`)

### Run Specific Cases

```bash
# Chạy case cụ thể: #2, #4, #6 (MultiToken, MiniQFormer, Full Freeze)
python scripts/training_runner.py --experiments 2,4,6

# Chạy range: #1-3 (BetterMLP, MultiToken, AttentionBridge)
python scripts/training_runner.py --experiments 1-3

# Chạy lại case từ đầu (xóa kết quả cũ)
python scripts/training_runner.py --rerun 1,6
```

### Resume & Control

```bash
# Tiếp tục từ progress đã lưu (bỏ qua completed)
python scripts/training_runner.py

# Chạy lại từ đầu (bỏ qua progress cũ)
python scripts/training_runner.py --no-resume

# Xem progress chi tiết
cat outputs/training/progress.json
```

### Compare Results

```bash
# In bảng so sánh chi tiết
python scripts/collect_ablation_results.py

# Export sang CSV để phân tích
python scripts/collect_ablation_results.py --export-csv training_comparison.csv
```

Bảng output:
```
Experiment                         Bridge Type       Status
─────────────────────────────────────────────────────────────────────
Exp 1: BetterMLP                  better_mlp        ✓ Ready
Exp 2: MultiToken                 multi_token       ✓ Ready
Exp 3: AttentionBridge            attention_bridge  ⏳ Pending
Exp 4: MiniQFormer                mini_qformer      ✓ Ready
Exp 5: QFormer                    qformer           ✓ Ready
Exp 6: Linear Baseline           linear_baseline  ✓ Ready
─────────────────────────────────────────────────────────────────────
Completed: 5/6 experiments
```

### View Detailed Results

```bash
# Results JSON
cat outputs/training/results.json

# Progress tracking
cat outputs/training/progress.json

# Model checkpoints
ls checkpoints/{better_mlp,multi_token,attention_bridge,mini_qformer,qformer,full_freeze}/best_model.pt

# Training logs from all runs
find logs -name "log_*.txt" -type f | head -10
```

### Configuration

Chỉnh tùy chỉnh trong `configs/ablation_config.yaml`:

```yaml
training:
  num_epochs: 10
  batch_size: 8
  learning_rate: 2e-4
  patience: 3
  
cases:
  - exp1_better_mlp
  - exp2_multi_token
  - exp3_attention_bridge
  - exp4_mini_qformer
  - exp5_qformer
  - baseline_full_freeze
```

### Key Directories

```
checkpoints/
├── better_mlp/              # Exp 1
├── multi_token/             # Exp 2
├── attention_bridge/        # Exp 3
├── mini_qformer/            # Exp 4
├── qformer/                 # Exp 5
└── linear_baseline/         # Exp 6: Linear Baseline

outputs/training/
├── progress.json            # Real-time tracking
├── results.json             # Summary results
└── detailed_logs/           # Per-experiment logs
```

---

## 📊 Training Configuration

Default settings in each experiment:

```python
TrainConfig(
    # Paths
    output_dir="checkpoints/exp1_better_mlp",
    
    # Training
    num_epochs=10,
    batch_size=8,
    learning_rate=2e-4,
    weight_decay=0.01,
    
    # Checkpointing
    eval_steps=100,
    save_steps=500,
    
    # Early stopping
    early_stopping=True,
    patience=5,
    min_delta=0.001,
)
```

To customize, edit the experiment script or create a new one from `scripts/base_exp.py`.

---

## 📁 Project Structure

```
VLM-Benchmark/
├── src/
│   ├── modeling/
│   │   └── bridge_modules.py              # 5 bridge implementations
│   │       ├── BetterMLP
│   │       ├── MultiTokenMLP
│   │       ├── AttentionBridge
│   │       ├── MiniQFormer
│   │       └── QFormer
│   │
│   ├── training/
│   │   ├── finetune_setup.py              # VisionLanguageBridge wrapper
│   │   ├── trainer.py                     # BridgeTrainer main class
│   │   ├── __init__.py                    # Module exports
│   │   └── finetune_trainer.py            # Backward compatibility
│   │
│   ├── data/                              # Data utilities
│   ├── middleware/                        # Logger setup
│   └── utils/                             # Config loading, paths
│
├── scripts/
│   ├── exp1_better_mlp.py                 # Experiment 1
│   ├── exp2_multi_token.py                # Experiment 2
│   ├── exp3_attention_bridge.py           # Experiment 3
│   ├── exp4_mini_qformer.py               # Experiment 4
│   ├── exp5_qformer.py                    # Experiment 5
│   ├── base_exp.py                        # Template for custom experiments
│   └── run_all_experiments.py             # Run all in sequence
│
├── configs/                               # Config files (optional)
├── data/                                  # Training data location
├── checkpoints/                           # Saved model checkpoints
│   ├── exp1_better_mlp/
│   ├── exp2_multi_token/
│   ├── exp3_attention_bridge/
│   ├── exp4_mini_qformer/
│   └── exp5_qformer/
│
├── pyproject.toml                         # Project metadata
├── README.md                              # This file
└── LICENSE                                # MIT License
```

---

## 🔑 Key Classes

### VisionLanguageBridge (`src/training/finetune_setup.py`)

Wraps the base model and manages freezing:

```python
from src.training import create_finetune_model

model = create_finetune_model(
    base_model,
    bridge_type='qformer',  # or 'better_mlp', 'multi_token', etc
    bridge_config={...}
)

# Only model.bridge parameters are trainable
# model.vision_model and model.language_model are frozen
```

### BridgeTrainer (`src/training/trainer.py`)

Handles the training loop:

```python
from src.training import BridgeTrainer, TrainConfig

trainer = BridgeTrainer(
    model=model,
    train_dataset=train_data,
    val_dataset=val_data,
    config=TrainConfig(...)
)

trainer.train()  # Training loop with checkpointing & early stopping
```

### TrainConfig

Configuration dataclass:

```python
from dataclasses import dataclass
from src.training import TrainConfig

config = TrainConfig(
    output_dir="checkpoints/exp1",
    num_epochs=10,
    batch_size=8,
    learning_rate=2e-4,
    # ... many more options
)
```

---

## 📊 Model Specifications

| Component | Model | Input | Output | Status |
|-----------|-------|-------|--------|--------|
| Vision | InternVL-Chat | Images | 4096 dims | 🔒 Frozen |
| Bridge | [5 options] | 4096 | 896 | ✅ Trainable |
| Language | Qwen2 | 896 dims | Logits | 🔒 Frozen |

---

## ❓ FAQ & Troubleshooting

### Q: Can I run this on CPU?
**A:** Yes, but training will be very slow (~5-10 hours per epoch). For development/testing, reduce `batch_size` and use `max_samples` in data loading.

```python
config = TrainConfig(batch_size=2)  # Smaller batches on CPU
```

### Q: GPU out of memory
**A:** Reduce batch size or image size:

```bash
# In experiment script, modify:
config.batch_size = 4  # Instead of 8
```

Or modify data loading:
```python
train_ds, val_ds = load_datasets(
    ...,
    img_size=224  # Instead of 448
)
```

### Q: Training too slow / taking too long
**A:** Use faster bridge (BetterMLP) or reduce data:

```python
# Use BetterMLP instead of QFormer
python scripts/exp1_better_mlp.py

# Or limit training data
train_ds, val_ds = load_datasets(..., max_samples=5000)

# Or skip validation steps
config.eval_steps = 500  # Check less frequently
```

### Q: Model not improving / loss stuck
**A:** This is expected at the start. Early stopping helps:

```python
config = TrainConfig(
    patience=5,        # Wait 5 evals without improvement
    min_delta=0.0001   # Minimum improvement threshold
)
```

### Q: Where are the training results?
**A:** All results saved in `./checkpoints/`:

```bash
# View best model
ls -lh checkpoints/better_mlp/best_model.pt

# View all checkpoints for an experiment  
ls checkpoints/better_mlp/

# View training logs
cat logs/data_loader/log_*.txt
```

### Q: How do I resume training from checkpoint?
**A:** Use the `resume_from` parameter in TrainConfig:

```python
config = TrainConfig(
    resume_from="checkpoints/better_mlp/step_500.pt",
    # ... other params
)
```

### Q: Can I train multiple models in parallel?
**A:** Use different output directories:

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python scripts/exp1_better_mlp.py

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python scripts/exp2_multi_token.py
```

---

## 📈 Expected Results

### Training Progress

**Epoch 1:**
- Loss: 3.5-4.0 (high, model is learning)
- Validation loss: 3.2-3.8

**Epoch 5:**
- Loss: 2.2-2.5 (improving)
- Validation loss: 2.1-2.4

**Epoch 10:**
- Loss: 1.8-2.2 (good convergence)
- Validation loss: 1.9-2.3

### Bridge Comparison

Expected relative performance:
```
BetterMLP      ████░░░░░░ (baseline)
MultiToken     █████░░░░░ (+5%)
AttentionBridge ██████░░░░ (+15%)
MiniQFormer    ███████░░░ (+25%)
QFormer        ████████░░ (+35%)
```

Note: Performance gains depend on data quality and training duration.

---

## 🚀 Next Steps

1. **Run Setup** (if not done)
   ```bash
   bash setup.sh
   ```

2. **Start Training** (try BetterMLP first)
   ```bash
   source activate_vlm.sh
   python scripts/exp1_better_mlp.py
   ```

3. **Once successful, run all experiments:**
   ```bash
   python scripts/run_all_experiments.py
   ```

4. **Create custom experiment:**
   - Copy `scripts/base_exp.py` template
   - Modify bridge type and hyperparameters
   - Run your custom training

5. **Deploy best model:**
   - Export from `checkpoints/{best_bridge}/best_model.pt`
   - Use with your application

---

## 📚 Additional Resources

- **Vintern-1B Model**: https://huggingface.co/5CD-AI/Vintern-1B-v3_5
- **PyTorch Documentation**: https://pytorch.org/docs
- **Transformers Library**: https://huggingface.co/docs/transformers

For detailed training pipeline information, see [TRAINING_PIPELINE_GUIDE.md](TRAINING_PIPELINE_GUIDE.md).

---

## ⚡ Performance Tips

- **GPU Memory**: ~4-6GB VRAM (works on T4, A100, etc)
- **Training Speed**: ~30-50 min per epoch (depends on bridge type)
- **Faster Development**: Use `max_samples=5000` with smaller image size
- **Better Results**: Use full dataset with larger image size

---

## 📝 License

MIT License - see [LICENSE](LICENSE)
