# 🇻🇳 VLM-Benchmark: Bridge Module Fine-Tuning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Train only bridge modules that convert vision embeddings to language model embeddings**

[Quick Start](#-quick-start) • [Architecture](#-architecture) • [5 Experiments](#-5-bridge-experiments) • [Installation](#-installation)

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

### Step 1: Automatic Setup (Recommended)

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
- ✅ Installs all dependencies from `requirements.txt`
- ✅ Creates project directories (data/, checkpoints/, logs/)
- ✅ Checks GPU/CUDA availability
- ✅ Tests installation
- ✅ Creates activation helper script

### Step 2: Activate Environment

```bash
# Quick activation using helper script
source activate_vlm.sh

# Or manually with conda
conda activate vlm-bridge
```

### Manual Setup (if needed)

If `setup.sh` doesn't work, install manually:

```bash
# Install PyTorch (with CUDA 12.1 support)
conda create -n vlm-bridge python=3.11 -y
conda activate vlm-bridge
conda install pytorch::pytorch pytorch::torchvision pytorch::torchaudio -c pytorch -y

# Install dependencies
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

## 🔍 Comparing Results

After running all 5 experiments:

```bash
# (Coming soon: comparison script)
python scripts/compare_experiments.py
```

This will show:
- Best validation loss per architecture
- Training time comparison  
- Model size comparison
- Recommendations

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
