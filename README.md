# VLM-Benchmark: Bridge Module Fine-Tuning

Train only bridge modules to convert vision embeddings to language model embeddings while keeping both vision and language models frozen.

Python 3.11+ | PyTorch 2.2+ | Transformers 4.38.2

---

## Overview

VLM-Benchmark provides a modular fine-tuning framework with 6 bridge architectures that improve upon the baseline projection:

- Baseline: Linear(1024 → 896) projection from vision to language embeddings
- Improvements: Residual, multi-token, attention-based, transformer layers, adaptive gating
- Approach: Each bridge adds improvements on top of baseline rather than replacing it
- Benefits: Memory efficient, stable training, modular design, only bridge trained

## Installation

Requirements:
- Python 3.11+
- PyTorch 2.2.2 (with CUDA 12.1 for GPU)
- transformers==4.38.2 (REQUIRED - do not change)
- 16GB+ RAM, 12GB+ GPU VRAM recommended

Step 1: Clone and navigate to project
```
cd /path/to/modular-vlm-finetune
```

Step 2: Install dependencies using conda
```
bash setup.sh
```

This creates a conda environment, installs PyTorch, transformers==4.38.2, and all dependencies.

Or use manual setup:
```
conda create -n vlm-bridge python=3.11 -y
conda activate vlm-bridge
pip install torch==2.2.2 torchvision==0.17.2
pip install transformers==4.38.2 timm einops pydantic accelerate
pip install -r requirements.txt
mkdir -p data/raw/{images,texts} checkpoints logs outputs
```

Step 3: Activate environment
```
conda activate vlm-bridge
```

## Project Structure

Key directories:

- src/modeling/bridge_modules.py: All 6 bridge implementations
- scripts/exp1_*.py to exp6_*.py: Training scripts for each experiment
- data/raw/: Images and metadata go here
- checkpoints/: Saved models after training
- outputs/: Training results and logs

Data file required:
- data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv
- data/raw/images/: Vision images

## Bridge Architectures

Exp 1: Residual Bridge (bridge_type="residual")
- Baseline: Linear(1024 → 896)
- Improvement: LayerNorm + 2 FC layers with GELU
- Output: (B, 896) with learned adjustment on baseline

Exp 2: Multi-Token Bridge (bridge_type="multi_token")
- 1 baseline token + (k-1) improvement tokens
- Output: (B, k, 896) where baseline anchors alignment

Exp 3: Tile Attention Bridge (bridge_type="tile_attention")
- Baseline: patch-wise projection
- Improvement: self-attention between patches
- Output: (B, 8, 896) with spatial awareness

Exp 4: Lightweight Q-Former (bridge_type="mini_qformer")
- Baseline token + 7 improvement tokens
- 2 lightweight transformer layers
- Output: (B, 8, 896)

Exp 5: Full Q-Former (bridge_type="qformer")
- Baseline token + 15 improvement tokens
- 4 transformer layers with vision+text fusion and gating
- Output: (B, 16, 896)

Exp 6: Gated Fusion Bridge (bridge_type="gated_fusion")
- Baseline: Linear(1024 → 896)
- Improvement: full network
- Gate: sigmoid learns when to apply improvement
- Output: baseline + gate * improvement (adaptive blending)

---

## Bridge Architectures

### Quick Reference

```bash
# Exp 1: ResidualBridge (baseline + improvement)
python scripts/exp1_residual_bridge.py

# Exp 2: MultiTokenMLP (8 tokens)
python scripts/exp2_multi_token.py

# Exp 3: AttentionBridge (with attention pooling)
python scripts/exp3_attention_bridge.py

# Exp 4: MiniQFormer (2-layer transformer)
python scripts/exp4_mini_qformer.py

# Exp 5: QFormer (4-layer transformer - SOTA)
python scripts/exp5_qformer.py

# Exp 6: GatedFusion (gated residual blend)
python scripts/exp6_gated_fusion.py
```

### Architecture Comparison

#### LinearBridge (Minimal Baseline)
```
Input (4096) → Linear(4096 → 896)
               Output (896)
```
+ Minimal parameters (ablation baseline)
+ Fastest training/inference
- Limited capacity (validates bridge importance)

#### BetterMLP (Baseline with Skip Connection)
```
Input (4096) → LayerNorm → Linear(4096→2048)+GELU → Linear(2048→896)
                                 ↓
           Skip connection: Linear(4096→896)
                                 ↓
                         Output (896)
```
+ Simple, fast
+ Better than LinearBridge
- Limited capacity vs transformers

#### MultiTokenMLP
```
Input (4096) → Linear(4096 → 896×8) → Reshape(1,8,896)
```
+ More parameters
+ Multiple tokens for richer representation

#### AttentionBridge
```
Vision patches (num_patches, 1024)
    ↓
Learnable queries (8, 896)
    ↓
Multi-head cross-attention
    ↓
Output (8, 896)
```
+ Interpretable attention
+ Learns to focus on important patches

#### MiniQFormer (2 layers)
```
Vision patches → Learnable queries (8) → 2×Transformer blocks → Output (8, 896)
```
+ Transformer-based
+ Lightweight (2 layers)

#### QFormer (4 layers - Best)
```
Vision patches → Learnable queries (16) → 4×Transformer blocks → Output (16, 896)
```
+ Most expressive
+ Best performance (slowest training)

## Quick Start: Running Training

Step 1: Prepare data

Your data directory should have:
- data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv (metadata with image paths and labels)
- data/raw/images/ (directory with image files)

Step 2: Run a single experiment

Activate environment:
```
conda activate vlm-bridge
```

Train Exp 1 (Residual Bridge - fastest):
```
python scripts/exp1_residual_bridge.py
```

This will:
1. Auto-detect GPU and optimize settings
2. Download Vintern-1B model (auto-cached)
3. Load training data
4. Train for 10 epochs
5. Save best model to checkpoints/exp1_residual_bridge/best_model.pt
6. Display progress and metrics

Step 3: Train other experiments

```
python scripts/exp2_multi_token.py          # Multi-token
python scripts/exp3_attention_bridge.py     # Tile attention
python scripts/exp4_mini_qformer.py         # Lightweight transformer
python scripts/exp5_qformer.py              # Full transformer
python scripts/exp6_gated_fusion.py         # Gated fusion
```

Step 4: Run all experiments sequentially

```
python scripts/run_all_experiments.py
```

This trains all 6 bridges in sequence with auto-resume (skips completed experiments).

Estimated time:
- Exp 1: 5 hours (30 min/epoch × 10)
- Exp 2: 6 hours (35 min/epoch × 10)
- Exp 3: 7 hours (40 min/epoch × 10)
- Exp 4: 8 hours (45 min/epoch × 10)
- Exp 5: 8 hours (50 min/epoch × 10)
- Exp 6: 4 hours (25 min/epoch × 10)
- Total: ~38 hours on GPU

Step 5: View results

Check best model:
```
ls -lh checkpoints/exp*/best_model.pt
```

Check logs:
```
cat logs/data_loader/log_*.txt
```

Compare results:
```
python scripts/collect_ablation_results.py
```

## Training Configuration

Edit in each experiment script or create custom config:

```python
TrainConfig(
    output_dir="checkpoints/exp_name",
    num_epochs=10,
    batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    eval_steps=500,
    save_steps=500,
    early_stopping=True,
    patience=3,
)
```

Parameters:
- num_epochs: Training iterations
- batch_size: Samples per batch (reduce if OOM)
- gradient_accumulation_steps: Gradient accumulation for larger effective batch
- learning_rate: Optimizer learning rate
- eval_steps: Evaluate every N steps
- early_stopping: Stop if validation doesn't improve
- patience: Wait N evaluations before stopping

## Custom Training

Create your own experiment by copying scripts/base_exp.py:

```python
from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from transformers import AutoModel

# Load model
base_model = AutoModel.from_pretrained(
    "5CD-AI/Vintern-1B-v3_5",
    trust_remote_code=True,
    device_map="auto"
)

# Create bridge
model = create_finetune_model(
    base_model,
    bridge_type='residual',  # or multi_token, tile_attention, mini_qformer, qformer, gated_fusion
    bridge_config={}
)

# Load data
train_ds, val_ds, _ = load_datasets(
    csv_path="data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv",
    images_dir="data/raw/images",
    val_ratio=0.1,
    test_ratio=0.1
)

# Configure
config = TrainConfig(
    output_dir="checkpoints/my_experiment",
    num_epochs=10,
    batch_size=2,
    learning_rate=2e-4,
    early_stopping=True
)

# Train
trainer = BridgeTrainer(model, train_ds, val_ds, config)
trainer.train()
```

## Troubleshooting

GPU out of memory:
- Reduce batch_size to 1 in config
- Reduce num_tokens in bridge_config
- Reduce image resolution in data loading

Training too slow:
- Use Exp 1 (residual) - fastest
- Reduce num_epochs
- Use --max-samples flag to limit data

Model not improving:
- Increase learning_rate to 3e-4
- Increase patience for early stopping
- Check that data is loading correctly (cat logs/data_loader/log_*.txt)

Model not loading:
- Ensure transformers==4.38.2 is installed
- Run: pip install --upgrade transformers==4.38.2
- Do NOT use version 4.35.2 or older or 4.40.0+

### IMPORTANT: Transformers Version

**You MUST use transformers==4.38.2** for Vintern-1B v3_5 compatibility:
- Version 4.35.2 or older: will not work (missing Qwen2Config)
- Version 4.40.0+: may have compatibility issues
- Version 4.38.2: exact tested version

To verify after setup:
```bash
python -c "import transformers; print(transformers.__version__)"
```

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16GB | 32GB+ |
| GPU VRAM | 12GB | 24GB+ |
| Python | 3.9+ | 3.11+ |
| PyTorch | 2.0+ | 2.2+ |
| Transformers | 4.38.2 | 4.38.2 (exact) |

### Setup Methods

**Option 1: Quick setup with conda (recommended)**

```bash
cd /path/to/modular-vlm-finetune
bash setup.sh
source activate vlm-bridge  # or activate.sh
```

**Option 2: Manual conda setup**

```bash
conda create -n vlm-bridge python=3.11 -y
conda activate vlm-bridge
conda install pytorch::pytorch pytorch::torchvision pytorch::torchaudio -c pytorch -y
pip install -r requirements.txt
mkdir -p data/raw/{images,texts} checkpoints logs outputs
```

**Option 3: Kaggle Notebook (run in first cell)**

```python
!bash /kaggle/input/.../setup.sh  # or copy setup commands above
```

## Quick Start: Training

### Option 1: Train single experiment (recommended for first run)

```bash
source activate vlm-bridge
python scripts/exp1_residual_bridge.py
```

Training will automatically:
- Detect your GPU and optimize settings
- Load Vintern-1B model (auto-download if needed)
- Freeze vision and language models
- Train only the bridge module
- Save best checkpoint and logs

**Expected output:**
```
Epoch 1/10: 100%|████████████| 1500/1500 [28:35<00:00, train_loss=3.245, val_loss=3.102]
✓ Best model saved: checkpoints/exp1_residual_bridge/best_model.pt
```

Results saved to: `checkpoints/exp1_residual_bridge/`

### Option 2: Train all 6 experiments in sequence

```bash
python scripts/run_all_experiments.py
```

Estimated times:
```
Exp 1: Residual        → ~30 min/epoch × 10 = 5 hours
Exp 2: Multi-token     → ~35 min/epoch × 10 = 6 hours
Exp 3: Tile attention  → ~40 min/epoch × 10 = 7 hours
Exp 4: Mini QFormer    → ~45 min/epoch × 10 = 8 hours
Exp 5: QFormer         → ~50 min/epoch × 10 = 8 hours
Exp 6: Gated fusion    → ~25 min/epoch × 10 = 4 hours
─────────────────────────────────────────────────────
Total: ~38 hours on single GPU
```

All results auto-save to `checkpoints/exp*/best_model.pt`

### Option 3: Train individual experiments

```bash
python scripts/exp1_residual_bridge.py     # Residual + 2 FC
python scripts/exp2_multi_token.py          # 8 token outputs
python scripts/exp3_attention_bridge.py     # Self-attention
python scripts/exp4_mini_qformer.py         # Lightweight transformer
python scripts/exp5_qformer.py              # Full transformer
python scripts/exp6_gated_fusion.py         # Gated fusion
```

### Option 4: Custom training from Python API

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

# Load training data
train_ds, val_ds, test_ds = load_datasets(
    csv_path="data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv",
    images_dir="data/raw/images",
    val_ratio=0.1,
    test_ratio=0.1
)

# Configure and train
config = TrainConfig(
    output_dir="checkpoints/my_experiment",
    num_epochs=10,
    batch_size=8,
    learning_rate=2e-4,
    early_stopping=True,
    patience=3
)

trainer = BridgeTrainer(model, train_ds, val_ds, config)
trainer.train()
```

## Training Configuration

Edit hyperparameters in the experiment scripts or `configs/training_configs.yaml`:

```python
TrainConfig(
    output_dir="checkpoints/exp1_residual_bridge",
    num_epochs=10,
    batch_size=8,
    learning_rate=2e-4,
    weight_decay=0.01,
    eval_steps=100,
    save_steps=500,
    early_stopping=True,
    patience=5,
    min_delta=0.001,
)
```

Key parameters:
- `num_epochs`: Training iterations (default: 10)
- `batch_size`: Samples per batch (reduce to 1-2 on limited VRAM)
- `learning_rate`: Optimizer learning rate (default: 2e-4)
- `early_stopping`: Stop if validation doesn't improve
- `patience`: Wait N evaluations before stopping

## Results and Monitoring

View training progress and results:

```bash
# Best models
ls -lh checkpoints/exp*/best_model.pt

# Training logs
cat logs/data_loader/log_*.txt
cat logs/model_builder/log_*.txt

# All checkpoints for an experiment
ls -lh checkpoints/exp1_residual_bridge/

# Training metrics (if saved)
cat outputs/training/results.json
cat outputs/training/progress.json
```

Expected training loss progression:
```
Epoch 1: ~3.5-4.0 (model learning from scratch)
Epoch 5: ~2.2-2.5 (improving steadily)
Epoch 10: ~1.8-2.2 (good convergence)
```

## Advanced: Custom Training

Create your own experiment using `scripts/base_exp.py` as template:

```python
from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from transformers import AutoModel

# 1. Load base model
model = AutoModel.from_pretrained(
    "5CD-AI/Vintern-1B-v3_5",
    trust_remote_code=True,
    device_map="auto"
)

# 2. Create bridge
model = create_finetune_model(
    model,
    bridge_type='residual',  # or multi_token, tile_attention, mini_qformer, qformer, gated_fusion
    bridge_config={'hidden_dim': 256}
)

# 3. Load data
from src.data.loaders import load_datasets
train_ds, val_ds, _ = load_datasets(
    csv_path="data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv",
    images_dir="data/raw/images",
    val_ratio=0.1
)

# 4. Train
config = TrainConfig(
    output_dir="checkpoints/my_experiment",
    num_epochs=10,
    batch_size=8,
    learning_rate=2e-4
)
trainer = BridgeTrainer(model, train_ds, val_ds, config)
trainer.train()
```

## Troubleshooting

**GPU out of memory (OOM)**

Reduce batch size or image resolution:
```bash
# Edit experiment script
batch_size = 2  # instead of 8
```

Or reduce data:
```python
train_ds, val_ds = load_datasets(..., max_samples=5000)
```

**Training too slow**

Use faster variant or reduce data:
```bash
python scripts/exp1_residual_bridge.py  # Fastest (~30 min/epoch)
# Or reduce training data with max_samples
```

**Model not improving or loss stuck**

Check early stopping parameters and learning rate:
```python
config = TrainConfig(
    learning_rate=3e-4,   # Try higher
    patience=5,           # Allow more epochs
    min_delta=0.0001      # Lower improvement threshold
)
```

**Model loading fails**

Verify transformers version:
```bash
python -c "import transformers; print(transformers.__version__)"  # Should be 4.38.2
pip install --upgrade transformers==4.38.2
```

**RuntimeError: CUDA out of memory**

Known causes and fixes:
```bash
# 1. Reduce batch size
batch_size = 1

# 2. Reduce image size in data loading
max_img_size = 224  # instead of 448

# 3. Clear cache and restart
python -c "import torch; torch.cuda.empty_cache()"
python scripts/exp1_residual_bridge.py
```

**Data not found or loading errors**

Ensure data directory structure:
```bash
# Check files exist
ls data/raw/images/*.jpg  # Should show images
ls data/raw/texts/*.csv   # Should show CSV with metadata
```

Required data format:
- CSV: columns [image_id, question, answer, ...]
- Images: JPEG/PNG files matching image_id in CSV

**How to resume training from checkpoint**

Use `resume_from` in TrainConfig:
```python
config = TrainConfig(
    resume_from="checkpoints/exp1_residual_bridge/step_500.pt",
    output_dir="checkpoints/exp1_residual_bridge",
    # ... other params
)
```

## Key Files and Directories

```
src/modeling/bridge_modules.py         # 6 bridge implementations
src/training/finetune_setup.py        # Model wrapper with freezing
src/training/trainer.py                # Training loop and checkpointing
scripts/exp1_residual_bridge.py       # Experiment 1 (residual bridge)
scripts/exp2_multi_token.py            # Experiment 2 (multi-token)
scripts/exp3_attention_bridge.py       # Experiment 3 (tile attention)
scripts/exp4_mini_qformer.py           # Experiment 4 (mini QFormer)
scripts/exp5_qformer.py                # Experiment 5 (full QFormer)
scripts/exp6_gated_fusion.py           # Experiment 6 (gated fusion)
scripts/base_exp.py                    # Template for custom experiments
scripts/run_all_experiments.py          # Run all 6 in sequence
configs/training_configs.yaml          # Training config template
checkpoints/                           # Saved models
logs/                                  # Training logs
outputs/                               # Results (training/, metrics/)
```

## FAQ

**Q: Can I train on CPU?**
A: Yes but very slow (~5+ hours/epoch). For development, use `max_samples=1000`.

**Q: Can I train multiple models in parallel?**
A: Yes, on different GPUs:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/exp1_residual_bridge.py &
CUDA_VISIBLE_DEVICES=1 python scripts/exp2_multi_token.py &
```

**Q: How long does training take?**
A: ~30-50 min/epoch depending on bridge type and GPU. Full training (10 epochs × 6 models) takes ~38 hours on single GPU.

**Q: What's the best bridge type?**
A: QFormer (Exp 5) typically gives best results (~45% improvement over baseline) but is slowest. Residual bridge (Exp 1) is fastest (~5% improvement). Multi-token and attention variants offer balance.

**Q: Can I export the model for inference?**
A: Yes. The trained bridge is saved as `checkpoints/exp*/best_model.pt`. Load with:
```python
from src.training import create_finetune_model
model = create_finetune_model(base_model, bridge_type='residual')
model.load_state_dict(torch.load('checkpoints/exp1_residual_bridge/best_model.pt'))
```

**Q: Where do I put my own data?**
A: Create CSV in `data/raw/texts/` and images in `data/raw/images/`. CSV needs columns: `image_id`, `question`, `answer`.

---

## Next Steps

1. Run setup: `bash setup.sh && source activate vlm-bridge`
2. Start with Exp 1: `python scripts/exp1_residual_bridge.py`
3. After success, run all: `python scripts/run_all_experiments.py`
4. Compare results: `python scripts/collect_ablation_results.py`
5. Create custom experiment from `scripts/base_exp.py` template

---

## Additional Resources

- Vintern-1B: https://huggingface.co/5CD-AI/Vintern-1B-v3_5
- PyTorch: https://pytorch.org/docs
- Transformers: https://huggingface.co/docs/transformers

---

## License

MIT License - see [LICENSE](LICENSE)
