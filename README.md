# VLM Fine-tuning Framework

Modular Vision-Language Model fine-tuning framework with easy model swapping. Optimized for Kaggle T4 GPU.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python run_train.py --config configs/kaggle_t4.yaml
```

## 📁 Structure

```
src/
├── models/           # Modular components
│   ├── vlm.py        # Main VLMModel class
│   ├── vision_encoders.py
│   ├── projectors.py
│   ├── registry.py   # Model configs
│   └── legacy/       # Old code (reference only)
├── data/             # Dataset + preprocessing
└── utils/            # Logging, checkpoints, metrics

configs/
├── config.yaml       # Default settings
└── kaggle_t4.yaml    # T4 optimized

metrics/              # 9 evaluation metrics
data/                 # Training data (images + JSON)
```

## ⚙️ Config - Swap Models Easily

```yaml
model:
  vision_encoder_type: "internvit"  # or: siglip, clip
  projector_type: "mlp"              # or: linear, mlp_gelu, downsample
  llm_type: "qwen2-0.5b"             # or: qwen2-1.5b, phi-2

lora:
  enabled: true  # LoRA fine-tuning
  r: 16
```

**Available Models:**
- Vision: InternViT (1024), SigLIP (1152), CLIP (1024)
- LLM: Qwen2-0.5B, Qwen2-1.5B, Phi-2

## 📊 Data Format

```json
[
  {
    "image": "img_001.jpg",
    "question": "Describe this",
    "answer": "This is..."
  }
]
```

Place images in `data/images/`, JSON in `data/`

## 🏃 Training

```bash
# Train
python run_train.py --config configs/kaggle_t4.yaml

# Override args
python run_train.py --config configs/config.yaml \
    --batch_size 2 --learning_rate 1e-5 --epochs 5

# Resume
python run_train.py --config configs/config.yaml --resume latest
```

## 💾 Key Features

- **Modular**: Swap vision encoder, projector, LLM via config
- **Registry**: Pre-configured LLM models
- **Dynamic Tiling**: Auto-split large images into 448×448 tiles
- **LoRA**: Memory-efficient fine-tuning
- **Label Masking**: Loss only on assistant responses
- **Checkpoints**: Auto-save best models
- **Metrics**: 9 evaluation metrics (BLEU, ROUGE, F1, etc.)

## 🔥 Kaggle T4 Tips

T4 GPU has 16GB VRAM. Config optimizations:
- Use `qwen2-0.5b` LLM
- `batch_size: 1`, `gradient_accumulation: 16`
- `max_tiles: 4`, `max_length: 512`
- Disable wandb: `use_wandb: false`
- Enable gradient checkpointing: `gradient_checkpointing: true`

## 📝 Changes Made

- Removed unused `src/utils/loss.py`
- Moved legacy code to `src/models/legacy/`
- Refactored metrics to use `metrics/` module
- Cleaned up duplicate code
- Ready for production training

