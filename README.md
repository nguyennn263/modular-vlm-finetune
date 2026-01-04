# VLM Fine-tuning Framework

Modular Vision-Language Model fine-tuning framework, optimized for Kaggle T4 GPU.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train with default config
python run_train.py --config configs/config.yaml

# Train with Kaggle T4 optimized config
python run_train.py --config configs/kaggle_t4.yaml
```

## 📁 Project Structure

```
VLM-Benchmark/
├── configs/
│   ├── config.yaml          # Default config
│   └── kaggle_t4.yaml       # T4 GPU optimized
├── src/
│   ├── models/
│   │   ├── registry.py      # Model registry (swap models easily)
│   │   ├── vision_encoders.py
│   │   ├── projectors.py
│   │   └── vlm.py           # Main VLM architecture
│   ├── data/
│   │   ├── processor.py     # Dynamic HR tiling
│   │   ├── dataset.py
│   │   └── collator.py
│   └── utils/
├── data/                    # Your training data
├── run_train.py            # Main training script
└── kaggle_notebook.py      # Kaggle notebook entry
```

## 🔧 Configuration - Swap Models Easily

```yaml
model:
  # Vision Encoder: "internvit", "siglip", "clip"
  vision_encoder_type: "internvit"
  
  # Projector: "mlp", "mlp_gelu", "linear", "downsample"
  projector_type: "mlp"
  
  # LLM: "qwen2-0.5b", "qwen2-1.5b", "qwen2-7b", "phi-2"
  llm_type: "qwen2-0.5b"
```

### Available Models

| Vision Encoder | Model | Hidden Size |
|---------------|-------|-------------|
| `internvit` | InternViT-300M-448px | 1024 |
| `siglip` | SigLIP-SO400M-384 | 1152 |
| `clip` | CLIP-ViT-L-336 | 1024 |

| LLM | Model | Size |
|-----|-------|------|
| `qwen2-0.5b` | Qwen2-0.5B-Instruct | 0.5B |
| `qwen2-1.5b` | Qwen2-1.5B-Instruct | 1.5B |
| `phi-2` | Phi-2 | 2.7B |

## 📊 Data Format

```json
[
  {
    "image": "image_001.jpg",
    "question": "Mô tả hình ảnh này?",
    "answer": "Đây là một bức ảnh..."
  }
]
```

## 🏃 Training

```bash
# Basic training
python run_train.py --config configs/config.yaml

# Override parameters
python run_train.py --config configs/config.yaml \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --epochs 5

# Resume from checkpoint
python run_train.py --config configs/config.yaml --resume outputs/checkpoint-100
```

## 💡 Tips for Kaggle T4 (16GB VRAM)

1. Use `qwen2-0.5b` (smallest LLM)
2. Set `max_tiles: 4` (fewer image patches)
3. Set `max_length: 512` (shorter sequences)
4. Use `batch_size: 1` with `gradient_accumulation: 16`
5. Disable wandb: `use_wandb: false`

## 🔑 Key Features

- **Modular Design**: Easily swap vision encoder, projector, LLM
- **Model Registry**: Pre-configured model combinations
- **Dynamic HR Tiling**: Auto-split large images into 448x448 tiles
- **LoRA Fine-tuning**: Memory-efficient training
- **Label Masking**: Only compute loss on assistant responses
- **Kaggle Ready**: Optimized configs for T4 GPU

## 📝 License

MIT
