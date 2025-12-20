# VietVLM-Finetune

Fine-tune Vision-Language Model (VLM) cho tiếng Việt dựa trên kiến trúc InternVL2/Vintern.

## Kiến trúc

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Vision Tower   │────▶│  MLP Projector  │────▶│      LLM        │
│  (InternViT)    │     │  (2-layer MLP)  │     │  (Qwen2/BartPho)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ▲
        │
┌─────────────────┐
│ Dynamic Tiling  │  ← Chia ảnh thành max 12 tiles 448x448 + thumbnail
└─────────────────┘
```

## Cấu trúc thư mục

```
VLM-Benchmark/
├── configs/
│   ├── model_config.yaml    # Cấu hình model, LoRA
│   └── data_config.yaml     # Cấu hình data, preprocessing
├── src/
│   ├── models/
│   │   ├── vision_tower.py  # InternViT wrapper
│   │   ├── projector.py     # MLP Projector
│   │   └── architecture.py  # VinternVLM chính
│   ├── data/
│   │   ├── processor.py     # Dynamic High Resolution tiling
│   │   ├── dataset.py       # VLM Dataset
│   │   └── collator.py      # Data collator với Label Masking
│   └── utils/
│       ├── metrics.py       # VQA metrics
│       └── logger.py        # W&B logger
├── train.py                 # Entry point training
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chuẩn bị data

Format data JSON:
```json
[
  {
    "image": "image_001.jpg",
    "question": "Mô tả hình ảnh này?",
    "answer": "Đây là một bức ảnh..."
  }
]
```

## Training

```bash
# Training với cấu hình mặc định
python train.py

# Custom config
python train.py \
    --model_config configs/model_config.yaml \
    --data_config configs/data_config.yaml \
    --output_dir outputs/exp1

# Resume training
python train.py --resume outputs/checkpoint-1000
```

## Tính năng chính

- **Dynamic High Resolution**: Tự động chia ảnh thành tối đa 12 tiles 448x448 dựa trên aspect ratio
- **LoRA/QLoRA**: Fine-tune hiệu quả với PEFT, chỉ train ~0.5% parameters
- **Label Masking**: Chỉ tính loss trên phần Assistant response
- **Gradient Checkpointing**: Giảm memory usage
- **W&B Integration**: Theo dõi metrics và sample predictions

## Cấu hình LoRA

```yaml
lora:
  enabled: true
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
```

## License

MIT
