"""
Experiment 4: MiniQFormer (2 Transformer Layers)

Architecture:
- Learnable queries: (8, 896)
- 2 Transformer layers with:
  - Self-attention on queries
  - Cross-attention (queries ↔ vision)
  - Feed-forward network
  - Residual connections, layer normalization
- Output: (B, 8, 896)

Why:
- Self-attention allows tokens to communicate and refine representations
- Cross-attention enables vision-to-query information flow
- 2 layers balance expressiveness vs computational cost
- Skip connections improve optimization
"""

import torch
from pathlib import Path
from transformers import AutoModel

from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from src.data.loaders import load_datasets
from utils.path_management import RAW_TEXT_CSV, RAW_IMAGES_DIR


class Exp4Config:
    """Exp4 configuration."""
    
    base_model_name = "5CD-AI/Vintern-1B-v3_5"
    bridge_type = "mini_qformer"
    bridge_config = {
        "num_tokens": 8,
        "num_heads": 8
    }
    
    num_epochs = 10
    batch_size = 8
    learning_rate = 2e-4
    eval_steps = 100
    save_steps = 500
    
    output_dir = "checkpoints/exp4_mini_qformer"


def load_datasets_exp4():
    """Load training and validation datasets."""
    return load_datasets(
        csv_path=str(RAW_TEXT_CSV),
        images_dir=str(RAW_IMAGES_DIR),
        val_ratio=0.1
    )


def main():
    """Run Experiment 4."""
    
    print("=" * 80)
    print("EXPERIMENT 4: MiniQFormer (2 Transformer Layers)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    print("Loading base model...")
    base_model = AutoModel.from_pretrained(
        Exp4Config.base_model_name,
        trust_remote_code=True,
        device_map="auto"
    )
    
    print("Creating MiniQFormer bridge (2 transformer layers)...")
    model = create_finetune_model(
        base_model,
        bridge_type=Exp4Config.bridge_type,
        bridge_config=Exp4Config.bridge_config
    )
    
    print("Loading datasets...")
    train_dataset, val_dataset = load_datasets_exp4()
    
    print("Starting training...\n")
    config = TrainConfig(
        output_dir=Exp4Config.output_dir,
        num_epochs=Exp4Config.num_epochs,
        batch_size=Exp4Config.batch_size,
        learning_rate=Exp4Config.learning_rate,
        eval_steps=Exp4Config.eval_steps,
        save_steps=Exp4Config.save_steps,
    )
    
    trainer = BridgeTrainer(model, train_dataset, val_dataset, config)
    trainer.train()
    
    print(f"\n✓ Experiment 4 completed!")
    print(f"  Best model: {trainer.best_model_path}")


if __name__ == "__main__":
    main()
