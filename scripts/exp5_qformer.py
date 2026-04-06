"""
Experiment 5: QFormer (Full 4-Layer Transformer)

Architecture:
- Based on BLIP-2 Q-Former
- Learnable queries: (16, 896)
- 4 Transformer layers with:
  - Self-attention on queries
  - Cross-attention (queries ↔ vision)
  - Feed-forward network
  - Residual connections, layer normalization
- Output: (B, 16, 896)

Why:
- Most expressive and powerful bridge design
- 4 layers allow deep feature transformation
- 16 queries provide better representation capacity
- Q-Former architecture proven effective in BLIP-2
"""

import torch
from pathlib import Path
from transformers import AutoModel

from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from src.data.loaders import load_datasets
from utils.path_management import RAW_TEXT_CSV, RAW_IMAGES_DIR


class Exp5Config:
    """Exp5 configuration."""
    
    base_model_name = "5CD-AI/Vintern-1B-v3_5"
    bridge_type = "qformer"
    bridge_config = {
        "num_queries": 16,
        "num_heads": 8,
        "num_layers": 4
    }
    
    num_epochs = 10
    batch_size = 8
    learning_rate = 2e-4
    eval_steps = 100
    save_steps = 500
    
    output_dir = "checkpoints/exp5_qformer"


def load_datasets_exp5():
    """Load training and validation datasets."""
    return load_datasets(
        csv_path=str(RAW_TEXT_CSV),
        images_dir=str(RAW_IMAGES_DIR),
        val_ratio=0.1
    )


def main():
    """Run Experiment 5."""
    
    print("=" * 80)
    print("EXPERIMENT 5: QFormer (4-Layer Transformer)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    print("Loading base model...")
    base_model = AutoModel.from_pretrained(
        Exp5Config.base_model_name,
        trust_remote_code=True,
        device_map="auto"
    )
    
    print("Creating QFormer bridge (4 transformer layers, 16 queries)...")
    model = create_finetune_model(
        base_model,
        bridge_type=Exp5Config.bridge_type,
        bridge_config=Exp5Config.bridge_config
    )
    
    print("Loading datasets...")
    train_dataset, val_dataset = load_datasets_exp5()
    
    print("Starting training...\n")
    config = TrainConfig(
        output_dir=Exp5Config.output_dir,
        num_epochs=Exp5Config.num_epochs,
        batch_size=Exp5Config.batch_size,
        learning_rate=Exp5Config.learning_rate,
        eval_steps=Exp5Config.eval_steps,
        save_steps=Exp5Config.save_steps,
    )
    
    trainer = BridgeTrainer(model, train_dataset, val_dataset, config)
    trainer.train()
    
    print(f"\n✓ Experiment 5 completed!")
    print(f"  Best model: {trainer.best_model_path}")


if __name__ == "__main__":
    main()
