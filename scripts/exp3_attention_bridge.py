"""
Experiment 3: AttentionBridge

Architecture:
- Project vision features: Linear(1024 → 896)
- Learnable queries: (8, 896)
- Multi-head attention: queries attend to vision features
- Output: (B, 8, 896)

Why:
- Attention enables dynamic selection of relevant visual features
- Multiple heads capture different visual aspects independently
- Learnable queries serve as "information slots"
- Soft selection is differentiable and learns with task
"""

import torch
from pathlib import Path
from transformers import AutoModel

from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from src.data.loaders import load_datasets
from utils.path_management import RAW_TEXT_CSV, RAW_IMAGES_DIR


class Exp3Config:
    """Exp3 configuration."""
    
    base_model_name = "5CD-AI/Vintern-1B-v3_5"
    bridge_type = "attention"
    bridge_config = {
        "num_tokens": 8,
        "num_heads": 8
    }
    
    num_epochs = 10
    batch_size = 8
    learning_rate = 2e-4
    eval_steps = 100
    save_steps = 500
    
    output_dir = "checkpoints/exp3_attention_bridge"


def load_datasets_exp3():
    """Load training and validation datasets."""
    return load_datasets(
        csv_path=str(RAW_TEXT_CSV),
        images_dir=str(RAW_IMAGES_DIR),
        val_ratio=0.1
    )


def main():
    """Run Experiment 3."""
    
    print("=" * 80)
    print("EXPERIMENT 3: AttentionBridge")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    print("Loading base model...")
    base_model = AutoModel.from_pretrained(
        Exp3Config.base_model_name,
        trust_remote_code=True,
        device_map="auto"
    )
    
    print("Creating AttentionBridge (learnable queries + multi-head attention)...")
    model = create_finetune_model(
        base_model,
        bridge_type=Exp3Config.bridge_type,
        bridge_config=Exp3Config.bridge_config
    )
    
    print("Loading datasets...")
    train_dataset, val_dataset = load_datasets_exp3()
    
    print("Starting training...\n")
    config = TrainConfig(
        output_dir=Exp3Config.output_dir,
        num_epochs=Exp3Config.num_epochs,
        batch_size=Exp3Config.batch_size,
        learning_rate=Exp3Config.learning_rate,
        eval_steps=Exp3Config.eval_steps,
        save_steps=Exp3Config.save_steps,
    )
    
    trainer = BridgeTrainer(model, train_dataset, val_dataset, config)
    trainer.train()
    
    print(f"\n✓ Experiment 3 completed!")
    print(f"  Best model: {trainer.best_model_path}")


if __name__ == "__main__":
    main()
