"""
Experiment 2: MultiTokenMLP

Architecture:
- Linear(4096 → 896*8)
- Reshape to (B, 8, 896)

Why:
- Single token creates information bottleneck
- Multiple tokens increase representation capacity
- Each token can specialize in different visual aspects
"""

import torch
from pathlib import Path
from transformers import AutoModel

from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from src.data.loaders import load_datasets
from utils.path_management import RAW_TEXT_CSV, RAW_IMAGES_DIR


class Exp2Config:
    """Exp2 configuration."""
    
    base_model_name = "5CD-AI/Vintern-1B-v3_5"
    bridge_type = "multi_token"
    bridge_config = {"num_tokens": 8}
    
    num_epochs = 10
    batch_size = 8
    learning_rate = 2e-4
    eval_steps = 100
    save_steps = 500
    
    output_dir = "checkpoints/exp2_multi_token"


def load_datasets_exp2():
    """Load training and validation datasets."""
    return load_datasets(
        csv_path=str(RAW_TEXT_CSV),
        images_dir=str(RAW_IMAGES_DIR),
        val_ratio=0.1
    )


def main():
    """Run Experiment 2."""
    
    print("=" * 80)
    print("EXPERIMENT 2: MultiTokenMLP")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    print("Loading base model...")
    base_model = AutoModel.from_pretrained(
        Exp2Config.base_model_name,
        trust_remote_code=True,
        device_map="auto"
    )
    
    print("Creating MultiTokenMLP bridge (8 tokens)...")
    model = create_finetune_model(
        base_model,
        bridge_type=Exp2Config.bridge_type,
        bridge_config=Exp2Config.bridge_config
    )
    
    print("Loading datasets...")
    train_dataset, val_dataset = load_datasets_exp2()
    
    print("Starting training...\n")
    config = TrainConfig(
        output_dir=Exp2Config.output_dir,
        num_epochs=Exp2Config.num_epochs,
        batch_size=Exp2Config.batch_size,
        learning_rate=Exp2Config.learning_rate,
        eval_steps=Exp2Config.eval_steps,
        save_steps=Exp2Config.save_steps,
    )
    
    trainer = BridgeTrainer(model, train_dataset, val_dataset, config)
    trainer.train()
    
    print(f"\n✓ Experiment 2 completed!")
    print(f"  Best model: {trainer.best_model_path}")


if __name__ == "__main__":
    main()
