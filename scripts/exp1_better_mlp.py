"""
Experiment 1: BetterMLP with Skip Connection

Architecture:
- LayerNorm(4096)
- Linear(4096 → 2048) + GELU
- Linear(2048 → 896)
- Skip: Linear(4096 → 896)
- Output = main + skip

Why:
- LayerNorm stabilizes large vision features
- Bottleneck reduces computational cost
- Skip connections improve gradient flow
"""

import torch
from pathlib import Path
from transformers import AutoModel

from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from src.data.loaders import load_datasets
from utils.data_loader_helper import AblationDataLoader
from utils.path_management import RAW_TEXT_CSV, RAW_IMAGES_DIR


class Exp1Config:
    """Exp1 configuration."""
    
    base_model_name = "5CD-AI/Vintern-1B-v3_5"
    bridge_type = "better_mlp"
    bridge_config = {}
    
    num_epochs = 10
    batch_size = 8
    learning_rate = 2e-4
    eval_steps = 100
    save_steps = 500
    
    output_dir = "checkpoints/exp1_better_mlp"


def load_datasets_exp1():
    """Load training and validation datasets using unified loader."""
    # Use new unified loader that handles both Kaggle and local
    ablation_loader = AblationDataLoader()
    train_samples, val_samples = ablation_loader.load_train_val_split(val_ratio=0.1)
    
    # Convert OneSample objects to VLMDataset format if needed
    # Can also pass the OneSample list directly to VLMDataset
    return train_samples, val_samples


def main():
    """Run Experiment 1."""
    
    print("=" * 80)
    print("EXPERIMENT 1: BetterMLP with Skip Connection")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModel.from_pretrained(
        Exp1Config.base_model_name,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Create fine-tune model
    print("Creating BetterMLP bridge...")
    model = create_finetune_model(
        base_model,
        bridge_type=Exp1Config.bridge_type,
        bridge_config=Exp1Config.bridge_config
    )
    
    # Load datasets
    print("Loading datasets...")
    train_samples, val_samples = load_datasets_exp1()
    
    # Training
    print("Starting training...\n")
    config = TrainConfig(
        output_dir=Exp1Config.output_dir,
        num_epochs=Exp1Config.num_epochs,
        batch_size=Exp1Config.batch_size,
        learning_rate=Exp1Config.learning_rate,
        eval_steps=Exp1Config.eval_steps,
        save_steps=Exp1Config.save_steps,
    )
    
    trainer = BridgeTrainer(model, train_samples, val_samples, config)
    trainer.train()
    
    print(f"\n✓ Experiment 1 completed!")
    print(f"  Best model: {trainer.best_model_path}")


if __name__ == "__main__":
    main()

