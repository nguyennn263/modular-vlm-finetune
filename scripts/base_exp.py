"""
Base template for bridge fine-tuning experiments.
"""

import torch
from pathlib import Path
from transformers import AutoModel

from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from src.data.loaders import load_datasets
from utils.path_management import RAW_TEXT_CSV, RAW_IMAGES_DIR


class ExpConfig:
    """Experiment configuration."""
    
    # Model
    base_model_name: str = "5CD-AI/Vintern-1B-v3_5"
    bridge_type: str = "better_mlp"
    bridge_config: dict = None
    
    # Training
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 2e-4
    eval_steps: int = 100
    save_steps: int = 500
    
    # Data
    csv_path: str = str(RAW_TEXT_CSV)
    images_dir: str = str(RAW_IMAGES_DIR)
    val_ratio: float = 0.1
    max_samples: int = None
    
    # Output
    output_dir: str = "checkpoints"
    
    def __post_init__(self):
        """Ensure bridge_config is a dict."""
        if self.bridge_config is None:
            self.bridge_config = {}





def run_experiment(exp_config: ExpConfig):
    """Run fine-tuning experiment."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: {exp_config.bridge_type}")
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModel.from_pretrained(
        exp_config.base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    base_model = base_model.to(device)
    
    # Create fine-tune model with selected bridge
    print(f"Creating fine-tune model with {exp_config.bridge_type} bridge...")
    model = create_finetune_model(
        base_model,
        bridge_type=exp_config.bridge_type,
        bridge_config=exp_config.bridge_config
    )
    
    # Load datasets
    print("Loading datasets...")
    result = load_datasets(
        csv_path=exp_config.csv_path,
        images_dir=exp_config.images_dir,
        val_ratio=exp_config.val_ratio,
        max_samples=exp_config.max_samples,
        test_ratio=0.0  # No test split by default
    )
    
    if len(result) == 3:
        train_dataset, val_dataset, test_dataset = result
    else:
        train_dataset, val_dataset = result
        test_dataset = None
    
    # Training config
    train_config = TrainConfig(
        output_dir=Path(exp_config.output_dir) / exp_config.bridge_type,
        num_epochs=exp_config.num_epochs,
        batch_size=exp_config.batch_size,
        learning_rate=exp_config.learning_rate,
        eval_steps=exp_config.eval_steps,
        save_steps=exp_config.save_steps,
    )
    
    # Train
    print("Starting training...")
    trainer = BridgeTrainer(
        model, 
        train_dataset, 
        val_dataset, 
        train_config,
        test_dataset=test_dataset
    )
    trainer.train()
    
    # Evaluate on test set if available
    if test_dataset is not None:
        print("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_dataset)
        print(f"Test metrics: {test_metrics}")
    
    print(f"✓ Experiment {exp_config.bridge_type} completed!")
    print(f"  Best model: {trainer.best_model_path}")


if __name__ == "__main__":
    config = ExpConfig()
    run_experiment(config)
