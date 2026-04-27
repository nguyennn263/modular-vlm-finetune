"""
Experiment 1: Residual Bridge (Improvement over Baseline)

Architecture:
- Baseline: Linear(1024 → 896)
- Improvement: LayerNorm → Linear(1024 → 2048) → GELU → Linear(2048 → 896)
- Output: baseline(x) + improvement(x)

Why:
- Keeps baseline alignment intact
- Learns "adjustment" not replacement
- Improves upon MLP1 rather than destroying it
- Stable training with residual connections
"""

import argparse
import torch
from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from scripts.base_experiment import BaseExperiment, ExperimentConfig, is_kaggle


class Exp1Config(ExperimentConfig):
    """Experiment 1 configuration."""
    
    base_model_name = "5CD-AI/Vintern-1B-v3_5"
    torch_dtype = torch.bfloat16
    use_flash_attn = False
    
    # Bridge config - improved residual approach
    bridge_type = "residual"
    bridge_config = {}
    
    # Training
    num_epochs = 10
    batch_size = 2  # Memory-optimized for 14GB GPU
    gradient_accumulation_steps = 4  # Effective batch size = 2 * 4 = 8
    learning_rate = 2e-4
    eval_steps = 1000
    save_steps = 1000
    
    output_dir = "checkpoints/exp1_residual_bridge"


class Experiment1(BaseExperiment):
    """Residual Bridge experiment."""
    
    def __init__(self, config: Exp1Config):
        super().__init__(config)
        self.config = config
        self.bridge_model = None
    
    def create_model(self) -> torch.nn.Module:
        """Create Residual Bridge model."""
        print("Creating Residual Bridge (improvement-based)...")
        self.bridge_model = create_finetune_model(
            self.model,
            bridge_type=self.config.bridge_type,
            bridge_config=self.config.bridge_config
        )
        return self.bridge_model
    
    def train(self):
        """Run training."""
        print("Starting training...\n")
        
        config = TrainConfig(
            model_name=self.config.base_model_name,
            output_dir=self.config.output_dir,
            num_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
        )
        
        trainer = BridgeTrainer(
            self.bridge_model,
            self.train_samples,
            self.val_samples,
            config
        )
        trainer.train()


def main():
    """Run Experiment 1."""
    parser = argparse.ArgumentParser(description="Run Experiment 1: Residual Bridge")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to use (e.g., 100 for testing)")
    args = parser.parse_args()
    
    # Auto-detect Kaggle and limit to 100 samples if not specified
    if args.max_samples is None and is_kaggle():
        args.max_samples = 100
        print("🔍 Kaggle detected → Auto-limit to 100 samples")
    
    config = Exp1Config()
    experiment = Experiment1(config)
    experiment.run(max_samples=args.max_samples)


if __name__ == "__main__":
    main()

