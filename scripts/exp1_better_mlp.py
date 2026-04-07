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
from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from scripts.base_experiment import BaseExperiment, ExperimentConfig


class Exp1Config(ExperimentConfig):
    """Experiment 1 configuration."""
    
    base_model_name = "5CD-AI/Vintern-1B-v3_5"
    torch_dtype = torch.bfloat16
    use_flash_attn = False
    
    # Bridge config
    bridge_type = "better_mlp"
    bridge_config = {}
    
    # Training
    num_epochs = 10
    batch_size = 8
    learning_rate = 2e-4
    eval_steps = 100
    save_steps = 500
    
    output_dir = "checkpoints/exp1_better_mlp"


class Experiment1(BaseExperiment):
    """BetterMLP bridge experiment."""
    
    def __init__(self, config: Exp1Config):
        super().__init__(config)
        self.config = config
        self.bridge_model = None
    
    def create_model(self) -> torch.nn.Module:
        """Create BetterMLP bridge model."""
        print("Creating BetterMLP bridge...")
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
    config = Exp1Config()
    experiment = Experiment1(config)
    experiment.run(max_samples=None)  # Use all data


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()

