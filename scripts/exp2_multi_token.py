"""
Experiment 2: MultiTokenMLP

Architecture:
- Linear(4096 -> 896*8)
- Reshape to (B, 8, 896)

Why:
- Single token creates information bottleneck
- Multiple tokens increase representation capacity
- Each token can specialize in different visual aspects
"""

import torch
from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from scripts.base_experiment import BaseExperiment, ExperimentConfig


class Exp2Config(ExperimentConfig):
    """Experiment 2 configuration."""
    
    base_model_name = "5CD-AI/Vintern-1B-v3_5"
    torch_dtype = torch.bfloat16
    use_flash_attn = False
    
    bridge_type = "multi_token"
    bridge_config = {"num_tokens": 8}
    
    num_epochs = 10
    batch_size = 8
    learning_rate = 2e-4
    eval_steps = 100
    save_steps = 500
    
    output_dir = "checkpoints/exp2_multi_token"


class Experiment2(BaseExperiment):
    """MultiTokenMLP bridge experiment."""
    
    def __init__(self, config: Exp2Config):
        super().__init__(config)
        self.config = config
        self.bridge_model = None
    
    def create_model(self) -> torch.nn.Module:
        """Create MultiTokenMLP bridge model."""
        print("Creating MultiTokenMLP bridge (8 tokens)...")
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
    """Run Experiment 2."""
    config = Exp2Config()
    experiment = Experiment2(config)
    experiment.run(max_samples=None)


if __name__ == "__main__":
    main()
