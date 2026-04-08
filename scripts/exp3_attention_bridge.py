"""
Experiment 3: AttentionBridge

Architecture:
- Project vision features: Linear(1024 -> 896)
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
from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from scripts.base_experiment import BaseExperiment, ExperimentConfig


class Exp3Config(ExperimentConfig):
    """Experiment 3 configuration."""
    
    base_model_name = "5CD-AI/Vintern-1B-v3_5"
    torch_dtype = torch.bfloat16
    use_flash_attn = False
    
    bridge_type = "attention"
    bridge_config = {
        "num_tokens": 8,
        "num_heads": 8
    }
    
    num_epochs = 10
    batch_size = 2  # Memory-optimized for 14GB GPU
    gradient_accumulation_steps = 4  # Effective batch size = 2 * 4 = 8
    learning_rate = 2e-4
    eval_steps = 500
    save_steps = 500
    
    output_dir = "checkpoints/exp3_attention_bridge"


class Experiment3(BaseExperiment):
    """AttentionBridge experiment."""
    
    def __init__(self, config: Exp3Config):
        super().__init__(config)
        self.config = config
        self.bridge_model = None
    
    def create_model(self) -> torch.nn.Module:
        """Create AttentionBridge model."""
        print("Creating AttentionBridge (learnable queries + multi-head attention)...")
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
    """Run Experiment 3."""
    config = Exp3Config()
    experiment = Experiment3(config)
    experiment.run(max_samples=None)


if __name__ == "__main__":
    main()
