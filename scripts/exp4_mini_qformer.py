"""
Experiment 4: MiniQFormer (2 Transformer Layers)

Architecture:
- Learnable queries: (8, 896)
- 2 Transformer layers with:
  - Self-attention on queries
  - Cross-attention (queries <-> vision)
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
from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from scripts.base_experiment import BaseExperiment, ExperimentConfig


class Exp4Config(ExperimentConfig):
    """Experiment 4 configuration."""
    
    base_model_name = "5CD-AI/Vintern-1B-v3_5"
    torch_dtype = torch.bfloat16
    use_flash_attn = False
    
    bridge_type = "mini_qformer"
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
    
    output_dir = "checkpoints/exp4_mini_qformer"


class Experiment4(BaseExperiment):
    """MiniQFormer bridge experiment."""
    
    def __init__(self, config: Exp4Config):
        super().__init__(config)
        self.config = config
        self.bridge_model = None
    
    def create_model(self) -> torch.nn.Module:
        """Create MiniQFormer bridge model."""
        print("Creating MiniQFormer bridge (2 transformer layers)...")
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
    """Run Experiment 4."""
    config = Exp4Config()
    experiment = Experiment4(config)
    experiment.run(max_samples=None)


if __name__ == "__main__":
    main()
