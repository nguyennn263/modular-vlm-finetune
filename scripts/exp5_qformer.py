"""
Experiment 5: QFormer (Full 4-Layer Transformer)

Architecture:
- Based on BLIP-2 Q-Former
- Learnable queries: (16, 896)
- 4 Transformer layers with:
  - Self-attention on queries
  - Cross-attention (queries <-> vision)
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
from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from scripts.base_experiment import BaseExperiment, ExperimentConfig


class Exp5Config(ExperimentConfig):
    """Experiment 5 configuration."""
    
    base_model_name = "5CD-AI/Vintern-1B-v3_5"
    torch_dtype = torch.bfloat16
    use_flash_attn = False
    
    bridge_type = "qformer"
    bridge_config = {
        "num_queries": 16,
        "num_heads": 8,
        "num_layers": 4
    }
    
    num_epochs = 10
    batch_size = 2  # Memory-optimized for 14GB GPU
    gradient_accumulation_steps = 4  # Effective batch size = 2 * 4 = 8
    learning_rate = 2e-4
    eval_steps = 100
    save_steps = 500
    
    output_dir = "checkpoints/exp5_qformer"


class Experiment5(BaseExperiment):
    """QFormer bridge experiment."""
    
    def __init__(self, config: Exp5Config):
        super().__init__(config)
        self.config = config
        self.bridge_model = None
    
    def create_model(self) -> torch.nn.Module:
        """Create QFormer bridge model."""
        print("Creating QFormer bridge (4 transformer layers, 16 queries)...")
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
    """Run Experiment 5."""
    config = Exp5Config()
    experiment = Experiment5(config)
    experiment.run(max_samples=None)


if __name__ == "__main__":
    main()
