"""
Experiment 2: Multi-Token Bridge (Multiple View Tokens)

Architecture:
- Baseline: Linear(1024 → 896) generates anchor token
- Improvement: Linear(1024 → 896*(k-1)) generates additional tokens
- Output: [baseline_token, improvement_tokens_1, ..., improvement_tokens_(k-1)]
  Shape: (B, k, 896)

Why:
- Single token creates information bottleneck
- Multiple tokens capture different visual aspects
- Baseline anchors alignment, improvements add capacity
- LLM process richer representation
"""

import argparse
import torch
from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from scripts.base_experiment import BaseExperiment, ExperimentConfig, is_kaggle


class Exp2Config(ExperimentConfig):
    """Experiment 2 configuration."""
    
    base_model_name = "5CD-AI/Vintern-1B-v3_5"
    torch_dtype = torch.bfloat16
    use_flash_attn = False
    
    bridge_type = "multi_token"
    bridge_config = {"num_tokens": 8}
    
    num_epochs = 10
    batch_size = 2  # Memory-optimized for 14GB GPU
    gradient_accumulation_steps = 4  # Effective batch size = 2 * 4 = 8
    learning_rate = 2e-4
    eval_steps = 1000
    save_steps = 1000
    
    output_dir = "checkpoints/exp2_multi_token"


class Experiment2(BaseExperiment):
    """Multi-Token bridge experiment."""
    
    def __init__(self, config: Exp2Config):
        super().__init__(config)
        self.config = config
        self.bridge_model = None
    
    def create_model(self) -> torch.nn.Module:
        """Create Multi-Token bridge model."""
        print("Creating Multi-Token Bridge (8 tokens with baseline anchor)...")
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
    """Run Experiment 2."""
    parser = argparse.ArgumentParser(description="Run Experiment 2: MultiToken")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to use (e.g., 100 for testing)")
    args = parser.parse_args()
    
    # Auto-detect Kaggle and limit to 100 samples if not specified
    if args.max_samples is None and is_kaggle():
        args.max_samples = 100
        print("🔍 Kaggle detected → Auto-limit to 100 samples")
    
    config = Exp2Config()
    experiment = Experiment2(config)
    experiment.run(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
