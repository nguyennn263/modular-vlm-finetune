"""
Experiment 5: Full Q-Former (Advanced Semantic Filtering)

Architecture:
- Baseline: Linear(1024 → 896) from pooled vision
- Improvement:
  * Project patches: Linear(1024 → 896)
  * Learnable queries: (15, 896) - 15 improvement tokens
  * 4 Q-Former layers with semantic filtering:
    - Vision cross-attention: queries ↔ patches
    - Question cross-attention: queries ↔ text
    - Gating: adaptive fusion of vision vs text signals
    - Self-attention: refine queries
    - FFN: non-linear transformation
- Output: (B, 16, 896) = [baseline_token, 15_improvement_tokens]

Why:
- Baseline anchors to stable alignment
- Vision cross-attn extracts visual features
- Question cross-attn provides semantic context
- Gating prevents over-reliance on one modality
- 4 layers with full semantic awareness
- Most expressive but still maintains baseline stability
"""

import argparse
import torch
from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from scripts.base_experiment import BaseExperiment, ExperimentConfig, is_kaggle


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
    eval_steps = 500
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
        print("Creating Full Q-Former (4 layers, vision+text fusion, +15 improvement tokens)...")
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
    parser = argparse.ArgumentParser(description="Run Experiment 5: QFormer")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to use (e.g., 100 for testing)")
    args = parser.parse_args()
    
    # Auto-detect Kaggle and limit to 100 samples if not specified
    if args.max_samples is None and is_kaggle():
        args.max_samples = 100
        print("🔍 Kaggle detected → Auto-limit to 100 samples")
    
    config = Exp5Config()
    experiment = Experiment5(config)
    experiment.run(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
