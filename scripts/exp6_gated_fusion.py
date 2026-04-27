"""
Experiment 6: Gated Fusion Bridge (Adaptive Blending)

Architecture:
- Baseline: Linear(1024 → 896)
- Improvement: LayerNorm → Linear(1024 → 2048) → GELU → Linear(2048 → 896)
- Gate: sigmoid(Linear(1024 → 896)) - learned per-element
- Output: baseline + gate * improvement

Why:
- Simple residual can push baseline out of optimal range
- Gating learns when to trust baseline vs improvement
- High gate = apply improvement, Low gate = keep baseline
- More stable blending prevents saturation
- Prevents catastrophic forgetting of baseline alignment
"""

import argparse
import torch
from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from scripts.base_experiment import BaseExperiment, ExperimentConfig, is_kaggle


class Exp6Config(ExperimentConfig):
    """Experiment 6 configuration."""
    
    base_model_name = "5CD-AI/Vintern-1B-v3_5"
    torch_dtype = torch.bfloat16
    use_flash_attn = False
    
    bridge_type = "gated_fusion"
    bridge_config = {}
    
    num_epochs = 10
    batch_size = 2  # Memory-optimized for 14GB GPU
    gradient_accumulation_steps = 4  # Effective batch size = 2 * 4 = 8
    learning_rate = 2e-4
    eval_steps = 1000
    save_steps = 1000
    
    output_dir = "checkpoints/exp6_gated_fusion"


class Experiment6(BaseExperiment):
    """Gated Fusion bridge experiment."""
    
    def __init__(self, config: Exp6Config):
        super().__init__(config)
        self.config = config
        self.bridge_model = None
    
    def create_model(self) -> torch.nn.Module:
        """Create Gated Fusion bridge model."""
        print("Creating Gated Fusion Bridge (adaptive blending with learned gate)...")
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
    """Run Experiment 6."""
    parser = argparse.ArgumentParser(description="Run Experiment 6: Linear Bridge (Minimal Baseline)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to use (e.g., 100 for testing)")
    args = parser.parse_args()
    
    # Auto-detect Kaggle and limit to 100 samples if not specified
    if args.max_samples is None and is_kaggle():
        args.max_samples = 100
        print("🔍 Kaggle detected → Auto-limit to 100 samples")
    
    config = Exp6Config()
    experiment = Experiment6(config)
    experiment.run(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
