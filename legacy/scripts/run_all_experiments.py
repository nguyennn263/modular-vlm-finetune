"""
Run all 6 bridge experiments sequentially.

This script trains all 6 bridge variants:
1. Exp1: Residual Bridge - Baseline + improvement path with residual connection
2. Exp2: MultiTokenMLP - Multiple output tokens (baseline + improvements)
3. Exp3: TileAttention - Learnable queries + multi-head attention over patches
4. Exp4: MiniQFormer - 2 transformer layers with baseline + improvement tokens
5. Exp5: QFormer - 4 transformer layers with text-aware semantic filtering
6. Exp6: GatedFusion - Adaptive blending with learned gating

Usage:
    python scripts/run_all_experiments.py

Results will be saved in checkpoints/exp{1-6}_*/
"""

import sys
import torch
from pathlib import Path

# Import experiment modules
from exp1_residual_bridge import main as exp1_main
from exp2_multi_token import main as exp2_main
from exp3_tile_attention import main as exp3_main
from exp4_mini_qformer import main as exp4_main
from exp5_qformer import main as exp5_main
from exp6_gated_fusion import main as exp6_main


def run_all_experiments():
    """Run all 6 experiments sequentially."""
    
    experiments = [
        ("Exp1: ResidualBridge", exp1_main),
        ("Exp2: MultiTokenMLP", exp2_main),
        ("Exp3: TileAttention", exp3_main),
        ("Exp4: MiniQFormer", exp4_main),
        ("Exp5: QFormer", exp5_main),
        ("Exp6: GatedFusion", exp6_main),
    ]
    
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("Running all 6 bridge experiments")
    print("=" * 80)
    print(f"Device: {device}\n")
    
    for idx, (exp_name, exp_func) in enumerate(experiments, 1):
        print(f"\n{'=' * 80}")
        print(f"Running {exp_name} ({idx}/6)")
        print("=" * 80)
        
        try:
            exp_func()
            results[exp_name] = "✓ Completed"
            print(f"✓ {exp_name} completed successfully")
        except Exception as e:
            results[exp_name] = f"✗ Failed: {str(e)}"
            print(f"✗ {exp_name} failed: {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for exp_name, status in results.items():
        print(f"{exp_name:<30} {status}")
    
    # Final info
    print("\n" + "=" * 80)
    print("Experiment results saved in:")
    for i in range(1, 7):
        print(f"  checkpoints/exp{i}_*/best_model.pt")
    print("=" * 80)


if __name__ == "__main__":
    run_all_experiments()
