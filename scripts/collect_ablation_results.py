#!/usr/bin/env python3
"""
Collect and compare ablation study results
"""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd

def collect_results(ablation_dir: str = "outputs/ablation") -> pd.DataFrame:
    """Collect results from all checkpoint directories"""
    ablation_dir = Path(ablation_dir)
    results = []

    # Checkpoint mapping
    checkpoints = {
        "baseline_full": ("Baseline: Full", "residual"),
        "residual": ("Exp 1: ResidualBridge", "residual"),
        "multi_token": ("Exp 2: MultiToken", "multi_token"),
        "tile_attention": ("Exp 3: TileAttention", "tile_attention"),
        "mini_qformer": ("Exp 4: MiniQFormer", "mini_qformer"),
        "qformer": ("Exp 5: QFormer", "qformer"),
        "ablation_no_bridge": ("Ablation: No Bridge", "no_bridge"),
    }

    for key, (name, bridge_type) in checkpoints.items():
        checkpoint_dir = Path(f"checkpoints/{key}")
        best_model = checkpoint_dir / "best_model.pt"

        if best_model.exists():
            # Try to load metrics from trainer state
            results.append({
                "name": name,
                "bridge_type": bridge_type,
                "checkpoint": str(best_model),
                "exists": True
            })
        else:
            results.append({
                "name": name,
                "bridge_type": bridge_type,
                "checkpoint": str(best_model),
                "exists": False
            })

    return pd.DataFrame(results)

def print_comparison(df: pd.DataFrame):
    """Print comparison table"""
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS".center(80))
    print("=" * 80)

    print(f"\n{'Experiment':<35} {'Bridge Type':<25} {'Status':<15}")
    print("-" * 80)

    for _, row in df.iterrows():
        status = "✓ Ready" if row['exists'] else "⏳ Pending"
        print(f"{row['name']:<35} {row['bridge_type']:<25} {status:<15}")

    print("-" * 80)
    completed = df['exists'].sum()
    total = len(df)
    print(f"\nCompleted: {completed}/{total} experiments\n")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-dir", default="outputs/ablation")
    parser.add_argument("--export-csv", help="Export results to CSV")
    args = parser.parse_args()

    df = collect_results(args.ablation_dir)
    print_comparison(df)

    if args.export_csv:
        df.to_csv(args.export_csv, index=False)
        print(f"Exported to {args.export_csv}")

if __name__ == "__main__":
    main()
