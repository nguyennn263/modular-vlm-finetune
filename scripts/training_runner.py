#!/usr/bin/env python3
"""
Bridge Ablation Study Runner
Tự động chạy ablation experiments: full bridge → no bridge → components
"""

import sys
import argparse
import json
import time
import os
from pathlib import Path
from datetime import datetime
import subprocess
import yaml
from typing import List, Dict, Tuple

BLUE = "\033[0;34m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
NC = "\033[0m"

def print_header(text):
    print(f"\n{BLUE}{'=' * 70}{NC}")
    print(f"{BLUE}{text:^70}{NC}")
    print(f"{BLUE}{'=' * 70}{NC}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{NC}")

def print_info(text):
    print(f"{YELLOW}ℹ {text}{NC}")

def print_error(text):
    print(f"{RED}✗ {text}{NC}")

class AblationExperiment:
    def __init__(self, name: str, bridge_type: str, config_override: Dict = None):
        self.name = name
        self.bridge_type = bridge_type
        self.config_override = config_override or {}
        self.script = self._get_script_path()
        self.status = "pending"
        self.result = None
        self.start_time = None
        self.end_time = None
        
    def _get_script_path(self) -> str:
        if self.bridge_type == "full_freeze":
            return None
        script_map = {
            "better_mlp": "scripts/exp1_better_mlp.py",
            "multi_token": "scripts/exp2_multi_token.py",
            "attention_bridge": "scripts/exp3_attention_bridge.py",
            "mini_qformer": "scripts/exp4_mini_qformer.py",
            "qformer": "scripts/exp5_qformer.py"
        }
        return script_map.get(self.bridge_type)

    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) / 60
        return 0.0

    def to_dict(self):
        return {
            "name": self.name,
            "bridge_type": self.bridge_type,
            "status": self.status,
            "duration_minutes": round(self.duration, 1),
            "config_override": self.config_override,
            "result": self.result
        }

class AblationStudy:
    def __init__(self, config_path: str = "configs/ablation_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.experiments = self._create_experiments()
        self.output_dir = Path(self.config.get("output_dir", "outputs/training"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "results.json"
        self.progress_file = self.output_dir / "progress.json"

    def _load_config(self) -> Dict:
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return self._default_config()

    def _default_config(self) -> Dict:
        return {
            "name": "bridge_training_comparison",
            "num_epochs": 10,
            "batch_size": 8,
            "learning_rate": 2e-4,
            "output_dir": "outputs/training",
            "resume": True,
            "training": {
                "include_no_bridge": True
            }
        }

    def _create_experiments(self) -> List[AblationExperiment]:
        exps = []
        train_cfg = self.config.get("training", {})

        exps.extend([
            AblationExperiment("Exp 1: BetterMLP", "better_mlp"),
            AblationExperiment("Exp 2: MultiToken", "multi_token"),
            AblationExperiment("Exp 3: AttentionBridge", "attention_bridge"),
            AblationExperiment("Exp 4: MiniQFormer", "mini_qformer"),
            AblationExperiment("Exp 5: QFormer", "qformer"),
        ])

        if train_cfg.get("include_no_bridge", True):
            exps.append(AblationExperiment(
                "Baseline: Full Vintern Freeze",
                "full_freeze",
                {"experiment_name": "baseline_full_freeze"}
            ))

        return exps

    def dry_run(self):
        """List experiments without running"""
        print_header(f"DRY RUN: {self.config['name']}")
        print(f"Total experiments: {len(self.experiments)}\n")
        
        for i, exp in enumerate(self.experiments, 1):
            bridge_info = f"{exp.bridge_type}"
            print(f"  [{i:2d}] {exp.name:<40} — {bridge_info}")
        
        print(f"\n{' Experiment Type Summary ':-^70}")
        experiments = sum(1 for e in self.experiments if "Exp" in e.name)
        baselines = sum(1 for e in self.experiments if "Baseline" in e.name)
        
        if experiments > 0:
            print(f"  Bridge Experiments: {experiments}")
        if baselines > 0:
            print(f"  Baselines:          {baselines}")

    def load_progress(self) -> Dict:
        """Load experiment progress"""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                return json.load(f)
        return {}

    def save_progress(self, progress: Dict):
        """Save experiment progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def run_single(self, exp: AblationExperiment, force_rerun: bool = False):
        """Run single experiment"""
        progress = self.load_progress()
        exp_key = exp.name

        if not force_rerun and progress.get(exp_key, {}).get("status") == "completed":
            print_info(f"Skipping {exp.name} (already completed)")
            exp.status = "skipped"
            return True

        if exp.bridge_type == "full_freeze":
            return self._run_ablation_no_bridge(exp)
        else:
            return self._run_bridge_experiment(exp)

    def _run_bridge_experiment(self, exp: AblationExperiment) -> bool:
        """Run bridge experiment script"""
        print_header(f"Running: {exp.name}")
        exp.start_time = time.time()
        
        try:
            # Set PYTHONPATH to project root so imports work
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path.cwd())
            
            result = subprocess.run(
                ["python", exp.script],
                check=True,
                capture_output=False,
                env=env
            )
            exp.status = "completed"
            exp.end_time = time.time()
            print_success(f"{exp.name} completed ({exp.duration:.1f}m)")
            return True
        except subprocess.CalledProcessError as e:
            exp.status = "failed"
            exp.end_time = time.time()
            print_error(f"{exp.name} failed")
            return False

    def _run_ablation_no_bridge(self, exp: AblationExperiment) -> bool:
        """Create and run no-bridge ablation (frozen models only)"""
        print_header(f"Running: {exp.name}")
        exp.start_time = time.time()

        script_content = '''
import sys
from pathlib import Path

# Add workspace root to path so src module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModel
from src.training import BridgeTrainer, TrainConfig
from src.data.loaders import load_datasets
from utils.path_management import RAW_TEXT_CSV, RAW_IMAGES_DIR

# Disable meta device to prevent meta tensor issues
import os
os.environ["TRANSFORMERS_NO_META_DEVICE"] = "1"

# Load base model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = AutoModel.from_pretrained(
    "5CD-AI/Vintern-1B-v3_5",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    trust_remote_code=True,
).eval().to(device)

# Create minimal wrapper (no bridge = frozen models only)
class NoOpBridge(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 896, device=x.device)

if not hasattr(base_model, 'bridge'):
    base_model.bridge = NoOpBridge()

# Load data
train_ds, val_ds = load_datasets(
    csv_path=str(RAW_TEXT_CSV),
    images_dir=str(RAW_IMAGES_DIR),
    val_ratio=0.1
)

# Freeze all
for param in base_model.parameters():
    param.requires_grad = False

# Train config
config = TrainConfig(
    output_dir="checkpoints/ablation_no_bridge",
    num_epochs=10,
    batch_size=8,
    learning_rate=2e-4,
    eval_steps=100
)

# Trainer
trainer = BridgeTrainer(base_model, train_ds, val_ds, config)
trainer.train()

print("✓ No-bridge ablation completed")
'''
        
        # Create ablation script in workspace root
        workspace_root = Path(__file__).parent.parent
        ablation_script = workspace_root / "scripts" / "_ablation_no_bridge.py"
        ablation_script.write_text(script_content)

        try:
            # Run from workspace root so imports work correctly
            result = subprocess.run(
                ["python", "scripts/_ablation_no_bridge.py"],
                cwd=workspace_root,
                check=True
            )
            exp.status = "completed"
            ablation_script.unlink()
            print_success(f"{exp.name} completed ({exp.duration:.1f}m)")
            return True
        except Exception as e:
            exp.status = "failed"
            print_error(f"{exp.name} failed: {e}")
            ablation_script.unlink(missing_ok=True)
            return False
        finally:
            exp.end_time = time.time()

    def run(self, experiment_ids: str = None, rerun_ids: str = None, resume: bool = True):
        """Run selected experiments"""
        print_header(f"ABLATION STUDY: {self.config['name']}")
        
        # Parse experiment IDs
        selected = self._parse_exp_ids(experiment_ids) if experiment_ids else None
        rerun = self._parse_exp_ids(rerun_ids) if rerun_ids else set()

        progress = {} if not resume else self.load_progress()

        results = []
        for i, exp in enumerate(self.experiments, 1):
            if selected and i not in selected:
                continue

            force_rerun = i in rerun
            if self.run_single(exp, force_rerun=force_rerun):
                results.append(exp.to_dict())

            progress[exp.name] = {
                "status": exp.status,
                "duration_minutes": exp.duration,
                "timestamp": datetime.now().isoformat()
            }
            self.save_progress(progress)

        self._save_results(results)
        self._print_summary(results)

    def _parse_exp_ids(self, ids_str: str) -> set:
        """Parse experiment IDs: '1,3,5-7' → {1, 3, 5, 6, 7}"""
        result = set()
        for part in ids_str.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                result.update(range(start, end + 1))
            else:
                result.add(int(part))
        return result

    def _save_results(self, results: List[Dict]):
        """Save results to JSON"""
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print_success(f"Results saved: {self.results_file}")

    def _print_summary(self, results: List[Dict]):
        """Print summary table"""
        print_header("SUMMARY")
        print(f"{'Experiment':<45} {'Status':<12} {'Duration (m)':<15}")
        print("-" * 72)
        
        total_time = 0
        for result in results:
            status = f"{GREEN}{result['status'].upper()}{NC}"
            duration = f"{result['duration_minutes']:.1f}"
            print(f"{result['name']:<45} {status:<12} {duration:<15}")
            total_time += result['duration_minutes']
        
        print("-" * 72)
        print(f"Total time: {total_time/60:.1f} hours\n")

def main():
    parser = argparse.ArgumentParser(description="Bridge Ablation Study Runner")
    parser.add_argument("--config", default="configs/ablation_config.yaml",
                        help="Ablation config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="List experiments without running")
    parser.add_argument("--experiments", help="Run specific experiments: 1,3,5-7")
    parser.add_argument("--rerun", help="Rerun specific experiments from scratch: 1,3,5")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore previous progress, run all from scratch")
    
    args = parser.parse_args()
    
    study = AblationStudy(args.config)
    
    if args.dry_run:
        study.dry_run()
    else:
        resume = not args.no_resume
        study.run(
            experiment_ids=args.experiments,
            rerun_ids=args.rerun,
            resume=resume
        )

if __name__ == "__main__":
    main()
