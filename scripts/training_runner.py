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
        script_map = {
            "residual": "scripts/exp1_residual_bridge.py",
            "better_mlp": "scripts/exp1_residual_bridge.py",  # Backward compat
            "multi_token": "scripts/exp2_multi_token.py",
            "attention_bridge": "scripts/exp3_attention_bridge.py",
            "mini_qformer": "scripts/exp4_mini_qformer.py",
            "qformer": "scripts/exp5_qformer.py",
            "gated_fusion": "scripts/exp6_gated_fusion.py",
            "linear_bridge": "scripts/exp6_gated_fusion.py"  # Backward compat
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
    def __init__(self, config_path: str = "configs/bridge_config.yaml", max_samples: int = None):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.experiments = self._create_experiments()
        self.output_dir = Path(self.config.get("output_dir", "outputs/training"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "results.json"
        self.progress_file = self.output_dir / "progress.json"
        self.max_samples = max_samples

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

        if train_cfg.get("include_linear_bridge", True):
            exps.append(AblationExperiment(
                "Exp 6: Linear",
                "linear_bridge",
                {"experiment_name": "linear_bridge"}
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

        return self._run_bridge_experiment(exp)

    def _run_bridge_experiment(self, exp: AblationExperiment) -> bool:
        """Run bridge experiment script"""
        print_header(f"Running: {exp.name}")
        exp.start_time = time.time()
        
        try:
            # Set PYTHONPATH to project root so imports work
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path.cwd())
            
            cmd = ["python", exp.script]
            if self.max_samples:
                cmd.extend(["--max-samples", str(self.max_samples)])
            
            result = subprocess.run(
                cmd,
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
    parser.add_argument("--config", default="configs/bridge_config.yaml",
                        help="Ablation config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="List experiments without running")
    parser.add_argument("--experiments", help="Run specific experiments: 1,3,5-7")
    parser.add_argument("--rerun", help="Rerun specific experiments from scratch: 1,3,5")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore previous progress, run all from scratch")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to use (e.g., 100 for testing)")
    
    args = parser.parse_args()
    
    # Auto-detect Kaggle environment and limit samples if not specified
    if args.max_samples is None and os.path.exists('/kaggle/working'):
        args.max_samples = 100
        print_info(f"🔍 Kaggle detected → Auto-limit to 100 samples (override with --max-samples)")
    
    study = AblationStudy(args.config, max_samples=args.max_samples)
    
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
