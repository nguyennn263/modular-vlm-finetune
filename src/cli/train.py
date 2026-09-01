"""Unified bridge training entry point.

    python -m src.cli.train --bridge residual
    python -m src.cli.train --bridge qformer --limit 2000 --epochs 5
    python -m src.cli.train --bridge multi_token --smoke
    python -m src.cli.train --bridge residual --resume checkpoints/residual/last.pt

This replaces the six near-identical ``scripts/expN_*.py`` scripts. There is NO
implicit sample cap on any platform — pass ``--limit`` (or ``--smoke``) yourself
when you want a short run. On Kaggle the only change is the default output dir
(``/kaggle/working/checkpoints/<bridge>``); the full dataset is used unless limited.

Config precedence (low -> high):
    configs/train.yaml  <  configs/bridges/<bridge>.yaml  <  --smoke preset  <  CLI flags
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGES = [
    "residual",
    "multi_token",
    "tile_attention",
    "mini_qformer",
    "qformer",
    "gated_fusion",
]


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _is_kaggle() -> bool:
    # Kept dependency-free on purpose (no torch import at module load).
    return Path("/kaggle/working").exists() or Path("/kaggle/input").exists()


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the full run configuration. Pure / import-light (no torch)."""
    cfg = _load_yaml(REPO_ROOT / "configs" / "train.yaml")
    bridge_cfg = _load_yaml(REPO_ROOT / "configs" / "bridges" / f"{args.bridge}.yaml")
    if not bridge_cfg:
        raise SystemExit(f"No config for bridge '{args.bridge}' at configs/bridges/{args.bridge}.yaml")
    cfg = _deep_merge(cfg, bridge_cfg)

    smoke_preset = cfg.pop("smoke", {}) or {}
    cfg.setdefault("limit", None)

    if args.smoke:
        cfg = _deep_merge(cfg, smoke_preset)

    # Explicit CLI flags win over everything.
    cli_overrides = {
        "limit": args.limit,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "resume_from": args.resume,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            cfg[key] = value

    # Kaggle: only the default output location changes.
    if args.output_dir is None and _is_kaggle():
        cfg["output_dir"] = f"/kaggle/working/checkpoints/{args.bridge}"
    cfg.setdefault("output_dir", f"checkpoints/{args.bridge}")

    cfg["bridge"] = args.bridge
    cfg.setdefault("resume_from", None)

    if cfg.get("resume_from") == "auto":
        cfg["resume_from"] = _latest_checkpoint(Path(cfg["output_dir"]))
        print(f"[resume] auto -> {cfg['resume_from']}")
    return cfg


def _latest_checkpoint(output_dir: Path) -> str | None:
    """Newest ``step_*.pt`` in ``output_dir`` (falls back to ``best_model.pt``)."""
    steps = sorted(output_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    if steps:
        return str(steps[-1])
    best = output_dir / "best_model.pt"
    return str(best) if best.exists() else None


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.cli.train",
        description="Train a single bridge module on top of frozen Vintern-1B.",
    )
    p.add_argument("--bridge", required=True, choices=BRIDGES, help="Bridge architecture to train.")
    p.add_argument("--limit", type=int, default=None,
                   help="Use at most N samples total (default: whole dataset).")
    p.add_argument("--smoke", action="store_true",
                   help="Fast sanity run: applies the 'smoke' preset from configs/train.yaml.")
    p.add_argument("--epochs", type=int, default=None, help="Override num_epochs.")
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--lr", type=float, default=None, help="Override learning_rate.")
    p.add_argument("--eval-steps", type=int, default=None, dest="eval_steps")
    p.add_argument("--save-steps", type=int, default=None, dest="save_steps")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None, dest="output_dir",
                   help="Checkpoint dir (default: checkpoints/<bridge>, or /kaggle/working/... on Kaggle).")
    p.add_argument("--resume", default=None, nargs="?", const="auto",
                   help="Checkpoint to resume from. Bare --resume (or --resume auto) "
                        "picks the newest step_*.pt in the output dir.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the resolved config and exit (no model load, no training).")
    return p


def run(cfg: dict[str, Any]) -> None:
    """Execute training. Heavy imports live here so --help / --dry-run need no torch."""
    os.environ.setdefault("TRANSFORMERS_NO_META_DEVICE", "1")
    import torch
    from transformers import AutoModel

    from src.training import BridgeTrainer, TrainConfig, create_finetune_model
    from src.utils.data_loader_helper import AblationDataLoader

    split = cfg.get("split", {})
    loader = AblationDataLoader(str(REPO_ROOT))
    train_samples, val_samples, test_samples = loader.load_train_val_test_split(
        max_samples=cfg.get("limit"),
        train_ratio=split.get("train_ratio", 0.8),
        val_ratio=split.get("val_ratio", 0.1),
        test_ratio=split.get("test_ratio", 0.1),
        seed=cfg["seed"],
    )
    print(f"[data] train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    base_model = AutoModel.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    ).eval()
    model = create_finetune_model(
        base_model,
        bridge_type=cfg["bridge_type"],
        bridge_config=cfg.get("bridge_config") or {},
    )

    train_config = TrainConfig(
        model_name=cfg["model_name"],
        output_dir=cfg["output_dir"],
        num_epochs=cfg["num_epochs"],
        batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=cfg["warmup_steps"],
        max_grad_norm=cfg["max_grad_norm"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        seed=cfg["seed"],
        num_workers=cfg["num_workers"],
        early_stopping=cfg["early_stopping"],
        patience=cfg["patience"],
        min_delta=cfg["min_delta"],
        save_best=cfg["save_best"],
        resume_from=cfg.get("resume_from"),
    )
    trainer = BridgeTrainer(model, train_samples, val_samples, train_config, test_dataset=test_samples or None)
    trainer.train()


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    cfg = build_run_config(args)
    print("[config]\n" + json.dumps(cfg, indent=2, default=str))
    if args.dry_run:
        return
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    run(cfg)


if __name__ == "__main__":
    main()
