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
# The 5 bridges of the study (capacity ladder: 1-token -> multi-token -> patch
# attention -> light transformer -> full transformer + text fusion). gated_fusion
# is still a valid bridge_type in the model code but is not part of the suite
# (it is a near-duplicate of `residual`, the weakest arm).
BRIDGES = [
    "residual",
    "multi_token",
    "tile_attention",
    "mini_qformer",
    "qformer",
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
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "seed": args.seed,
        "split_dir": args.split_dir,
        "n_tiles": args.n_tiles,
        "text_metrics_every": args.text_metrics_every,
        "text_metrics_max_samples": args.text_metrics_max_samples,
        "patience": args.patience,
        "resume_from": args.resume,
    }
    if args.no_early_stopping:
        cfg["early_stopping"] = False
    for key, value in cli_overrides.items():
        if value is not None:
            cfg[key] = value

    # Checkpoint dir is <base>/<bridge>[_t<n_tiles>]. --output-dir sets the base;
    # default base is /kaggle/working/checkpoints on Kaggle, else ./checkpoints.
    base = args.output_dir or ("/kaggle/working/checkpoints" if _is_kaggle() else "checkpoints")
    nt = cfg.get("n_tiles") or 1
    cfg["output_dir"] = str(Path(base) / (args.bridge if nt == 1 else f"{args.bridge}_t{nt}"))

    cfg["bridge"] = args.bridge
    cfg.setdefault("resume_from", None)

    if cfg.get("resume_from") == "auto":
        cfg["resume_from"] = _latest_checkpoint(Path(cfg["output_dir"]))
        print(f"[resume] auto -> {cfg['resume_from']}")
    return cfg


def _latest_checkpoint(output_dir: Path) -> str | None:
    """Newest resumable checkpoint: prefer the most recently modified of
    ``last_model.pt`` / ``step_*.pt``, then fall back to ``best_model.pt``."""
    cands = list(output_dir.glob("step_*.pt"))
    last = output_dir / "last_model.pt"
    if last.exists():
        cands.append(last)
    if cands:
        return str(max(cands, key=lambda p: p.stat().st_mtime))
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
    p.add_argument("--grad-accum", type=int, default=None, dest="grad_accum")
    p.add_argument("--lr", type=float, default=None, help="Override learning_rate.")
    p.add_argument("--eval-steps", type=int, default=None, dest="eval_steps")
    p.add_argument("--save-steps", type=int, default=None, dest="save_steps")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-tiles", type=int, default=None, dest="n_tiles",
                   help="Visual budget: InternViT tiles per image (1 = single 336px image).")
    p.add_argument("--text-metrics-every", type=int, default=None, dest="text_metrics_every",
                   help="Generate val text metrics (CIDEr/BLEU/...) every N epochs (default 1). "
                        "The last epoch always runs regardless.")
    p.add_argument("--text-metrics-max-samples", type=int, default=None, dest="text_metrics_max_samples",
                   help="Cap the per-epoch text-metric generation to a seeded N-sample "
                        "subset of val (0 = full val). The final `evaluate` still scores all.")
    p.add_argument("--no-early-stopping", action="store_true", dest="no_early_stopping",
                   help="Disable val-loss early stopping. Recommended for bridge training: "
                        "teacher-forced CE bottoms out in ~1 epoch while greedy-decode CIDEr "
                        "keeps climbing for many more.")
    p.add_argument("--patience", type=int, default=None,
                   help="Early-stopping patience (evals without val-loss improvement). "
                        "Ignored when --no-early-stopping is set.")
    p.add_argument("--split-dir", default=None, dest="split_dir",
                   help="Use data/splits/{train,val,test}.jsonl (final-plan grouped split) "
                        "instead of a random split of the raw CSV.")
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
    limit = cfg.get("limit")
    limit = None if limit is None else max(1, int(limit))
    if cfg.get("split_dir"):
        from src.data.split import load_split

        d = cfg["split_dir"]
        sub = None if limit is None else max(1, limit // 5)
        train_samples = load_split("train", d)[:limit]
        val_samples = load_split("val", d)[:sub]
        test_samples = load_split("test", d)[:sub]
        print(f"[data] grouped split '{d}' -> train={len(train_samples)} "
              f"val={len(val_samples)} test={len(test_samples)}")
    else:
        from src.utils.data_loader_helper import AblationDataLoader

        split = cfg.get("split", {})
        loader = AblationDataLoader(str(REPO_ROOT))
        train_samples, val_samples, test_samples = loader.load_train_val_test_split(
            max_samples=limit,
            train_ratio=split.get("train_ratio", 0.8),
            val_ratio=split.get("val_ratio", 0.1),
            test_ratio=split.get("test_ratio", 0.1),
            seed=cfg["seed"],
        )
        print(f"[data] random split -> train={len(train_samples)} "
              f"val={len(val_samples)} test={len(test_samples)}")

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
        n_tiles=cfg.get("n_tiles") or 1,
        tile_choices=cfg.get("tile_choices"),
        text_metrics_every=cfg.get("text_metrics_every") or 1,
        text_metrics_max_samples=cfg.get("text_metrics_max_samples") or 0,
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
