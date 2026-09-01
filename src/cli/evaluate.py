"""Evaluate a trained bridge checkpoint on a data split.

    python -m src.cli.evaluate --bridge residual --checkpoint checkpoints/residual/best_model.pt
    python -m src.cli.evaluate --bridge qformer  --checkpoint <path> --split test --limit 500

Reports loss / perplexity plus generation metrics (BLEU-4, METEOR, ROUGE-L, CIDEr,
precision/recall/F1, exact-match, WUPS) and writes them to ``<checkpoint_dir>/eval_<split>.json``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from src.cli.train import BRIDGES, REPO_ROOT, _load_yaml


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.cli.evaluate",
        description="Evaluate a trained bridge checkpoint on a data split.",
    )
    p.add_argument("--bridge", required=True, choices=BRIDGES)
    p.add_argument("--checkpoint", required=True, help="Path to a *.pt checkpoint (bridge_state).")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=None, help="Evaluate at most N samples.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="Where to write the JSON report.")
    return p


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("TRANSFORMERS_NO_META_DEVICE", "1")
    import torch
    from transformers import AutoModel

    from src.training import BridgeTrainer, TrainConfig, create_finetune_model
    from src.utils.data_loader_helper import AblationDataLoader

    train_cfg = _load_yaml(REPO_ROOT / "configs" / "train.yaml")
    bridge_cfg = _load_yaml(REPO_ROOT / "configs" / "bridges" / f"{args.bridge}.yaml")
    split_cfg = train_cfg.get("split", {})

    loader = AblationDataLoader(str(REPO_ROOT))
    train_s, val_s, test_s = loader.load_train_val_test_split(
        max_samples=None,
        train_ratio=split_cfg.get("train_ratio", 0.8),
        val_ratio=split_cfg.get("val_ratio", 0.1),
        test_ratio=split_cfg.get("test_ratio", 0.1),
        seed=args.seed,
    )
    chosen = {"train": train_s, "val": val_s, "test": test_s}[args.split]
    if args.limit:
        chosen = chosen[: args.limit]
    print(f"[data] evaluating on {len(chosen)} {args.split} samples")

    base_model = AutoModel.from_pretrained(
        train_cfg["model_name"], torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False, trust_remote_code=True,
    ).eval()
    model = create_finetune_model(
        base_model, bridge_type=bridge_cfg["bridge_type"],
        bridge_config=bridge_cfg.get("bridge_config") or {},
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("bridge_state", ckpt)
    model.bridge.load_state_dict(state)
    print(f"[ckpt] loaded bridge weights from {args.checkpoint}")

    # The trainer wires tokenizer + loaders + the generation-metric routines.
    # Put the chosen split in the val slot so its eval helpers operate on it.
    tc = TrainConfig(model_name=train_cfg["model_name"], output_dir=str(Path(args.checkpoint).parent))
    trainer = BridgeTrainer(model, [], chosen, tc)

    report = {"split": args.split, "n": len(chosen), "checkpoint": args.checkpoint}
    report.update(trainer.evaluate())
    try:
        report.update(trainer._compute_epoch_text_metrics(0))
    except Exception as exc:  # generation metrics are best-effort
        report["generation_metrics_error"] = repr(exc)

    out = Path(args.output or Path(args.checkpoint).parent / f"eval_{args.split}.json")
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[report] {out}\n" + json.dumps(report, indent=2, default=str))
    return report


def main(argv: list[str] | None = None) -> None:
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
