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
    p.add_argument("--split-dir", default=None, dest="split_dir",
                   help="Use data/splits/<split>.jsonl (grouped split, carries `category`) "
                        "instead of a random split of the raw CSV.")
    p.add_argument("--n-tiles", type=int, default=1, dest="n_tiles")
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

    if args.split_dir:
        from src.data.split import load_split
        chosen = load_split(args.split, args.split_dir)
    else:
        loader = AblationDataLoader(str(REPO_ROOT))
        parts = loader.load_train_val_test_split(
            max_samples=None,
            train_ratio=split_cfg.get("train_ratio", 0.8),
            val_ratio=split_cfg.get("val_ratio", 0.1),
            test_ratio=split_cfg.get("test_ratio", 0.1),
            seed=args.seed,
        )
        chosen = dict(zip(("train", "val", "test"), parts))[args.split]
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

    # BridgeTrainer only builds the tokenizer + collate_fn when train_dataset is
    # non-empty, and its generation-metric helper reads self.val_dataset — so pass
    # `chosen` in both slots. We never call trainer.train() here.
    tc = TrainConfig(model_name=train_cfg["model_name"],
                     output_dir=str(Path(args.checkpoint).parent), n_tiles=args.n_tiles)
    trainer = BridgeTrainer(model, chosen, chosen, tc)

    report = {"split": args.split, "n": len(chosen), "bridge": args.bridge,
              "n_tiles": args.n_tiles, "checkpoint": args.checkpoint}
    try:
        report.update(trainer.evaluate())
    except Exception as exc:
        report["loss_eval_error"] = repr(exc)
    try:
        report.update(trainer._compute_epoch_text_metrics(0))
    except Exception as exc:  # generation metrics are best-effort
        report["generation_metrics_error"] = repr(exc)

    ckpt_dir = Path(args.checkpoint).parent
    out = Path(args.output or ckpt_dir / f"eval_{args.split}.json")
    out.write_text(json.dumps(report, indent=2, default=str))

    # Per-sample scores + category (for Exp B). Aligns with `chosen` order.
    _dump_per_sample(ckpt_dir, chosen, args)

    print(f"[report] {out}\n" + json.dumps(
        {k: v for k, v in report.items() if not isinstance(v, (list, dict))}, indent=2))
    return report


def _dump_per_sample(ckpt_dir: Path, chosen, args) -> None:
    metrics_file = ckpt_dir / "results" / "text_metrics_epoch_1.json"
    if not metrics_file.exists():
        return
    details = json.loads(metrics_file.read_text()).get("details", {})
    per = {m: details.get(m, {}).get("per_sample", []) for m in
           ("meteor", "rouge_l", "cider", "exact_match", "wups@0.9")}
    out = ckpt_dir / f"eval_{args.split}_samples.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for i, s in enumerate(chosen):
            rec = {
                "image_id": (s.metadata or {}).get("image_id"),
                "question": s.question,
                "category": (s.metadata or {}).get("category"),
                "bridge": args.bridge,
                "n_tiles": args.n_tiles,
            }
            for m, arr in per.items():
                rec[m.replace("@0.9", "")] = float(arr[i]) if i < len(arr) else None
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[per-sample] {out}")


def main(argv: list[str] | None = None) -> None:
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
