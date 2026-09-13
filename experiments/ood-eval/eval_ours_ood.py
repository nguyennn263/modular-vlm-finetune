"""Run our best checkpoint (Multi-Token bridge + decoder LoRA r16, 3ep, seed42)
on an OOD dataset built by build_ood_data.py. Same model-loading path as
`src.cli.evaluate` (this repo's real eval entrypoint) minus the AutoViVQA-specific
`load_split`/`resolve_dirs` step -- everything else (BridgeTrainer, LoRA
auto-detection, generation settings, output format) is identical, so the number
is produced by the SAME code that produced every other row in this project's
results tables.

Requires the `feat/decoder-lora` branch (LoRA-aware create_finetune_model /
src.cli.evaluate). Writes results/text_predictions_epoch_1.json (bridge-pipeline
format) under --output-dir; score with experiments/vintern-ft/score_local.py.

    python experiments/ood-eval/eval_ours_ood.py \
        --data /kaggle/working/data/vitextvqa/ours.jsonl \
        --images-dir /kaggle/working/data/vitextvqa/images \
        --checkpoint /kaggle/working/ck/multi_token/model.pt \
        --output-dir /kaggle/working/out/vitextvqa/ours
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="ours.jsonl from build_ood_data.py")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bridge", default="multi_token")
    p.add_argument("--n-tiles", type=int, default=1, dest="n_tiles")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True, dest="output_dir")
    p.add_argument("--limit", type=int, default=None)
    return p


def main() -> None:
    a = _parser().parse_args()
    import torch
    from transformers import AutoModel

    from src.cli.train import BRIDGES, _load_yaml as _ly  # noqa: F401 (sanity: branch has the LoRA-aware CLI)
    from src.training import BridgeTrainer, TrainConfig, create_finetune_model
    from src.schema.data_schema import OneSample

    images_dir = Path(a.images_dir)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]
    chosen = [OneSample(image_path=str(images_dir / r["image_name"]), question=r["question"],
                        answers=list(r["answers"]), metadata={"id": r.get("id")}) for r in rows]
    print(f"[data] {len(chosen)} OOD samples from {a.data}")

    train_cfg = _load_yaml(REPO_ROOT / "configs" / "train.yaml")
    bridge_cfg = _load_yaml(REPO_ROOT / "configs" / "bridges" / f"{a.bridge}.yaml")

    base_model = AutoModel.from_pretrained(
        train_cfg["model_name"], torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False, trust_remote_code=True).eval()
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    lora_cfg = {} if "lora_state" in ckpt else None
    model = create_finetune_model(base_model, bridge_type=bridge_cfg["bridge_type"],
                                  bridge_config=bridge_cfg.get("bridge_config") or {}, lora=lora_cfg)
    model.bridge.load_state_dict(ckpt.get("bridge_state", ckpt))
    if "lora_state" in ckpt and hasattr(model, "load_lora_state_dict"):
        model.load_lora_state_dict(ckpt["lora_state"])
        print("[ckpt] loaded LoRA adapter")
    print(f"[ckpt] loaded bridge weights from {a.checkpoint}")

    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    tc = TrainConfig(model_name=train_cfg["model_name"], output_dir=a.output_dir, n_tiles=a.n_tiles)
    trainer = BridgeTrainer(model, chosen, chosen, tc)

    report = {"dataset": Path(a.data).parent.name, "n": len(chosen), "bridge": a.bridge,
              "n_tiles": a.n_tiles, "checkpoint": a.checkpoint}
    try:
        report.update(trainer._compute_epoch_text_metrics(0))
    except Exception as exc:
        report["generation_metrics_error"] = repr(exc)

    out = Path(a.output_dir) / "eval_ood.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[report] {out}\n" + json.dumps(
        {k: v for k, v in report.items() if not isinstance(v, (list, dict))}, indent=2))


if __name__ == "__main__":
    main()
