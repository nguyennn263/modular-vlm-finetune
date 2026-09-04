"""Oracle utility-cost sweep (final-plan D2 / P4).

    python -m src.cli.oracle --bridges mini_qformer,qformer,residual \
        --n-tiles 1,3,6 --subset 7500 --ckpt-dir checkpoints/expA/seed42

For every (sample, n_tiles, bridge) it greedily generates an answer and scores
per-sample CIDEr -> M(a;x). C(a) = n_tiles / max(n_tiles). Writes
outputs/oracle/{table.parquet, labels.parquet} (the latter = a*(x, λ) for the
7-point λ grid). Heavy: |subset| * |n_tiles| * |bridges| generate() calls —
launch deliberately.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.config.loader import repo_root


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.cli.oracle", description=__doc__)
    p.add_argument("--bridges", default="mini_qformer,qformer,residual",
                   help="comma list; top-3 from Exp B fork")
    p.add_argument("--n-tiles", default="1,3,6", dest="n_tiles")
    p.add_argument("--split", default="train")
    p.add_argument("--split-dir", default="data/splits")
    p.add_argument("--subset", type=int, default=7500, help="stratified by category")
    p.add_argument("--ckpt-dir", default="checkpoints/expA/seed42", dest="ckpt_dir",
                   help="<ckpt-dir>/<bridge>/best_model.pt (tile-augmented bridge preferred)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard", default=None, help="I/N — process only shard I of N (0-indexed).")
    p.add_argument("--out", default="outputs/oracle")
    p.add_argument("--dry-run", action="store_true")
    return p


def _stratified_subset(samples, k: int, seed: int):
    import random
    from collections import defaultdict

    by_cat = defaultdict(list)
    for s in samples:
        by_cat[(s.metadata or {}).get("category")].append(s)
    rng = random.Random(seed)
    per = max(1, k // max(1, len(by_cat)))
    out = []
    for cat, lst in by_cat.items():
        rng.shuffle(lst)
        out += lst[:per]
    rng.shuffle(out)
    return out[:k]


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("TRANSFORMERS_NO_META_DEVICE", "1")
    import pandas as pd
    import torch
    from transformers import AutoModel

    from src.analysis.oracle import oracle_labels
    from src.config.loader import load_config
    from src.data.split import load_split
    from src.training import BridgeTrainer, TrainConfig, create_finetune_model

    bridges = [b.strip() for b in args.bridges.split(",") if b.strip()]
    n_tiles = [int(x) for x in args.n_tiles.split(",")]
    model_name = load_config(repo_root() / "configs" / "train.yaml")["model_name"]
    bridge_cfgs = {b: load_config(repo_root() / "configs" / "bridges" / f"{b}.yaml") for b in bridges}

    samples = _stratified_subset(load_split(args.split, args.split_dir), args.subset, args.seed)
    out_name = "table.parquet"
    if args.shard:
        i, nsh = (int(x) for x in args.shard.split("/"))
        samples = samples[i::nsh]                       # deterministic strided shard
        out_name = f"table.shard{i}of{nsh}.parquet"
    sample_id = [f"{(s.metadata or {}).get('image_id')}::{s.question}" for s in samples]
    print(f"[oracle] {len(samples)} samples x {len(n_tiles)} n_tiles x {len(bridges)} bridges "
          f"= {len(samples) * len(n_tiles) * len(bridges)} generate() calls  (shard={args.shard})")
    if args.dry_run:
        return

    rows = []
    for b in bridges:
        base = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                        low_cpu_mem_usage=False, trust_remote_code=True).eval()
        bdir = Path(args.ckpt_dir) / b
        if not bdir.is_absolute():
            bdir = repo_root() / bdir
        ckpt_path = bdir / "last_model.pt" if (bdir / "last_model.pt").exists() else bdir / "best_model.pt"
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = create_finetune_model(base, bridge_type=bridge_cfgs[b]["bridge_type"],
                                      bridge_config=bridge_cfgs[b].get("bridge_config") or {},
                                      lora={} if "lora_state" in ck else None)
        model.bridge.load_state_dict(ck.get("bridge_state", ck))
        if "lora_state" in ck and hasattr(model, "load_lora_state_dict"):
            model.load_lora_state_dict(ck["lora_state"])
        print(f"[oracle] {b}: loaded {ckpt_path}"
              + (" (+LoRA)" if "lora_state" in ck else ""))

        for n in n_tiles:
            tc = TrainConfig(model_name=model_name, output_dir=str(repo_root() / args.out / f"{b}_t{n}"),
                             n_tiles=n)
            trainer = BridgeTrainer(model, samples, samples, tc)
            trainer._compute_epoch_text_metrics(0)  # noqa: SLF001 — writes per-sample cider json
            per_sample_cider = _read_per_sample_cider(repo_root() / args.out / f"{b}_t{n}")
            for sid, cider in zip(sample_id, per_sample_cider):
                rows.append({"sample_id": sid, "action": f"{b}|t{n}", "bridge": b,
                             "n_tiles": n, "M": float(cider), "C": n / max(n_tiles)})
        del model, base
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    table = pd.DataFrame(rows)
    out = repo_root() / args.out
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / out_name, index=False)
    if not args.shard:                       # full run -> also emit labels
        oracle_labels(table).to_parquet(out / "labels.parquet", index=False)
        print(f"[oracle] {len(table)} rows -> {out}/table.parquet + labels.parquet")
    else:
        print(f"[oracle] {len(table)} rows -> {out}/{out_name}  (merge shards with src.analysis.merge)")


def _read_per_sample_cider(results_root: Path) -> list[float]:
    import json
    d = json.loads((results_root / "results" / "text_metrics_epoch_1.json").read_text())
    return d["details"]["cider"]["per_sample"]


def main(argv: list[str] | None = None) -> None:
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
