"""Build f(I,Q): the cheap visual-state features (final-plan P4).

    python -m src.cli.build_fiq --split-dir data/splits --splits train,val

Per sample: pooled InternViT CLS at n_tiles=1 (1024-d, optionally PCA-reduced) +
a few cheap signals (question length, and image clarity/occlusion/object-density
from the quality CSV if data/labeled.parquet is present). Does NOT use `category`.
Writes outputs/fiq/<split>.parquet keyed by sample_id.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.config.loader import repo_root

_META_COLS = ["eip_img_clarity_Score", "eip_img_occlusion_Score", "eip_img_object_density_Score"]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.cli.build_fiq", description=__doc__)
    p.add_argument("--split-dir", default="data/splits")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--pca", type=int, default=64, help="reduce the 1024-d CLS to this (0 = keep raw)")
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    p.add_argument("--out", default="outputs/fiq")
    p.add_argument("--dry-run", action="store_true")
    return p


def run(args: argparse.Namespace) -> None:
    from src.data.split import load_split

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if args.dry_run:
        for s in splits:
            print(f"[fiq] {s}: {len(load_split(s, args.split_dir))} samples")
        return

    os.environ.setdefault("TRANSFORMERS_NO_META_DEVICE", "1")
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModel

    from src.config.loader import load_config
    from src.data.tiling import load_image_tiles

    model_name = load_config(repo_root() / "configs" / "train.yaml")["model_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                      low_cpu_mem_usage=False, trust_remote_code=True
                                      ).eval().to(device).vision_model
    dtype = next(vision.parameters()).dtype

    meta = None
    lp = repo_root() / "data" / "labeled.parquet"
    if lp.exists():
        m = pd.read_parquet(lp)
        keep = [c for c in _META_COLS if c in m.columns]
        if keep:
            m["sample_id"] = m["image_id"].astype(str) + "::" + m["question"]
            meta = m.drop_duplicates("sample_id").set_index("sample_id")[keep]

    outdir = repo_root() / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    pca_path = outdir / "pca.npz"
    proj = mean = None
    if args.pca and pca_path.exists():
        z = np.load(pca_path)
        proj, mean = z["proj"], z["mean"]
        print(f"[fiq] loaded PCA basis {proj.shape} from {pca_path}")

    for split in splits:
        samples = load_split(split, args.split_dir)
        ids = [f"{(s.metadata or {}).get('image_id')}::{s.question}" for s in samples]
        feats = []
        for i in range(0, len(samples), args.batch_size):
            batch = samples[i:i + args.batch_size]
            pv = torch.stack([load_image_tiles(s.image_path, 1).squeeze(0) for s in batch])
            with torch.no_grad():
                vo = vision(pv.to(device=device, dtype=dtype))
                hs = vo.last_hidden_state if hasattr(vo, "last_hidden_state") else vo
            feats.append(hs[:, 0].float().cpu().numpy())  # CLS
        cls = np.concatenate(feats)  # (N, 1024)

        if args.pca:
            if proj is None:  # fit once (on the first split seen, normally train) then persist
                mean = cls.mean(0)
                _, _, vt = np.linalg.svd(cls - mean, full_matrices=False)
                proj = vt[: args.pca].T
                np.savez(pca_path, proj=proj, mean=mean)
                print(f"[fiq] fit PCA basis {proj.shape} on '{split}' -> {pca_path}")
            cls = (cls - mean) @ proj

        df = pd.DataFrame(cls, columns=[f"f{j}" for j in range(cls.shape[1])])
        df.insert(0, "sample_id", ids)
        df["q_len"] = [len(s.question.split()) for s in samples]
        if meta is not None:
            df = df.merge(meta.reset_index(), on="sample_id", how="left")
            df[[c for c in meta.columns]] = df[[c for c in meta.columns]].fillna(0.0)

        df = df.drop_duplicates("sample_id").reset_index(drop=True)
        df.to_parquet(outdir / f"{split}.parquet", index=False)
        print(f"[fiq] {split}: {df.shape} -> {outdir}/{split}.parquet")


def main(argv: list[str] | None = None) -> None:
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
