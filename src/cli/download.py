"""Download the AutoViVQA dataset from Kaggle into ``data/raw/``.

    python -m src.cli.download                # images + all text files
    python -m src.cli.download --texts-only   # skip the ~5 GB image folder

On Kaggle you do not need this — add the ``nguynrichard/auto-vqabest`` dataset to
the notebook and it is mounted read-only under ``/kaggle/input/``.

Needs ``kagglehub`` (``pip install kagglehub``); the dataset is public so no token
is required.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.config.loader import load_config, repo_root

CONFIG = repo_root() / "configs" / "data.yaml"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.cli.download", description=__doc__)
    p.add_argument("--texts-only", action="store_true", help="Skip the image folder (~5 GB).")
    p.add_argument("--force", action="store_true", help="Re-copy even if targets already exist.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    import kagglehub

    cfg = load_config(CONFIG)
    kaggle = cfg["kaggle_setup"]
    paths = cfg["data_path"]
    root = repo_root()
    images_dir = root / paths["raw_images"]
    texts_dir = root / paths["raw_texts"]
    images_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)

    src = Path(kagglehub.dataset_download(kaggle["kaggle_project"]))
    print(f"[kagglehub] cached at {src}")

    # All text files (CSV + the label JSON + split temp files).
    text_src = src / Path(kaggle["text_file"]).parent
    if text_src.is_dir():
        for f in sorted(text_src.iterdir()):
            if f.is_file():
                dst = texts_dir / f.name
                if dst.exists() and not args.force:
                    continue
                shutil.copy2(f, dst)
                print(f"  text  {f.name}")

    if not args.texts_only:
        img_src = src / kaggle["images_folder"]
        if img_src.is_dir():
            existing = {p.name for p in images_dir.iterdir()} if images_dir.exists() else set()
            n = 0
            for f in img_src.iterdir():
                if f.is_file() and (args.force or f.name not in existing):
                    shutil.copy2(f, images_dir / f.name)
                    n += 1
            print(f"  images {n} file(s) -> {images_dir}")

    print("[done] data ready under data/raw/")


if __name__ == "__main__":
    main()
