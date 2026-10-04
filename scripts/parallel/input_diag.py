#!/usr/bin/env python3
"""Eval-only diagnostic of how pooled bridges see the image -- one job per account.

Two things the reported numbers never controlled for:
  1. generation at n_tiles=1 fed the FIRST 448px crop of the dynamic tiling
     (top-left ~1/6 of every val image) while training saw the whole image at 336px
     -> --gen-image full
  2. multi_token reads the CLS token at one tile but the mean of every token at
     T > 1, so RQ3's tile "collapse" mixes tile count with input type
     -> --pooled-input {mean_all, cls_mean}

    python scripts/parallel/input_diag.py launch
    python scripts/parallel/input_diag.py collect
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, ACCT_DIR, _kaggle, _user, _code, _clone_cell, _nb, load_ledger, save_ledger, _register  # noqa

BRANCH = "exp/eval-input-diagnostic"
PLAIN_DS = "mvlm-test-ckpt"      # <label>/model.pt, multi_token 2-epoch plain bridge
LORA_DS = "mvlm-lora-mt-ckpt"    # <label>.pt, multi_token + decoder LoRA r16
# Kaggle moved new kernels to a Python 3.13 image after 2026-09-22; setup_kaggle.sh's
# torch 2.2.2 / numpy<2 / transformers 4.38.2 pins have no 3.13 wheels. This is the
# image the 2026-09-22 token-sweep kernels ran on.
DOCKER_IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
                "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461")
ACCS = ["acc1", "acc2", "acc3", "acc4", "acc5", "acc7", "acc8",
        "acc10", "acc11", "acc12", "acc13", "acc15", "acc16"]

# (tag, ckpt glob under /kaggle/input, n_tiles, pooled_input, gen_image)
JOBS = [
    ("mt-s42-t1-full",       "mt-s42/model.pt",   1, "default",  "full"),
    ("mt-s123-t1-full",      "mt-s123/model.pt",  1, "default",  "full"),
    ("mt-s2026-t1-full",     "mt-s2026/model.pt", 1, "default",  "full"),
    ("mt-s3407-t1-full",     "mt-s3407/model.pt", 1, "default",  "full"),
    ("mt-s42-t1-full-mean",  "mt-s42/model.pt",   1, "mean_all", "full"),
    ("mt-s42-t3-meanall",    "mt-s42/model.pt",   3, "default",  "first_tile"),
    ("mt-s42-t3-clsmean",    "mt-s42/model.pt",   3, "cls_mean", "first_tile"),
    ("mt-s42-t6-meanall",    "mt-s42/model.pt",   6, "default",  "first_tile"),
    ("mt-s42-t6-clsmean",    "mt-s42/model.pt",   6, "cls_mean", "first_tile"),
    ("l3ep-s42-t1-full",     "l3ep-s42.pt",       1, "default",  "full"),
    ("l3ep-s123-t1-full",    "l3ep-s123.pt",      1, "default",  "full"),
    ("l3ep-s3407-t1-full",   "l3ep-s3407.pt",     1, "default",  "full"),
    ("l3ep-s42-t1-crop",     "l3ep-s42.pt",       1, "default",  "first_tile"),
]


def _cells(tag: str, ckpt: str, n_tiles: int, pooled: str, gen_image: str) -> list[dict]:
    # checkpoint lives in /tmp so the 600MB LoRA .pt never lands in the kernel output
    return [
        _clone_cell(BRANCH),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!python scripts/phase0_build_data.py 2>&1 | tail -6"),
        _code("import glob, os, shutil",
              f"src = glob.glob('/kaggle/input/**/{ckpt}', recursive=True)",
              "assert src, 'ckpt not found: ' + repr(os.listdir('/kaggle/input'))",
              "os.makedirs('/tmp/ck/multi_token', exist_ok=True)",
              "shutil.copy(src[0], '/tmp/ck/multi_token/model.pt')",
              "print('ckpt ready', src[0])"),
        _code(f"!python -m src.cli.evaluate --bridge multi_token --split-dir data/splits --split val "
              f"--n-tiles {n_tiles} --pooled-input {pooled} --gen-image {gen_image} "
              f"--checkpoint /tmp/ck/multi_token/model.pt "
              f"--output /tmp/ck/multi_token/eval_val.json"),
        _code("import os, shutil",
              f"out = '/kaggle/working/out/{tag}'",
              "os.makedirs(out, exist_ok=True)",
              "shutil.copy('/tmp/ck/multi_token/eval_val.json', out)",
              "for f in glob.glob('/tmp/ck/multi_token/results/text_predictions*.json'):",
              "    shutil.copy(f, out)",
              "print(open(out + '/eval_val.json').read()[:1200])"),
    ]


def cmd_launch(only: list[str] | None = None) -> None:
    led = load_ledger()
    for i, (tag, ckpt, n_tiles, pooled, gen_image) in enumerate(JOBS):
        job = f"input-diag:{tag}"
        if only and tag not in only:
            continue
        if led["jobs"].get(job, {}).get("status") == "done":
            print(f"[skip] {job} {led['jobs'][job]['status']}"); continue
        acc = ACCS[i % len(ACCS)]
        slug = f"mvlm-diag-{tag}"
        kid = f"{_user(acc)}/{slug}"
        wd = ROOT / "outputs" / "parallel" / "workers" / slug
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "worker.ipynb").write_text(json.dumps(_nb(_cells(tag, ckpt, n_tiles, pooled, gen_image))))
        (wd / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": slug[:50], "code_file": "worker.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "docker_image": DOCKER_IMAGE,
            "dataset_sources": ["nguynrichard/auto-vqabest",
                                f"{_user('acc1')}/{PLAIN_DS}", f"{_user('acc1')}/{LORA_DS}"],
        }, indent=2))
        _kaggle(acc, "kernels", "push", "-p", str(wd))
        _register(led, job, acc, kid, {"n_tiles": n_tiles, "pooled_input": pooled,
                                       "gen_image": gen_image, "ckpt": ckpt})
        time.sleep(2)


def cmd_collect() -> None:
    led = load_ledger()
    out_root = ROOT / "outputs" / "input_diag"
    done = []
    for job, j in led["jobs"].items():
        if not job.startswith("input-diag:") or j.get("status") == "done":
            continue
        st = _kaggle(j["account"], "kernels", "status", j["kernel"], check=False)
        if "COMPLETE" not in st:
            print(f"[wait] {job}: {st.strip()[-40:]}"); continue
        tag = job.split(":", 1)[1]
        _kaggle(j["account"], "kernels", "output", j["kernel"], "-p", str(out_root / tag), check=False)
        f = next((out_root / tag).rglob("eval_val.json"), None)
        if f:
            d = json.loads(f.read_text())
            done.append(job)
            print(f"[ok] {tag}: F1 {d.get('f1', 0) * 100:.2f}  CIDEr {d.get('cider', 0) * 100:.2f}  "
                  f"loss {d.get('loss', 0):.3f}  n={d.get('n')}")
    # another launcher shares this ledger -- re-read so its entries written while
    # we were downloading are not overwritten
    led = load_ledger()
    for job in done:
        led["jobs"][job]["status"] = "done"
    save_ledger(led)


if __name__ == "__main__":
    if sys.argv[1] == "launch":
        cmd_launch(sys.argv[2:] or None)
    else:
        cmd_collect()
