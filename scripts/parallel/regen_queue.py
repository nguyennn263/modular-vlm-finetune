#!/usr/bin/env python3
"""Re-score existing checkpoints with --gen-image full for the paper tables, one job
per idle account. Only uses checkpoints already on Kaggle. Ledger keys regen-full:<tag>
(the same keys the jobs handed over from the other session use).

    python scripts/parallel/regen_queue.py fill
    python scripts/parallel/regen_queue.py collect
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, _kaggle, _user, _code, _clone_cell, _nb, load_ledger, save_ledger  # noqa
from input_diag import BRANCH, DOCKER_IMAGE
from ood_full import _idle

OWNER = "nguyennn263"
# accounts handed back by the other session -- ood_full.py fills the rest
ACCS = ["acc1", "acc3", "acc4", "acc5", "acc12"]

# (tag, bridge, dataset, file inside it, splits)
JOBS = [
    ("mt-s123",    "multi_token",    "mvlm-test-ckpt",    "mt-s123/model.pt",  ["test"]),
    ("mt-s2026",   "multi_token",    "mvlm-test-ckpt",    "mt-s2026/model.pt", ["test"]),
    ("mt-s3407",   "multi_token",    "mvlm-test-ckpt",    "mt-s3407/model.pt", ["test"]),
    ("l3ep-s123",  "multi_token",    "mvlm-lora-mt-ckpt", "l3ep-s123.pt",      ["test"]),
    ("l3ep-s3407", "multi_token",    "mvlm-lora-mt-ckpt", "l3ep-s3407.pt",     ["test"]),
    ("mq-s42",     "mini_qformer",   "mvlm-test-ckpt",    "mq-s42/model.pt",   ["val"]),
    ("res-s42",    "residual",       "mvlm-test-ckpt",    "res-s42/model.pt",  ["val"]),
    ("ta-s42",     "tile_attention", "mvlm-test-ckpt",    "ta-s42/model.pt",   ["val"]),
    ("qf-s3407",   "qformer",        "mvlm-qf-test-ckpt", "qf-s3407.pt",       ["val"]),
]


def _cells(tag: str, bridge: str, ckpt: str, splits: list[str]) -> list[dict]:
    cells = [
        _clone_cell(BRANCH),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!python scripts/phase0_build_data.py 2>&1 | tail -6"),
        _code("import glob, os, shutil",
              f"src = glob.glob('/kaggle/input/**/{ckpt}', recursive=True)",
              "assert src, 'ckpt not found: ' + repr(os.listdir('/kaggle/input'))",
              f"os.makedirs('/tmp/ck/{bridge}', exist_ok=True)",
              f"shutil.copy(src[0], '/tmp/ck/{bridge}/model.pt')",
              "print('ckpt ready', src[0])"),
    ]
    for split in splits:
        out = f"/kaggle/working/out/{tag}/{split}"
        cells += [
            _code(f"!python -m src.cli.evaluate --bridge {bridge} --split-dir data/splits --split {split} "
                  f"--n-tiles 1 --gen-image full --checkpoint /tmp/ck/{bridge}/model.pt "
                  f"--output /tmp/ck/{bridge}/eval_{split}.json"),
            _code(f"os.makedirs('{out}', exist_ok=True)",
                  f"shutil.copy('/tmp/ck/{bridge}/eval_{split}.json', '{out}')",
                  f"for f in glob.glob('/tmp/ck/{bridge}/results/text_predictions*.json'):",
                  f"    shutil.copy(f, '{out}')",
                  f"print(open('{out}/eval_{split}.json').read()[:600])"),
        ]
    return cells


def _launch(acc: str, tag: str, bridge: str, ds: str, ckpt: str, splits: list[str]) -> None:
    job = f"regen-full:{tag}"
    slug = f"mvlm-regen-{tag}"
    kid = f"{_user(acc)}/{slug}"
    wd = ROOT / "outputs" / "parallel" / "workers" / slug
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "worker.ipynb").write_text(json.dumps(_nb(_cells(tag, bridge, ckpt, splits))))
    (wd / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": slug[:50], "code_file": "worker.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "docker_image": DOCKER_IMAGE,
        "dataset_sources": ["nguynrichard/auto-vqabest", f"{OWNER}/{ds}"],
    }, indent=2))
    _kaggle(acc, "kernels", "push", "-p", str(wd))
    led = load_ledger()  # re-read: shared with ood_full.py
    led["jobs"][job] = {"account": acc, "kernel": kid, "status": "running",
                        "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "bridge": bridge, "splits": splits, "gen_image": "full"}
    save_ledger(led)
    print(f"[launch] {job} -> {acc} ({kid}) {splits}", flush=True)


def cmd_fill() -> None:
    queue = [j for j in JOBS if f"regen-full:{j[0]}" not in load_ledger()["jobs"]]
    while queue:
        busy = {j["account"] for k, j in load_ledger()["jobs"].items()
                if k.startswith("regen-full:") and j["status"] == "running"}
        for acc in ACCS:
            if queue and acc not in busy and _idle(acc):
                _launch(acc, *queue.pop(0))
                time.sleep(5)
        if queue:
            print(time.strftime("%H:%M:%S"), f"{len(queue)} queued", flush=True)
            time.sleep(300)
    print("ALL LAUNCHED", flush=True)


def cmd_collect() -> None:
    out_root = ROOT / "outputs" / "regen_full"
    done = []
    for job, j in load_ledger()["jobs"].items():
        if not job.startswith("regen-full:") or j.get("status") == "done":
            continue
        st = _kaggle(j["account"], "kernels", "status", j["kernel"], check=False)
        if "COMPLETE" not in st:
            print(f"[wait] {job}: {st.strip()[-30:]}"); continue
        tag = job.split(":", 1)[1]
        _kaggle(j["account"], "kernels", "output", j["kernel"], "-p", str(out_root / tag), check=False)
        evals = sorted((out_root / tag).rglob("eval_*.json"))
        if evals:
            done.append(job)
            for f in evals:
                d = json.loads(f.read_text())
                print(f"[ok] {tag} {f.stem}: F1 {d.get('f1', 0) * 100:.2f}  CIDEr {d.get('cider', 0) * 100:.2f}")
    led = load_ledger()
    for job in done:
        led["jobs"][job]["status"] = "done"
    save_ledger(led)


if __name__ == "__main__":
    {"fill": cmd_fill, "collect": cmd_collect}[sys.argv[1]]()
