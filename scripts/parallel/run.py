#!/usr/bin/env python3
"""Drive Exp A / oracle sweep across the 5 Kaggle accounts.

    python scripts/parallel/run.py launch expa            # 5 bridges, seed 42, one per account
    python scripts/parallel/run.py poll                   # collect finished, save to repo
    python scripts/parallel/run.py status
    python scripts/parallel/run.py resume expa:qformer:s42 # relaunch with the partial checkpoint

State: outputs/parallel/ledger.json  (idempotent — safe to re-run after a crash).
Account configs: ~/.kaggle-accounts/acc{1..5}/kaggle.json
Kernel outputs land in checkpoints/expA/... and outputs/expA/... in this repo.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "outputs" / "parallel" / "ledger.json"
ACCT_DIR = Path.home() / ".kaggle-accounts"
ACCOUNTS = [f"acc{i}" for i in range(1, 6)]
REPO_URL = "https://github.com/nguyennn263/modular-vlm-finetune.git"
BRIDGES = ["residual", "multi_token", "tile_attention", "mini_qformer", "qformer"]


# ------------------------------------------------------------------ kaggle CLI
def _kaggle(acc: str, *args: str, check: bool = True, capture: bool = True) -> str:
    env = {**os.environ, "KAGGLE_CONFIG_DIR": str(ACCT_DIR / acc)}
    r = subprocess.run(["kaggle", *args], env=env, text=True,
                       capture_output=capture, check=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} [{acc}] -> {r.returncode}\n{r.stdout}\n{r.stderr}")
    return (r.stdout or "") + (r.stderr or "")


def _user(acc: str) -> str:
    return json.loads((ACCT_DIR / acc / "kaggle.json").read_text())["username"]


# ------------------------------------------------------------------ ledger
def load_ledger() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {"jobs": {}}


def save_ledger(led: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=2))


# ------------------------------------------------------------------ worker notebook
def _nb(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3",
            "display_name": "Python 3", "language": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def _code(*lines: str) -> dict:
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def expa_worker(bridge: str, seed: int, branch: str, resume_ds: str | None, epochs: int) -> list[dict]:
    ck = f"/kaggle/working/ck/seed{seed}"
    resume_cp = (f"!mkdir -p {ck}/{bridge} && cp /kaggle/input/{resume_ds.split('/')[-1]}/* "
                 f"{ck}/{bridge}/ 2>/dev/null && echo RESUMED || echo FRESH") if resume_ds else "print('FRESH')"
    return [
        _code("import os",
              "os.chdir('/kaggle/working')",
              f"os.system('git clone -q {REPO_URL} repo || (cd repo && git fetch -q)')",
              "os.chdir('/kaggle/working/repo')",
              f"os.system('git checkout -q {branch} && git pull -q')"),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!python scripts/phase0_build_data.py 2>&1 | tail -6"),
        _code(resume_cp),
        _code(f"!python -m src.cli.train --bridge {bridge} --split-dir data/splits --seed {seed} "
              f"--epochs {epochs} --batch-size 8 --grad-accum 1 --eval-steps 800 --save-steps 800 "
              f"--output-dir {ck} --resume"),
        _code(f"!python -m src.cli.evaluate --bridge {bridge} --split-dir data/splits --split val "
              f"--checkpoint {ck}/{bridge}/best_model.pt"),
        _code(f"!mkdir -p /kaggle/working/out && cp -r {ck} /kaggle/working/out/ && "
              f"cp -r data/splits /kaggle/working/out/ 2>/dev/null; ls -R /kaggle/working/out | tail -20"),
    ]


def _push_worker(acc: str, slug: str, cells: list[dict], resume_ds: str | None) -> str:
    user = _user(acc)
    kid = f"{user}/{slug}"
    d = ROOT / "outputs" / "parallel" / "workers" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "worker.ipynb").write_text(json.dumps(_nb(cells)))
    meta = {
        "id": kid, "title": slug[:50], "code_file": "worker.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ["nguynrichard/auto-vqabest"] + ([resume_ds] if resume_ds else []),
        "competition_sources": [], "kernel_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    _kaggle(acc, "kernels", "push", "-p", str(d))
    return kid


# ------------------------------------------------------------------ commands
def cmd_launch(args) -> None:
    phase = args.phase
    led = load_ledger()
    if phase == "expa":
        branch = _current_branch()
        for i, bridge in enumerate(BRIDGES):
            acc = ACCOUNTS[i % len(ACCOUNTS)]
            job = f"expa:{bridge}:s{args.seed}"
            if job in led["jobs"] and led["jobs"][job].get("status") not in ("failed", None):
                print(f"[skip] {job} already {led['jobs'][job]['status']}")
                continue
            slug = f"mvlm-expa-{bridge.replace('_','-')}-s{args.seed}"
            cells = expa_worker(bridge, args.seed, branch, None, args.epochs)
            kid = _push_worker(acc, slug, cells, None)
            led["jobs"][job] = {"account": acc, "kernel": kid, "bridge": bridge,
                                "seed": args.seed, "status": "running",
                                "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "collected": False}
            print(f"[launch] {job} -> {acc} ({kid})")
            save_ledger(led)
    else:
        raise SystemExit(f"unknown phase {phase!r}")


def cmd_poll(args) -> None:
    led = load_ledger()
    for job, j in led["jobs"].items():
        if j.get("collected") or j.get("status") in ("failed",):
            continue
        out = _kaggle(j["account"], "kernels", "status", j["kernel"], check=False)
        st = next((s for s in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED")
                   if s in out.upper()), "?")
        j["status"] = {"COMPLETE": "complete", "ERROR": "error", "CANCEL": "cancelled",
                       "RUNNING": "running", "QUEUED": "queued"}.get(st, "unknown")
        print(f"[{job}] {j['status']}")
        if j["status"] in ("complete", "error"):
            _collect(job, j)
        save_ledger(led)
    save_ledger(led)


def _collect(job: str, j: dict) -> None:
    dst = ROOT / "outputs" / "parallel" / "pulled" / job.replace(":", "_")
    dst.mkdir(parents=True, exist_ok=True)
    _kaggle(j["account"], "kernels", "output", j["kernel"], "-p", str(dst), check=False)

    # checkpoints + eval samples -> repo layout
    for pt in dst.rglob("best_model.pt"):
        tgt = ROOT / "checkpoints" / "expA" / f"seed{j['seed']}" / j["bridge"]
        tgt.mkdir(parents=True, exist_ok=True)
        for f in pt.parent.iterdir():
            if f.is_file():
                (tgt / f.name).write_bytes(f.read_bytes())
    for sm in dst.rglob("eval_val_samples.jsonl"):
        tgt = ROOT / "outputs" / "expA" / j["bridge"]
        tgt.mkdir(parents=True, exist_ok=True)
        (tgt / "eval_val_samples.jsonl").write_bytes(sm.read_bytes())
    for sj in dst.rglob("summary.json"):
        try:
            s = json.loads(sj.read_text())
            j["epochs_trained"] = s.get("epochs_trained")
            j["best_val_loss"] = s.get("best_val_loss")
        except Exception:
            pass

    got_ckpt = (ROOT / "checkpoints" / "expA" / f"seed{j['seed']}" / j["bridge"] / "best_model.pt").exists()
    j["collected"] = bool(got_ckpt)
    j["status"] = "done" if got_ckpt else ("error" if j["status"] == "error" else "incomplete")
    print(f"   -> {j['status']}  (checkpoint: {'yes' if got_ckpt else 'NO'})")


def cmd_status(args) -> None:
    led = load_ledger()
    if not led["jobs"]:
        print("(no jobs)")
        return
    print(f"{'job':32} {'acc':5} {'status':11} {'ep':>3} {'val_loss':>9}  kernel")
    for job, j in led["jobs"].items():
        print(f"{job:32} {j['account']:5} {j['status']:11} "
              f"{str(j.get('epochs_trained','-')):>3} {str(j.get('best_val_loss','-'))[:9]:>9}  {j['kernel']}")


def cmd_resume(args) -> None:
    led = load_ledger()
    job = args.job
    j = led["jobs"][job]
    partial = ROOT / "checkpoints" / "expA" / f"seed{j['seed']}" / j["bridge"]
    if not any(partial.glob("*.pt")):
        raise SystemExit(f"no partial checkpoint at {partial} — nothing to resume from")
    # publish partial as a public dataset under the same account
    ds_dir = ROOT / "outputs" / "parallel" / "resume" / job.replace(":", "_")
    ds_dir.mkdir(parents=True, exist_ok=True)
    for f in partial.glob("*.pt"):
        (ds_dir / f.name).write_bytes(f.read_bytes())
    user = _user(j["account"])
    ds_id = f"{user}/mvlm-resume-{job.split(':',1)[1].replace(':','-').replace('_','-')}"
    (ds_dir / "dataset-metadata.json").write_text(json.dumps(
        {"id": ds_id, "title": ds_id.split('/')[-1], "licenses": [{"name": "unknown"}]}))
    try:
        _kaggle(j["account"], "datasets", "create", "-p", str(ds_dir), "--public")
    except RuntimeError:
        _kaggle(j["account"], "datasets", "version", "-p", str(ds_dir), "-m", "resume", "--dir-mode", "zip")
    time.sleep(20)
    slug = f"mvlm-expa-{j['bridge'].replace('_','-')}-s{j['seed']}"
    cells = expa_worker(j["bridge"], j["seed"], _current_branch(), ds_id, args.epochs)
    kid = _push_worker(j["account"], slug, cells, ds_id)
    j.update(status="running", collected=False, resumed_from=ds_id,
             pushed_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    save_ledger(led)
    print(f"[resume] {job} -> {kid} (resume dataset {ds_id})")


def _current_branch() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "branch", "--show-current"],
                          text=True, capture_output=True).stdout.strip() or "chore/repo-restructure"


def main() -> None:
    p = argparse.ArgumentParser(prog="run.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("launch"); lp.add_argument("phase"); lp.add_argument("--seed", type=int, default=42)
    lp.add_argument("--epochs", type=int, default=10); lp.set_defaults(fn=cmd_launch)
    sub.add_parser("poll").set_defaults(fn=cmd_poll)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    rp = sub.add_parser("resume"); rp.add_argument("job"); rp.add_argument("--epochs", type=int, default=10)
    rp.set_defaults(fn=cmd_resume)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
