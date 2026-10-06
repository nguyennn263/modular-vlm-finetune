#!/usr/bin/env python3
"""Retrain Full Q-Former after the answer-leak fix (4a8eb6d), as chains of Kaggle
sessions -- one epoch or one eval per session so each fits the 12h cap. A chain stays
on one account; each phase mounts the previous phase's private output through
kernel_sources. Same procedure as every other bridge in the paper:
  plain : epoch 1 -> epoch 2 (--resume) -> val eval (--gen-image full)
  +LoRA : one joint run from scratch (bridge + decoder LoRA r16, 1 epoch) -> val eval

    python scripts/parallel/train_queue.py fill [--smoke]   # --smoke: --limit 40 everywhere
    python scripts/parallel/train_queue.py collect

--gl: the global_local study instead (plain bridge, no LoRA, AutoViVQA only): Multi-Token on
CLS with g in {8, 14} global tokens + Vintern's mlp1 tokens pooled to k in {1, 9, 36, 144} in
the image slot; epoch 1 -> epoch 2 (--resume) -> val + test eval (--gen-image full).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, ACCT_DIR, _kaggle, _user, _code, _clone_cell, _nb, load_ledger, save_ledger  # noqa
from input_diag import BRANCH, DOCKER_IMAGE
from ood_full import _free_slots

SMOKE = "--smoke" in sys.argv
GL = "--gl" in sys.argv
SET = "gl" if GL else "qfx"
PREFIX = f"{SET}-smoke" if SMOKE else SET
GL_GLOBAL = [8, 14]
GL_GRIDS = [1, 3, 6, 12]                    # k = grid^2 = 1 / 9 / 36 / 144 local tokens
# accounts with room for two ~8h chains each (accounts under ~2h left failed to start a session)
GL_ACCS = ["acc13", "acc17", "acc15", "acc16"]
# relaunches get a new chain name, so every attempt has its own kernels (no version overwrite)
GL_RETRY = {(8, 12): "r2", (14, 12): "r2"}   # k=144 ep1 OOM'd at the fixed 400-token padding
SEEDS = [42] if SMOKE else [42, 123, 3407]
TRAIN = ("--split-dir data/splits --batch-size 8 --grad-accum 1 --eval-steps 800 --save-steps 800 "
         "--no-early-stopping --text-metrics-every 99")
LIMIT = " --limit 40" if SMOKE else ""
CK = "/kaggle/working/ck"


def _chains() -> dict[str, list[dict]]:
    chains = {}
    if GL:
        for g in GL_GLOBAL:
            for grid in GL_GRIDS:
                flags = f"--bridge-num-tokens {g} --local-grid {grid}"
                base = (f"python -m src.cli.train --bridge global_local {flags} --seed 42 {TRAIN}"
                        f" --text-metrics-max-samples 200{LIMIT} --output-dir {CK}")
                chains[f"{PREFIX}-g{g}-k{grid * grid}{GL_RETRY.get((g, grid), '')}-s42"] = [
                    {"name": "ep1", "bridge": "global_local", "cmd": f"{base} --epochs 1"},
                    {"name": "ep2", "bridge": "global_local", "cmd": f"{base} --epochs 2 --resume PREV"},
                    {"name": "eval", "bridge": "global_local", "eval": True, "splits": ["val", "test"],
                     "eval_flags": f" {flags}"},
                ]
        return chains
    for s in SEEDS:
        chains[f"{PREFIX}-plain-s{s}"] = [
            {"name": "ep1", "cmd": f"python -m src.cli.train --bridge qformer --seed {s} --epochs 1 {TRAIN}{LIMIT} --output-dir {CK}"},
            {"name": "ep2", "cmd": f"python -m src.cli.train --bridge qformer --seed {s} --epochs 2 {TRAIN}{LIMIT} --output-dir {CK} --resume PREV"},
            {"name": "eval", "eval": True},
        ]
        chains[f"{PREFIX}-lora-s{s}"] = [
            {"name": "ep1", "cmd": f"python -m src.cli.train --bridge qformer --seed {s} --epochs 1 --lora --lora-r 16 {TRAIN}{LIMIT} --output-dir {CK}"},
            {"name": "eval", "eval": True},
        ]
    return chains


def _checked(cmd: str) -> dict:
    """A `!cmd` cell that fails the kernel when cmd exits non-zero -- a bare `!` line does not,
    so a crashed run would otherwise show up as COMPLETE."""
    return _code(f"!{cmd} || echo failed > /tmp/cell_failed", "import os",
                 "assert not os.path.exists('/tmp/cell_failed'), 'command failed, see the output above'")


def _cells(phase: dict, has_prev: bool) -> list[dict]:
    b = phase.get("bridge", "qformer")
    cells = [_clone_cell(BRANCH), _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
             _code("!python -c \"import torch; print('GPU:', torch.cuda.get_device_name(0))\""),
             _code("!python scripts/phase0_build_data.py 2>&1 | tail -4")]
    if has_prev:  # previous phase's checkpoint, mounted read-only through kernel_sources
        cells.append(_code("import glob, os, shutil",
                           f"prev = sorted(glob.glob('/kaggle/input/**/{b}/last_model.pt', recursive=True))",
                           "assert prev, 'previous phase checkpoint not found: ' + repr(os.listdir('/kaggle/input'))",
                           "os.makedirs('/tmp/prev', exist_ok=True); shutil.copy(prev[0], '/tmp/prev/last_model.pt')",
                           "print('prev ckpt', prev[0])"))
    if phase.get("eval"):
        limit = " --limit 40" if SMOKE else ""
        cells.append(_code(f"!mkdir -p /tmp/ck/{b} && cp /tmp/prev/last_model.pt /tmp/ck/{b}/model.pt"))
        for split in phase.get("splits", ["val"]):
            out = "/kaggle/working/out" + ("" if "splits" not in phase else f"/{split}")
            cells += [_checked(f"python -m src.cli.evaluate --bridge {b} --split-dir data/splits --split {split} "
                               f"--n-tiles 1 --gen-image full --checkpoint /tmp/ck/{b}/model.pt{limit}"
                               f"{phase.get('eval_flags', '')} --output /tmp/ck/{b}/eval_{split}.json"),
                      _code(f"!mkdir -p {out} && cp /tmp/ck/{b}/eval_{split}.json "
                            f"/tmp/ck/{b}/results/text_predictions_epoch_1.json {out}/ && "
                            f"head -c 700 {out}/eval_{split}.json")]
    else:
        cells += [_checked(phase["cmd"].replace("PREV", "/tmp/prev/last_model.pt")),
                  _code(f"!ls -la {CK}/{b}/ && grep -h 'Train Loss\\|Val Loss' {CK}/{b}/results/training_*.log | tail -4")]
    return cells


def _launch(acc: str, chain: str, idx: int, phases: list[dict], prev_kernel: str | None) -> None:
    phase = phases[idx]
    slug = f"mvlm-{chain}-{phase['name']}"
    kid = f"{_user(acc)}/{slug}"
    wd = ROOT / "outputs" / "parallel" / "workers" / slug
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "worker.ipynb").write_text(json.dumps(_nb(_cells(phase, prev_kernel is not None))))
    (wd / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": slug[:50], "code_file": "worker.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "docker_image": DOCKER_IMAGE,
        "dataset_sources": ["nguynrichard/auto-vqabest"],
        "kernel_sources": [prev_kernel] if prev_kernel else [],
    }, indent=2))
    _kaggle(acc, "kernels", "push", "-p", str(wd))
    led = load_ledger()
    led["jobs"][f"{chain}:{phase['name']}"] = {"account": acc, "kernel": kid, "status": "running",
                                               "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                               "chain": chain, "phase": idx}
    save_ledger(led)
    print(f"[launch] {chain}:{phase['name']} -> {acc} ({kid})", flush=True)


def _status(acc: str, kid: str) -> str:
    st = _kaggle(acc, "kernels", "status", kid, check=False)
    return st.split("KernelWorkerStatus.")[-1].strip(' "\n') if "KernelWorkerStatus." in st else "UNKNOWN"


def cmd_fill() -> None:
    chains = _chains()
    accounts = GL_ACCS if GL else sorted(
        (p.name for p in ACCT_DIR.glob("acc*") if (p / "kaggle.json").exists() and p.name[3:].isdigit()),
        key=lambda s: int(s[3:]), reverse=True)
    while True:
        jobs = load_ledger()["jobs"]
        pending = False
        for chain, phases in chains.items():
            done_idx, acc, prev = -1, None, None
            for i, ph in enumerate(phases):
                j = jobs.get(f"{chain}:{ph['name']}")
                if not j:
                    break
                acc, prev = j["account"], j["kernel"]
                st = j["status"] if j["status"] == "done" else _status(acc, prev)
                if st in ("COMPLETE", "done"):
                    done_idx = i
                    continue
                if st in ("ERROR", "CANCEL_ACKNOWLEDGED"):
                    print(f"[FAILED] {chain}:{ph['name']} on {acc} -- needs a look", flush=True)
                break
            nxt = done_idx + 1
            if nxt >= len(phases):
                continue
            pending = True
            if f"{chain}:{phases[nxt]['name']}" in jobs:
                continue                      # that phase is running (or failed, reported above)
            if acc is None:                   # chain start: any account with a free slot
                acc = next((a for a in accounts if _free_slots(a) > 0), None)
            elif _free_slots(acc) == 0:
                continue                      # chain must stay on its account
            if acc:
                try:
                    _launch(acc, chain, nxt, phases, prev if nxt > 0 else None)
                except Exception as exc:
                    print(f"[error] {chain}:{phases[nxt]['name']} on {acc}: {exc}", flush=True)
        if not pending:
            print("ALL CHAINS DONE", flush=True)
            return
        time.sleep(300)


def cmd_collect() -> None:
    out_root = ROOT / "outputs" / f"train_{SET}"
    led = load_ledger()
    done = []
    for key, j in led["jobs"].items():
        if not key.startswith(PREFIX + "-") or j.get("status") != "running":
            continue
        if _status(j["account"], j["kernel"]) != "COMPLETE":
            continue
        dst = out_root / key.replace(":", "_")
        _kaggle(j["account"], "kernels", "output", j["kernel"], "--file-pattern", r".*\.(json|log)$",
                "-p", str(dst), check=False)
        done.append(key)
        evs = sorted(dst.rglob("eval_*.json"))
        for ev in evs:
            d = json.loads(ev.read_text())
            print(f"[ok] {key} {ev.stem}: F1 {d.get('f1', 0) * 100:.2f}  CIDEr {d.get('cider', 0) * 100:.2f}  "
                  f"loss {d.get('loss', 0):.3f}")
        if not evs:
            print(f"[ok] {key}")
    led = load_ledger()
    for key in done:
        led["jobs"][key]["status"] = "done"
    save_ledger(led)


if __name__ == "__main__":
    {"fill": cmd_fill, "collect": cmd_collect}[sys.argv[1]]()
