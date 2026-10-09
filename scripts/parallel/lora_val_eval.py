#!/usr/bin/env python3
"""Re-eval the 6 re-run LoRA multi_token checkpoints on the VAL split ONLY, to
recover full-val text predictions (the expa_worker's test-eval step overwrites
results/text_predictions_epoch_1.json with the test set, so the re-run kernels
kept clean val *metrics* but not val *predictions* -> no val corpus rescore).

    python scripts/parallel/lora_val_eval.py bundle    # flat dataset of the 6 .pt
    python scripts/parallel/lora_val_eval.py launch     # 6 val-eval kernels
    python scripts/parallel/lora_val_eval.py collect     # pull text_predictions_epoch_1.json (val)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, ACCT_DIR, _kaggle, _user, _code, _clone_cell, _nb, load_ledger, save_ledger, _register  # noqa

DS = "mvlm-lora-mt-ckpt"
BRANCH = "feat/decoder-lora"
# (repo ckpt path, label, seed, epochs)
SPECS = [
    ("checkpoints/expA-lora16-3ep/seed42/multi_token/last_model.pt",   "l3ep-s42",   42,   3),
    ("checkpoints/expA-lora16-3ep/seed123/multi_token/last_model.pt",  "l3ep-s123",  123,  3),
    ("checkpoints/expA-lora16-3ep/seed3407/multi_token/last_model.pt", "l3ep-s3407", 3407, 3),
    ("checkpoints/expA-lora16/seed42/multi_token/last_model.pt",       "l1ep-s42",   42,   1),
    ("checkpoints/expA-lora16/seed123/multi_token/last_model.pt",      "l1ep-s123",  123,  1),
    ("checkpoints/expA-lora16/seed3407/multi_token/last_model.pt",     "l1ep-s3407", 3407, 1),
]
ACCS = ["acc1", "acc5", "acc6", "acc10", "acc2", "acc16"]


def cmd_bundle():
    d = ROOT / "outputs" / "parallel" / "lora_val_bundle"
    if d.exists():
        import shutil; shutil.rmtree(d)
    d.mkdir(parents=True)
    for pt, label, *_ in SPECS:
        src = ROOT / pt
        assert src.exists() and src.stat().st_size > 0, f"missing {pt}"
        (d / f"{label}.pt").write_bytes(src.read_bytes())
        print(f"[bundle] {label}.pt {src.stat().st_size//1024//1024}MB")
    user = _user("acc1")
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"id": f"{user}/{DS}", "title": DS, "licenses": [{"name": "unknown"}]}))
    try:
        _kaggle("acc1", "datasets", "create", "-p", str(d), "--public")
    except RuntimeError:
        _kaggle("acc1", "datasets", "version", "-p", str(d), "-m", "lora mt ckpts")
    print(f"[bundle] -> {user}/{DS}")


def _cells(label: str, seed: int) -> list[dict]:
    ds = f"{_user('acc1')}/{DS}"
    return [
        _clone_cell(BRANCH),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!python scripts/phase0_build_data.py 2>&1 | tail -6"),
        _code("import glob, os, shutil",
              f"src = glob.glob('/kaggle/input/**/{label}.pt', recursive=True)",
              "assert src, 'ckpt: ' + repr(os.listdir('/kaggle/input'))",
              "os.makedirs('/kaggle/working/ck/multi_token', exist_ok=True)",
              "shutil.copy(src[0], '/kaggle/working/ck/multi_token/model.pt')"),
        # feat/decoder-lora evaluate.py auto-detects `lora_state` in the ckpt and
        # rebuilds the adapter -- no --lora flag needed.
        _code(f"!python -m src.cli.evaluate --bridge multi_token --split-dir data/splits "
              f"--split val --seed {seed} "
              f"--checkpoint /kaggle/working/ck/multi_token/model.pt "
              f"--output /kaggle/working/eval_val.json"),
        _code("import os, shutil, glob",
              "os.makedirs('/kaggle/working/out', exist_ok=True)",
              "shutil.copy('/kaggle/working/eval_val.json', '/kaggle/working/out/')",
              "[shutil.copy(p, '/kaggle/working/out/') for p in "
              "glob.glob('/kaggle/working/ck/multi_token/results/text_predictions_epoch_1.json')]",
              "[shutil.copy(p, '/kaggle/working/out/') for p in "
              "glob.glob('/kaggle/working/ck/multi_token/eval_val_samples.jsonl')]",
              "print('out:', os.listdir('/kaggle/working/out'))"),
    ]


def cmd_launch():
    led = load_ledger()
    for i, (pt, label, seed, ep) in enumerate(SPECS):
        job = f"lora-val-eval:{label}"
        if led["jobs"].get(job, {}).get("status") == "done":
            print(f"[skip] {job}"); continue
        acc = ACCS[i % len(ACCS)]
        slug = f"mvlm-lora-val-{label}"
        user = _user(acc)
        kid = f"{user}/{slug}"
        wd = ROOT / "outputs" / "parallel" / "workers" / slug
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "worker.ipynb").write_text(json.dumps(_nb(_cells(label, seed))))
        (wd / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": slug[:50], "code_file": "worker.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": True,
            "dataset_sources": ["nguynrichard/auto-vqabest", f"{_user('acc1')}/{DS}"],
        }, indent=2))
        _kaggle(acc, "kernels", "push", "-p", str(wd))
        _register(led, job, acc, kid, {"label": label, "seed": seed, "epochs": ep})
        time.sleep(2)


def cmd_collect():
    led = load_ledger()
    out = ROOT / "outputs" / "lora_val_eval"
    out.mkdir(parents=True, exist_ok=True)
    for job, j in led["jobs"].items():
        if not job.startswith("lora-val-eval:") or j.get("status") == "done":
            continue
        st = _kaggle(j["account"], "kernels", "status", j["kernel"], check=False)
        if "COMPLETE" not in st:
            print(f"[wait] {job}: {st.strip()[:50]}"); continue
        dst = out / job.split(":")[1]
        _kaggle(j["account"], "kernels", "output", j["kernel"], "-p", str(dst), check=False)
        j["status"] = "done"
        print(f"[ok] {job} -> {dst}")
    save_ledger(led)


if __name__ == "__main__":
    {"bundle": cmd_bundle, "launch": cmd_launch, "collect": cmd_collect}[sys.argv[1]]()
