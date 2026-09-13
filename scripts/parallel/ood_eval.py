#!/usr/bin/env python3
"""OOD eval: Vintern-1B-v3_5 zero-shot vs our best checkpoint (Multi-Token
bridge + decoder LoRA r16 3ep, seed42) on 2 external Vietnamese VQA datasets
that neither model has ever trained on (ViTextVQA, ViVQA-X) -- 1000 sampled
test-split rows each, seed=42, single seed, traced via manifest.json.

    python scripts/parallel/ood_eval.py launch                 # push 2 kernels
    python scripts/parallel/ood_eval.py collect                 # pull + score

Reuses the existing `mvlm-lora-mt-ckpt` Kaggle dataset (uploaded by
scripts/parallel/lora_val_eval.py) for the checkpoint -- no re-bundle needed.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, ACCT_DIR, _kaggle, _user, _code, _clone_cell, _nb, load_ledger, save_ledger, _register  # noqa

BRANCH = "feat/decoder-lora"
CKPT_DS = "mvlm-lora-mt-ckpt"
CKPT_LABEL = "l3ep-s42"           # checkpoints/expA-lora16-3ep/seed42/multi_token/last_model.pt
DATASETS = ["vitextvqa", "vivqax"]
N_SAMPLES = 1000
SEED = 42
ACCS = ["acc7", "acc8", "acc9"]    # pick accounts NOT busy with the vintern-ft-minimal repro (acc2, acc15)


def _cells(dataset: str) -> list[dict]:
    ck_ds = f"{_user('acc1')}/{CKPT_DS}"
    return [
        _clone_cell(BRANCH),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!pip -q install requests"),
        _code("import glob, os, shutil",
              f"src = glob.glob('/kaggle/input/**/{CKPT_LABEL}.pt', recursive=True)",
              "assert src, 'ckpt not found: ' + repr(os.listdir('/kaggle/input'))",
              "os.makedirs('/kaggle/working/ck/multi_token', exist_ok=True)",
              "shutil.copy(src[0], '/kaggle/working/ck/multi_token/model.pt')",
              "print('ckpt ready')"),
        _code(f"!python experiments/ood-eval/build_ood_data.py --dataset {dataset} --n {N_SAMPLES} "
              f"--seed {SEED} --out /kaggle/working/data/{dataset} 2>&1 | tail -40"),
        _code(f"!python experiments/ood-eval/gen_vintern_base.py "
              f"--data /kaggle/working/data/{dataset}/internvl.jsonl "
              f"--images-dir /kaggle/working/data/{dataset}/images "
              f"--out /kaggle/working/out/{dataset}/vintern_base 2>&1 | tail -60"),
        _code(f"!python experiments/ood-eval/eval_ours_ood.py "
              f"--data /kaggle/working/data/{dataset}/ours.jsonl "
              f"--images-dir /kaggle/working/data/{dataset}/images "
              f"--checkpoint /kaggle/working/ck/multi_token/model.pt "
              f"--output-dir /kaggle/working/out/{dataset}/ours 2>&1 | tail -60"),
        _code("import os, shutil",
              f"os.makedirs('/kaggle/working/out/{dataset}', exist_ok=True)",
              f"shutil.copy('/kaggle/working/data/{dataset}/manifest.json', '/kaggle/working/out/{dataset}/manifest.json')",
              f"print('out:', os.listdir('/kaggle/working/out/{dataset}'))"),
    ]


def cmd_launch() -> None:
    led = load_ledger()
    for i, dataset in enumerate(DATASETS):
        job = f"ood-eval:{dataset}"
        if led["jobs"].get(job, {}).get("status") == "done":
            print(f"[skip] {job} done"); continue
        acc = ACCS[i % len(ACCS)]
        slug = f"mvlm-ood-eval-{dataset}"
        user = _user(acc)
        kid = f"{user}/{slug}"
        wd = ROOT / "outputs" / "parallel" / "workers" / slug
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "worker.ipynb").write_text(json.dumps(_nb(_cells(dataset))))
        (wd / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": slug[:50], "code_file": "worker.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [f"{_user('acc1')}/{CKPT_DS}"],
        }, indent=2))
        _kaggle(acc, "kernels", "push", "-p", str(wd))
        _register(led, job, acc, kid, {"dataset": dataset, "n": N_SAMPLES, "seed": SEED})
        time.sleep(2)


def cmd_collect() -> None:
    led = load_ledger()
    out_root = ROOT / "outputs" / "ood_eval"
    out_root.mkdir(parents=True, exist_ok=True)
    for job, j in led["jobs"].items():
        if not job.startswith("ood-eval:") or j.get("status") == "done":
            continue
        st = _kaggle(j["account"], "kernels", "status", j["kernel"], check=False)
        if "COMPLETE" not in st:
            print(f"[wait] {job}: {st.strip()[:60]}"); continue
        dst = out_root / j["dataset"]
        _kaggle(j["account"], "kernels", "output", j["kernel"], "-p", str(dst), check=False)
        vb = next(dst.rglob("vintern_base/results/text_predictions_epoch_1.json"), None)
        ours = next(dst.rglob("ours/results/text_predictions_epoch_1.json"), None)
        j["collected"] = bool(vb and ours)
        j["status"] = "done" if j["collected"] else "incomplete"
        print(f"[{'ok' if j['collected'] else 'partial'}] {job} -> {dst} "
              f"(vintern_base={'y' if vb else 'N'} ours={'y' if ours else 'N'})")
    save_ledger(led)


if __name__ == "__main__":
    {"launch": cmd_launch, "collect": cmd_collect}[sys.argv[1]]()
