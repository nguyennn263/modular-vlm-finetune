#!/usr/bin/env python3
"""OOD eval: Vintern-1B-v3_5 zero-shot vs our best checkpoint (Multi-Token
bridge + decoder LoRA r16 3ep, seed42) on external Vietnamese VQA datasets
that neither model has ever trained on (ViTextVQA, ViVQA-X, OpenViVQA, ...) --
1000 sampled rows each (test split, except OpenViVQA which uses its DEV split
since its test-split answers are a placeholder). Multi-seed (42/123/3407): the
seed controls which 1000 rows get SAMPLED from the split (neither model is
retrained -- Vintern gốc is zero-shot, ours is a fixed checkpoint -- this is
purely about sampling variance on a 1000-row subset). Results recorded in
experiments/ood-eval/RESULTS.md.

    python scripts/parallel/ood_eval.py launch                 # push kernels (skips already-done jobs)
    python scripts/parallel/ood_eval.py collect                 # pull + score

Reuses the existing `mvlm-lora-mt-ckpt` Kaggle dataset (uploaded by
scripts/parallel/lora_val_eval.py) for the checkpoint -- no re-bundle needed.

Job-key note: the first (dataset, seed=42) run for vitextvqa/vivqax predates
multi-seed and used a bare `ood-eval:<dataset>` key (already collected,
recorded in RESULTS.md) -- kept as-is rather than re-running. Every other
(dataset, seed) combo uses `ood-eval:<dataset>:s<seed>`.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, ACCT_DIR, _kaggle, _user, _code, _clone_cell, _nb, load_ledger, save_ledger, _register  # noqa

BRANCH = "feat/decoder-lora"
CKPT_DS = "mvlm-lora-mt-ckpt"
CKPT_LABEL = "l3ep-s42"           # checkpoints/expA-lora16-3ep/seed42/multi_token/last_model.pt
DATASETS = ["vitextvqa", "vivqax", "openvivqa", "vivqa", "vietcult"]
SEEDS = [42, 123, 3407]
N_SAMPLES = 1000
# legacy bare-key jobs (pre-dates multi-seed, already collected/running under
# the old `ood-eval:<dataset>` naming) -- don't re-launch these under a seed key.
# vivqa is NOT in here: added after multi-seed, all 3 of its seeds are new jobs.
LEGACY_S42 = {"vitextvqa", "vivqax", "openvivqa"}
# fresh accounts, untouched by vintern-ft-minimal (acc2, acc15), RQ6 reruns
# (acc9, acc10), or the first-round OOD jobs (acc7, acc8, acc11).
ACCS = ["acc7", "acc8", "acc14", "acc3", "acc4", "acc5", "acc6", "acc12", "acc13", "acc16"]


def _job_key(dataset: str, seed: int) -> str:
    if seed == 42 and dataset in LEGACY_S42:
        return f"ood-eval:{dataset}"
    return f"ood-eval:{dataset}:s{seed}"


def _tag(dataset: str, seed: int) -> str:
    """Path/slug-safe tag: bare dataset name for the legacy seed42 jobs, dataset_sNNN otherwise."""
    if seed == 42 and dataset in LEGACY_S42:
        return dataset
    return f"{dataset}_s{seed}"


def _cells(dataset: str, seed: int) -> list[dict]:
    tag = _tag(dataset, seed)
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
              f"--seed {seed} --out /kaggle/working/data/{tag} 2>&1 | tail -40"),
        _code(f"!python experiments/ood-eval/gen_vintern_base.py "
              f"--data /kaggle/working/data/{tag}/internvl.jsonl "
              f"--images-dir /kaggle/working/data/{tag}/images "
              f"--out /kaggle/working/out/{tag}/vintern_base 2>&1 | tail -60"),
        _code(f"!python experiments/ood-eval/eval_ours_ood.py "
              f"--data /kaggle/working/data/{tag}/ours.jsonl "
              f"--images-dir /kaggle/working/data/{tag}/images "
              f"--checkpoint /kaggle/working/ck/multi_token/model.pt "
              f"--output-dir /kaggle/working/out/{tag}/ours 2>&1 | tail -60"),
        _code("import os, shutil",
              f"os.makedirs('/kaggle/working/out/{tag}', exist_ok=True)",
              f"shutil.copy('/kaggle/working/data/{tag}/manifest.json', '/kaggle/working/out/{tag}/manifest.json')",
              f"print('out:', os.listdir('/kaggle/working/out/{tag}'))"),
    ]


def cmd_launch() -> None:
    led = load_ledger()
    combos = [(d, s) for d in DATASETS for s in SEEDS]
    i = 0
    for dataset, seed in combos:
        job = _job_key(dataset, seed)
        if led["jobs"].get(job, {}).get("status") == "done":
            print(f"[skip] {job} done"); continue
        if led["jobs"].get(job, {}).get("status") == "running":
            print(f"[skip] {job} already running"); continue
        acc = ACCS[i % len(ACCS)]
        i += 1
        tag = _tag(dataset, seed)
        slug = f"mvlm-ood-eval-{tag}".replace("_", "-")
        user = _user(acc)
        kid = f"{user}/{slug}"
        wd = ROOT / "outputs" / "parallel" / "workers" / slug
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "worker.ipynb").write_text(json.dumps(_nb(_cells(dataset, seed))))
        (wd / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": slug[:50], "code_file": "worker.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [f"{_user('acc1')}/{CKPT_DS}"],
        }, indent=2))
        _kaggle(acc, "kernels", "push", "-p", str(wd))
        _register(led, job, acc, kid, {"dataset": dataset, "n": N_SAMPLES, "seed": seed, "tag": tag})
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
        tag = j.get("tag", j["dataset"])
        dst = out_root / tag
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
