#!/usr/bin/env python3
"""Re-run OUR side of the OOD eval with --gen-image full (the original run fed
generation the first 448px crop). Vintern-base predictions are unaffected and
are reused from outputs/ood_eval/; score both locally on the shared sample ids.

Jobs only go to accounts in ACCS whose latest kernels are all finished -- another
session launches onto the other accounts, and an account may already be running
one of our input-diag kernels.

    python scripts/parallel/ood_full.py fill      # launch onto idle accounts, until the queue is empty
    python scripts/parallel/ood_full.py collect
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, _kaggle, _user, _code, _clone_cell, _nb, load_ledger, save_ledger  # noqa
from input_diag import BRANCH, DOCKER_IMAGE, LORA_DS

CKPT = "l3ep-s42.pt"   # the checkpoint the original OOD run used
DATASETS = ["vitextvqa", "vivqax", "openvivqa", "vivqa"]
SEEDS = [42, 123, 3407]
ACCS = ["acc2", "acc7", "acc8", "acc10", "acc11", "acc13", "acc15", "acc16"]
BUSY = ("RUNNING", "QUEUED", "NEW")


def _cells(dataset: str, seed: int) -> list[dict]:
    tag = f"{dataset}_s{seed}"
    data = f"/kaggle/working/data/{tag}"
    out = f"/kaggle/working/out/{tag}"
    return [
        _clone_cell(BRANCH),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!pip -q install requests"),
        _code("import glob, os, shutil",
              f"src = glob.glob('/kaggle/input/**/{CKPT}', recursive=True)",
              "assert src, 'ckpt not found: ' + repr(os.listdir('/kaggle/input'))",
              "os.makedirs('/tmp/ck/multi_token', exist_ok=True)",
              "shutil.copy(src[0], '/tmp/ck/multi_token/model.pt')",
              "print('ckpt ready', src[0])"),
        _code(f"!python experiments/ood-eval/build_ood_data.py --dataset {dataset} --n 1000 "
              f"--seed {seed} --out {data} 2>&1 | tail -20"),
        _code(f"!python experiments/ood-eval/eval_ours_ood.py --data {data}/ours.jsonl "
              f"--images-dir {data}/images --checkpoint /tmp/ck/multi_token/model.pt "
              f"--gen-image full --output-dir /tmp/ours 2>&1 | tail -30"),
        _code(f"os.makedirs('{out}', exist_ok=True)",
              f"for f in ['{data}/ours.jsonl', '{data}/manifest.json', '/tmp/ours/eval_ood.json',",
              "          '/tmp/ours/results/text_predictions_epoch_1.json']:",
              f"    shutil.copy(f, '{out}')",
              f"print(os.listdir('{out}'))"),
    ]


def _idle(acc: str) -> bool:
    refs = [l.split(",")[0] for l in
            _kaggle(acc, "kernels", "list", "--mine", "--sort-by", "dateRun", "--page-size", "4",
                    "--csv", check=False).splitlines()[1:] if l.count(",") and "/" in l.split(",")[0]]
    for ref in refs:
        st = _kaggle(acc, "kernels", "status", ref, check=False)
        if any(f"KernelWorkerStatus.{b}" in st for b in BUSY):
            return False
    return True


def _launch(acc: str, dataset: str, seed: int) -> None:
    job = f"ood-full:{dataset}:s{seed}"
    slug = f"mvlm-oodfull-{dataset}-s{seed}"
    kid = f"{_user(acc)}/{slug}"
    wd = ROOT / "outputs" / "parallel" / "workers" / slug
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "worker.ipynb").write_text(json.dumps(_nb(_cells(dataset, seed))))
    (wd / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": slug[:50], "code_file": "worker.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "docker_image": DOCKER_IMAGE,
        "dataset_sources": [f"{_user('acc1')}/{LORA_DS}"],
    }, indent=2))
    _kaggle(acc, "kernels", "push", "-p", str(wd))
    led = load_ledger()  # re-read: another session writes this file too
    led["jobs"][job] = {"account": acc, "kernel": kid, "status": "running",
                        "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "dataset": dataset, "seed": seed, "gen_image": "full"}
    save_ledger(led)
    print(f"[launch] {job} -> {acc} ({kid})", flush=True)


def cmd_fill() -> None:
    led = load_ledger()
    queue = [(d, s) for d in DATASETS for s in SEEDS if f"ood-full:{d}:s{s}" not in led["jobs"]]
    while queue:
        used = {j["account"] for k, j in load_ledger()["jobs"].items()
                if k.startswith("ood-full:") and j["status"] == "running"}
        for acc in ACCS:
            if queue and acc not in used and _idle(acc):
                _launch(acc, *queue.pop(0))
                time.sleep(5)
        if queue:
            print(time.strftime("%H:%M:%S"), f"{len(queue)} queued, waiting for an idle account", flush=True)
            time.sleep(300)
    print("ALL LAUNCHED", flush=True)


def cmd_collect() -> None:
    led = load_ledger()
    out_root = ROOT / "outputs" / "ood_full"
    done = []
    for job, j in led["jobs"].items():
        if not job.startswith("ood-full:") or j.get("status") == "done":
            continue
        st = _kaggle(j["account"], "kernels", "status", j["kernel"], check=False)
        if "COMPLETE" not in st:
            print(f"[wait] {job}: {st.strip()[-40:]}"); continue
        tag = job.split(":", 1)[1].replace(":", "_")
        _kaggle(j["account"], "kernels", "output", j["kernel"], "-p", str(out_root / tag), check=False)
        if next((out_root / tag).rglob("text_predictions_epoch_1.json"), None):
            done.append(job)
            print(f"[ok] {job}")
    led = load_ledger()
    for job in done:
        led["jobs"][job]["status"] = "done"
    save_ledger(led)


if __name__ == "__main__":
    {"fill": cmd_fill, "collect": cmd_collect}[sys.argv[1]]()
