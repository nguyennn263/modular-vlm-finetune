#!/usr/bin/env python3
"""Out-of-distribution eval of the global-local bridge (g=14, k=36 and k=144, seed 42) on
the same 1000-question subsets as the OOD table (build_ood_data.py, sampling seeds
42/123/3407). One Kaggle session per dataset. The checkpoints are a private dataset on
the launching account (`<user>/mvlm-gl-ckpt`), so every job runs on that account.

    python scripts/parallel/ood_gl.py fill --acc=acc5      # launch as slots free up
    python scripts/parallel/ood_gl.py collect --acc=acc5

--lora: the g14-k36 + decoder LoRA model instead (seed 42, private dataset
`<user>/mvlm-gl-lora-ckpt`); its own job names, kernels and outputs/ood_gl_lora/.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, _kaggle, _user, _code, _clone_cell, _nb, load_ledger, save_ledger  # noqa
from input_diag import BRANCH, DOCKER_IMAGE
from ood_full import _free_slots
from train_queue import _checked

ACC = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--acc="))
DATASETS = ["vitextvqa", "openvivqa", "vivqax", "vivqa"]
SEEDS = [42, 123, 3407]
LORA = "--lora" in sys.argv
# name -> (local_grid, checkpoint file in the private dataset)
MODELS = ({"k36lora": (6, "gl-g14-k36-lora1-s42.pt")} if LORA else
          {"k36": (6, "gl-g14-k36-s42.pt"), "k144": (12, "gl-g14-k144-s42.pt")})
CKPT_DS = "mvlm-gl-lora-ckpt" if LORA else "mvlm-gl-ckpt"
JOB = "ood-gllora" if LORA else "ood-gl"
FINISHED = ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED")


def _cells(dataset: str) -> list[dict]:
    cells = [_clone_cell(BRANCH), _code("!bash setup_kaggle.sh 2>&1 | tail -5"), _code("!pip -q install requests"),
             _code("import glob, os, shutil",
                   f"ck = {{m: glob.glob(f'/kaggle/input/**/{{f}}', recursive=True) for m, f in "
                   f"{ {m: f for m, (_, f) in MODELS.items()} }.items()}}",
                   "assert all(ck.values()), 'checkpoints not found: ' + repr(ck)",
                   "os.makedirs('/tmp/ck', exist_ok=True)",
                   "for m, f in ck.items(): shutil.copy(f[0], f'/tmp/ck/{m}.pt')",
                   "print(ck, os.listdir('/tmp/ck'))")]
    for seed in SEEDS:
        tag = f"{dataset}_s{seed}"
        data, out = f"/tmp/data/{tag}", f"/kaggle/working/out/{tag}"
        cells.append(_checked(f"python experiments/ood-eval/build_ood_data.py --dataset {dataset} --n 1000 "
                              f"--seed {seed} --out {data} 2>&1 | tail -5; test -s {data}/ours.jsonl"))
        cells.append(_code(f"os.makedirs('{out}', exist_ok=True)",
                           f"for f in ['{data}/ours.jsonl', '{data}/manifest.json']: shutil.copy(f, '{out}')"))
        for m, (grid, _) in MODELS.items():
            res = f"/tmp/res/{tag}/{m}"
            cells.append(_checked(
                f"python experiments/ood-eval/eval_ours_ood.py --data {data}/ours.jsonl --images-dir {data}/images "
                f"--bridge global_local --bridge-num-tokens 14 --local-grid {grid} "
                f"--checkpoint /tmp/ck/{m}.pt --gen-image full --output-dir {res} 2>&1 | tail -25; "
                f"test -s {res}/results/text_predictions_epoch_1.json"))
            cells.append(_code(f"os.makedirs('{out}/{m}', exist_ok=True)",
                               f"for f in ['{res}/eval_ood.json', '{res}/results/text_predictions_epoch_1.json']: "
                               f"shutil.copy(f, '{out}/{m}')"))
    return cells


def _launch(dataset: str) -> None:
    job, slug = f"{JOB}:{dataset}", f"mvlm-{JOB.replace('-', '')}-{dataset}"
    kid = f"{_user(ACC)}/{slug}"
    wd = ROOT / "outputs" / "parallel" / "workers" / slug
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "worker.ipynb").write_text(json.dumps(_nb(_cells(dataset))))
    (wd / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": slug[:50], "code_file": "worker.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "docker_image": DOCKER_IMAGE,
        "dataset_sources": [f"{_user(ACC)}/{CKPT_DS}"], "kernel_sources": [],
    }, indent=2))
    _kaggle(ACC, "kernels", "push", "-p", str(wd))
    led = load_ledger()
    led["jobs"][job] = {"account": ACC, "kernel": kid, "status": "running",
                        "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset": dataset}
    save_ledger(led)
    print(f"[launch] {job} -> {ACC} ({kid})", flush=True)


def cmd_fill() -> None:
    queue = [d for d in DATASETS if f"{JOB}:{d}" not in load_ledger()["jobs"]]
    while queue:
        if _free_slots(ACC) > 0:
            _launch(queue.pop(0))
            time.sleep(60)            # let the new kernel show up before counting slots again
            continue
        time.sleep(300)
    print("ALL LAUNCHED", flush=True)


def cmd_collect() -> None:
    led = load_ledger()
    done = []
    for job, j in led["jobs"].items():
        if not job.startswith(f"{JOB}:") or j.get("status") != "running":
            continue
        st = _kaggle(j["account"], "kernels", "status", j["kernel"], check=False)
        if "COMPLETE" not in st:
            print(f"[wait] {job}: {st.strip()[-40:]}")
            continue
        dst = ROOT / "outputs" / ("ood_gl_lora" if LORA else "ood_gl") / job.split(":", 1)[1]
        _kaggle(j["account"], "kernels", "output", j["kernel"], "--file-pattern", r".*\.(json|jsonl|log)$",
                "-p", str(dst), check=False)
        evs = sorted(dst.rglob("eval_ood.json"))
        for ev in evs:
            d = json.loads(ev.read_text())
            print(f"[ok] {job} {ev.parent.parent.name}/{ev.parent.name}: F1 {100 * d.get('f1', 0):.2f}")
        if len(evs) == len(SEEDS) * len(MODELS):
            done.append(job)
    led = load_ledger()
    for job in done:
        led["jobs"][job]["status"] = "done"
    save_ledger(led)


if __name__ == "__main__":
    {"fill": cmd_fill, "collect": cmd_collect}[sys.argv[1]]()
