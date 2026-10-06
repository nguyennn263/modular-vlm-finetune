#!/usr/bin/env python3
"""Run experiments/token-select/vintern_topk.py (training-free top-k of Vintern's mlp1
tokens) on the OOD sets, one Kaggle session per dataset, seed-42 1000-question subset
(the same questions as the OOD table).

    python scripts/parallel/token_select.py fill [--smoke] [--limit=N] [--datasets=a,b]   # --smoke: 5 questions, vitextvqa only
    python scripts/parallel/token_select.py collect
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, ACCT_DIR, _kaggle, _user, _code, _clone_cell, _nb, load_ledger, save_ledger  # noqa
from input_diag import BRANCH, DOCKER_IMAGE
from ood_full import _free_slots

SMOKE = "--smoke" in sys.argv
# --limit=N: first N questions of the seed-42 subset (already a random sample of the set)
LIMIT = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--limit=")), 5 if SMOKE else 0)
SET = "tsel-smoke" if SMOKE else (f"tsel{LIMIT}" if LIMIT else "tsel")
DATASETS = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--datasets=")),
                ["vitextvqa"] if SMOKE else ["vitextvqa", "openvivqa"])
KS = "7,14,32"


def _cells(dataset: str) -> list[dict]:
    data = f"/kaggle/working/data/{dataset}"
    limit = f" --limit {LIMIT}" if LIMIT else ""
    keep = (f"import json, shutil, os; os.makedirs('/kaggle/working/out/{dataset}/images', exist_ok=True); "
            f"[shutil.copy('{data}/images/' + json.loads(l)['image'], '/kaggle/working/out/{dataset}/images/') "
            f"for l in list(open('{data}/internvl.jsonl'))[:{LIMIT}]]") if LIMIT else "pass"
    return [
        _clone_cell(BRANCH),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!pip -q install requests"),
        _code(f"!python experiments/ood-eval/build_ood_data.py --dataset {dataset} --n 1000 "
              f"--seed 42 --out {data} 2>&1 | tail -5"),
        _code(f"!python experiments/token-select/vintern_topk.py --data {data}/internvl.jsonl "
              f"--images-dir {data}/images --out /kaggle/working/out/{dataset} --ks {KS}{limit}"),
        _code(keep),
        _code(f"!rm -rf {data}/images {data}/_raw && ls -la /kaggle/working/out/{dataset}"),
    ]


def _launch(acc: str, dataset: str) -> None:
    job = f"{SET}:{dataset}"
    slug = f"mvlm-{SET}-{dataset}"
    kid = f"{_user(acc)}/{slug}"
    wd = ROOT / "outputs" / "parallel" / "workers" / slug
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "worker.ipynb").write_text(json.dumps(_nb(_cells(dataset))))
    (wd / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": slug[:50], "code_file": "worker.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "docker_image": DOCKER_IMAGE,
        "dataset_sources": [], "kernel_sources": [],
    }, indent=2))
    _kaggle(acc, "kernels", "push", "-p", str(wd))
    led = load_ledger()
    led["jobs"][job] = {"account": acc, "kernel": kid, "status": "running",
                        "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset": dataset}
    save_ledger(led)
    print(f"[launch] {job} -> {acc} ({kid})", flush=True)


def cmd_fill() -> None:
    accounts = sorted((p.name for p in ACCT_DIR.glob("acc*") if (p / "kaggle.json").exists()
                       and p.name[3:].isdigit()), key=lambda s: int(s[3:]))
    queue = [d for d in DATASETS if f"{SET}:{d}" not in load_ledger()["jobs"]]
    for acc in accounts:
        if not queue:
            break
        if _free_slots(acc) > 0:
            _launch(acc, queue.pop(0))
    print("queued but not launched:", queue or "none", flush=True)


def cmd_collect() -> None:
    led = load_ledger()
    done = []
    for job, j in led["jobs"].items():
        if not job.startswith(SET + ":") or j.get("status") != "running":
            continue
        st = _kaggle(j["account"], "kernels", "status", j["kernel"], check=False)
        if "COMPLETE" not in st:
            print(f"[wait] {job}: {st.strip()[-40:]}")
            continue
        dst = ROOT / "outputs" / "token_select" / job.split(":", 1)[1]
        _kaggle(j["account"], "kernels", "output", j["kernel"], "--file-pattern", r".*\.(json|log|jpg|jpeg|png)$",
                "-p", str(dst), check=False)
        summ = next(dst.rglob("summary.json"), None)
        if summ:
            done.append(job)
            for c, m in json.loads(summ.read_text())["summary"].items():
                print(f"[ok] {job} {c:12s} F1 {m['f1']:6.2f}  CIDEr {m['cider']:7.2f}")
    led = load_ledger()
    for job in done:
        led["jobs"][job]["status"] = "done"
    save_ledger(led)


if __name__ == "__main__":
    {"fill": cmd_fill, "collect": cmd_collect}[sys.argv[1]]()
