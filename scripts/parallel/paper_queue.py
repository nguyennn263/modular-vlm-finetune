#!/usr/bin/env python3
"""Single scheduler for every remaining --gen-image full re-score the paper needs,
across all accounts, one kernel per idle account (replaces running ood_full.py and
regen_queue.py fill side by side, which raced for the same accounts).

Order: OOD -> test rows of the main table -> one-seed bridge rows (checkpoints already
on Kaggle) -> RQ5 + remaining bridge seeds (local checkpoints, uploaded just before
launch as a PRIVATE dataset on the account that runs the job).

    python scripts/parallel/paper_queue.py fill
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, ACCT_DIR, _kaggle, _user, load_ledger, save_ledger  # noqa
import ood_full, regen_queue
from ood_full import _idle
from regen_queue import _cells
from input_diag import DOCKER_IMAGE

LOCAL = Path("/Users/tcx/Documents/personal/Repo/modular-vlm-finetune/checkpoints")
STAGE = Path("/private/tmp/claude-501/-Users-tcx-Documents-personal-Repo-modular-vlm-finetune/"
             "7534ec64-b50f-4da3-b323-052086c8df60/scratchpad/pv")
ACCS = sorted((p.name for p in ACCT_DIR.glob("acc*") if (p / "kaggle.json").exists()
               and p.name[3:].isdigit()), key=lambda s: int(s[3:]))

# (tag, bridge, local checkpoint) -- val only
PRIVATE = (
    [(f"rq5-{t}-s{s}", "multi_token", LOCAL / d / f"seed{s}" / "multi_token" / "last_model.pt")
     for t, d in [("feat", "expA-align-feat"), ("logit01", "expA-align-logit-a01"),
                  ("logit1", "expA-align-logit"), ("random", "expA-random")]
     for s in (42, 123, 3407)]
    + [(f"{t}-s{s}", b, LOCAL / "expA" / f"seed{s}" / b / "last_model.pt")
       for t, b, seeds in [("res", "residual", (123, 3407)), ("ta", "tile_attention", (123, 3407)),
                           ("mq", "mini_qformer", (123, 3407)), ("qf", "qformer", (42, 123))]
       for s in seeds]
    # tab:bridges "+LoRA" column: decoder LoRA r16, 1 epoch (tile_attention only has seed 42)
    + [(f"l1ep-{t}-s{s}", b, LOCAL / "expA-lora16" / f"seed{s}" / b / "last_model.pt")
       for t, b, seeds in [("res", "residual", (42, 123, 3407)), ("ta", "tile_attention", (42,)),
                           ("mq", "mini_qformer", (42, 123, 3407)), ("qf", "qformer", (42, 123, 3407))]
       for s in seeds]
)


def _queue() -> list[tuple]:
    jobs = load_ledger()["jobs"]
    q = [("ood", d, s) for d in ood_full.DATASETS for s in ood_full.SEEDS
         if f"ood-full:{d}:s{s}" not in jobs]
    q += [("regen",) + j for j in regen_queue.JOBS
          if j[1] == "multi_token" and f"regen-full:{j[0]}" not in jobs]
    q += [("private",) + j for j in PRIVATE if j[0].startswith("rq5-") and f"regen-full:{j[0]}" not in jobs]
    if "--bridges" in sys.argv:  # tab:bridges plain rows -- only once the LoRA side is settled
        q += [("regen",) + j for j in regen_queue.JOBS
              if j[1] != "multi_token" and f"regen-full:{j[0]}" not in jobs]
        q += [("private",) + j for j in PRIVATE
              if not j[0].startswith("rq5-") and f"regen-full:{j[0]}" not in jobs]
    return q


def _upload_private(acc: str, tag: str, pt: Path) -> str:
    ds = f"{_user(acc)}/mvlm-pv-{tag}"
    d = STAGE / tag
    d.mkdir(parents=True, exist_ok=True)
    if not (d / f"{tag}.pt").exists():
        (d / f"{tag}.pt").hardlink_to(pt)
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"id": ds, "title": f"mvlm-pv-{tag}", "licenses": [{"name": "unknown"}]}))
    if "ready" not in _kaggle(acc, "datasets", "status", ds, check=False):  # retry-safe
        _kaggle(acc, "datasets", "create", "-p", str(d))  # private by default
    for _ in range(60):
        if "ready" in _kaggle(acc, "datasets", "status", ds, check=False):
            return ds
        time.sleep(20)
    raise RuntimeError(f"{ds} not ready after 20 min")


def _launch_private(acc: str, tag: str, bridge: str, pt: Path) -> None:
    ds = _upload_private(acc, tag, pt)
    job, slug = f"regen-full:{tag}", f"mvlm-regen-{tag}"
    kid = f"{_user(acc)}/{slug}"
    wd = ROOT / "outputs" / "parallel" / "workers" / slug
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "worker.ipynb").write_text(json.dumps(regen_queue._nb(_cells(tag, bridge, f"{tag}.pt", ["val"]))))
    (wd / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": slug[:50], "code_file": "worker.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "docker_image": DOCKER_IMAGE,
        "dataset_sources": ["nguynrichard/auto-vqabest", ds],
    }, indent=2))
    _kaggle(acc, "kernels", "push", "-p", str(wd))
    led = load_ledger()
    led["jobs"][job] = {"account": acc, "kernel": kid, "status": "running",
                        "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "bridge": bridge, "splits": ["val"], "gen_image": "full", "dataset": ds}
    save_ledger(led)
    print(f"[launch] {job} -> {acc} ({kid}) via private {ds}", flush=True)


def cmd_fill() -> None:
    queue = _queue()
    print(f"{len(queue)} jobs queued on {len(ACCS)} accounts", flush=True)
    while queue:
        for acc in ACCS:
            if not queue or not _idle(acc):
                continue
            kind, *spec = queue.pop(0)
            try:
                {"ood": ood_full._launch, "regen": regen_queue._launch,
                 "private": _launch_private}[kind](acc, *spec)
            except Exception as exc:  # keep the loop alive; retry this job on another account
                print(f"[error] {kind} {spec[0]} on {acc}: {exc}", flush=True)
                queue.insert(0, (kind, *spec))
            time.sleep(5)
        if queue:
            print(time.strftime("%H:%M:%S"), f"{len(queue)} queued", flush=True)
            time.sleep(300)
    print("ALL LAUNCHED", flush=True)


if __name__ == "__main__":
    {"fill": cmd_fill}[sys.argv[1]]()
