#!/usr/bin/env python3
"""RQ6 re-run with the bridge ACTUALLY frozen during the LoRA stage (see
src/training/trainer.py's `freeze_bridge` fix, feat/decoder-lora).

The original RQ6 attn+MLP / MLP-only LoRA runs left the bridge trainable the
whole time (a real bug, confirmed by direct code read -- _setup_optimization()
never touched self.model.bridge). This reruns just those 2 diverging configs,
seed42 only for now, with --freeze-bridge added, to see whether the divergence
survives once the confound is removed.

    python scripts/parallel/rq6_freeze_bridge.py smoke     # --limit 20 sanity check (1 job)
    python scripts/parallel/rq6_freeze_bridge.py launch     # real reruns (2 jobs)
    python scripts/parallel/rq6_freeze_bridge.py collect
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, ACCT_DIR, _kaggle, _user, _code, _clone_cell, _nb, load_ledger, save_ledger, _register  # noqa

BRANCH = "feat/decoder-lora"
# (label, lora_targets, resume ckpt dataset holding the pretrained seed42 bridge)
SPECS = [
    ("lora16-all-fb", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"),
    ("lora16-mlp-fb", "gate_proj,up_proj,down_proj"),
]
# the pretrained (2-epoch, plain) multi_token bridge checkpoint, seed42 -- same
# starting point the original (buggy) RQ6 runs resumed from.
BRIDGE_CKPT_DS = "mvlm-expa-ckpt"  # produced by scripts/parallel/run.py bundle
ACCS = ["acc9", "acc10"]


def _cells(label: str, lora_targets: str, limit: int) -> list[dict]:
    ck = f"/kaggle/working/ck-{label}/seed42"
    limit_arg = f"--limit {limit} " if limit else ""
    # smoke (--limit) also caps the post-train eval to a few hundred samples --
    # otherwise the eval alone (full 5463-sample generation-based val) dwarfs
    # the point of a "quick" sanity check.
    eval_limit_arg = f" --limit {min(limit * 10, 500)}" if limit else ""
    return [
        _clone_cell(BRANCH),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!python scripts/phase0_build_data.py 2>&1 | tail -6"),
        # --init-bridge (NOT --resume): loads only bridge_state, no
        # optimizer/scheduler/step restore. --resume would try to restore the
        # OLD (bridge-only) optimizer state into a differently-shaped LoRA
        # optimizer and crash ("parameter group doesn't match the size") --
        # confirmed the hard way in the first smoke test attempt. Resolve the
        # ckpt path and run training in the SAME cell (avoids relying on
        # cross-cell shell/python variable interpolation).
        _code("import os, glob",
              "pts = glob.glob('/kaggle/input/**/multi_token/best_model.pt', recursive=True) or "
              "glob.glob('/kaggle/input/**/best_model.pt', recursive=True)",
              "assert pts, 'pretrained bridge ckpt not found: ' + repr(os.listdir('/kaggle/input'))",
              "print('bridge warm-start source:', pts[0])",
              "rc = os.system(f\"python -m src.cli.train --bridge multi_token --split-dir data/splits "
              f"--seed 42 --epochs 1 {limit_arg}--batch-size 8 --grad-accum 1 --eval-steps 800 "
              f"--save-steps 800 --no-early-stopping --text-metrics-every 2 --text-metrics-max-samples 600 "
              f"--lora --lora-r 16 --lora-targets {lora_targets} --freeze-bridge "
              "--init-bridge {pts[0]!r} "
              f"--output-dir {ck}\")",
              "assert rc == 0, f'training exited {rc}'"),
        _code(f"!python -m src.cli.evaluate --bridge multi_token --split-dir data/splits --split val "
              f"--checkpoint {ck}/multi_token/last_model.pt{eval_limit_arg}"),
        _code(f"!mkdir -p /kaggle/working/out && cp -r {ck} /kaggle/working/out/ && "
              "ls -R /kaggle/working/out | tail -20"),
    ]


def _push(acc: str, slug: str, label: str, lora_targets: str, limit: int) -> str:
    user = _user(acc)
    kid = f"{user}/{slug}"
    wd = ROOT / "outputs" / "parallel" / "workers" / slug
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "worker.ipynb").write_text(json.dumps(_nb(_cells(label, lora_targets, limit))))
    (wd / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": slug[:50], "code_file": "worker.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ["nguynrichard/auto-vqabest", f"{_user('acc1')}/{BRIDGE_CKPT_DS}"],
    }, indent=2))
    _kaggle(acc, "kernels", "push", "-p", str(wd))
    return kid


def cmd_smoke() -> None:
    """1 job, --limit 20, just to confirm the trainable-param log line drops to
    ~LoRA-only% once --freeze-bridge is on (cheap, no ledger tracking)."""
    label, targets = SPECS[0]
    kid = _push("acc9", f"mvlm-rq6-smoke-{label}", label, targets, limit=20)
    print(f"[smoke] pushed {kid} -- check the log for '[freeze_bridge] bridge parameters frozen' "
          f"and the trainable-parameters %.")


def cmd_launch() -> None:
    led = load_ledger()
    for i, (label, targets) in enumerate(SPECS):
        job = f"rq6-freeze:{label}"
        if led["jobs"].get(job, {}).get("status") == "done":
            print(f"[skip] {job} done"); continue
        acc = ACCS[i % len(ACCS)]
        slug = f"mvlm-rq6-{label}"
        kid = _push(acc, slug, label, targets, limit=0)
        _register(led, job, acc, kid, {"label": label, "lora_targets": targets})
        time.sleep(2)


def cmd_collect() -> None:
    led = load_ledger()
    out_root = ROOT / "outputs" / "rq6_freeze_bridge"
    out_root.mkdir(parents=True, exist_ok=True)
    for job, j in led["jobs"].items():
        if not job.startswith("rq6-freeze:") or j.get("status") == "done":
            continue
        st = _kaggle(j["account"], "kernels", "status", j["kernel"], check=False)
        if "COMPLETE" not in st:
            print(f"[wait] {job}: {st.strip()[:60]}"); continue
        dst = out_root / j["label"]
        _kaggle(j["account"], "kernels", "output", j["kernel"], "-p", str(dst), check=False)
        f = next(dst.rglob("eval_val.json"), None)
        if f:
            d = json.loads(f.read_text())
            j["status"] = "done"; j["collected"] = True
            print(f"[ok] {job}: F1 {d.get('f1', 0)*100:.2f}  loss {d.get('loss', 0):.3f}")
        else:
            print(f"[partial] {job}: no eval_val.json yet under {dst}")
    save_ledger(led)


if __name__ == "__main__":
    {"smoke": cmd_smoke, "launch": cmd_launch, "collect": cmd_collect}[sys.argv[1]]()
