#!/usr/bin/env python3
"""Advisor follow-up ablation 1: num_tokens sweep for the Multi-Token bridge.

Why num_tokens=8 and not 6/10/12? The paper's recipe bridge (Multi-Token) has
always used num_tokens=8 as a shared default across several bridge types (for
comparability), never actually swept for Multi-Token specifically. This sweeps
it, plain bridge (no LoRA), 2 epochs, matching the original Exp A protocol.
num_tokens=8 already has full 4-seed Exp A data (F1 49.55+-0.07) -- not rerun.

    python scripts/parallel/token_sweep.py smoke     # 1 job, --limit 20 sanity check
    python scripts/parallel/token_sweep.py launch     # 4 real jobs (seed 42)
    python scripts/parallel/token_sweep.py collect
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, _kaggle, _user, _code, _clone_cell, _nb, _push_worker, load_ledger, save_ledger, _register  # noqa

BRANCH = "feat/bridge-design-ablation"
SEED = 42
# (label, num_tokens). num_tokens=8 already has 4-seed Exp A data -- not rerun here.
SPECS = [("tok4", 4), ("tok6", 6), ("tok10", 10), ("tok12", 12)]
# top-quota-remaining accounts as of 2026-09-14 (excludes acc2/acc15, exhausted)
ACCS = ["acc16", "acc14", "acc11", "acc12"]


def _cells(label: str, n: int, limit: int) -> list[dict]:
    ck = f"/kaggle/working/toksweep-{label}/seed{SEED}"
    limit_arg = f"--limit {limit} " if limit else ""
    eval_limit_arg = f" --limit {min(limit * 10, 500)}" if limit else ""
    return [
        _clone_cell(BRANCH),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!python scripts/phase0_build_data.py 2>&1 | tail -6"),
        _code(f"!python -m src.cli.train --bridge multi_token --bridge-num-tokens {n} "
              f"--split-dir data/splits --seed {SEED} --epochs 2 {limit_arg}"
              f"--batch-size 8 --grad-accum 1 --eval-steps 800 --save-steps 800 "
              f"--no-early-stopping --text-metrics-every 2 --text-metrics-max-samples 600 "
              f"--output-dir {ck}"),
        _code(f"!python -m src.cli.evaluate --bridge multi_token --bridge-num-tokens {n} "
              f"--split-dir data/splits --split val "
              f"--checkpoint {ck}/multi_token/last_model.pt{eval_limit_arg}"),
        _code(f"!mkdir -p /kaggle/working/out && cp -r {ck} /kaggle/working/out/ && "
              "ls -R /kaggle/working/out | tail -20"),
    ]


def cmd_smoke() -> None:
    label, n = SPECS[0]
    kid = _push_worker(ACCS[0], f"mvlm-toksweep-smoke-{label}", _cells(label, n, limit=20), None)
    print(f"[smoke] pushed {kid} -- check the log for num_tokens={n} in the bridge_config and "
          f"a successful eval_val.json write.")


def cmd_launch() -> None:
    led = load_ledger()
    for i, (label, n) in enumerate(SPECS):
        job = f"toksweep:{label}"
        if led["jobs"].get(job, {}).get("status") == "done":
            print(f"[skip] {job} done"); continue
        acc = ACCS[i % len(ACCS)]
        slug = f"mvlm-toksweep-{label}"
        kid = _push_worker(acc, slug, _cells(label, n, limit=0), None)
        _register(led, job, acc, kid, {"label": label, "num_tokens": n})
        time.sleep(2)


def cmd_collect() -> None:
    led = load_ledger()
    out_root = ROOT / "outputs" / "token_sweep"
    out_root.mkdir(parents=True, exist_ok=True)
    for job, j in led["jobs"].items():
        if not job.startswith("toksweep:") or j.get("status") == "done":
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
            print(f"[ok] {job} (num_tokens={j['num_tokens']}): "
                  f"F1 {d.get('f1', 0)*100:.2f}  CIDEr {d.get('cider', 0)*100:.2f}  loss {d.get('loss', 0):.3f}")
        else:
            print(f"[partial] {job}: no eval_val.json in {dst}")
    save_ledger(led)


if __name__ == "__main__":
    {"smoke": cmd_smoke, "launch": cmd_launch, "collect": cmd_collect}[sys.argv[1]]()
