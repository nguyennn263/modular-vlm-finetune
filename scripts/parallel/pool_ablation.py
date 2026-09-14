#!/usr/bin/env python3
"""Advisor follow-up ablation 2: patches->tokens pooling operator, mean vs max
vs attention (fair, controlled comparison).

Only mean/max are NEW runs here. The "attention-pool" arm is `tile_attention`
(AttentionBridge), which already has full 3-seed Exp A data (F1 45.17+-0.94,
CIDEr 84.21+-1.71 plain) -- reused as-is, zero new training needed.

CAVEAT (report this alongside any result): PatchPoolBridge has only the shared
per-patch Linear(1024->896) (~0.92M params) -- no other learnable weights,
deliberately, to isolate the pooling *operator*. AttentionBridge has ~4.14M
params (self-attn + cross-attn + learnable queries). If attention wins, that
is NOT clean evidence "attention as an operator" beats pooling -- it also has
~4x the capacity. State the params next to F1, not just in prose.

    python scripts/parallel/pool_ablation.py smoke     # 1 job, --limit 20 sanity check
    python scripts/parallel/pool_ablation.py launch     # 2 real jobs (seed 42)
    python scripts/parallel/pool_ablation.py collect
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, _kaggle, _user, _code, _clone_cell, _nb, _push_worker, load_ledger, save_ledger, _register  # noqa

BRANCH = "feat/bridge-design-ablation"
SEED = 42
# (label, bridge_type). bridge_type alone disambiguates output_dir -- no suffix needed.
SPECS = [("mean", "patch_pool_mean"), ("max", "patch_pool_max")]
ACCS = ["acc13", "acc7"]


def _cells(label: str, bridge_type: str, limit: int) -> list[dict]:
    ck = f"/kaggle/working/poolabl-{label}/seed{SEED}"
    limit_arg = f"--limit {limit} " if limit else ""
    eval_limit_arg = f" --limit {min(limit * 10, 500)}" if limit else ""
    return [
        _clone_cell(BRANCH),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!python scripts/phase0_build_data.py 2>&1 | tail -6"),
        _code(f"!python -m src.cli.train --bridge {bridge_type} "
              f"--split-dir data/splits --seed {SEED} --epochs 2 {limit_arg}"
              f"--batch-size 8 --grad-accum 1 --eval-steps 800 --save-steps 800 "
              f"--no-early-stopping --text-metrics-every 2 --text-metrics-max-samples 600 "
              f"--output-dir {ck}"),
        _code(f"!python -m src.cli.evaluate --bridge {bridge_type} --split-dir data/splits --split val "
              f"--checkpoint {ck}/{bridge_type}/last_model.pt{eval_limit_arg}"),
        _code(f"!mkdir -p /kaggle/working/out && cp -r {ck} /kaggle/working/out/ && "
              "ls -R /kaggle/working/out | tail -20"),
    ]


def cmd_smoke() -> None:
    label, bt = SPECS[0]
    kid = _push_worker(ACCS[0], f"mvlm-poolabl-smoke-{label}", _cells(label, bt, limit=20), None)
    print(f"[smoke] pushed {kid} -- check the log for bridge_type={bt} and a successful "
          f"eval_val.json write.")


def cmd_launch() -> None:
    led = load_ledger()
    for i, (label, bt) in enumerate(SPECS):
        job = f"poolabl:{label}"
        if led["jobs"].get(job, {}).get("status") == "done":
            print(f"[skip] {job} done"); continue
        acc = ACCS[i % len(ACCS)]
        slug = f"mvlm-poolabl-{label}"
        kid = _push_worker(acc, slug, _cells(label, bt, limit=0), None)
        _register(led, job, acc, kid, {"label": label, "bridge_type": bt})
        time.sleep(2)


def cmd_collect() -> None:
    led = load_ledger()
    out_root = ROOT / "outputs" / "pool_ablation"
    out_root.mkdir(parents=True, exist_ok=True)
    for job, j in led["jobs"].items():
        if not job.startswith("poolabl:") or j.get("status") == "done":
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
            print(f"[ok] {job} ({j['bridge_type']}): "
                  f"F1 {d.get('f1', 0)*100:.2f}  CIDEr {d.get('cider', 0)*100:.2f}  loss {d.get('loss', 0):.3f}")
        else:
            print(f"[partial] {job}: no eval_val.json in {dst}")
    save_ledger(led)


if __name__ == "__main__":
    {"smoke": cmd_smoke, "launch": cmd_launch, "collect": cmd_collect}[sys.argv[1]]()
