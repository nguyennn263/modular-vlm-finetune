#!/usr/bin/env python3
"""Advisor follow-up ablation 3: conv/deconv-based bridge (HoneyBee's
C-Abstractor, Cha et al. arXiv:2312.06742): L ResNet blocks -> adaptive avg
pool -> L ResNet blocks. Preserves LOCAL spatial context via convolution --
unlike every other bridge, which either discards spatial structure entirely
(MultiTokenMLP) or has no 2D inductive bias (AttentionBridge/MiniQFormer).

M=9 (3x3 grid, closest perfect square to the study's usual num_tokens=8) is
the primary run. M=4 (2x2) is an optional secondary run, only launched if M=9
looks competitive (cross-references the token_sweep.py results).

    python scripts/parallel/conv_abstractor.py smoke        # --limit 20 sanity check
    python scripts/parallel/conv_abstractor.py launch        # M=9 (primary)
    python scripts/parallel/conv_abstractor.py launch --m4   # + M=4 (secondary, optional)
    python scripts/parallel/conv_abstractor.py collect
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, _kaggle, _user, _code, _clone_cell, _nb, _push_worker, load_ledger, save_ledger, _register  # noqa

BRANCH = "feat/bridge-design-ablation"
SEED = 42
SPECS_PRIMARY = [("m9", 9)]
SPECS_SECONDARY = [("m4", 4)]
ACCS = ["acc8", "acc6"]


def _cells(label: str, m: int, limit: int) -> list[dict]:
    ck = f"/kaggle/working/convabs-{label}/seed{SEED}"
    limit_arg = f"--limit {limit} " if limit else ""
    eval_limit_arg = f" --limit {min(limit * 10, 500)}" if limit else ""
    return [
        _clone_cell(BRANCH),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!python scripts/phase0_build_data.py 2>&1 | tail -6"),
        _code(f"!python -m src.cli.train --bridge conv_abstractor --bridge-num-tokens {m} "
              f"--split-dir data/splits --seed {SEED} --epochs 2 {limit_arg}"
              f"--batch-size 8 --grad-accum 1 --eval-steps 800 --save-steps 800 "
              f"--no-early-stopping --text-metrics-every 2 --text-metrics-max-samples 600 "
              f"--output-dir {ck}"),
        _code(f"!python -m src.cli.evaluate --bridge conv_abstractor --bridge-num-tokens {m} "
              f"--split-dir data/splits --split val "
              f"--checkpoint {ck}/conv_abstractor/last_model.pt{eval_limit_arg}"),
        _code(f"!mkdir -p /kaggle/working/out && cp -r {ck} /kaggle/working/out/ && "
              "ls -R /kaggle/working/out | tail -20"),
    ]


def cmd_smoke() -> None:
    label, m = SPECS_PRIMARY[0]
    kid = _push_worker(ACCS[0], f"mvlm-convabs-smoke-{label}", _cells(label, m, limit=20), None)
    print(f"[smoke] pushed {kid} -- check the log for num_tokens={m}, the perfect-square "
          f"assertion NOT firing, and a successful eval_val.json write.")


def cmd_launch(m4: bool = False) -> None:
    led = load_ledger()
    specs = SPECS_PRIMARY + (SPECS_SECONDARY if m4 else [])
    for i, (label, m) in enumerate(specs):
        job = f"convabs:{label}"
        if led["jobs"].get(job, {}).get("status") == "done":
            print(f"[skip] {job} done"); continue
        acc = ACCS[i % len(ACCS)]
        slug = f"mvlm-convabs-{label}"
        kid = _push_worker(acc, slug, _cells(label, m, limit=0), None)
        _register(led, job, acc, kid, {"label": label, "num_tokens": m})
        time.sleep(2)


def cmd_collect() -> None:
    led = load_ledger()
    out_root = ROOT / "outputs" / "conv_abstractor"
    out_root.mkdir(parents=True, exist_ok=True)
    for job, j in led["jobs"].items():
        if not job.startswith("convabs:") or j.get("status") == "done":
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
    cmd = sys.argv[1]
    if cmd == "launch":
        cmd_launch(m4="--m4" in sys.argv)
    else:
        {"smoke": cmd_smoke, "collect": cmd_collect}[cmd]()
