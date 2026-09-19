#!/usr/bin/env python3
"""Advisor follow-up ablation 1: num_tokens sweep for the Multi-Token bridge.

Why num_tokens=8 and not 6/10/12? The paper's recipe bridge (Multi-Token) has
always used num_tokens=8 as a shared default across several bridge types (for
comparability), never actually swept for Multi-Token specifically. This sweeps
it, plain bridge (no LoRA), 2 epochs, matching the original Exp A protocol.
num_tokens=8 already has full 4-seed Exp A data (F1 49.55+-0.07) -- not rerun.

Round 1 (seed 42 only, all 4 done): 4/6/10/12 showed a MONOTONIC increase in
F1 with no peak at 8 (48.85/49.38/49.55/50.06/50.27) -- 8 was never optimal.
Round 2 (this update, per user steer after seeing round 1): extend the sweep
upward to 14/16 to look for an actual ceiling, and 3-seed tok10/tok12 (the two
round-1 winners) to confirm the gain isn't sampling noise on a single seed.

    python scripts/parallel/token_sweep.py smoke     # 1 job, --limit 20 sanity check
    python scripts/parallel/token_sweep.py launch     # pushes whatever in SPECS isn't done yet
    python scripts/parallel/token_sweep.py collect
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, _kaggle, _user, _code, _clone_cell, _nb, _push_worker, load_ledger, save_ledger, _register  # noqa

BRANCH = "feat/bridge-design-ablation"
# (label, num_tokens, seed). num_tokens=8 already has 4-seed Exp A data -- not
# rerun here. Round 1 (seed 42 only): tok4/tok6/tok10/tok12, all done.
# Round 2: tok14/tok16 (extend the sweep, still seed 42) + tok10/tok12 at
# seeds 123/3407 (3-seed confirmation of the two round-1 winners).
SPECS = [
    ("tok4", 4, 42), ("tok6", 6, 42), ("tok10", 10, 42), ("tok12", 12, 42),
    ("tok14", 14, 42), ("tok16", 16, 42),
    ("tok10-s123", 10, 123), ("tok10-s3407", 10, 3407),
    ("tok12-s123", 12, 123), ("tok12-s3407", 12, 3407),
    # Round 3 (2026-09-19, quota reset -> 14 accounts idle at 30h): 3-seed
    # confirm tok16 (matches tok10/tok12's treatment) + push the sweep
    # further to 18/20 since 14->16 hadn't shown a plateau yet when this was
    # launched. All run in parallel with everything else -- zero extra
    # wall-clock cost, quota was the only constraint and it just reset.
    ("tok16-s123", 16, 123), ("tok16-s3407", 16, 3407),
    ("tok18", 18, 42), ("tok20", 20, 42),
    ("tok14-s123", 14, 123), ("tok14-s3407", 14, 3407),  # same 3-seed treatment as 10/12/16
]
# 10/14 accounts now in flight as of this round-3 spam-launch; acc10-13 kept
# in reserve for relaunching whatever the confirmed-recurring Kaggle
# slowdown cancels next, rather than opening yet more new sweep points.
# top-quota-remaining accounts as of 2026-09-15 (excludes acc2/acc15, exhausted).
# Round-1 accounts (acc16/14/11/12) are free again -- their jobs finished.
ACCS = ["acc6", "acc10", "acc16", "acc14", "acc12", "acc11", "acc13", "acc7", "acc8", "acc9"]

# Round-2 hit the Kaggle 12h session cap TWICE in a row (0/6 completions across
# both attempts) on acc7/8/9/11/12/13 -- verified via `kaggle quota` per account
# (not assumed): all 6 are now at 0.00h/30h weekly quota (acc7/8/9 even
# slightly over, 31.9-32.9h used). Root cause was quota exhaustion mid-session
# (repeated relaunches on the same accounts burn through the weekly cap without
# yielding a completed run), NOT a training-loop/config slowdown -- that theory
# was checked and disproved (round-1's real per-job training-loop time was a
# flat ~2.42-2.44h regardless of num_tokens). Relaunched 2026-09-16 on 6 fresh
# accounts with real headroom instead: acc6/acc10/acc1/acc16/acc14/acc3.


# Round 3 (2026-09-16/17): relaunched all 6 on brand-new accounts (acc6/10/1/
# 16/14/3, none previously used for token_sweep except acc1/16/14's round-1
# jobs) -- STILL hit the same ~2.2x per-epoch slowdown and got cancelled at
# the 12h cap (confirmed via real training_*.log Time: values pulled from the
# cancelled kernels: tok14 20126.7s/epoch, tok10-s123 21428.2s/epoch, vs.
# round-1's flat 8725-8787s/epoch). Investigated and RULED OUT a code/repo
# regression: `git diff` between round-1's commit and round-3's launch touches
# ONLY token_sweep.py itself (SPECS/collect-logic) + result files, zero
# changes to bridge_modules.py/setup.py/trainer.py/train.py/collator.py/
# setup_kaggle.sh; torch==2.2.2 and transformers==4.38.2 are pinned exact; the
# unpinned deps (accelerate/timm/einops/sentence-transformers) had NO new PyPI
# release in the 09-14->09-16 window, so `pip install --upgrade` resolved to
# the identical version both times. Root cause is genuinely external (Kaggle
# GPU-type assignment or system load), not fixable from this repo -- so
# instead of relaunching a 4th combined train+eval attempt, SPLIT into two
# separate kernel sessions: a train-only session (fits the confirmed slow
# pace: 2 epochs ~11.3h, comfortably under 12h once eval is NOT also packed
# into the same session) that bundles its checkpoint into a public Kaggle
# dataset, then a short eval-only session that pulls that dataset in. This
# works regardless of whether Kaggle's infra ever speeds back up.


def _cells_train(label: str, n: int, seed: int, limit: int) -> list[dict]:
    ck = f"/kaggle/working/toksweep-{label}/seed{seed}"
    limit_arg = f"--limit {limit} " if limit else ""
    return [
        _clone_cell(BRANCH),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        # GPU type is NOT visible in the setup_kaggle.sh tail -5 above --
        # check_gpu() runs at step 4/8, and 4 more steps print after it, so
        # `tail -5` silently drops the "GPU: <name>" line before it ever
        # reaches the kernel log. Print it again here, standalone, so it
        # always survives -- this is how the round-2/3 ~2.2x per-epoch
        # slowdown (confirmed via real training_*.log Time: values, same
        # num_tokens/config as round-1, code/deps ruled out -- see the
        # comment above SPECS) could finally be attributed to a GPU type
        # change (e.g. P100 vs T4x2) if a future run reveals it.
        _code("!python -c \"import torch; print('GPU count:', torch.cuda.device_count()); "
              "print('GPU:', torch.cuda.get_device_name(0))\""),
        _code("!python scripts/phase0_build_data.py 2>&1 | tail -6"),
        _code(f"!python -m src.cli.train --bridge multi_token --bridge-num-tokens {n} "
              f"--split-dir data/splits --seed {seed} --epochs 2 {limit_arg}"
              f"--batch-size 8 --grad-accum 1 --eval-steps 800 --save-steps 800 "
              f"--no-early-stopping --text-metrics-every 2 --text-metrics-max-samples 600 "
              f"--output-dir {ck}"),
        # NOTE: no evaluate.py call here -- that's the whole point of the
        # split. Just bundle the checkpoint for a separate eval session.
        _code(f"!mkdir -p /kaggle/working/out && cp -r {ck} /kaggle/working/out/ && "
              "ls -R /kaggle/working/out | tail -20"),
    ]


def _cells_eval(label: str, n: int, seed: int, ds_id: str, limit: int) -> list[dict]:
    # REAL mount path confirmed via a diagnostic kernel (tok14, 2026-09-17):
    # Kaggle now mounts dataset_sources at /kaggle/input/datasets/<owner>/
    # <slug>/, NOT the classic /kaggle/input/<slug>/ this repo's other
    # scripts assume (e.g. run.py's expa_worker `resume_cp` uses the old
    # path too and would silently no-op via its `2>/dev/null || echo FRESH`
    # fallback -- worth checking if that's ever masked a real resume). 3
    # eval attempts (cross-account, then same-account, then same-account
    # again after a 90s wait) all hit an identical FileNotFoundError before
    # this was found with a minimal `ls -la /kaggle/input/` probe kernel.
    # `evaluate.py` derives BridgeTrainer's output_dir (where it mkdir's a
    # results/ subdir) from Path(checkpoint).parent -- fine when evaluating
    # right after training (checkpoint on writable /kaggle/working/), but
    # /kaggle/input/ is always read-only, so pointing straight at the mounted
    # dataset there hits `OSError: [Errno 30] Read-only file system` (tok14,
    # 2026-09-17, right after the mount-path fix got it past the earlier
    # FileNotFoundError). Copy the checkpoint into /kaggle/working/ first so
    # its parent dir is writable -- avoids touching evaluate.py's shared path.
    limit_arg = f" --limit {min(limit * 10, 500)}" if limit else ""
    return [
        _clone_cell(BRANCH),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!python scripts/phase0_build_data.py 2>&1 | tail -6"),
        _code(f"!mkdir -p /kaggle/working/ckpt_for_eval /kaggle/working/out && "
              f"cp /kaggle/input/datasets/{ds_id}/last_model.pt /kaggle/working/ckpt_for_eval/ && "
              f"python -m src.cli.evaluate --bridge multi_token --bridge-num-tokens {n} "
              f"--split-dir data/splits --split val "
              f"--checkpoint /kaggle/working/ckpt_for_eval/last_model.pt{limit_arg} "
              f"--output /kaggle/working/out/eval_val.json"),
        _code("!ls -la /kaggle/working/out"),
    ]


def _bundle_ckpt(acc: str, label: str, local_pt: Path) -> str:
    """Package one trained checkpoint into a public per-label Kaggle dataset
    (~55-77MB, well within limits) so a separate eval-only kernel can pull it
    in via dataset_sources -- this is the train/eval split's connective step."""
    user = _user(acc)
    ds_id = f"{user}/mvlm-toksweep-ckpt-{label}"
    d = ROOT / "outputs" / "parallel" / "bundle" / f"toksweep-{label}"
    if d.exists():
        import shutil
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / "last_model.pt").write_bytes(local_pt.read_bytes())
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"id": ds_id, "title": f"mvlm-toksweep-ckpt-{label}", "licenses": [{"name": "unknown"}]}))
    try:
        _kaggle(acc, "datasets", "create", "-p", str(d), "--public")
    except RuntimeError:
        _kaggle(acc, "datasets", "version", "-p", str(d), "-m", "update")
    print(f"[bundle] {label} -> dataset {ds_id}")
    return ds_id


def cmd_smoke() -> None:
    label, n, seed = SPECS[0]
    kid = _push_worker(ACCS[0], f"mvlm-toksweep-smoke-{label}", _cells_train(label, n, seed, limit=20), None)
    print(f"[smoke] pushed {kid} -- check the log for num_tokens={n} in the bridge_config and "
          f"a successful training completion.")


def cmd_launch() -> None:
    led = load_ledger()
    for i, (label, n, seed) in enumerate(SPECS):
        job = f"toksweep-eval:{label}"
        if led["jobs"].get(job, {}).get("status") == "done":
            print(f"[skip] {job} done"); continue
        train_job = f"toksweep-train:{label}"
        if led["jobs"].get(train_job):
            print(f"[skip] {train_job} already launched (phase=train)"); continue
        acc = ACCS[i % len(ACCS)]
        slug = f"mvlm-toksweep-{label}"
        kid = _push_worker(acc, slug, _cells_train(label, n, seed, limit=0), None)
        _register(led, train_job, acc, kid, {"label": label, "num_tokens": n, "seed": seed, "phase": "train"})
        time.sleep(2)
    save_ledger(led)


def cmd_collect() -> None:
    led = load_ledger()
    out_root = ROOT / "outputs" / "token_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    # phase 1: train jobs -> on COMPLETE, bundle checkpoint + launch eval job
    for job, j in list(led["jobs"].items()):
        if not job.startswith("toksweep-train:") or j.get("status") == "done":
            continue
        label = j["label"]
        st = _kaggle(j["account"], "kernels", "status", j["kernel"], check=False)
        if "CANCEL_ACKNOWLEDGED" in st or "ERROR" in st:
            print(f"[CANCELLED] {job}: {st.strip()[:80]} -- needs relaunch"); continue
        if "COMPLETE" not in st:
            print(f"[wait] {job}: {st.strip()[:60]}"); continue
        dst = out_root / label / "train"
        _kaggle(j["account"], "kernels", "output", j["kernel"], "-p", str(dst), check=False)
        pt = next(dst.rglob("last_model.pt"), None)
        if not pt:
            print(f"[partial] {job}: training COMPLETE but no last_model.pt in {dst}"); continue
        j["status"] = "done"
        ds_id = _bundle_ckpt(j["account"], label, pt)
        # Kaggle needs a beat to finish processing a just-created/updated
        # dataset before it's mountable via dataset_sources -- launching the
        # eval kernel immediately after upload once hit a real
        # FileNotFoundError on /kaggle/input/<slug>/last_model.pt (tok14,
        # 2026-09-17) even though `kaggle datasets files` showed it uploaded.
        time.sleep(90)
        eval_job = f"toksweep-eval:{label}"
        # IMPORTANT: eval MUST run on the same account that owns the bundled
        # dataset. tok14 (handled manually, off this automated path) hit the
        # identical FileNotFoundError even after the 90s wait when the
        # dataset (acc1) and the eval kernel (acc3) were different accounts
        # -- cross-account dataset_sources mounting a *freshly created*
        # public dataset appears unreliable even though `datasets files`
        # already lists it. This automated path avoids that by construction
        # (same `acc` bundles and evals), so never split those.
        acc = j["account"]  # same account already has the dataset locally-owned
        kid = _push_worker(acc, f"mvlm-toksweep-eval-{label}", _cells_eval(label, j["num_tokens"], j["seed"], ds_id, 0), ds_id)
        _register(led, eval_job, acc, kid, {"label": label, "num_tokens": j["num_tokens"], "seed": j.get("seed"), "phase": "eval"})
        print(f"[ok] {job} trained -> bundled -> launched {eval_job} ({kid})")

    # phase 2: eval jobs -> on COMPLETE, read eval_val.json
    for job, j in led["jobs"].items():
        if not job.startswith("toksweep-eval:") or j.get("status") == "done":
            continue
        st = _kaggle(j["account"], "kernels", "status", j["kernel"], check=False)
        if "CANCEL_ACKNOWLEDGED" in st or "ERROR" in st:
            print(f"[CANCELLED] {job}: {st.strip()[:80]} -- needs relaunch"); continue
        if "COMPLETE" not in st:
            print(f"[wait] {job}: {st.strip()[:60]}"); continue
        dst = out_root / j["label"] / "eval"
        _kaggle(j["account"], "kernels", "output", j["kernel"], "-p", str(dst), check=False)
        f = next(dst.rglob("eval_val.json"), None)
        if f:
            d = json.loads(f.read_text())
            j["status"] = "done"; j["collected"] = True
            print(f"[ok] {job} (num_tokens={j['num_tokens']}, seed={j.get('seed', '?')}): "
                  f"F1 {d.get('f1', 0)*100:.2f}  CIDEr {d.get('cider', 0)*100:.2f}  loss {d.get('loss', 0):.3f}")
        else:
            print(f"[partial] {job}: no eval_val.json in {dst}")
    save_ledger(led)


if __name__ == "__main__":
    {"smoke": cmd_smoke, "launch": cmd_launch, "collect": cmd_collect}[sys.argv[1]]()
