#!/usr/bin/env python3
"""Drive Exp A / oracle sweep across the 5 Kaggle accounts.

    python scripts/parallel/run.py launch expa            # 5 bridges, seed 42, one per account
    python scripts/parallel/run.py poll                   # collect finished, save to repo
    python scripts/parallel/run.py status
    python scripts/parallel/run.py resume expa:qformer:s42 # relaunch with the partial checkpoint

State: outputs/parallel/ledger.json  (idempotent — safe to re-run after a crash).
Account configs: ~/.kaggle-accounts/acc{1..5}/kaggle.json
Kernel outputs land in checkpoints/expA/... and outputs/expA/... in this repo.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "outputs" / "parallel" / "ledger.json"
ACCT_DIR = Path.home() / ".kaggle-accounts"
# every ~/.kaggle-accounts/accN/ holding a kaggle.json, natural-sorted
ACCOUNTS = sorted((p.name for p in ACCT_DIR.glob("acc*") if (p / "kaggle.json").exists()),
                  key=lambda s: int(s[3:]) if s[3:].isdigit() else 0) or [f"acc{i}" for i in range(1, 6)]
REPO_URL = "https://github.com/nguyennn263/modular-vlm-finetune.git"
BRIDGES = ["residual", "multi_token", "tile_attention", "mini_qformer", "qformer"]


# ------------------------------------------------------------------ kaggle CLI
def _kaggle(acc: str, *args: str, check: bool = True, capture: bool = True) -> str:
    env = {**os.environ, "KAGGLE_CONFIG_DIR": str(ACCT_DIR / acc)}
    r = subprocess.run(["kaggle", *args], env=env, text=True,
                       capture_output=capture, check=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} [{acc}] -> {r.returncode}\n{r.stdout}\n{r.stderr}")
    return (r.stdout or "") + (r.stderr or "")


def _user(acc: str) -> str:
    return json.loads((ACCT_DIR / acc / "kaggle.json").read_text())["username"]


# ------------------------------------------------------------------ ledger
def load_ledger() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {"jobs": {}}


def save_ledger(led: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=2))


# ------------------------------------------------------------------ worker notebook
def _nb(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3",
            "display_name": "Python 3", "language": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def _code(*lines: str) -> dict:
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def _clone_cell(branch: str) -> dict:
    """Robust clone to /tmp (keeps it OUT of /kaggle/working so the kernel output
    stays small -- otherwise every `kernels output` pull drags the ~200MB repo)."""
    return _code(
        "import os, subprocess, time",
        "os.makedirs('/tmp/wk', exist_ok=True); os.chdir('/tmp/wk')",
        "for _ in range(6):",
        f"    subprocess.call('git clone -q {REPO_URL} repo || (cd repo && git fetch -q)', shell=True)",
        "    if os.path.isdir('repo'): break",
        "    time.sleep(15)",
        "os.chdir('/tmp/wk/repo')",
        f"os.system('git checkout -q {branch} && git pull -q')",
    )


def expa_worker(bridge: str, seed: int, branch: str, resume_ds: str | None, epochs: int,
                tile_choices: str | None = None, answer_sampling: str | None = None,
                align: str | None = None) -> list[dict]:
    _sub = ("ck-tiled" if tile_choices
            else "ck-align-" + align if align
            else "ck-" + answer_sampling if answer_sampling else "ck")
    ck = f"/kaggle/working/{_sub}/seed{seed}"
    resume_cp = (f"!mkdir -p {ck}/{bridge} && cp /kaggle/input/{resume_ds.split('/')[-1]}/* "
                 f"{ck}/{bridge}/ 2>/dev/null && echo RESUMED || echo FRESH") if resume_ds else "print('FRESH')"
    tc = f"--tile-choices {tile_choices} " if tile_choices else ""
    asamp = f"--answer-sampling {answer_sampling} " if answer_sampling else ""
    algn = f"--align-distill --align-type {align} " if align else ""
    # align logit adds a full teacher Qwen2 forward (256 vision + text tokens) ->
    # OOMs the 16GB P100 at bs 8. Halve the micro-batch, keep effective batch 8.
    bs, ga = (4, 2) if align == "logit" else (8, 1)
    # tile augmentation makes each step ~3x slower (avg InternViT tiles). Fewer
    # mid-training val passes + a checkpoint every ~half epoch keeps the whole
    # kernel well under the 12h Kaggle cap (a CANCEL there persists nothing).
    step = 3000 if tile_choices else 800
    # tile-augmented runs skip the ~2h final full-val eval (the oracle scores per
    # n_tiles itself) so training-only fits the 12h cap.
    metrics = "--text-metrics-every 99 --text-metrics-max-samples 300" if tile_choices \
              else "--text-metrics-every 2 --text-metrics-max-samples 600"
    cells = [
        _clone_cell(branch),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!python scripts/phase0_build_data.py 2>&1 | tail -6"),
        _code(resume_cp),
        _code(f"!python -m src.cli.train --bridge {bridge} --split-dir data/splits --seed {seed} "
              f"--epochs {epochs} --batch-size {bs} --grad-accum {ga} --eval-steps {step} --save-steps {step} "
              f"--no-early-stopping {metrics} "
              f"{tc}{asamp}{algn}--output-dir {ck} --resume"),
    ]
    if not tile_choices:
        cells.append(_code(f"!python -m src.cli.evaluate --bridge {bridge} --split-dir data/splits --split val "
                           f"--checkpoint {ck}/{bridge}/last_model.pt"))
    cells.append(_code(f"!mkdir -p /kaggle/working/out && cp -r {ck} /kaggle/working/out/ && "
                       f"cp -r data/splits /kaggle/working/out/ 2>/dev/null; ls -R /kaggle/working/out | tail -20"))
    return cells


def oracle_worker(shard: int, nshards: int, bridges: str, branch: str,
                  ckpt_ds: str, split: str, subset: int, out: str) -> list[dict]:
    ds = ckpt_ds.split("/")[-1]
    ckdir = "checkpoints/expA-tiled/seed42"
    return [
        _clone_cell(branch),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!python scripts/phase0_build_data.py 2>&1 | tail -4"),
        _code(f"import os, glob, shutil",
              f"os.system('find /kaggle/input -maxdepth 5 -name best_model.pt')",
              f"BR = ('multi_token','qformer','mini_qformer','residual','tile_attention')",
              f"pts = [p for p in glob.glob('/kaggle/input/**/best_model.pt', recursive=True) if os.path.basename(os.path.dirname(p)) in BR]",
              f"assert pts, 'no <bridge>/best_model.pt under /kaggle/input -- '+repr(os.listdir('/kaggle/input'))",
              f"os.makedirs('{ckdir}', exist_ok=True)",
              f"[ (os.makedirs('{ckdir}/'+os.path.basename(os.path.dirname(p)), exist_ok=True), shutil.copy(p, '{ckdir}/'+os.path.basename(os.path.dirname(p))+'/best_model.pt')) for p in pts ]",
              f"print('checkpoints ready:', glob.glob('{ckdir}/*/best_model.pt'))"),
        _code(f"!python -m src.cli.oracle --bridges {bridges} --n-tiles 1,3,6 "
              f"--split {split} --subset {subset} --shard {shard}/{nshards} "
              f"--ckpt-dir {ckdir} --out {out}"),
        _code(f"!mkdir -p /kaggle/working/out && cp {out}/*.parquet /kaggle/working/out/ && ls -la /kaggle/working/out"),
    ]


def fiq_worker(branch: str, splits: str) -> list[dict]:
    return [
        _clone_cell(branch),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!python scripts/phase0_build_data.py 2>&1 | tail -4"),
        _code(f"!python -m src.cli.build_fiq --split-dir data/splits --splits {splits} --pca 64"),
        _code("!python -m src.cli.train_router --split-dir data/splits --epochs 3 "
              "--predict-splits train,val,test"),
        _code("!mkdir -p /kaggle/working/out && cp -r outputs/fiq outputs/router checkpoints/router "
              "/kaggle/working/out/ && ls -R /kaggle/working/out | tail -20"),
    ]


def cmd_bundle(args) -> None:
    """Package trained bridge checkpoints into a public Kaggle dataset for the oracle workers."""
    seed = args.seed
    bridges = args.bridges.split(",") if args.bridges else BRIDGES
    exp = "expA-tiled" if args.tiled else "expA"
    d = ROOT / "outputs" / "parallel" / "bundle"
    if d.exists():
        import shutil
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    got = []
    for b in bridges:
        bdir = ROOT / "checkpoints" / exp / f"seed{seed}" / b
        src = bdir / "last_model.pt" if (bdir / "last_model.pt").exists() else bdir / "best_model.pt"
        if src.exists():
            (d / b).mkdir(exist_ok=True)
            (d / b / "best_model.pt").write_bytes(src.read_bytes())   # oracle expects <bridge>/best_model.pt
            got.append(b)
    if not got:
        raise SystemExit(f"no checkpoints in checkpoints/{exp}/seed{seed} — run `poll` first")
    user = _user("acc1")
    ds_id = f"{user}/mvlm-expa-ckpt"
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"id": ds_id, "title": "mvlm-expa-ckpt", "licenses": [{"name": "unknown"}]}))
    try:
        _kaggle("acc1", "datasets", "create", "-p", str(d), "--public")
    except RuntimeError:
        _kaggle("acc1", "datasets", "version", "-p", str(d), "-m", "update")
    print(f"[bundle] {got} -> dataset {ds_id}  (wait ~1 min for Kaggle to process)")


def _push_worker(acc: str, slug: str, cells: list[dict], resume_ds: str | None) -> str:
    user = _user(acc)
    kid = f"{user}/{slug}"
    d = ROOT / "outputs" / "parallel" / "workers" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "worker.ipynb").write_text(json.dumps(_nb(cells)))
    meta = {
        "id": kid, "title": slug[:50], "code_file": "worker.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ["nguynrichard/auto-vqabest"] + ([resume_ds] if resume_ds else []),
        "competition_sources": [], "kernel_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    _kaggle(acc, "kernels", "push", "-p", str(d))
    return kid


# ------------------------------------------------------------------ commands
def _register(led, job, acc, kid, extra):
    led["jobs"][job] = {"account": acc, "kernel": kid, "status": "running",
                        "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "collected": False, **extra}
    print(f"[launch] {job} -> {acc} ({kid})")
    save_ledger(led)


def cmd_launch(args) -> None:
    led = load_ledger()
    branch = _current_branch()
    pool = [a.strip() for a in args.accounts.split(",")] if args.accounts else ACCOUNTS
    print(f"[launch] account pool: {pool}")

    if args.phase == "expa":
        seeds = [int(s) for s in str(args.seed).split(",")]
        blist = [b.strip() for b in args.bridges.split(",")] if args.bridges != ",".join(BRIDGES) else BRIDGES
        combos = [(b, s) for s in seeds for b in blist]
        tag = ("-tiled" if args.tiles
               else "-align-" + args.align if args.align
               else "-" + args.answer_sampling if args.answer_sampling else "")
        for i, (bridge, seed) in enumerate(combos):
            acc = pool[i % len(pool)]
            job = f"expa{tag}:{bridge}:s{seed}"
            if led["jobs"].get(job, {}).get("status") not in (None, "failed", "error", "incomplete"):
                print(f"[skip] {job} = {led['jobs'][job]['status']}")
                continue
            slug = f"mvlm-expa{tag}-{bridge.replace('_','-')}-s{seed}"
            cells = expa_worker(bridge, seed, branch, None, args.epochs, tile_choices=args.tiles or None,
                                answer_sampling=args.answer_sampling, align=args.align)
            kid = _push_worker(acc, slug, cells, None)
            _register(led, job, acc, kid, {"bridge": bridge, "seed": seed, "tiles": args.tiles,
                                           "answer_sampling": args.answer_sampling, "align": args.align})

    elif args.phase == "oracle":
        ds = args.ckpt_ds or f"{_user(args.bundle_acc)}/mvlm-expa-ckpt"
        n = args.shards
        tag = f"-{args.tag}" if args.tag else ""
        out = f"outputs/oracle_{args.split}{('_' + args.tag) if args.tag else ''}"
        for i in range(n):
            acc = pool[i % len(pool)]
            job = f"oracle:{args.split}{tag}:shard{i}of{n}"
            if led["jobs"].get(job, {}).get("status") not in (None, "failed", "error", "incomplete"):
                print(f"[skip] {job} = {led['jobs'][job]['status']}")
                continue
            slug = f"mvlm-oracle-{args.split}{tag}-{i}of{n}"
            cells = oracle_worker(i, n, args.bridges, branch, ds, args.split,
                                  args.subset, out)
            kid = _push_worker(acc, slug, cells, ds)
            _register(led, job, acc, kid, {"split": args.split, "shard": f"{i}/{n}", "tag": args.tag})

    elif args.phase == "fiq":
        acc = args.account or "acc1"
        job = "fiq:all"
        kid = _push_worker(acc, "mvlm-fiq", fiq_worker(branch, "train,val,test"), None)
        _register(led, job, acc, kid, {})

    else:
        raise SystemExit(f"unknown phase {args.phase!r}")


def cmd_poll(args) -> None:
    led = load_ledger()
    for job, j in led["jobs"].items():
        if j.get("collected") or j.get("status") in ("failed",):
            continue
        out = _kaggle(j["account"], "kernels", "status", j["kernel"], check=False)
        st = next((s for s in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED")
                   if s in out.upper()), "?")
        j["status"] = {"COMPLETE": "complete", "ERROR": "error", "CANCEL": "cancelled",
                       "RUNNING": "running", "QUEUED": "queued"}.get(st, "unknown")
        print(f"[{job}] {j['status']}")
        if j["status"] in ("complete", "error"):
            _collect(job, j)
        save_ledger(led)
    save_ledger(led)


def _copytree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.rglob("*"):
        if f.is_file():
            rel = f.relative_to(src)
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            (dst / rel).write_bytes(f.read_bytes())


def _collect(job: str, j: dict) -> None:
    dst = ROOT / "outputs" / "parallel" / "pulled" / job.replace(":", "_").replace("/", "-")
    dst.mkdir(parents=True, exist_ok=True)
    _kaggle(j["account"], "kernels", "output", j["kernel"], "-p", str(dst), check=False)
    ok = False

    if job.startswith("expa"):
        seed, bridge = j["seed"], j["bridge"]
        exp = ("expA-tiled" if j.get("tiles")
               else f"expA-align-{j['align']}" if j.get("align")
               else f"expA-{j['answer_sampling']}" if j.get("answer_sampling") else "expA")
        # Only the worker's own output tree (`out/seed<S>/<bridge>/` or `ck/...`);
        # NEVER files that rode along inside the cloned `repo/` checkout.
        def _own(name: str):
            pref = [p for p in dst.rglob(name)
                    if f"/seed{seed}/{bridge}/" in str(p) and "/repo/" not in str(p)]
            pref.sort(key=lambda p: (0 if f"{os.sep}out{os.sep}" in str(p) else 1))
            return pref

        cdir = ROOT / "checkpoints" / exp / f"seed{seed}" / bridge
        srcs = _own("last_model.pt") or _own("best_model.pt")
        if srcs:
            cdir.mkdir(parents=True, exist_ok=True)
            for f in srcs[0].parent.iterdir():
                if f.is_file():
                    (cdir / f.name).write_bytes(f.read_bytes())
            rdir = srcs[0].parent / "results"
            if rdir.is_dir():
                _copytree(rdir, cdir / "results")
        for sm in _own("eval_val_samples.jsonl")[:1]:
            t = ROOT / "outputs" / exp / f"seed{seed}" / bridge
            t.mkdir(parents=True, exist_ok=True)
            (t / "eval_val_samples.jsonl").write_bytes(sm.read_bytes())
        for sj in _own("summary.json")[:1]:
            try:
                s = json.loads(sj.read_text())
                j["epochs_trained"], j["best_val_loss"] = s.get("epochs_trained"), s.get("best_val_loss")
            except Exception:
                pass
        ok = (cdir / "last_model.pt").exists() or (cdir / "best_model.pt").exists()

    elif job.startswith("oracle:"):
        tagpart = f"_{j['tag']}" if j.get("tag") else ""
        t = ROOT / "outputs" / f"oracle_{j['split']}{tagpart}"; t.mkdir(parents=True, exist_ok=True)
        for p in dst.rglob("table.shard*.parquet"):
            (t / p.name).write_bytes(p.read_bytes()); ok = True

    elif job.startswith("fiq:"):
        for sub in ("fiq", "router"):
            src = next((p for p in dst.rglob(sub) if p.is_dir()), None)
            if src:
                _copytree(src, ROOT / "outputs" / sub); ok = True
        src = next((p for p in dst.rglob("router") if p.is_dir() and (p / "best.pt").exists()), None)
        if src:
            _copytree(src, ROOT / "checkpoints" / "router")

    j["collected"] = bool(ok)
    j["status"] = "done" if ok else ("error" if j["status"] == "error" else "incomplete")
    print(f"   -> {j['status']}  (artifacts: {'yes' if ok else 'NO'})")


def cmd_status(args) -> None:
    led = load_ledger()
    if not led["jobs"]:
        print("(no jobs)")
        return
    print(f"{'job':32} {'acc':5} {'status':11} {'ep':>3} {'val_loss':>9}  kernel")
    for job, j in led["jobs"].items():
        print(f"{job:32} {j['account']:5} {j['status']:11} "
              f"{str(j.get('epochs_trained','-')):>3} {str(j.get('best_val_loss','-'))[:9]:>9}  {j['kernel']}")


def cmd_resume(args) -> None:
    led = load_ledger()
    job = args.job
    j = led["jobs"][job]
    partial = ROOT / "checkpoints" / "expA" / f"seed{j['seed']}" / j["bridge"]
    if not any(partial.glob("*.pt")):
        raise SystemExit(f"no partial checkpoint at {partial} — nothing to resume from")
    # publish partial as a public dataset under the same account
    ds_dir = ROOT / "outputs" / "parallel" / "resume" / job.replace(":", "_")
    ds_dir.mkdir(parents=True, exist_ok=True)
    for f in partial.glob("*.pt"):
        (ds_dir / f.name).write_bytes(f.read_bytes())
    user = _user(j["account"])
    ds_id = f"{user}/mvlm-resume-{job.split(':',1)[1].replace(':','-').replace('_','-')}"
    (ds_dir / "dataset-metadata.json").write_text(json.dumps(
        {"id": ds_id, "title": ds_id.split('/')[-1], "licenses": [{"name": "unknown"}]}))
    try:
        _kaggle(j["account"], "datasets", "create", "-p", str(ds_dir), "--public")
    except RuntimeError:
        _kaggle(j["account"], "datasets", "version", "-p", str(ds_dir), "-m", "resume", "--dir-mode", "zip")
    time.sleep(20)
    slug = f"mvlm-expa-{j['bridge'].replace('_','-')}-s{j['seed']}"
    cells = expa_worker(j["bridge"], j["seed"], _current_branch(), ds_id, args.epochs)
    kid = _push_worker(j["account"], slug, cells, ds_id)
    j.update(status="running", collected=False, resumed_from=ds_id,
             pushed_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    save_ledger(led)
    print(f"[resume] {job} -> {kid} (resume dataset {ds_id})")


def _current_branch() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "branch", "--show-current"],
                          text=True, capture_output=True).stdout.strip() or "chore/repo-restructure"


def main() -> None:
    p = argparse.ArgumentParser(prog="run.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("launch")
    lp.add_argument("phase", choices=["expa", "oracle", "fiq"])
    lp.add_argument("--seed", default="42", help="expa: comma list e.g. 43,44")
    lp.add_argument("--epochs", type=int, default=2,
                    help="DEFAULT 2 — CIDEr plateaus by epoch 2 (ep2 1.025 -> ep4 1.029) and 2 "
                         "epochs stays safely under the 12h Kaggle cap for every config incl. "
                         "tile-augmented + align (both ~2x slower/step). A CANCEL at the cap "
                         "persists nothing beyond the last saved epoch checkpoint. Use 1 for a "
                         "fast probe of a new experiment type.")
    lp.add_argument("--shards", type=int, default=5)
    lp.add_argument("--split", default="train", help="oracle: train|val|test")
    lp.add_argument("--subset", type=int, default=7500)
    lp.add_argument("--bridges", default=",".join(BRIDGES),
                    help="expa/oracle: comma list of bridges (expa default = all 5)")
    lp.add_argument("--tiles", default=None,
                    help="expa: '1,3,6' -> tile-count-augmented retrain (checkpoints/expA-tiled/)")
    lp.add_argument("--answer-sampling", default=None, dest="answer_sampling",
                    choices=["random", "majority"],
                    help="expa: train target picks among all 5 refs instead of ref[0]")
    lp.add_argument("--align", default=None, choices=["logit", "feat"],
                    help="expa: KD the bridge toward Vintern's mlp1 projector")
    lp.add_argument("--accounts", default=None,
                    help="comma list e.g. acc6,acc7 — restrict the account pool (default: all)")
    lp.add_argument("--ckpt-ds", default=None, dest="ckpt_ds",
                    help="oracle: override the checkpoint dataset id (user/slug)")
    lp.add_argument("--tag", default=None,
                    help="oracle: disambiguate job-key/slug/out-dir from a prior sweep on the "
                         "same split+shards but a DIFFERENT checkpoint dataset (e.g. 'tiled' -> "
                         "job oracle:val-tiled:..., out outputs/oracle_val_tiled/). Needed because "
                         "job identity is otherwise just phase:split:shard, which collides across "
                         "checkpoint generations.")
    lp.add_argument("--bundle-acc", default="acc1", dest="bundle_acc",
                    help="oracle: account whose user owns mvlm-expa-ckpt (default acc1)")
    lp.add_argument("--account", default=None)
    lp.set_defaults(fn=cmd_launch)

    sub.add_parser("poll").set_defaults(fn=cmd_poll)
    sub.add_parser("status").set_defaults(fn=cmd_status)

    bp = sub.add_parser("bundle"); bp.add_argument("--seed", type=int, default=42)
    bp.add_argument("--bridges", default=""); bp.add_argument("--tiled", action="store_true",
                    help="bundle from checkpoints/expA-tiled/ (the tile-augmented retrain)")
    bp.set_defaults(fn=cmd_bundle)

    rp = sub.add_parser("resume"); rp.add_argument("job"); rp.add_argument("--epochs", type=int, default=5)
    rp.set_defaults(fn=cmd_resume)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
