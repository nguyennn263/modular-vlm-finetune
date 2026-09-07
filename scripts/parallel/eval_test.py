#!/usr/bin/env python3
"""Bundle trained bridge checkpoints and launch test-split eval kernels, one
per (bridge, seed), spread across accounts by remaining GPU quota.

    python scripts/parallel/eval_test.py bundle       # upload .pt -> mvlm-test-ckpt
    python scripts/parallel/eval_test.py launch        # push eval kernels
    python scripts/parallel/eval_test.py collect       # pull eval_test.json

Eval is cheap (~0.3-0.6h GPU); this picks the least-loaded accounts with enough
quota so nothing is cut mid-run.
"""
from __future__ import annotations
import json, sys, subprocess, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import (ROOT, ACCT_DIR, _kaggle, _user, _code, _clone_cell, _nb,
                 load_ledger, save_ledger, _register)  # noqa

DS = "mvlm-test-ckpt"
BRANCH_PLAIN = "chore/repo-restructure"

# (local .pt path, bridge, seed, label)  -- plain bridges only (LoRA needs feat branch + big ckpts)
SPECS = [
    ("checkpoints/expA/seed42/multi_token/last_model.pt",       "multi_token", 42,   "mt-s42"),
    ("checkpoints/expA/seed123/multi_token/last_model.pt",      "multi_token", 123,  "mt-s123"),
    ("checkpoints/expA/seed2026/multi_token/last_model.pt",     "multi_token", 2026, "mt-s2026"),
    ("checkpoints/expA/seed3407/multi_token/last_model.pt",     "multi_token", 3407, "mt-s3407"),
    ("checkpoints/expA/seed42/qformer/last_model.pt",           "qformer",     42,   "qf-s42"),
    ("checkpoints/expA/seed42/mini_qformer/last_model.pt",      "mini_qformer",42,   "mq-s42"),
    ("checkpoints/expA/seed42/residual/last_model.pt",          "residual",    42,   "res-s42"),
    ("checkpoints/expA/seed42/tile_attention/last_model.pt",    "tile_attention",42, "ta-s42"),
]


def _quota_h(acc: str) -> float:
    out = _kaggle(acc, "quota", check=False)
    for ln in out.splitlines():
        if ln.strip().startswith("GPU"):
            try:
                return float(ln.split()[2].rstrip("h"))
            except Exception:
                return 0.0
    return 0.0


def cmd_bundle():
    d = ROOT / "outputs" / "parallel" / "test_bundle"
    if d.exists():
        import shutil; shutil.rmtree(d)
    d.mkdir(parents=True)
    got = []
    for pt, bridge, seed, label in SPECS:
        src = ROOT / pt
        if not src.exists() or src.stat().st_size == 0:
            alt = src.parent / "best_model.pt"
            if alt.exists() and alt.stat().st_size > 0:
                print(f"[note] {label}: {src.name} empty/missing -> using best_model.pt")
                src = alt
            else:
                print(f"[skip] missing/empty {pt}"); continue
        sub = d / label
        sub.mkdir()
        (sub / "model.pt").write_bytes(src.read_bytes())
        got.append(f"{label} ({src.stat().st_size//1024//1024}MB)")
    user = _user("acc1")
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"id": f"{user}/{DS}", "title": DS, "licenses": [{"name": "unknown"}]}))
    try:
        _kaggle("acc1", "datasets", "create", "-p", str(d), "--dir-mode", "zip", "--public")
    except RuntimeError:
        _kaggle("acc1", "datasets", "version", "-p", str(d), "-m", "test-ckpts", "--dir-mode", "zip")
    print(f"[bundle] {got} -> {user}/{DS} (wait ~1-2 min for Kaggle to process)")


def _eval_cells(bridge: str, seed: int, label: str) -> list[dict]:
    ds = DS
    return [
        _clone_cell(BRANCH_PLAIN),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!python scripts/phase0_build_data.py 2>&1 | tail -6"),
        _code("import glob, os, shutil",
              f"src = [p for p in glob.glob('/kaggle/input/**/{label}/model.pt', recursive=True)]",
              "assert src, 'ckpt not found: '+repr(os.listdir('/kaggle/input'))",
              f"os.makedirs('/kaggle/working/ck/{bridge}', exist_ok=True)",
              f"shutil.copy(src[0], '/kaggle/working/ck/{bridge}/model.pt')",
              "print('ckpt ready')"),
        _code(f"!python -m src.cli.evaluate --bridge {bridge} --split-dir data/splits "
              f"--split test --seed {seed} "
              f"--checkpoint /kaggle/working/ck/{bridge}/model.pt "
              f"--output /kaggle/working/eval_test.json"),
        _code("!mkdir -p /kaggle/working/out && cp /kaggle/working/eval_test.json /kaggle/working/out/ && "
              "cp /kaggle/working/ck/*/results/*.json /kaggle/working/out/ 2>/dev/null; "
              "import json; print(json.dumps(json.load(open('/kaggle/working/eval_test.json')), indent=2)[:800])"),
    ]


def _push(acc: str, slug: str, cells: list[dict]) -> str:
    user = _user(acc)
    kid = f"{user}/{slug}"
    wd = ROOT / "outputs" / "parallel" / "workers" / slug
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "worker.ipynb").write_text(json.dumps(_nb(cells)))
    (wd / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": slug[:50], "code_file": "worker.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ["nguynrichard/auto-vqabest", f"{_user('acc1')}/{DS}"],
    }, indent=2))
    _kaggle(acc, "kernels", "push", "-p", str(wd))
    return kid


def cmd_launch():
    led = load_ledger()
    accs = sorted((p.name for p in ACCT_DIR.glob("acc*") if (p / "kaggle.json").exists()),
                  key=lambda s: int(s[3:]))
    quotas = {a: _quota_h(a) for a in accs}
    # need ~0.7h headroom per eval; rank accounts by free quota, drop the exhausted
    pool = sorted([a for a in accs if quotas[a] >= 1.5], key=lambda a: -quotas[a])
    print("[launch] quota:", {a: round(quotas[a], 1) for a in accs})
    print("[launch] pool:", pool)
    if len(pool) < len(SPECS):
        print(f"[warn] only {len(pool)} accounts >=1.5h for {len(SPECS)} jobs — some will double up")
    for i, (pt, bridge, seed, label) in enumerate(SPECS):
        job = f"test-eval:{bridge}:s{seed}"
        if led["jobs"].get(job, {}).get("status") == "done":
            print(f"[skip] {job} done"); continue
        acc = pool[i % len(pool)]
        slug = f"mvlm-test-eval-{label}"
        kid = _push(acc, slug, _eval_cells(bridge, seed, label))
        _register(led, job, acc, kid, {"bridge": bridge, "seed": seed, "split": "test"})
        time.sleep(2)


def cmd_collect():
    led = load_ledger()
    out = ROOT / "outputs" / "test_eval"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for job, j in led["jobs"].items():
        if not job.startswith("test-eval:"):
            continue
        acc, kid = j["account"], j["kernel"]
        st = _kaggle(acc, "kernels", "status", kid, check=False)
        if "COMPLETE" not in st:
            print(f"[wait] {job}: {st.strip()[:60]}"); continue
        dst = out / job.replace(":", "_")
        _kaggle(acc, "kernels", "output", kid, "-p", str(dst), check=False)
        f = next(dst.rglob("eval_test.json"), None)
        if f:
            d = json.loads(f.read_text())
            rows.append((job, d))
            j["status"] = "done"
            print(f"[ok] {job}: F1 {d.get('f1', 0)*100:.2f}  CIDEr {d.get('cider', 0)*100:.2f}")
    save_ledger(led)
    (out / "summary.json").write_text(json.dumps([{"job": r[0], **r[1]} for r in rows], indent=2, default=str))


if __name__ == "__main__":
    {"bundle": cmd_bundle, "launch": cmd_launch, "collect": cmd_collect}[sys.argv[1]]()
