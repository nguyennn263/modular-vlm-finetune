#!/usr/bin/env python3
"""Full-image re-evaluation of the Multi-Token num_tokens sweep (g = 4, 6, 10..20).

The sweep (scripts/parallel/token_sweep.py, branch feat/bridge-design-ablation, Sept 2026) was
evaluated with the old first-tile generation input (--gen-image first_tile). The g=8 runs were
re-evaluated on the full image by input_diag.py; this does the same for g = 10..20 with the
identical command (eval-only, --n-tiles 1 --pooled-input default --gen-image full). Checkpoints:
the public per-run datasets mvlm-toksweep-ckpt-<label> (last_model.pt), except tok4 / tok6 / tok10 / tok12 at
seed 42, which live only in their training kernels' output (mounted through kernel_sources, so
those two run on the owning account). tok20-s3407 is unreachable (its account is gone).
MultiTokenMLP is identical on both branches; at one tile the pooled input is the CLS token, as in
training.

    python scripts/parallel/tok_full.py launch [tag ...]
    python scripts/parallel/tok_full.py collect
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, _kaggle, _user, _code, _clone_cell, _nb, load_ledger, save_ledger, _register  # noqa

BRANCH = "exp/eval-input-diagnostic"
DOCKER_IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
                "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461")

# tag -> (num_tokens, account, dataset_source or None, kernel_source or None, splits)
JOBS = {
    "g4-s42":    (4, "acc16", None, "nguynrichard/mvlm-toksweep-tok4", ["val"]),
    "g4-s123":   (4, "acc17", "nguyennn263/mvlm-toksweep-ckpt-tok4-s123", None, ["val"]),
    "g4-s3407":  (4, "acc17", "nguyennn251/mvlm-toksweep-ckpt-tok4-s3407", None, ["val"]),
    "g6-s42":    (6, "acc14", None, "nguyendinhdoanh/mvlm-toksweep-tok6", ["val"]),
    "g6-s123":   (6, "acc13", "kffddk/mvlm-toksweep-ckpt-tok6-s123", None, ["val"]),
    "g6-s3407":  (6, "acc13", "avanrager/mvlm-toksweep-ckpt-tok6-s3407", None, ["val"]),
    "g10-s42":   (10, "acc11", None, "chalicetrncthnh/mvlm-toksweep-tok10", ["val"]),
    "g10-s123":  (10, "acc3", "nguyennn2603/mvlm-toksweep-ckpt-tok10-s123", None, ["val"]),
    "g10-s3407": (10, "acc3", "nguyennn251/mvlm-toksweep-ckpt-tok10-s3407", None, ["val"]),
    "g12-s42":   (12, "acc12", None, "kffddk/mvlm-toksweep-tok12", ["val"]),
    "g12-s123":  (12, "acc12", "nguyendinhdoanh/mvlm-toksweep-ckpt-tok12-s123", None, ["val"]),
    "g12-s3407": (12, "acc15", "nguynrichard/mvlm-toksweep-ckpt-tok12-s3407", None, ["val"]),
    "g14-s42":   (14, "acc1", "nguyennn263/mvlm-toksweep-ckpt-tok14", None, ["val", "test"]),
    "g14-s123":  (14, "acc1", "lelynhan/mvlm-toksweep-ckpt-tok14-s123", None, ["val", "test"]),
    "g14-s3407": (14, "acc16", "nguynrichard/mvlm-toksweep-ckpt-tok14-s3407", None, ["val", "test"]),
    "g16-s42":   (16, "acc2", "nguyennn263/mvlm-toksweep-ckpt-tok16", None, ["val"]),
    "g16-s123":  (16, "acc2", "nguyennn2501/mvlm-toksweep-ckpt-tok16-s123", None, ["val"]),
    "g16-s3407": (16, "acc5", "nguyennguyen263/mvlm-toksweep-ckpt-tok16-s3407", None, ["val"]),
    "g18-s42":   (18, "acc11", "chalicetrncthnh/mvlm-toksweep-ckpt-tok18", None, ["val"]),
    "g18-s123":  (18, "acc5", "nguyennguyen263/mvlm-toksweep-ckpt-tok18-s123", None, ["val"]),
    "g18-s3407": (18, "acc10", "giapht/mvlm-toksweep-ckpt-tok18-s3407", None, ["val"]),
    "g20-s42":   (20, "acc10", "tinquc220300nguyn/mvlm-toksweep-ckpt-tok20", None, ["val"]),
    "g20-s123":  (20, "acc7", "tinquc220300nguyn/mvlm-toksweep-ckpt-tok20-s123", None, ["val"]),
}
# 2026-10-12: six kernels were still RUNNING after ~9.5h (an eval takes ~1h); relaunch under new
# kernel names on accounts with free slots, leaving the old kernels alone (first to finish wins)
RETRY = {
    "g10-s123-r2": (10, "acc18", "nguyennn2603/mvlm-toksweep-ckpt-tok10-s123", None, ["val"]),
    "g10-s3407-r2": (10, "acc18", "nguyennn251/mvlm-toksweep-ckpt-tok10-s3407", None, ["val"]),
    "g12-s123-r2": (12, "acc19", "nguyendinhdoanh/mvlm-toksweep-ckpt-tok12-s123", None, ["val"]),
    "g14-s123-r2": (14, "acc19", "lelynhan/mvlm-toksweep-ckpt-tok14-s123", None, ["val", "test"]),
    "g6-s42-r2": (6, "acc14", None, "nguyendinhdoanh/mvlm-toksweep-tok6", ["val"]),
}
if "--retry" in sys.argv:
    JOBS = RETRY
CK = "/tmp/ck/multi_token"


def _cells(tag: str, n: int, splits: list[str]) -> list[dict]:
    cells = [
        _clone_cell(BRANCH),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!python scripts/phase0_build_data.py 2>&1 | tail -6"),
        # the run's last_model.pt: one file in a dataset; training-kernel outputs hold it twice
        # (two directories) -- use it only if every copy is byte-identical. Copied to /tmp so
        # the eval's results/ dir is writable and the .pt stays out of the output
        _code("import glob, hashlib, os, shutil",
              "src = sorted(glob.glob('/kaggle/input/**/last_model.pt', recursive=True))",
              "md5 = {p: hashlib.md5(open(p, 'rb').read()).hexdigest() for p in src}",
              "print(md5)",
              "assert src and len(set(md5.values())) == 1, 'missing or differing last_model.pt: ' + repr(md5)",
              f"os.makedirs('{CK}', exist_ok=True)",
              f"shutil.copy(src[0], '{CK}/model.pt')",
              f"print('ckpt ready', src[0], os.path.getsize('{CK}/model.pt'))"),
    ]
    for split in splits:
        cells += [
            _code(f"!python -m src.cli.evaluate --bridge multi_token --bridge-num-tokens {n} "
                  f"--split-dir data/splits --split {split} --n-tiles 1 --pooled-input default "
                  f"--gen-image full --checkpoint {CK}/model.pt --output {CK}/eval_{split}.json "
                  f"|| echo failed > /tmp/cell_failed",
                  "import os",
                  "assert not os.path.exists('/tmp/cell_failed'), 'evaluate failed'"),
            _code("import glob, os, shutil",
                  f"out = '/kaggle/working/out/{tag}/{split}'",
                  "os.makedirs(out, exist_ok=True)",
                  f"shutil.copy('{CK}/eval_{split}.json', out)",
                  f"for f in glob.glob('{CK}/results/text_predictions*.json'):",
                  "    shutil.copy(f, out)",
                  f"print(open(out + '/eval_{split}.json').read()[:900])"),
        ]
    return cells


def cmd_launch(only: list[str] | None) -> None:
    led = load_ledger()
    for tag, (n, acc, ds, kern, splits) in JOBS.items():
        job = f"tokfull:{tag}"
        if only and tag not in only:
            continue
        if job in led["jobs"]:
            print(f"[skip] {job} already pushed ({led['jobs'][job]['status']})"); continue
        slug = f"mvlm-tokfull-{tag}"
        kid = f"{_user(acc)}/{slug}"
        wd = ROOT / "outputs" / "parallel" / "workers" / slug
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "worker.ipynb").write_text(json.dumps(_nb(_cells(tag, n, splits))))
        (wd / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": slug[:50], "code_file": "worker.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "docker_image": DOCKER_IMAGE,
            "dataset_sources": ["nguynrichard/auto-vqabest"] + ([ds] if ds else []),
            "kernel_sources": [kern] if kern else [],
        }, indent=2))
        _kaggle(acc, "kernels", "push", "-p", str(wd))
        _register(led, job, acc, kid, {"num_tokens": n, "ckpt": ds or kern, "splits": splits,
                                       "gen_image": "full"})
        time.sleep(2)


def cmd_collect() -> None:
    led = load_ledger()
    out_root = ROOT / "outputs" / "tok_full"
    done = []
    for job, j in led["jobs"].items():
        if not job.startswith("tokfull:") or j.get("status") != "running":
            continue
        st = _kaggle(j["account"], "kernels", "status", j["kernel"], check=False)
        if "COMPLETE" not in st:
            print(f"[wait] {job}: {st.strip()[-40:]}"); continue
        tag = job.split(":", 1)[1]
        _kaggle(j["account"], "kernels", "output", j["kernel"], "--file-pattern", r".*\.(json|jsonl)$",
                "-p", str(out_root / tag), check=False)
        evals = sorted((out_root / tag).rglob("eval_*.json"))
        if evals:
            done.append(job)
            for f in evals:
                d = json.loads(f.read_text())
                print(f"[ok] {tag} {f.stem}: F1 {d.get('f1', 0) * 100:.2f}  n={d.get('n')}  gen_image={d.get('gen_image')}")
    led = load_ledger()
    for job in done:
        led["jobs"][job]["status"] = "done"
    save_ledger(led)


if __name__ == "__main__":
    if sys.argv[1] == "launch":
        cmd_launch([a for a in sys.argv[2:] if not a.startswith("--")] or None)
    else:
        cmd_collect()
