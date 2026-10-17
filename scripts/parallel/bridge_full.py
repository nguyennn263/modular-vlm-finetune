#!/usr/bin/env python3
"""Full-image re-evaluation of the September C-Abstractor and Patch-pool (mean) bridges.

Both were trained on 2026-09-14 (feat/bridge-design-ablation @80be77a, seed 42, 2 epochs, 336px)
and scored with the old first-tile generation input. This re-scores them with the command used
for every other Table 2 bridge (eval-only, --n-tiles 1 --pooled-input default --gen-image full).
Checkpoints live only in the training kernels' output, mounted through kernel_sources, so each
job runs on the owning account. Bridge classes and configs are identical between 80be77a and
this branch (checked 2026-10-17).

    python scripts/parallel/bridge_full.py launch [tag ...]
    python scripts/parallel/bridge_full.py collect
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, _kaggle, _user, _code, _clone_cell, _nb, load_ledger, save_ledger, _register  # noqa

BRANCH = "exp/eval-input-diagnostic"
DOCKER_IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
                "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461")

# tag -> (bridge, account, kernel_source, splits)
JOBS = {
    "convabs-s42": ("conv_abstractor", "acc8", "lelynhan/mvlm-convabs-m9", ["val"]),
    "poolmean-s42": ("patch_pool_mean", "acc13", "avanrager/mvlm-poolabl-mean", ["val"]),
    # 2026-10-17: local tokens only (g = 0, k = 36) with nothing trained -- Vintern-1B reading the
    # 36 pooled projector tokens zero-shot; no checkpoint (the bridge has no trainable weights)
    "g0k36-notrain": ("global_local", "acc17", None, ["val"]),
}
EXTRA_FLAGS = {"g0k36-notrain": " --bridge-num-tokens 0 --local-grid 6"}
CK = "/tmp/ck/bridge"



def _cells(tag: str, bridge: str, splits: list[str]) -> list[dict]:
    cells = [
        _clone_cell(BRANCH),
        _code("!bash setup_kaggle.sh 2>&1 | tail -5"),
        _code("!python scripts/phase0_build_data.py 2>&1 | tail -6"),
    ]
    if tag not in EXTRA_FLAGS:
        cells += [
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
    else:
        cells.append(_code(f"!mkdir -p {CK}"))
    ckpt = "none" if tag in EXTRA_FLAGS else f"{CK}/model.pt"
    flags = EXTRA_FLAGS.get(tag, "")
    for split in splits:
        cells += [
            _code(f"!python -m src.cli.evaluate --bridge {bridge} "
                  f"--split-dir data/splits --split {split} --n-tiles 1 --pooled-input default "
                  f"--gen-image full --checkpoint {ckpt}{flags} --output {CK}/eval_{split}.json "
                  f"|| echo failed > /tmp/cell_failed",
                  "import os",
                  "assert not os.path.exists('/tmp/cell_failed'), 'evaluate failed'"),
            _code("import glob, os, shutil",
                  f"os.makedirs('{CK}', exist_ok=True)",
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
    for tag, (bridge, acc, kern, splits) in JOBS.items():
        job = f"bridgefull:{tag}"
        if only and tag not in only:
            continue
        if job in led["jobs"]:
            print(f"[skip] {job} already pushed ({led['jobs'][job]['status']})"); continue
        slug = f"mvlm-bridgefull-{tag}"
        kid = f"{_user(acc)}/{slug}"
        wd = ROOT / "outputs" / "parallel" / "workers" / slug
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "worker.ipynb").write_text(json.dumps(_nb(_cells(tag, bridge, splits))))
        (wd / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": slug[:50], "code_file": "worker.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "docker_image": DOCKER_IMAGE,
            "dataset_sources": ["nguynrichard/auto-vqabest"],
            "kernel_sources": [kern] if kern else [],
        }, indent=2))
        _kaggle(acc, "kernels", "push", "-p", str(wd))
        _register(led, job, acc, kid, {"bridge": bridge, "ckpt": kern, "splits": splits,
                                       "gen_image": "full"})
        time.sleep(2)


def cmd_collect() -> None:
    led = load_ledger()
    out_root = ROOT / "outputs" / "bridge_full"
    done = []
    for job, j in led["jobs"].items():
        if not job.startswith("bridgefull:") or j.get("status") != "running":
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
