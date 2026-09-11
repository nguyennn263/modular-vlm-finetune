#!/usr/bin/env python3
"""Minimal, cookbook-native, multi-session Vintern-1B fine-tune on Kaggle.

No repo clone, no phase0 — just the cookbook recipe (untouched: 6 tiles,
freeze all + LoRA-16, lr 4e-5, Hermes-2) against our pre-built AutoViVQA
splits, uploaded once as a tiny Kaggle dataset (duongcubu/autovivqa-internvl-sft).

Follows the real-world pattern (github.com/taitruong256's Vintern-chart-VQA
kernel): training does not fit one 12h Kaggle session at 6 tiles, so each
session's checkpoint is promoted to a Kaggle dataset and the next session
resumes from it via --resume_from_checkpoint, exactly like that notebook does.

    python experiments/vintern-ft/run_kaggle_minimal.py push --acc acc15
    python experiments/vintern-ft/run_kaggle_minimal.py push --acc acc15 --resume  # after promote
    python experiments/vintern-ft/run_kaggle_minimal.py status --acc acc15
    python experiments/vintern-ft/run_kaggle_minimal.py fetch  --acc acc15
    python experiments/vintern-ft/run_kaggle_minimal.py promote --acc acc15       # ckpt -> dataset
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACCT_DIR = Path.home() / ".kaggle-accounts"
VINTERN_URL = "https://github.com/5CD-AI/Vintern.git"
SLUG = "mvlm-vintern-ft-minimal"
DATA_DS = "duongcubu/autovivqa-internvl-sft"
IMAGES_DS = "nguynrichard/auto-vqabest"
CKPT_DS_SLUG = "vintern-ft-ckpt"
IMAGES = "/kaggle/input/auto-vqabest/preprocessed_images"
DATA_DIR = "/kaggle/input/autovivqa-internvl-sft"
OUT_DIR_LOCAL = ROOT / "experiments" / "vintern-ft" / "out_min"
CKPT_STAGE = ROOT / "experiments" / "vintern-ft" / "ckpt_stage"


def kaggle(acc: str, *args: str, check: bool = True) -> str:
    env = {**os.environ, "KAGGLE_CONFIG_DIR": str(ACCT_DIR / acc)}
    r = subprocess.run(["kaggle", *args], env=env, text=True, capture_output=True, check=False)
    out = (r.stdout or "") + (r.stderr or "")
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} [{acc}] -> {r.returncode}\n{out}")
    return out


def user(acc: str) -> str:
    return json.loads((ACCT_DIR / acc / "kaggle.json").read_text())["username"]


def code(*lines: str) -> dict:
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def _embed_file(dst: str, local_path: Path) -> dict:
    """Write a local helper file's content into /tmp/wk/<name> at kernel start,
    the same way the cookbook itself builds train_bash_content as a Python
    string (no multi-file kernel upload -- that mechanism isn't proven)."""
    content = local_path.read_text()
    return code(
        "import os; os.makedirs('/tmp/wk', exist_ok=True)",
        f"open({dst!r}, 'w').write({content!r})",
        f"print('wrote {dst}', os.path.getsize({dst!r}), 'bytes')",
    )


def cells(seed: int, resume: bool, epochs: int) -> list[dict]:
    ft_out = "/kaggle/working/work_dirs/vintern_lora"
    merged = ft_out + "_merge"
    resume_ckpt_dir = "/kaggle/input/" + CKPT_DS_SLUG
    here = ROOT / "experiments" / "vintern-ft"

    c = [
        _embed_file("/tmp/wk/finetune_cookbook.sh", here / "finetune_lora.sh"),
        _embed_file("/tmp/wk/merge_lora.py", here / "merge_lora.py"),
        _embed_file("/tmp/wk/gen_vintern_standalone.py", here / "gen_vintern_standalone.py"),
        code(
            "import subprocess",
            f"subprocess.call('git clone -q --depth 1 {VINTERN_URL} /tmp/wk/Vintern || "
            f"(mkdir -p /tmp/wk && cd /tmp/wk && git clone -q --depth 1 {VINTERN_URL} Vintern)', shell=True)",
            "# InternVL patch/__init__ hard-imports flash_attn monkey-patches we don't need",
            "# (Kaggle GPUs are pre-Ampere -> can't run flash-attn v2 anyway); strip the 2 lines.",
            "import os",
            "os.system(\"sed -i '/flash_attn_monkey_patch import/d' /tmp/wk/Vintern/internvl_chat/internvl/patch/__init__.py\")",
            "os.system('ls /tmp/wk/Vintern/internvl_chat && head -4 /tmp/wk/Vintern/internvl_chat/internvl/patch/__init__.py')",
        ),
        code(
            "# cookbook env, in full, with the two Kaggle-2026-compat pins we proved necessary:",
            "#  - torch 2.5.1 (Kaggle's default torch is too new for transformers 4.47 / this 2024 trainer)",
            "#  - deepspeed==0.15.4, peft==0.14.0 (latest deepspeed/peft break on torch 2.5's schema infer",
            "#    and Kaggle's old torchao respectively) -- installed IMPORT-ONLY (DS_BUILD_OPS=0), the",
            "#    trainer runs plain (no --deepspeed arg, single GPU LoRA doesn't need ZeRO)",
            "#  - NO flash_attn install (P100/T4 can't run it; every model file gates it behind try/except)",
            "!pip -q install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121",
            "!DS_BUILD_OPS=0 pip -q install timm einops 'peft==0.14.0' wandb deepspeed==0.15.4 bitsandbytes decord tensorboardX gdown imageio opencv-python-headless",
            "!pip -q install -U datasets",
            "!pip -q install transformers==4.47.0 'accelerate>=1.1,<1.3' 'numpy<2.1'",
            "!pip -q uninstall -y torchao 2>/dev/null; echo done",
        ),
        code(
            "import os",
            "print('=== /kaggle/input ==='); os.system('ls -la /kaggle/input')",
            "os.system('ls -la /kaggle/input/autovivqa-internvl-sft 2>&1')",
            "os.system('ls -la /kaggle/input/datasets 2>&1')",
            "os.system('find /kaggle/input -iname \"autovivqa_train*\" 2>&1')",
        ),
        code(
            "import subprocess, sys, os",
            "os.chdir('/tmp/wk/Vintern/internvl_chat')",
            "r = subprocess.run([sys.executable, '-c',",
            "  'import torch,transformers,deepspeed,decord,timm,peft,cv2,imageio;'",
            "  'import internvl.patch, internvl.train.dataset, internvl.model.internvl_chat;'",
            "  'from internvl.train.trainer_monkey_patch import replace_create_optimizer;'",
            "  'print(\"OK\", torch.__version__, transformers.__version__)'],",
            "  capture_output=True, text=True)",
            "print(r.stdout); print(r.stderr)",
            "assert 'OK' in r.stdout, 'import chain broken'",
        ),
        code(
            "import os",
            "os.chdir('/tmp/wk/Vintern')",
            "!mkdir -p pretrained && huggingface-cli download --resume-download "
            "--local-dir-use-symlinks False 5CD-AI/Vintern-1B-v3_5 --local-dir pretrained/Vintern-1B-v3_5 "
            "2>&1 | tail -3",
        ),
        code(
            "# our pre-built splits (already committed as a tiny Kaggle dataset -- no repo clone needed).",
            "# Resolve the actual mount path by search -- Kaggle's input layout has varied",
            "# (flat /kaggle/input/<slug>/ vs nested /kaggle/input/datasets/<owner>/<slug>/).",
            "import json, os, glob",
            "hits = glob.glob('/kaggle/input/**/autovivqa_train.jsonl', recursive=True)",
            "assert hits, 'autovivqa_train.jsonl not found anywhere under /kaggle/input -- see the ls above'",
            "data_dir = os.path.dirname(hits[0])",
            "print('resolved DATA_DIR =', data_dir)",
            f"dst = '/tmp/wk/Vintern/internvl_chat/shell/data'; os.makedirs(dst, exist_ok=True)",
            f"meta = {{'autovivqa-train': {{'root': '{IMAGES}',",
            "  'annotation': f'{data_dir}/autovivqa_train.jsonl', 'data_augment': False, 'repeat_time': 1,",
            "  'length': sum(1 for _ in open(f'{data_dir}/autovivqa_train.jsonl'))}}",
            "json.dump(meta, open(f'{dst}/meta_autovivqa.json', 'w'), ensure_ascii=False, indent=2)",
            "open('/tmp/wk/DATA_DIR', 'w').write(data_dir)",
            "print(meta)",
        ),
    ]

    resume_setup = []
    resume_arg = ""
    if resume:
        resume_setup = [code(
            "# resume from the previous session's checkpoint (promoted to a Kaggle dataset)",
            "import shutil, os, glob",
            f"os.makedirs('{ft_out}', exist_ok=True)",
            f"ckpts = sorted(glob.glob('{resume_ckpt_dir}/checkpoint-*'), key=lambda p: int(p.rsplit('-',1)[1]))",
            "assert ckpts, 'no checkpoint found in resume dataset'",
            "src = ckpts[-1]; step = os.path.basename(src)",
            f"dst = os.path.join('{ft_out}', step)",
            "shutil.copytree(src, dst)",
            f"print('resuming from', dst, '->', os.listdir(dst))",
        )]
        resume_arg = f'''os.environ["RESUME_ARG"] = f"--resume_from_checkpoint {ft_out}/{{step}}"'''

    c += resume_setup
    c.append(code(
        "%cd /tmp/wk/Vintern/internvl_chat",
        "import os, torch",
        "ng = torch.cuda.device_count()",
        "print('GPUs:', ng, [torch.cuda.get_device_name(i) for i in range(ng)])",
        "os.environ['GPUS'] = str(ng)",
        "os.environ['BATCH_SIZE'] = '16'",
        "os.environ['PER_DEVICE_BATCH_SIZE'] = '4' if ng >= 2 else '2'",
        "os.environ['PYTHONPATH'] = os.getcwd()",
        "os.environ['MODEL_PATH'] = '/tmp/wk/Vintern/pretrained/Vintern-1B-v3_5'",
        "os.environ['META_PATH'] = 'shell/data/meta_autovivqa.json'",
        f"os.environ['OUTPUT_DIR'] = '{ft_out}'",
        f"os.environ['SEED'] = '{seed}'",
        f"os.environ['EPOCHS'] = '{epochs}'",
        "os.environ['SKIP_DEEPSPEED'] = '1'",
        (resume_arg if resume else "os.environ['RESUME_ARG'] = ''"),
        "print('RESUME_ARG=', os.environ['RESUME_ARG'])",
        "!bash /tmp/wk/finetune_cookbook.sh 2>&1 | tail -80",
        f"print('checkpoints now:', sorted(os.listdir('{ft_out}')) if os.path.isdir('{ft_out}') else 'MISSING')",
    ))

    # Only merge + generate once training has actually reached the end of
    # an epoch (the training shell itself decides; if it's mid-epoch this
    # cell is skipped so the session doesn't waste time on a stale merge).
    c.append(code(
        "import os, glob",
        f"done = os.path.exists('{ft_out}/adapter_model.safetensors') or os.path.exists('{ft_out}/pytorch_model.bin')",
        "print('epoch finished (final adapter present)?', done)",
        "os.environ['EPOCH_DONE'] = '1' if done else '0'",
    ))
    c.append(code(
        "if os.environ.get('EPOCH_DONE') == '1':",
        f"    !python /tmp/wk/merge_lora.py {ft_out} {merged}",
        "    !cp /tmp/wk/Vintern/pretrained/Vintern-1B-v3_5/*.py " + merged + "/",
        "    !cp /tmp/wk/Vintern/pretrained/Vintern-1B-v3_5/config.json " + merged + "/",
        f"    !ls {merged}",
        "else:",
        "    print('skip merge/generate -- epoch not finished yet, checkpoint will be promoted for resume')",
    ))
    c.append(code(
        "if os.environ.get('EPOCH_DONE') == '1':",
        "    import sys; sys.path.append('/tmp/wk')",
        "    import subprocess",
        "    data_dir = open('/tmp/wk/DATA_DIR').read().strip()",
        f"    subprocess.run(['python', '/tmp/wk/gen_vintern_standalone.py', '--model-path', '{merged}',",
        "      '--data', f'{data_dir}/autovivqa_val.jsonl', '--images-dir', '" + IMAGES + "',",
        "      '--out', '/kaggle/working/out/val', '--max-num', '6'])",
    ))
    c.append(code(
        "if os.environ.get('EPOCH_DONE') == '1':",
        "    import subprocess",
        "    data_dir = open('/tmp/wk/DATA_DIR').read().strip()",
        f"    subprocess.run(['python', '/tmp/wk/gen_vintern_standalone.py', '--model-path', '{merged}',",
        "      '--data', f'{data_dir}/autovivqa_test.jsonl', '--images-dir', '" + IMAGES + "',",
        "      '--out', '/kaggle/working/out/test', '--max-num', '6'])",
    ))
    return c


def nb(cs: list[dict]) -> dict:
    return {"cells": cs, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
            "language": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def cmd_push(a):
    acc = a.acc
    kid = f"{user(acc)}/{SLUG}"
    d = ROOT / "experiments" / "vintern-ft" / "worker_min"
    d.mkdir(parents=True, exist_ok=True)
    (d / "worker.ipynb").write_text(json.dumps(nb(cells(a.seed, a.resume, a.epochs))))
    ds = [DATA_DS, IMAGES_DS]
    if a.resume:
        ds.append(f"{user(acc)}/{CKPT_DS_SLUG}")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": SLUG, "code_file": "worker.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ds, "competition_sources": [], "kernel_sources": [],
    }, indent=2))
    print(kaggle(acc, "kernels", "push", "-p", str(d)))
    print(f"[pushed] {kid} resume={a.resume} epochs={a.epochs}")


def cmd_status(a):
    print(kaggle(a.acc, "kernels", "status", f"{user(a.acc)}/{SLUG}"))


def cmd_fetch(a):
    OUT_DIR_LOCAL.mkdir(parents=True, exist_ok=True)
    print(kaggle(a.acc, "kernels", "output", f"{user(a.acc)}/{SLUG}", "-p", str(OUT_DIR_LOCAL)))
    print(f"[fetched] -> {OUT_DIR_LOCAL}")


def cmd_promote(a):
    """Find the latest checkpoint under the fetched output and push it as/into
    the resume dataset so the next --resume session can load it."""
    ckpts = sorted(
        glob.glob(str(OUT_DIR_LOCAL / "work_dirs" / "vintern_lora" / "checkpoint-*")),
        key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)),
    )
    if not ckpts:
        raise SystemExit(f"no checkpoint-* found under {OUT_DIR_LOCAL}")
    latest = Path(ckpts[-1])
    size = sum(f.stat().st_size for f in latest.rglob("*") if f.is_file())
    print(f"[promote] latest checkpoint: {latest} ({size/1e6:.1f} MB)")
    if CKPT_STAGE.exists():
        shutil.rmtree(CKPT_STAGE)
    CKPT_STAGE.mkdir(parents=True)
    shutil.copytree(latest, CKPT_STAGE / latest.name)
    meta = {"title": "vintern-ft-ckpt", "id": f"{user(a.acc)}/{CKPT_DS_SLUG}", "licenses": [{"name": "CC0-1.0"}]}
    (CKPT_STAGE / "dataset-metadata.json").write_text(json.dumps(meta))
    try:
        print(kaggle(a.acc, "datasets", "create", "-p", str(CKPT_STAGE), "--dir-mode", "zip"))
    except RuntimeError:
        print(kaggle(a.acc, "datasets", "version", "-p", str(CKPT_STAGE), "-m", latest.name, "--dir-mode", "zip"))
    print(f"[promote] -> {user(a.acc)}/{CKPT_DS_SLUG} ({latest.name})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("push", "status", "fetch", "promote"):
        p = sub.add_parser(name)
        p.add_argument("--acc", default="acc15")
        if name == "push":
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--resume", action="store_true")
            p.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()
    {"push": cmd_push, "status": cmd_status, "fetch": cmd_fetch, "promote": cmd_promote}[args.cmd](args)
