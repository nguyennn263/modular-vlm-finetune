#!/usr/bin/env python3
"""Push / poll the Vintern-1B fine-tune reproduction as a single Kaggle kernel.

    python experiments/vintern-ft/run_kaggle.py push  --acc acc15 [--seed 42]
    python experiments/vintern-ft/run_kaggle.py status --acc acc15
    python experiments/vintern-ft/run_kaggle.py fetch  --acc acc15   # -> experiments/vintern-ft/out/

Everything runs in one kernel: clone repos, install cookbook deps, download
Vintern-1B-v3_5, fine-tune (experiments/vintern-ft/finetune_lora.sh, verbatim
cookbook hyper-params), merge LoRA, eval val+test with our metrics, rescore
corpus. Outputs land under /kaggle/working/out.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACCT_DIR = Path.home() / ".kaggle-accounts"
REPO_URL = "https://github.com/nguyennn263/modular-vlm-finetune.git"
VINTERN_URL = "https://github.com/5CD-AI/Vintern.git"
IMAGES = "/kaggle/input/auto-vqabest/preprocessed_images"
SLUG = "mvlm-vintern-ft-repro"


def kaggle(acc: str, *args: str, check: bool = True) -> str:
    env = {**os.environ, "KAGGLE_CONFIG_DIR": str(ACCT_DIR / acc)}
    r = subprocess.run(["kaggle", *args], env=env, text=True, capture_output=True, check=False)
    out = (r.stdout or "") + (r.stderr or "")
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} [{acc}] -> {r.returncode}\n{out}")
    return out


def user(acc: str) -> str:
    return json.loads((ACCT_DIR / acc / "kaggle.json").read_text())["username"]


def branch() -> str:
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()


def code(*lines: str) -> dict:
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def cells(seed: int, br: str) -> list[dict]:
    ft_out = "/kaggle/working/work_dirs/vintern_lora"
    merged = ft_out + "_merge"
    return [
        code(
            "import os, subprocess, time",
            "os.makedirs('/tmp/wk', exist_ok=True); os.chdir('/tmp/wk')",
            "for _ in range(6):",
            f"    subprocess.call('git clone -q {REPO_URL} repo || (cd repo && git fetch -q)', shell=True)",
            "    if os.path.isdir('repo'): break",
            "    time.sleep(15)",
            f"os.chdir('/tmp/wk/repo'); os.system('git checkout -q {br} && git pull -q')",
            f"subprocess.call('git clone -q --depth 1 {VINTERN_URL} /tmp/wk/Vintern', shell=True)",
            "print('repo:', os.getcwd()); os.system('ls /tmp/wk && ls /tmp/wk/Vintern/internvl_chat')",
        ),
        code(
            "# minimal deps for phase0 (keep Kaggle's native torch; no setup_kaggle.sh downgrade)",
            "%cd /tmp/wk/repo",
            "!pip -q install pyarrow pyyaml pydantic 'pandas>=2' && pip -q install -e . --no-deps",
        ),
        code(
            "# build data/splits/{train,val,test}.jsonl (gitignored -> must regenerate on Kaggle)",
            "%cd /tmp/wk/repo",
            "!python scripts/phase0_build_data.py 2>&1 | tail -12",
            "import os; assert os.path.exists('data/splits/train.jsonl'), 'phase0 did not produce data/splits'",
            "print('splits:', {s: sum(1 for _ in open(f'data/splits/{s}.jsonl')) for s in ['train','val','test']})",
        ),
        code(
            "# cookbook deps (colab cells 1-2) -- transformers 4.47 for the InternVL trainer.",
            "# NO flash_attn: Kaggle P100/T4 are pre-Ampere, flash-attn v2 cannot run -> eager attn.",
            "!pip -q install transformers==4.47.0 peft deepspeed accelerate timm einops bitsandbytes datasets tensorboardX",
        ),
        code(
            "import os",
            "os.chdir('/tmp/wk/Vintern')",
            "!mkdir -p pretrained && huggingface-cli download --resume-download "
            "--local-dir-use-symlinks False 5CD-AI/Vintern-1B-v3_5 --local-dir pretrained/Vintern-1B-v3_5 "
            "2>&1 | tail -3",
        ),
        code(
            "# convert our splits -> InternVL chat SFT format",
            "%cd /tmp/wk/repo",
            f"!python experiments/vintern-ft/build_data.py --images-dir {IMAGES} "
            f"--meta-image-root {IMAGES} --out-dir experiments/vintern-ft/data 2>&1 | tail -8",
            "import json, shutil, os",
            "src='/tmp/wk/repo/experiments/vintern-ft/data'",
            "dst='/tmp/wk/Vintern/internvl_chat/shell/data'; os.makedirs(dst, exist_ok=True)",
            "for f in ['autovivqa_train.jsonl','autovivqa_val.jsonl','autovivqa_test.jsonl']:",
            "    shutil.copy(f'{src}/{f}', f'{dst}/{f}')",
            "meta={'autovivqa-train':{'root':'" + IMAGES + "',",
            "  'annotation':f'{dst}/autovivqa_train.jsonl','data_augment':False,'repeat_time':1,",
            "  'length':sum(1 for _ in open(f'{dst}/autovivqa_train.jsonl'))}}",
            "json.dump(meta, open(f'{dst}/meta_autovivqa.json','w'), ensure_ascii=False, indent=2)",
            "print(meta)",
            "shutil.copy('/tmp/wk/repo/experiments/vintern-ft/finetune_lora.sh',",
            "  '/tmp/wk/Vintern/internvl_chat/shell/internvl2.0/2nd_finetune/autovivqa_lora.sh')",
        ),
        code(
            "%cd /tmp/wk/Vintern/internvl_chat",
            "import os, torch",
            "ng = torch.cuda.device_count()",
            "print('GPUs:', ng, [torch.cuda.get_device_name(i) for i in range(ng)])",
            "# cookbook total batch = 16; smaller per-device on a single 16GB card",
            "os.environ['GPUS'] = str(ng)",
            "os.environ['BATCH_SIZE'] = '16'",
            "os.environ['PER_DEVICE_BATCH_SIZE'] = '4' if ng >= 2 else '2'",
            "os.environ['PYTHONPATH'] = os.getcwd()",
            "os.environ['MODEL_PATH'] = '/tmp/wk/Vintern/pretrained/Vintern-1B-v3_5'",
            "os.environ['META_PATH'] = './shell/data/meta_autovivqa.json'",
            f"os.environ['OUTPUT_DIR'] = '{ft_out}'",
            f"os.environ['SEED'] = '{seed}'",
            "print('zero config present:', os.path.exists('zero_stage1_config.json'))",
            "!bash shell/internvl2.0/2nd_finetune/autovivqa_lora.sh 2>&1 | tail -60",
            f"import os; print('train outputs:', sorted(os.listdir('{ft_out}')) if os.path.isdir('{ft_out}') else 'MISSING')",
        ),
        code(
            "# merge LoRA (cookbook cell 41-47)",
            "%cd /tmp/wk/Vintern/internvl_chat",
            f"import os; assert any('adapter' in f or f=='pytorch_model.bin' or f.endswith('.safetensors') for f in os.listdir('{ft_out}')), 'no adapter/weights in {ft_out}'",
            f"!python /tmp/wk/repo/experiments/vintern-ft/merge_lora.py {ft_out} {merged}",
            f"!cp /tmp/wk/Vintern/pretrained/Vintern-1B-v3_5/*.py {merged}/",
            f"!cp /tmp/wk/Vintern/pretrained/Vintern-1B-v3_5/config.json {merged}/",
            f"!ls {merged}",
        ),
        code(
            "# generation only (stays in the 4.47 / InternVL env); scoring is local",
            "%cd /tmp/wk/repo",
            f"!python experiments/vintern-ft/gen_vintern.py --model-path {merged} "
            f"--split val --data experiments/vintern-ft/data/autovivqa_val.jsonl "
            f"--images-dir {IMAGES} --out /kaggle/working/out/val --max-num 6 2>&1 | tail -15",
        ),
        code(
            "%cd /tmp/wk/repo",
            f"!python experiments/vintern-ft/gen_vintern.py --model-path {merged} "
            f"--split test --data experiments/vintern-ft/data/autovivqa_test.jsonl "
            f"--images-dir {IMAGES} --out /kaggle/working/out/test --max-num 6 2>&1 | tail -15",
        ),
        code(
            "import shutil, os",
            f"os.makedirs('/kaggle/working/out/lora_adapter', exist_ok=True)",
            f"for f in os.listdir('{ft_out}'):",
            f"    p=os.path.join('{ft_out}',f)",
            "    if os.path.isfile(p) and (f.endswith('.json') or f.endswith('.txt') or 'adapter' in f or f.endswith('.safetensors')):",
            "        shutil.copy(p, '/kaggle/working/out/lora_adapter/')",
            "print('=== predictions written ===')",
            "os.system('du -sh /kaggle/working/out; find /kaggle/working/out -name text_predictions_epoch_1.json -exec wc -l {} +')",
            "os.system('ls -R /kaggle/working/out | head -40')",
        ),
    ]


def nb(cs: list[dict]) -> dict:
    return {"cells": cs, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
            "language": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def cmd_push(a):
    acc = a.acc
    kid = f"{user(acc)}/{SLUG}"
    d = ROOT / "experiments" / "vintern-ft" / "worker"
    d.mkdir(parents=True, exist_ok=True)
    (d / "worker.ipynb").write_text(json.dumps(nb(cells(a.seed, branch()))))
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": SLUG, "code_file": "worker.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ["nguynrichard/auto-vqabest"], "competition_sources": [], "kernel_sources": [],
    }, indent=2))
    print(kaggle(acc, "kernels", "push", "-p", str(d)))
    print(f"[pushed] {kid}  (branch {branch()}, seed {a.seed})")


def cmd_status(a):
    print(kaggle(a.acc, "kernels", "status", f"{user(a.acc)}/{SLUG}"))


def cmd_fetch(a):
    out = ROOT / "experiments" / "vintern-ft" / "out"
    out.mkdir(parents=True, exist_ok=True)
    print(kaggle(a.acc, "kernels", "output", f"{user(a.acc)}/{SLUG}", "-p", str(out)))
    print(f"[fetched] -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("push", "status", "fetch"):
        p = sub.add_parser(name)
        p.add_argument("--acc", default="acc15")
        if name == "push":
            p.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    {"push": cmd_push, "status": cmd_status, "fetch": cmd_fetch}[args.cmd](args)
