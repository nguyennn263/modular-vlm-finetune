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
            "# InternVL patch/__init__ hard-imports flash_attn monkey-patches; the finetune script",
            "# does not need them (Kaggle GPUs are pre-Ampere anyway). Strip those 2 import lines.",
            "os.system(\"sed -i '/flash_attn_monkey_patch import/d' \"",
            "          \"/tmp/wk/Vintern/internvl_chat/internvl/patch/__init__.py\")",
            "print('repo:', os.getcwd()); os.system('ls /tmp/wk && head -4 /tmp/wk/Vintern/internvl_chat/internvl/patch/__init__.py')",
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
            "# resolve the REAL mounted image dir (same resolver the bridge pipeline uses)",
            "from src.data.labeled_table import resolve_dirs",
            "_texts, _imgs = resolve_dirs()",
            "open('/tmp/wk/IMAGES', 'w').write(str(_imgs))",
            "print('IMAGES =', _imgs, '| exists:', os.path.isdir(_imgs))",
        ),
        code(
            "# ---- cookbook environment, installed in FULL (colab cells 1-2), with two deltas ----",
            "# delta 1: pin torch 2.5.1 — Kaggle ships a torch too new for the 2024-era InternVL",
            "#          trainer + transformers 4.47.",
            "# delta 2: SKIP flash_attn (5h source build; Kaggle P100/T4 are pre-Ampere and cannot",
            "#          run flash-attn v2 anyway). The unconditional flash import in patch/__init__",
            "#          is sed-stripped above; every model file gates flash behind try/except.",
            "!pip -q install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121",
            "# deepspeed==0.15.4: latest deepspeed uses torch.library.custom_op(list[int]) which",
            "# torch 2.5.1's infer_schema rejects (fixed in torch 2.6). 0.15.4 imports clean on 2.5.",
            "!DS_BUILD_OPS=0 pip -q install timm einops 'peft==0.14.0' wandb deepspeed==0.15.4 bitsandbytes decord tensorboardX gdown imageio opencv-python-headless",
            "!pip -q install -U datasets",
            "!pip -q install transformers==4.47.0 'accelerate>=1.1,<1.3' 'numpy<2.1'",
            "# peft's LoRA dispatcher raises on torchao<0.16 (Kaggle ships 0.10); we don't use it",
            "!pip -q uninstall -y torchao 2>/dev/null; echo 'torchao removed (not needed)'",
        ),
        code(
            "# import smoke test BEFORE the slow model download — catches any remaining import gap",
            "import os, subprocess, sys",
            "os.chdir('/tmp/wk/Vintern/internvl_chat'); sys.path.insert(0, os.getcwd())",
            "r = subprocess.run([sys.executable, '-c',",
            "  'import torch,transformers,deepspeed,decord,timm,peft,cv2,imageio;'",
            "  'import internvl.patch, internvl.train.dataset, internvl.model.internvl_chat;'",
            "  'from internvl.train.trainer_monkey_patch import replace_create_optimizer;'",
            "  'print(\"OK\", torch.__version__, transformers.__version__)'],",
            "  capture_output=True, text=True)",
            "print(r.stdout); print(r.stderr)",
            "assert 'OK' in r.stdout, 'internvl import chain still broken -- see stderr above'",
        ),
        code(
            "import os",
            "os.chdir('/tmp/wk/Vintern')",
            "!mkdir -p pretrained && huggingface-cli download --resume-download "
            "--local-dir-use-symlinks False 5CD-AI/Vintern-1B-v3_5 --local-dir pretrained/Vintern-1B-v3_5 "
            "2>&1 | tail -3",
        ),
        code(
            "# convert our splits -> InternVL chat SFT format (runtime-resolved image dir)",
            "%cd /tmp/wk/repo",
            "import subprocess, json, shutil, os",
            "IMAGES = open('/tmp/wk/IMAGES').read().strip()",
            "subprocess.run(['python', 'experiments/vintern-ft/build_data.py', '--images-dir', IMAGES,",
            "                '--meta-image-root', IMAGES, '--out-dir', 'experiments/vintern-ft/data'], check=True)",
            "src='/tmp/wk/repo/experiments/vintern-ft/data'",
            "dst='/tmp/wk/Vintern/internvl_chat/shell/data'; os.makedirs(dst, exist_ok=True)",
            "for f in ['autovivqa_train.jsonl','autovivqa_val.jsonl','autovivqa_test.jsonl']:",
            "    shutil.copy(f'{src}/{f}', f'{dst}/{f}')",
            "meta={'autovivqa-train':{'root':IMAGES,'annotation':f'{dst}/autovivqa_train.jsonl',",
            "  'data_augment':False,'repeat_time':1,'length':sum(1 for _ in open(f'{dst}/autovivqa_train.jsonl'))}}",
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
            "# single-GPU LoRA + grad-ckpt does not need ZeRO; skip deepspeed (one less failure surface)",
            "os.environ['SKIP_DEEPSPEED'] = '1'",
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
            "import subprocess; IMAGES = open('/tmp/wk/IMAGES').read().strip()",
            f"subprocess.run(['python','experiments/vintern-ft/gen_vintern.py','--model-path','{merged}',",
            "  '--split','val','--data','experiments/vintern-ft/data/autovivqa_val.jsonl',",
            "  '--images-dir',IMAGES,'--out','/kaggle/working/out/val','--max-num','6'])",
        ),
        code(
            "%cd /tmp/wk/repo",
            "import subprocess; IMAGES = open('/tmp/wk/IMAGES').read().strip()",
            f"subprocess.run(['python','experiments/vintern-ft/gen_vintern.py','--model-path','{merged}',",
            "  '--split','test','--data','experiments/vintern-ft/data/autovivqa_test.jsonl',",
            "  '--images-dir',IMAGES,'--out','/kaggle/working/out/test','--max-num','6'])",
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
