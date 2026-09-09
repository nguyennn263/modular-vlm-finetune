"""Merge LoRA adapters into the base weights — verbatim from the Vintern
fine-tune cookbook (colab cell 41 == Vintern repo tools/merge_lora.py).

    python merge_lora.py <lora_output_dir> <merged_output_dir>

Run from inside the Vintern repo's `internvl_chat/` directory (it imports
`internvl.model.internvl_chat`).
"""
import argparse
import sys

import torch

sys.path.append("/tmp/wk/Vintern/internvl_chat")
sys.path.append(".")

from internvl.model.internvl_chat import InternVLChatModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("input_path", type=str)
ap.add_argument("output_path", type=str)
args = ap.parse_args()

print("Loading model...")
model = InternVLChatModel.from_pretrained(
    args.input_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval()
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.input_path, trust_remote_code=True)

if model.config.use_backbone_lora:
    model.vision_model.merge_and_unload()
    model.vision_model = model.vision_model.model
    model.config.use_backbone_lora = 0
if model.config.use_llm_lora:
    model.language_model.merge_and_unload()
    model.language_model = model.language_model.model
    model.config.use_llm_lora = 0

print("Saving model...")
model.save_pretrained(args.output_path)
print("Saving tokenizer...")
tokenizer.save_pretrained(args.output_path)
print("Done!")
