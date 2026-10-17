# Ablations launched 2026-10-17 (for ACIIDS 2027 Table 2 / §4.4)

## Jobs
| job (ledger key) | account | what | collect with |
|---|---|---|---|
| bridgefull:convabs-s42 | acc8 | C-Abstractor (Sept ckpt, seed 42) re-scored on the full image, val | `bridge_full.py collect` |
| bridgefull:poolmean-s42 | acc13 | Patch-pool mean (Sept ckpt, seed 42) re-scored on the full image, val | same |
| bridgefull:g0k36-notrain | acc17 | g=0, k=36, nothing trained (Vintern reads 36 pooled projector tokens), val | same |
| gl-g0-k36-lora1-s{42,123,3407}:ep1 -> :eval | acc3 / acc10 / acc7 | g=0, k=36 + decoder LoRA r16, 1 joint epoch (g14-k36-lora1 recipe without global tokens), val + test | `train_queue.py collect --gl --gl-lora --local-only` (fill chains the eval phase) |

Outputs: `outputs/bridge_full/<tag>/`, `outputs/train_gl/gl-g0-k36-lora1-s*_eval/`.

## Rules agreed with the user (decided before seeing results)
1. Every number goes in, whatever it is. C-Abstractor / Patch-pool rows in Table 2 (bridge-only block),
   marked one seed; C-Abstractor described as 9 tokens (Honeybee normally uses >= 144).
2. One seed is enough only for a large gap: if a bridge lands within ~1 F1 of Multi-Token (50.74), write
   "on par" (or run more seeds), never "below". Rewrite "3.7-5.6 F1 below Multi-Token" from the real numbers.
3. Patch-pool mean = per-patch Linear(1024->896) + 1D adaptive average pooling of the row-major token
   sequence into 8 tokens (bands, not 2D blocks; CLS included) -- describe it as such.
4. g=0 + LoRA is compared with ViBridge-VQA (k=36) + LoRA (58.21 val / 58.17 test); the untrained g=0
   number shows what the frozen VLM does with the local tokens alone.
