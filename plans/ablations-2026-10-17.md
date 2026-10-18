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

## Added 2026-10-17 23:10: extra seeds for C-Abstractor / Patch-pool (mean)
`train_queue.py fill --xb`: chains xb-{cab,ppm}-s{123,3407} on acc1 / acc2 / acc4 / acc5, each
train (September recipe: --epochs 2 in one kernel, batch 8, same split; in-training text metrics
off) -> val eval (--gen-image full). With the re-scored seed-42 runs this gives three seeds, so
the two Table 2 rows get mean +- std and lose the one-run marker.

## 2026-10-18 11:20: Patch-pool seeds 123/3407 hit the 12h cap
Both train kernels were cancelled with epoch 2 at 3178/3222 (global step 6400/6444); one epoch
now takes ~5h on Kaggle T4 (the September seed-42 run took 4.9h for both epochs). Their
step_6400.pt checkpoints are scored by `bridge_full.py` (ppm-s{123,3407}-st6400): the cosine
learning rate is ~2e-8 at step 6400, so the 44 missing steps (0.7%) change the weights negligibly.
The same applies to the C-Abstractor chains if they are cancelled.
C-Abstractor seed 123 was cancelled at step 5600/6444 (LR still ~4% of its peak, 13% of the
steps missing), too early to score. Epoch 2 is redone from its end-of-epoch-1 last_model.pt
(`train_queue.py fill --xb --xb-resume`, chain xb-cab-s123c on acc1, mid-epoch validation off).
Kaggle does not mount the output of a cancelled kernel, so these checkpoints were downloaded and
uploaded as private datasets: nguyennn251/mvlm-ppm-s123-st6400, nguyennguyen263/mvlm-ppm-s3407-st6400
(step_6400.pt) and nguyennn263/mvlm-cab-s123-ep1 (epoch-1 last_model.pt). Jobs: bridge_full
ppm-s{123,3407}-st6400-ds; train_queue --xb --xb-resume chain xb-cab-s123d.
