# global_local: how many CLS (global) and patch (local) tokens

Plain bridge (no LoRA), AutoViVQA only, 336px single image, 2 epochs (epoch 2 resumed),
seed 42, `--gen-image full`, metrics from `metrics.vqa_metrics.score_answers` (x100).
Global = Multi-Token on the CLS token (g tokens, trainable, prefixed). Local = Vintern's
own mlp1 tokens (frozen), average-pooled to k = grid^2, written into Vintern's image slot
`<img><IMG_CONTEXT>...</img>`. Code: `GlobalLocalBridge` (commit e9620fe, padding fix
84dd815). Raw files: `outputs/train_gl/gl-g{g}-k{k}[r2]-s42_eval/out/{val,test}/`.

Baseline k = 0: Multi-Token 8, 4 seeds: val 50.74 / 99.07, test 50.47 / 96.19.

| g | k | tokens | val F1 | val CIDEr | test F1 | test CIDEr | val CE |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | 1 | 9 | 51.18 | 99.02 | 50.93 | 97.99 | 1.492 |
| 8 | 9 | 17 | 52.76 | 106.45 | 51.76 | 100.22 | 1.466 |
| 8 | 36 | 44 | 53.79 | 109.87 | 53.72 | 107.12 | 1.425 |
| 8 | 144 | 152 | 56.25 | 118.77 | 55.15 | 114.49 | 1.375 |
| 14 | 1 | 15 | 51.98 | 103.20 | 51.83 | 100.84 | 1.468 |
| 14 | 9 | 23 | 53.09 | 107.04 | 52.87 | 103.71 | 1.433 |
| 14 | 36 | 50 | 54.75 | 112.37 | 54.47 | 109.39 | 1.398 |
| 14 | 144 | 158 | **56.62** | **120.55** | **55.79** | **115.67** | **1.366** |

k = 144 runs are the `k144r2` relaunch: the first attempt OOM'd at step 1 with the fixed
400-token padding (fixed in 84dd815; padding is masked, results unaffected).

Single seed. For reference, Multi-Token 8 + decoder LoRA, 3 epochs: val 54.78, test 54.52.
