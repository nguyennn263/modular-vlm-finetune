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

## 3 seeds (42 / 123 / 3407), mean ± std (ddof=0) -- 2026-10-09

All 24 runs: `outputs/train_gl/gl-g{g}-k{k}[r2]-s{seed}_eval/`. Columns: Acc, P, R, F1, BLEU,
ROUGE-L, METEOR, CIDEr (x100), CE.

```
== val  (mean ± std over seeds, ddof=0)
g   k    tok  n        accura        precis        recall            f1          bleu        rouge_        meteor         cider            CE
8   1    9    3   9.33± 0.29  52.03± 0.12  52.75± 0.10  51.03± 0.12  16.54± 0.27  49.21± 0.13  41.37± 0.11  99.27± 0.58  1.496±0.004
8   9    17   3  10.25± 0.26  53.63± 0.04  54.72± 0.16  52.82± 0.11  17.58± 0.03  50.96± 0.05  43.56± 0.20 106.11± 0.55  1.460±0.004
8   36   44   3  12.15± 0.44  55.19± 0.43  56.14± 0.40  54.32± 0.41  18.22± 0.39  52.48± 0.43  45.31± 0.42 111.88± 1.43  1.421±0.004
8   144  152  3  13.83± 0.12  56.94± 0.20  58.10± 0.15  56.15± 0.08  19.82± 0.00  54.21± 0.06  47.51± 0.05 119.05± 0.25  1.379±0.002
14  1    15   3   9.67± 0.14  52.50± 0.40  53.48± 0.14  51.67± 0.25  17.30± 0.36  49.93± 0.24  42.36± 0.19 101.66± 1.11  1.467±0.001
14  9    23   3  10.55± 0.22  53.96± 0.10  54.66± 0.11  52.98± 0.08  17.83± 0.23  51.17± 0.10  43.71± 0.13 106.02± 0.80  1.437±0.004
14  36   50   3  12.38± 0.22  55.86± 0.21  56.72± 0.33  54.96± 0.28  18.73± 0.23  53.10± 0.27  46.01± 0.42 113.29± 1.01  1.396±0.003
14  144  158  3  14.19± 0.36  57.36± 0.07  58.58± 0.03  56.61± 0.04  20.18± 0.13  54.70± 0.05  48.00± 0.05 120.71± 0.12  1.365±0.003
== test  (mean ± std over seeds, ddof=0)
g   k    tok  n        accura        precis        recall            f1          bleu        rouge_        meteor         cider            CE
8   1    9    3   9.06± 0.16  51.49± 0.15  52.60± 0.26  50.71± 0.16  16.08± 0.19  48.91± 0.17  41.28± 0.21  96.90± 0.87  1.506±0.002
8   9    17   3   9.83± 0.09  52.92± 0.18  54.10± 0.24  52.11± 0.25  16.85± 0.26  50.14± 0.25  42.79± 0.30 101.30± 0.87  1.475±0.006
8   36   44   3  11.40± 0.25  54.65± 0.28  55.80± 0.20  53.82± 0.22  17.98± 0.25  51.92± 0.21  44.74± 0.14 107.67± 0.53  1.437±0.003
8   144  152  3  12.81± 0.08  55.94± 0.08  57.33± 0.13  55.20± 0.07  19.27± 0.10  53.33± 0.03  46.45± 0.11 114.13± 0.46  1.394±0.002
14  1    15   3   9.44± 0.34  52.41± 0.29  53.59± 0.01  51.67± 0.12  16.89± 0.29  49.84± 0.15  42.28± 0.08  99.53± 0.93  1.476±0.003
14  9    23   3  10.55± 0.27  53.65± 0.22  54.60± 0.38  52.76± 0.28  17.56± 0.10  50.82± 0.32  43.49± 0.39 103.63± 1.24  1.447±0.005
14  36   50   3  11.75± 0.21  55.36± 0.21  56.56± 0.21  54.56± 0.19  18.52± 0.18  52.64± 0.19  45.53± 0.33 110.58± 0.95  1.412±0.003
14  144  158  3  13.19± 0.23  56.64± 0.13  58.17± 0.26  56.04± 0.19  19.79± 0.15  54.11± 0.16  47.38± 0.26 116.74± 0.76  1.377±0.003
```

Selection (val F1, smallest config within 1 std of the best): best = g14-k144, 56.61 ± 0.04;
g8-k144 56.15 ± 0.08 is 0.46 below (> 5 std), so **g = 14, k = 144** (158 tokens). Test:
56.04 ± 0.19 F1, 116.74 CIDEr. Cost-aware alternative: g14-k36 (50 tokens), val 54.96 ±
0.28 / test 54.56 ± 0.19, on par with Multi-Token 8 + LoRA 3 epochs (54.78 / 54.52)
without LoRA.

## Choosing one configuration that is both good and cheap (2026-10-09)

Costs: tokens into the LLM (g + k), measured eval wall-clock (val + test, mean of 3 seeds:
k=1 57-59 min, k=9 58-59, k=36 60-61, k=144 70-71), trainable params (g8 7.35M, g14 12.86M).

1. **Pareto front:** accuracy rises with every token step, so all 8 points are Pareto-optimal;
   the front alone does not pick one.
2. **Knee of the front (Kneedle: max normalised height above the cheapest-best chord)** --
   **g14-k36** for every metric (Acc, P, R, F1, BLEU, ROUGE-L, METEOR, CIDEr) on both val and
   test, with cost = tokens (16/16) and cost = measured eval time (16/16); with cost =
   log(tokens), 11/16.
3. **TOPSIS** (benefits = gain over Multi-Token 8 in F1/CIDEr/METEOR; costs = extra tokens,
   extra eval time, params, F1 std): accuracy weight 0.5 -> g14-k36 first; 0.3-0.4 -> k=9
   configs; 0.6-0.7 -> g14-k144. g14-k36 is the only configuration in the top 4 at every
   weight from 0.3 to 0.7.
4. **Parity target:** g14-k36 is the cheapest configuration at or above Multi-Token 8 + LoRA
   3 epochs (54.78 / 54.52): val 54.96 ± 0.28, test 54.56 ± 0.19, without LoRA.
5. Share of the maximum gain: g14-k36 gets +4.22 of the +5.87 F1 that g14-k144 gets
   (72%) with 50 of 158 tokens (32%), and ~15% less eval time.

**Choice: g14-k36** (14 global + 36 local = 50 tokens) as the efficient configuration;
g14-k144 as the most accurate one. g = 14 over g = 8 at k = 36: +0.64 val / +0.74 test F1
(> 2 std) for 6 more tokens.
