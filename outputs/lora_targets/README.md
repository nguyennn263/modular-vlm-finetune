# Uncollected LoRA-target runs (Sept 2026), pulled 2026-10-17

Multi-Token 8 + rank-16 decoder LoRA, 1 epoch, old first-tile generation eval (Sept), full val.
Ledger listed them as "running"; the kernels had COMPLETE.

| run | LoRA targets | F1 | CIDEr |
|---|---|--:|--:|
| expa-lora16-mlp s42 / s123 / s3407 | gate, up, down | 18.68 / 19.74 / 22.30 | 46.71 / 49.87 / 44.04 |
| expa-lora16-all s42 / s123 | q, k, v, o, gate, up, down | 38.11 / 39.22 | 71.49 / 69.19 |
| rq6-lora16-mlp-fb s42 | gate, up, down; bridge frozen (stage 2) | 50.66 | 98.69 |

For reference, attention-only LoRA (q, k, v, o) 1 epoch is 53.65 (full-image eval). The MLP/all
numbers are far below the bridge alone (~50) -- not investigated; not used in the paper.
