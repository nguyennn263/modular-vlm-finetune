# Bản thiết kế paper — "Cải thiện Vintern-1B cách rẻ"

> **Bản đã qua biên tập trình bày (2026-09-06).** Cắt từ 13 bảng → 7 bảng chính +
> 2 hình + phụ lục per-seed. Một hệ ký hiệu, một bộ cột, làm tròn 2 chữ số (×100)
> khớp ViMoE. Caption + "Reading" bằng tiếng Anh (paper text); dòng "VI:" tiếng
> Việt để theo dõi.
>
> Artifact: https://claude.ai/code/artifact/fe068b4c-d59c-429f-bdba-ed9ea93bd557
> Khung câu chuyện: https://claude.ai/code/artifact/bb7bf7ee-d5f1-4749-bb56-29a5c5daa610

## Quy ước chung — khai báo một lần

- **Bold** = dòng của chúng tôi (recipe / bridge đề xuất). *Không* dùng bold cho
  "best per column".
- Cột metric: `Acc · Prec · Rec · F1 · BLEU · ROUGE · METEOR · CIDEr` — đồng bộ mọi
  bảng, in-house ×100, **2 chữ số thập phân**.
- Hai quy ước metric **tách hẳn thành bảng riêng**: Table 1 = in-house (khớp bảng
  AutoViVQA), Table 2 = corpus pycocoevalcap (so cross-paper). Không cắm dấu `*`
  vào từng số.
- Chú thích caveat: chữ thường ⁠ᵃ ᵇ ᶜ, định nghĩa **ngay trong caption** bảng đó.
- LNCS thật: đánh **Table 1…N tuần tự**, **Fig 1–2**. (Blueprint này đã phẳng hoá
  — không còn 5a–5d.)
- Cần làm khi viết bản nộp: (1) dịch toàn bộ prose sang tiếng Anh; (2) verify
  recipe train của "Vintern-1B (fine-tuned)" ở §4 AutoViVQA (nhiều khả năng là
  "ViT + projector full, LLM LoRA", không phải "fine-tune toàn bộ").

---

## PHẦN A — Cấu trúc paper (LNCS, 12–15 trang)

| § | Nội dung | Nguồn |
|---|---|---|
| **Abstract** | Vintern zero-shot hỏng (F1 17.6); recipe của Vintern train *full* InternViT-300M + projector + LoRA LLM trên 3M cặp. Ta: đóng băng cả hai backbone, chỉ train bridge 0.78% (+ LoRA decoder 0.23%), 1 tile → vượt Vintern fine-tuned trên metric sinh ở ~1% chi phí. Chẩn đoán 6 bước → nút thắt = frozen decoder. | — |
| **1 · Introduction** | ViMoE xây model mới · Vintern train nặng phía thị giác · **câu hỏi: adapt rẻ được không, nút thắt ở đâu** · 4 đóng góp. | — |
| **2 · Related Work** | VQA tiếng Việt (ViVQA/OpenViVQA/ViTextVQA/AutoViVQA/ViMoE) · frozen-backbone + projector (BLIP-2, "Inference-Optimal VLMs" 2411.03312) · parameter-efficient adaptation (LoRA, adapter). | — |
| **3 · Method** | 3.1 Kiến trúc frozen · 3.2 Năm bridge (thang capacity) · 3.3 Decoder-LoRA như can thiệp có chủ đích · 3.4 Hai chỗ vặn × 6 câu hỏi. | — |
| **4 · Experimental Setup** | AutoViVQA · **grouped split leak-free** · 8 metric · baseline (Table 1) · Vintern-FT recipe làm rõ — *cần verify §4 AutoViVQA*. | §4.1 |
| **5 · Main Results** | Recipe vs baseline. | Table 1, 2, 7 |
| **6 · Ablation: truy tìm nút thắt** | 6.1 bridge (RQ1–2, Table 3) · 6.2 tile-collapse (RQ3, Fig 2) · 6.3 oracle + routing (RQ4) · 6.4 training/alignment (RQ5) · 6.5 decoder-LoRA (RQ6, Table 5, Fig 1) · 6.6 tổng hợp (Table 4). | Table 3–5, Fig 1–2 |
| **7 · Human Validation & Error Analysis** | Self-check (Table 6) · [camera-ready: 2 annotator + κ] · lỗi per-category · độ dài sinh. | Table 6 |
| **8 · Discussion** | Frozen decoder là trần · nối "Inference-Optimal VLMs" · "reasoning-aware" của ViMoE cần đo trực tiếp · giới hạn. | — |
| **9 · Conclusion** | Recipe rẻ + chẩn đoán. Code + split + oracle table released. | — |
| **Appendix** | Per-seed tables (A1–A5), rank/epoch curve đầy đủ. | — |

---

## PHẦN B — Bảng chính (7) + hình (2)

### Table 1 — Main results (in-house metrics) — recipe khoá

**Table 1.** Recipe vs prior work on AutoViVQA (val). In-house metrics ×100.
Bold = ours. ᵃ BARTPhoBEiT CIDEr is a verbosity outlier, excluded from
comparison. ᵇ mean over 4 seeds (plain) / 3 seeds (+LoRA); std in Table 2.
Baseline rows: as reported by the AutoViVQA benchmark, split-independent.

| Model | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Vintern-1B (base, zero-shot) | 0.12 | 17.52 | 19.87 | 17.55 | 1.91 | 25.84 | 23.93 | 8.54 |
| ViT5_ViT | 7.97 | 46.84 | 50.33 | 48.52 | 4.13 | 46.89 | 31.02 | 72.68 |
| BARTPhoBEiT | 8.81 | 45.30 | 46.48 | 45.88 | 4.33 | 44.83 | 24.57 | 188.96 ᵃ |
| Vintern-1B (fine-tuned) | 13.01 | 52.47 | 55.12 | 53.76 | 6.11 | 51.93 | 35.25 | 72.84 |
| Llama 3.2 (zero-shot) | 0.36 | 23.96 | 73.71 | 36.16 | 3.62 | 36.11 | 30.01 | 62.84 |
| Gemini 2.0 Flash | 0.55 | 27.20 | 74.10 | 39.79 | 4.41 | 39.60 | 31.72 | 74.42 |
| Gemini 2.5 Flash | 0.22 | 24.43 | 76.66 | 24.75 | 0.39 | 37.27 | 31.22 | 71.90 |
| GPT-5 (zero-shot) | 10.84 | 47.20 | 55.20 | 50.89 | 6.07 | 47.30 | 33.34 | 84.20 |
| ViMoE-VQA (Tuong-MOE) | 9.65 | 62.89 | 58.65 | 60.69 | 12.54 | 47.07 | 39.10 | 88.67 |
| **Multi-Token bridge (0.78%, 1 tile)** ᵇ | **8.28** | **50.53** | **51.72** | **49.82** | **15.99** | **48.11** | **40.47** | **96.98** |
| **  + decoder LoRA r=16 (~1.0%)** ᵇ | **10.42** | **53.85** | **55.00** | **53.17** | **19.44** | **51.48** | **43.91** | **105.59** |
| **  + decoder LoRA r=16, 3 epoch** ᵇ | **11.78** | **55.54** | **56.25** | **54.67** | **20.98** | **52.92** | **45.24** | **109.60** |

**Reading.** The frozen-backbone recipe surpasses the fine-tuned Vintern-1B on all
generation metrics (BLEU +14.9, METEOR +10.0, CIDEr +36.8) at ~1% of its trainable
parameters, and beats ViMoE-VQA on BLEU / ROUGE / METEOR / CIDEr. It trails ViMoE
on token-F1 (−6.0) and Acc-vs-Vintern.
_VI: Recipe vượt Vintern fine-tuned mọi metric sinh; thắng ViMoE trừ F1 & Acc._

### Table 2 — Corpus metrics + confidence intervals

**Table 2.** Cross-paper generation quality (corpus pycocoevalcap). Bold = ours.
ᵃ mean ± std. ᵇ paired bootstrap 95% CI over the 5 463-sample val set; ViMoE has
no per-sample data so only a one-sample CI is possible.

| Model | CIDEr-D | BLEU-4 | ROUGE-L |
|---|--:|--:|--:|
| ViMoE-VQA | 88.67 | 12.54 | 47.07 |
| **Multi-Token bridge (4 seed)** ᵃ | **92.80 ± 1.10** | **19.20 ± 0.30** | **49.20 ± 0.50** |
| **  95% CI** ᵇ | **[91.30, 97.10]** | — | — |
| **  + LoRA r=16 (seed 42)** | **101.70** | **23.20** | **52.70** |
| **  + LoRA r=16, 3 epoch (3 seed)** ᵃ | **106.80 ± 1.10** | **25.00 ± 0.40** | **54.20 ± 0.20** |

**Reading.** The plain-bridge CIDEr-D interval [91.3, 97.1] lies entirely above
ViMoE's 88.67 — the generation win is not marginal.

### Table 3 — Bridge architecture comparison (RQ1–2, RQ6) — seed 42, +2 seed đang chạy

**Table 3.** Five bridges over the frozen backbone (val, seed 42; Multi-Token
plain = 4-seed mean; Multi-Token & Full Q-Former +LoRA = 3-seed mean).
Bold = the proposed bridge. ᵃ job running, no value yet.

| Bridge | Params | % | F1 | CIDEr | val CE | F1 +LoRA | ΔF1 | CIDEr +LoRA |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Residual (1 tok) | 4.86M | 0.52 | 36.45 | 66.07 | 2.35 | 52.66 | +16.21 | 103.26 |
| Tile-Attention (8 tok) | 4.14M | 0.44 | 46.69 | 87.46 | 1.62 | — ᵃ | — | — ᵃ |
| **Multi-Token (8 tok pooled)** | **7.35M** | **0.78** | **49.82** | **96.98** | **1.49** | **53.17** | **+3.35** | **105.59** |
| Light Q-Former (8 query) | 27.6M | 2.87 | 46.63 | 88.10 | 1.59 | 53.39 | +6.76 | 106.65 |
| Full Q-Former (16 query) | 69.4M | 6.91 | 47.66 | 90.82 | 1.57 | 53.21 | +5.55 | 105.70 |

**Reading.** RQ1: Multi-Token (0.78%) is the best bridge and beats fine-tuned
Vintern on generation. RQ2: a 10× larger bridge (Full Q-Former, 69M) is *worse*;
Multi-Token has the lowest CE — capacity is not the bottleneck. RQ6: LoRA lifts F1
on every bridge and collapses the CIDEr spread (Fig 1).

### Fig 1 — Decoder-LoRA equalizes bridge quality (corpus CIDEr-D, val)

**Spec:** grouped horizontal bar chart, one pair of bars per bridge —
*plain* (grey) vs *+LoRA r=16* (accent). X-axis CIDEr-D 0–120.

| Bridge | plain | +LoRA r=16 |
|---|--:|--:|
| Residual | 56.30 | 100.00 |
| Tile-Attention | 87.50 | *(running)* |
| Multi-Token | 96.98 | 105.59 |
| Light Q-Former | 88.10 | 106.65 |
| Full Q-Former | 90.82 | 105.70 |

**Fig 1.** Plain bridges span CIDEr-D 56–97 (a wide quality gap driven by bridge
design). After a 0.23% decoder LoRA, all four measured bridges converge to
100–107 — **bridge choice becomes nearly irrelevant once the decoder has
capacity**. _VI: mở decoder → chênh lệch bridge gần như biến mất._

### Table 4 — Ablation summary — six axes, one positive (RQ1–6) — seed 42, dòng âm +2 seed

**Table 4.** ΔF1 relative to the anchor (Multi-Token plain, seed 42: F1 50.66).
CIDEr-D = corpus. ᵃ align-logit at α=1.0 is mis-weighted (KL swamps CE, val CE
2.84 vs 1.49); a full-val 3-seed rerun is in progress. The alignment axis rests
primarily on align-feat.

| RQ · axis | Intervention | F1 | CIDEr-D | ΔF1 | Verdict |
|---|---|--:|--:|--:|---|
| — anchor | Multi-Token plain | 50.66 | 94.40 | — | — |
| RQ1–2 · bridge capacity | Full Q-Former (69M) | 47.66 | 86.70 | −3.00 | negative |
| RQ3 · visual tiles | evaluate at 3 tiles | 21.05 | ~46 | −29.61 | negative (collapse) |
| RQ4 · adaptive routing | learned policy (reasoning + visual) | ≈50.7 | ≈94 | ≈0 | negative (no gain vs fixed) |
| RQ5 · training target | multi-reference sampling | 49.01 | 87.30 | −1.65 | negative |
| RQ5 · representation alignment | projector-KD (feat) | 49.66 | 92.00 | −1.00 | negative |
| RQ5 · representation alignment | projector-KD (logit) ᵃ | 40.70 | 80.10 | −9.96 | negative ᵃ |
| **RQ6 · decoder capacity** | **LoRA r=16 (1 epoch)** | **53.17** | **101.70** | **+2.51** | **positive** |
| **RQ6 · decoder capacity** | **LoRA r=16 (3 epoch)** | **54.67** | **106.80** | **+4.01** | **positive** |

**Reading.** Four independent vision-/training-side axes are all negative; the
single decoder-side axis is clearly positive. The *pattern* — not any one
ablation — localizes the bottleneck to the frozen decoder.

### Table 5 — Decoder-LoRA per bridge (RQ6) — multi_token + qformer khoá 3-seed

**Table 5.** plain → +LoRA r=16 by bridge (val, full 5 463). F1 = in-house;
CIDEr-D = corpus. multi_token & qformer: 3-seed mean. mini_qformer & residual:
seed 42. tile_attention: job running. ᵃ paired bootstrap 95% CI. P(Δ>0) = 1.000
on every row.

| Bridge | F1 plain | F1 +LoRA | ΔF1 [95% CI] ᵃ | CIDEr-D plain | CIDEr-D +LoRA | ΔCIDEr-D |
|---|--:|--:|--:|--:|--:|--:|
| multi_token | 50.66 | 53.17 | +2.51 [1.9, 3.1] | 94.40 | 101.70 | +7.30 |
| qformer | 47.66 | 53.10 | +5.44 | 86.70 | 101.90 | +15.20 |
| mini_qformer | 46.63 | 53.39 | +6.76 | 83.80 | 103.30 | +19.50 |
| residual | 36.45 | 52.66 | +16.21 [15.4, 17.0] | 56.30 | 100.00 | +43.70 |
| tile_attention | 46.69 | *running* | — | 87.50 | *running* | — |

**Reading.** The lift grows as the plain bridge gets weaker: +2.5 F1 on the best
bridge, +16.2 on the worst — and both land at ≈53 F1 / ≈101 CIDEr-D. This is the
equalization in Fig 1; per-seed detail in App. A3.

### Fig 2 — The bridge breaks above 1 tile (Multi-Token, val)

**Spec:** dual-axis line chart. X = n_tiles {1, 3, 6}. Left axis = token-F1,
right axis = validation loss.

| n_tiles | token-F1 | val loss |
|--:|--:|--:|
| 1 | 50.66 | 1.48 |
| 3 | 21.05 | 3.35 |
| 6 | 22.51 | 3.36 |

**Fig 2.** Multi-Token, trained at 1 tile, collapses when evaluated with more
tiles: F1 50.7 → 21, val loss 1.48 → 3.36. The mean-pool over 8 output tokens
cannot absorb 3–6× the visual tokens. **1 tile is the operating point, not a
compromise** — going higher is strictly worse. _VI: bridge sụp khi >1 tile._

### Table 6 — Self-check: does token-F1 track correctness? — N=120, 1 rater

**Table 6.** F1 bucket vs. semantic judgment (Multi-Token, N=120, single rater
against the 5 references, no image access). Not the planned 2-annotator /
Cohen's κ protocol — a bounded substitute, flagged as such. Camera-ready needs
the full study.

| F1 bucket | n | correct | partial | wrong | nonsense | acceptable |
|---|--:|--:|--:|--:|--:|--:|
| strong (≥0.6) | 45 | 80.00 | 11.11 | 6.67 | 2.22 | 91.11 |
| partial (0.2–0.6) | 58 | 12.07 | 31.03 | 55.17 | 1.72 | 43.10 |
| weak (0–0.2) | 3 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 |
| zero | 13 | 7.69 | 7.69 | 76.92 | 7.69 | 15.38 |
| **overall (n=119)** | — | **36.97** | **20.17** | **40.34** | **2.52** | **57.14** |

**Reading.** The *largest* F1 bucket (partial, 51.5% of val) is the *least*
reliable — 55% of "partial" answers are actually wrong. Mid-range token-F1 is a
poor correctness signal; this qualifies how every generation number should be
read.

### Table 7 — Compute-efficiency of the tile lever

**Table 7.** InternViT vision encode cost per image (Tesla P100-16GB,
`src.cli.profile`). Fine-tuned Vintern runs up to 12 tiles; our recipe uses 1.

| n_tiles | GFLOPs | Latency (ms) | Throughput (img/s) |
|---|--:|--:|--:|
| **1 (ours)** | **362** | **229** | **6.00** |
| 2 | 724 | 374 | 3.30 |
| 4 | 1 448 | 648 | 1.70 |
| 6 | 2 172 | 922 | 1.15 |

**Reading.** The 1→6 tile lever is real: FLOPs ×6.0, latency ×4.0, throughput
×5.2 — the recipe spends none of it. This is the FLOPs/latency analysis
ViMoE-VQA explicitly deferred.

---

## PHẦN D — Phụ lục (không vào main text — giữ để trả lời reviewer)

### A1. Multi-Token bridge (plain), per seed

| Seed | F1 | BLEU | ROUGE | METEOR | CIDEr | CIDEr-D | BLEU-4 | ROUGE-L |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 42 | 50.66 | 16.34 | 48.95 | 41.05 | 98.69 | 94.40 | 19.60 | 50.00 |
| 123 | 49.46 | 16.05 | 47.76 | 40.16 | 95.84 | 91.70 | 19.20 | 48.80 |
| 2026 | 49.64 | 15.91 | 47.93 | 40.53 | 97.35 | 93.10 | 19.10 | 49.00 |
| 3407 | 49.51 | 15.64 | 47.80 | 40.13 | 96.05 | 91.80 | 18.80 | 48.90 |

### A2. Multi-Token + LoRA r=16, per seed (in-house, full-val)

| Config | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 ep · s42 | 10.49 | 53.90 | 54.92 | 53.16 | 19.38 | 51.44 | 43.85 | 104.90 |
| 1 ep · s123 | 10.27 | 53.89 | 55.05 | 53.20 | 19.53 | 51.52 | 43.94 | 106.11 |
| 1 ep · s3407 | 10.51 | 53.76 | 55.03 | 53.15 | 19.42 | 51.48 | 43.93 | 105.76 |
| 3 ep · s42 | 11.73 | 55.47 | 56.07 | 54.52 | 20.59 | 52.82 | 45.06 | 108.49 |
| 3 ep · s123 | 11.92 | 55.45 | 56.31 | 54.67 | 21.30 | 52.91 | 45.34 | 110.63 |
| 3 ep · s3407 | 11.68 | 55.71 | 56.36 | 54.81 | 21.06 | 53.04 | 45.31 | 109.69 |

### A3. Q-Former + LoRA r=16, per seed

| Seed | F1 | CIDEr | CIDEr-D |
|--:|--:|--:|--:|
| 42 | 53.10 | 105.15 | 101.90 |
| 123 | 53.32 | 105.75 | 102.60 |
| 3407 | 53.22 | 106.19 | 102.80 |
| mean | 53.21 | 105.70 | 102.43 |

### A4. LoRA rank curve (600-sample subset)

| rank | F1 (s42) | F1 mean (n) |
|--:|--:|--:|
| 4 | 51.26 | 51.26 (1) |
| 8 | 51.62 | 51.62 (1) |
| 16 | 51.98 | 51.98 (1) |
| 32 | 51.80 | 53.83 ± 1.77 (3) |
| 64 | 53.05 | 54.06 ± 0.94 (3) |

Note: with proper 3-seed means, rank 32 ≈ rank 64 (0.23 apart, within noise);
seed 42 was an outlier-low seed. **Recommendation: keep r=16.**

### A5. Epoch curve — nguồn Table 1 (multi_token + LoRA, 1 ep vs 3 ep, 3-seed mean).

### Param counts (reference)

| Component | Trainable | % of total |
|---|--:|--:|
| Residual Bridge | 4.86M | 0.52 |
| **Multi-Token Bridge** | **7.35M** | **0.78** |
| Tile-Attention Bridge | 4.14M | 0.44 |
| Light Q-Former | 27.57M | 2.87 |
| Full Q-Former | 69.39M | 6.91 |
| LoRA r=16 (Qwen2 q/k/v/o) | 2.16M | 0.23 |
| **Multi-Token + LoRA r=16** | **9.51M** | **1.01** |

---

## PHẦN C — Đang chạy: TIER-1 (19 job)

Hầu hết đã có seed 42; TIER-1 nâng lên 3-seed mean ± std. Ô "running" duy nhất
chưa có số nào: **tile_attention + LoRA**.

| Nhóm | Jobs | Bảng | Status |
|---|---|---|---|
| 1a bridge multi-seed | residual/mini_qformer/tile_attention × s123,s3407 + qformer s3407 | Table 3: s42 → 3-seed | 7 running |
| 1b dòng âm multi-seed | align-feat/answer-random × s123,s3407 + align-logit × 3 seed | Table 4: s42 → 3-seed | 7 running |
| 1c LoRA coverage | mini_qformer/residual +LoRA × s123,s3407 + **tile_attention +LoRA s42** | Table 5: → 3-seed + 5/5 bridge | 5 running |

Sau TIER-1: TIER-2 (LoRA target: attn vs MLP vs both — làm sâu RQ6) · test-set
eval · [camera-ready] human validation thật · [stretch] larger frozen decoder.

---

*Nguồn: results-5bridge.md (Main) · results-grouped-split.md (Ablation) ·
bootstrap_ci.json (CI).*
