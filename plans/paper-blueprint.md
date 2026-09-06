# Bản thiết kế paper — "Cải thiện Vintern-1B cách rẻ"

> Cấu trúc theo paper AutoViVQA / ViMoE (cùng nhóm). Mọi bảng metric đầy đủ 8 cột
> (Acc / Prec / Rec / F1 / BLEU / ROUGE-L / METEOR / CIDEr, in-house ×100) để theo dõi.
> Cập nhật 2026-09-06. Ô đánh dấu **(TIER-1)** đang chạy, sẽ điền khi land.
>
> Khung câu chuyện: artifact "Cải Thiện Vintern-1B Cách Rẻ"
> (https://claude.ai/code/artifact/bb7bf7ee-d5f1-4749-bb56-29a5c5daa610)

---

## PHẦN A — Cấu trúc paper (LNCS, ~12–15 trang)

| § | Nội dung | Nguồn số liệu |
|---|---|---|
| **Abstract** | Vintern zero-shot hỏng (F1 17.6); finetune-toàn-bộ 53.8 nhưng ~100× chi phí. Recipe adapt rẻ: frozen backbone + bridge 0.78% + LoRA decoder 0.23% (~1.0% tổng), 1 tile → vượt finetune-toàn-bộ trên metric sinh. Chuỗi chẩn đoán 6 bước → nút thắt = frozen decoder. | — |
| **1 · Introduction** | Bối cảnh VLM Việt · Vintern-1B mạnh nhưng đắt để adapt · ViMoE chọn xây model mới · **câu hỏi: adapt rẻ được không, nút thắt ở đâu** · 4 đóng góp. | — |
| **2 · Related Work** | (a) VQA tiếng Việt: ViVQA, OpenViVQA, ViTextVQA, AutoViVQA, ViMoE · (b) frozen-backbone VLM + projector: BLIP-2, Frozen, LLaVA, "Inference-Optimal VLMs" (arXiv 2411.03312) · (c) parameter-efficient adaptation: LoRA, adapter, projector-only. | — |
| **3 · Method** | 3.1 Kiến trúc: InternViT-300M (frozen) + bridge + Qwen2-0.5B (frozen) · 3.2 Năm bridge (residual / multi-token / tile-attention / mini-QFormer / QFormer) · 3.3 Decoder-LoRA (q/k/v/o) như can thiệp có chủ đích · 3.4 Không gian chẩn đoán: 2 chỗ vặn × 6 câu hỏi. | — |
| **4 · Experimental Setup** | AutoViVQA · **grouped split 70/15/15 theo image_id (leak-free)** · 8 metric · corpus pycocoevalcap cho so cross-paper · baseline: 9 model (Bảng 1) · 4 seed, epoch 1, 1×A100/P100. | §4.1 |
| **5 · Main Results** | Recipe vs 9 baseline (8 metric) · bootstrap CI · compute-efficiency. | Bảng 1, 5d |
| **6 · Ablation: Truy tìm nút thắt** | 6.1 So 5 bridge (RQ1–2) · 6.2 Tile-sweep (RQ3) · 6.3 Oracle + routing (RQ4) · 6.4 Training signal + alignment (RQ5) · 6.5 Decoder-LoRA (RQ6): rank, epoch, bridge-equalizing · 6.6 Bảng tổng hợp 6-trục. | Bảng 2, 3, 4, 5a-c |
| **7 · Human Validation & Error Analysis** | Self-check N=120 (F1-bucket vs đúng-sai) · [camera-ready: 2 annotator, Cohen's κ] · phân tích lỗi per-category · noun-omission · độ dài sinh. | Bảng 6 |
| **8 · Discussion** | Frozen decoder là trần (không phải thị giác) · nối "Inference-Optimal VLMs" · "reasoning-aware" routing của ViMoE cần đo trực tiếp · giới hạn: 1 dataset, 1 backbone, self-check. | — |
| **9 · Conclusion** | Recipe rẻ + chẩn đoán 6-bước. Không xây model mới. Code + split + oracle table released. | — |

---

## PHẦN B — Toàn bộ bảng metric

### Bảng 1 — Kết quả chính: Recipe vs TOÀN BỘ baseline AutoViVQA

*val, in-house metric ×100. `multi_token` đã khoá 4 seed; LoRA khoá 3 seed.*

| Model | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vintern-1B (base, zero-shot) | 0.12 | 17.52 | 19.87 | 17.55 | 1.91 | 25.84 | 23.93 | 8.54 |
| ViT5_ViT | 7.97 | 46.84 | 50.33 | 48.52 | 4.13 | 46.89 | 31.02 | 72.68 |
| BARTPhoBEiT | 8.81 | 45.30 | 46.48 | 45.88 | 4.33 | 44.83 | 24.57 | 188.96¹ |
| **Vintern-1B (finetune toàn bộ, ≤12 tile)** | **13.01** | 52.47 | 55.12 | 53.76 | 6.11 | **51.93** | 35.25 | 72.84 |
| Llama 3.2 (zero-shot) | 0.36 | 23.96 | 73.71 | 36.16 | 3.62 | 36.11 | 30.01 | 62.84 |
| Gemini 2.0 Flash | 0.55 | 27.20 | 74.10 | 39.79 | 4.41 | 39.60 | 31.72 | 74.42 |
| Gemini 2.5 Flash | 0.22 | 24.43 | 76.66 | 24.75 | 0.39 | 37.27 | 31.22 | 71.90 |
| GPT-5 (zero-shot) | 10.84 | 47.20 | 55.20 | 50.89 | 6.07 | 47.30 | 33.34 | 84.20 |
| **ViMoE-VQA (Tuong-MOE, 5 seed)** | 9.65 | **62.89** | 58.65 | **60.69** | 12.54 | 47.07 | **39.10** | 88.67 |
| Ours · Residual Bridge (s42) | 1.87 | 34.57 | 43.70 | 36.45 | 6.11 | 34.12 | 30.33 | 66.07 |
| Ours · Full Q-Former (s42) | 7.34 | 48.31 | 49.78 | 47.66 | 14.58 | 45.96 | 38.25 | 90.82 |
| Ours · Light Q-Former (s42) | 5.99 | 47.18 | 49.00 | 46.63 | 13.80 | 44.81 | 37.30 | 88.10 |
| Ours · Tile-Attention (s42) | 6.06 | 47.11 | 49.13 | 46.69 | 13.52 | 44.91 | 37.36 | 87.46 |
| Ours · **Multi-Token Bridge** (0.78%, 1 tile), mean 4 seed | 8.28 | 50.53 | 51.72 | 49.82 | 15.99 | 48.11 | 40.47 | 96.98 |
| Ours · **★ Multi-Token + LoRA r=16** (~1.0% param), mean 3 seed | 10.42 | 53.85 | 55.00 | 53.17 | 19.44 | 51.48 | 43.91 | 105.59 |
| Ours · **★ Multi-Token + LoRA r=16, 3 epoch**, mean 3 seed | 11.78 | 55.54 | 56.25 | 54.67 | 20.98 | 52.92 | 45.24 | 109.60 |

¹ BARTPhoBEiT CIDEr là outlier (sinh câu dài), không so.
Baseline (9 dòng đầu): từ bảng AutoViVQA, độc lập split. Ours: grouped split leak-free.

**Đọc:** Recipe thắng generation (BLEU +7–9 / CIDEr +17–21 / ROUGE +5 so ViMoE) & Acc;
thua F1 token-level (−6 với 3-epoch). Corpus pycocoevalcap dưới đây.

#### Bảng 1b — corpus pycocoevalcap (chuẩn so cross-paper)

| Model | CIDEr-D | BLEU-4 | ROUGE-L |
|---|---:|---:|---:|
| ViMoE-VQA | 88.7 | 12.5 | 47.1 |
| Multi-Token Bridge (mean 4 seed) | **92.8 ± 1.1** | **19.2 ± 0.3** | **49.2 ± 0.5** |
| Multi-Token + LoRA r=16 (seed 42) | **101.7** | **23.2** | **52.7** |
| Multi-Token + LoRA r=16, 3 epoch (mean 3 seed) | **106.8 ± 1.1** | **25.0 ± 0.4** | **54.2 ± 0.2** |

Bootstrap: multi_token CIDEr-D 95% CI **[91.3, 97.1]** — hoàn toàn trên ViMoE 88.7.
Mọi cải thiện plain→LoRA significant (P(Δ>0) = 1.000).

#### Bảng 1c — Multi-Token plain, từng seed

| Seed | F1 | BLEU | ROUGE | METEOR | CIDEr | CIDEr-D* | BLEU-4* | ROUGE-L* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 50.66 | 16.34 | 48.95 | 41.05 | 98.69 | 94.4 | 19.6 | 50.0 |
| 123 | 49.46 | 16.05 | 47.76 | 40.16 | 95.84 | 91.7 | 19.2 | 48.8 |
| 2026 | 49.64 | 15.91 | 47.93 | 40.53 | 97.35 | 93.1 | 19.1 | 49.0 |
| 3407 | 49.51 | 15.64 | 47.80 | 40.13 | 96.05 | 91.8 | 18.8 | 48.9 |
| **mean ± std** | 49.82±.57 | 15.99±.29 | 48.11±.56 | 40.47±.43 | 96.98±1.3 | 92.8±1.1 | 19.2±.3 | 49.2±.5 |

`*` = corpus pycocoevalcap.

---

### Bảng 2 — So 5 bridge (RQ1–2) + LoRA (RQ6)

*seed 42; multi_token & qformer-LoRA: mean 3-seed. **(TIER-1)** = đang chạy thêm seed.*

| Bridge | Param | % | F1 plain | F1 +LoRA | ΔF1 | CIDEr plain | CIDEr +LoRA | val CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Residual (1 tok) | 4.86M | 0.52 | 36.45 | **52.66** | **+16.2** | 66.07 | **103.26** | 2.35 |
| Tile-Attention (8 tok) | 4.14M | 0.44 | 46.69 | *(TIER-1)* | — | 87.46 | *(TIER-1)* | 1.62 |
| Multi-Token (8 tok pooled) | 7.35M | 0.78 | 49.82 | 53.17 | **+3.4** | 96.98 | 105.59 | 1.49 |
| Light Q-Former (8 query) | 27.6M | 2.87 | 46.63 | **53.39** | **+6.8** | 88.10 | **106.65** | 1.59 |
| Full Q-Former (16 query) | 69.4M | 6.91 | 47.66 | **53.21** | **+5.4** | 90.82 | **105.70** | 1.57 |

**RQ1:** multi_token (0.78% param) tốt nhất, vượt Vintern-FT trên generation. Thua F1.
**RQ2:** Full Q-Former (69M, ×10 param) *tệ hơn* multi_token; multi_token CE thấp nhất → không phải capacity bridge.
**RQ6 / bridge-equalizing:** 5 bridge từ CIDEr 66–97 → sau LoRA đều 103–107. Chênh lệch gần như biến mất → capacity decoder, không phải độ tinh vi bridge, quyết định chất lượng.

---

### Bảng 3 — Ablation 6-RQ: bảng tổng hợp

*ΔF1 so anchor (multi_token plain seed 42: F1 50.7 / CIDEr-D 94.4). **(TIER-1)** dòng ÂM đang lên 3-seed.*

| RQ · trục | Can thiệp | F1 | CIDEr-D | ΔF1 | Verdict |
|---|---|---:|---:|---:|---|
| — anchor | multi_token plain | 50.7 | 94.4 | — | — |
| RQ1–2 · bridge capacity | Q-Former 69M param | 47.7 | 86.7 | −3.0 | **âm** |
| RQ3 · số tile | eval @ 3 tile | 21.1 | — | −29.6 | **âm (sụp)** |
| RQ4 · routing động | policy (reasoning + visual) | ≈50.7 | ≈94 | ≈0 | **âm** (không thắng fixed) |
| RQ5 · training target | answer-sampling = random | 49.0 | 90.6 | −1.7 | **âm** |
| RQ5 · alignment | align-feat KD α=1.0 | 49.7 | 96.4 | −1.0 | **âm** |
| **RQ6 · decoder capacity** | **LoRA r=16 (1 epoch)** | **53.2** | **101.7** | **+2.5** | **DƯƠNG** |
| **RQ6 · decoder capacity** | **LoRA r=16 (3 epoch)** | **54.7** | **106.8** | **+4.0** | **DƯƠNG** |

*align-logit α=1.0 mis-weighted (KL chèn CE); TIER-1 đang chạy 3-seed để chốt "âm nhất quán".*

---

### Bảng 4 — Decoder-LoRA (RQ6)

#### Bảng 4a — plain → +LoRA r=16, per bridge (seed 42, full-val)

*multi_token & qformer: mean 3-seed. mini_qf & residual: seed 42, **(TIER-1)** thêm 2 seed.*

| Bridge | F1 plain→LoRA | ΔF1 | ΔF1 95%CI | CIDEr-D plain→LoRA | ΔCIDEr-D | P(Δ>0) |
|---|---|---:|---|---|---:|---:|
| multi_token | 50.7 → 53.2 | +2.5 | [1.9, 3.1] | 94.4 → 101.7 | +7.3 | 1.000 |
| qformer | 47.7 → 53.1 | +5.4 | — | 86.7 → 101.9 | +15.2 | 1.000 |
| mini_qformer | 46.6 → 53.4 | +6.8 | — | 83.8 → 103.3 | +19.5 | 1.000 |
| **residual** (tệ nhất) | 36.5 → 52.7 | **+16.2** | [15.4, 17.0] | 56.3 → 100.0 | **+43.7** | 1.000 |
| tile_attention | 46.7 → *(TIER-1)* | — | — | 87.5 → *(TIER-1)* | — | — |

#### Bảng 4b — multi_token + LoRA r=16, từng seed đầy đủ 8 metric

| Config | Acc | EM | P | R | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 epoch · s42 | 10.49 | 11.17 | 53.90 | 54.92 | 53.16 | 19.38 | 51.44 | 43.85 | 104.90 |
| 1 epoch · s123 | 10.27 | 10.87 | 53.89 | 55.05 | 53.20 | 19.53 | 51.52 | 43.94 | 106.11 |
| 1 epoch · s3407 | 10.51 | 11.04 | 53.76 | 55.03 | 53.15 | 19.42 | 51.48 | 43.93 | 105.76 |
| **1 epoch · mean ± std** | 10.42±.13 | 11.03±.15 | 53.85±.08 | 55.00±.07 | 53.17±.03 | 19.44±.08 | 51.48±.04 | 43.91±.05 | 105.59±.62 |
| 3 epoch · s42 | 11.73 | 12.41 | 55.47 | 56.07 | 54.52 | 20.59 | 52.82 | 45.06 | 108.49 |
| 3 epoch · s123 | 11.92 | 12.65 | 55.45 | 56.31 | 54.67 | 21.30 | 52.91 | 45.34 | 110.63 |
| 3 epoch · s3407 | 11.68 | 12.41 | 55.71 | 56.36 | 54.81 | 21.06 | 53.04 | 45.31 | 109.69 |
| **3 epoch · mean ± std** | 11.78±.13 | 12.49±.14 | 55.54±.14 | 56.25±.15 | 54.67±.15 | 20.98±.36 | 52.92±.11 | 45.24±.16 | 109.60±1.1 |

#### Bảng 4c — qformer + LoRA r=16, 3-seed

| Seed | Acc | F1 | BLEU | ROUGE | METEOR | CIDEr | CIDEr-D* |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 10.91 | 53.10 | 19.33 | 51.58 | 43.66 | 105.15 | 101.9 |
| 123 | 10.65 | 53.32 | 19.79 | 51.59 | 43.80 | 105.75 | 102.6 |
| 3407 | 10.80 | 53.22 | 19.76 | 51.55 | 44.14 | 106.19 | 102.8 |
| **mean ± std** | 10.79±.13 | 53.21±.11 | 19.63±.26 | 51.57±.02 | 43.87±.25 | 105.70±.52 | 102.4±.5 |

---

### Bảng 5 — Tile-sweep · rank curve · epoch curve · compute

#### Bảng 5a — Tile-sweep: multi_token @ N tile (full-val) — RQ3

| n_tiles | F1 | BLEU | ROUGE | METEOR | CIDEr | val loss | PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| **1** | **50.66** | 16.34 | 48.95 | 41.05 | 98.69 | 1.478 | 4.4 |
| 3 | 21.05 | 1.63 | 15.97 | 26.10 | 48.75 | 3.351 | 28.5 |
| 6 | 22.51 | 1.76 | 17.14 | 27.80 | 52.36 | 3.364 | 28.9 |
| 12 | *bị Kaggle cắt ở 12h — không cần (trend 1→3→6 rõ)* | | | | | | |

Pool 8-token xoá tín hiệu khi có 3–6× token. "1 tile" là điểm vận hành đúng.

#### Bảng 5b — Rank curve (multi_token + LoRA, 600-mẫu subset lúc train)

| rank | F1 (seed 42) | F1 mean (n seed) | CIDEr subset | val loss |
|---|---:|---:|---:|---:|
| 4 | 51.26 | 51.26 (1) | 103.5 | 1.410 |
| 8 | 51.62 | 51.62 (1) | 106.0 | 1.408 |
| 16 | 51.98 | 51.98 (1) | 106.4 | 1.371 |
| 32 | 51.80 | 53.83 ± 1.77 (3) | — | 1.368 |
| 64 | 53.05 | 54.06 ± 0.94 (3) | — | 1.366 |

Đã RÚT LẠI "rank cao hơn tốt hơn": khi 3-seed, r=32 ≈ r=64 (chênh 0.23, trong nhiễu).
seed 42 là seed thấp bất thường. → giữ **r=16** làm điểm chính.

#### Bảng 5c — Epoch curve (multi_token + LoRA r=16, full-val, mean 3-seed)

| | F1 | CIDEr-D* | BLEU-4* | ROUGE-L* | val loss |
|---|---:|---:|---:|---:|---:|
| 1 epoch | 53.17 ± .03 | 101.7 | 23.2 | 52.7 | ~1.37 |
| **3 epoch** | 54.67 ± .15 | 106.8 ± 1.1 | 25.0 ± .4 | 54.2 ± .2 | ~1.32 |

Train thêm giúp nhẹ (+1.5 F1); bão hoà từ epoch 2. 1 epoch bắt ~80% lợi ích. Gap F1 tới ViMoE: 7.5 → 6.0.

#### Bảng 5d — Compute-efficiency (InternViT, P100, per image)

| n_tiles | GFLOPs | Latency (ms) | Throughput (img/s) |
|---|---:|---:|---:|
| **1 (ours)** | 362 | 229 | 6.0 |
| 2 | 724 | 374 | 3.3 |
| 4 | 1 448 | 648 | 1.7 |
| 6 | 2 172 | 922 | 1.15 |

Vintern-FT chạy ≤12 tile. Ours ở 1 tile: FLOPs ×6, latency ×4, throughput ×5 rẻ hơn — bảng ViMoE bỏ ngỏ.

---

### Bảng 6 — Self-check (§7): F1 bucket vs phán đoán ngữ nghĩa

*multi_token, N=120, 1 rater. [camera-ready: 2 annotator + Cohen's κ]*

| F1 bucket | n | đúng | 1 phần | sai | vô nghĩa | chấp nhận được |
|---|---:|---:|---:|---:|---:|---:|
| strong (≥0.6) | 45 | 80.0% | 11.1% | 6.7% | 2.2% | **91.1%** |
| partial (0.2–0.6) | 58 | 12.1% | 31.0% | 55.2% | 1.7% | **43.1%** |
| weak (0–0.2) | 3 | 0% | 0% | 100% | 0% | 0% |
| zero | 13 | 7.7% | 7.7% | 76.9% | 7.7% | 15.4% |
| **tổng (n=119)** | | **37.0%** | **20.2%** | **40.3%** | **2.5%** | **57.1%** |

**Phát hiện:** bucket "partial" — lớn nhất (51.5% val) — chỉ 43% chấp nhận được → token-F1 tầm trung không đáng tin.

---

## PHẦN C — Đang chạy: TIER-1 (19 job)

| Nhóm | Jobs | Điền vào bảng | Status |
|---|---|---|---|
| 1a bridge multi-seed | residual/mini_qformer/tile_attention × s123,s3407 + qformer s3407 | Bảng 2 → 3-seed | 7 running |
| 1b dòng ÂM multi-seed | align-feat/answer-random × s123,s3407 + align-logit × 3 seed | Bảng 3 → 3-seed | 7 running |
| 1c LoRA coverage | mini_qformer/residual +LoRA × s123,s3407 + tile_attention +LoRA s42 | Bảng 4 → 3-seed + 5/5 bridge | 5 running |

**Sau TIER-1:** TIER-2 (LoRA target: attn vs MLP vs cả hai — làm sâu RQ6) · test-set eval ·
[camera-ready] human validation thật · [stretch] decoder frozen to hơn (G7).

---

## Số param (tham chiếu)

| Thành phần | Trainable | % total |
|---|---:|---:|
| Residual Bridge | 4.86M | 0.52% |
| **Multi-Token Bridge** | **7.35M** | **0.78%** |
| Tile Attention Bridge | 4.14M | 0.44% |
| Lightweight Q-Former | 27.57M | 2.87% |
| Full Q-Former | 69.39M | 6.91% |
| LoRA r=16 (Qwen2 q/k/v/o) | 2.16M | +0.23% |
| **Multi-Token + LoRA r=16** | **9.51M** | **1.01%** |

---

*Nguồn: plans/results-5bridge.md (Main Results) · plans/results-grouped-split.md (Ablation) ·
outputs/bootstrap_ci.json (CI) · outputs/parallel/ledger.json (job tracking)*
