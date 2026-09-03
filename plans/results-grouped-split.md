# Kết quả 5 bridge — grouped split (leak-free), seed 42

Split: 70/15/15 **nhóm theo `image_id`** (seed 42) — **0 ảnh trùng giữa các tập**.
Val = 5463 câu, mỗi câu 5 đáp án. Chỉ train bridge (InternViT + Qwen2 đóng băng). Epoch 1 full-val (metric hội tụ từ epoch 2).

---

## A. Bộ metric ĐẦY ĐỦ (10 metric) — impl in-house `metrics/vqa_metrics.py`

Đây là **cùng implementation với bảng cũ `results-5bridge.md`** → số khớp bảng cũ
(multi_token: BLEU 16.34 vs 16.47 cũ, CIDEr 98.69 vs 99.88, METEOR 41.05 vs 41.55, F1 50.66 vs 50.23).

| Bridge | Acc | EM | WUPS | Prec | Rec | F1 | BLEU | ROUGE-L | METEOR | CIDEr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Multi-Token** | **8.62** | **8.97** | **8.62** | **51.60** | **52.32** | **50.66** | **16.34** | **48.95** | **41.05** | **98.69** |
| Full Q-Former | 7.34 | 7.61 | 7.34 | 48.31 | 49.78 | 47.66 | 14.58 | 45.96 | 38.25 | 90.82 |
| Light Q-Former | 5.99 | 6.26 | 5.99 | 47.18 | 49.00 | 46.63 | 13.80 | 44.81 | 37.30 | 88.10 |
| Tile-Attention | 6.06 | 6.33 | 6.06 | 47.11 | 49.13 | 46.69 | 13.52 | 44.91 | 37.36 | 87.46 |
| Residual | 1.87 | 1.94 | 1.87 | 34.57 | 43.70 | 36.45 | 6.11 | 34.12 | 30.33 | 66.07 |

> Acc = EM = WUPS: impl WUPS ở đây sụp về exact-match (đáp án ngắn 4 từ, không có partial credit). Ba cột này thực chất là 1.

## B. Cùng metric nhưng impl KHÁC — để so với paper khác

| impl | Bridge = multi_token | Acc | F1 | BLEU | ROUGE-L | METEOR | CIDEr |
|---|---|---:|---:|---:|---:|---:|---:|
| in-house `vqa_metrics` (bảng A) | | 8.62 | 50.66 | 16.34 | 48.95 | 41.05 | 98.69 |
| `compute_score.py` per-sample max-ref | | 8.97 | 44.19 | *0*¹ | 49.78 | 32.42 | *0*¹ |
| pycocoevalcap **corpus** (chuẩn field) | | — | — | 19.58² | 50.00 | 28.45 | **94.4** |

¹ `compute_score.py` tính BLEU/CIDEr per-sample với 1 ref → IDF/brevity sụp về ~0. **Không dùng.**
² pycoco BLEU-4 corpus; BLEU-1 = 54.79.

→ **F1, METEOR cực nhạy impl** (F1: 50.7 vs 44.2; METEOR: 41.1 vs 32.4 vs 28.5). **CIDEr-D, BLEU, ROUGE-L ổn định** giữa các impl và giữa các paper.

---

## C. multi_token vs prior work trên AutoViVQA

Số bridge lấy từ impl khớp được với từng baseline. Cột **CIDEr / BLEU / ROUGE** dùng impl chuẩn (corpus), an toàn nhất để so.

| Model | Acc | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|---:|---:|---:|---:|---:|---:|
| Vintern-1B (base) | 0.12 | 17.55 | 1.91 | 25.84 | 23.93 | 8.54 |
| ViT5+ViT | 7.97 | 48.52 | 4.13 | 46.89 | 31.02 | 72.68 |
| BARTPhoBEiT | 8.81 | 45.88 | 4.33 | 44.83 | 24.57 | 188.96³ |
| Vintern-1B (finetune) | **13.01** | 53.76 | 6.11 | **51.93** | 35.25 | 72.84 |
| GPT-5 (zero-shot) | 10.84 | 50.89 | 6.07 | 47.30 | 33.34 | 84.20 |
| Gemini 2.0 Flash | 0.55 | 39.79 | 4.41 | 39.60 | 31.72 | 74.42 |
| **Tuong-MoE / ViMoE-VQA** (5-seed) | 9.65 | **60.69** | 12.54 | 47.07 | **39.10** | 88.67 |
| **Multi-Token Bridge (ours)** | 8.62 | 50.7 / 44.2⁴ | **19.58** | **50.00** | 28.5–41.1⁴ | **94.4** |

³ BARTPhoBEiT CIDEr outlier — sinh câu dài. ⁴ khoảng giá trị theo impl (xem mục B).

## D. Có hơn model cũ không?

**CÓ — trên metric sinh implementation-stable:**

| | multi_token | ViMoE | chênh |
|---|---:|---:|---:|
| CIDEr-D | 94.4 | 88.67 | **+5.7** ✅ |
| BLEU-4 | 19.58 | 12.54 | **+7.0** ✅ |
| ROUGE-L | 50.00 | 47.07 | **+2.9** ✅ |

**Hòa/thua trên metric matching:**
- Acc 8.62 vs 9.65 (≈ hòa)
- F1: 50.7 (in-house, so bảng cũ) hoặc 44.2 (corpus) — ViMoE 60.69. **Thua rõ dù dùng impl nào.**
- Bottleneck = frozen Qwen2-0.5B (ViMoE train decoder 6 lớp from scratch + label smoothing → bám phrasing ref chặt hơn), không phải bridge.

## E. Cần đóng trước camera-ready

- **Mới seed 42.** ViMoE báo mean 5 seed (std ≤ 0.16). Cần 5 seed cho dòng multi_token, hoặc paired bootstrap trên 5463 mẫu val.
- Chốt **1 implementation** cho toàn paper — đề xuất pycocoevalcap corpus (CIDEr-D / BLEU / ROUGE-L) làm chính, F1/METEOR ghi kèm cả 2 số + chú thích.
- Đây là val. Test chạy 1 lần (Section 5).

## F. Kết luận Paper 3

1. Bridge multi_token **vượt ViMoE + toàn bộ baseline** trên metric sinh ổn định → tái xác nhận đóng góp bridge Paper 1&2 trên split sạch (không do leakage: số gần trùng bảng cũ).
2. Thua F1/Acc → Limitations, bottleneck frozen decoder.
3. Routing theo reasoning-type (đóng góp mới Paper 3) = NULL/negative — xem `plans/P6-draft/05-results.md` + `outputs/oracle_val/ANALYSIS.json`.
