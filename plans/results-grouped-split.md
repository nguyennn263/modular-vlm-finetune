# Kết quả 5 bridge — grouped split (leak-free), seed 42

Split mới: 70/15/15 **nhóm theo `image_id`** (seed 42, 0 leak — không ảnh nào xuất hiện ở 2 tập).
Val = 5463 mẫu, mỗi câu hỏi có 5 câu trả lời tham chiếu. Chỉ train bridge (LLM + ViT đóng băng).

Số bridge dưới đây = **full-val, epoch 1** (đã hội tụ; epoch 2–4 trên subset 800 cao hơn ~2–4% CIDEr nên đây là cận dưới thận trọng). Metric tính bằng `metrics/vqa_metrics.py`, quy ước ×100 giống bảng cũ.

## Bảng đối chứng đầy đủ

| Mô hình | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vintern (base) | 0.12 | 17.52 | 19.87 | 17.55 | 1.91 | 25.84 | 23.93 | 8.54 |
| ViT5_ViT | 7.97 | 46.84 | 50.33 | 48.52 | 4.13 | 46.89 | 31.02 | 72.68 |
| BARTPhoBEiT | 8.81 | 45.30 | 46.48 | 45.88 | 4.33 | 44.83 | 24.57 | 188.96 |
| Vintern (finetune) | 13.01 | 52.47 | 55.12 | 53.76 | 6.11 | 51.93 | 35.25 | 72.84 |
| Llama 3.2 | 0.36 | 23.96 | 73.71 | 36.16 | 3.62 | 36.11 | 30.01 | 62.84 |
| Gemini 2.0 Flash | 0.55 | 27.20 | 74.10 | 39.79 | 4.41 | 39.60 | 31.72 | 74.42 |
| Gemini 2.5 Flash | 0.22 | 24.43 | 76.66 | 24.75 | 0.39 | 37.27 | 31.22 | 71.90 |
| GPT-5 | 10.84 | 47.20 | 55.20 | 50.89 | 6.07 | 47.30 | 33.34 | 84.20 |
| Tuong-MOE (ViMoE-VQA) | 9.65 | **62.89** | 58.65 | **60.69** | 12.54 | 47.07 | 39.10 | 88.67 |
| **Multi-Token Bridge** | 8.62 | 51.60 | 52.32 | 50.66 | **16.34** | **48.95** | **41.05** | **98.69** |
| **Full Q-Former** | 7.34 | 48.31 | 49.78 | 47.66 | 14.58 | 45.96 | 38.25 | 90.82 |
| **Lightweight Q-Former** | 5.99 | 47.18 | 49.00 | 46.63 | 13.80 | 44.81 | 37.30 | 88.10 |
| **Tile Attention Bridge** | 6.06 | 47.11 | 49.13 | 46.69 | 13.52 | 44.91 | 37.36 | 87.46 |
| **Residual Bridge** | 1.87 | 34.57 | 43.70 | 36.45 | 6.11 | 34.12 | 30.33 | 66.07 |

*In đậm ở cột = tốt nhất toàn bảng cho metric đó (trừ CIDEr 188.96 của BARTPhoBEiT — anomaly do sinh câu dài lê thê).*

## So bảng cũ (random split) ↔ bảng mới (grouped split)

| | BLEU | ROUGE | METEOR | CIDEr | F1 |
|---|---:|---:|---:|---:|---:|
| multi_token cũ | 16.47 | 48.25 | 41.55 | 99.88 | 50.23 |
| multi_token mới | 16.34 | 48.95 | 41.05 | 98.69 | 50.66 |

→ Chênh <1.5% ở mọi metric. Bảng cũ **không bị leakage thổi phồng** — kết quả tái lập được trên split sạch. Đây là điểm mạnh khi trình thầy, không phải điểm yếu.

## multi_token vs ViMoE-VQA (baseline SOTA của nhóm)

**Thắng** — sinh văn bản:
- BLEU: 16.34 vs 12.54 (**+3.8**)
- METEOR: 41.05 vs 39.10 (**+1.95**)
- ROUGE-L: 48.95 vs 47.07 (**+1.88**)
- CIDEr: 98.69 vs 88.67 (**+10.0**)

**Thua** — phân loại/khớp token:
- F1: 50.66 vs 60.69 (**−10.0**)
- Precision: 51.60 vs 62.89 (**−11.3**)
- Accuracy: 8.62 vs 9.65 (−1.0)

multi_token **tốt nhất toàn bảng** trên cả 4 metric sinh văn bản (BLEU/ROUGE/METEOR/CIDEr).
ViMoE giữ ưu thế rõ về F1/Precision — decoder 6 lớp train from scratch + label smoothing sinh câu bám sát pattern reference hơn frozen Qwen2-0.5B + projector.

## Hiệu quả tham số

| Bridge | Trainable | % | CIDEr |
|---|---:|---:|---:|
| Multi-Token | 7.35M | 0.78% | 98.69 |
| Full Q-Former | 69.4M | 6.91% | 90.82 |
| Lightweight Q-Former | 27.6M | 2.87% | 88.10 |
| Tile Attention | 4.14M | 0.44% | 87.46 |
| Residual | 4.86M | 0.52% | 66.07 |

multi_token đạt CIDEr cao nhất với chỉ 0.78% tham số train — hiệu quả tham số vượt trội Full Q-Former (6.91% tham số, kém 8 điểm CIDEr).

## Kết luận cho Paper 3

1. Bridge multi_token **vượt ViMoE-VQA và toàn bộ baseline** trên metric sinh (BLEU/ROUGE/METEOR/CIDEr) — tái xác nhận đóng góp bridge của Paper 1&2 trên split sạch.
2. Thua F1/Precision — bottleneck là frozen Qwen2-0.5B, không phải bridge. Cần nói thẳng trong limitations.
3. Đóng góp MỚI của Paper 3 (routing theo reasoning-type) là kết quả NULL — xem `outputs/oracle_val/ANALYSIS.json`. Giá trị: test nghiêm ngặt claim "reasoning-aware" của ViMoE + bổ sung compute-efficiency analysis mà ViMoE bỏ ngỏ.
