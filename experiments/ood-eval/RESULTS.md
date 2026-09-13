# OOD eval results — Vintern-1B-v3_5 zero-shot vs our best checkpoint

Methodology: see `build_ood_data.py` / `gen_vintern_base.py` / `eval_ours_ood.py` in
this dir. Both models scored with the SAME code (`metrics.compute_score.compute_all_data`
+ pycocoevalcap-style corpus CIDEr-D/BLEU-4/ROUGE-L, via
`experiments/vintern-ft/score_local.py`) on the SAME 1000-sample seeded (seed=42)
subset of each dataset's official test split (ViVQA-X, ViTextVQA) -- neither model
has ever trained on either dataset.

- **Vintern gốc**: `5CD-AI/Vintern-1B-v3_5`, zero-shot, 6 tiles, greedy, max_new_tokens=64.
- **Model của mình**: Multi-Token bridge + decoder LoRA r16 (3 epoch, seed42,
  `checkpoints/expA-lora16-3ep/seed42/multi_token/last_model.pt`), 1 tile, same
  generation settings.

Raw score files: `outputs/ood_eval/<dataset>[_s<seed>]/out/.../{vintern_base,ours}/results/scored.json`
(not committed here -- large/derived; this file is the durable record).

Multi-seed (42/123/3407): the seed controls which 1000 rows get SAMPLED from
each split; neither model is retrained (Vintern gốc zero-shot, ours a fixed
checkpoint) -- this is purely to get a mean±std across sampling variance,
matching this project's usual multi-seed convention.

## ViTextVQA (scene-text / OCR-in-photo VQA, arXiv:2404.10652, UIT 2024)

n=1000/seed, sampled from the official test split (3,353 img / 10,028 QA).

**Vintern gốc (6 tile):**

| seed | acc | prec | recall | F1 | CIDEr-D (corpus) | BLEU-4 (corpus) | ROUGE-L (corpus) |
|---|--:|--:|--:|--:|--:|--:|--:|
| 42 | 0.90 | 22.43 | 64.82 | 27.03 | 122.5 | 14.5 | 34.3 |
| 123 | 0.50 | 22.29 | 62.29 | 26.84 | 127.6 | 15.0 | 33.6 |
| 3407 | 0.60 | 22.60 | 65.11 | 26.51 | 130.8 | 14.7 | 34.3 |
| **mean±std (n=3)** | **0.67±0.21** | **22.44±0.16** | **64.07±1.55** | **26.79±0.26** | **127.0±4.2** | **14.7±0.3** | **34.1±0.4** |

**Model của mình (1 tile):**

| seed | acc | prec | recall | F1 | CIDEr-D (corpus) | BLEU-4 (corpus) | ROUGE-L (corpus) |
|---|--:|--:|--:|--:|--:|--:|--:|
| 42 | 0.10 | 4.52 | 3.88 | 31.32 | 10.7 | 0.6 | 3.7 |
| 123 | 0.10 | 4.86 | 4.31 | 31.25 | 12.9 | 1.2 | 4.1 |
| 3407 | 0.10 | 4.64 | 4.12 | 31.39 | 12.1 | 0.8 | 4.0 |
| **mean±std (n=3)** | **0.10±0.00** | **4.67±0.17** | **4.10±0.22** | **31.32±0.07** | **11.9±1.1** | **0.9±0.3** | **3.9±0.2** |

**Vintern gốc thắng rõ trên hầu hết metric, ổn định qua cả 3 seed** (std nhỏ so
với khoảng cách giữa 2 model — không phải nhiễu ngẫu nhiên). Ngoại lệ F1_token
(31.3 vs 26.8) không có ý nghĩa so sánh — xem note dưới (precision/recall/F1
không cùng công thức). Xem mẫu dự đoán thật của model mình (seed 42):
```
Q: mức giá khuyến mãi đầu của bảng đầu tiên là bao nhiêu?  GT: 32500          PRED: "1000 đồng"
Q: số điện thoại nơi này là gì?                            GT: 08.3722.0539  PRED: "1111111111111111"
Q: chùa do đơn vị nào quản lý?                             GT: giáo hội phật giáo việt nam  PRED: "Cổng chùa"
```
→ Model của mình KHÔNG đọc chữ trong ảnh -- chỉ đoán mò kiểu caption chung chung.
Hợp lý: chỉ train trên AutoViVQA (không phải OCR task) ở **1 tile** (độ phân giải
thấp hơn nhiều so với 6 tile của Vintern gốc) → mất khả năng đọc text nhỏ. Đây là
một trade-off thật (generalization gap), không phải bug.

## ViVQA-X (VQA tự do tổng quát + giải thích, Springer ICISN 2025, VLAI-AIVN)

n=1000/seed, sampled from the official test split (1,970 QA, images = COCO2014).

**Vintern gốc (6 tile):**

| seed | acc | prec | recall | F1 | CIDEr-D (corpus) | BLEU-4 (corpus) | ROUGE-L (corpus) |
|---|--:|--:|--:|--:|--:|--:|--:|
| 42 | 0.40 | 8.19 | 60.02 | 16.80 | 21.6 | 0.0 | 14.6 |
| 123 | 0.20 | 8.26 | 61.48 | 16.87 | 21.3 | 0.2 | 14.9 |
| 3407 | 0.50 | 9.02 | 61.90 | 17.55 | 26.4 | 0.2 | 15.9 |
| **mean±std (n=3)** | **0.37±0.15** | **8.49±0.46** | **61.13±0.99** | **17.07±0.41** | **23.1±2.9** | **0.1±0.1** | **15.2±0.7** |

**Model của mình (1 tile):**

| seed | acc | prec | recall | F1 | CIDEr-D (corpus) | BLEU-4 (corpus) | ROUGE-L (corpus) |
|---|--:|--:|--:|--:|--:|--:|--:|
| 42 | 2.30 | 12.59 | 26.00 | 33.37 | 32.1 | 0.0 | 16.0 |
| 123 | 2.50 | 12.77 | 26.38 | 33.57 | 32.6 | 0.0 | 16.3 |
| 3407 | 2.50 | 13.54 | 28.48 | 34.09 | 35.0 | 0.6 | 17.3 |
| **mean±std (n=3)** | **2.43±0.12** | **12.97±0.50** | **26.95±1.34** | **33.68±0.37** | **33.2±1.6** | **0.2±0.4** | **16.5±0.7** |

**Model của mình thắng trên hầu hết metric, ổn định qua cả 3 seed** (F1 33.7 vs
17.1, accuracy 2.4 vs 0.4, CIDEr-D 33.2 vs 23.1) — chỉ thua recall (Vintern gốc
trả lời dài/verbose hơn → trùng từ nhiều hơn). Mẫu dự đoán (seed 42): câu trả lời
đúng format, đúng ngữ pháp, hợp lý (kể cả câu yes/no đúng), sai nội dung do ảnh
COCO ngoài domain — không có dấu hiệu hỏng:
```
Q: Trời đang mưa à?                       GT: có           PRED: "Có, trời đang mưa"  (đúng)
Q: Con chó là thật hay giả?                GT: giả          PRED: "Có vẻ là thật"       (sai nội dung, hợp lý)
Q: Đây là phòng nào?                       GT: nhà bếp      PRED: "Phòng khách"         (sai nội dung, hợp lý)
```

## OpenViVQA (Information Fusion 2023, arXiv:2305.04183, UIT) -- seed 42 only so far

n=1000, seed=42, sampled from the **dev** split (test split's answers are a
placeholder, see build_ood_data.py). Street-scene photos, ~44% of QA require
reading embedded scene text per the paper (hybrid of plain VQA + OCR).

| | acc | prec | recall | F1 | CIDEr-D (corpus) | BLEU-4 (corpus) | ROUGE-L (corpus) |
|---|--:|--:|--:|--:|--:|--:|--:|
| Vintern gốc (6 tile) | 14.30 | 54.14 | 72.63 | 32.63 | 388.6 | 38.2 | 60.0 |
| Model của mình (1 tile) | 1.30 | 34.96 | 18.56 | 31.95 | 63.4 | 4.5 | 21.4 |

Vintern gốc thắng rõ (CIDEr-D 388.6 vs 63.4) -- nhất quán với ViTextVQA: OpenViVQA
cũng cần đọc chữ trong ảnh ở ~44% câu hỏi, nơi 1 tile của model mình bất lợi.
Seeds 123/3407 đang chạy, sẽ cập nhật mean±std khi xong.

## Kết luận chung

Model của mình **tốt hơn trên VQA tổng quát, ảnh không cần đọc chữ (ViVQA-X)** —
gần domain train (AutoViVQA) hơn — nhưng **thua hẳn trên các tập cần đọc chữ
trong ảnh (ViTextVQA, OpenViVQA)** — đúng dự đoán vì bridge chỉ train 1 tile,
không phải tác vụ OCR, và kết quả ổn định qua nhiều seed (không phải may rủi).

CIDEr (in-house, cột "cider") = 0.00 cho **cả 4 hàng** — đây là hệ quả toán học tất
yếu, không phải model tệ: `compute_score.py`'s `cider_score()` gọi `Cider().compute_score()`
riêng cho TỪNG sample (corpus size = 1 lúc tính IDF), nên `ref_len = log(len(crefs)) = log(1) = 0`
→ trọng số tf-idf luôn suy biến về 0. Metric CIDEr chỉ có ý nghĩa ở mức **corpus**
(toàn tập), đúng như cách Table 1 chính của paper dùng — cột "CIDEr-D (corpus)" ở
trên mới phản ánh đúng. BLEU (in-house, sentence-level) cũng gần 0 ở hầu hết các
hàng vì câu trả lời VQA ngắn (1-3 từ) hiếm khi đạt đủ 4-gram trùng khớp — cùng lý
do dùng BLEU-4 (corpus) làm số chính.

precision/recall/F1 không khớp công thức 2pr/(p+r) vì `metrics/f1/f1.py` tính F1
độc lập (max qua các references) chứ không suy từ 2 cột precision/recall riêng
(`metrics/precision`, `metrics/recall`) -- pattern có sẵn của project, không phải
lỗi mới.
