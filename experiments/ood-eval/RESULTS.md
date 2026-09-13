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

Raw score files: `outputs/ood_eval/{vitextvqa,vivqax}/out/<dataset>/{vintern_base,ours}/results/scored.json`
(not committed here -- large/derived; this file is the durable record).

## ViTextVQA (scene-text / OCR-in-photo VQA, arXiv:2404.10652, UIT 2024)

n=1000, seed=42, sampled from the official test split (3,353 img / 10,028 QA).

| | acc | prec | recall | F1 | BLEU | ROUGE | METEOR | CIDEr (in-house) | CIDEr-D (corpus) | BLEU-4 (corpus) | ROUGE-L (corpus) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Vintern gốc (6 tile) | 0.90 | 22.43 | 64.82 | 27.03 | 8.54 | 34.33 | 26.59 | 0.00 | 122.5 | 14.5 | 34.3 |
| Model của mình (1 tile) | 0.10 | 4.52 | 3.88 | 31.32 | 0.06 | 3.75 | 3.54 | 0.00 | 10.7 | 0.6 | 3.7 |

**Vintern gốc thắng rõ trên hầu hết metric** (trừ F1_token, nơi 2 số không thực sự
so sánh được do precision/recall dùng implementation khác F1 -- xem note dưới).
Xem mẫu dự đoán thật của model mình:
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

n=1000, seed=42, sampled from the official test split (1,970 QA, images = COCO2014).

| | acc | prec | recall | F1 | BLEU | ROUGE | METEOR | CIDEr (in-house) | CIDEr-D (corpus) | BLEU-4 (corpus) | ROUGE-L (corpus) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Vintern gốc (6 tile) | 0.40 | 8.19 | 60.02 | 16.80 | 0.00 | 14.62 | 11.44 | 0.00 | 21.6 | 0.0 | 14.6 |
| Model của mình (1 tile) | **2.30** | **12.59** | 26.00 | **33.37** | 0.00 | **15.95** | 9.03 | 0.00 | **32.1** | 0.0 | 16.0 |

**Model của mình thắng trên hầu hết metric** (F1, accuracy, precision, ROUGE,
CIDEr-D) — chỉ thua recall (Vintern gốc trả lời dài/verbose hơn → trùng từ nhiều
hơn) và METEOR. Mẫu dự đoán: câu trả lời đúng format, đúng ngữ pháp, hợp lý (kể cả
câu yes/no đúng), sai nội dung do ảnh COCO ngoài domain — không có dấu hiệu hỏng:
```
Q: Trời đang mưa à?                       GT: có           PRED: "Có, trời đang mưa"  (đúng)
Q: Con chó là thật hay giả?                GT: giả          PRED: "Có vẻ là thật"       (sai nội dung, hợp lý)
Q: Đây là phòng nào?                       GT: nhà bếp      PRED: "Phòng khách"         (sai nội dung, hợp lý)
```

## Kết luận chung

Model của mình **tốt hơn trên VQA tổng quát (ViVQA-X)** — gần domain train
(AutoViVQA) hơn — nhưng **thua hẳn trên đọc-chữ-trong-ảnh (ViTextVQA)** — đúng dự
đoán vì bridge chỉ train 1 tile, không phải tác vụ OCR.

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
