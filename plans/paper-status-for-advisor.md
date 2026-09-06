# Paper 3 — Hiện trạng để báo cáo thầy

*Cập nhật 2026-09-06. Deadline ~2026-09-27 (ACIIDS 2027 / Trust4NLP).*
*Bản chi tiết (đầy đủ 7 bảng + 2 hình + phụ lục per-seed): `plans/paper-blueprint.md`.*

---

## 1. Câu hỏi nghiên cứu

> Thay vì **xây một model mới** (như ViMoE-VQA), có thể **cải thiện Vintern-1B**
> trên AutoViVQA bằng cách **chỉ cập nhật một phần nhỏ tham số** (~1% tổng,
> đóng băng toàn bộ backbone) để đạt ngang mức fine-tune không? Nếu chưa đạt,
> **nút thắt nằm ở đâu**?

**Định vị:**

| | Cách làm | Tham số được cập nhật |
|---|---|---|
| Vintern-1B (fine-tuned) | Full fine-tune InternViT-300M + projector; Qwen2-0.5B dùng LoRA. 3M cặp, 4×RTX-3090 | phần lớn tham số phía thị giác + projector |
| ViMoE-VQA | Thiết kế kiến trúc MoE **mới** | toàn bộ model mới |
| **Của chúng tôi** | Đóng băng **cả** InternViT-300M lẫn Qwen2-0.5B, chỉ train **bridge 0.78%** + **LoRA decoder 0.23%**, 1 tile | **~1% tổng tham số** |

*Lưu ý: hiện mới so được theo **số tham số train được** và **phạm vi đóng băng
backbone**; chưa đo training time / GPU memory nên chưa khẳng định con số "rẻ hơn
N lần" về chi phí huấn luyện.*

---

## 2. Trả lời hiện tại

Trong các can thiệp đã khảo sát, **tăng capacity ở phía thị giác** (bridge lớn
hơn, nhiều tile hơn, routing thích ứng) **không đem lại cải thiện đáng kể**.
**Decoder là trục duy nhất cho thấy tín hiệu cải thiện rõ ràng** trong các thí
nghiệm hiện tại. → Công thức: **bridge pooling rẻ (cố định) + LoRA decoder nhẹ.**

Recipe này **đạt hoặc vượt Vintern-1B fine-tuned trên các metric sinh**
(BLEU +14.9, METEOR +10.0, CIDEr +36.8), **trong khi chỉ cập nhật ~1% tổng số
tham số và đóng băng toàn bộ backbone** — giảm đáng kể chi phí adaptation. So với
ViMoE-VQA: cao hơn ở BLEU / ROUGE / METEOR / CIDEr, thấp hơn ở token-F1 (−6.0) —
chính khoảng cách F1 này dẫn tới phần chẩn đoán.

---

## 3. Cấu trúc paper (LNCS, 12–15 trang)

| § | Nội dung |
|---|---|
| 1 · Introduction | ViMoE xây model mới · Vintern train nặng phía thị giác · câu hỏi: adapt tiết kiệm tham số được không, nút thắt ở đâu · 3 đóng góp |
| 2 · Related Work | VQA tiếng Việt · frozen-backbone + projector (BLIP-2, "Inference-Optimal VLMs") · adapt tiết kiệm tham số (LoRA) |
| 3 · Method | Kiến trúc frozen · 5 bridge (thang capacity) · decoder-LoRA như can thiệp có chủ đích · 2 chỗ vặn × 6 câu hỏi |
| 4 · Experimental Setup | AutoViVQA · **grouped split không rò rỉ** (chia theo image_id) · 8 metric · nhiều seed |
| 5 · Main Results | Recipe so với 9 baseline · khoảng tin cậy bootstrap · phân tích efficiency của visual tiles |
| 6 · Ablation: truy tìm nút thắt | 6 câu hỏi RQ1–6 (xem §5 dưới đây) |
| 7 · Human Validation & Error Analysis | Tự kiểm + [cần: 2 người chấm, Cohen's κ] · phân tích lỗi |
| 8 · Discussion | Frozen decoder là bottleneck đáng kể · claim "reasoning-aware" của ViMoE cần đo trực tiếp · giới hạn |
| 9 · Conclusion | Recipe adapt tiết kiệm tham số + quy trình chẩn đoán. Không xây model mới |

---

## 4. Kết quả chính (tập val, metric đo nội bộ ×100)

| Mô hình | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Vintern-1B (gốc, zero-shot) | 0.12 | 17.52 | 19.87 | 17.55 | 1.91 | 25.84 | 23.93 | 8.54 |
| Vintern-1B (fine-tuned) | 13.01 | 52.47 | 55.12 | 53.76 | 6.11 | 51.93 | 35.25 | 72.84 |
| GPT-5 (zero-shot) | 10.84 | 47.20 | 55.20 | 50.89 | 6.07 | 47.30 | 33.34 | 84.20 |
| ViMoE-VQA | 9.65 | 62.89 | 58.65 | 60.69 | 12.54 | 47.07 | 39.10 | 88.67 |
| **Bridge Multi-Token (0.78%, 1 tile)** | **8.28** | **50.53** | **51.72** | **49.82** | **15.99** | **48.11** | **40.47** | **96.98** |
| **  + decoder LoRA r=16 (~1.0%)** | **10.42** | **53.85** | **55.00** | **53.17** | **19.44** | **51.48** | **43.91** | **105.59** |
| **  + decoder LoRA r=16, 3 epoch** | **11.78** | **55.54** | **56.25** | **54.67** | **20.98** | **52.92** | **45.24** | **109.60** |

*(Bảng đầy đủ 9 baseline + so sánh metric corpus với ViMoE + khoảng tin cậy: xem
`paper-blueprint.md` Bảng 1–2.)*

Đo kiểu corpus (so với paper khác): Multi-Token **CIDEr-D 92.8 ± 1.1** (khoảng
tin cậy 95% [91.3, 97.1] — nằm hoàn toàn trên mức 88.7 của ViMoE); + LoRA 3
epoch **106.8 ± 1.1**. Điểm yếu còn lại: token-F1 và Acc vẫn dưới ViMoE.

---

## 5. Chẩn đoán nút thắt — sáu trục, một trục dương

ΔF1 so với mốc (Multi-Token thường, seed 42: F1 50.66):

| RQ · trục | Can thiệp | ΔF1 | Nhận xét |
|---|---|--:|---|
| RQ1–2 · capacity của bridge | Full Q-Former (69M, gấp 10×) | −3.00 | bridge to hơn *không* tốt hơn trong khảo sát này |
| RQ3 · số tile thị giác | **train 1 tile → eval 3 tile** | −29.61 | bridge train single-tile suy giảm mạnh khi inference multi-tile ᵃ |
| RQ4 · routing thích ứng | policy học được theo loại câu hỏi | ≈0 | không hơn cấu hình cố định |
| RQ5 · tín hiệu huấn luyện | lấy mẫu nhiều câu tham chiếu | −1.65 | không cải thiện |
| RQ5 · căn chỉnh biểu diễn | KD projector (feat) | −1.00 | không cải thiện |
| **RQ6 · capacity của decoder** | **LoRA r=16 (1 epoch)** | **+2.51** | **cải thiện nhất quán** |
| **RQ6 · capacity của decoder** | **LoRA r=16 (3 epoch)** | **+4.01** | **cải thiện nhất quán** |

ᵃ Đây là thí nghiệm *train 1 tile → test 3/6 tile*, nên chỉ kết luận: bridge
train với input single-tile **generalize kém** sang inference multi-tile. **Chưa**
khảo sát *train 3 tile → test 3 tile* (thí nghiệm đáng làm nếu còn quota) —
chưa kết luận "multi-tile training không work".

**Ý nghĩa:** hiệu ứng LoRA decoder xuất hiện *nhất quán trên mọi bridge* (không
phải một cấu hình may mắn), trong khi bốn trục phía thị giác đều không có tín
hiệu. Các kết quả hiện tại cho thấy **frozen decoder là một bottleneck đáng kể,
trong khi tăng capacity ở bridge không mang lại lợi ích tương ứng**. Claim mạnh
hơn — "decoder capacity là bottleneck chính" — cần thí nghiệm decoder frozen lớn
hơn (0.5B → 1B/3B), xem §7 mục 6.

Hai hình minh hoạ (xem `paper-blueprint.md`):
- **Hình 1 — san bằng bridge:** các bridge thường trải CIDEr-D 56–97; sau LoRA
  decoder 0.23% đều hội tụ về 100–107.
- **Hình 2 — tile-collapse:** F1 50.7 → 21 khi eval từ 1 lên 3 tile (bridge train
  ở 1 tile); val loss 1.48 → 3.36.

---

## 6. Hiện trạng số liệu

**Đã khoá (nhiều seed, đủ vững để viết):**
- Multi-Token bridge (thường) — trung bình 4 seed
- Multi-Token + LoRA r=16, 1 epoch & 3 epoch — trung bình 3 seed
- Q-Former + LoRA r=16 — trung bình 3 seed
- Grouped split không rò rỉ đã xác nhận: số bridge gần như không đổi so với split
  cũ → kết quả trước không bị thổi phồng do rò rỉ
- Khoảng tin cậy bootstrap cho mọi so sánh chính

**Đang chạy (TIER-1, 19 job Kaggle, xong dự kiến rạng sáng 07/09):**
- 4 bridge còn lại (residual / mini_qformer / tile_attention / qformer) → nâng
  lên 3 seed
- Các dòng âm (align-feat / align-logit / answer-sampling) → nâng lên 3 seed
- Phủ LoRA cho đủ 5/5 bridge

**Cần thầy + 1 người (TIER-4):**
- Human validation thật: 300–500 mẫu, 2 người chấm, Cohen's κ (cho Trust4NLP).
  Hiện mới có bản tự kiểm N=120, 1 người chấm — đủ để nêu vấn đề "nhóm F1 giữa
  không đáng tin" nhưng chưa đủ để claim mạnh.

---

## 7. Việc còn lại tới deadline

| Thứ tự | Việc | Phụ thuộc |
|---|---|---|
| 1 | TIER-1 land → tính lại toàn bộ số theo 3 seed, cập nhật bảng | ~07/09 |
| 2 | Viết lại §5–§6 với số mới (một lượt) | sau (1) |
| 3 | TIER-2: LoRA giúp ở đâu trong decoder (attn / MLP / cả hai) — làm sâu RQ6 | ~10 job |
| 4 | Eval trên tập test (chốt số, hiện hầu hết là val) | quota |
| 5 | **Human validation** (cần thầy) | lên lịch |
| 6 | (nếu kịp) Decoder frozen lớn hơn (0.5B → 1B/3B) — kiểm tra claim "decoder capacity là bottleneck chính" | stretch |
| 7 | (nếu kịp) Train 3 tile → test 3 tile — trả lời câu "multi-tile training thì sao" | stretch |
| 8 | Ráp bản thảo → chỉnh → nộp | ~25/09 |

---

## 8. Ba đóng góp

1. **Parameter-efficient adaptation recipe** — frozen vision + lightweight bridge
   + decoder LoRA, chỉ ~1% trainable parameters nhưng đạt/vượt baseline
   fine-tuned trên các generation metric.
2. **Systematic bottleneck diagnosis** — khảo sát có hệ thống bridge capacity,
   tile scaling, routing, supervision / alignment và decoder adaptation; kết quả
   cho thấy decoder adaptation là hướng duy nhất đem lại cải thiện nhất quán
   trong không gian can thiệp đã thử.
3. **Reliable evaluation protocol** — grouped split chống leakage, đánh giá nhiều
   seed, bootstrap confidence intervals, human validation + error analysis; kèm
   phân tích efficiency của visual tiles.
