# Paper 3 — Hiện trạng để báo cáo thầy

*Cập nhật 2026-09-06. Deadline ~2026-09-27 (ACIIDS 2027 / Trust4NLP).*
*Bản chi tiết (đầy đủ 7 bảng + 2 hình + phụ lục per-seed): `plans/paper-blueprint.md`.*

---

## 1. Câu hỏi nghiên cứu

> Thay vì **xây một model mới** (như ViMoE-VQA), có thể **cải thiện Vintern-1B**
> trên AutoViVQA một cách **rẻ** — chỉ train ~1% tham số, đóng băng toàn bộ
> backbone — để đạt ngang mức fine-tune toàn bộ không? Nếu chưa đạt, **nút thắt
> nằm ở đâu**?

**Định vị:**

| | Cách làm | Chi phí |
|---|---|---|
| Vintern-1B (fine-tuned) | Train **toàn bộ** InternViT-300M + projector + LoRA cho LLM, 3M cặp, 4×RTX-3090 | rất cao |
| ViMoE-VQA | Thiết kế kiến trúc MoE **mới** | cao (model mới) |
| **Của chúng tôi** | Đóng băng cả InternViT-300M lẫn Qwen2-0.5B, chỉ train **bridge 0.78%** (+ **LoRA decoder 0.23%**), 1 tile | **~1% tham số** |

---

## 2. Trả lời hiện tại

Phía **thị giác đã bão hoà** — bốn hướng can thiệp độc lập đều không cải thiện.
**Decoder là trục duy nhất còn dư địa.** → Công thức:
**bridge pooling rẻ (cố định) + LoRA decoder nhẹ.**

Recipe này **vượt Vintern-1B fine-tuned ở mọi metric sinh** (BLEU +14.9,
METEOR +10.0, CIDEr +36.8) với ~1% chi phí, và **thắng ViMoE-VQA** ở
BLEU / ROUGE / METEOR / CIDEr. Còn kém ViMoE ở token-F1 (−6.0) — chính khoảng
cách này dẫn tới phần chẩn đoán.

---

## 3. Cấu trúc paper (LNCS, 12–15 trang)

| § | Nội dung |
|---|---|
| 1 · Introduction | ViMoE xây model mới · Vintern train nặng phía thị giác · câu hỏi: adapt rẻ được không, nút thắt ở đâu · 4 đóng góp |
| 2 · Related Work | VQA tiếng Việt · frozen-backbone + projector (BLIP-2, "Inference-Optimal VLMs") · adapt tiết kiệm tham số (LoRA) |
| 3 · Method | Kiến trúc frozen · 5 bridge (thang capacity) · decoder-LoRA như can thiệp có chủ đích · 2 chỗ vặn × 6 câu hỏi |
| 4 · Experimental Setup | AutoViVQA · **grouped split không rò rỉ** (chia theo image_id) · 8 metric |
| 5 · Main Results | Recipe so với 9 baseline · khoảng tin cậy bootstrap · hiệu quả tính toán |
| 6 · Ablation: truy tìm nút thắt | 6 câu hỏi RQ1–6 (xem §5 dưới đây) |
| 7 · Human Validation & Error Analysis | Tự kiểm + [cần: 2 người chấm, Cohen's κ] · phân tích lỗi |
| 8 · Discussion | Frozen decoder là trần · claim "reasoning-aware" của ViMoE cần đo trực tiếp · giới hạn |
| 9 · Conclusion | Recipe rẻ + quy trình chẩn đoán. Không xây model mới |

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
epoch **106.8 ± 1.1**.

---

## 5. Chẩn đoán nút thắt — sáu trục, một trục dương

ΔF1 so với mốc (Multi-Token thường, seed 42: F1 50.66):

| RQ · trục | Can thiệp | ΔF1 | Kết luận |
|---|---|--:|---|
| RQ1–2 · capacity của bridge | Full Q-Former (69M, gấp 10×) | −3.00 | âm — bridge to hơn *tệ hơn* |
| RQ3 · số tile thị giác | đánh giá ở 3 tile | −29.61 | âm — bridge sụp khi > 1 tile |
| RQ4 · routing thích ứng | policy học được theo loại câu hỏi | ≈0 | âm — không hơn cấu hình cố định |
| RQ5 · tín hiệu huấn luyện | lấy mẫu nhiều câu tham chiếu | −1.65 | âm |
| RQ5 · căn chỉnh biểu diễn | KD projector (feat) | −1.00 | âm |
| **RQ6 · capacity của decoder** | **LoRA r=16 (1 epoch)** | **+2.51** | **dương** |
| **RQ6 · capacity của decoder** | **LoRA r=16 (3 epoch)** | **+4.01** | **dương** |

**Ý nghĩa:** không phải một mẹo may mắn — mà là *chỉ* hướng thêm capacity cho
decoder mới có tác dụng, bất kể bridge nào. → với lớp VLM này (ViT đóng băng,
decoder nhỏ 0.5B đóng băng, vài token thị giác đã pool), **frozen decoder là
trần** chứ không phải pipeline thị giác.

Hai hình minh hoạ (xem `paper-blueprint.md`):
- **Hình 1 — san bằng bridge:** các bridge thường trải CIDEr-D 56–97; sau LoRA
  decoder 0.23% đều hội tụ về 100–107.
- **Hình 2 — tile-collapse:** F1 50.7 → 21 khi tăng từ 1 lên 3 tile; val loss
  1.48 → 3.36.

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
| 6 | (nếu kịp) Thử decoder frozen lớn hơn — kiểm tra claim "decoder nhỏ là trần" | stretch |
| 7 | Ráp bản thảo → chỉnh → nộp | ~25/09 |

---

## 8. Bốn đóng góp

1. **Recipe adapt rẻ:** frozen backbone + bridge pooling 0.78% + LoRA decoder
   0.23% → vượt fine-tune toàn bộ trên metric sinh với ~1% chi phí.
2. **Quy trình chẩn đoán 6 bước** khoanh nút thắt về frozen decoder — bằng *mẫu
   hình* 4 trục âm / 1 trục dương, không phải một ablation lẻ.
3. **Đặc tả hiệu quả tính toán** của đòn bẩy tile (FLOPs ×6, độ trễ ×4 từ 1→6
   tile) — phân tích mà ViMoE-VQA để lại sau.
4. **Grouped split không rò rỉ** + bảng oracle + toàn bộ code công bố.
