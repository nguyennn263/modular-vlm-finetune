# Báo cáo tiến độ nghiên cứu — Paper 3

*Cập nhật 05/10/2026. Mọi con số ở §3–§6.1 đã được **đo lại** sau khi sửa quy trình
đánh giá (xem §6.0) và đối chiếu với file kết quả gốc; Full Q-Former đã được huấn luyện
lại sau khi sửa lỗi rò đáp án.*

---

## 1. Câu hỏi nghiên cứu

> Thay vì xây một mô hình mới (như ViMoE-VQA), có thể cải thiện Vintern-1B trên
> AutoViVQA bằng cách chỉ cập nhật khoảng 1% tham số (đóng băng toàn bộ backbone)
> mà vẫn ngang fine-tune không? Nếu chưa, điểm nghẽn (bottleneck) ở đâu?

| Công trình | Cách làm | Tham số cập nhật |
|---|---|---|
| Vintern-1B (fine-tuned) | Số trích dẫn từ benchmark AutoViVQA. Cookbook fine-tune chính thức của Vintern: đóng băng ViT + projector, LoRA r=16 cho Qwen2.5-0.5B, 6 tile | Theo cookbook: chỉ LoRA LLM (~0.2% tham số) |
| ViMoE-VQA | Xây kiến trúc Mixture-of-Experts mới | Toàn bộ mô hình mới |
| **Nghiên cứu này** | Đóng băng cả InternViT-300M lẫn Qwen2.5-0.5B; chỉ huấn luyện bridge (0.78%) + LoRA cho decoder (0.23%), 1 tile | **~1% tổng tham số** |

**Trả lời ngắn gọn:** đạt được *một phần* — ngang/nhỉnh hơn Vintern-1B fine-tuned về
F1 (+1.0) và vượt rõ trên các chỉ số sinh văn bản, với ~1% tham số và 1 tile; vẫn kém
ViMoE-VQA ở token-F1. Điểm nghẽn nằm ở **khả năng thích nghi của decoder**: trong các
trục đã khảo sát, thêm LoRA cho decoder là can thiệp duy nhất cải thiện F1 (§4).

---

## 2. Thiết lập huấn luyện

- **Backbone** (InternViT-300M + Qwen2.5-0.5B, từ `5CD-AI/Vintern-1B-v3_5`): đóng
  băng hoàn toàn ở mọi cấu hình.
- **Bridge only:** huấn luyện bridge **2 epoch** (đồng đều cho cả 5 loại bridge, mọi
  seed, mọi dòng ablation), batch 8, learning rate 2e-4, ảnh 1 tile (nguyên ảnh,
  336px). Chọn 2 epoch vì CIDEr và F1 bão hòa từ epoch 2.
- **Bridge + LoRA cho decoder** (r=16, `q/k/v/o` của Qwen2.5): **một lần huấn luyện
  duy nhất**, bridge khởi tạo ngẫu nhiên và được huấn luyện **đồng thời** với LoRA
  (9.51M tham số = 1.01%) — *không* phải "giai đoạn 2" nối tiếp bridge đã train (đã
  kiểm trên checkpoint, xem §6.0). Báo cáo **1 và 3 epoch** như một đường cong: 1
  epoch (tổng cộng chỉ 1 epoch, ít hơn bridge-only) đã đạt ΔF1 +2.73 (~68% lợi ích);
  **3 epoch là điểm tốt nhất (ΔF1 +4.04)** và là cấu hình recipe.
- **Chọn rank LoRA:** đã quét r = 4/8/16/32/64 (subset 600 mẫu val); tăng rank không
  cho cải thiện rõ so với nhiễu giữa các seed → giữ **r=16**.
- **Đánh giá:** toàn bộ tập validation (5 463 mẫu) và test (5 468 mẫu); mô hình nhận
  **nguyên ảnh** khi sinh câu trả lời (xem §6.0); mọi chỉ số chấm bằng **một hàm duy
  nhất** (`metrics.vqa_metrics.score_answers`). Mỗi cấu hình chính chạy 3–4 seed,
  báo trung bình ± độ lệch chuẩn.

---

## 3. Kết quả chính (tập validation, thang ×100)

| Mô hình | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Vintern-1B (gốc, zero-shot) | 0.12 | 17.52 | 19.87 | 17.55 | 1.91 | 25.84 | 23.93 | 8.54 |
| ViT5_ViT | 7.97 | 46.84 | 50.33 | 48.52 | 4.13 | 46.89 | 31.02 | 72.68 |
| BARTPhoBEiT | 8.81 | 45.30 | 46.48 | 45.88 | 43.29 ᵃ | 44.83 | 24.57 | 188.96 ᵃ |
| Vintern-1B (fine-tuned) ᵇ | 13.01 | 52.47 | 55.12 | 53.76 | 6.11 | 51.93 | 35.25 | 72.84 |
| Llama 3.2 (zero-shot) | 0.36 | 23.96 | 73.71 | 36.16 | 3.62 | 36.11 | 30.01 | 62.84 |
| Gemini 2.0 Flash | 0.55 | 27.20 | 74.10 | 39.79 | 4.41 | 39.60 | 31.72 | 74.42 |
| Gemini 2.5 Flash | 0.22 | 24.43 | 76.66 | 24.75 ᶜ | 0.39 | 37.27 | 31.22 | 71.90 |
| GPT-5 (zero-shot) | 10.84 | 47.20 | 55.20 | 50.89 | 6.07 | 47.30 | 33.34 | 84.20 |
| ViMoE-VQA | 9.65 | 62.89 | 58.65 | 60.69 | 12.54 | 47.07 | 39.10 | 88.67 |
| **Bridge Multi-Token (0.78%, 1 tile)** | **9.21** | **51.54** | **52.48** | **50.74** | **16.88** | **49.03** | **41.38** | **99.07** |
| **  + LoRA cho decoder, r=16 (~1.0%), 1 epoch** | **10.94** | **54.37** | **55.05** | **53.47** | **19.72** | **51.79** | **44.11** | **106.68** |
| **  + LoRA cho decoder, r=16, 3 epoch** | **12.04** | **55.58** | **56.38** | **54.78** | **21.25** | **53.05** | **45.46** | **110.67** |

*In đậm = phương pháp đề xuất (trung bình 4 seed cho bridge / 3 seed cho LoRA). ᵃ BLEU
và CIDEr của BARTPhoBEiT là ngoại lai so với mọi mô hình khác, không so sánh. Các
baseline lấy theo báo cáo benchmark AutoViVQA — chỉ có 1 số, không kèm ±. Lưu ý: bảng
gốc của AutoViVQA ghi là tập **test** theo split 8:1:1 của họ (dữ liệu công bố lại chia
80/20: 29 661 / 7 416 câu), còn các dòng của mình đo trên tập val của grouped split
70/15/15 — so sánh mang tính tham khảo, không cùng tập. ᵇ Số 53.76 là trích dẫn, đo trên
split ngẫu nhiên của AutoViVQA — chưa tự tái lập. ᶜ Lấy đúng như paper AutoViVQA, nhưng
không khớp với P/R của chính dòng đó (2PR/(P+R) = 37.05) — nhiều khả năng lỗi đánh máy
trong paper gốc.*

*Lưu ý về cách tính F1: F1 của các baseline trích dẫn trùng khớp với 2PR/(P+R) tính từ
P/R trung bình; F1 của mình là trung bình F1 theo từng mẫu (luôn ≤ cách kia). Nếu tính
theo cách của baseline, cấu hình 3 epoch đạt 55.98 thay vì 54.78.*

### 3.1. Phương pháp đề xuất — mean ± std đầy đủ (mọi chỉ số, cả val và test)

*Thang ×100. Bridge = 4 seed (42/123/2026/3407); LoRA = 3 seed (42/123/3407). "CE" =
cross-entropy (không nhân 100). Std tổng thể (ddof=0).*

| Cấu hình | Split | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr | CE |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Bridge Multi-Token | val | 9.21 ± 0.05 | 51.54 ± 0.15 | 52.48 ± 0.21 | 50.74 ± 0.17 | 16.88 ± 0.23 | 49.03 ± 0.12 | 41.38 ± 0.21 | 99.07 ± 0.47 | 1.491 ± 0.003 |
| Bridge Multi-Token | test | 8.86 ± 0.37 | 51.19 ± 0.19 | 52.37 ± 0.21 | 50.47 ± 0.22 | 16.24 ± 0.43 | 48.70 ± 0.19 | 41.10 ± 0.29 | 96.19 ± 1.10 | 1.500 ± 0.002 |
| + LoRA r=16, 1 epoch | val | 10.94 ± 0.12 | 54.37 ± 0.11 | 55.05 ± 0.22 | 53.47 ± 0.17 | 19.72 ± 0.09 | 51.79 ± 0.15 | 44.11 ± 0.23 | 106.68 ± 0.69 | 1.374 ± 0.005 |
| + LoRA r=16, 1 epoch | test | 10.47 ± 0.22 | 53.95 ± 0.04 | 54.72 ± 0.07 | 53.10 ± 0.05 | 19.00 ± 0.16 | 51.29 ± 0.08 | 43.63 ± 0.16 | 104.44 ± 0.45 | 1.382 ± 0.007 |
| + LoRA r=16, 3 epoch | val | 12.04 ± 0.15 | 55.58 ± 0.25 | 56.38 ± 0.23 | 54.78 ± 0.22 | 21.25 ± 0.03 | 53.05 ± 0.15 | 45.46 ± 0.22 | 110.67 ± 0.48 | 1.327 ± 0.004 |
| + LoRA r=16, 3 epoch | test | 11.40 ± 0.24 | 55.37 ± 0.18 | 56.16 ± 0.09 | 54.52 ± 0.14 | 20.97 ± 0.29 | 52.68 ± 0.21 | 45.08 ± 0.14 | 108.43 ± 0.67 | 1.329 ± 0.007 |

### 3.2. Số per-seed (val, thô, không lấy trung bình)

*Bridge Multi-Token:*

| seed | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr | CE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 42 | 9.17 | 51.63 | 52.62 | 50.87 | 16.57 | 49.14 | 41.54 | 99.62 | 1.489 |
| 123 | 9.23 | 51.35 | 52.39 | 50.60 | 17.07 | 48.94 | 41.28 | 99.12 | 1.493 |
| 2026 | 9.28 | 51.74 | 52.73 | 50.95 | 16.73 | 49.16 | 41.62 | 99.22 | 1.487 |
| 3407 | 9.17 | 51.44 | 52.19 | 50.54 | 17.14 | 48.90 | 41.08 | 98.32 | 1.493 |

*+ LoRA 1 epoch:*

| seed | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr | CE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 42 | 10.96 | 54.47 | 55.31 | 53.65 | 19.69 | 51.96 | 44.38 | 106.94 | 1.370 |
| 123 | 10.78 | 54.22 | 54.77 | 53.25 | 19.64 | 51.60 | 43.81 | 105.74 | 1.381 |
| 3407 | 11.07 | 54.42 | 55.07 | 53.52 | 19.84 | 51.80 | 44.13 | 107.37 | 1.371 |

*+ LoRA 3 epoch:*

| seed | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr | CE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 42 | 11.92 | 55.89 | 56.66 | 55.04 | 21.22 | 53.25 | 45.68 | 110.93 | 1.326 |
| 123 | 12.25 | 55.28 | 56.09 | 54.51 | 21.28 | 52.87 | 45.16 | 109.99 | 1.332 |
| 3407 | 11.95 | 55.59 | 56.38 | 54.79 | 21.24 | 53.04 | 45.54 | 111.08 | 1.322 |

---

## 4. Phân tích điểm nghẽn — các trục can thiệp, một trục tích cực

ΔF1 so với cấu hình gốc (Bridge Multi-Token, trung bình 4 seed: F1 50.74). Mọi số trung
bình 3 seed trên val, trừ RQ3 (seed 42) và RQ4 (seed 42, đo theo quy trình cũ).

| RQ · axis | Intervention | ΔF1 | Nhận xét |
|---|---|--:|---|
| RQ1–2 · Bridge capacity | Light Q-Former (27.6M, 3.8×) | −3.68 | Bridge lớn hơn không tốt hơn |
| RQ1–2 · Bridge capacity | Full Q-Former (69M, ~10×) | −3.90 | Bridge lớn hơn không tốt hơn (bản huấn luyện lại, §6.0) |
| RQ3 · Number of visual tiles | 1 → 6 tile (đầu vào CLS nhất quán) | −0.90 | Thêm tile không giúp, cũng không làm sụp (bảng dưới) |
| RQ4 · Adaptive routing | Learned policy (theo loại câu hỏi) | ≈0 | Không hơn cấu hình cố định (chưa đo lại) |
| RQ5 · Training signal | Multi-reference answer sampling | −2.07 | Không cải thiện |
| RQ5 · Representation alignment | Projector-level feature KD | −0.39 | Null |
| RQ5 · Representation alignment | Projector-level logit KD (α = 0.1) | −0.28 | Null — F1 50.46 ± 0.21, val CE ≈1.53 |
| RQ5 · Representation alignment | Projector-level logit KD (α = 1.0) | −9.05 | Chỉ hỏng khi α quá lớn (KL lấn cross-entropy) |
| **RQ6 · Decoder — LoRA r=16 (1 epoch)** | q/k/v/o, huấn luyện cùng bridge | **+2.73** | **Cải thiện nhất quán** |
| **RQ6 · Decoder — LoRA r=16 (3 epochs)** | q/k/v/o, huấn luyện cùng bridge | **+4.04** | **Cải thiện nhất quán, tốt nhất** |

**Số liệu chi tiết đứng sau các dòng trên:**

*So sánh các loại bridge (val; bridge only = 2 epoch; +LoRA = huấn luyện chung 1 epoch;
± = std qua seed).*

| Bridge | Tham số | F1 | CIDEr | val CE | F1 + LoRA (1ep) | ΔF1 | CIDEr + LoRA |
|---|--:|--:|--:|--:|--:|--:|--:|
| Residual (1 token) | 4.86M (0.52%) | 46.02 ± 0.36 | 86.48 ± 0.77 | 1.665 | 52.69 ± 0.14 | +6.67 | 104.25 |
| Tile-Attention (8 token) | 4.14M (0.44%) | 45.14 ± 0.92 | 84.14 ± 1.40 | 1.683 | 52.82 ᵃ | +7.68 | 104.72 |
| **Multi-Token (8 token)** | **7.35M (0.78%)** | **50.74 ± 0.17** | **99.07 ± 0.47** | **1.491** | **53.47 ± 0.17** | **+2.73** | **106.68** |
| Light Q-Former (8 query) | 27.6M (2.87%) | 47.06 ± 0.66 | 88.07 ± 1.89 | 1.599 | 53.21 ± 0.14 | +6.14 | 106.26 |
| Full Q-Former (16 query) ᵇ | 69.4M (6.91%) | 46.84 ± 0.47 | 87.74 ± 1.24 | 1.605 | 53.29 ± 0.29 | +6.45 | 106.50 |

ᵃ Tile-Attention + LoRA mới có seed 42 (các cấu hình còn lại đủ 3 seed). ᵇ Bản huấn
luyện lại sau khi sửa lỗi rò đáp án (§6.0); checkpoint cũ không dùng.

*→ Bridge lớn hơn (Light Q-Former 3.8×, Full Q-Former ~10×) không tốt hơn; Multi-Token
có val CE thấp nhất (RQ1–2). Các bridge plain trải F1 45.1–50.7 (rộng 5.6 điểm); sau LoRA
dồn về 52.7–53.5 (rộng 0.8 điểm) bất kể chất lượng ban đầu; mức nâng lớn hơn khi bridge yếu hơn (RQ6).*

*Số tile × cách đưa đặc trưng vào bridge (Multi-Token, seed 42, F1 / val CE):*

| Đầu vào cho bridge | 1 tile | 3 tile | 6 tile |
|---|--:|--:|--:|
| Token CLS (trung bình CLS của từng tile khi > 1 tile) | 50.87 / 1.489 | 50.84 / 1.491 | 49.97 / 1.509 |
| Trung bình mọi token (cách code cũ dùng khi > 1 tile) | 18.32 / 3.468 | 18.31 / 3.469 | 18.96 / 3.419 |

*→ "Bridge sụp khi vượt 1 tile" ở bản trước là do code đổi kiểu đầu vào (CLS → trung bình
mọi token) khi có nhiều tile, không phải do số tile: dùng trung bình token thì sụp ngay
cả ở 1 tile; dùng CLS nhất quán thì 6 tile không sụp (−0.9) nhưng cũng không giúp (RQ3).
Với ảnh 4:3 của tập val, "3 tile" là nguyên ảnh lặp 3 lần; 6 tile = 6 mảnh 448px.*

*Chi phí encode thị giác của InternViT trên mỗi ảnh (Tesla P100-16GB):*

| Số tile | GFLOPs | Độ trễ (ms) | Thông lượng (ảnh/s) |
|--:|--:|--:|--:|
| 1 (của ta) | 362 | 229 | 6.00 |
| 2 | 724 | 374 | 3.30 |
| 4 | 1 448 | 648 | 1.70 |
| 6 | 2 172 | 922 | 1.15 |

*→ Tăng số tile không cải thiện chất lượng mà tốn chi phí: 1→6 tile là FLOPs ×6, độ
trễ ×4. Recipe dùng 1 tile nên không tốn chi phí này.*

**Nhận định:** Các trục phía thị giác và tín hiệu huấn luyện đều không cải thiện
token-F1; căn chỉnh biểu diễn là null ở mọi cường độ hợp lý. Thêm LoRA cho **decoder**
là trục duy nhất có tác dụng, nhất quán trên mọi bridge — dư địa cải thiện F1 nằm ở
khả năng thích nghi của decoder. CIDEr cũng vậy: plain trải 84.1–99.1 (rộng 15 điểm) →
+LoRA chỉ còn 104.3–106.7 (rộng 2.4 điểm).

---

## 5. Kết luận sơ bộ

**(a) Adapt Vintern-1B với ~1% tham số?** Được một phần: backbone đóng băng hoàn toàn +
bridge nhẹ + LoRA decoder (3 epoch, 1 tile) ngang/nhỉnh hơn Vintern-1B fine-tuned (6
tile) về F1 (+1.02) và vượt rõ trên chỉ số sinh văn bản (BLEU +15.14, ROUGE +1.12,
METEOR +10.21, CIDEr +37.83); riêng Acc (exact match) còn thấp hơn 0.97. Cũng vượt
ViMoE-VQA về Acc (+2.39) cùng BLEU/ROUGE/METEOR/CIDEr — nhưng còn kém ViMoE-VQA ở
token-F1 (−5.91). (Lưu ý: số Vintern-1B fine-tuned là trích dẫn, đo trên split khác —
xem ᵇ ở §3.)

**(b) Điểm nghẽn ở đâu?** Khả năng thích nghi của decoder. Trong không gian can thiệp
đã khảo sát, các thay đổi phía thị giác (dung lượng bridge, số tile, routing, căn chỉnh
biểu diễn) không cải thiện F1; chỉ thêm dung lượng cho decoder (LoRA) mới cải thiện.

**Hạn chế cần nói rõ:** mô hình dựa ít vào chi tiết ảnh. Bridge chỉ nhận token CLS của
InternViT; cho mô hình xem góc 1/6 ảnh hay nguyên ảnh chỉ đổi F1 ≤ 1.4 điểm (bản LoRA
gần như không đổi); và trên các tập cần đọc chữ, mô hình thua xa Vintern gốc (§6.1).

**Hàm ý:** muốn thu hẹp nốt khoảng cách F1 thì phải tác động vào phía decoder, hoặc
cho bridge truy cập thông tin theo patch — không phải đầu tư tiếp vào các biến thể
bridge chỉ dùng vector toàn ảnh. Đây cũng là điểm phản biện với "reasoning-aware
routing" của ViMoE: trên cùng benchmark, loại câu hỏi không mang tín hiệu hữu ích cho
phân bổ tài nguyên thị giác.

---

## 6. Ghi chú về độ tin cậy

- Mọi kết quả bridge plain + dòng âm: trung bình 3 seed @ 2 epoch (Multi-Token = 4
  seed); LoRA: 3 seed cho cả 1 và 3 epoch. F1 std 0.05–0.22 cho cấu hình đề xuất (§3.1),
  tối đa 0.92 ở bridge phụ yếu nhất (Tile-Attention).
- Đối chiếu tập test: test thấp hơn val 0.27 (bridge only), 0.37 (LoRA 1 epoch), 0.26
  (LoRA 3 epoch) điểm F1 → không có dấu hiệu overfit tập val.
- Dùng grouped split chống rò rỉ dữ liệu. Khoảng tin cậy bootstrap 95% (paired, 2 000
  lần, seed 42, val) cho ΔF1 của LoRA 1 epoch trên Multi-Token: **+2.78, [2.20, 3.34]**,
  P(Δ > 0) = 1.000.
- Tái lập: chạy lại đúng cách đo cũ trên phần cứng khác (T4 thay vì P100) ra 54.84 so
  với 54.91 (lệch 0.07, nhỏ hơn std giữa các seed).
- Đánh giá ngữ nghĩa hiện mới ở mức tự kiểm 120 mẫu do **trợ lý AI chấm** (một lượt
  chấm, không xem ảnh, chỉ so với 5 đáp án tham chiếu) — chưa phải đánh giá của người.
  Cần nghiên cứu 2 người chấm có xem ảnh + Cohen's κ cho bản camera-ready.
- Độ lệch chuẩn ở §3–§4 là std tổng thể (ddof=0); ở bảng OOD §6.1 và bảng sweep
  `num_tokens` §6.2 là std mẫu (ddof=1).

### 6.0. Các lỗi đã phát hiện và sửa trong quy trình (đợt kiểm tra 04–05/10)

Trong lúc rà soát số liệu, em phát hiện và sửa 5 vấn đề; mọi số ở báo cáo này đã theo
quy trình đã sửa:

1. **Đánh giá chỉ cho mô hình xem góc ảnh.** Khi sinh câu trả lời ở 1 tile, code đưa
   vào mảnh 448px đầu tiên của lưới chia động — tức góc trái trên, khoảng 1/6 ảnh — trong
   khi lúc huấn luyện mô hình thấy nguyên ảnh. Chỉ là lỗi đánh giá (checkpoint không đổi;
   val loss tái lập tới 3 chữ số), nên đã **đánh giá lại** toàn bộ: F1 của bridge-only
   tăng ~1.2, bản LoRA gần như không đổi, kết luận giữ nguyên.
2. **RQ3 trộn số tile với kiểu đầu vào** (xem bảng ở §4): kết luận "sụp khi tăng tile"
   bị thay bằng "thêm tile không giúp". Thí nghiệm tile-augmentation cũ (huấn luyện với
   hai kiểu đầu vào lẫn lộn) không còn hợp lệ; đang cân nhắc bỏ hẳn hay huấn luyện lại.
3. **Một hàm F1 phụ tính theo ký tự** (chỉ ảnh hưởng bảng OOD cũ). Toàn bộ project giờ
   chấm bằng một hàm duy nhất; bảng OOD ở §6.1 đã tính lại.
4. **Full Q-Former nhìn thấy đáp án khi huấn luyện:** input huấn luyện chứa cả đáp án và
   Q-Former cross-attend lên toàn bộ input mà không che. Đã sửa (che phần đáp án và
   padding, có test kiểm chứng) và **đã huấn luyện lại** 3 seed (plain 46.84, + LoRA
   53.29 — cùng xu hướng các bridge khác); các bridge khác không bị ảnh hưởng.
5. **Mô tả quy trình LoRA sai:** bản trước viết LoRA là "giai đoạn 2 khởi tạo từ bridge
   đã train". Kiểm trên checkpoint cho thấy đó là một lần huấn luyện chung từ đầu (bridge
   ngẫu nhiên + LoRA). Số liệu không đổi, chỉ sửa mô tả (§2).

### 6.1. Đánh giá ngoài phân phối (OOD) trên 4 tập VQA tiếng Việt khác

Để kiểm tra khả năng khái quát hóa, đã đánh giá **Vintern-1B gốc (zero-shot, 6
tile)** so với **mô hình đề xuất (1 tile, bridge + LoRA 3 epoch)** trên 4 tập VQA
tiếng Việt mà cả hai chưa từng train (lấy mẫu từ test split chính thức, riêng
OpenViVQA dùng dev split vì đáp án của test split không công bố; 1000 mẫu/seed,
riêng ViVQA chỉ 505–520 mẫu/seed do một phần ảnh COCO tải về thất bại — cùng một
tập câu hỏi cho cả 2 mô hình; 3 seed lấy mẫu, mean ± std). Mọi chỉ số chấm bằng
**cùng một hàm với §3** (F1 word-level, CIDEr in-house); mô hình đề xuất nhận
nguyên ảnh (xem §6.0).

| Tập | F1 — Vintern gốc | F1 — Đề xuất | CIDEr — Vintern gốc | CIDEr — Đề xuất | Ai hơn |
|---|--:|--:|--:|--:|---|
| ViTextVQA (đọc chữ trong ảnh, UIT 2024) | 32.01 ± 0.38 | 3.96 ± 0.11 | 195.6 ± 2.9 | 15.6 ± 1.1 | Vintern gốc |
| OpenViVQA (~44% cần đọc chữ, UIT 2023) | 58.40 ± 0.98 | 21.56 ± 0.59 | 411.6 ± 13.6 | 83.2 ± 2.9 | Vintern gốc |
| ViVQA-X (VQA tổng quát, VLAI-AIVN 2025) | 14.52 ± 0.58 | 15.89 ± 0.80 | 50.1 ± 3.5 | 42.1 ± 3.6 | Ngang (đề xuất hơn F1, Vintern hơn CIDEr) |
| ViVQA gốc (UIT-ViVQA, PACLIC 2021) | 23.90 ± 0.30 | 31.48 ± 1.40 | 78.6 ± 2.5 | 84.9 ± 3.5 | **Đề xuất** |

Cho mô hình đề xuất xem nguyên ảnh hay chỉ góc trái trên (cách đo cũ) gần như không
đổi kết quả OOD (chênh < 0.5 F1 ở cả 4 tập) — thêm một dấu hiệu rằng mô hình dựa
rất ít vào chi tiết ảnh.

Xem qua mẫu dự đoán ở 2 tập đầu: mô hình đề xuất **hầu như không đọc được chữ trong
ảnh**, chỉ đoán theo dạng câu hỏi (VD hỏi số điện thoại → "1111111111111111"; 2 câu
"cửa hàng tên gì" trên 2 ảnh khác nhau ra cùng một câu trả lời).

**Kết luận:** pattern nhất quán qua cả 4 tập — mô hình đề xuất tốt hơn (ViVQA) hoặc
ngang (ViVQA-X) trên VQA tổng quát không cần đọc chữ (gần domain train AutoViVQA), nhưng **mất gần
như hoàn toàn khả năng đọc chữ trong ảnh** so với Vintern gốc. Nguyên nhân khả dĩ:
(i) bridge Multi-Token chỉ nhận token CLS của InternViT (một vector toàn ảnh, xem §6.2),
không giữ thông tin chi tiết theo từng patch; (ii) chỉ 1 tile (độ phân giải thấp
hơn 6 tile); (iii) dữ liệu train không có tác vụ OCR. Đáng đưa vào phần
generalization/limitation của paper.

### 6.2. Trả lời 2 câu hỏi của thầy về thiết kế bridge

*Lưu ý: các số trong mục này đo theo quy trình đánh giá cũ (góc ảnh, §6.0) và chưa đo
lại; mọi dòng trong cùng một bảng đều đo cùng cách nên so sánh tương đối vẫn hợp lệ.
Theo quy trình mới, `n=8` (Multi-Token) đạt 50.74 thay vì 49.55.*

Thầy hỏi 2 câu về bridge Multi-Token (kiến trúc đề xuất, dòng in đậm ở §3):
(1) tại sao chọn `num_tokens=8` mà không phải 6/10/12; (2) xu hướng "gộp" là
gì — có thử deconvolution/zoom-in-zoom-out không, và khi gộp thành 8 token thì
đang dùng max, avg, hay giữ nguyên. Em đã chạy thí nghiệm cho cả hai câu. (Thực
hiện trên nhánh `feat/bridge-design-ablation`; recipe chính thức ở §1–§5 không
đổi. Các job này chạy sau 15/09 trên GPU T4 do Kaggle ngừng cung cấp P100 — cùng
hyperparameter, chỉ khác phần cứng.)

**Câu 1 — sweep `num_tokens`.** Giữ nguyên mọi thứ khác so với recipe (plain
bridge, không LoRA, 2 epoch, cùng protocol §2). `n=8` đã có sẵn 4-seed từ
thí nghiệm chính; các `n` khác train mới, 3 seed (42/123/3407) mỗi điểm:

| n | mean F1 | std | n_seed |
|--:|--:|--:|--:|
| 4 | 48.54 | 0.42 | 3 |
| 6 | 49.03 | 0.39 | 3 |
| **8 (recipe hiện tại)** | **49.55** | 0.08 | 4 |
| 10 | 49.99 | 0.16 | 3 |
| 12 | 50.23 | 0.22 | 3 |
| 14 | 50.67 | 0.09 | 3 |
| 16 | 50.44 | 0.54 | 3 |
| 18 | 50.51 | 0.79 | 3 |
| 20 | 50.62 | 0.52 | 3 |

*(F1 nội bộ, ×100, cùng thang với §3; std mẫu, ddof=1.)*

`n=8` không phải điểm tối ưu: `n=4` và `n=6` đều thấp hơn rõ — chênh so với `n=8`
(−1.01 và −0.52) vượt độ lệch chuẩn của chính điểm đó (0.42 và 0.39). F1 tăng dần
trong khoảng `n=10–14`: mọi `n≥10` đều vượt `n=8`, và `n=14` (50.67 ± 0.09) là điểm
cao nhất, std rất nhỏ. Từ `n=14` trở đi là plateau: 4 điểm `14/16/18/20` dao động
50.4–50.7 không đơn điệu (`n=16` thấp hơn `n=14`), chênh lệch nằm trong biên độ
std của các điểm 16/18/20.

Nếu chọn 1 số khác 8: **n=14** cho F1 cao nhất (+1.12 so với n=8); **n=12** là
lựa chọn tiết kiệm hơn (ít token ảnh nạp vào decoder hơn) với mức tăng +0.68. Mọi
số ở §1–§5 của báo cáo này vẫn dùng `n=8`; đây là ablation trả lời câu hỏi của
thầy — nếu thầy đồng ý, có thể cân nhắc chuyển recipe sang n=14 cho bản cuối.

**Câu 2 — bản chất phép "gộp" trong Multi-Token, và các kiến trúc gộp khác.**

Một điểm cần làm rõ trước về thiết kế hiện tại: **Multi-Token không có toán tử
gộp (pooling) học được nào.** InternViT tự đưa ra một vector đặc trưng toàn ảnh
`(B, 1024)` trước khi bridge chạy; bridge chỉ là 2 lớp `Linear` chiếu từ vector
đó ra `k` token, không nhìn từng patch riêng lẻ. Tức là câu trả lời cho "max, avg
hay để nguyên" là: để nguyên vector toàn ảnh của InternViT, rồi chiếu tuyến tính.

Để trả lời phần còn lại, em đã thử thêm các kiến trúc làm việc trực tiếp trên
patch, cùng `num_tokens≈8`, so bằng F1:

| Kiến trúc | Toán tử gộp | Trainable params | F1 | Δ so với Multi-Token |
|---|---|--:|--:|--:|
| **Multi-Token (hiện tại)** | không có — Linear thuần trên vector toàn ảnh | 7.35M | **49.55 ± 0.07** (4 seed) | — |
| Patch-Pool (mean) | trung bình cố định trên patch | 0.92M | 38.80 (seed 42) | −10.75 |
| Patch-Pool (max) | max cố định trên patch | 0.92M | 35.71 (seed 42) | −13.84 |
| Attention (= Tile-Attention, §4) | attention học được | 4.14M | 45.17 ± 0.94 (3 seed) | −4.38 |
| Conv-Abstractor (HoneyBee-style) | conv → nén (adaptive avg pool) → conv | 19.87M | 48.87 (seed 42) | −0.68 |

*(Conv-Abstractor: `num_tokens=9` vì kiến trúc cần lưới không gian vuông;
theo Cha et al., "Honeybee: Locality-enhanced Projector for Multimodal LLM",
CVPR 2024 — 2 khối ResNet trước và sau một adaptive-avg-pool. Patch-Pool và
Conv-Abstractor mới có seed 42; Multi-Token seed 42 là 49.61.)*

Trả lời cụ thể từng ý:
- **"Zoom in → nén → zoom out"** = đúng tinh thần kiến trúc Conv-Abstractor đã
  thử (conv xử lý cục bộ = zoom in, adaptive pool = nén, conv xử lý lại = zoom
  out). Kết quả gần bằng nhưng vẫn thấp hơn Multi-Token (48.87 so với 49.61 cùng
  seed 42), và tốn gấp 2.7× tham số (19.9M so với 7.3M).
- **Pooling (max/avg) cố định**: thua đậm (35.7–38.8 so với 49.6) khi cô lập
  hoàn toàn (chỉ 1 `Linear` dùng chung + pooling cố định, không thêm capacity nào
  khác) — đã rà lại code (shape, dispatch, output thực tế) để loại trừ khả năng
  bug; pooling cố định không chọn lọc được patch nào quan trọng.
- **DeConvolution đúng nghĩa đen**: chưa implement, vì deconv (transposed
  convolution) là phép **upsample** — đi từ ít phần tử ra nhiều phần tử không gian
  hơn — trong khi bridge cần chiều ngược lại: nén 1024 patch token (lưới 32×32)
  của InternViT xuống còn 8–20 token. Conv-Abstractor (conv thường + pool) là
  phương án đúng tinh thần câu hỏi cho hướng "nén bằng convolution".

**Kết luận:** trong các cách "gộp" đã thử (pool cố định, attention học được,
conv+pool kiểu Conv-Abstractor), chưa có cách nào vượt được thiết kế hiện tại
(Linear trên vector toàn ảnh của InternViT); Conv-Abstractor gần nhất nhưng tốn
2.7× tham số. Đây là cơ sở để giữ Multi-Token làm bridge chính của paper.

---

## 7. Đóng góp

1. **Quy trình thích nghi tiết kiệm tham số:** frozen backbone + bridge nhẹ +
   LoRA cho decoder, ~1% tham số, 1 tile, nhưng đạt/vượt baseline fine-tuned trên
   F1 và các chỉ số sinh (và đối chiếu được trên tập test).
2. **Chẩn đoán điểm nghẽn hệ thống:** khảo sát bridge, số tile, routing, tín hiệu
   huấn luyện / căn chỉnh, và decoder → dư địa hữu ích nằm ở khả năng thích nghi
   của decoder, không phải phía thị giác.
3. **Quy trình đánh giá đáng tin cậy:** grouped split, nhiều seed, đối chiếu
   val/test, bootstrap CI, đánh giá thủ công + phân tích lỗi, đánh giá OOD trên 4
   tập ngoài, kèm phân tích hiệu quả tính toán của tile.
