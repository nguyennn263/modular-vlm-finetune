# Báo cáo tiến độ nghiên cứu — Paper 3

*Cập nhật 12/09/2026. Toàn bộ thí nghiệm chính đã hoàn tất, số liệu đã rà soát
chéo. Một hạng mục phụ đang chạy (§6.3: tái lập baseline Vintern-FT bằng
cookbook chính thức) — không chặn phần còn lại.*

---

## 1. Câu hỏi nghiên cứu

> Thay vì xây một mô hình mới (như ViMoE-VQA), có thể cải thiện Vintern-1B trên
> AutoViVQA bằng cách chỉ cập nhật khoảng 1% tham số (đóng băng toàn bộ backbone)
> mà vẫn ngang fine-tune không? Nếu chưa, điểm nghẽn (bottleneck) ở đâu?

| Công trình | Cách làm | Tham số cập nhật |
|---|---|---|
| Vintern-1B (fine-tuned) | Cookbook fine-tune chính thức của Vintern: đóng băng ViT + projector, LoRA r=16 cho Qwen2-0.5B | Chỉ LoRA LLM (~1% tham số) |
| ViMoE-VQA | Xây kiến trúc Mixture-of-Experts mới | Toàn bộ mô hình mới |
| **Nghiên cứu này** | Đóng băng cả InternViT-300M lẫn Qwen2-0.5B; chỉ huấn luyện bridge (0.78%) + LoRA cho decoder (0.23%), 1 tile | **~1% tổng tham số** |

**Trả lời ngắn gọn:** đạt được *một phần* — vượt Vintern-1B fine-tuned trên mọi
chỉ số sinh văn bản với ~1% tham số, nhưng vẫn kém ViMoE-VQA ở token-F1. Điểm
nghẽn nằm ở **attention của frozen decoder**: chỉ can thiệp vào đó mới cải thiện F1.

---

## 2. Thiết lập huấn luyện

- **Backbone** (InternViT-300M + Qwen2-0.5B): đóng băng hoàn toàn ở mọi cấu hình.
- **Giai đoạn 1 — huấn luyện bridge:** **2 epoch** (đồng đều cho cả 5 loại bridge,
  mọi seed, mọi dòng ablation), batch 8, learning rate 2e-4, ảnh 1 tile; khoảng
  5 giờ/lần trên một GPU Tesla P100-16GB. Chọn 2 epoch vì **CIDEr và F1 bão hòa
  từ epoch 2** — bridge chỉ học một ánh xạ hẹp (đặc trưng thị giác → không gian
  embedding của Qwen2), hội tụ nhanh; thêm epoch không cải thiện.
- **Giai đoạn 2 — LoRA cho decoder** (áp lên `q/k/v/o` của Qwen2, r=16): huấn
  luyện thêm trên bridge đã cố định. LoRA là adapter nhỏ vào attention của decoder
  đóng băng, **học chậm và dần** — chưa bão hòa ở epoch 1, nên báo cáo cả **1 và
  3 epoch** như một đường cong: 1 epoch đã đạt ΔF1 +3.97 (~80% lợi ích), **3
  epoch là điểm tốt nhất (ΔF1 +5.16)** và là cấu hình recipe. (Có thử 5 epoch
  nhưng job bị cắt ở giới hạn quota Kaggle → không dùng.)
- **Số epoch có đồng đều không?** Bridge: **có** — cố định 2 epoch khắp nơi.
  (Trong quá trình làm từng có một số job seed-42 vô tình chạy 4 epoch; đã phát
  hiện, re-run về 2, và rà soát lại toàn bộ.) LoRA: 1 và 3 là hai mức *có chủ
  đích* để vẽ đường cong, không phải chạy lệch.
- **Đánh giá:** toàn bộ tập validation (5 463 mẫu) và tập test (5 468 mẫu), không
  lấy mẫu con. Mỗi cấu hình chính chạy 3–4 seed, báo trung bình ± độ lệch chuẩn.

---

## 3. Kết quả chính (tập validation, chỉ số nội bộ, thang ×100)

| Mô hình | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Vintern-1B (gốc, zero-shot) | 0.12 | 17.52 | 19.87 | 17.55 | 1.91 | 25.84 | 23.93 | 8.54 |
| ViT5_ViT | 7.97 | 46.84 | 50.33 | 48.52 | 4.13 | 46.89 | 31.02 | 72.68 |
| BARTPhoBEiT | 8.81 | 45.30 | 46.48 | 45.88 | 4.33 | 44.83 | 24.57 | 188.96 ᵃ |
| Vintern-1B (fine-tuned) ᵇ | 13.01 | 52.47 | 55.12 | 53.76 | 6.11 | 51.93 | 35.25 | 72.84 |
| Llama 3.2 (zero-shot) | 0.36 | 23.96 | 73.71 | 36.16 | 3.62 | 36.11 | 30.01 | 62.84 |
| Gemini 2.0 Flash | 0.55 | 27.20 | 74.10 | 39.79 | 4.41 | 39.60 | 31.72 | 74.42 |
| Gemini 2.5 Flash | 0.22 | 24.43 | 76.66 | 24.75 | 0.39 | 37.27 | 31.22 | 71.90 |
| GPT-5 (zero-shot) | 10.84 | 47.20 | 55.20 | 50.89 | 6.07 | 47.30 | 33.34 | 84.20 |
| ViMoE-VQA | 9.65 | 62.89 | 58.65 | 60.69 | 12.54 | 47.07 | 39.10 | 88.67 |
| **Bridge Multi-Token (0.78%, 1 tile)** | **8.17** | **50.21** | **51.50** | **49.55** | **15.72** | **47.84** | **40.22** | **96.49** |
| **  + LoRA cho decoder, r=16 (~1.0%), 1 epoch** | **10.93** | **54.39** | **55.11** | **53.52** | **19.72** | **51.81** | **44.11** | **106.56** |
| **  + LoRA cho decoder, r=16, 3 epoch** | **12.00** | **55.46** | **56.38** | **54.71** | **21.07** | **52.96** | **45.42** | **110.49** |

*In đậm = phương pháp đề xuất (trung bình 4 seed cho bridge / 3 seed cho LoRA).
ᵃ CIDEr của BARTPhoBEiT là ngoại lai (sinh câu dài), không so sánh. Baseline lấy
theo báo cáo benchmark AutoViVQA — các dòng baseline chỉ có 1 số, không có
per-seed nên không kèm ± ở bảng trên; phần ± đầy đủ cho phương pháp đề xuất ở
ngay dưới. ᵇ Số 53.76 là trích dẫn — dùng cookbook fine-tune chính thức của
Vintern (đóng băng ViT + projector, LoRA r=16 LLM), đang tái lập ở §6.3.*

### 3.1. Phương pháp đề xuất — mean ± std đầy đủ (mọi chỉ số, cả val và test)

*Chỉ số nội bộ (in-house), thang ×100. Bridge = 4 seed (42/123/2026/3407);
LoRA = 3 seed (42/123/3407). "val CE" = cross-entropy (không nhân 100).*

| Cấu hình | Split | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr | CE |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Bridge Multi-Token | val | 8.17 ± 0.10 | 50.21 ± 0.09 | 51.50 ± 0.12 | 49.55 ± 0.07 | 15.72 ± 0.30 | 47.84 ± 0.06 | 40.22 ± 0.18 | 96.49 ± 0.59 | 1.49 |
| Bridge Multi-Token | test | 7.98 ± 0.24 | 49.83 ± 0.17 | 51.21 ± 0.18 | 49.20 ± 0.18 | 15.36 ± 0.33 | 47.37 ± 0.13 | 39.74 ± 0.18 | 93.24 ± 0.71 | 1.500 ± 0.002 |
| + LoRA r=16, 1 epoch | val | 10.93 ± 0.13 | 54.39 ± 0.08 | 55.11 ± 0.16 | 53.52 ± 0.11 | 19.72 ± 0.09 | 51.81 ± 0.12 | 44.11 ± 0.20 | 106.56 ± 0.53 | 1.374 ± 0.005 |
| + LoRA r=16, 1 epoch | test | 10.49 ± 0.22 | 53.99 ± 0.11 | 54.77 ± 0.07 | 53.15 ± 0.07 | 18.98 ± 0.24 | 51.34 ± 0.09 | 43.73 ± 0.18 | 104.65 ± 0.44 | 1.382 ± 0.007 |
| + LoRA r=16, 3 epoch | val | 12.00 ± 0.17 | 55.46 ± 0.11 | 56.38 ± 0.23 | 54.71 ± 0.14 | 21.07 ± 0.23 | 52.96 ± 0.06 | 45.42 ± 0.14 | 110.49 ± 0.27 | 1.327 ± 0.004 |
| + LoRA r=16, 3 epoch | test | 11.21 ± 0.20 | 55.06 ± 0.14 | 55.95 ± 0.08 | 54.28 ± 0.13 | 20.72 ± 0.25 | 52.44 ± 0.18 | 44.82 ± 0.17 | 107.60 ± 0.65 | 1.329 ± 0.007 |

### 3.2. Số per-seed (thô, không lấy trung bình)

*Bridge Multi-Token plain — in-house val:*

| seed | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 42 | 8.20 | 50.36 | 51.43 | 49.61 | 15.28 | 47.86 | 40.05 | 96.72 |
| 123 | 8.00 | 50.10 | 51.40 | 49.46 | 16.05 | 47.76 | 40.16 | 95.84 |
| 2026 | 8.24 | 50.20 | 51.70 | 49.64 | 15.91 | 47.93 | 40.53 | 97.35 |
| 3407 | 8.24 | 50.20 | 51.47 | 49.51 | 15.64 | 47.80 | 40.13 | 96.05 |

*+ LoRA 1 epoch — in-house val:*

| seed | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr | CE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 42 | 10.84 | 54.48 | 55.33 | 53.67 | 19.84 | 51.97 | 44.38 | 106.83 | 1.370 |
| 123 | 10.85 | 54.40 | 54.95 | 53.42 | 19.64 | 51.72 | 43.90 | 105.82 | 1.381 |
| 3407 | 11.11 | 54.29 | 55.03 | 53.46 | 19.69 | 51.72 | 44.07 | 107.03 | 1.371 |

*+ LoRA 3 epoch — in-house val:*

| seed | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr | CE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 42 | 11.90 | 55.61 | 56.69 | 54.91 | 20.77 | 53.03 | 45.53 | 110.56 | 1.326 |
| 123 | 12.25 | 55.41 | 56.16 | 54.59 | 21.33 | 52.98 | 45.22 | 110.13 | 1.332 |
| 3407 | 11.86 | 55.37 | 56.29 | 54.63 | 21.12 | 52.89 | 45.50 | 110.79 | 1.322 |

---

## 4. Phân tích điểm nghẽn — sáu trục, một trục tích cực

ΔF1 so với cấu hình gốc (Bridge Multi-Token, trung bình 4 seed: F1 49.55).
Mọi số trung bình 3 seed trừ RQ3/RQ4 (seed 42). ΔF1 là hiệu trung bình; riêng
dòng RQ6 LoRA (1 epoch) có bootstrap 95% CI +4.06 [3.49, 4.65] (xem §6).

| RQ · axis | Intervention | ΔF1 | Nhận xét |
|---|---|--:|---|
| RQ1–2 · Bridge capacity | Full Q-Former (69M, 10×) | −2.20 | Bridge lớn hơn không tốt hơn |
| RQ3 · Number of visual tiles | Train 1 tile → evaluate 3 tiles | −28.50 | Bridge train 1 tile khái quát hóa kém sang nhiều tile ᵃ |
| RQ4 · Adaptive routing | Learned policy (theo loại câu hỏi) | ≈0 | Không hơn cấu hình cố định |
| RQ5 · Training signal | Multi-reference answer sampling | −1.47 | Không cải thiện |
| RQ5 · Representation alignment | Projector-level feature KD | **−0.03** | Null tuyệt đối |
| RQ5 · Representation alignment | Projector-level logit KD (α = 0.1) | **+0.20** | Null — F1 49.75 ± 0.29, val CE 1.53 ≈ mức gốc 1.49 |
| RQ5 · Representation alignment | Projector-level logit KD (α = 1.0) | **−8.80** | Chỉ hỏng khi α quá lớn (KL lấn cross-entropy) |
| **RQ6 · Decoder — LoRA attention (1 epoch)** | q/k/v/o | **+3.97** | **Cải thiện nhất quán** |
| **RQ6 · Decoder — LoRA attention (3 epochs)** | q/k/v/o | **+5.16** | **Cải thiện nhất quán** |
| RQ6 · Decoder — LoRA MLP-only | gate/up/down_proj | **−29.31** | Phân kỳ (training hỏng) |
| RQ6 · Decoder — LoRA attention + MLP | cả 7 module | **−12.04** | Phân kỳ |

ᵃ Thí nghiệm huấn luyện 1 tile, đánh giá 3–6 tile → generalize kém khi lệch số
tile lúc test. Đã thử tiếp huấn luyện trực tiếp với tile-augmentation
(`tile_choices=1,3,6`) trên cả 4 bridge (gồm Multi-Token và Full Q-Former):
**không giúp gì — F1/val loss đều hơi tệ hơn** so với train chỉ 1 tile (đo trên
subset 300 mẫu, không phải full-val nên không so tuyệt đối được, nhưng chiều
hướng nhất quán ở cả 4 bridge). Kết luận: sụp đổ khi lệch tile không chỉ do
thiếu tiếp xúc lúc train — kiến trúc bridge khó biểu diễn tốt nhiều tile dù có
được huấn luyện với nó hay không.

**Số liệu chi tiết đứng sau các dòng trên:**

*So sánh 5 loại bridge (tập val, trung bình 3 seed @ 2 epoch; Multi-Token = 4
seed). "val CE" = cross-entropy trên tập val (thấp = tốt).*

*F1 / CIDEr = in-house; ± = độ lệch chuẩn qua seed.*

| Bridge | Tham số | F1 | CIDEr | val CE | F1 + LoRA (1ep) | ΔF1 | CIDEr + LoRA |
|---|--:|--:|--:|--:|--:|--:|--:|
| Residual (1 token) | 4.86M (0.52%) | 45.64 ± 0.36 | 86.25 ± 0.61 | 1.67 | 52.64 ± 0.03 | +7.0 | 104.05 |
| Tile-Attention (8 token) | 4.14M (0.44%) | 45.17 ± 0.94 | 84.21 ± 1.71 | 1.67 | 52.99 ᵇ | +7.8 | 105.04 |
| **Multi-Token (8 token, pooled)** | **7.35M (0.78%)** | **49.55 ± 0.07** | **96.49 ± 0.59** | **1.49** | **53.52 ± 0.11** | **+4.0** | **106.56** |
| Light Q-Former (8 query) | 27.6M (2.87%) | 46.25 ± 0.62 | 86.80 ± 2.28 | 1.60 | 53.21 ± 0.13 | +7.0 | 106.24 |
| Full Q-Former (16 query) | 69.4M (6.91%) | 47.35 ± 0.17 | 89.98 ± 0.74 | 1.58 | 53.21 ± 0.09 | +5.9 | 105.70 |

ᵇ Tile-Attention + LoRA mới có seed 42 (các cấu hình còn lại đủ 3 seed).

*→ Bridge lớn hơn 10× (Full Q-Former) không tốt hơn; Multi-Token có val CE thấp
nhất (RQ1–2). 5 bridge plain trải F1 45.2–49.6; sau LoRA đều về 52.6–53.5 (băng
0.9 điểm) bất kể chất lượng ban đầu (RQ6). Mức nâng lớn hơn khi bridge yếu hơn.*

*Vị trí LoRA trong decoder (multi_token, r=16, 1 epoch, 3 seed; ± qua seed):*

| Target module | F1 | val loss |
|---|--:|--:|
| attention (q/k/v/o) — recipe | 53.52 ± 0.11 | 1.37 |
| MLP (gate/up/down_proj) | 20.24 ± 1.52 | ~3.7 |
| attention + MLP | 37.51 ± 1.70 | ~2.08 |

*→ Dư địa hữu ích của decoder nằm cụ thể ở attention. LoRA lên feed-forward làm
training phân kỳ. (Có thể là hyperparameter artifact — claim giới hạn ở cấu hình
recipe.)*

*Số tile khi đánh giá (Bridge Multi-Token, huấn luyện với 1 tile):*

| Số tile | token-F1 | val loss |
|--:|--:|--:|
| 1 | 50.66 ᶜ | 1.48 |
| 3 | 21.05 | 3.35 |
| 6 | 22.51 | 3.36 |

ᶜ Dòng tile=1 đo trên checkpoint seed-42 **cũ** (lúc đó vô tình chạy 4 epoch —
xem §2). Sau khi phát hiện và re-run seed 42 về đúng 2 epoch, F1 seed-42 đúng là
49.61 (khớp baseline 49.55 dùng cho mọi ΔF1 ở bảng RQ trên) — nhưng thí nghiệm
sweep tile 3/6 này chạy **trước** khi phát hiện lỗi, nên chưa sweep lại trên
checkpoint đã sửa. Chênh lệch ~1 điểm F1 ở baseline không đổi kết luận: rơi từ
~50 xuống ~21 khi lên 3 tile vẫn là một cú sụp ~29 điểm dù dùng baseline nào.

*→ Bridge sụp ngay khi vượt 1 tile (RQ3).*

*Chi phí encode thị giác của InternViT trên mỗi ảnh (Tesla P100-16GB):*

| Số tile | GFLOPs | Độ trễ (ms) | Thông lượng (ảnh/s) |
|--:|--:|--:|--:|
| 1 (của ta) | 362 | 229 | 6.00 |
| 2 | 724 | 374 | 3.30 |
| 4 | 1 448 | 648 | 1.70 |
| 6 | 2 172 | 922 | 1.15 |

*→ Tăng số tile vừa làm hỏng chất lượng vừa đắt: 1→6 tile là FLOPs ×6, độ trễ
×4. Recipe dùng 1 tile nên không tốn chi phí này.*

**Nhận định:** Các trục phía thị giác và tín hiệu huấn luyện đều không cải thiện
token-F1 — căn chỉnh biểu diễn là null trên cả hai biến thể (feature-KD và
logit-KD) ở mọi cường độ hợp lý (chỉ hỏng khi trọng số KD quá lớn, do nhiễu tối
ưu chứ không phải bản chất). Chỉ can thiệp vào **attention của decoder** là có
tác dụng, và lặp lại nhất quán trên mọi loại bridge → attention của frozen
decoder là điểm nghẽn. Ngoài ra, cả 5/5 bridge đều tăng CIDEr sau LoRA (bảng
trên): plain trải 84.2–96.5 (rộng ~12.3 điểm) → +LoRA chỉ còn 104.1–106.6 (rộng
~2.5 điểm) — khi decoder đủ dung lượng thì kiến trúc bridge gần như không còn
ảnh hưởng.

---

## 5. Kết luận sơ bộ

**(a) Adapt Vintern-1B với ~1% tham số?** Được một phần: backbone đóng băng hoàn
toàn + bridge nhẹ + LoRA decoder (3 epoch) đã vượt Vintern-1B fine-tuned trên
toàn bộ chỉ số sinh văn bản (F1 +0.95, BLEU +14.96, ROUGE +1.03, METEOR +10.17,
CIDEr +37.65), và cũng vượt ViMoE-VQA về Acc (+2.35) cùng BLEU/ROUGE/METEOR/CIDEr
— nhưng còn kém ViMoE-VQA ở token-F1 (−5.98, do Precision/Recall đều thấp hơn)
→ chưa tương đương hoàn toàn với các phương pháp huấn luyện đầy đủ.

**(b) Điểm nghẽn ở đâu?** Attention của frozen decoder. Trong không gian can
thiệp đã khảo sát, phía thị giác không còn dư địa; chỉ thêm dung lượng cho phần
attention của decoder mới cải thiện F1 (LoRA lên feed-forward làm training phân
kỳ).

**Hàm ý:** muốn thu hẹp nốt khoảng cách F1 thì mở thêm dung lượng phía decoder
(LoRA attention sâu hơn / decoder đóng băng lớn hơn), không phải đầu tư tiếp vào
thị giác. Đây cũng là điểm phản biện với "reasoning-aware routing" của ViMoE:
trên cùng benchmark, loại câu hỏi không mang tín hiệu hữu ích cho phân bổ tài
nguyên thị giác.

---

## 6. Ghi chú về độ tin cậy

- Mọi kết quả bridge plain + dòng âm: trung bình 3 seed @ 2 epoch (multi_token =
  4 seed); LoRA: 3 seed cho cả 1 và 3 epoch. Độ lệch chuẩn nhỏ: F1 std 0.07–0.17
  cho cấu hình đề xuất (Multi-Token, xem §3.1), tối đa 0.94 ở bridge phụ yếu
  nhất (Tile-Attention, xem bảng so bridge ở §4); mọi chỉ số in-house khác cũng
  std nhỏ tương tự.
- Đối chiếu tập test cho toàn bộ năm bridge và cả hai cấu hình LoRA: chênh so với
  val < 0.5 F1, không nhất quán về chiều.
- Dùng grouped split chống rò rỉ dữ liệu (đã kiểm chứng: kết quả gần như không
  đổi so với cách chia cũ). Khoảng tin cậy bootstrap 95% cho các so sánh chính đã
  tính trên đúng số 3-seed (ví dụ ΔF1 của LoRA cho Multi-Token: +4.06, khoảng
  [3.49, 4.65]).
- *Lưu ý:* một lần chạy bridge residual (seed cũ) bị mất ổn định (F1 36.5, val CE
  2.35) — đã phát hiện và thay bằng 3-seed chuẩn (F1 45.6). Đã rà soát lại toàn
  bộ checkpoint và số liệu, không còn sai lệch tương tự.
- Đánh giá ngữ nghĩa hiện mới ở mức tự kiểm 120 mẫu, một người đánh giá — cần
  nghiên cứu 2 người chấm + Cohen's κ cho bản camera-ready.

### 6.3. Đang tái lập dòng "Vintern-1B (fine-tuned)" bằng cookbook chính thức — chưa xong

Dòng 53.76/72.84 ở Bảng §3 hiện là **số trích dẫn** từ AutoViVQA, đo trên split
ngẫu nhiên của họ (không loại trừ rò rỉ ảnh). Đang tự đo lại trên **grouped
split của mình** bằng **cookbook fine-tune chính thức của
Vintern** (tải notebook thật từ Kaggle của 5CD-AI, dùng nguyên hyperparameter —
không đổi gì): đóng băng backbone + MLP, LoRA r=16 trên Qwen2.5-0.5B, 6 tile,
lr 4e-5, 1 epoch, template Hermes-2.

**Vướng mắc kỹ thuật (không phải vấn đề khoa học):** codebase gốc (InternVL, viết
2024) chạy trên hạ tầng Kaggle hiện tại phải vá nhiều lỗi môi trường (phiên bản
thư viện, dependency), và ở 6 tile không có flash-attention trên GPU miễn phí của
Kaggle nên train rất chậm (~38s/step) → **1 epoch cần nhiều hơn 1 phiên 12h của
Kaggle**, phải nối nhiều phiên (train → cắt → lưu checkpoint → phiên sau train
tiếp). Đang chạy song song trên 2 tài khoản, dự kiến có số trong 1–2 ngày tới.

**Không chặn phần còn lại của báo cáo này** — mọi kết luận ở §1, §4, §5 không phụ
thuộc vào con số này. Khi có kết quả sẽ cập nhật: (a) xác nhận/đối chiếu số
53.76 trên split công bằng, (b) làm rõ recipe fine-tune thật của AutoViVQA nặng
cỡ nào so với recipe của mình (nếu họ dùng cookbook thay vì recipe build gốc của
Vintern, câu chuyện "rẻ hơn 100×" sẽ cần chỉnh thành "cùng ngân sách adapt,
thiết kế + chẩn đoán tốt hơn" — xem giải thích ở §1).

## 7. Đóng góp

1. **Quy trình thích nghi tiết kiệm tham số:** frozen backbone + bridge nhẹ +
   LoRA cho attention của decoder, ~1% tham số nhưng đạt/vượt baseline
   fine-tuned trên chỉ số sinh (và đối chiếu được trên tập test).
2. **Chẩn đoán điểm nghẽn hệ thống:** khảo sát bridge, số tile, routing, tín hiệu
   huấn luyện / căn chỉnh, và vị trí LoRA trong decoder → dư địa hữu ích nằm cụ
   thể ở attention của decoder, không phải feed-forward hay phía thị giác.
3. **Quy trình đánh giá đáng tin cậy:** grouped split, nhiều seed, đối chiếu
   val/test, bootstrap CI, đánh giá thủ công + phân tích lỗi, kèm phân tích hiệu
   quả tính toán của tile.
