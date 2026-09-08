# Báo cáo tiến độ nghiên cứu — Paper 3

*Cập nhật 08/09/2026. Toàn bộ thí nghiệm đã hoàn tất, số liệu đã rà soát chéo.*

---

## 1. Câu hỏi nghiên cứu

> Thay vì xây một mô hình mới (như ViMoE-VQA), có thể cải thiện Vintern-1B trên
> AutoViVQA bằng cách chỉ cập nhật khoảng 1% tham số (đóng băng toàn bộ backbone)
> mà vẫn ngang fine-tune không? Nếu chưa, điểm nghẽn (bottleneck) ở đâu?

| Công trình | Cách làm | Tham số cập nhật |
|---|---|---|
| Vintern-1B (fine-tuned) | Fine-tune toàn bộ InternViT-300M + projector; LoRA cho Qwen2-0.5B | Phần lớn phía thị giác + projector |
| ViMoE-VQA | Xây kiến trúc Mixture-of-Experts mới | Toàn bộ mô hình mới |
| **Nghiên cứu này** | Đóng băng cả InternViT-300M lẫn Qwen2-0.5B; chỉ huấn luyện bridge (0.78%) + LoRA cho decoder (0.23%), 1 tile | **~1% tổng tham số** |

**Trả lời ngắn gọn:** đạt được *một phần* — vượt Vintern-1B fine-tuned trên mọi
chỉ số sinh văn bản với ~1% tham số, nhưng vẫn kém ViMoE-VQA ở token-F1. Điểm
nghẽn nằm ở **attention của frozen decoder**: chỉ can thiệp vào đó mới cải thiện F1.

---

## 2. Thiết lập huấn luyện

- **Backbone** (InternViT-300M + Qwen2-0.5B): đóng băng hoàn toàn ở mọi cấu hình.
- **Giai đoạn 1 — huấn luyện bridge:** 2 epoch, batch 8, learning rate 2e-4, ảnh
  1 tile; khoảng 5 giờ/lần trên một GPU Tesla P100-16GB. (CIDEr bão hòa từ epoch
  2, thêm epoch không cải thiện đáng kể.)
- **Giai đoạn 2 — LoRA cho decoder** (áp lên `q/k/v/o` của Qwen2, r=16): huấn
  luyện thêm trên bridge đã cố định. Thử 1 epoch và 3 epoch; **3 epoch cho kết
  quả tốt nhất**, là cấu hình dùng trong bảng kết quả.
- **Đánh giá:** toàn bộ tập validation (5 463 mẫu) và tập test (5 468 mẫu), không
  lấy mẫu con. Mỗi cấu hình chính chạy 3–4 seed, báo trung bình ± độ lệch chuẩn.

---

## 3. Kết quả chính (tập validation, chỉ số nội bộ, thang ×100)

| Mô hình | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Vintern-1B (gốc, zero-shot) | 0.12 | 17.52 | 19.87 | 17.55 | 1.91 | 25.84 | 23.93 | 8.54 |
| ViT5_ViT | 7.97 | 46.84 | 50.33 | 48.52 | 4.13 | 46.89 | 31.02 | 72.68 |
| BARTPhoBEiT | 8.81 | 45.30 | 46.48 | 45.88 | 4.33 | 44.83 | 24.57 | 188.96 ᵃ |
| Vintern-1B (fine-tuned) | 13.01 | 52.47 | 55.12 | 53.76 | 6.11 | 51.93 | 35.25 | 72.84 |
| Llama 3.2 (zero-shot) | 0.36 | 23.96 | 73.71 | 36.16 | 3.62 | 36.11 | 30.01 | 62.84 |
| Gemini 2.0 Flash | 0.55 | 27.20 | 74.10 | 39.79 | 4.41 | 39.60 | 31.72 | 74.42 |
| Gemini 2.5 Flash | 0.22 | 24.43 | 76.66 | 24.75 | 0.39 | 37.27 | 31.22 | 71.90 |
| GPT-5 (zero-shot) | 10.84 | 47.20 | 55.20 | 50.89 | 6.07 | 47.30 | 33.34 | 84.20 |
| ViMoE-VQA | 9.65 | 62.89 | 58.65 | 60.69 | 12.54 | 47.07 | 39.10 | 88.67 |
| **Bridge Multi-Token (0.78%, 1 tile)** | **8.20** | **50.36** | **51.43** | **49.55** | **15.47** | **47.84** | **40.22** | **96.49** |
| **  + LoRA cho decoder, r=16 (~1.0%), 1 epoch** | **10.93** | **54.39** | **55.11** | **53.52** | **19.72** | **51.81** | **44.11** | **106.56** |
| **  + LoRA cho decoder, r=16, 3 epoch** | **12.00** | **55.46** | **56.38** | **54.71** | **21.07** | **52.96** | **45.42** | **110.49** |

*In đậm = phương pháp đề xuất (trung bình 4 seed cho bridge / 3 seed cho LoRA).
ᵃ CIDEr của BARTPhoBEiT là ngoại lai (sinh câu dài), không so sánh. Baseline lấy
theo báo cáo benchmark AutoViVQA.*

Đo theo corpus (để so với công trình khác): Bridge Multi-Token đạt CIDEr-D
92.3 ± 0.6 / BLEU-4 18.9 / ROUGE-L 48.9 — trên mức 88.7 / 12.5 / 47.1 của ViMoE;
thêm LoRA 3 epoch đạt CIDEr-D 107.5 ± 0.3 / BLEU-4 25.1 / ROUGE-L 54.2 (1 epoch:
103.2 ± 0.5). Điểm yếu còn lại: token-F1 (−6.0 so với ViMoE) và Acc vẫn thấp hơn.

**Đối chiếu tập test:** Bridge Multi-Token (4 seed) F1 **49.20** / CIDEr **93.24**
— chênh so với val −0.35 / −3.25. Recipe cũng giữ vững trên test: + LoRA 1 epoch
F1 **53.15** (val 53.52), + LoRA 3 epoch F1 **54.28** (val 54.71). Cả năm loại
bridge: test ≈ val trong khoảng ±0.5 F1, không nhất quán về chiều → **không
overfit vào tập validation**.

---

## 4. Phân tích điểm nghẽn — sáu trục, một trục tích cực

ΔF1 so với cấu hình gốc (Bridge Multi-Token, trung bình 4 seed: F1 49.55).
Mọi số trung bình 3 seed trừ RQ3/RQ4 (seed 42).

| RQ · axis | Intervention | ΔF1 | Nhận xét |
|---|---|--:|---|
| RQ1–2 · Bridge capacity | Full Q-Former (69M, 10×) | −2.20 | Bridge lớn hơn không tốt hơn |
| RQ3 · Number of visual tiles | Train 1 tile → evaluate 3 tiles | −28.50 | Bridge train 1 tile khái quát hóa kém sang nhiều tile ᵃ |
| RQ4 · Adaptive routing | Learned policy (theo loại câu hỏi) | ≈0 | Không hơn cấu hình cố định |
| RQ5 · Training signal | Multi-reference answer sampling | −1.47 | Không cải thiện |
| RQ5 · Representation alignment | Projector-level feature KD | **−0.03** | Null tuyệt đối |
| RQ5 · Representation alignment | Projector-level logit KD (α = 0.1) | **+0.20** | Null (val CE 1.53 ≈ mức gốc 1.49) |
| RQ5 · Representation alignment | Projector-level logit KD (α = 1.0) | **−8.80** | Chỉ hỏng khi α quá lớn (KL lấn cross-entropy) |
| **RQ6 · Decoder — LoRA attention (1 epoch)** | q/k/v/o | **+3.97** | **Cải thiện nhất quán** |
| **RQ6 · Decoder — LoRA attention (3 epochs)** | q/k/v/o | **+5.16** | **Cải thiện nhất quán** |
| RQ6 · Decoder — LoRA MLP-only | gate/up/down_proj | **−29.31** | Phân kỳ (training hỏng) |
| RQ6 · Decoder — LoRA attention + MLP | cả 7 module | **−12.04** | Phân kỳ |

ᵃ Thí nghiệm huấn luyện 1 tile, đánh giá 3–6 tile → chỉ kết luận về khả năng
khái quát hóa; chưa khảo sát huấn luyện đa tile.

**Số liệu chi tiết đứng sau các dòng trên:**

*So sánh 5 loại bridge (tập val, trung bình 3 seed @ 2 epoch; Multi-Token = 4
seed). "val CE" = cross-entropy trên tập val (thấp = tốt).*

| Bridge | Tham số | F1 | CIDEr | val CE | F1 + LoRA | ΔF1 | CIDEr + LoRA |
|---|--:|--:|--:|--:|--:|--:|--:|
| Residual (1 token) | 4.86M (0.52%) | 45.64 | 86.25 | 1.67 | 52.64 | +7.0 | 104.05 |
| Tile-Attention (8 token) | 4.14M (0.44%) | 45.17 | 84.21 | 1.67 | 52.99 | +7.8 | 105.04 |
| **Multi-Token (8 token, pooled)** | **7.35M (0.78%)** | **49.55** | **96.49** | **1.49** | **53.52** | **+4.0** | **106.56** |
| Light Q-Former (8 query) | 27.6M (2.87%) | 46.25 | 86.80 | 1.60 | 53.21 | +7.0 | 106.24 |
| Full Q-Former (16 query) | 69.4M (6.91%) | 47.35 | 89.98 | 1.58 | 53.21 | +5.9 | 105.70 |

*→ Bridge lớn hơn 10× (Full Q-Former) không tốt hơn; Multi-Token có val CE thấp
nhất (RQ1–2). 5 bridge plain trải F1 45.2–49.6; sau LoRA đều về ≈53 F1 bất kể
chất lượng ban đầu (RQ6). Mức nâng lớn hơn khi bridge yếu hơn.*

*Vị trí LoRA trong decoder (multi_token, r=16, 1 epoch, 3 seed):*

| Target module | F1 | val loss |
|---|--:|--:|
| attention (q/k/v/o) — recipe | 53.52 | 1.37 |
| MLP (gate/up/down_proj) | 20.24 | ~3.7 |
| attention + MLP | 37.51 | ~2.08 |

*→ Dư địa hữu ích của decoder nằm cụ thể ở attention. LoRA lên feed-forward làm
training phân kỳ. (Có thể là hyperparameter artifact — claim giới hạn ở cấu hình
recipe.)*

*Số tile khi đánh giá (Bridge Multi-Token, huấn luyện với 1 tile):*

| Số tile | token-F1 | val loss |
|--:|--:|--:|
| 1 | 50.66 | 1.48 |
| 3 | 21.05 | 3.35 |
| 6 | 22.51 | 3.36 |

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
decoder là điểm nghẽn. Ngoài ra, các bridge vốn chênh lệch CIDEr-D 79–92 đều hội
tụ về ~101–103 sau khi thêm LoRA — khi decoder đủ dung lượng thì kiến trúc bridge
gần như không còn ảnh hưởng.

---

## 5. Kết luận sơ bộ

**(a) Adapt Vintern-1B với ~1% tham số?** Được một phần: backbone đóng băng hoàn
toàn + bridge nhẹ + LoRA decoder (3 epoch) đã vượt Vintern-1B fine-tuned trên
toàn bộ chỉ số sinh văn bản (F1 +0.95, BLEU +14.96, ROUGE +1.03, METEOR +10.17,
CIDEr +37.65), nhưng còn kém ViMoE-VQA ở token-F1 (−5.98) và Acc → chưa tương
đương hoàn toàn với các phương pháp huấn luyện đầy đủ.

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
  4 seed); LoRA: 3 seed cho cả 1 và 3 epoch. Độ lệch chuẩn nhỏ (F1 std 0.07–0.94
  cho bridge, 0.11–0.29 cho LoRA và các dòng RQ5).
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
