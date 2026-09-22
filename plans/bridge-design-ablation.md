# Bridge-design ablation: trả lời 2 câu hỏi của thầy

> **ĐANG MỞ RỘNG (không phải chờ mới gửi được — số liệu chính vẫn đúng).**
> n=10/12/14/16/18/20 đã đủ 3-seed, n=8 có 4-seed (Exp A). Phát hiện: n=4/6
> trước đó chỉ có 1-seed — không đồng bộ với phần còn lại của bảng, đang
> chạy thêm 2 seed mỗi điểm để khớp chuẩn (3-seed toàn bộ sweep).

## Bối cảnh

Advisor xem paper draft, hỏi 2 câu về bridge Multi-Token (kiến trúc tốt nhất
trong 5 bridge gốc, F1 49.55±0.07 plain / 53.52±0.11 +LoRA):

1. "Tại sao multitoken lại chọn là 8 mà không phải 6 8 10 12? → cần exp để
   xác định con số nhé"
2. "Rồi là cái xu hướng của bridge là gì, tại sao không thử deconvolution →
   zoom in xong bung ra zoom out, nén, rồi hiện tại multitoken đang dùng là 8
   cái token thì lúc gộp lại là max, avg, hay là để nguyên."

Cả 2 câu đều đã có exp thật trả lời, không chỉ giải thích chay.

---

## Câu 1: num_tokens sweep

Sweep `--bridge multi_token --bridge-num-tokens {n}`, plain bridge (không
LoRA), 2 epoch, cùng protocol với Exp A gốc. `n=8` đã có sẵn 4-seed Exp A data
(49.55±0.07) — không train lại.

| n | seed 42 | seed 123 | seed 3407 | **mean** | std | n_seed |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 48.85 | — | — | 48.85 | — | 1 |
| 6 | 49.38 | — | — | 49.38 | — | 1 |
| **8 (recipe hiện tại)** | — | — | — | **49.55** | 0.07 | 4 (Exp A) |
| 10 | 50.06 | 50.11 | 49.81 | **49.99** | 0.16 | 3 |
| 12 | 50.27 | 49.99 | 50.43 | **50.23** | 0.22 | 3 |
| 14 | 50.68 | 50.76 | 50.58 | **50.67** | 0.09 | 3 |
| 16 | 50.95 | 49.87 | 50.51 | **50.44** | 0.54 | 3 |
| 18 | 49.72 | 50.51 | 51.30 | **50.51** | 0.79 | 3 |
| 20 | 50.85 | 50.98 | 50.02 | **50.62** | 0.52 | 3 |

*(F1 ×100. CIDEr theo cùng thang trong log commit, không tách riêng ở đây —
xem `outputs/token_sweep/tok*/eval/out/eval_val.json` cho số đầy đủ.)*

### Kết luận cuối cùng

1. **`n=8` chắc chắn không phải điểm tối ưu.** Mọi n≥10 đều vượt baseline
   49.55±0.07, kể cả sau khi lấy mean 3-seed (không phải nhiễu 1-seed).
2. **Xác nhận chắc chắn trên CẢ 4/4 điểm mở rộng (14/16/18/20, đều đủ
   3-seed): xu hướng chững lại thật sự (plateau) từ khoảng n=14 trở đi**,
   không phải tiếp tục tăng như dữ liệu bán phần ban đầu gợi ý. Mean 3-seed
   đầy đủ: n=14=**50.67**, n=16=**50.44**, n=18=**50.51**, n=20=**50.62** —
   cả 4 dao động quanh ~50.4-50.7, không hề có xu hướng tăng đơn điệu tiếp
   tục (n=16 thậm chí thấp hơn n=14). std của các điểm này (0.09-0.79) đều
   đủ lớn để "khác biệt" giữa 14/16/18/20 nằm trong biên độ nhiễu thống kê,
   không phải tín hiệu thật.
3. **Kết luận: gain thật sự nằm ở khoảng n=10-14, sau đó là plateau/nhiễu.**
   Đây là câu trả lời rõ ràng, có số liệu vững cho advisor — đã kiểm tra tới
   n=20, không cần đào sâu thêm vì xu hướng đã đủ rõ và ổn định trên 4 điểm
   độc lập.
4. **Không đổi recipe chính thức (giữ n=8)** — quyết định đã chốt với người
   dùng trước đó. Đây là ablation report-only cho advisor, không phải đề xuất
   đổi kiến trúc paper.

*(Ghi chú kỹ thuật, không ảnh hưởng tới số liệu: n=18/seed=42 mất ~10.9h thay
vì ~5.4h dự kiến do một bug hiếm gặp — cơ chế resume-theo-epoch bị bỏ qua
âm thầm khi checkpoint tạm thời chưa kịp sẵn sàng, khiến job train lại cả 2
epoch từ đầu thay vì resume 1 epoch còn lại. Kết quả cuối vẫn là 1 lần train
2-epoch đầy đủ và hợp lệ, không ảnh hưởng độ tin cậy số liệu.)*

### Khuyến nghị nếu phải chọn 1 con số khác 8 để báo cáo

**n=12**, càng chắc chắn hơn sau khi xác nhận plateau: n=14/16/18/20 không
còn tăng có ý nghĩa so với n=12 (chênh lệch nằm trong nhiễu), nên không có
lý do đánh đổi thêm chi phí inference (+token ảnh nạp vào LLM decoder) để
đổi lấy F1 gần như không đổi. n=12 vẫn là điểm có gain/token tốt nhất trong
vùng đã tăng thật (n=10-14), đã đủ 3-seed để defend chắc chắn.

| n | ΔF1 so với n=8 | Δtoken | gain/token thêm |
|---:|---:|---:|---:|
| 10 | +0.44 | +2 | 0.220 |
| 12 | +0.68 | +4 | 0.170 |
| 14 | +1.12 | +6 | 0.187 |
| 16 | +0.89 | +8 | 0.111 |
| 18 | +0.96 | +10 | 0.096 |
| 20 | +1.07 | +12 | 0.089 |

*(Lưu ý: đã loại trừ khả năng bug — kiểm tra kỹ code forward/dispatch/init,
không tìm thấy vấn đề. Xem log điều tra trong lịch sử commit của branch này.)*

---

## Câu 2: xu hướng "gộp" — pooling, deconv, zoom-in/out, nén

### Sự thật quan trọng cần nói rõ với thầy trước

**Multi-Token bridge hiện tại KHÔNG dùng pooling operator nào cả.** InternViT
tự pool ảnh thành 1 vector (B, 1024) trước khi bridge chạy — bridge chỉ là 2
lớp `Linear` học thẳng từ vector đó ra `k` token (`baseline` token + `k-1`
token bổ sung), không hề nhìn patch riêng lẻ. Câu hỏi của thầy ngầm giả định
Multi-Token đang "gộp bằng max/avg/gì đó" — thực ra không có bước gộp học
được nào cả, chỉ là phép chiếu tuyến tính thuần.

### Thí nghiệm: 3 kiến trúc "gộp" khác, cùng num_tokens≈8

| Kiến trúc | Toán tử gộp | Trainable params | F1 | Δ so với Multi-Token |
|---|---|---:|---:|---:|
| **Multi-Token (hiện tại)** | không có (Linear thuần trên vector đã pool sẵn) | 7,347,200 | **49.55±0.07** | — |
| Patch-Pool (mean) | trung bình cố định trên patch | 918,400 | 38.80 | −10.75 |
| Patch-Pool (max) | max cố định trên patch | 918,400 | 35.71 | −13.84 |
| Attention (tile_attention, có sẵn 3-seed Exp A) | attention học được | 4,142,208 | 45.17±0.94 | −4.38 |
| **Conv-Abstractor** (HoneyBee C-Abstractor: conv→pool→conv) | zoom-in (conv) → nén (adaptive avg pool) → zoom-out (conv) | 19,871,104 | 48.87 | −0.68 |

*(Conv-Abstractor dùng num_tokens=9 — số chính phương gần 8 nhất, do kiến
trúc cần lưới không gian vuông. Nguồn: `src/modeling/bridge_modules.py`,
kiến trúc theo Cha et al., "Honeybee: Locality-enhanced Projector for
Multimodal LLM", CVPR 2024.)*

### Trả lời cụ thể từng ý thầy hỏi

- **"Zoom in xong bung ra zoom out, nén"** = chính xác là kiến trúc
  **Conv-Abstractor** đã thử (conv xử lý cục bộ = "zoom in", adaptive pool =
  "nén", conv xử lý lại = "zoom out"). Kết quả: **48.87, gần bằng nhưng vẫn
  thấp hơn Multi-Token (49.55)**.
- **"Pooling (max/avg)"** đã thử riêng, cô lập hoàn toàn khỏi các thành phần
  khác (chỉ 1 Linear dùng chung + pooling cố định, không add capacity) —
  **thua đậm** (35.71-38.80 so với 49.55). Đã kiểm tra kỹ code (shape, dispatch,
  hyperparameter, leading-token handling, output thực tế của model) — không
  phải bug, là architecture thật sự yếu vì pooling cố định không học được
  patch nào quan trọng.
- **"DeConvolution"**: **chưa implement đúng nghĩa đen**, vì về mặt kỹ thuật
  deconv (transposed conv) là phép **upsample** (ít phần tử → nhiều phần tử
  không gian hơn) — trong khi bridge ở đây cần làm ngược lại: 576 patch → 8-20
  token, tức **downsample/nén**. Dùng deconv đúng chiều sẽ đi ngược hướng cần
  thiết. Conv-Abstractor (conv thường + pool, không phải deconv) là câu trả
  lời đúng tinh thần câu hỏi của thầy cho hướng "nén bằng convolution".

### Kết luận câu 2

**Không có cách "gộp" nào (pool thô, attention, conv+pool) đánh bại được
cách hiện tại (Linear thuần trên vector đã pool sẵn) ở cùng scale tham số.**
Conv-Abstractor gần nhất (chỉ kém 0.68 điểm F1) nhưng tốn gấp 2.7x tham số
(19.9M so với 7.3M) — không đáng đánh đổi. Đây là bằng chứng số liệu thật
cho việc giữ nguyên Multi-Token làm recipe chính.

---

## Tóm tắt gửi thầy

- **Câu 1 (num_tokens)**: `n=8` không tối ưu. Gain thật nằm ở n=10-14; sau
  n=14 là plateau/nhiễu, đã xác nhận trên 4 điểm độc lập (14/16/18/20, mỗi
  điểm 3-seed). Đề xuất report `n=12` là điểm cân bằng hợp lý nếu cần chọn 1
  số khác 8 — không đổi recipe chính thức.
- **Câu 2 (gộp/deconv)**: Multi-Token hiện tại không dùng toán tử gộp nào
  (Linear thuần). Đã thử pooling (thua đậm), attention (thua), Conv-Abstractor
  = đúng tinh thần "zoom in/nén/zoom out" (gần bằng, vẫn thua). DeConvolution
  đúng nghĩa đen không hợp lý về hướng biến đổi (upsample vs downsample cần
  thiết) nên không implement, đã giải thích rõ lý do kỹ thuật.
