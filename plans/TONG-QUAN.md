# Paper 3 — Tổng quan

*Cập nhật 2026-09-03 · Trust4NLP @ ACIIDS 2027 · deadline 27/09/2026*

---

## 0. Một đoạn: đây là cái gì

Mình có một model VQA tiếng Việt (trả lời câu hỏi về ảnh). Trước khi trả lời, model
phải **chọn cách xử lý ảnh**: nhìn thô cho nhanh, hay cắt nhỏ nhìn kỹ cho tốn kém?

Paper 3 hỏi: **có thể dựa vào "loại suy luận" của câu hỏi (đếm / quan hệ / nhân quả...)
để đoán câu nào đáng nhìn kỹ, câu nào không — nhằm tiết kiệm tính toán mà vẫn giữ chất
lượng?**

Câu trả lời sau toàn bộ thí nghiệm: **KHÔNG.** Loại suy luận không dự đoán được điều đó;
mà thực ra với model nhẹ này, "nhìn kỹ" cũng gần như không giúp trả lời tốt hơn. Cứ chọn
một cấu hình tốt rồi dùng cho mọi câu là đủ.

Nhưng trên đường đi mình thu được vài thứ chắc chắn có giá trị: một **bridge** nhỏ
(7M tham số) vượt SOTA cũ trên các metric sinh, một **split sạch** không rò rỉ, và một
**kết quả negative nghiêm ngặt** kiểm tra thẳng claim "reasoning-aware" của ViMoE-VQA.

---

## 1. Bức tranh lớn

```
                       +---------------------------------------------+
   Ảnh I  ------------>|  InternViT-300M  (ĐÓNG BĂNG)                 |
   "con chó ngoài sân" |     nhìn ảnh, 1 hoặc 3 hoặc 6 mảnh (n_tiles) |
                       +----------------------+----------------------+
                                              | patch token
                                              v
                       +---------------------------------------------+
                       |  BRIDGE  (PHẦN DUY NHẤT ĐƯỢC TRAIN)          |
                       |     nén thành vài "vision token"             |
                       |     3 kiểu: multi_token / qformer / mini_qf  |
                       +----------------------+----------------------+
                                              v
                       +---------------------------------------------+
   Câu hỏi Q  -------->|  Qwen2-0.5B  (ĐÓNG BĂNG)                     |
   "Có mấy con chó?"   |     đọc vision token + câu hỏi -> sinh câu   |
                       +----------------------+----------------------+
                                              v
                                     "Có hai con chó"

   -- song song, RẺ ------------------------------------------------------
   Câu hỏi Q  --> Router PhoBERT --> loại suy luận r      (vd: "đếm")
   Ảnh + hỏi  --> probe rẻ f(I,Q) --> vài đặc trưng ảnh nhìn-lướt
```

**2 nút vặn khi xử lý ảnh** — gộp lại gọi là **action**:

| Nút | Lựa chọn | Ảnh hưởng |
|---|---|---|
| `n_tiles` | 1 / 3 / 6 mảnh | 6 mảnh = chi tiết hơn nhưng **chậm gấp ~4 lần** |
| `bridge` | multi_token / qformer / mini_qformer | kiểu nén vision token khác nhau |

→ `3 x 3 = 9` action. Model phải chọn 1 trong 9 cho mỗi câu hỏi.

---

## 2. Bảng ký hiệu (đọc 1 lần rồi tham chiếu lại)

| Ký hiệu | Nghĩa | Ví dụ |
|---|---|---|
| **x** | 1 mẫu = (ảnh + câu hỏi) | mẫu #4213 |
| **I** | phần **ảnh** | ảnh con chó ngoài sân |
| **Q** | phần **câu hỏi** (chữ) | "Có mấy con chó?" |
| **r** | **loại suy luận** câu hỏi đòi hỏi (8 loại) | `counting` (đếm) |
| **P(r\|Q)** | router đọc Q, đoán phân phối xác suất trên 8 loại r. Không nhìn ảnh | `{đếm: 0.92, không_gian: 0.05, ...}` |
| **f(I,Q)** | vector đặc trưng **rẻ**: nhìn ảnh lướt 1 lần + độ dài câu + độ nét / che khuất / mật độ vật thể | `[64 số ảnh, 8, 0.9, 0.1, 0.7]` |
| **a** | **action** = (n_tiles, bridge) | `(3 mảnh, qformer)` |
| **M(a;x)** | action a trả lời mẫu x **đúng tới đâu** — đo bằng CIDEr (0–2) | `M = 0.88` |
| **C(a)** | action a **tốn compute bao nhiêu** = n_tiles / 6, thang [0,1] | `C = 3/6 = 0.5` |
| **λ** (lambda) | con số **mình tự đặt**: "ghét tốn compute tới mức nào". Nhỏ = chấp nhận tốn; lớn = phải tiết kiệm | quét 0 → 1 |
| **a\*(x,λ)** | action **tốt nhất** cho mẫu x ở mức λ đó = `argmax_a [M − λ·C]` | `qformer\|3` |
| **π** (policy) | mạng nhỏ phải **đoán** action, chỉ từ P(r\|Q), f(I,Q), λ — **không được chạy thử** | `π(...) -> multi_token\|1` |

---

## 3. Oracle + action đang làm gì? (giải thích kỹ)

### Bước 1 — Với MỖI câu hỏi, chạy thử CẢ 9 action, ghi điểm

Ví dụ *"Có mấy con chó trong ảnh?"* (mẫu #4213):

| action | M (chất lượng) | C (chi phí) |
|---|---:|---:|
| multi_token \| 1 mảnh | 0.85 | 0.17 |
| multi_token \| 3 mảnh | 0.40 | 0.50 |
| qformer \| 1 mảnh | 0.80 | 0.17 |
| qformer \| 3 mảnh | 0.88 | 0.50 |
| qformer \| 6 mảnh | 0.91 | 1.00 |
| ... (9 dòng) | ... | ... |

Làm việc này cho **toàn bộ** train (5547 câu) + val (3727) + test (3739). Đây là
"**oracle sweep**" — phần tốn GPU nhất của cả paper.

### Bước 2 — "Oracle" = kẻ biết trước bảng đó

Oracle luôn chọn được action tốt nhất cho từng câu. **Không hệ thống thật nào làm được**
(phải chạy thử hết mới biết). Oracle chỉ là **trần lý thuyết** để so sánh.

Công thức oracle chọn: `a*(x,λ) = action có [M − λ·C] cao nhất`

- `λ = 0` (bất chấp chi phí): chọn `qformer|6` vì M = 0.91 cao nhất
- `λ = 0.2` (tiếc compute): `qformer|6` -> 0.91 − 0.2 × 1.0 = 0.71; `qformer|3` -> 0.88 − 0.2 × 0.5 = **0.78** -> chọn `qformer|3`
- `λ = 0.7` (rất tiếc): action 1 mảnh thắng

→ quét λ để vẽ đường "chất lượng đổi lấy chi phí".

### Bước 3 — Câu hỏi: một router THẬT có bắt kịp oracle không?

Train **policy π** — mạng nhỏ đoán action chỉ từ `P(r|Q)`, `f(I,Q)`, `λ`, học bằng
cách bắt chước lựa chọn `a*` của oracle trên tập train, rồi test trên tập test (không
gian lận).

**3 phiên bản** để tách xem tín hiệu nào quan trọng:

| Policy | Được xem | Kiểm tra điều gì |
|---|---|---|
| `ours` | loại suy luận **+** đặc trưng ảnh | (đầy đủ) |
| `rt_only` | **chỉ** loại suy luận r | ← giả thuyết kiểu ViMoE: loại câu hỏi là tín hiệu đủ |
| `visual_only` | **chỉ** đặc trưng ảnh f | ← chỉ tín hiệu nội tại (model tự liếc, không cần nhãn) |

Cách đọc: nếu `rt_only` ≈ `ours` >> `visual_only` -> loại suy luận là tín hiệu chính.
Nếu `visual_only` ≈ `ours` >> `rt_only` -> reasoning-type thừa. Nếu cả 3 ≈ nhau ≈ fixed
-> không có gì để route.

### Bước 4 — Kết quả: cả 3 đều thất bại

| Policy | mean CIDEr (test) | Chọn gì |
|---|---:|---|
| Oracle (trần, gian lận) | **1.25** | — |
| **Fixed: luôn dùng `multi_token\|1`** | **0.90** | 1 action cố định |
| `ours` | 0.90 | multi_token\|1 (94%) |
| `rt_only` | 0.90 | multi_token\|1 (**100%**) |
| `visual_only` | 0.82 | lệch, tốn compute hơn |
| Random | 0.77 | — |

- **Không policy nào thắng "fixed"** — cả 3 tự học ra rằng nước đi tốt nhất là *luôn chọn
  một action*, không route gì cả.
- **`rt_only` ≈ `visual_only` ≈ `ours`** -> biết loại suy luận **không thêm gì**.
- Oracle *trông* hơn +40% nhưng đó là **nhiễu**: CIDEr chấm trên câu 4 từ dao động mạnh,
  oracle chỉ ăn may. Bằng chứng: phân phối `a*` của val và test **khác hẳn nhau** dù chia
  cùng cách -> không có quy luật để học.

---

## 4. Dữ liệu

**AutoViVQA**: 19,411 ảnh / 37,077 câu / mỗi câu **5 đáp án đa dạng** / có nhãn loại suy luận.

**Split mới — grouped 70/15/15**: chia theo `image_id` -> **0 ảnh trùng giữa train / val / test**
(chặn rò rỉ qua caption / bối cảnh chung ảnh; bản gốc chỉ có 80/20, không có test).

| Split | Câu hỏi | Ảnh |
|---|---:|---:|
| Train | 25,933 | 13,576 |
| Val | 5,544 | 2,908 |
| Test | 5,503 | 2,914 |

8 loại suy luận: relational ~30%, recognition ~19%, spatial ~15%, causal ~13%,
counting ~12%, action / context / yesno (đuôi).

---

## 5. Kết quả

### 5A. So 5 bridge (val, metric chuẩn pycocoevalcap corpus)

| Bridge | Tham số train | % | **CIDEr-D** | **BLEU-4** | **ROUGE-L** |
|---|---:|---:|---:|---:|---:|
| **Multi-Token** | 7.35M | 0.78% | **94.4** | **19.6** | **50.0** |
| Full Q-Former | 69.4M | 6.91% | 86.7 | 17.5 | 47.1 |
| Light Q-Former | 27.6M | 2.87% | 83.8 | 16.8 | 46.0 |
| Tile-Attention | 4.14M | 0.44% | 82.7 | 16.3 | 46.1 |
| Residual (1 token) | 4.86M | 0.52% | 56.3 | 8.1 | 36.0 |

→ **multi_token tốt nhất mọi mặt, chỉ 0.78% tham số.** Residual (nén ảnh về 1 vector) thua
xa -> biểu diễn ảnh 1 token là nút thắt.

### 5B. Multi-Token vs các model trước trên AutoViVQA

| Model | Acc | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|---:|---:|---:|---:|---:|---:|
| Vintern-1B (base) | 0.1 | 17.6 | 1.9 | 25.8 | 23.9 | 8.5 |
| Vintern-1B (**finetune toàn bộ**) | **13.0** | 53.8 | 6.1 | **51.9** | 35.3 | 72.8 |
| GPT-5 (zero-shot) | 10.8 | 50.9 | 6.1 | 47.3 | 33.3 | 84.2 |
| **ViMoE-VQA / Tuong-MoE** (5 seed) | 9.7 | **60.7** | 12.5 | 47.1 | **39.1** | 88.7 |
| **★ Multi-Token Bridge (của mình)** | 8.6 | 44–51* | **19.6** | **50.0** | ~28–41* | **94.4** |

<small>* F1 / METEOR chênh theo implementation, không so được chắc chắn. BARTPhoBEiT bỏ khỏi bảng: CIDEr 189 là outlier do sinh câu dài.</small>

**THẮNG rõ** (metric sinh, ổn định giữa các implementation):

| | Multi-Token | ViMoE | chênh |
|---|---:|---:|---:|
| CIDEr-D | 94.4 | 88.7 | **+5.7** |
| BLEU-4 | 19.6 | 12.5 | **+7.0** |
| ROUGE-L | 50.0 | 47.1 | **+2.9** |

**THUA**: F1 (44–51 vs 60.7), Acc (8.6 vs 13 của Vintern-finetune).
Nút thắt = **frozen Qwen2-0.5B** — ViMoE train decoder 6 lớp from scratch + label smoothing
nên bám phrasing đáp án chặt hơn. **Không phải lỗi của bridge.**

### 5C. Kết quả routing (đóng góp mới của Paper 3)

> **Không policy học được nào thắng được một fixed policy chọn khéo.**
> Loại suy luận (reasoning-type) **không** dự đoán được nhu cầu tính toán thị giác —
> per-category không có category nào n_tiles có tác dụng (paired bootstrap CI đều chứa 0).
> Với lớp VLM nhẹ này, phân bổ compute thích ứng không đáng.

Chi tiết bảng ở mục 3 bước 4.

---

## 6. "Compute lever" (đòn bẩy tính toán) — vì sao phải đo

"Route" chỉ có nghĩa nếu cái nút `n_tiles` **thật sự thay đổi chi phí đáng kể**. Nếu vặn
từ 1 lên 6 mảnh mà máy chỉ tốn thêm 3% -> chẳng có gì để tiết kiệm, cả câu hỏi nghiên
cứu vô nghĩa.

Mình đã đo (phase P1):

| | n_tiles = 1 | n_tiles = 6 |
|---|---:|---:|
| FLOPs phần thị giác (InternViT) | 1x | **~6x** |
| Latency thực trên GPU P100 | 1x | **~4x** |

-> Cái nút **nặng thật**. Nhờ vậy, khi phát hiện "route không giúp gì", mình loại được
cách giải thích tầm thường *"vì cái nút vốn quá nhẹ"*. Kết luận đúng là: **có thứ để
route hẳn hoi (tốn gấp 4–6 lần), nhưng route cũng không ăn thua** — vì frozen Qwen2-0.5B
không tận dụng được chi tiết thị giác thêm vào.

---

## 7. Đã trả lời được gì / Còn cần confirm gì

### 7A. Đã trả lời (đủ chắc để viết)

| Câu hỏi | Trả lời | Bằng chứng |
|---|---|---|
| Reasoning-type có dự đoán nhu cầu visual compute không? | **KHÔNG** | Per-category, paired bootstrap CI đều chứa 0 — cả \|A\|=6 lẫn \|A\|=9 |
| Policy học được có thắng fixed policy không? | **KHÔNG** | \|A\|=9 held-out (train 5547 -> test 3739): `ours` / `rt_only` / `visual_only` đều = fixed `multi_token\|1` (0.90) |
| Reasoning-type có thêm gì so với chỉ liếc ảnh? | **KHÔNG** | `rt_only` ≈ `visual_only` ≈ `ours` |
| `n_tiles` có phải compute lever thật không? | **CÓ** (FLOPs ×6, latency ×4) | -> negative KHÔNG phải do "lever quá nhẹ" (mục 6) |
| Bridge nào tốt nhất? | **multi_token**, chỉ 0.78% params | Exp A, corpus metrics |
| multi_token vs ViMoE (generation) | **Thắng** CIDEr-D +5.7 / BLEU +7.0 / ROUGE +2.9; **thua** F1 | val, seed 42 |
| Split cũ có bị leak thổi phồng không? | **KHÔNG** | grouped split ≈ số cũ |
| Có "bridge fork" theo category không? (Exp B) | **KHÔNG** | multi_token top ở cả 8 category |

-> Câu chuyện chính của paper đã đứng được.

### 7B. Còn cần confirm

**Ảnh hưởng đến kết luận (ưu tiên cao):**

| Cần gì | Vì sao | Trạng thái |
|---|---|---|
| Oracle sweep cho **tiled multi_token** (train *với* tiles) | Biến cuối: nếu bản này khai thác được tiles -> phải làm lại phân tích "headroom là noise" (§3 bước 4). Nếu collapse như bản n_tiles=1 -> không đổi | đang chạy, ~23:00 |
| **Bootstrap oracle M** (resample 5 ref) | Củng cố lập luận "dư địa +40% là nhiễu CIDEr" — cho reviewer thấy headroom co lại | chưa làm, rẻ (~1h) |
| **Bridge eval trên TEST** (1 lần) | Bảng bridge hiện là VAL; bảng cuối cần test | chưa chạy |

**Rigor (reviewer sẽ hỏi):**

| Cần gì | Vì sao |
|---|---|
| **≥3–5 seed** cho multi_token (42, 123, 3407, 2026, 8668) + std / CI | Hiện 1 seed. ViMoE có 5 seed |
| **Human validation** 300–500 mẫu, 2 annotator, Cohen's κ | Signature #8 của "công thức rigor" P1&P2; cũng validate CIDEr per-sample. **Cần người** |
| **Error analysis định lượng** | Phân loại lỗi trên test (noun omission, vague attribute...) |

**Chỉ để "khả quan" hơn, không đổi kết luận:**

| Cần gì | Trạng thái |
|---|---|
| `--answer-sampling random` (train trên cả 5 ref) -> F1 có chạm Vintern-finetune (53.8)? | đang chạy, ~05:30 |

---

## 8. Ba đóng góp của Paper 3

Paper **KHÔNG** bán "bridge đánh bại SOTA". Ba thứ:

1. **Framework** để nghiên cứu adaptive visual computation: VLM chỉ-train-bridge +
   action space (tiles × bridge) có cost chuẩn hoá + oracle sweep offline.
2. **Kết quả negative nghiêm ngặt** + truy được nguyên nhân: reasoning-type không giúp
   route; không policy nào thắng fixed policy — với VLM frozen-backbone.
3. **Benchmark bridge leak-free** trên AutoViVQA + **phân tích compute-efficiency
   (FLOPs / latency)** — ViMoE khẳng định hiệu quả tính toán nhưng không báo con số cụ thể;
   Paper 3 bổ sung phần đo đó.

Positive kèm theo: multi_token (7M tham số, frozen) vượt ViMoE MoE trên CIDEr-D / BLEU / ROUGE.

---

## 9. Thầy có thể chất vấn gì

| Chất vấn | Trả lời |
|---|---|
| "Kết quả không khả quan" | Đúng nếu định nghĩa = thắng SOTA trên F1 — không đạt được khi còn frozen decoder. Giá trị: **efficiency + negative result + benchmark sạch**, đúng phạm vi Trust4NLP (nhận analysis / negative papers). |
| "Mới 1 seed" | Đang bổ sung 5 seed / dựa paired bootstrap trên 5000+ mẫu. |
| "F1 / METEOR số nào đúng" | Chốt 1 bộ metric cho toàn paper — đề xuất pycocoevalcap corpus (CIDEr-D / BLEU / ROUGE ổn định). |
| "Negative result đủ mới để đăng?" | Có — kiểm tra thẳng hướng "reasoning-aware" của ViMoE trên cùng benchmark + bổ sung phần efficiency. |
| "Sao dùng CIDEr" | Bài toán sinh câu nhiều-ref cần metric liên tục per-sample; EM / BLEU per-sample ≈ 0 nên vô dụng cho oracle argmax. CIDEr là chuẩn của Vietnamese VQA. (Nhiễu per-sample là hạn chế đã ghi nhận.) |
| Framing an toàn với ViMoE | Không nói "ViMoE sai". Nói: "ViMoE đề xuất hướng reasoning-aware; chúng tôi kiểm tra thẳng bằng oracle và thấy tín hiệu reasoning-type không đủ để route -> gợi ý lợi ích MoE đến từ capacity / ensemble hơn là chuyên biệt hoá theo loại suy luận." (Thầy Tung Le là corresponding author của ViMoE.) |
