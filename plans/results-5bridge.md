# Kết quả — Cải thiện Vintern-1B cách rẻ (leak-free grouped split)

> **Đây là doc bảng kết quả CHÍNH cho paper** (Main Results + Bridge comparison).
> Split: grouped 70/15/15 theo `image_id` (không rò rỉ ảnh). Convention metric
> giống bảng AutoViVQA (Acc/Prec/Rec/F1/BLEU/ROUGE-L/METEOR/CIDEr, in-house ×100).
> Chi tiết ablation (6-RQ, tile-sweep, self-check, per-seed) ở `results-grouped-split.md`.
> Cập nhật 2026-09-06. Số seed-42-only đang được TIER-1 nâng lên 3-seed.

---

## Bối cảnh

Vintern-1B (InternViT-300M + Qwen2-0.5B) là VLM tiếng Việt mạnh, nhưng:
- **Zero-shot trên AutoViVQA: gần như hỏng** (F1 17.55).
- **Finetune toàn bộ: F1 53.76** — nhưng huấn luyện lại toàn bộ ViT + LoRA LLM,
  dynamic tiling tới 12 tile → chi phí train ~100× lớn, chi phí vision inference ~×4–6.

**Câu hỏi:** thay vì xây model mới (như ViMoE-VQA / Tuong-MOE), có thể **cải thiện
Vintern-1B một cách rẻ** — giữ cố định cả hai backbone, chỉ huấn luyện một *bridge
module* nhỏ giữa chúng — mà đạt tương đương finetune-toàn-bộ không? Và nếu chưa đạt
thì **nút thắt nằm ở đâu?**

---

## Các kiến trúc bridge đề xuất

Giữ cố định mô hình thị giác và mô hình ngôn ngữ của Vintern-1B, chỉ huấn luyện
thành phần bridge để ánh xạ đặc trưng ảnh sang không gian embedding ngôn ngữ. 5 kiến
trúc, xếp theo thang capacity, nhằm cải thiện lớp linear projection gốc (MLP 2 lớp).

| Exp | Tên | Mô tả |
|---|---|---|
| 1 | **Residual Bridge** | Bridge tuyến tính + nhánh residual (LayerNorm + 2 FC + GELU), học phần hiệu chỉnh trên biểu diễn gốc. 1 token ảnh. |
| 2 | **Multi-Token Bridge** | Sinh **nhiều token** đầu ra: một token anchor + các token bổ sung ngữ nghĩa (8 token pooled). |
| 3 | **Tile Attention Bridge** | Chia đặc trưng ảnh thành patch, self-attention giữa patch để tận dụng thông tin không gian (8 token). |
| 4 | **Lightweight Q-Former** | 8 query token + 2 lớp transformer nhẹ học tương tác thị giác ↔ ngôn ngữ. |
| 5 | **Full Q-Former** | 16 query token + 4 lớp transformer + fusion ảnh–văn bản, tăng khả năng căn chỉnh đa phương thức. |

**Can thiệp thứ 2 (RQ6):** sau khi cạn kiệt phía bridge, LoRA r=16 trên `q/k/v/o`
của Qwen2-0.5B (~2.16M param) như một can thiệp có chủ đích vào decoder.

---

## Bảng 1 — Kết quả chính (val, in-house metric ×100)

| Mô hình | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vintern-1B (base, zero-shot) | 0.12 | 17.52 | 19.87 | 17.55 | 1.91 | 25.84 | 23.93 | 8.54 |
| ViT5_ViT | 7.97 | 46.84 | 50.33 | 48.52 | 4.13 | 46.89 | 31.02 | 72.68 |
| BARTPhoBEiT | 8.81 | 45.30 | 46.48 | 45.88 | 4.33 | 44.83 | 24.57 | 188.96¹ |
| **Vintern-1B (finetune toàn bộ, ≤12 tile)** | **13.01** | 52.47 | 55.12 | **53.76** | 6.11 | **51.93** | 35.25 | 72.84 |
| Llama 3.2 (zero-shot) | 0.36 | 23.96 | 73.71 | 36.16 | 3.62 | 36.11 | 30.01 | 62.84 |
| Gemini 2.0 Flash | 0.55 | 27.20 | 74.10 | 39.79 | 4.41 | 39.60 | 31.72 | 74.42 |
| Gemini 2.5 Flash | 0.22 | 24.43 | 76.66 | 24.75 | 0.39 | 37.27 | 31.22 | 71.90 |
| GPT-5 (zero-shot) | 10.84 | 47.20 | 55.20 | 50.89 | 6.07 | 47.30 | 33.34 | 84.20 |
| **ViMoE-VQA (Tuong-MOE, 5 seed)** | 9.65 | **62.89** | 58.65 | **60.69** | 12.54 | 47.07 | **39.10** | 88.67 |
| — *Ours (frozen backbone, 1 tile, 2 epoch, mean 3 seed²):* | | | | | | | | |
| Residual Bridge | 4.83 | 46.09 | 48.79 | 45.64 | 12.70 | 44.34 | 36.49 | 86.25 |
| Tile-Attention Bridge | 3.97 | 44.59 | 47.86 | 45.17 | 12.35 | 43.35 | 36.14 | 84.21 |
| Light Q-Former (mini) | 6.04 | 47.34 | 49.57 | 46.25 | 13.48 | 44.53 | 36.82 | 86.80 |
| Full Q-Former | 7.32 | 48.26 | 49.56 | 47.36 | 13.28 | 45.54 | 37.91 | 88.31 |
| **Multi-Token Bridge** (0.78% param, mean 4 seed) | 8.20 | 50.36 | 51.43 | 49.55 | **15.47** | 47.84 | **40.22** | **96.49** |
| **★ Multi-Token + LoRA r=16** (~1.0% param, mean 3 seed) | **10.42** | 53.85 | 55.00 | 53.17 | **19.44** | 51.48 | **43.91** | **105.59** |
| **★ Multi-Token + LoRA r=16, 3 epoch** (mean 3 seed) | **11.78** | 55.54 | 56.25 | 54.67 | **20.98** | **52.92** | **45.24** | **109.60** |

<small>¹ BARTPhoBEiT CIDEr là outlier (sinh câu dài lê thê), không so.
Baseline (dòng 1–9): lấy từ bảng AutoViVQA, độc lập với split.
Bridge (dòng 10–14): grouped split leak-free, **2 epoch, trung bình 3 seed**
(multi_token = 4 seed). ² Acc/Prec/Rec: seed 42; các cột còn lại: mean 3 seed.
Bản trước dùng seed-42-only + residual có 1 lần chạy hỏng (F1 36.45 → thật 45.64).</small>

**Đọc:**
- **Chỉ train bridge (Multi-Token, 0.78% param, 1 tile)** đã **vượt Vintern finetune-toàn-bộ** trên BLEU (+9.4), METEOR (+5.0), CIDEr (+23.7) — ở ~1% chi phí train.
- **+ LoRA decoder (~1.0% param tổng)** đẩy F1 49.6 → 53.2 → 54.7 (3 epoch), thắng cả Vintern-FT lẫn GPT-5 trên F1, và **vượt ViMoE trên mọi metric sinh** (BLEU, CIDEr; ROUGE, METEOR ở 3-epoch).
- **Vẫn thua ViMoE trên F1 token-level** (54.7 vs 60.7, gap 6.0). → xem chẩn đoán §RQ.
- **5 bridge plain chụm F1 45.2–49.6** (chênh ~4.4 điểm) — residual KHÔNG còn là ngoại lai.

### Corpus pycocoevalcap (chuẩn so cross-paper)

| Mô hình | CIDEr-D | BLEU-4 | ROUGE-L |
|---|---:|---:|---:|
| ViMoE-VQA | 88.7 | 12.5 | 47.1 |
| Multi-Token Bridge (mean 4 seed, 2ep) | **92.3 ± 0.6** | **18.9 ± 0.3** | **48.9 ± 0.1** |
| Multi-Token + LoRA r=16 (seed 42) | **101.7** | **23.2** | **52.7** |
| Multi-Token + LoRA r=16, 3 epoch (mean 3 seed) | **106.8 ± 1.1** | **25.0 ± 0.4** | **54.2 ± 0.2** |

Bootstrap: multi_token CIDEr-D 95% CI **[91.3, 97.1]** — hoàn toàn trên ViMoE 88.7.
Mọi cải thiện plain→LoRA significant (P(Δ>0) = 1.000).

---

## Bảng 2 — So 5 bridge (RQ1–2) + LoRA (RQ6)

Mọi số plain = **2 epoch, mean 3 seed** (multi_token = 4 seed). LoRA = r=16, 1 ep
(multi_token/qformer/mini_qformer/residual = mean 3 seed; tile_attention = seed 42).
F1/CIDEr in-house ×100.

| Bridge | Param | % | F1 plain | F1 +LoRA | ΔF1 | CIDEr plain | CIDEr +LoRA | val CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Residual (1 tok) | 4.86M | 0.52 | 45.64 | 52.64 | **+7.0** | 86.25 | 104.05 | 1.67 |
| Tile-Attention (8 tok) | 4.14M | 0.44 | 45.17 | 52.99 | **+7.8** | 84.21 | 105.04 | 1.67 |
| Multi-Token (8 tok pooled) | 7.35M | 0.78 | 49.55 | 53.17 | **+3.6** | 96.49 | 105.59 | 1.49 |
| Light Q-Former (8 query) | 27.6M | 2.87 | 46.25 | 53.21 | **+7.0** | 86.80 | 106.24 | 1.60 |
| Full Q-Former (16 query) | 69.4M | 6.91 | 47.36 | 53.21 | **+5.9** | 88.31 | 105.70 | 1.57 |

**RQ1** — chỉ train bridge, đóng băng hết: Multi-Token (0.78% param) tốt nhất, vượt
Vintern-FT trên generation. Nhưng thua F1.
**RQ2** — F1 thua tại bridge chưa đủ to? **Không**: Full Q-Former (69M param, ×10) *tệ hơn*
Multi-Token; Multi-Token có CE thấp nhất → không phải capacity bridge.
**RQ6** — mở decoder (LoRA): trục **duy nhất** nhích F1. **Bridge-equalizing**: 5 bridge
plain trải F1 45.2–49.6 / CIDEr 84–96 → sau LoRA đều F1 52.6–53.2 / CIDEr 104–106
(chênh lệch gần như biến mất; mức nâng lớn hơn khi bridge yếu hơn). → capacity
decoder, không phải độ tinh vi bridge, quyết định chất lượng.
**RQ6 sâu (§4d results-grouped-split)** — LoRA chỉ giúp khi gắn vào **attention**
của decoder (q/k/v/o). Gắn vào feed-forward (gate/up/down) làm training **phân kỳ**
(F1 ~20, val loss ~3.7). → dư địa của decoder nằm cụ thể ở attention layers.

---

## Bảng 3 — Số lượng tham số

| Thành phần | Total params | Trainable | % | Frozen |
|---|---:|---:|---:|---:|
| Residual Bridge | 939.48M | 4.86M | 0.52% | 934.63M |
| **Multi-Token Bridge** | 941.98M | **7.35M** | **0.78%** | 934.63M |
| Tile Attention Bridge | 938.77M | 4.14M | 0.44% | 934.63M |
| Lightweight Q-Former | 962.20M | 27.57M | 2.87% | 934.63M |
| Full Q-Former | 1004.02M | 69.39M | 6.91% | 934.63M |
| — LoRA r=16 (Qwen2 q/k/v/o) | — | 2.16M | +0.23% | — |
| **Multi-Token + LoRA r=16** | 944.14M | **9.51M** | **1.01%** | 934.63M |

> Multi-Token Bridge đạt kết quả tốt nhất trong 5 kiến trúc chỉ với 0.78% tham số
> — hiệu quả tham số cao nhất (so với Full Q-Former tốn 6.91% nhưng kém hơn).
> **Recipe cuối = bridge + LoRA = 1.01% tham số**, vượt finetune-toàn-bộ trên generation.

---

## Nhận xét (chốt cho §5–§6)

1. **Cải thiện Vintern-1B bằng cách chỉ thay tầng bridge** tạo cải thiện đáng kể trên
   metric sinh (BLEU/METEOR/CIDEr vượt cả finetune-toàn-bộ) mà không train lại backbone.
2. **Kiến trúc nhiều token** (Multi-Token, Tile-Attention, Q-Former) hơn hẳn tinh chỉnh
   MLP một token (Residual). Nhưng **thêm capacity bridge quá mức phản tác dụng**
   (Full Q-Former 69M < Multi-Token 7M).
3. **F1 token-level là nút thắt** — và nó **không** phải vấn đề phía thị giác: 4 trục
   độc lập (kiến trúc bridge, số tile, routing động, alignment) đều không nhích được F1
   (chi tiết `results-grouped-split.md` §RQ3–5).
4. **Chỉ mở decoder (LoRA ~0.23% param) mới nhích F1** (+3.4 → +5.0), và nó **san bằng
   chênh lệch giữa các bridge** → *frozen decoder Qwen2-0.5B là trần cho khớp phrasing*.
5. **Recipe rẻ nhất đạt SOTA-generation:** `Vintern-1B (frozen) + Multi-Token bridge
   (0.78%) + decoder LoRA r=16 (0.23%)`, 1 tile, ~1 GPU-giờ. Đối lập trực tiếp với hướng
   "xây model MoE mới" của ViMoE-VQA.

---

## Lịch sử

- Bản trước (`git log`): split cũ (leak — ảnh chung giữa train/test). Multi-Token F1
  50.23, CIDEr 99.88. Grouped split leak-free chỉ thấp hơn ~0.5 F1 / ~3 CIDEr →
  **kết quả bridge cũ KHÔNG bị thổi phồng nhiều do leak** (một điểm cho paper).
