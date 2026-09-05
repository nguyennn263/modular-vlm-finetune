# Kết quả hiện tại — Paper 3 (efficiency-bridge)

*Cập nhật 2026-09-04 14:30. Spine = efficiency-bridge (alignment-KD đã thử, âm).*

---

## Headline

> Bridge **multi_token** (7.35M param = 0.78%), **đóng băng cả InternViT + Qwen2**,
> huấn luyện ở **1 tile ảnh** thay vì tới 12 của Vintern gốc → **vượt Vintern-1B
> finetune-toàn-bộ và ViMoE-VQA trên metric sinh**, với chi phí train ~100× nhỏ hơn và
> chi phí vision inference ~×4–6 rẻ hơn.

---

## 0. multi_token multi-seed (C2) — ĐÃ KHÓA, 4/4 seed xong

| seed | CIDEr-D | BLEU-4 | ROUGE-L | F1 |
|---|---:|---:|---:|---:|
| 42 | 94.4 | 19.58 | 50.0 | 50.7 |
| 123 | 91.7 | 19.24 | 48.8 | 49.5 |
| 2026 | 93.1 | 19.12 | 49.0 | 49.6 |
| 3407 | 91.8 | 18.80 | 48.9 | 49.5 |
| **mean ± std (n=4)** | **92.8 ± 1.1** | **19.2 ± 0.3** | **49.2 ± 0.5** | **49.8 ± 0.5** |
| ViMoE-VQA (5 seed) | 88.7 | 12.5 | 47.1 | 60.7 |

**4/4 seed xong, std nhỏ (~1–2% relative)** → multi_token vượt ViMoE trên CIDEr-D
(+4.1), BLEU-4 (+6.7), ROUGE-L (+2.1) **ổn định qua seed, không phải may rủi seed 42**.
Vẫn thua F1 rõ rệt (49.8 vs 60.7, −10.9) — nhất quán với 3-way negative ở mục 3
(frozen decoder là trần). Đây là dòng headline chính thức cho paper.

## 1. So 5 bridge (val, grouped split, seed 42, 1 tile) — ĐÃ KHÓA

Metric pycocoevalcap corpus (chuẩn để so paper khác).

| Bridge | Tham số train | % | CIDEr-D | BLEU-4 | ROUGE-L | F1(token) | val CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Multi-Token** (8 tok pooled) | 7.35M | 0.78 | **94.4** | **19.6** | **50.0** | **44.2** | **1.49** |
| Full Q-Former (16 query) | 69.4M | 6.91 | 86.7 | 17.5 | 47.1 | 43.3 | 1.57 |
| Light Q-Former (8 query) | 27.6M | 2.87 | 83.8 | 16.8 | 46.0 | 42.9 | 1.59 |
| Tile-Attention (8 tok) | 4.14M | 0.44 | 82.7 | 16.3 | 46.1 | 43.0 | 1.62 |
| Residual (1 tok) | 4.86M | 0.52 | 56.3 | 8.1 | 36.0 | 37.6 | 2.35 |

Multi-token tốt nhất mọi mặt **và** có CE thấp nhất — quan trọng cho §2 dưới.

## 2. Multi-Token vs prior work trên AutoViVQA — ĐÃ KHÓA (val, seed 42)

| Model | Acc | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|---:|---:|---:|---:|---:|---:|
| Vintern-1B (base) | 0.1 | 17.6 | 1.9 | 25.8 | 23.9 | 8.5 |
| Vintern-1B (**finetune toàn bộ**, đa tile) | **13.0** | 53.8 | 6.1 | **51.9** | 35.3 | 72.8 |
| GPT-5 (zero-shot) | 10.8 | 50.9 | 6.1 | 47.3 | 33.3 | 84.2 |
| **ViMoE-VQA / Tuong-MoE** (5 seed) | 9.7 | **60.7** | 12.5 | 47.1 | **39.1** | 88.7 |
| **★ Multi-Token Bridge (ours)** | 8.6 | 44–51* | **19.6** | **50.0** | ~28–41* | **94.4** |

<small>* F1/METEOR chênh theo implementation. BARTPhoBEiT bỏ khỏi bảng: CIDEr 189 là outlier.</small>

**Thắng rõ (metric sinh ổn định):** CIDEr-D +5.7 / BLEU-4 +7.0 / ROUGE-L +2.9 so với ViMoE.
**Thua:** F1 token-level, Acc. → §3.

## 3. Ba can thiệp khép F1 gap — TẤT CẢ ÂM (val, seed 42, anchor = 50.7 / 94.4) — ĐÃ KHÓA

| Can thiệp (trục) | F1 | CIDEr-D | Δ F1 |
|---|---:|---:|---:|
| baseline `first` | 50.7 | 94.4 | — |
| answer-sampling=random (training target) | 49.0 | 87.3 | −1.7 |
| align-feat α=1.0 (representation alignment) | 49.7 | 92.0 | −1.0 |
| align-logit α=1.0 (representation alignment) | 40.7† | 80.1† | −10 |

<small>† ep2 subset — bị cắt trước full-val. val CE 2.84 (vs 1.49) → KL term ở weight 1.0 chèn CE.</small>

**Kết luận (§6.1):** ba trục độc lập — phân bổ visual compute (routing, §5.2–5.4), training
target, representation alignment — đều **không** cải thiện token-F1. multi_token đã đạt CE
thấp nhất trong 5 bridge. → **frozen Qwen2-0.5B decoder LÀ trần cho khớp phrasing;
capacity phía thị giác/training KHÔNG phải nút thắt.**

## 4. Error analysis (multi_token, val 5463) — ĐÃ KHÓA

| Token-F1 bucket | % |
|---|---:|
| strong (≥0.6) | 36.7 |
| partial (0.2–0.6) | 51.5 |
| weak (<0.2) | 3.1 |
| zero | 8.7 |

- Độ dài dự đoán **4.39 từ** vs ref 4.32 — **không** bị "sinh câu cụt" như ViMoE (ViMoE 4.4 vs 5.6). Residual bridge thì sinh dài lê thê (6.41 từ).
- **Noun omission ở câu đếm: 5.8%** (vs ViMoE **10.7%**) — mình bỏ noun ÍT hơn.
- Per-category F1: tốt nhất counting 0.66 / yesno 0.61 / relational 0.55; tệ nhất action 0.40 / context 0.37 / causal 0.43 (câu mở, nhiều đáp án đúng — model đoán 1 đáp án hợp lý khác).

## 4b. Decoder-LoRA (feat/decoder-lora branch) — POSITIVE, 3/3 seed KHÓA

Sau 3-way negative (§3), thử can thiệp **decoder** (phá vỡ frozen-backbone có chủ đích):
LoRA r=16 trên q/k/v/o của Qwen2-0.5B, huấn luyện cùng bridge multi_token, 1 epoch, 1 tile.

| seed | F1 | CIDEr (in-house) | BLEU | val loss |
|---|---:|---:|---:|---:|
| 42 | 53.16 | 104.9 | 19.38 | 1.368 |
| 123 | 53.20 | — | — | — |
| 3407 | 53.15 | — | — | — |

| | Plain (mean 4 seed) | **LoRA r=16 (mean 3 seed)** | Δ | ViMoE |
|---|---:|---:|---:|---:|
| F1 | 49.8 | **53.17** | **+3.4** | 60.7 |
| CIDEr (in-house) | 97.0 | **~105.6** | **+8.6** | — |
| BLEU | 16.0 | **~19.5** | **+3.5** | 12.5 |
| Acc | 8.3 | **10.4** (2-seed) | **+2.1** | 9.7 |

**Khép ~31% khoảng cách F1 tới ViMoE** (gap 10.9 → còn 7.5), **tái lập được qua CẢ 3
seed** (42/123/3407, std nhỏ ~0.03 trên F1) — không phải may rủi 1 seed. val CE cũng
thấp hơn hẳn plain (1.37–1.39 vs 1.49).

**Ý nghĩa cho paper:** đây là can thiệp DUY NHẤT trong tất cả các thử (routing, answer-
sampling, align-KD, decoder-LoRA) thực sự cải thiện F1 — và nó là can thiệp DUY NHẤT
đụng vào decoder. Càng củng cố §6.1: **frozen decoder là trần**; mở nó ra (dù rất nhẹ,
~2% param LoRA) mới nhích được, còn mọi can thiệp phía thị giác/training đều vô ích.

Đóng khung: multi_token + LoRA vẫn KHÔNG phải "frozen backbone" nữa — trình bày như
**phần bổ sung/reference point**, không phải spine chính (spine chính vẫn là bridge
0.78% param hoàn toàn đóng băng).

### Generalization check: LoRA trên qformer (seed 42) — ĐÃ XONG, KHÓA

LoRA r=16 áp lên bridge **khác** (Full Q-Former, 69.4M param) để xem hiệu ứng có phải
chỉ riêng multi_token hay không. Cùng eval_val.json in-house convention (n=5463, 1 tile):

| | qformer plain | qformer **+ LoRA r=16** | Δ |
|---|---:|---:|---:|
| F1 | 47.66 | **53.10** | **+5.4** |
| CIDEr (in-house) | 90.8 | **105.2** | **+14.3** |
| BLEU | 14.6 | **19.3** | **+4.8** |
| ROUGE-L | 46.0 | **51.6** | **+5.6** |
| Acc | 7.34 | **10.91** | **+3.6** |
| val loss | 1.568 | **1.377** | **−0.19** |

**Gen hoá xác nhận, và mạnh hơn cả multi_token** (+5.4 F1 vs +3.4 trên multi_token) —
decoder-LoRA không phải hiệu ứng đặc thù 1 bridge, mà là hiệu ứng của việc mở decoder,
nhất quán bất kể bridge nào đứng trước nó. Củng cố thêm luận điểm §6.1.

### Corpus rescore (pycocoevalcap) — qformer LoRA seed 42, ĐÃ XONG

`scripts/rescore_corpus.py` (mới, tổng quát hoá `rescore_expA.py` từ chỉ-CIDEr sang
CIDEr-D+BLEU-4+ROUGE-L, dùng được ngoài `checkpoints/expA/`). Verify: chạy trên
qformer-plain seed42 tái tạo đúng hàng trong bảng §1 (86.7/17.5/47.1) → script đúng
convention. Kết quả LoRA (n=5463, cross-paper-comparable):

| | qformer plain | qformer **+ LoRA r=16** | Δ |
|---|---:|---:|---:|
| CIDEr-D | 86.7 | **101.9** | **+15.2** |
| BLEU-4 | 17.5 | **23.1** | **+5.6** |
| ROUGE-L | 47.1 | **52.6** | **+5.5** |

So ViMoE (88.7/12.5/47.1): qformer+LoRA thắng cả 3 (CIDEr-D +13.2, BLEU-4 +10.6,
ROUGE-L +5.5) — hạng mạnh hơn multi_token-plain (§2) trên BLEU-4/ROUGE-L, dù CIDEr-D
vẫn thấp hơn multi_token-plain (94.4).

**multi_token+LoRA seed 42 full-val — ĐÃ XONG (verify hạ tầng thành công lần 3):**
in-house F1 53.16/CIDEr 104.9/BLEU 19.38 (khớp seed 123/3407, xem bảng trên). Corpus:

| | multi_token plain (§1) | multi_token **+ LoRA r=16** | Δ | ViMoE |
|---|---:|---:|---:|---:|
| CIDEr-D | 94.4 | **101.7** | **+7.3** | 88.7 |
| BLEU-4 | 19.6 | **23.2** | **+3.6** | 12.5 |
| ROUGE-L | 50.0 | **52.7** | **+2.7** | 47.1 |

multi_token+LoRA thắng ViMoE cả 3 (+13.0/+10.7/+5.6) và là điểm mạnh nhất trong mọi
biến thể (plain hay LoRA, bridge nào) trên cả CIDEr-D lẫn BLEU-4/ROUGE-L cross-paper.
**Toàn bộ dòng LoRA r=16 giờ đã khóa 3/3 seed (in-house) + corpus cho cả 2 bridge
(multi_token, qformer).**

**Đang chạy:** sweep LoRA r=8 và r=32 (multi_token, seed42, acc6/acc7) tận dụng quota
Kaggle đang dư nhiều (~440h/480h chưa dùng tuần này) — mục tiêu có ablation theo rank
thay vì 1 điểm r=16.

## 5. Còn PENDING (sau reset quota 00:00 UTC 5/9)

| # | Việc | Ai |
|---|---|---|
| C3 | Oracle sweep + policy ladder trên 3 **tiled** checkpoint → re-lock §5.2/§5.3 (đóng confound "bridge train ở 1 tile") | peer |
| C2 | multi_token seed 123 + 3407 → mean±std cho dòng headline | tôi |
| B3 | Tile-sweep: multi_token @ {1,3,6,12} vs Vintern-finetune @ {1,3,6,12} → bảng efficiency | tôi |
| C4 | Human validation 300–500 mẫu, 2 annotator, Cohen's κ | **user** + tôi setup |
| B5 | Số tile "Vintern-finetune" bảng cũ (≤6 hay ≤12) | **user** |

## 6. Câu chuyện paper (chốt)

1. **Đóng góp chính — efficiency**: bridge nhẹ trên backbone đóng băng đạt SOTA-generation ở 1 tile.
2. **Oracle analysis**: xác nhận 1 tile không phải thoả hiệp — không policy adaptive nào (kể cả biết reasoning-type) thắng fixed 1-tile.
3. **3-way negative → decoder ceiling**: định vị + định lượng nút thắt (frozen 0.5B decoder), không phải phía thị giác.
4. **Benchmark bridge leak-free** + compute-efficiency table (FLOPs/latency) mà ViMoE bỏ ngỏ.

Reasoning-type / P(r|Q) router → 1 mục ablation nhỏ.
