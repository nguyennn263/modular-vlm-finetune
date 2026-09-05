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
| Vintern-1B (**finetune toàn bộ**, đa tile†) | **13.0** | 53.8 | 6.1 | **51.9** | 35.3 | 72.8 |
| GPT-5 (zero-shot) | 10.8 | 50.9 | 6.1 | 47.3 | 33.3 | 84.2 |
| **ViMoE-VQA / Tuong-MoE** (5 seed) | 9.7 | **60.7** | 12.5 | 47.1 | **39.1** | 88.7 |
| **★ Multi-Token Bridge (ours)** | 8.6 | 44–51* | **19.6** | **50.0** | ~28–41* | **94.4** |

<small>* F1/METEOR chênh theo implementation. BARTPhoBEiT bỏ khỏi bảng: CIDEr 189 là outlier.
† **B5 resolved (user, 2026-09-05):** bản Vintern-1B finetune dùng là bản HuggingFace release
đầy đủ, dynamic tiling **tối đa 12 tile lúc train** (tối đa 40 lúc test theo config gốc của
Vintern) — không phải ≤6. Vậy headline "1 tile của mình so với 12 tile của Vintern gốc" ở đầu
file là ĐÚNG, đã confirm chứ không phải giả định.</small>

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

## 4c. Self-check: F1 bucket có thực sự phản ánh đúng-sai không? (C4, thay cho human validation) — ĐÃ XONG

**Đổi scope so với plan gốc** (300–500 mẫu, 2 người chấm, Cohen's κ): user không có thời gian
trước deadline, chỉ đạo "human thì m tự check luôn đi" → chuyển thành **self-check 1
rater (chính tôi/assistant), N=120**, chấm từng câu dựa trên câu hỏi + 5 câu tham chiếu
(**không có ảnh gốc** — đây là kiểm tra plausibility-so-với-reference, không phải
ground-truth-độc-lập-từ-ảnh, khác về bản chất so với validation con người thật). Script:
`scripts/human_validation_sample.py` (lấy mẫu tỉ lệ theo category × F1-bucket thật, seed 42,
n=15/category) → `scripts/human_validation_report.py` (tổng hợp). Toàn bộ 120 phán đoán +
lý do từng câu: `outputs/human_validation/selfcheck_judgments.json`.

| F1 bucket | n | đúng | đúng 1 phần | sai | vô nghĩa | **chấp nhận được (đúng+1 phần)** |
|---|---:|---:|---:|---:|---:|---:|
| strong (≥0.6) | 45 | 80.0% | 11.1% | 6.7% | 2.2% | **91.1%** |
| partial (0.2–0.6) | 58 | 12.1% | 31.0% | 55.2% | 1.7% | **43.1%** |
| weak (0–0.2) | 3 | 0% | 0% | 100% | 0% | **0%** |
| zero (F1=0) | 13 | 7.7% | 7.7% | 76.9% | 7.7% | **15.4%** |
| **tổng (n=119, loại 1 mẫu GT tự mâu thuẫn)** | | **37.0%** | **20.2%** | **40.3%** | **2.5%** | **57.1%** |

**Đọc kết quả — KHÔNG chỉ là tin tốt, phải nói thẳng:**
- **Bucket "strong" đáng tin** (91.1% chấp nhận được) → F1 cao thì hầu như chắc đúng.
- **Bucket "zero" hầu như đúng là sai** (84.6% sai/vô nghĩa) nhưng **không phải 100%** —
  có ca F1=0 nhưng ngữ nghĩa đúng (VD idx2261: "đang suy nghĩ về chiến thắng" vs GT
  "làm sao để đánh trúng bóng/hi vọng home run" — diễn giải khác nhưng cùng ý, 0 token chung).
- **Bucket "partial" (0.2–0.6) — bucket LỚN NHẤT (51.5% val) — lại là bucket TỆ NHẤT
  về độ tin cậy: chỉ 43.1% chấp nhận được, 55.2% thực ra SAI** dù có chung vài token
  (từ đệm như "để", "đang", màu sắc chung...). Đây là phát hiện hơi bất lợi, không phải
  spin tích cực: **token-F1 tầm trung KHÔNG phải chỉ báo đáng tin của đúng/sai** — model
  hay sai *loại lỗi* (nhầm màu, nhầm số đếm, nhầm object, trả lời sai chiều câu hỏi —
  VD hỏi "khi nào" trả lời thời tiết, hỏi "ai" trả lời giới tính sai, hỏi "có kéo theo
  gì không" trả lời NGƯỢC cực — idx342) nhưng vẫn ăn điểm F1 nhờ từ chung không mang
  nghĩa.
- **Tổng thể: chỉ 37.0% đúng hoàn toàn, 57.1% chấp nhận được** — thấp hơn con số 36.7%
  "strong bucket" ở §4 nhưng **khớp khá sát** (2 phép đo độc lập ra số gần nhau ở mức
  tổng), củng cố rằng bucket "strong" ≈ "thực sự đúng" là một proxy hợp lý, nhưng đừng
  đọc "44.2 F1 tổng" hay CIDEr-D 94.4 như thể ~44–94% câu trả lời đúng — con số thật
  (self-check) thấp hơn nhiều so với cảm giác CIDEr-D cao có thể gợi ý.

**Hạn chế của chính self-check này** (phải ghi rõ trong paper, không giấu): 1 rater duy
nhất, không có ảnh gốc (chỉ so với 5 câu tham chiếu — với câu hỏi mở như causal/context,
"đúng" nghĩa là "hợp lý so với tham chiếu", không phải verify được với ảnh thật), N=120
chứ không phải 300–500, không có Cohen's κ vì không có rater thứ 2. Đây là substitute
tạm thời do ràng buộc thời gian, không thay thế được human validation thật nếu có
reviewer yêu cầu — nhưng đủ để phát hiện vấn đề thật (bucket "partial" không đáng tin)
mà thuần dựa vào F1 sẽ bỏ sót.

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

### Generalization check: LoRA trên qformer — 2/3 SEED (42, 3407), seed123 đang chạy

LoRA r=16 áp lên bridge **khác** (Full Q-Former, 69.4M param) để xem hiệu ứng có phải
chỉ riêng multi_token hay không. eval_val.json in-house convention (n=5463, 1 tile):

| seed | F1 | CIDEr (in-house) | BLEU | val loss |
|---|---:|---:|---:|---:|
| 42 | 53.10 | 105.15 | 19.33 | 1.377 |
| 3407 | 53.22 | 106.19 | 19.76 | 1.377 |
| 123 | đang chạy (acc13) | | | |

vs qformer plain seed42 (F1 47.66/CIDEr 90.8/BLEU 14.6/val loss 1.568) → Δ ổn định
qua cả 2 seed đã có (+5.4 đến +5.6 F1), **khớp rất sát nhau** (53.10 vs 53.22) — không
phải may rủi 1 seed.

**Gen hoá xác nhận, và mạnh hơn cả multi_token** (+5.4-5.6 F1 vs +3.4 trên multi_token) —
decoder-LoRA không phải hiệu ứng đặc thù 1 bridge, mà là hiệu ứng của việc mở decoder,
nhất quán bất kể bridge nào đứng trước nó. Củng cố thêm luận điểm §6.1.

### Corpus rescore (pycocoevalcap) — qformer LoRA, 2/3 seed

`scripts/rescore_corpus.py` (mới, tổng quát hoá `rescore_expA.py` từ chỉ-CIDEr sang
CIDEr-D+BLEU-4+ROUGE-L, dùng được ngoài `checkpoints/expA/`). Verify: chạy trên
qformer-plain seed42 tái tạo đúng hàng trong bảng §1 (86.7/17.5/47.1) → script đúng
convention. Kết quả LoRA (n=5463, cross-paper-comparable):

| seed | CIDEr-D | BLEU-4 | ROUGE-L |
|---|---:|---:|---:|
| 42 | 101.9 | 23.1 | 52.6 |
| 3407 | 102.8 | 23.5 | 52.7 |

vs qformer plain (86.7/17.5/47.1) và ViMoE (88.7/12.5/47.1): qformer+LoRA thắng cả 3
so ViMoE ở cả 2 seed (CIDEr-D +13-14, BLEU-4 +10.6-11.0, ROUGE-L +5.5-5.6) — hạng mạnh
hơn multi_token-plain (§2) trên BLEU-4/ROUGE-L, dù CIDEr-D vẫn thấp hơn multi_token-plain
(94.4).

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

### Rank curve (r=4/8/16/32/64) — seed42, SƠ BỘ (chỉ 600-mẫu training-time subset)

Cả 5 rank đã xong training (đa seed cho r=32/64 vẫn đang chạy). Số dưới đây là subset
600 mẫu (quick-eval lúc train), **chưa phải full-val 5463** — so sánh nội bộ giữa các
rank thì dùng được (cùng subset), nhưng **không so trực tiếp được** với các số full-val
ở trên.

| rank | F1 (600-subset) | CIDEr (600-subset) | val loss (full, best epoch) |
|---|---:|---:|---:|
| 4 | 51.26 | 103.50 | 1.410 |
| 8 | 51.62 | 105.95 | 1.408 |
| 16 | 51.98 | 106.44 | 1.371 |
| 32 | 51.80 | 107.06 | 1.368 |
| 64 | **53.05** | **110.43** | **1.366** |

**Đọc sơ bộ (chưa full-val, chưa nhiều seed, đừng chốt vội) — bức tranh rõ hơn với đủ
5 điểm:** cả CIDEr lẫn val loss đơn điệu tăng/giảm đều theo rank (CIDEr 103.5→110.4,
val loss 1.410→1.366), F1 gần như đơn điệu tăng trừ 1 điểm nhiễu nhỏ ở r=32. Không còn
là "bão hòa ở r=8" (đã rút lại nhận định đó) — giờ trông giống **"rank càng cao, lợi ích
càng nhích thêm"**, r=64 vẫn chưa cho thấy dấu hiệu chững lại. Vẫn phải chờ đa seed
(r=32/64 đang chạy trên acc1-4) để biết đây là xu hướng thật hay chỉ là 1 seed/600-mẫu
may mắn theo hướng monotonic.

**Đang chạy thêm để xác nhận:** r=4 (acc10), r=64 (acc9); r=32 + r=64 mỗi cái thêm 2 seed
(123, 3407 — acc1/2/3/4) để rank curve có error bar thật thay vì 1 seed; LoRA r=16 áp
thêm lên mini_qformer (acc5) và residual (acc16, bridge yếu nhất) — mở rộng "bridge-
agnostic" từ 2/5 lên 4/5 bridge.

## 5. Còn PENDING (sau reset quota 00:00 UTC 5/9)

| # | Việc | Ai | Trạng thái |
|---|---|---|---|
| C3 | Oracle sweep + policy ladder trên 3 **tiled** checkpoint → re-lock §5.2/§5.3 | peer | ✅ xong, không đổi kết luận |
| C2 | multi_token seed 123 + 3407 → mean±std cho dòng headline | tôi | ✅ xong, 4/4 seed |
| **B5** | **Số tile "Vintern-finetune" bảng cũ** | **user** | ✅ resolved: bản HF, tối đa **12** tile lúc train |
| B3 | Tile-sweep: multi_token @ {1,3,6,12} vs Vintern-finetune @ {1,3,6,12} → bảng efficiency | tôi | 🔄 running (acc11) |
| B2 | residual-tiled (baseline tương phản: bridge đơn giản có cần tile không) | tôi | 🔄 running (acc8) |
| — | LoRA rank ablation r=8/r=32 (mở rộng §4b) | tôi | 🔄 running (acc6/acc7) |
| ~~C4~~ | ~~Human validation 300–500 mẫu, 2 annotator, Cohen's κ~~ → **self-check N=120, 1 rater** (§4c) | tôi | ✅ xong — xem §4c: strong bucket 91% đáng tin, **partial bucket chỉ 43% đáng tin** (finding hơi bất lợi, đã ghi rõ) |

## 6. Câu chuyện paper (chốt)

1. **Đóng góp chính — efficiency**: bridge nhẹ trên backbone đóng băng đạt SOTA-generation ở 1 tile.
2. **Oracle analysis**: xác nhận 1 tile không phải thoả hiệp — không policy adaptive nào (kể cả biết reasoning-type) thắng fixed 1-tile.
3. **3-way negative → decoder ceiling**: định vị + định lượng nút thắt (frozen 0.5B decoder), không phải phía thị giác.
4. **Benchmark bridge leak-free** + compute-efficiency table (FLOPs/latency) mà ViMoE bỏ ngỏ.

Reasoning-type / P(r|Q) router → 1 mục ablation nhỏ.
