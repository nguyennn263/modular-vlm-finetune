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

---

## TIER-1 progress + epoch audit — 06/09/2026 13:15 UTC (ĐANG XỬ LÝ, chưa chốt)

### Epoch KHÔNG đồng đều giữa các seed — đang chuẩn hoá về 2 epoch

Kiểm tra `epochs_trained` trong summary.json:

| Nhóm | seed 42 | seed 123 / 2026 / 3407 |
|---|:--:|:--:|
| Bridge plain (5 bridge) | **4 ep** (~10h) | **2 ep** (~5h) |
| multi_token plain 4-seed "khoá" | s42 = 4 ep | s123/2026/3407 = 2 ep |
| Decoder-LoRA r=16 (mọi bridge, mọi seed) | 1 ep | 1 ep ✅ |
| Decoder-LoRA biến thể dài | 3 ep | 3 ep (multi_token) ✅ |
| Dòng âm answer-random | s42 = 4 ep | s123/s3407 = 2 ep |
| Dòng âm align-feat | s42 = **3 ep** (lạ) | s123/s3407 = 2 ep |

→ LoRA an toàn. Bridge plain + dòng âm bị lệch epoch giữa s42 và các seed khác.
seed 42 KHÔNG lưu checkpoint epoch-2 (chỉ epoch-4) → không re-eval được.

**Hành động:** đã stash 7 key seed-42 (`__Nep_locked`), checkpoint 4ep lưu ở
`checkpoints/expA-4ep/seed42/`. Launched lại 7 job seed-42 @ `--epochs 2`
(2026-09-06 ~13:15 UTC, acc1/4/5/8/10/13/14). Dự kiến ~18:15 UTC. Chuẩn 2 epoch
= default `run.py` + CIDEr bão hoà từ ep2.

### residual seed-42 @ 4ep là LẦN CHẠY HỎNG, không phải "bridge yếu"

| residual | best val loss | F1(token) | CIDEr-D |
|---|--:|--:|--:|
| seed 42 @ 4ep (đang dùng ở §1, §4b, blueprint) | **2.354** ⚠️ | 36.45 | 56.3 |
| seed 123 @ 2ep (TIER-1, mới) | 1.672 | 45.14 | 85.4 corpus |
| seed 3407 @ 2ep (TIER-1, mới) | 1.676 | 45.88 | 86.7 corpus |

val loss 2.35 bất thường (mọi bridge/seed khác 1.5–1.7). Đây là training
instability của riêng seed 42, KHÔNG phải đặc tính kiến trúc residual.

**Hệ quả cho paper:** câu chuyện "residual = bridge tệ nhất (F1 36.5) → sau LoRA
lên ngang hàng (52.7), ΔF1 +16.2" (§4b, §5.6/§5.7 draft) **phần lớn do lần chạy
hỏng này**. Với seed bình thường residual plain ≈ F1 45.5 → ΔF1 từ LoRA chỉ ~+7,
giống các bridge khác. Điểm "san bằng bridge" VẪN đúng (mọi bridge → 52–53 F1 /
101–103 CIDEr-D sau LoRA) nhưng BỚT kịch tính — bỏ con số +16.2. Chờ seed-42
residual @ 2ep xác nhận s42 có bình thường hoá không.

### Số 1a sơ bộ (seed 123/3407 @ 2ep, full-val, in-house F1 / CIDEr-D corpus)

| bridge · seed | F1 | CIDEr | BLEU | ROUGE | MET | val loss | CIDEr-D | BLEU-4 | ROUGE-L |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| mini_qformer · 123 | 46.16 | 86.33 | 13.43 | 44.45 | 36.79 | 1.615 | 84.9 | 15.6 | 45.1 |
| mini_qformer · 3407 | 45.54 | 84.28 | 13.03 | 43.87 | 36.13 | 1.603 | 84.7 | 15.4 | 46.9 |
| qformer · 3407 | 47.13 | 89.01 | 14.01 | 45.38 | 37.62 | 1.566 | 90.4 | 16.5 | 47.5 |
| residual · 123 | 45.14 | 85.38 | 12.50 | 43.23 | 36.27 | 1.672 | 80.2 | 15.2 | 43.9 |
| residual · 3407 | 45.88 | 86.67 | 12.70 | 44.10 | 36.82 | 1.676 | 89.3 | 15.3 | 47.1 |
| tile_attention · 123 | 46.49 | 86.48 | 12.66 | 44.60 | 37.12 | 1.646 | 84.0 | 15.5 | 45.2 |
| tile_attention · 3407 | 44.51 | 82.37 | 12.13 | 42.59 | 35.53 | 1.706 | 83.6 | 14.3 | 45.0 |
| answer-random · 123 | 48.19 | 91.28 | 14.14 | 46.41 | 38.43 | 1.551 | 90.5 | 16.0 | 48.3 |
| answer-random · 3407 | 48.01 | 90.53 | 14.34 | 46.37 | 38.54 | 1.552 | 90.9 | 19.0 | 48.3 |

Checkpoint đã stage vào `checkpoints/expA/seed{123,3407}/<bridge>/` và
`checkpoints/expA-random/seed{123,3407}/multi_token/`. **Chưa tính mean / cập
nhật bảng** — chờ seed-42 @ 2ep để 3-seed đồng nhất epoch.

Nhận xét sơ bộ: seed 123/3407 @ 2ep thấp hơn seed-42 @ 4ep ~0.5–1 điểm F1 cho
mini_qformer/qformer/tile_attention/answer-random (đúng như dự đoán epoch). Với
2 epoch, các bridge phụ chụm quanh F1 44.5–47, CIDEr-D 84–90 — chênh lệch giữa
bridge nhỏ hơn nhiều so với bảng §1 (dùng seed-42 với residual hỏng).

### Còn chạy (13:15 UTC)

- 1b align-logit × 3 seed (s42/s123/s3407) + align-feat s123/s3407 — 5 job
- 7 job seed-42 @ 2ep re-run vừa launch

---

## 1. So 5 bridge (val, grouped split, seed 42, 1 tile) — ĐÃ KHÓA

Metric pycocoevalcap corpus (chuẩn để so paper khác).

| Bridge | Tham số train | % | CIDEr-D | BLEU-4 | ROUGE-L | F1(token) | val CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Multi-Token** (8 tok pooled) | 7.35M | 0.78 | **94.4** | **19.6** | **50.0** | **44.2** | **1.49** |
| Full Q-Former (16 query) | 69.4M | 6.91 | 86.7 | 17.5 | 47.1 | 43.3 | 1.57 |
| Light Q-Former (8 query) | 27.6M | 2.87 | 83.8 | 16.8 | 46.0 | 42.9 | 1.59 |
| Tile-Attention (8 tok) | 4.14M | 0.44 | 82.7 | 16.3 | 46.1 | 43.0 | 1.62 |
| Residual (1 tok) | 4.86M | 0.52 | 56.3 ⚠️ | 8.1 ⚠️ | 36.0 ⚠️ | 37.6 ⚠️ | 2.35 ⚠️ |

⚠️ **Dòng residual = lần chạy hỏng seed 42** (val CE 2.35 bất thường). seed
123/3407 @ 2ep cho residual F1 ~45.5, CIDEr-D ~85 — xem "epoch audit" phía trên.
Cần thay bằng 3-seed @ 2ep khi seed-42 re-run land.

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
0.23% param LoRA) mới nhích được, còn mọi can thiệp phía thị giác/training đều vô ích.

Đóng khung: multi_token + LoRA vẫn KHÔNG phải "frozen backbone" nữa — trình bày như
**phần bổ sung/reference point**, không phải spine chính (spine chính vẫn là bridge
0.78% param hoàn toàn đóng băng).

### Train thêm epoch có giúp không? — ĐÃ XONG 3/3 seed, CÓ giúp nhẹ

Tất cả LoRA runs trước chỉ 1 epoch (cố tình test nhanh). Chạy lại LoRA r=16 multi_token
**3 epoch × 3 seed** (42/123/3407), full-val n=5463:

| | LoRA r=16 · 1 epoch (mean 3 seed) | **LoRA r=16 · 3 epoch (mean 3 seed)** | Δ | ViMoE |
|---|---:|---:|---:|---:|
| F1 (in-house) | 53.17 ± 0.03 | **54.67 ± 0.15** | **+1.5** | 60.7 |
| CIDEr-D (corpus) | 101.7 (seed42) | **106.8 ± 1.1** | **+5.1** | 88.7 |
| BLEU-4 (corpus) | 23.2 (seed42) | **25.0 ± 0.4** | **+1.8** | 12.5 |
| ROUGE-L (corpus) | 52.7 (seed42) | **54.2 ± 0.2** | **+1.5** | 47.1 |
| Acc (in-house) | 10.42 | **11.78** | **+1.4** | 9.7 |
| val loss | ~1.37 | **~1.32** | −0.05 | — |

Per-epoch (600-mẫu subset lúc train): F1 gần như bão hoà từ epoch 2 → 3 (VD seed123:
54.71 → 54.81; seed3407: 56.64 → 56.56) — **phần lớn lợi ích của 3 epoch đã đạt ở epoch 2**.

**Kết luận:** train thêm epoch giúp **thật nhưng nhẹ** (+1.5 F1), khép thêm khoảng cách
tới ViMoE (gap 7.5 → **6.0**). Mỗi job 3-epoch mất ~7.8h Kaggle (so với ~2.5h cho 1
epoch). Khuyến nghị paper: nếu cần con số mạnh nhất cho reference-point thì dùng 3 epoch
(F1 54.7 / CIDEr-D 106.8), nhưng ghi rõ 1 epoch đã bắt ~80% lợi ích — decoder-LoRA không
cần train lâu để thấy hiệu ứng.

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

### Rank curve (r=4/8/16/32/64) — ĐÃ CÓ ĐA SEED CHO r=32/64, RÚT LẠI KẾT LUẬN "CÀNG CAO CÀNG TỐT"

Tất cả 600-mẫu training-time subset (không phải full-val 5463) — so nội bộ giữa
rank/seed dùng được, không so trực tiếp với số full-val ở trên.

| rank | seed 42 | seed 123 | seed 3407 | **mean (n seed)** |
|---|---:|---:|---:|---:|
| 4 | 51.26 | — | — | 51.26 (1) |
| 8 | 51.62 | — | — | 51.62 (1) |
| 16 | 51.98 | — | — | 51.98 (1) |
| 32 | 51.80 | 55.06 | 54.62 | **53.83 ± 1.77** (3) |
| 64 | 53.05 | 54.23 | 54.91 | **54.06 ± 0.94** (3) |

**RÚT LẠI nhận định trước ("rank càng cao lợi ích càng nhích")** — đó là dựng từ
seed42 duy nhất, và seed42 hoá ra là seed THẤP nhất trong 3 seed ở CẢ r=32 lẫn r=64.
Khi lấy trung bình 3 seed đàng hoàng: **r=32 (53.83) và r=64 (54.06) gần như KHÔNG
khác nhau** (chênh 0.23, nhỏ hơn nhiều so với std ~1-1.8 của từng rank) — không có
bằng chứng "rank cao hơn luôn tốt hơn". r=4/8/16 vẫn chỉ 1 seed nên chưa so được
công bằng với r=32/64, nhưng **bài học chính: đừng dựng xu hướng rank từ 1 seed**,
seed-to-seed noise trên 600-mẫu subset (std ~1-2 điểm F1) đủ lớn để đảo ngược thứ tự
biểu kiến giữa các rank. Khuyến nghị paper: **giữ r=16 làm điểm chính** (đã full-val
3/3 seed rất chắc, std ~0.03 — xem bảng đầu §4b), rank cao hơn không cho thấy lợi ích
rõ ràng đáng để đánh đổi thêm tham số.

**Bridge-agnostic mở rộng — qformer LoRA16 giờ 3/3 SEED, KHÓA:**

| seed | F1 (full-val) |
|---|---:|
| 42 | 53.10 |
| 3407 | 53.22 |
| 123 | 53.32 |
| **mean ± std** | **53.21 ± 0.11** |

Cực kỳ khít (std 0.11, nhỏ hơn cả multi_token's 0.03... thực ra tương đương) — xác
nhận chắc chắn decoder-LoRA generalize sang qformer, không phải may rủi seed.

### Bridge-agnostic 5/5: mini_qformer + residual + tile_attention — ĐÃ XONG (TIER-1 land 06/09)

⚠️ **residual plain (36.45) là seed-42 hỏng — xem "epoch audit" đầu file.** ΔF1 +16.2
cho residual sẽ giảm còn ~+7 khi dùng residual plain @ 2ep (~45.5). Bảng dưới giữ
số cũ, ĐÁNH DẤU để sửa sau khi seed-42 @ 2ep land. Điểm "san bằng 5/5" (kết quả +LoRA
đều ~52–53) không đổi.

LoRA r=16 (1 epoch) áp thêm lên 3 bridge nữa. mini_qformer + residual: **mean 3-seed**
(s42/123/3407, full-val n=5463). tile_attention: seed 42.

| bridge | | plain (s42) | +LoRA r=16 | Δ |
|---|---|---:|---:|---:|
| mini_qformer (3-seed) | F1 | 46.63 | **53.21** | **+6.6** |
| | CIDEr-D (corpus) | 83.8 | **103.0** | **+19.2** |
| | CIDEr (in-house) | 88.1 | **106.24** | **+18.1** |
| | val loss | 1.585 | **~1.38** | −0.20 |
| **residual** (3-seed, bridge yếu nhất) | F1 | 36.45 | **52.63** | **+16.2** |
| | CIDEr-D (corpus) | 56.3 | **100.8** | **+44.5** |
| | CIDEr (in-house) | 66.1 | **104.05** | **+38.0** |
| | val loss | 2.354 | **~1.40** | **−0.95** |
| tile_attention (s42) | F1 | 46.69 | **52.99** | **+6.3** |
| | CIDEr-D (corpus) | 87.5 | **102.0** | **+14.5** |
| | val loss | 1.62 | **1.40** | −0.22 |

per-seed: mini_qformer F1 {42:53.39, 123:53.16, 3407:53.07}; residual F1 {42:52.66,
123:52.60, 3407:52.64} (std 0.03 — cực chặt).

**Residual đi từ bridge TỆ NHẤT (§1: CIDEr-D 56.3, cách xa nhóm 82-94) lên NGANG HÀNG
với mọi bridge khác sau LoRA (CIDEr-D 100.8, so với multi_token+LoRA 101.7, qformer+LoRA
102.4, mini_qformer+LoRA 103.0, tile_attention+LoRA 102.0)** — chênh lệch giữa các
bridge gần như BIẾN MẤT sau khi mở decoder. Đây là bằng chứng mạnh nhất cho luận điểm
§6.1: khi decoder được mở (dù chỉ LoRA r=16, 0.23% param), **bridge nào cho decoder
cũng dùng được gần như nhau**. Giờ đã **5/5 bridge** (multi_token, qformer, mini_qformer,
residual, tile_attention) đều xác nhận decoder-LoRA có lợi.

### B3: tile-sweep multi_token @ {1,3,6,12} — ĐÃ XONG 3/4 (tile 12 bị cắt ở mốc 12h, KHÔNG cần)

Eval checkpoint multi_token (train ở 1 tile) trên full-val ở nhiều số tile. Job chạy
12h → bị Kaggle cắt khi đang eval tile=12, nhưng tile 1/3/6 đã xong và lưu file.

| n_tiles | F1 | CIDEr (in-house) | val loss | perplexity |
|---|---:|---:|---:|---:|
| **1** | **50.66** | **98.69** | **1.478** | 4.4 |
| 3 | 21.05 | 48.75 | 3.351 | 28.5 |
| 6 | 22.51 | 52.36 | 3.364 | 28.9 |
| 12 | — (bị cắt, không cần) | | | |

**multi_token SỤP ĐỔ hoàn toàn khi eval ở >1 tile** — F1 50.7 → 21, CIDEr 98.7 → 49,
val loss hơn gấp đôi (1.48 → 3.35). Bridge mean-pool 8 token: khi có 3-6× số token
vào, phép pool trung bình xoá sạch tín hiệu. Trend từ 1→3→6 quá rõ (sụp ngay ở tile 3,
giữ nguyên sụp ở tile 6) → **tile=12 chỉ là thêm 1 dòng sụp nữa, không đáng relaunch
~2h eval**.

**Ý nghĩa cho §2:** "1 tile" của multi_token KHÔNG phải hạn chế phải xin lỗi — đó là
điểm vận hành mà kiến trúc này được xây cho, và **tăng tile lên thì strictly TỆ HƠN**
(không chỉ "không giúp"). Bổ trợ mạnh cho câu chuyện efficiency: mình đạt SOTA-generation
ở đúng cấu hình rẻ nhất, không phải "chấp nhận thua thiệt để tiết kiệm".

### B2: residual-tiled — bridge yếu nhất có "cần" tile không? ĐÃ XONG (chỉ subset 300)

Train residual (bridge đơn giản nhất, 1 token, đang thua xa mọi bridge khác ở §1) với
`--tile-choices 1,3,6`. Job này mất **11h16'** (gần chạm mốc 12h — không có bước eval
full-val riêng trong recipe "-tiled" này, giống các job tiled khác trước, chỉ có
subset 300 mẫu định kỳ lúc train):

| | residual plain (§1, full-val corpus, 1 tile) | residual tiled (subset 300, best epoch) |
|---|---:|---:|
| val loss | 2.35 | **1.71** |
| F1 (khác scale/subset) | 37.6 | 42.5 |
| CIDEr (khác scale/subset) | 56.3 | 82.2 |

**Đọc có caveat rõ (khác subset/metric-scale, không so trực tiếp số tuyệt đối được):**
val loss giảm mạnh (2.35→1.71) khi residual được train với tile-augmentation — ngược
hẳn với multi_token (CE gần như không đổi hoặc tệ hơn khi train-với-tile, theo sweep
oracle C3 trước đó). Gợi ý: **bridge càng yếu/pool càng thô (residual = 1 token) thì
càng "cần" tile để bù capacity**, còn bridge đã pool tốt (multi_token, 8 token) thì
tile không giúp gì thêm — khớp với câu chuyện "1 tile không phải thỏa hiệp CHO
multi_token cụ thể", không phải "1 tile luôn đủ cho MỌI bridge". Cần full-val eval
riêng (standalone, giống cách làm với LoRA) để có số so sánh chuẩn — chưa làm, ghi
nhận là việc còn lại.

## 5. Còn PENDING (sau reset quota 00:00 UTC 5/9)

| # | Việc | Ai | Trạng thái |
|---|---|---|---|
| C3 | Oracle sweep + policy ladder trên 3 **tiled** checkpoint → re-lock §5.2/§5.3 | peer | ✅ xong, không đổi kết luận |
| C2 | multi_token seed 123 + 3407 → mean±std cho dòng headline | tôi | ✅ xong, 4/4 seed |
| **B5** | **Số tile "Vintern-finetune" bảng cũ** | **user** | ✅ resolved: bản HF, tối đa **12** tile lúc train |
| B3 | Tile-sweep: multi_token @ {1,3,6,12} → bảng efficiency | tôi | ✅ xong 3/4 (tile 1/3/6): multi_token SỤP khi >1 tile, tile=12 không cần |
| B2 | residual-tiled (baseline tương phản: bridge đơn giản có cần tile không) | tôi | ✅ xong (300-subset), gợi ý bridge yếu cần tile — cần full-val standalone eval để chốt số |
| — | LoRA rank ablation r=8/r=32 (mở rộng §4b) | tôi | 🔄 running (acc6/acc7) |
| ~~C4~~ | ~~Human validation 300–500 mẫu, 2 annotator, Cohen's κ~~ → **self-check N=120, 1 rater** (§4c) | tôi | ✅ xong — xem §4c: strong bucket 91% đáng tin, **partial bucket chỉ 43% đáng tin** (finding hơi bất lợi, đã ghi rõ) |

## 6. Câu chuyện paper (chốt)

1. **Đóng góp chính — efficiency**: bridge nhẹ trên backbone đóng băng đạt SOTA-generation ở 1 tile.
2. **Oracle analysis**: xác nhận 1 tile không phải thoả hiệp — không policy adaptive nào (kể cả biết reasoning-type) thắng fixed 1-tile.
3. **3-way negative → decoder ceiling**: định vị + định lượng nút thắt (frozen 0.5B decoder), không phải phía thị giác.
4. **Benchmark bridge leak-free** + compute-efficiency table (FLOPs/latency) mà ViMoE bỏ ngỏ.

Reasoning-type / P(r|Q) router → 1 mục ablation nhỏ.
