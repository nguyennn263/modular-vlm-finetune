# Kế hoạch hoàn thiện paper — "Ngân sách tham số nhỏ nên tiêu vào đâu?"

*Lập 2026-09-06. Deadline ~2026-09-27 (ACIIDS 2027 / Trust4NLP). Quota: ~290h/tuần, reset 09-12.*

Khung câu chuyện: xem artifact "Ngân Sách Tham Số"
(https://claude.ai/code/artifact/bb7bf7ee-d5f1-4749-bb56-29a5c5daa610).

---

## Câu hỏi tổng quát

> Khi adapt frozen VLM → VQA ngôn ngữ ít tài nguyên với ngân sách tham số nhỏ,
> nên tiêu ngân sách vào **phía thị giác** (bridge / tile) hay **phía ngôn ngữ** (decoder)?

**Trả lời hiện tại:** phía thị giác đã bão hoà (4 trục âm); decoder là trục duy nhất
dương. → công thức: `bridge pool rẻ (cố định) + LoRA decoder nhẹ`.

---

## Còn thiếu gì (gap → việc)

| # | Gap | Mức độ | Việc |
|---|---|---|---|
| G1 | Chỉ 1 dataset (AutoViVQA) → không "tổng quát hoá" được | **NẶNG** | Chạy pipeline trên dataset Việt thứ 2 (ViVQA) |
| G2 | Bảng so 5 bridge (§5.1) chỉ multi_token đa seed, 4 bridge còn lại seed 42 | vừa | Chạy 4 bridge × 2 seed nữa |
| G3 | Các dòng ÂM (§5.5: align-feat/logit, answer-sampling) chỉ seed 42 → reviewer nói "1 seed, nhiễu" | vừa | Chạy 3 dòng âm × 2-3 seed |
| G4 | align-logit bị cắt ở ep2 subset, chưa full-val chuẩn | vừa | Chạy lại full-val, 3 seed |
| G5 | LoRA bridge coverage: mini_qformer + residual chỉ 1 seed; tile_attention chưa test LoRA | vừa | Đủ 5/5 bridge, 3 seed cho 2 bridge chính |
| G6 | Chưa biết LoRA giúp ở **đâu trong decoder** (attn? MLP? cả hai?) | nhẹ (làm sâu) | LoRA target ablation |
| G7 | Chưa có **upper bound**: mở decoder hết cỡ recover được bao nhiêu gap? | nhẹ | Full-finetune decoder / LoRA-all rank cao, 1 seed |
| G8 | Hầu hết số là val, chưa chốt test | vừa | Eval checkpoint đã khoá trên test split |
| G9 | Human validation chỉ self-check N=120, 1 rater | vừa (Trust4NLP) | **Cần user + 1 người**: 300 mẫu, 2 annotator, Cohen's κ |
| G10 | Chưa có paired-bootstrap CI cho các so sánh chính | nhẹ | Peer / script offline |

---

## TIER 1 — Multi-seed hoá (pure rerun, ~26 job, 1-2 wave, KHÔNG code mới)

Mục đích: mọi kết quả hiện có thành "3-seed defensible".

### 1a. Bridge comparison → 3 seed (G2)
- `expa:qformer:s3407`
- `expa:residual:s123 s3407`
- `expa:mini_qformer:s123 s3407`
- `expa:tile_attention:s123 s3407`
→ 7 job, `--epochs 2`

### 1b. Dòng ÂM → 3 seed (G3, G4)
- `expa-align-feat:multi_token:s123 s3407`
- `expa-random:multi_token:s123 s3407`
- `expa-align-logit:multi_token:s42 s123 s3407` (chạy lại full-val, α=1.0 — báo "âm nhất quán dù mis-weight")
→ 7 job

### 1c. LoRA bridge coverage → 5/5 (G5)
- `expa-lora16:mini_qformer:s123 s3407`
- `expa-lora16:residual:s123 s3407`
- `expa-lora16:tile_attention:s42` (+ plain đã có ở 1a)
→ 5 job

### 1d. Test-set eval (G8)
- multi_token plain s42/s123/s2026/s3407 trên test
- multi_token LoRA s42/s123/s3407 trên test
→ ~7 eval job (rẻ)

---

## TIER 2 — Làm sâu câu chuyện decoder (~10 job, code nhẹ)

### 2a. LoRA giúp ở ĐÂU trong decoder? (G6)
- attn-only (`q,k,v,o`) — ĐÃ CÓ (= r=16 hiện tại)
- MLP-only (`gate_proj,up_proj,down_proj`)
- attn+MLP (cả 7)
→ multi_token, 3 seed × 2 config mới = 6 job. Dùng `--lora-targets`.

### 2b. Upper bound: mở decoder hết cỡ (G7)
- multi_token + full-finetune decoder (check OOM ở bs4; nếu OOM → LoRA-all r=64)
- 1 seed
→ 1-2 job. Cho biết "trần lý thuyết" — bao nhiêu % gap tới ViMoE recover được.

---

## TIER 3 — Dataset thứ 2 (G1, NẶNG NHẤT, tuần sau)

### ViVQA (khuyến nghị)
- **Vì sao**: ảnh COCO (cùng nguồn AutoViVQA, dễ lấy), nhỏ (~15k pair), có SOTA
  công bố (ViVQA-TranConI 71% acc), style **phân loại/câu ngắn** ↔ tương phản với
  AutoViVQA (sinh mở) → chứng minh câu trả lời budget-allocation đúng qua **cả 2 style**.
- **Setup**: tải ViVQA (HF `nngocson2002/ViVQA` hoặc GitHub) + ảnh COCO val2014 →
  `data/splits_vivqa/{train,val,test}.jsonl` cùng schema → upload Kaggle dataset →
  `run.py` thêm `--data-dir` / dataset override.
- **Chạy**: multi_token plain 3 seed + LoRA 3 seed + residual plain+LoRA 3 seed
  → ~12 job.
- **Kỳ vọng**: cùng pattern (vision phẳng, LoRA là đòn bẩy) → generalization mạnh.

### OpenViVQA (nếu còn thời gian)
- Sinh mở, khớp pipeline chính xác hơn, nhưng ảnh riêng (không COCO) → tải nặng hơn.

---

## TIER 4 — Cần user

### G9. Human validation thật
- 300–500 mẫu, 2 annotator, Cohen's κ. Assistant chuẩn bị sample + form; user + 1
  người chấm. Cho phép claim trustworthiness (tốt cho Trust4NLP).

---

## Thứ tự thực thi

1. **Ngay**: launch TIER 1a + 1b + 1c (19 training job) — 1 wave.
2. Khi land: TIER 1d (test eval) + TIER 2a (decoder localization).
3. Song song: assistant làm data engineering cho TIER 3 (ViVQA).
4. TIER 2b + TIER 3 chạy.
5. TIER 4: hẹn lịch với user.
6. Peer: paired-bootstrap CI (G10) + viết §5-§6 với số mới.
