# Kế hoạch hoàn thiện paper — "Cải thiện Vintern-1B cách rẻ"

*Lập 2026-09-06. Deadline ~2026-09-27 (ACIIDS 2027 / Trust4NLP). Quota: ~290h/tuần, reset 09-12.*

Khung: artifact "Cải Thiện Vintern-1B Cách Rẻ" (https://claude.ai/code/artifact/bb7bf7ee-d5f1-4749-bb56-29a5c5daa610).
Bản thiết kế + bảng metric: artifact "Bản Thiết Kế Paper" (https://claude.ai/code/artifact/fe068b4c-d59c-429f-bdba-ed9ea93bd557).

## REFRAME 2026-09-06 (user)
Bỏ khung "budget allocation" trừu tượng → khung "cải thiện Vintern-1B":
Vintern zero-shot hỏng (F1 17.6) / finetune-toàn-bộ đắt (F1 53.8, 100x cost).
Câu hỏi: adapt rẻ (~1% param, frozen) đạt tương đương không, nút thắt ở đâu?
Đối lập ViMoE (xây model mới). 1 dataset của mình là ĐỦ — không thêm dataset 2.
Cấu trúc paper theo AutoViVQA/ViMoE (8 metric: Acc/P/R/F1/BLEU/ROUGE-L/METEOR/CIDEr).

---

## Câu hỏi tổng quát

> Thay vì xây model mới (như ViMoE), cải thiện Vintern-1B trên AutoViVQA một
> cách rẻ (~1% param, backbone đóng băng) đạt tương đương finetune-toàn-bộ
> không? Chưa đạt thì nút thắt ở đâu? → chẩn đoán 6-bước (RQ1-6).

**Trả lời hiện tại:** phía thị giác đã bão hoà (4 trục âm); decoder là trục duy nhất
dương. → công thức: `bridge pool rẻ (cố định) + LoRA decoder nhẹ`.

---

## Còn thiếu gì (gap → việc)

| # | Gap | Mức độ | Việc |
|---|---|---|---|
| ~~G1~~ | ~~Dataset thứ 2~~ — **BỎ** (user): AutoViVQA là benchmark mình cải thiện; ViVQA quá cũ. Tổng quát hoá qua trục **kiến trúc** (G7) + 4-trục ablation + bridge-equalizing thay vì dataset |
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

## TIER 3 — Tổng quát hoá theo trục KIẾN TRÚC (thay dataset thứ 2)

### G7 mở rộng: decoder frozen TO HƠN có tự đóng gap không?
- Claim là về regime "decoder nhỏ đóng băng". Test nhân quả: swap Qwen2-0.5B
  → decoder frozen lớn hơn (VD Viet-InternVL2-4B của 5CD-AI, hoặc 1 LM Việt
  1.5B+). Khoảng cách F1 có tự đóng KHÔNG cần LoRA không?
- Assistant scope độ khó tích hợp trước. Dễ → làm (6 job key); khó → bỏ, dựa
  vào rigor + scope claim trung thực cho AutoViVQA + kiến trúc Vintern-1B.

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
