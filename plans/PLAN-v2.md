# Paper 3 — Kế hoạch v2 (pivot: alignment-supervised bridging)

*Chốt 2026-09-03. Deadline 2026-09-27 (~24 ngày). Quota Kaggle reset 2026-09-05.*

## Thesis mới

> **Với VLM frozen-backbone dùng LLM nhỏ (Qwen2-0.5B), bổ sung alignment supervision
> cho bridge — dùng chính đường projector gốc của Vintern-1B làm teacher — khép được
> khoảng cách tới cả full-finetune Vintern-1B và ViMoE-VQA, ở backbone đóng băng và 1
> tile ảnh.**

Đóng góp:
1. **[chính]** Alignment-supervised bridge: `L = L_CE + α·L_align`, teacher = `mlp1`
   gốc của Vintern (đã align qua pretraining). Đặc biệt quan trọng cho LLM nhỏ (SEA).
2. **[hỗ trợ]** Efficiency: 0.78% tham số, frozen, 1 tile thay vì tới 12 (×4–6 rẻ).
3. **[hỗ trợ]** Oracle analysis: adaptive tile allocation KHÔNG giúp → gains đến từ
   alignment, không phải compute. Reasoning-type routing = ablation nhỏ.
4. **[hỗ trợ]** Benchmark bridge leak-free trên AutoViVQA + bảng compute-efficiency.

Related work để định vị: BASIC (ICCV'25), SEA (EMNLP'25), LangBridge (ICCV'25),
LaVer (CVPR'26), VoCo-LLaMA (CVPR'25), EvoComp (CVPR'26 — pattern oracle→distill).

---

## Fact đã xác nhận

- CE thực tế wave-1b: multi_token **1.49** (thấp nhất), qformer 1.57, residual 2.35.
  "CE nổ lên 12" = term MSE distillation hỏng (teacher = linear random), đã tắt.
  → Bridge KHÔNG misalign nặng; alignment supervision là để **đẩy xa hơn**, không phải sửa.
- Vintern-1B ([arxiv 2408.12480](https://arxiv.org/pdf/2408.12480)): full-finetune ViT +
  LoRA LLM, dynamic tiling ≤12 (train) / ≤40 (test). `mlp1` = projector gốc:
  `LayerNorm(4096) → Linear(4096→896) → GELU → Linear(896→896)`, input 4096 do pixel
  shuffle (gộp 2×2 patch, giảm token còn 1/4), output 256 token 896-d mỗi tile.
- `base_model.extract_feature(pixel_values)` cho teacher visual tokens; `base_model.mlp1`
  truy cập được.
- Multi_token (1-tile, seed 42): CIDEr-D 94.4 / BLEU-4 19.6 / ROUGE-L 50.0 / F1 ~50 / Acc 8.6
  - vs Vintern-finetune (F1 53.8, CIDEr 72.8): thắng generation, thua F1/Acc/ROUGE chút
  - vs ViMoE (F1 60.7, CIDEr 88.7): thắng 4 metric sinh, thua F1 −10 / Prec −11

Mục tiêu: **beat Vintern-finetune** (gần rồi) → **beat ViMoE** (cần alignment để khép F1).

---

## Track A — Alignment supervision (SPINE)

| # | Việc | Ai | Chi phí | Phụ thuộc |
|---|---|---|---|---|
| A1 | `--align-distill` trong trainer: teacher forward (`mlp1` path) → KL logit answer-token vs student. `TrainConfig.align_distill`, `align_weight`, `align_type ∈ {logit, feat}` | tôi | code ~60 dòng | — |
| A2 | Verify local: import, `train.py --help`, dry-run, smoke CPU | tôi | — | A1 |
| A3 | Train multi_token + align (logit), seed 42, 1 tile, 4ep → so F1/CIDEr vs bản không align | Kaggle 1 acct | ~10h | A2 |
| A4 | Nếu A3 tăng: retrain 3 bridge (multi_token, qformer, mini_qformer) với align + tile-aug | Kaggle 3 acct | ~30h | A3 |
| A5 | Variant: feat-distill + BASIC `L_direction` (cosine bridge vs teacher pooled) | tôi + Kaggle | ~10h | A3 (nếu còn thời gian) |

**Teacher chi tiết:** `t_tokens = mlp1(pixel_shuffle(vision_model(pv)))` → (B, 256·T, 896).
`L_logit`: chạy Qwen2 với `[t_tokens; text_emb]` → logits teacher trên vị trí answer;
KL(student ‖ teacher.detach()) trên answer tokens. 1 forward Qwen2 thêm/batch (rẻ).

---

## Track B — Efficiency / fair comparison (SUPPORTING)

| # | Việc | Ai | Trạng thái |
|---|---|---|---|
| B1 | Tiled retrains: multi_token/qformer/mini_qformer `--tiles 1,3,6` | Kaggle acc11/9/10 | **ĐANG CHẠY** |
| B2 | `linear_bridge` (projector đơn giản) tile-sweep → contrast "bridge thay tile" | Kaggle | chưa (thấp ưu tiên) |
| B3 | Tile-sweep eval: mỗi bridge @ {1,3,6,12} + Vintern-finetune @ native | tôi | sau B1 |
| B4 | Bảng compute-efficiency P1 vào §5.5 | peer | **XONG** (c72740e) |
| B5 | Chốt số tile của "Vintern-finetune" trong bảng cũ | **user** | chờ |

---

## Track C — Rigor

| # | Việc | Ai | Trạng thái |
|---|---|---|---|
| C1 | answer-sampling random (train 5 ref) — probe F1 gap | Kaggle acc13 | **ĐANG CHẠY** |
| C2 | Multi-seed ≥3 (42, 123, 3407) cho dòng bridge headline | Kaggle | sau A4/B3 |
| C3 | Re-run oracle + policy trên checkpoint tiled/aligned | peer | sau A4/B1 |
| C4 | Human validation 300–500 mẫu, 2 annotator, Cohen's κ | **user** + tôi setup | chưa |
| C5 | Error analysis định lượng (noun omission như ViMoE 10.7%, vague attr) | tôi | sau A3 |

---

## Track D — Viết

| # | Việc | Ai |
|---|---|---|
| D1 | Reframe TOM-TAT.md + P6-draft quanh alignment-supervised bridging | tôi |
| D2 | Related work: BASIC/SEA/LangBridge/LaVer/VoCo/EvoComp | tôi |
| D3 | P6 hoàn chỉnh LNCS 12–15 trang | tôi + peer |

---

## Timeline

| Tuần | Việc chính |
|---|---|
| **1** (nay–10/9) | A1–A3, B1 lands, C1 lands, B3, D1–D2, chốt B5 |
| **2** (10–17/9) | A3 verdict → A4, C2, C3, C5, A5 |
| **3** (17–24/9) | C4 human eval, D3 viết, buffer cho seed |
| **4** (24–27/9) | Polish, nộp EasyChair |

**Điểm quyết định:** A3 xong (~10/9). Nếu align tăng F1 rõ → A4 + spine chốt. Nếu không
→ lùi về "efficiency bridge" thuần (Track B là chính), align thành 1 ablation âm.

---

## Job Kaggle đang chạy (2026-09-03 ~21:30)

| Job | Acct | Track |
|---|---|---|
| mvlm-expa-tiled-multi-token-s42 | acc11 | B1 |
| mvlm-expa-tiled-qformer-s42 | acc9 | B1 |
| mvlm-expa-tiled-mini-qformer-s42 | acc10 | B1 |
| mvlm-expa-random-multi-token-s42 | acc13 | C1 |

Free để launch A3: acc4/6/7/8 (quota thấp, reset 5/9) hoặc chờ 1 job B1 xong.
acc12 (trngtinanh) chờ user verify SĐT.
