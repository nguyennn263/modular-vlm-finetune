# Paper 3 — Kế hoạch v2 (SPINE = efficiency-bridge)

*Chốt 2026-09-03, cập nhật 2026-09-04. Deadline 2026-09-27. Quota Kaggle reset 2026-09-05 00:00 UTC.*

## ⚠️ UPDATE 2026-09-04: alignment-supervised bridging FAILED — spine reverts to efficiency-bridge

3 axes thử để khép F1 gap, **cả 3 âm** (seed 42, anchor `first` = F1 50.7 / CIDEr-D 94.4):

| intervention | F1 | CIDEr-D | verdict |
|---|---|---|---|
| answer-sampling=random | 49.0 | 87.3 | âm nhẹ |
| align-feat α=1.0 | 49.7 | 92.0 | âm nhẹ |
| align-logit α=1.0 | 40.7 (ep2 subset) | 80.1 | âm nặng — KL term chèn CE (val CE 2.84 vs 1.49) |

→ multi_token đã có CE thấp nhất trong 5 bridge (1.49), gần tối ưu với frozen Qwen2-0.5B.
**Frozen 0.5B decoder LÀ trần cho token-F1; capacity phía vision/training KHÔNG phải nút thắt.**
3 negative này → §6.1 synthesis (peer). §5.6 = bảng factual (peer đã commit da0e15d/d072cb4).

## Thesis (efficiency-bridge)

> **Thay projector MLP của Vintern-1B bằng bridge multi-token — chỉ train bridge (0.78%
> params), đóng băng InternViT + Qwen2 — đạt/vượt Vintern-1B finetune trên metric sinh,
> từ 1 tile ảnh thay vì tới 12 (rẻ hơn ×4–6 vision compute). Oracle analysis xác nhận
> 1 tile không phải thoả hiệp; 3-way negative xác nhận frozen decoder là trần.**

### (lịch sử — thesis alignment đã bỏ)

> ~~Với VLM frozen-backbone dùng LLM nhỏ, bổ sung alignment supervision cho bridge — dùng
> projector gốc Vintern-1B làm teacher — khép khoảng cách tới Vintern-finetune và ViMoE.~~

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

## Track A — Alignment supervision (DONE — NEGATIVE, dropped as spine)

| # | Việc | Trạng thái |
|---|---|---|
| A1–A2 | `--align-distill {logit,feat}` implement + verify | ✅ (commit b01ef30, 2164f66, 09ebfa3) |
| A3 | multi_token + align (feat @acc13, logit @acc11), seed 42, 1 tile | ✅ **cả 2 ÂM** — xem bảng trên. Staged `checkpoints/expA-align-{feat,logit}/` |
| A4 | retrain 3 bridge align+tiles | ❌ HỦY (A3 âm) |
| A5 | α sweep (0.1, 0.05) cho logit | tùy chọn — pattern đã rõ, ưu tiên thấp. Nếu làm: 1-epoch probe α=0.1 sau reset |

Code `--align-distill` giữ lại (off by default) cho reproducibility + §5.6.

---

## Track B — Efficiency / fair comparison (SUPPORTING → giờ là chính)

| # | Việc | Ai | Trạng thái |
|---|---|---|---|
| B1 | Tiled retrains: multi_token/qformer/mini_qformer `--tiles 1,3,6` | Kaggle | ✅ **XONG cả 3**, staged `checkpoints/expA-tiled/seed42/{multi_token,qformer,mini_qformer}/` (2ep mỗi bridge, val CE 1.55/1.60/1.65) |
| B2 | `linear_bridge` tile-sweep | — | ❌ BỎ (`linear_bridge` = alias của `ResidualBridge`, trùng `residual`). Nếu cần contrast: `residual --tiles 1,3,6` sau reset, ưu tiên thấp |
| B3 | Tile-sweep eval: mỗi bridge @ {1,3,6,12} + Vintern-finetune @ native | tôi | sau reset — dùng oracle output của C3 cho {1,3,6}; +eval riêng @12 |
| B4 | Bảng compute-efficiency P1 vào §5.5 | peer | ✅ (c72740e) |
| B5 | Chốt số tile "Vintern-finetune" bảng cũ | **user** | ⏳ chờ ("full tiles" — cần biết ≤6 hay ≤12) |

---

## Track C — Rigor

| # | Việc | Ai | Trạng thái |
|---|---|---|---|
| C1 | answer-sampling random | Kaggle | ✅ **ÂM** (F1 49.0 vs 50.7) — staged `checkpoints/expA-random/` |
| C2 | Multi-seed ≥3 (42, 123, 3407) cho multi_token headline (1-tile, plain, --epochs 2) | Kaggle | **sau reset** (00:00 UTC) — 2 acct |
| C3 | Oracle sweep + policy ladder trên `checkpoints/expA-tiled/seed42/{multi_token,qformer,mini_qformer}/` | peer | **sau reset** — re-lock §5.2/§5.3 trên tile-trained bridge (đóng confound 1-tile) |
| C4 | Human validation 300–500 mẫu, 2 annotator, Cohen's κ | **user** + tôi setup | chưa |
| C5 | Error analysis định lượng (noun omission, vague attr) | tôi | có thể làm ngay từ `checkpoints/expA/seed42/*/results/text_predictions_epoch_1.json` |

---

## Track D — Viết

| # | Việc | Ai |
|---|---|---|
| D1 | Reframe TOM-TAT.md + P6-draft quanh alignment-supervised bridging | tôi |
| D2 | Related work: BASIC/SEA/LangBridge/LaVer/VoCo/EvoComp | tôi |
| D3 | P6 hoàn chỉnh LNCS 12–15 trang | tôi + peer |

---

## Việc còn lại (sau khi A3 âm — spine đã chốt = efficiency-bridge)

**Sau 00:00 UTC 5/9 (quota reset, 8 acct × 30h):**
1. **C3** (peer): bundle 3 tiled checkpoint → oracle sweep val+test + policy ladder → re-lock §5.2/§5.3
2. **C2**: multi_token seed 123 + 3407, 1-tile, plain, `--epochs 2` → mean±std cho dòng headline
3. **B3**: eval multi_token/qformer/mini_qformer @ n_tiles 12 (bổ sung cho {1,3,6} từ C3) → bảng tile-sweep
4. (tùy) **A5**: α=0.1 logit 1-epoch probe — chỉ nếu muốn khép hẳn hướng alignment; ưu tiên thấp
5. (tùy) `residual --tiles 1,3,6` cho contrast "bridge thay tile"

**Làm ngay được (không cần Kaggle):**
- **C5** error analysis: phân loại lỗi trên `checkpoints/expA/seed42/*/results/text_predictions_epoch_1.json`
- **D1** reframe TOM-TAT.md quanh efficiency-bridge (peer đang giữ §5; tôi §1–4)
- **D2** related work: BASIC/SEA/LangBridge/LaVer/VoCo/EvoComp + note alignment-KD đã thử & âm

**Cần user:**
- **B5**: số tile "Vintern-finetune" bảng cũ (≤6 hay ≤12)
- **C4**: tổ chức human validation 300–500 mẫu, 2 annotator

## Job Kaggle (2026-09-04 13:20 — TẤT CẢ đã xong)

| Job | Kết quả |
|---|---|
| B1 tiled ×3 (mt/qf/mq) | ✅ staged `checkpoints/expA-tiled/seed42/` |
| C1 answer-sampling | ✅ ÂM, staged `checkpoints/expA-random/` |
| A3 align feat + logit | ✅ ÂM cả 2, staged `checkpoints/expA-align-{feat,logit}/` |
| B2 linear_bridge | ❌ ERROR (acc11 hết quota) — bỏ |

acc12 = `tn7012` (mới, 30h, chưa test github-clone). acc11 hết sạch quota tới reset.
