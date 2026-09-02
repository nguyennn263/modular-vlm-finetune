# Sau khi v12 xong — việc còn lại

> Trạng thái: toàn bộ **code P0–P5 đã viết + unit-test + verify smoke trên Kaggle**.
> Còn lại = chạy compute nặng (bạn launch) + phần thủ công + viết bài.

---

## EXECUTION LOG (cập nhật 2026-09-02)

Chạy 5-account song song qua `scripts/parallel/run.py` (ledger: `outputs/parallel/ledger.json`).

### P4 fiq — ✅ XONG (kernel `mvlm-fiq`, acc4, ~2h)
- `outputs/fiq/{train,val,test}.parquet` + `pca.npz` (PCA basis persist).
- `outputs/router/prq_{train,val,test}.parquet` + `best.pt` (515MB, không version).
- **Router P(r|Q)**: val macro-F1 **0.913**, acc 0.946. Mạnh: counting 1.0, relational .98, causal .96. Yếu: context .66 (nhầm recognition).

### Exp A wave 1 — ❌ HỎNG, chạy lại
- Bug: **early stopping theo val-CE** dừng ở epoch 2 (CE chạm đáy <1 epoch, CIDEr còn lên dốc). `best_model.pt` = under-trained.
- 4/5 kernel bị Kaggle **CANCEL ở 12h** (full-val generation mỗi epoch tốn ~1.7h) → Kaggle **không lưu output kernel bị cancel** → mất trắng.
- Chỉ `mini_qformer` COMPLETE: mốc tham khảo 2-epoch, val CIDEr 0.853, BLEU .135, ROUGE-L .444.
- Fix (`c9a2723`): `--no-early-stopping`, lưu `last_model.pt` mỗi epoch, evaluate `last_model.pt`, `--text-metrics-every 2 --text-metrics-max-samples 800`.

### Exp A wave 1b — 🔄 ĐANG CHẠY (launch 2026-09-02 ~07:53, 5 bridge × seed 42)
- `--epochs 4 --no-early-stopping --batch-size 8`. Dự kiến qformer ~9.3h (margin ~2.7h dưới cap 12h).
- Xong → `run.py poll` gom về `checkpoints/expA/seed42/<b>/last_model.pt` + `outputs/expA/seed42/<b>/eval_val_samples.jsonl`.
- Nếu chưa hội tụ → `run.py resume expa:<b>:s42 --epochs 8` (tiếp từ epoch 4).

### ⚠️ Quota
Wave 1 đốt ~12h/account (acc1/2/3/5). + wave 1b ~9h → ~21h/30h tuần này. Oracle sweep cần ~8h/account nữa → **có thể phải chờ Kaggle reset quota tuần** trước khi chạy oracle.

### Còn lại: Exp B → oracle (train/val/test) → policy ×3 → eval_ladder → human eval → viết P6.

---

## 0. Housekeeping (nhanh)

- Merge `chore/repo-restructure` → `main` (hoặc `feat/master-plan`).
- Sau merge: sửa biến `BRANCH` trong `notebooks/kaggle_runner.ipynb` (`build_kaggle_runner.py` cell 2) → `main`, rồi `kaggle kernels push -p notebooks/`.
- `notebooks/_kaggle_pulled/` đã gitignore — kệ.

---

## 1. QUYẾT ĐỊNH cần chốt trước khi chạy

| # | Quyết định | Lựa chọn |
|---|---|---|
| 1 | **Số seed Exp A** | 1 seed (staged, ~25 GPU-h) HAY 3 seed (~75 GPU-h) ngay |
| 2 | **Số epoch** | 10 (như run cũ) HAY 6–7 (check plateau ở seed đầu → nếu metric bão hoà thì cắt) |
| 3 | **Nơi chạy phần nặng** | Kaggle (30h/tuần → oracle sweep 1 mình đã ~30-40h → 4-5 tuần) HAY thuê GPU (A100 ~$1.5/h, ~$100-200, vài ngày) |
| 4 | **Bridge cho multi-tile** | Train lại top-3 bridge với `tile_choices=[1,3,6]` (chặt chẽ, +15h) HAY dùng checkpoint n_tiles=1 cho oracle (nhanh, ghi chú domain shift) |
| 5 | **Human validation** | Ai chấm 300–500 câu? 2 người + Cohen's κ |

---

## 2. Exp A — marathon (BẠN launch)

`scripts/phase2_expA.py` (hiện set 5 bridge × 3 seed). Mỗi run:

```
python -m src.cli.train --bridge <b> --split-dir data/splits --seed 42 \
    --output-dir checkpoints/expA/seed42 --resume
```

- 5 bridge: `residual multi_token tile_attention mini_qformer qformer`
- ~4.5h/bridge (residual/multi_token), ~6h (qformer) ở batch 2 × accum 4, 10 epoch
- Checkpoint ghi `/kaggle/working/checkpoints/expA/seed42/<bridge>/` — **Save Version** để persist qua session; `--resume` tự lấy `step_*.pt` mới nhất
- Sau khi có 5 checkpoint → so số với `plans/results-5bridge.md` (split cũ) để sanity-check split mới hợp lý

**Đề xuất staged:** 5×1 seed trước → chạy Exp B → nếu heatmap phẳng thì DỪNG ở đây (dùng 1 bridge tốt nhất, final-plan §10). Chỉ 5 run.

---

## 3. Sau khi có checkpoint Exp A — pipeline (gần như tự động)

```
python scripts/phase3_expB.py    # eval 5 bridge trên VAL + fork analysis
python scripts/phase4_oracle_policy.py
python scripts/phase5_eval.py
```

Chi tiết + chi phí:

| Bước | Script | Chi phí | Ghi chú |
|---|---|---|---|
| P3 Exp B | `phase3_expB.py` | ~5 phút/bridge (eval) + giây (bootstrap) | ra `outputs/expB/summary.json` → **điền top-3 vào `configs/action_space.yaml:bridges` bằng tay** |
| P1 leftover | `train --bridge <b> --n-tiles ...` với `tile_choices` | ~15h (3 bridge) | chỉ nếu chọn phương án 4a |
| P4 oracle sweep TRAIN | `src.cli.oracle --split train --subset 7500` | **~15-20 GPU-h** (67.5k generate) | ra `outputs/oracle/{table,labels}.parquet` |
| P4 f(I,Q) | `src.cli.build_fiq --splits train,val,test` | ~3h (chỉ InternViT) | verify ở v12 (val) |
| P4 oracle VAL | `src.cli.oracle --split val` | ~12h | cho policy a*-match accuracy |
| P4 policy | `src.cli.train_policy` (×3 nhánh: ours / rt_only / visual_only) | ~5 phút/nhánh | |
| P5 oracle TEST | `src.cli.oracle --split test` | **~12-15h** (50k generate) | |
| P5 ladder | `src.cli.eval_ladder` | giây | ra `outputs/eval/{ladder.csv, pareto.csv, behaviour.json}` |

**Tổng GPU-giờ tối thiểu (đường 5×1 seed): ~70-80h. Đầy đủ (5×3): ~130-150h.**

---

## 4. Thủ công / bán tự động (final-plan P5)

- **Bảng compute-efficiency**: FLOPs (từ `outputs/profile/pipeline_cost.json`) + latency + throughput + % trainable params + kích thước checkpoint. Ghép số có sẵn — viết 1 script nhỏ hoặc làm tay.
- **Human validation**: lấy mẫu 300–500 QA phân tầng theo `category`, 2 người chấm (a) độ đúng câu sinh, (b) nhãn `category` có hợp lý không. Báo cáo Cohen's κ + % đồng thuận với metric tự động.
- **Error analysis định lượng**: phân loại lỗi theo `category`, so Ours vs Fixed-best.

---

## 5. P6 — Viết bài (deadline **27/09/2026**)

- Related Work: bảng phân biệt với token-pruning / early-exit / mixture-of-resolution — **điểm khác: giám sát reasoning-type tường minh**
- Method 4.1–4.5 theo final-plan §5. Diagram: `InternViT (frozen) → Bridge (trainable) → Qwen2-0.5B (frozen)`
- Experiments 5.1–5.8 từ đầu ra `outputs/`
- Setup 5.1: bảng GPU-giờ theo hạng mục, phần cứng, seed, phiên bản lib
- **Nói thẳng**: `P(r|Q)` macro-F1 0.92 vì `category` do LLM sinh từ câu hỏi → "cognitive prior" ≈ "question-pattern prior" (nông nhưng hợp lệ vì so sánh là "nhãn tường minh vs tín hiệu nội tại")
- Nếu Exp B không fork → đổi tên bài sang *"Cognitive-Conditioned Visual Budget Allocation for Efficient Vietnamese VQA"*, trọng tâm Pareto Ours-vs-Fixed + cận Oracle
- Springer LNCS/LNAI 12–15 trang, EasyChair `aciids2027`, session Trust4NLP

---

## Đường tới hạn (critical path)

```
[chốt QĐ 1-4]
   → Exp A 5×1 seed  (~25h, bạn launch)
   → phase3_expB      → điền top-3 bridge
   → (tuỳ) tile-aug retrain top-3
   → phase4: oracle TRAIN + f(I,Q) + oracle VAL + policy  (~30-35h)
   → phase5: oracle TEST + ladder  (~15h)
   → compute table + human eval + error analysis  (thủ công)
   → P6 viết bài  → nộp 27/09
```

Nút thắt lớn nhất = **oracle sweep** (~40-50h generate). Nếu chỉ có Kaggle → cân nhắc thu nhỏ subset (5000 thay 7500) hoặc thuê GPU cho riêng bước này.
