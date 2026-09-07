# SESSION STATE — Paper 3 (snapshot 2026-09-07 11:00 UTC)

> Bản chốt trạng thái để không mất context khi compact. Số CANONICAL. Nếu số ở
> file khác lệch → tin file này + `results-grouped-split.md` (recompute mới nhất).

---

## 0. TL;DR trạng thái

- **Thực nghiệm: XONG** (còn 9 job đang chạy = re-run + probe, không đổi kết luận).
- **Số đã tính lại toàn bộ theo 2-epoch, 3-seed** (multi_token = 4 seed).
- **Peer đã restructure §5–§7** (05-results.md) + bootstrap CI trên số 2ep.
- **§1–4 ĐÃ VIẾT XONG** (English, framing mới; commits 27dc49b/c6a8d6f/49aafc1/abcefd2). Chờ peer consistency pass.
- **Còn lại: English consistency pass toàn draft (peer), 3 hình, human validation (user), §7 Conclusion (chưa).**
- Branch: `chore/repo-restructure` (docs + plain expa), `feat/decoder-lora` (LoRA + run.py patches). Ledger sync 2 branch.

---

## 1. SỐ CANONICAL (val, 2 epoch, 3-seed; multi_token = 4 seed; ×100)

### 1a. Recipe vs baseline (in-house)
| Model | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Vintern-1B base | 0.12 | 17.52 | 19.87 | 17.55 | 1.91 | 25.84 | 23.93 | 8.54 |
| Vintern-1B fine-tuned | 13.01 | 52.47 | 55.12 | 53.76 | 6.11 | 51.93 | 35.25 | 72.84 |
| GPT-5 zero-shot | 10.84 | 47.20 | 55.20 | 50.89 | 6.07 | 47.30 | 33.34 | 84.20 |
| ViMoE-VQA (5 seed) | 9.65 | 62.89 | 58.65 | 60.69 | 12.54 | 47.07 | 39.10 | 88.67 |
| **Multi-Token bridge (0.78%, 1 tile)** | 8.20 | 50.36 | 51.43 | **49.55** | 15.47 | 47.84 | 40.22 | 96.49 |
| **+ LoRA r=16 attn (1 epoch)** ‡ | 10.93 | 54.39 | 55.11 | **53.52** | 19.72 | 51.81 | 44.11 | 106.56 |
| **+ LoRA r=16 attn (3 epoch)** ‡ | 12.00 | 55.46 | 56.38 | **54.71** | 21.07 | 52.96 | 45.42 | 110.49 |

‡ **CẬP NHẬT 2026-09-07 ~15:30 UTC**: LoRA re-run sạch (v1 ckpt hỏng), 3-seed
@ full-val 5463. F1 std: 1ep 0.11, 3ep 0.14. Cũ (seed-42, đã thay): 1ep 53.17 / 3ep 54.67.

*Baseline khác: ViT5_ViT F1 48.52, BARTPhoBEiT F1 45.88 (CIDEr 188.96 = outlier bỏ),
Llama3.2 36.16, Gemini2.0 39.79, Gemini2.5 24.75. Full ở paper-blueprint.md Bảng 1.*

### 1b. Corpus (pycocoevalcap, để so cross-paper)
| Model | CIDEr-D | BLEU-4 | ROUGE-L |
|---|--:|--:|--:|
| ViMoE-VQA | 88.67 | 12.54 | 47.07 |
| Multi-Token bridge (4 seed, 2ep) — VAL | 92.3 ± 0.6 | 18.9 ± 0.3 | 48.9 ± 0.1 |
| + LoRA r=16 1ep (3 seed) — VAL | 103.2 ± 0.5 | 23.6 ± 0.1 | 53.0 ± 0.1 |
| + LoRA r=16 3ep (3 seed) — VAL | 107.5 ± 0.3 | 25.1 ± 0.2 | 54.2 ± 0.1 |
| + LoRA r=16 1ep (3 seed) — **TEST** | 101.3 ± 0.5 | 22.9 | 52.6 |
| + LoRA r=16 3ep (3 seed) — **TEST** | 104.8 ± 0.8 | 24.9 | 53.8 |
| bootstrap 95% CI (multi_token plain CIDEr-D) | **[89.9, 95.3]** — trên hẳn ViMoE 88.7 | | |

✅ **2026-09-07 16:50 UTC**: VAL corpus LoRA refresh XONG — từ 6 checkpoint re-run
sạch (6 job `lora-val-eval`, text-pred full-val 5463). Số cũ (1ep 101.7 / 3ep
106.8) từ run 3-seed đã mất → thay bằng số trên. Val predictions ở
`checkpoints/expA-lora16{,-3ep}/seed*/multi_token/results/text_predictions_val_epoch_1.json`
→ đưa peer rerun bootstrap. TEST corpus = rescore epoch_1 (test 5468).

### 1c. TEST split — đối chiếu val
| | val | test | Δ |
|---|--:|--:|--:|
| Multi-Token bridge F1 (n=5468, 4 seed) | 49.55 | **49.20** | −0.35 |
| Multi-Token bridge CIDEr(ih) | 96.49 | **93.24** | −3.25 |
| mini_qformer F1 (s42) | 47.05 | 47.25 | +0.20 |
| residual F1 (s42) | 45.91 | 45.49 | −0.42 |
| tile_attention F1 (s42) | 44.50 | 44.44 | −0.06 |
| qformer F1 (s3407) | 47.13 | **46.62** | −0.51 |
| **+ LoRA 1ep F1 (3 seed)** | 53.52 | **53.15** | −0.37 |
| **+ LoRA 1ep CIDEr(ih)** | 106.56 | **104.65** | −1.91 |
| **+ LoRA 3ep F1 (3 seed)** | 54.71 | **54.28** | −0.43 |
| **+ LoRA 3ep CIDEr(ih)** | 110.49 | **107.60** | −2.89 |

→ gap < 0.5 F1 mọi cấu hình, không nhất quán chiều → **KHÔNG overfit val**. Recipe
giữ vững trên test. (LoRA test đo trên cùng 6 ckpt sạch re-run 2026-09-07.)

### 1d. 5 bridge plain @ 2ep 3-seed + LoRA
| Bridge | Params | F1 plain | CIDEr-D plain | val CE | F1 +LoRA | ΔF1 | CIDEr-D +LoRA |
|---|---|--:|--:|--:|--:|--:|--:|
| multi_token | 7.35M (0.78%) | 49.55 ± 0.07 | 92.3 ± 0.6 | 1.49 | 53.52 | +4.0 | 103.2 |
| qformer (Full Q-Former) | 69.4M (6.91%) | 47.35 ± 0.17 | 85.4 ± 0.5 ‡ | 1.57 | 53.21 | +5.9 | 102.4 |
| mini_qformer (Light Q-Former) | 27.6M (2.87%) | 46.25 ± 0.62 | 81.7 ± 1.7 ‡ | 1.60 | 53.21 | +7.0 | 103.0 |
| residual | 4.86M (0.52%) | 45.64 ± 0.36 | 81.1 ± 0.6 | 1.67 | 52.64 | +7.0 | 100.8 |
| tile_attention | 4.14M (0.44%) | 45.17 ± 0.94 | 79.0 ± 2.1 | 1.67 | 52.99† | +7.8 | 102.0 |

† tile_attention +LoRA = seed 42 only.
§ multi_token +LoRA F1 = **re-run sạch 3-seed** (53.52, ΔF1 +4.0). 4 bridge kia
(qformer/mini_qf/residual/tile_attn +LoRA) = 3-seed sạch sẵn có, **KHÔNG đổi**
(53.21/53.21/52.64/52.99). Băng F1 sau LoRA: 52.6–53.5 (trước: 45.2–49.6). Kết
luận "LoRA san bằng mọi bridge" giữ nguyên. multi_token +LoRA VAL corpus (1ep 103.2 / 3ep 107.5) đã refresh từ 6 ckpt sạch.
‡ **SỬA 2026-09-07 (audit)**: qformer/mini_qformer CIDEr-D cũ (86.9 ± 2.3 / 83.7 ± 4.4)
là **giá trị sơ bộ** — tính khi chưa đủ 3 seed corpus. Nay đủ cả 3 file
`*_epoch_1_corpus.json`: qformer 85.4 ± 0.5 (BLEU-4 16.7, ROUGE-L 46.7),
mini_qformer 81.7 ± 1.7 (BLEU-4 16.3, ROUGE-L 45.5). **F1 không đổi.** qformer/mini_qf
+LoRA CIDEr-D (102.4 / 103.0) cũng cần verify lại từ corpus json khi rảnh.
Nguyên nhân: local `checkpoints/expA/seed42/qformer/` từng là **mix cũ 4ep** (summary.json
+ file dự đoán 4ep) — đã thay bằng bản 2ep sạch pull từ acc14 (F1 47.56, đúng số cũ đã dùng).

**Bootstrap CIs (clean 5-bridge re-run: 2ep plain vs 3-seed-clean LoRA val preds, seed-42 basis, `outputs/bootstrap_ci.json`, commit e06f824):**
| Bridge | ΔF1 [95% CI] | ΔCIDEr-D [95% CI] | P(Δ>0) |
|---|---|---|---|
| multi_token | +4.06 [3.49, 4.65] | +10.75 [8.83, 12.60] | 1.000 |
| qformer | +5.54 [4.92, 6.19] | +16.51 [14.36, 18.64] | 1.000 |
| mini_qformer | +6.75 [6.07, 7.43] | +19.53 [17.36, 21.89] | 1.000 |
| residual | +6.75 [6.07, 7.39] | +18.40 [16.24, 20.98] | 1.000 |
| tile_attention | +8.49 [7.80, 9.16] | +24.00 [21.86, 26.54] | 1.000 |

multi_token-plain CIDEr-D 92.5, 95% CI **[89.9, 95.3]** (one-sample; entirely above ViMoE 88.7).
F1 CI [48.9, 50.3] vs ViMoE 60.7. No paired test vs ViMoE (no per-sample data published).

**⚠️ residual: số cũ F1 36.45 / CIDEr-D 56.3 / val CE 2.35 là LẦN CHẠY SEED-42 HỎNG
(training instability). Số thật 45.64. "ΔF1 +16.2 từ bridge tệ nhất" ĐÃ BỎ.**

### 1e. Ablation 6-RQ (anchor = multi_token 49.55)
| RQ · trục | Can thiệp | F1 | ΔF1 | Verdict |
|---|---|--:|--:|---|
| RQ1–2 bridge capacity | Full Q-Former (69M, 10×) | 47.35 | −2.20 | âm |
| RQ3 tile | train 1 tile → eval 3 tile | 21.05 | −28.5 | âm (sụp; val loss 1.48→3.35) |
| RQ4 routing | learned policy theo loại câu hỏi | ≈50.7 | ≈0 | âm |
| RQ5 training signal | multi-reference answer sampling | 48.08 | −1.47 | âm |
| RQ5 alignment | projector feature-KD (align-feat) | 49.53 | **−0.03** | **âm — NULL TUYỆT ĐỐI** |
| RQ5 alignment | projector logit-KD α=0.1 (align-logit-a01) | 49.75 ± 0.29 | **+0.20** | **âm — NULL** (val CE 1.53, ≈ plain) |
| RQ5 alignment | projector logit-KD α=1.0 (align-logit) | 40.75 | −8.80 | âm (KL lấn CE, val CE ~2.05 — CHỈ do α quá lớn) |
| **RQ6 decoder** | **LoRA r=16 attn (1ep)** | **53.52** | **+4.0** | **DƯƠNG** |
| **RQ6 decoder** | **LoRA r=16 attn (3ep)** | **54.71** | **+5.2** | **DƯƠNG** |
| RQ6 decoder | LoRA r=16 MLP-only (gate/up/down) | 20.24 ± 1.52 | −29 | 💥 phân kỳ (val loss ~3.7) |
| RQ6 decoder | LoRA r=16 attn+MLP (cả 7) | 37.51 ± 1.70 | −12 | 💥 phân kỳ (val loss ~2.08) |

→ **Dư địa decoder nằm CỤ THỂ ở attention.** MLP LoRA phân kỳ (caveat: có thể HP
artifact, α=32 mạnh cho MLP dim ~4864 vs attn 896 — claim giới hạn ở cấu hình recipe).

### 1f. Đường cong epoch LoRA (multi_token attn, 3-seed)
| epoch | F1 | CIDEr(ih) | CIDEr-D |
|--:|--:|--:|--:|
| 1 | 53.52 | 106.56 | 103.2 |
| 3 | 54.71 | 110.49 | 107.5 |
| 5 | job bị cắt ở cap quota (~4ep, best_model.pt lưu ở epoch 1 — vô ích) |

### 1g. Chi phí tính toán tile (InternViT/ảnh, P100-16GB)
| tile | GFLOPs | latency ms | throughput img/s |
|--:|--:|--:|--:|
| 1 (ours) | 362 | 229 | 6.00 |
| 2 | 724 | 374 | 3.30 |
| 4 | 1448 | 648 | 1.70 |
| 6 | 2172 | 922 | 1.15 |

### 1h. Self-check (thay human validation tạm, N=120, 1 rater)
strong (≥0.6 F1): 91.1% chấp nhận được. **partial (0.2–0.6, bucket LỚN NHẤT 51.5%
val): chỉ 43.1%** — token-F1 tầm trung KHÔNG đáng tin. overall 57.1% acceptable.

### 1i. Param counts
Residual 4.86M/0.52% · Multi-Token 7.35M/0.78% · Tile-Attention 4.14M/0.44% ·
Light Q-Former 27.57M/2.87% · Full Q-Former 69.39M/6.91% · LoRA r=16 q/k/v/o
2.16M/0.23% · **Multi-Token + LoRA = 9.51M/1.01%**.

---

## 2. JOBS ĐANG CHẠY (check 2026-09-07 11:00 UTC — tất cả 9 job = RUNNING) — sẽ land trong ngày

| Job | Account | Kernel slug (feat/decoder-lora) | Pushed (UTC) | Cho ra | Xử lý khi land |
|---|---|---|---|---|---|
| LoRA 3ep re-run ×s42/123/3407 | acc14/acc7/acc3 | `<user>/mvlm-expa-lora16-multi-token-s<seed>` | ~04:00 (ETA ~12:00) | ckpt sạch + **val + test** cho recipe | pull → eval_test.json → §1c + Bảng 1 blueprint (3ep test row) |
| LoRA 1ep re-run ×s42/123/3407 | acc2/acc8/acc13 | `<user>/mvlm-expa-lora16-multi-token-s<seed>` | ~08:14 (ETA ~11:00) | ckpt sạch (v1 hỏng) + val + test | pull → pull → 1ep val 53.52 (3-seed) + test |
| align-logit α=0.1 ×s42/123 | acc12 (×2) | `kffddk/mvlm-expa-align-logit-a01-multi-token-s<seed>` | ~08:14 (ETA ~13:00) | RQ5: KL trọng số nhẹ có giúp F1? | pull → nếu vẫn âm → §1e bỏ caveat "sai trọng số" |
| align-logit α=0.1 ×s3407 | acc4 | `nguyennn251/mvlm-expa-align-logit-a01-multi-token-s3407` | ~08:14 | ⚠️ acc4 quota thấp, có thể bị cắt → 2/3 seed | chấp nhận 2 seed nếu 2 seed kia nhất quán |
| ~~qformer test-eval s3407~~ | acc10 | `giapht/mvlm-test-eval-qf-s3407` | — | **ERROR** (bỏ — mt 4-seed + 3 bridge đủ cho "test≈val") | không cần |

**Lưu ý giờ:** ledger `pushed_at` = giờ máy (UTC+7), không phải UTC. Trừ 7h.

**Cách pull LoRA:** kernel slug `<user>/mvlm-expa-lora16-multi-token-s<seed>` (feat branch).
eval_test.json ở `ck-lora/seed<N>/multi_token/`. Corpus rescore: **dùng
`text_predictions_epoch_1.json` (full-val 5463), KHÔNG epoch_2 (600-subset!).**

---

## 3. ĐÃ XONG / ĐÃ COMMIT

- `results-grouped-split.md` — §0/§1/§3/§4b/§4d/§4e recompute (commit 9bd2dd0, 6a8df84...)
- `results-5bridge.md` — Bảng 1/2 recompute (5559a51)
- `paper-blueprint.md` + artifact https://claude.ai/code/artifact/fe068b4c-d59c-429f-bdba-ed9ea93bd557 (02c44bd) — full 2ep 3-seed + TIER-2 Bảng 5b + test A1
- `paper-status-for-advisor.md` — full recompute (bcaabab)
- `P6-draft/06-discussion.md` — attention-localization + residual caveat + §5→§6 refs (0b288be)
- `P6-draft/05.1-bridge-baseline.md` — SUPERSEDED banner
- **Peer:** `P6-draft/05-results.md` — §5 Main Results / §6 Ablation 6-RQ / §7 Human Validation restructure (bfed6f2) + bootstrap CI (f2dd5d0)
- `scripts/bootstrap_ci.py` (peer), `scripts/parallel/eval_test.py` (test-eval infra),
  `scripts/parallel/run.py` feat branch: `--lora-targets`, `--align-weight`, worker eval test split
- Checkpoints staged local: `checkpoints/expA/seed{42,123,2026,3407}/<bridge>/`,
  `checkpoints/expA-4ep/seed42/` (backup 4ep), `checkpoints/expA-lora16{,-mlp,-all}/`,
  `checkpoints/expA-align-{feat,logit}/`. (.pt của LoRA-1ep s123/s3407 HỎNG — đang re-run.)
- `outputs/test_eval/` — 7 eval_test.json + summary.json
- **`paper/` LaTeX skeleton (assistant)** — llncs, §1–9 + 3 figs + 7 tables, density-passed 21→16pp.
  Number sync tới basis 53.52 + bootstrap CIs clean 5-bridge: commit **e06f824**. Builds clean, 16pp, no undefined refs.
  Còn: overfull hbox 1 dòng ở `03-method.tex` L36–47 (peer-owned); real `.bib`; 2-rater human validation.

---

## 4. CÒN LÀM (không cần Kaggle)

| # | Việc | Ai | Trạng thái |
|---|---|---|---|
| 1–4 | **§1 Intro / §2 Related Work / §3 Method / §4 Setup** — viết lại English, framing mới | mình | ✅ XONG (27dc49b / c6a8d6f / 49aafc1 / abcefd2). Cần verify Vintern-FT recipe ở §4 AutoViVQA. |
| 5 | **§7 Conclusion** — viết ngắn | mình | ⬜ chưa (peer's 05-results có §7 = Human Validation; conclusion riêng = 07-conclusion.md) |
| 6 | **English consistency pass toàn draft §1–7** | peer | ⬜ chờ — §1–4 đã land, ping peer |
| 7 | **3 hình matplotlib** | mình/peer | ⬜ (a) bridge-equalizing bar (79–92 → 100.8–103) (b) tile-collapse F1+loss vs n_tiles (c) sơ đồ kiến trúc |
| 8 | **Human validation THẬT** | **user + 1 người** | 🟡 TOOLING XONG (commit sau 9bd5d46): `scripts/human_validation_build.py` → 300 mẫu (plain multi_token s42, stratified theo category), 2 form trắng `outputs/human_validation/annotation_form_{A,B}.csv` + `answer_key.json` (token-F1 ẩn) + `ANNOTATION_README.md` (rubric 4 mức). User + 1 người annotate → `scripts/human_validation_kappa.py` cho Cohen's κ + cross-tab F1-bucket. Còn chờ: 2 người annotate. |
| 9 | (optional, sau Fri reset) | — | LoRA 5ep sạch (>12h cap, cần resume) · TIER-2 MLP retune HP · align-feat 5-seed |

---

## 5. QUYẾT ĐỊNH / GOTCHA quan trọng

1. **Chuẩn 2 epoch** cho mọi bridge plain (default run.py; CIDEr bão hòa từ ep2).
   Số cũ trộn seed-42-4ep + seeds-2ep. Đã re-run seed 42 @ 2ep cho 5 bridge + dòng âm.
   4ep backup: `checkpoints/expA-4ep/`.
2. **residual seed-42 4ep = bad run** (val CE 2.35). ĐỪNG cite từ đó. Số thật 45.64.
3. **Corpus rescore dùng `text_predictions_epoch_1.json`** (full-val 5463), KHÔNG
   `epoch_2` (600-subset mid-training). Đã dính lỗi này 1 lần.
4. **LoRA test-eval**: 1ep/3ep .pt local hỏng/mất → đang re-run với worker eval test.
   LoRA-3ep .pt cũ MẤT HẲN (kernel s42 bị 5ep ghi đè).
5. **Kaggle `kaggle quota`** cho quota/account — LUÔN check trước batch. acc12 mới
   thêm lại 2026-09-07 (username kffddk, 30h). acc9/acc11/acc16 gần cạn. acc15 vắng.
6. **KHÔNG cancel job launch nhầm account** (user chỉ đạo) — chọn account theo quota
   TRƯỚC khi launch. Push no-op để cancel thì bản cũ có thể vẫn ngốn quota.
7. Bootstrap CI cũ (bootstrap_ci.json) dựa trên số 4ep + residual bad run — peer đã
   regen trên 2ep (f2dd5d0). CIDEr-D CI [89.9, 95.3] (chặt hơn [91.3, 97.1] cũ).
8. **Peer session** = `repo-restructure-modular-vlm` (uds:/tmp/cc-socks/7393.sock).
   Phân công: mình §1–4 + blueprint tables; peer §5–7 prose + bootstrap.
9. Deadline ~2026-09-27 (ACIIDS 2027 / Trust4NLP). LNCS 12–15 trang.

---

## 6. FILE MAP

| Nội dung | File |
|---|---|
| Bản thiết kế + mọi bảng | `plans/paper-blueprint.md` + artifact fe068b4c |
| Ablation working log (chi tiết nhất) | `plans/results-grouped-split.md` |
| Main Results canonical | `plans/results-5bridge.md` |
| Báo cáo cho thầy (gọn) | `plans/paper-status-for-advisor.md` |
| Draft paper | `plans/P6-draft/0{1..7}*.md` (05 = peer's Results/Ablation/HumanVal) |
| State này | `plans/SESSION-STATE.md` |
| Job ledger | `outputs/parallel/ledger.json` (sync 2 branch) |
| Kế hoạch tổng | `plans/paper-completion-plan.md`, `plans/final-plan.md` |
