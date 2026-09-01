# Final Plan — Paper 3 · ACIIDS 2027 (special session Trust4NLP)

> Tài liệu này là bản thực thi chính thức. Ngôn ngữ khẳng định: mỗi mục mô tả thiết kế **là gì** và **làm gì**, kèm đầu ra cụ thể và tiêu chí hoàn thành để agent chạy tuần tự.

---

## 1. Tóm tắt một trang

**Câu hỏi nghiên cứu.**
> Giám sát loại suy luận (reasoning-type) tường minh có cải thiện việc phân bổ tính toán thị giác trong vision-language model so với chỉ dùng tín hiệu nội tại của mô hình hay không.

Đây là câu hỏi nghiên cứu, không phải tuyên bố phát minh.

**Hệ thống.** `InternViT-300M (đóng băng) → Bridge (huấn luyện) → Qwen2-0.5B (đóng băng)` — nền tảng Vintern-1B-v3_5, bridge custom thay projector gốc. Một **router** nhìn ảnh + câu hỏi sinh hai tín hiệu: prior nhận thức `P(r|Q)` (giám sát bằng nhãn `category`) và trạng thái thị giác rẻ `f(I,Q)`. Một **policy** học ngoại tuyến từ oracle, nhận `(P(r|Q), f(I,Q), λ)` và chọn action phân bổ tính toán thị giác.

**Lever tính toán.** `action = (n_tiles, bridge)`:
- `n_tiles` — số lần forward InternViT trên các tile của ảnh; đây là chi phí thị giác chi phối toàn pipeline.
- `bridge` — chọn từ tập rút gọn sau Exp B.

**Dữ liệu.** AutoViVQA. Nhãn reasoning-type lấy từ `data/raw/texts/final_vqa_dataset.json` (60.000 QA, trường `category`). Metadata chất lượng lấy từ `evaluate_60k_data_balanced_preprocessed.csv`. Split 70/15/15 tự chia, nhóm theo ảnh, seed cố định.

**Đánh giá.** FLOPs, wall-clock latency thực đo, throughput, Pareto frontier rời rạc, cận trên Oracle, human validation + error analysis định lượng.

**Mục tiêu nộp.** Springer LNCS/LNAI 12–15 trang, EasyChair `aciids2027`, chọn session Trust4NLP khi submit. Nộp trước **27/09/2026**.

---

## 2. Công thức chặt chẽ (8 chữ ký, bám theo Paper 1 & 2)

1. Contribution list đánh số.
2. Công thức hình thức hóa cơ chế lõi (oracle utility-cost, policy objective).
3. Bảng thống kê dataset.
4. Setup chi tiết: hyperparameter + phần cứng + GPU-giờ.
5. Bảng kết quả nhiều baseline.
6. Ablation nhiều tầng.
7. Thống kê nhiều seed: mean ± std, 95% CI, paired bootstrap/permutation.
8. Human validation + error analysis định lượng.

---

## 3. Đóng góp

1. **Motivating experiment 3 bước (A/B/C)** — instrumentation chứng minh có dư địa phân bổ tính toán thị giác theo loại suy luận. Không phải đóng góp chính.
2. **Router nhìn ảnh + câu hỏi** — prior nhận thức `P(r|Q)` (chỉ dùng câu hỏi, giám sát nhãn `category`) kết hợp trạng thái thị giác rẻ `f(I,Q)`.
3. **Policy học qua offline oracle-guided policy learning** — sinh nhãn action tối ưu từ oracle utility-cost, huấn luyện một mạng policy phân bổ tính toán.
4. **Đánh giá compute-aware đầy đủ** — FLOPs, latency thực, throughput, Pareto frontier rời rạc, cận trên Oracle, trên cùng một GPU.

---

## 4. Dữ liệu

### 4.1 Nguồn

| File | Vai trò | Quy mô |
|---|---|---|
| `data/raw/texts/final_vqa_dataset.json` | Nhãn reasoning-type (`category`) + `reason_depth` cho mọi QA | 60.000 QA, 27.496 ảnh, 0 null |
| `data/raw/texts/evaluate_60k_data_balanced_preprocessed.csv` | Metadata chất lượng (etp/eip/idp/vqac, 76 cột) — nguồn feature cho `f(I,Q)` và phân tích | 37.077 QA, 19.411 ảnh |
| `data/raw/images/` | Ảnh COCO đã tiền xử lý | 20.715 file |

Join `final_vqa_dataset.json` ↔ CSV theo khóa `(img_id, question)` (chuẩn hóa strip). Khóa này phủ **100%** số cặp `(image_id, question)` duy nhất của CSV (36.804/36.804).

### 4.2 Nhãn reasoning-type (`category`)

Dùng `category` làm tín hiệu reasoning-type **duy nhất**. Taxonomy chốt 8 lớp nominal (không thứ tự khó–dễ):

| Mã | Tên (AutoViVQA) | English | Tỉ lệ |
|---|---|---|---|
| REL | Mối quan hệ | relational | 28.5% |
| REC | Xác định đối tượng/thuộc tính | recognition | 18.0% |
| SPA | Mô tả vị trí/không gian | spatial | 13.0% |
| CAU | Lý do/Nhân quả | causal | 12.4% |
| ACT | Mô tả hành động | action | 12.3% |
| CNT | Xác định số lượng | counting | 11.3% |
| CTX | Suy luận ngữ cảnh | context inference | 2.7% |
| YNO | Câu hỏi có/không | yes-no | 1.5% |

- Loại bỏ ~200 dòng nhãn rác (typo, giá trị `reason_depth` lọt vào ô `category`, lớp "so sánh" 93 dòng) và lớp **"Text-in-Image"** (7 dòng) — reasoning-type text-in-image không xuất hiện đủ trong dữ liệu và không nằm trong phạm vi paper.
- Huấn luyện `P(r|Q)` với **class-balanced loss** (trọng số nghịch tần suất). Báo cáo macro-F1 và per-class F1.
- Giữ ký hiệu tốc ký REL/REC/… trong bảng, luôn kèm chú thích "nominal, not ordinal".

### 4.3 `reason_depth`

`reason_depth` (Level 1–5) **không** là tín hiệu đầu vào của model và **không** là mục tiêu giám sát trong vòng này. Lý do: trùng ~80% với `category` (recognition ≈ luôn Level 1, causal/context ≈ không bao giờ Level 1), khung ordinal của nó không được đảm bảo, Level 5 chỉ 364 dòng.

`reason_depth` chỉ dùng cho: (a) biến phân tầng phụ khi chia split, (b) chiều phân tích trong Policy Behavior Analysis và error analysis.

### 4.4 Split

- Tự chia **70/15/15** thành TRAIN / VAL / TEST.
- **Nhóm theo `image_id`**: một ảnh chỉ thuộc đúng một split (77% ảnh có nhiều câu hỏi → bắt buộc để tránh leak qua context ảnh chung).
- **Phân tầng theo `category`**, cân phụ theo `reason_depth`.
- `seed = 42`, cố định. Sơ đồ chia viết thành script tái lập: `scripts/data/build_split.py`, xuất `data/splits/{train,val,test}.jsonl`.
- TRAIN dùng để: sinh nhãn oracle + huấn luyện policy + huấn luyện bridge. VAL dùng để: chọn hyperparameter, early-stopping, chọn λ vận hành, quyết định fork Exp B. TEST: đánh giá một lần cuối cho bảng kết quả chính.

### 4.5 Bảng thống kê dataset (Section paper)

Xuất `outputs/dataset_stats.json` + bảng LaTeX: số ảnh/câu hỏi/đáp án mỗi split, phân phối `category` mỗi split, độ dài câu hỏi/đáp án, phân phối `reason_depth`, độ phủ join nhãn.

---

## 5. Kiến trúc và không gian action

### 5.1 Pipeline

```
Ảnh (1..N tile, 448×448)
   │  n_tiles lần forward
   ▼
InternViT-300M  (đóng băng)          ← chi phí tính toán thị giác chi phối
   │  T · 256 patch token
   ▼
Bridge  (huấn luyện, thay projector gốc)
   │  k vision token  (k tùy bridge)
   ▼
Qwen2-0.5B  (đóng băng)  ──►  câu trả lời
```

Router chạy song song, rẻ:
```
Question ──► PhoBERT head ──► P(r|Q) ∈ Δ^8
(I, Q)   ──► f(I,Q): probe thị giác rẻ (đặc trưng gộp + vài đặc trưng metadata rẻ)
```

### 5.2 Không gian action — CHỐT

`action a = (n_tiles, bridge)`. Hai trục, cả hai đều là lever tính toán thực:

- `n_tiles ∈ {1, 3, 6}` — số lần forward InternViT. Trục chi phí thị giác chính. Giá trị lưới có thể dịch nhẹ (vd `{1, 4, 8}`) theo đường cong FLOPs đo ở P1, nhưng **trục này luôn nằm trong action space**.
- `bridge ∈` top-3 bridge chọn ở **P3**. Mỗi bridge có số vision-token `k` khác nhau (1 với pooled, 8–16 với patch/qformer) → cũng là một lever chi phí ở tầng LLM prefill + FLOPs của chính bridge.

Tổng `|action| = 3 × 3 = 9`.

P1 **không** phải cổng go/no-go. P1 chỉ hiệu chỉnh giá trị lưới `n_tiles` và quyết định trục wall-clock nào báo cáo (latency đơn mẫu hay throughput theo batch). Không có nhánh "viết lại Method".

### 5.3 Chi phí `C(a)` và chất lượng `M(a)`

- `C(a)` = **số lần forward InternViT** (`= n_tiles`), chuẩn hoá `[0,1]` theo `n_tiles / max(n_tiles)`. Deterministic, phản ánh đúng chi phí chi phối trên trục FLOPs — có dải động thực (~2× FLOPs giữa 1 và 6 tile do InternViT-300M lớn hơn Qwen2-0.5B ở phần prefill thị giác).
- Nếu P1 cho thấy trục `bridge` cũng đóng góp chi phí đáng kể, dùng `C(a) = (α · n_tiles/max + β · k/max)` với `α, β` đặt theo tỉ lệ FLOPs thực đo. Mặc định `β = 0` cho tới khi P1 chứng minh ngược lại.
- `M(a; x)` = **CIDEr** giữa câu sinh ra (greedy decoding) và 5 đáp án tham chiếu. Metric liên tục.
- Latency wall-clock đo được để **riêng** ở bảng hiệu quả tính toán (5.6), **không** đưa vào hàm mục tiêu oracle.

### 5.4 Oracle utility-cost

Với mỗi mẫu `x` và mỗi trọng số `λ`:

```
U(a; x, λ) = M(a; x) − λ · C(a)
a*(x, λ)   = argmax_a U(a; x, λ)
```

Lưới `λ ∈ {0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0}` (7 điểm). `M` và `C` cùng thang `[0,1]` nên `λ` có nghĩa nhất quán.

### 5.5 Policy

- Mạng: MLP nhỏ. Đầu vào `(P(r|Q) ∈ R^8, f(I,Q) ∈ R^d, λ ∈ R)`. Đầu ra: phân phối trên 9 action.
- Mục tiêu: cross-entropy với nhãn `a*(x, λ)` từ oracle.
- Một policy duy nhất, điều kiện hoá theo `λ` (không huấn luyện lại mỗi `λ`).

---

## 6. Kế hoạch thực thi theo phase

> Mỗi phase: **Mục tiêu · Việc · Đầu ra · Tiêu chí hoàn thành**. Agent chạy tuần tự P0 → P6, không nhảy phase.

### P0 — Môi trường và dữ liệu

**Mục tiêu.** Có môi trường train tái lập và dữ liệu sạch tại chỗ.

**Việc.**
1. Tạo môi trường theo `setup.sh` (conda `vlm-bridge`, `torch==2.2.2`, `transformers==4.38.2`). Trên Kaggle dùng `setup_kaggle.sh` + Add Data `nguynrichard/auto-vqabest`.
2. Bổ sung phụ thuộc vào `requirements.txt`: `sentence-transformers`, `underthesea` (hoặc `pyvi`) cho PhoBERT tokenization, `scikit-learn`, `scipy`, `sacrebleu`, `rouge-score`. Ghi chú METEOR cần Java + `metrics/meteor/meteor-1.5.jar`.
3. Chuyển `metrics/compute_score.py` và `metrics/evaluate_model.py` sang import theo package (`from metrics.bleu import Bleu` …) để chạy được từ mọi thư mục.
4. Chạy `python -m src.data.download_data` (hoặc mount trên Kaggle). Xác nhận `data/raw/images/` và `data/raw/texts/*`.
5. Viết `scripts/data/build_labeled_table.py`: join `final_vqa_dataset.json` ↔ CSV theo `(img_id, question)`, lọc nhãn rác + "Text-in-Image", xuất `data/labeled.parquet` (cột: `image_id, question, answers, category, reason_depth, <metadata chất lượng>`).

**Đầu ra.** Môi trường hoạt động; `data/labeled.parquet`; `requirements.txt` cập nhật; metrics import gọn.

**Tiêu chí hoàn thành.** `pytest test_training_pipeline.py` chạy được; `data/labeled.parquet` có ≥ 36.000 dòng, 0 null ở `category`, đúng 8 lớp; `python -m metrics.compute_score` import không lỗi từ repo root.

---

### P1 — Multi-tile pipeline và hiệu chỉnh lưới  ·  **profiling DONE (kernel v8)**

**Kết quả đo (P100-16GB, mini_qformer, 32 mẫu, `outputs/profile/pipeline_cost.json`):**

| n_tiles | InternViT GFLOPs | latency (ms) | throughput (img/s) |
|---|---|---|---|
| 1 | 362 | 229 | 6.0 |
| 2 | 724 | 374 | 3.3 |
| 4 | 1448 | 648 | 1.7 |
| 6 | 2172 | 922 | 1.15 |

Dải động 1→6: **FLOPs ×6.0, latency ×4.0, throughput ×5.2** — vượt xa ngưỡng 15%. → **`n_tiles` là lever chính, giữ `action = (n_tiles, bridge)`, `C(a) = n_tiles/6` (β=0).** Lưới `{1, 3, 6}` chốt trong `configs/action_space.yaml`. Trục wall-clock: latency đơn mẫu (dải ×4, dùng trực tiếp).

**Còn lại của P1** (chưa làm): nối multi-tile vào collator + `trainer` forward/generate (hiện `profile.py` chạy đường multi-tile độc lập); retrain bridge lever với tile-count augmentation.

**Mục tiêu.** Pipeline tiêu thụ được `n_tiles` tile thực; số đo FLOPs/latency hiệu chỉnh giá trị lưới và chọn trục wall-clock để báo cáo. Action space `(n_tiles, bridge)` đã chốt ở §5.2, P1 không thay đổi điều đó.

**Việc.**
1. `forward` và đường sinh câu (`_generate_answer_for_sample`, `_batch_generate_answers`) nhận `pixel_values` dạng `(B, T, 3, H, W)`, flatten `(B·T, 3, H, W)` qua InternViT, gom token lại theo ảnh thành `(B, T·256, 1024)`.
2. Chọn **một** bridge tiêu thụ chuỗi token dài biến thiên `T·256` làm lever thị giác chính (nhánh attention/qformer). Bridge này nhận toàn bộ patch token, không CLS-pool.
3. Huấn luyện bridge lever với **tile-count augmentation** (train ở hỗn hợp `T ∈ {1..6}`) để không bị domain shift khi test đa tile.
4. `scripts/measure/profile_pipeline.py`: đo FLOPs (fvcore hoặc `torch.profiler`) và wall-clock end-to-end ở `n_tiles ∈ {1, 2, 3, 4, 6}`, bridge cố định, trên GPU mục tiêu, ≥ 200 mẫu, greedy decoding.
5. Chốt lưới `n_tiles`: chọn 3 giá trị trải đều dải chi phí thực đo (mặc định `{1, 3, 6}`).

**Đầu ra.** `src/training/finetune_setup.py` + `trainer.py` hỗ trợ đa tile; checkpoint bridge lever; `outputs/profile/pipeline_cost.json`; lưới `n_tiles` chốt trong `configs/action_space.yaml`.

**Tiêu chí hoàn thành.** Bảng chi phí `n_tiles → {FLOPs, latency đơn mẫu, throughput theo batch}` hoàn chỉnh; lưới `n_tiles` (3 giá trị) chốt trong `configs/action_space.yaml`; hệ số `α, β` của `C(a)` chốt. Quy tắc chọn trục wall-clock: nếu dải động **latency đơn mẫu** giữa `n_tiles=1` và giá trị lớn nhất **≥ 15%** → báo cáo latency đơn mẫu là trục hiệu quả chính; nếu **< 15%** (vòng sinh autoregressive của Qwen2 che lấp) → báo cáo **throughput theo batch** là trục chính (nơi chi phí InternViT chiếm tỉ trọng lớn) và vẫn liệt kê latency đơn mẫu để tham khảo. Trong cả hai trường hợp, trục FLOPs (`C(a)`) luôn có dải động thực và là trục Pareto sơ cấp.

---

### P2 — Split và Exp A (bridge baselines)

**Mục tiêu.** Có split chính thức và bảng hiệu năng **5 bridge** trên split đó.

**5 bridge = một thang capacity** (không dùng `gated_fusion` — gần trùng `residual`, nhánh yếu nhất; đã có kết quả sơ bộ ở `plans/results-5bridge.md`):
`residual` (1 token) → `multi_token` (nhiều token) → `tile_attention` (patch self-attn) → `mini_qformer` (2 lớp transformer nhẹ) → `qformer` (4 lớp + fusion văn bản).

**Việc.**
1. `python -m src.data.split`: chia 70/15/15 nhóm theo ảnh, phân tầng `category`, `seed=42` → `data/splits/{train,val,test}.jsonl`.
2. Huấn luyện 5 bridge trên TRAIN của split mới (`python -m src.cli.train --bridge <b> --split-dir data/splits --seed <s>`), chọn checkpoint theo VAL. `n_tiles=1` cho Exp A để so sánh bridge công bằng.
3. Chạy **3 seed** (42/43/44) mỗi bridge → 15 lần train.
4. Đánh giá trên VAL: CIDEr, BLEU-4, ROUGE-L, Accuracy, F1, METEOR, WUPS.

**Đầu ra.** `data/splits/*`; `checkpoints/expA/<bridge>/seed<i>/`; `outputs/expA/results.json` (mean ± std mỗi bridge mỗi metric).

**Tiêu chí hoàn thành.** 5 bridge × 3 seed hoàn tất, không NaN; bảng Exp A đầy đủ mean ± std trên split chính thức.

---

### P3 — Exp B: bridge × category và quyết định fork

**Mục tiêu.** Xác định bridge nào mạnh ở loại suy luận nào; chốt top-3 bridge cho oracle.

**Việc.**
1. Load kết quả 5 bridge trên **VALIDATION** (không đụng TEST). Join `category`. Tính CIDEr/Acc/F1 trung bình mỗi bridge × mỗi `category` → heatmap `outputs/expB/heatmap.png`.
2. Với mỗi `category`: **paired bootstrap + permutation test** giữa bridge tốt nhất và tốt nhì. Đây là phép thử quyết định duy nhất.
3. Kendall's W và "số category mà top-1 thay đổi" chỉ là số mô tả để đọc, không phải ngưỡng quyết định.
4. Bốn cổng cho kết luận "có phân hoá theo category": (a) ổn định qua ≥ 3 seed, (b) paired bootstrap significant (p < 0.05, hiệu chỉnh đa so sánh), (c) hợp lý ngữ nghĩa, (d) lợi ích compute thực đo được.
5. Chốt **top-3 bridge** (theo CIDEr trung bình có trọng số theo phân phối category của TRAIN) làm trục `bridge` của action space.

**Đầu ra.** `outputs/expB/heatmap.png`; `outputs/expB/fork_tests.json`; danh sách top-3 bridge trong `configs/action_space.yaml`; bridge baseline chính được đánh dấu.

**Tiêu chí hoàn thành.** Bảng test cho cả 8 category; top-3 bridge chốt; kết luận fork (có/không phân hoá) phát biểu rõ kèm bằng chứng thống kê.

---

### P4 — Oracle sweep và huấn luyện policy

**Mục tiêu.** Sinh nhãn action tối ưu và huấn luyện một policy.

**Việc.**
1. Chọn **subset TRAIN 7.500 mẫu**, phân tầng theo `category`.
2. Với `|action| = 9` (`n_tiles` lưới P1 × top-3 bridge P3): chạy `generate()` **greedy, 1 seed** cho từng `(mẫu, action)` → bảng `M(a; x)`, `C(a)`. Ghi tổng GPU-giờ dự kiến vào `outputs/oracle/estimate.json` **trước khi chạy**, ước từ throughput thực đo ở P1.
3. Sinh `a*(x, λ)` cho 7 điểm `λ`.
4. Huấn luyện router:
   - `P(r|Q)`: head phân loại 8 lớp trên PhoBERT, chỉ dùng câu hỏi, giám sát `category`, class-balanced loss. Đánh giá macro-F1 trên VAL.
   - `f(I,Q)`: probe rẻ — gộp đặc trưng InternViT ở `n_tiles=1` + một tập nhỏ đặc trưng metadata rẻ tính được lúc suy luận (độ nhòe/che khuất/mật độ vật thể từ ảnh, độ dài câu hỏi). Không dùng nhãn `category`.
5. Huấn luyện **một** policy MLP: đầu vào `(P(r|Q), f(I,Q), λ)`, đầu ra action, CE với `a*`. Chọn hyperparameter + early-stopping trên VAL. **Không** sinh thêm nhãn oracle ở VAL.
6. Chọn `λ` vận hành để báo cáo, dựa trên VAL.

**Đầu ra.** `outputs/oracle/table.parquet` (M, C mỗi mẫu mỗi action); `outputs/oracle/labels.parquet` (`a*` mỗi λ); `checkpoints/router/`; `checkpoints/policy/`; `configs/lambda_operating.yaml`.

**Tiêu chí hoàn thành.** Bảng oracle đầy đủ 7.500 × 9; `P(r|Q)` macro-F1 báo cáo trên VAL; policy hội tụ, val-accuracy so với `a*` báo cáo; GPU-giờ thực so với ước tính ghi lại.

---

### P5 — Đánh giá đầy đủ

**Mục tiêu.** Toàn bộ bảng kết quả và hình cho paper, đánh giá TEST một lần.

**Việc.**
1. **Ablation ladder** (7 nhánh), tất cả trên TEST, ≥ 3 seed:
   | Nhánh | Mô tả |
   |---|---|
   | Fixed-budget sweep | action cố định tại từng điểm của lưới (nhiều mức) |
   | Random | chọn action ngẫu nhiên theo ngân sách trung bình khớp |
   | Visual-state-only | policy chỉ nhận `f(I,Q)` |
   | Reasoning-type-only | policy chỉ nhận `P(r|Q)` |
   | **Ours** | policy nhận cả hai |
   | Oracle-cognitive-prior | policy nhận `category` thật (không phải dự đoán) |
   | Oracle | `a*` biết `M(a;x)` |
2. **Pareto frontier**: vẽ `M` theo `C` (và theo latency thực) cho từng nhánh; frontier = empirical frontier trên các điểm rời rạc. `outputs/eval/pareto.png`.
3. **Bảng hiệu quả tính toán (5.6)**: FLOPs, wall-clock latency, throughput, % trainable params, cho mỗi cấu hình. Một GPU, đo nhất quán.
4. **Policy Behavior Analysis**: phân phối action theo `category` và theo `reason_depth`; tần suất policy khớp `a*`; đường cong action ~ `λ`.
5. **Human validation** (chữ ký #8): lấy mẫu 300–500 QA phân tầng theo `category`, 2 người đánh giá độ đúng của câu sinh + tính hợp lệ nhãn `category`; báo cáo agreement (Cohen's κ) và tỉ lệ đồng thuận với metric tự động.
6. **Error analysis định lượng**: phân loại lỗi theo `category`, so sánh Ours vs Fixed-best, bảng tần suất.

**Đầu ra.** `outputs/eval/` đầy đủ: `ablation.json`, `pareto.png`, `compute_table.json`, `policy_behavior.json`, `human_eval.json`, `error_analysis.json`.

**Tiêu chí hoàn thành.** Mọi bảng/hình trong Section 5 sẵn sàng; TEST chỉ chạy một lần; mọi con số có mean ± std + 95% CI; mọi so sánh chính có paired bootstrap.

---

### P6 — Viết bài và nộp

**Việc.**
1. Related Work: bảng phân biệt với các router thị giác thích ứng (token pruning, early-exit, mixture-of-resolution), nêu rõ điểm khác: giám sát reasoning-type tường minh.
2. Method 4.1–4.5 theo Section 5 của plan này; diagram pipeline vẽ đúng `InternViT (frozen) → Bridge (trainable) → Qwen2-0.5B (frozen)`.
3. Experiments 5.1–5.8 từ đầu ra P5.
4. Setup 5.1: ghi tổng GPU-giờ theo hạng mục, phần cứng, seed, phiên bản thư viện.
5. Format Springer LNCS/LNAI, 12–15 trang. Trust4NLP là special session của ACIIDS 2027: cùng deadline, cùng EasyChair `aciids2027`, chọn session khi submit.
6. Nộp trước **27/09/2026**.

**Tiêu chí hoàn thành.** Bản PDF ≤ 15 trang, đủ 8 chữ ký chặt chẽ, nộp EasyChair có xác nhận.

---

## 7. Cấu trúc code đích

```
src/
├── modeling/
│   ├── bridge_modules.py         # 5 bridge của suite + GatedFusionBridge (không dùng)
│   ├── router.py                 # P(r|Q) head (PhoBERT) + f(I,Q) probe
│   └── policy.py                 # policy MLP: (P(r|Q), f(I,Q), λ) → action
├── training/
│   ├── finetune_setup.py         # hỗ trợ đa tile (P1)
│   └── trainer.py                # đa tile ở forward + generate (P1)
scripts/
├── data/
│   ├── build_labeled_table.py    # P0: join + làm sạch nhãn
│   └── build_split.py            # P2: split 70/15/15 nhóm theo ảnh
├── measure/
│   └── profile_pipeline.py       # P1: FLOPs + latency theo n_tiles
├── expA_bridge_baselines.py      # P2
├── expB_bridge_x_category.py     # P3
├── oracle_sweep.py               # P4
├── train_router.py               # P4
├── train_policy.py               # P4
└── eval_full.py                  # P5: ablation ladder + Pareto + compute table
configs/
├── action_space.yaml             # n_tiles grid + top-3 bridge
├── lambda_operating.yaml         # λ vận hành chốt trên VAL
data/
├── labeled.parquet
└── splits/{train,val,test}.jsonl
outputs/
├── dataset_stats.json
├── profile/pipeline_cost.json
├── expA/  expB/  oracle/  eval/
```

---

## 8. Ngân sách compute

Ước lần cuối theo throughput thực đo ở P1. Bảng dự kiến (điền số thực trước khi chạy từng phase):

| Hạng mục | Quy mô | GPU-giờ dự kiến |
|---|---|---|
| P1 profiling | 5 mức × 200 mẫu | thấp |
| P1 bridge lever + tile augmentation | 1 bridge | — |
| P2 Exp A | 5 bridge × 3 seed × 10 epoch | — |
| P3 Exp B eval | inference trên VAL | thấp |
| P4 oracle sweep | 7.500 × 9 generate × 1 seed | — |
| P4 router + policy | nhẹ | thấp |
| P5 ablation TEST | 7 nhánh × 3 seed | — |

Ràng buộc thực thi trên Kaggle 16 GB: VRAM dư (bridge fine-tune đỉnh ~4–11 GB). Nút thắt là **thời gian và quota** (session 12h, ~30h GPU/tuần). Bắt buộc checkpoint + `resume_from` xuyên session. Cân nhắc chuyển P4 (oracle sweep) và các lượt ≥ 3 seed sang GPU thuê nếu quota Kaggle không đủ. Đo nhất quán trên **một** GPU cho mọi con số so sánh; không PR phần cứng.

---

## 9. Thống kê và tính chặt chẽ

- So sánh chính: **paired bootstrap** (10.000 lần resample) + **permutation test**. Báo cáo hiệu số + 95% CI, không dùng CI-overlap.
- ≥ 3 seed cho: quyết định fork (P3) và bảng TEST cuối (P5). Oracle sweep (P4) 1 seed — mục đích là sinh nhãn huấn luyện, không phải báo cáo kết quả.
- Hiệu chỉnh đa so sánh (Holm) khi test trên 8 category.
- Pareto = empirical frontier trên tập điểm rời rạc, nêu rõ trong caption.
- Decoding **greedy cố định** cho mọi lượt sinh phục vụ metric và oracle.
- Mọi ngưỡng số ngoài paired bootstrap chỉ là mô tả, không phải tiêu chí quyết định.

---

## 10. Rủi ro và phương án

| Rủi ro | Phương án |
|---|---|
| Latency đơn mẫu của `n_tiles` bị vòng sinh Qwen2 che lấp (P1 < 15%) | Trục FLOPs vẫn có dải động ~2× → Pareto sơ cấp dùng FLOPs; trục wall-clock chuyển sang throughput theo batch. Action space `(n_tiles, bridge)` không đổi. Đã định nghĩa sẵn ở P1. |
| Cả hai trục action đều yếu (FLOPs range < 1.3×) | Mở rộng lưới `n_tiles` lên `{1, 6, 12}` (InternViT hỗ trợ tới 12 tile) để kéo dải chi phí; nếu vẫn yếu, thêm trục thứ 3 là độ phân giải tile (448 vs 224). |
| `P(r|Q)` macro-F1 thấp do lệch lớp | Gộp CTX+YNO vào lớp lân cận; báo cáo cả cấu hình 6 lớp. `reason_depth` (4 bucket) làm mục tiêu bổ trợ nếu cần. |
| `category` và `reason_depth` trùng nhau → "reasoning-type" chỉ là proxy của độ khó | Nêu thẳng sự trùng lặp trong Method; thesis vẫn hợp lệ vì so sánh là "nhãn tường minh vs tín hiệu nội tại", không phải "type vs difficulty". |
| Fork Exp B không có phân hoá | Paper vẫn đứng: kết quả âm tính có giá trị, chuyển trọng tâm sang Pareto Ours-vs-Fixed và cận Oracle. Đổi tên bài sang *"Cognitive-Conditioned Visual Budget Allocation for Efficient Vietnamese VQA"*. |
| Quota Kaggle không đủ | Giảm subset oracle xuống 5.000; giảm seed phụ; thuê GPU cho P4–P5. |
| Nhãn `category` auto-gen sai nhiều | Human validation P5 đo tỉ lệ sai; nếu > 15%, lọc theo độ tin cậy join + báo cáo trên tập đã lọc. |

---

## 11. Điểm mấu chốt để "nhìn perfect"

- Thesis là câu hỏi nghiên cứu, không phải claim phát minh.
- 5-bridge là instrumentation, không phải đóng góp chính.
- Action space `(n_tiles, bridge)` đã chốt. P1 chỉ hiệu chỉnh giá trị lưới và chọn trục wall-clock báo cáo — không phải cổng go/no-go.
- Nhãn reasoning-type: `category` từ `final_vqa_dataset.json`, 8 lớp nominal, phủ ~100% split.
- `reason_depth` không vào model; chỉ stratify + phân tích.
- Split 70/15/15 nhóm theo ảnh, phân tầng `category`, seed 42.
- `C(a)` deterministic + chuẩn hoá; latency thực đo riêng, không vào oracle objective.
- Fork/gate: paired bootstrap là quyết định duy nhất.
- Một GPU, đo nhất quán, không PR phần cứng.
- Nộp trước 27/09/2026, EasyChair `aciids2027`, session Trust4NLP.
