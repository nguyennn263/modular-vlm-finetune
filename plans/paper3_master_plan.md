# Master Plan — Paper 3 cho ACIIDS 2027 (bản chốt sau 7 vòng review)

> ⚠️ Round 7 là loại review KHÁC hẳn round 1-6: không phải polish methodology/writing nữa, mà là **audit khả thi kỹ thuật** — kiểm tra xem plan (đã rất chặt trên giấy) có thực sự build được không. Kết quả: Phần A-E (methodology) được duyệt làm định hướng, nhưng **CHẶN CODE cho tới khi xong Phần G** (gate kỹ thuật, làm trước Phần D bước 1). Đây là lần đầu có người tính chi phí GPU thật của oracle sweep — và con số đó không khả thi như plan cũ mô tả.

## Phần A — "Công thức rigor" từ Paper 1 & 2 (không đổi)

8 chữ ký: (1) contribution list đánh số, (2) công thức hình thức hóa cơ chế lõi, (3) bảng thống kê dataset, (4) setup chi tiết hyperparameter+phần cứng, (5) bảng kết quả nhiều baseline, (6) ablation nhiều tầng, (7) thống kê nhiều seed (mean±std, CI), (8) human validation + error analysis định lượng.

## Phần B — Lịch sử sửa qua các vòng

| Vòng | Vấn đề chính đã sửa |
|---|---|
| 1-2 | Bỏ bảng hard-code level→action; chuyển sang policy học từ oracle. Router nhìn Image+Question. Bỏ trục "reasoning budget/CoT". |
| 3 | Xác nhận ResAdapt thật (RL/bandit). Tách reasoning-type khỏi difficulty. Bridge×Level matrix làm fork, có 4 gate. |
| 4-5 | "Difficulty"→"uncertainty". Oracle dùng metric liên tục M(a), không phải Accuracy 0/1. Paired bootstrap thay CI-overlap. Pareto = "empirical frontier over discrete points". |
| 6 | Bỏ hẳn Head 2 (ΔM regression) — ΔM không phải intrinsic property của sample. Tách TRAIN(oracle+train policy)/VAL(fork+hyperparam)/TEST(1 lần). Thêm baseline Oracle-cognitive-prior + Fixed-budget sweep. |
| **7 (round này)** | **Audit khả thi:** (a) oracle sweep như mô tả tốn ~270-900 GPU-giờ, không khả thi → phải thu gọn action grid + subset sample; (b) "N_tile" có thể là compute lever yếu (bridge pool về vài token cố định, chi phí thật nằm ở ViT) → phải đo trước khi tin; (c) AutoViVQA chỉ có train/val 80/20, KHÔNG có test split công khai (đã verify arXiv 2603.09689) → phải tự chia 3 phần; (d) C(a) trong oracle objective chưa định nghĩa/chuẩn hoá; (e) vài mâu thuẫn nội bộ nhỏ (Kendall's W ngưỡng số vs nguyên tắc "không ngưỡng tùy tiện" ở Phần E). |

## Phần C — Outline Paper 3 (methodology giữ nguyên hướng, nhưng ĐIỀU KIỆN theo Phần G)

**⚠️ Toàn bộ Method dưới đây giả định action = (N_tile, bridge). Đây là GIẢ ĐỊNH CHƯA XÁC MINH — phụ thuộc kết quả gate G.0.3. Nếu N_tile không phải compute lever đủ mạnh (chênh lệch FLOPs/latency <15-20% giữa N_tile=1 và N_tile=8), action space thu về CHỈ CÒN bridge-capacity (hoặc số query-token của bridge). Không viết lại chi tiết Method ở đây cho tới khi có kết quả G.0.3 — tránh viết lại lần 8.**

**Tên tạm (điều kiện theo fork + G.0.3):**
- Nếu cả bridge routing VÀ tile-budget đều sống: *"Cognitive-Supervised Adaptive Visual Computation for Vietnamese VQA"*
- Nếu chỉ 1 trục sống (bridge-only hoặc tile-only): *"Cognitive-Conditioned Visual Budget Allocation for Efficient Vietnamese VQA"*

**Thesis (câu hỏi nghiên cứu, không phải claim phát minh):**
> *"We investigate whether explicit reasoning-level supervision can improve the allocation of visual computation in vision-language models beyond model-internal signals alone."*

**Dataset & split (SỬA — AutoViVQA không có test split công khai, đã verify):**
- AutoViVQA (arXiv 2603.09689): 19,411 ảnh / 37,077 câu hỏi / 5 đáp án/câu, **chỉ chia train/val 80/20**, không có test split chính thức.
- Paper 3 phải **tự chia lại 3 phần** từ phần train của AutoViVQA: đề xuất 70/15/15, **stratify theo reasoning-type VÀ theo ảnh** (không để cùng 1 ảnh xuất hiện ở nhiều split — tránh leak qua caption/context chung ảnh). Seed cố định, ghi rõ sơ đồ chia trong Method (đây cũng là lý do TRAIN/VAL/TEST ở các mục dưới không phải là train/val gốc của AutoViVQA mà là 3 phần tự chia từ 80% train).
- **Terminology:** dùng nhất quán **"reasoning-type"**, không dùng "reasoning-level/L1-L5" (ngụ ý ordinal sai — 5 loại là nominal: recognition/relational/compositional/causal/text-in-image, không có thứ tự khó-dễ đảm bảo). Giữ "L1-L5" chỉ như ký hiệu tốc ký khi cần liệt kê ngắn gọn, luôn kèm chú thích "nominal, not ordinal".
- **Cần xác minh trước khi dùng bất kỳ CSV nào có sẵn:** nếu có file dạng "60k" (khác 37k câu hỏi gốc của AutoViVQA) — phải xác định đây là gì (QA-pair nở ra theo 5 đáp án? trộn nguồn khác? "balanced" = resample?) trước khi dùng, và kiểm tra nhãn reasoning-type có phủ đủ không nếu có trộn nguồn (xem G.0.1).

**Contributions (4 gạch đầu dòng, giữ nguyên hướng round 6):**
1. Motivating experiment 3 bước (A/B/C) — instrumentation, không phải contribution chính.
2. Router nhìn Image+Question: cognitive prior P(r|Q) (giám sát bằng nhãn AutoViVQA) + trạng thái thị giác rẻ f(I,Q).
3. Policy học qua offline oracle-guided policy learning — action space **cụ thể tùy kết quả Phần G**.
4. Đánh giá đầy đủ: FLOPs, wall-clock latency thật, throughput, Pareto frontier, Oracle upper-bound.

**Related Work, Method 4.1-4.4, Experiments 5.1-5.8:** giữ nguyên nội dung đã chốt ở round 6 (P(r|Q) chỉ dùng Question; bỏ Head 2/ΔM, dùng f(I,Q) trực tiếp; oracle utility-cost U(a;x,λ)=M(a;x)−λ·C(a); ablation ladder 7 nhánh gồm Fixed-nhiều-mức/Random/Visual-state-only/Reasoning-type-only/Ours/Oracle-cognitive-prior/Oracle; Pareto = empirical frontier rời rạc; thống kê = paired bootstrap/permutation). **Chỉ sửa 2 chỗ:**
- C(a) trong U(a;x,λ) = M(a;x) − λ·C(a) **định nghĩa cụ thể**: C(a) = số lần chạy vision encoder (∝ số tile T), **chuẩn hoá về [0,1]** trước khi trừ với M (nếu không, lưới λ∈{0...1} vô nghĩa vì 2 đại lượng khác thang đo). Latency đo được (nhiễu, biến thiên theo batch/hardware) để RIÊNG ở bảng 5.6 hiệu quả tính toán, KHÔNG đưa vào hàm oracle objective.
- Diagram Method: vẽ đúng pipeline — **InternViT (frozen) → Bridge (trainable, thay thế MLP projector gốc của Vintern) → Qwen2-0.5B (frozen)** — không vẽ "Vintern-1B" như 1 khối đặc, vì bridge custom đã thay projector gốc.

## Phần D — Việc cần làm, đúng thứ tự (SỬA — chèn gate kỹ thuật TRƯỚC bước 1 cũ)

### D0 — Gate kỹ thuật, BẮT BUỘC làm trước, chặn mọi code khác

0. **[G.0.1] Xác minh dữ liệu:** cột reasoning-type có tồn tại trong CSV đang dùng? Phân phối thế nào? Nếu có file "60k" khác 37k câu hỏi gốc — xác định nguồn gốc, có phủ đủ nhãn reasoning-type không. Chốt dataset cuối cùng dùng cho toàn bộ paper.
0. **[G.0.2] Chốt sơ đồ split 3 phần:** 70/15/15 từ phần train AutoViVQA, seed cố định, stratify theo reasoning-type + theo ảnh (không leak ảnh giữa split). Ghi thành script tái lập được.
0. **[G.0.3 — GATE CỨNG, quyết định cả hướng Method]** Đo FLOPs & wall-clock end-to-end thật của pipeline ở N_tile = 1 / 4 / 8 (cùng 1 bridge cố định để so sánh công bằng).
   - Nếu chênh lệch **≥ ~15-20%** giữa N_tile=1 và N_tile=8 → N_tile là compute lever có nghĩa, giữ action = (N_tile, bridge) như Phần C mô tả.
   - Nếu chênh lệch **< 15-20%** → N_tile KHÔNG phải lever đủ mạnh (đúng lo ngại: bridge pool về vài token cố định bất kể T, LLM prefill gần như không đổi theo T, chi phí thật ở ViT quá nhỏ so với tổng pipeline) → **bỏ trục N_tile**, đổi action space sang **chỉ còn bridge-capacity** (hoặc số query-token của bridge làm lever duy nhất) → phải viết lại phần liên quan của Phần C Method (nhưng KHÔNG làm việc này trước khi có kết quả đo — tránh đoán mò).
0. **[G.0.4]** Chốt số bridge dùng (5 hay 6 — có Gated Fusion hay không) + xác định bridge nào là baseline chính.
0. **[G.0.5]** Xác nhận 5 (hoặc 6) checkpoint bridge đã train THẬT trên split mới (70/15/15 vừa chốt ở G.0.2, không phải split cũ) + có bảng Exp A (5-bridge overall) thật trên split mới — đây là tiền đề của toàn bộ paper, nếu chưa có phải train lại trước.
0. **[G.0.6, chỉ làm NẾU G.0.3 pass — tức N_tile sống]** Sửa pipeline multi-tile: (a) forward/generate nhận (B,T,3,H,W) → flatten (B·T,3,H,W) qua InternViT → gom lại theo ảnh; (b) bridge phải nuốt được chuỗi token dài biến thiên T·256 (residual/gated_fusion/multi_token hiện chỉ nhận 1 vector pooled — cần thiết kế lại hoặc khoá ở N_tile=1); (c) train lại bridge với tile-count augmentation (bridge train ở 1-tile sẽ không tự khai thác thêm tile lúc test — domain shift thật).
0. **[G.0.7]** Sửa lỗi code nhỏ: thêm PhoBERT + Vietnamese-SBERT vào requirements.txt; sửa import trần trong script tính metric (chỉ chạy đúng khi cwd đúng thư mục) thành import theo package, để oracle sweep script không vấp; ghi chú METEOR cần Java+.jar (subprocess) vào phần reproducibility hoặc cân nhắc bỏ METEOR nếu dependency này gây phiền.

### D1 — Bridge × Level (Exp B), làm sau khi xong D0

Load kết quả 5 bridge **trên VALIDATION set của split MỚI (70/15/15)** — không dùng test set, không dùng split cũ. Join nhãn reasoning-type. Tính Acc/F1/CIDEr trung bình mỗi bridge mỗi reasoning-type → heatmap.
- **Tiêu chí fork — SỬA (gỡ mâu thuẫn với nguyên tắc "không ngưỡng tùy tiện" ở Phần E):** phép thử quyết định DUY NHẤT là **paired bootstrap/permutation test** giữa bridge tốt nhất và tốt nhì mỗi reasoning-type. Kendall's W và "số reasoning-type mà top-1 đổi" **chỉ là số mô tả** (báo cáo cho dễ đọc), KHÔNG dùng làm ngưỡng quyết định (vd "< 0.7" hay "≥2/5") — vì bản thân các ngưỡng đó cũng tùy tiện y như "≥2 điểm" đã bỏ ở round 5.
- 4 gate giữ nguyên: (1) ổn định qua ≥3 seed, (2) paired bootstrap/permutation significant, (3) hợp lý ngữ nghĩa, (4) lợi ích compute thực đo được.

### D2 — Oracle utility-cost sweep, làm sau D1 (SỬA HOÀN TOÀN — bản cũ không khả thi)

**Bản cũ bị bỏ:** "chạy tất cả action trên toàn bộ TRAIN, 3-4 seed" → nếu |action|=36 (6 tile × 6 bridge), TRAIN ~30k sample, ×3 seed ≈ 3.2 triệu lần `generate()` ≈ 270-900 GPU-giờ. **Không khả thi trong 1 tháng, 1 GPU.**

**Bản sửa:**
1. Chỉ chạy oracle sweep trên **subset TRAIN 5,000-10,000 sample**, stratified theo reasoning-type (không phải toàn bộ ~30k).
2. **Thu gọn action grid** dựa trên kết quả G.0.3/D0.4/D1: nếu cả 2 trục sống, dùng N_tile ∈ {1, 4, 8} (3 giá trị, không phải 6) × bridge ∈ {top-3 từ fork D1} (3 giá trị, không phải 5-6) = **9 action**, không phải 36.
3. **Decoding greedy cố định** (bỏ sampling) cho toàn bộ sweep — đảm bảo tái lập, giảm nhiễu.
4. **Chỉ chạy ≥3 seed cho 2 chỗ:** (a) fork decision ở D1, (b) bảng kết quả TEST cuối cùng. Oracle sweep để sinh nhãn train policy chỉ cần **1 seed** — không cần lặp lại nhiều lần vì mục đích là sinh nhãn huấn luyện, không phải báo cáo kết quả cuối.
5. **Ghi tổng GPU-giờ dự kiến vào Setup** (Section 5.1) trước khi chạy — với quy mô đã thu gọn (9 action × ~7,500 sample × 1 seed ≈ 67,500 lần generate), ước lượng lại theo throughput generate thật đo được của Vintern-1B trên RTX 5090, không đoán.
6. Với mỗi sample: bảng M(a)/C(a) — M(a) là metric liên tục (CIDEr hoặc kết hợp), **C(a) = số lần chạy vision encoder chuẩn hoá [0,1]** (theo định nghĩa đã chốt ở Phần C).
7. Sinh oracle action a*(λ) cho lưới λ (7 điểm: {0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0}).
8. Train **một** policy network trên subset TRAIN này, input (P(r|Q), f(I,Q), λ), output action, cross-entropy với oracle label.
9. **VAL** (của split mới): chọn hyperparameter/early-stopping cho policy, chọn λ vận hành để báo cáo — KHÔNG sinh thêm oracle label ở đây.
10. **TEST** (của split mới): frozen policy, đánh giá 1 lần cuối, ≥3 seed cho bảng kết quả chính.

### D3 — Sau khi D0-D2 xong, mới quay lại code router/policy

11. Sửa `difficulty_router.py` — 1 head P(r|Q) (PhoBERT, giám sát nhãn thật) + nhánh f(I,Q) rẻ, KHÔNG Head 2.
12. Viết lại `policy.py` — policy network học từ oracle (D2), input (P(r|Q), f(I,Q), λ), output action đúng action-space đã chốt ở G.0.3.
13. Code training loop, chạy Pareto frontier + ablation ladder đầy đủ (bao gồm Oracle-cognitive-prior, Fixed-budget sweep nhiều mức).
14. Đo wall-clock latency thật, throughput, trainable params %.
15. Dựng bảng Policy Behavior Analysis.
16. Viết Motivating Experiment + Related Work (bảng TRouter differentiation) + Method + Experiments.
17. Format Springer LNCS/LNAI 12-15 trang. **Làm rõ: Trust4NLP LÀ một special session CỦA ACIIDS 2027, không phải venue riêng** — cùng 1 deadline, cùng hệ thống nộp EasyChair (aciids2027), chỉ chọn session Trust4NLP lúc submit. Nộp trước 27/09 để có buffer.

## Phần E — Điểm mấu chốt để "nhìn perfect"

- Thesis là câu hỏi nghiên cứu, không phải claim phát minh.
- 5-bridge chỉ là instrumentation, không phải contribution chính.
- **G.0.3 là gate quan trọng nhất của toàn bộ paper** — quyết định action space thật sự là gì trước khi viết Method chi tiết.
- Oracle sweep: quy mô đã thu gọn (subset + action grid nhỏ + 1 seed cho sweep), có ước lượng GPU-giờ ghi trong Setup.
- Split: tự chia 70/15/15 từ AutoViVQA train (không có test gốc), stratify reasoning-type + ảnh, seed cố định.
- Terminology: "reasoning-type" (nominal), không phải "reasoning-level" (ngụ ý ordinal).
- Fork/gate: paired bootstrap là quyết định DUY NHẤT, mọi ngưỡng số khác chỉ mô tả.
- C(a) định nghĩa rõ + chuẩn hoá; latency thật đo riêng, không vào oracle objective.
- Không PR phần cứng; đo nhất quán 1 GPU để so sánh công bằng.

## Phần F — Trạng thái code (TẠM DỪNG, chưa code — chờ xong Phần G trước)

```
adaptive_router/
├── src/modeling/
│   ├── difficulty_router.py   ⚠️ thiết kế CŨ (2-head, có Head 2/ΔM) → cần sửa theo Phần C, nhưng
│   │                              CHỜ G.0.3 xong (biết action space thật) mới sửa, tránh sửa 2 lần
│   └── policy.py               ⚠️ thiết kế CŨ (bảng hard-code) → viết lại sau D0-D2, không phải bây giờ
└── (chưa có: oracle sweep script, split script, FLOPs/latency measurement script — cần làm TRƯỚC
    2 file trên, theo đúng thứ tự D0 → D1 → D2 → D3)
```

**Việc tiếp theo KHÔNG phải sửa router/policy.** Việc tiếp theo là **G.0.1 → G.0.2 → G.0.3** — xác minh dữ liệu, chốt split, đo FLOPs/latency thật theo N_tile. G.0.3 là phép đo đơn giản (chạy vài chục sample qua pipeline hiện có ở N_tile=1/4/8, bấm giờ + đếm FLOPs), không cần code kiến trúc mới, làm được ngay và cho câu trả lời quyết định hướng đi tiếp theo.
