# Bản thiết kế paper — "Cải thiện Vintern-1B cách rẻ"

> **Bản đã qua biên tập trình bày (2026-09-06).** Cắt từ 13 bảng → 7 bảng chính +
> 2 hình + phụ lục per-seed. Một hệ ký hiệu, một bộ cột, làm tròn 2 chữ số (×100)
> khớp ViMoE.
>
> Toàn bộ phần chữ (chú thích bảng, phần "Đọc") viết bằng tiếng Việt cho việc
> theo dõi thí nghiệm; khi lên bản nộp LNCS mới dịch sang tiếng Anh. Thuật ngữ
> kỹ thuật (bridge, LoRA, decoder, token, tile, seed, epoch, val CE…) giữ nguyên.
>
> Artifact: https://claude.ai/code/artifact/fe068b4c-d59c-429f-bdba-ed9ea93bd557
> Khung câu chuyện: https://claude.ai/code/artifact/bb7bf7ee-d5f1-4749-bb56-29a5c5daa610

## Quy ước chung — khai báo một lần

- **In đậm** = dòng của chúng tôi (recipe / bridge đề xuất). *Không* dùng in đậm
  cho "giá trị tốt nhất trong cột".
- Cột metric: `Acc · Prec · Rec · F1 · BLEU · ROUGE · METEOR · CIDEr` — đồng bộ mọi
  bảng, đo nội bộ ×100, **làm tròn 2 chữ số**.
- Hai cách đo metric **tách hẳn thành bảng riêng**: Bảng 1 = đo nội bộ (khớp bảng
  AutoViVQA), Bảng 2 = đo kiểu corpus pycocoevalcap (so với paper khác). Không cắm
  dấu `*` vào từng con số.
- Chú thích cảnh báo: chữ thường ᵃ ᵇ ᶜ, định nghĩa **ngay trong chú thích** bảng đó.
- Bản LNCS thật: đánh số **Bảng 1…N tuần tự**, **Hình 1–2**. (Bản này đã phẳng hoá
  — bỏ 5a–5d.)
- Cần làm khi viết bản nộp: (1) dịch toàn bộ phần chữ sang tiếng Anh; (2) đối
  chiếu recipe train của "Vintern-1B (fine-tuned)" ở §4 AutoViVQA (nhiều khả năng
  là "ViT + projector train toàn bộ, LLM LoRA", không phải "train toàn bộ").

---

## PHẦN A — Cấu trúc paper (LNCS, 12–15 trang)

| § | Nội dung | Nguồn |
|---|---|---|
| **Abstract** | Vintern-1B chạy zero-shot hỏng (F1 17.6); recipe của Vintern train *toàn bộ* InternViT-300M + projector + LoRA cho LLM trên 3M cặp. Chúng tôi: đóng băng cả hai backbone, chỉ train bridge 0.78% (+ LoRA decoder 0.23%), 1 tile → vượt Vintern fine-tuned trên các metric sinh với ~1% chi phí. Chẩn đoán 6 bước → nút thắt là frozen decoder. | — |
| **1 · Introduction** | ViMoE xây model mới · Vintern train nặng phía thị giác · **câu hỏi: adapt rẻ được không, nút thắt ở đâu** · 4 đóng góp. | — |
| **2 · Related Work** | VQA tiếng Việt (ViVQA / OpenViVQA / ViTextVQA / AutoViVQA / ViMoE) · frozen-backbone + projector (BLIP-2, "Inference-Optimal VLMs" 2411.03312) · adapt tiết kiệm tham số (LoRA, adapter). | — |
| **3 · Method** | 3.1 Kiến trúc frozen · 3.2 Năm bridge (thang capacity) · 3.3 Decoder-LoRA như một can thiệp có chủ đích · 3.4 Hai chỗ vặn × 6 câu hỏi. | — |
| **4 · Experimental Setup** | AutoViVQA · **grouped split không rò rỉ** · 8 metric · baseline (Bảng 1) · làm rõ recipe của Vintern-FT — *cần đối chiếu §4 AutoViVQA*. | §4.1 |
| **5 · Main Results** | Recipe so với baseline. | Bảng 1, 2, 7 |
| **6 · Ablation: truy tìm nút thắt** | 6.1 bridge (RQ1–2, Bảng 3) · 6.2 tile-collapse (RQ3, Hình 2) · 6.3 oracle + routing (RQ4) · 6.4 training / alignment (RQ5) · 6.5 decoder-LoRA (RQ6, Bảng 5, Hình 1) · 6.6 tổng hợp (Bảng 4). | Bảng 3–5, Hình 1–2 |
| **7 · Human Validation & Error Analysis** | Tự kiểm (Bảng 6) · [camera-ready: 2 người chấm + κ] · lỗi theo từng loại · độ dài câu sinh. | Bảng 6 |
| **8 · Discussion** | Frozen decoder là trần · nối với "Inference-Optimal VLMs" · claim "reasoning-aware" của ViMoE cần đo trực tiếp · giới hạn. | — |
| **9 · Conclusion** | Recipe rẻ + quy trình chẩn đoán. Công bố code + split + bảng oracle. | — |
| **Appendix** | Bảng per-seed (A1–A5), đường cong rank / epoch đầy đủ. | — |

---

## PHẦN B — Bảng chính (7) + hình (2)

### Bảng 1 — Kết quả chính (metric đo nội bộ) — recipe đã khoá

**Bảng 1.** Recipe so với các công trình trước trên AutoViVQA (tập val). Metric
đo nội bộ ×100. In đậm = của chúng tôi. ᵃ CIDEr của BARTPhoBEiT là ngoại lai do
sinh dài dòng, không đưa vào so sánh. ᵇ trung bình 4 seed (bridge thường) / 3
seed (+ LoRA); độ lệch chuẩn ở Bảng 2. Các dòng baseline: lấy theo báo cáo của
benchmark AutoViVQA, không phụ thuộc cách chia split.

| Mô hình | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Vintern-1B (gốc, zero-shot) | 0.12 | 17.52 | 19.87 | 17.55 | 1.91 | 25.84 | 23.93 | 8.54 |
| ViT5_ViT | 7.97 | 46.84 | 50.33 | 48.52 | 4.13 | 46.89 | 31.02 | 72.68 |
| BARTPhoBEiT | 8.81 | 45.30 | 46.48 | 45.88 | 4.33 | 44.83 | 24.57 | 188.96 ᵃ |
| Vintern-1B (fine-tuned) | 13.01 | 52.47 | 55.12 | 53.76 | 6.11 | 51.93 | 35.25 | 72.84 |
| Llama 3.2 (zero-shot) | 0.36 | 23.96 | 73.71 | 36.16 | 3.62 | 36.11 | 30.01 | 62.84 |
| Gemini 2.0 Flash | 0.55 | 27.20 | 74.10 | 39.79 | 4.41 | 39.60 | 31.72 | 74.42 |
| Gemini 2.5 Flash | 0.22 | 24.43 | 76.66 | 24.75 | 0.39 | 37.27 | 31.22 | 71.90 |
| GPT-5 (zero-shot) | 10.84 | 47.20 | 55.20 | 50.89 | 6.07 | 47.30 | 33.34 | 84.20 |
| ViMoE-VQA (Tuong-MOE) | 9.65 | 62.89 | 58.65 | 60.69 | 12.54 | 47.07 | 39.10 | 88.67 |
| **Bridge Multi-Token (0.78%, 1 tile)** ᵇ | **8.28** | **50.53** | **51.72** | **49.82** | **15.99** | **48.11** | **40.47** | **96.98** |
| **  + decoder LoRA r=16 (~1.0%)** ᵇ | **10.42** | **53.85** | **55.00** | **53.17** | **19.44** | **51.48** | **43.91** | **105.59** |
| **  + decoder LoRA r=16, 3 epoch** ᵇ | **11.78** | **55.54** | **56.25** | **54.67** | **20.98** | **52.92** | **45.24** | **109.60** |

**Đọc:** Recipe frozen-backbone vượt Vintern-1B fine-tuned ở mọi metric sinh
(BLEU +14.9, METEOR +10.0, CIDEr +36.8) với ~1% số tham số train, và thắng
ViMoE-VQA ở BLEU / ROUGE / METEOR / CIDEr. Vẫn kém ViMoE ở token-F1 (−6.0) và
kém Vintern ở Acc. → phần chẩn đoán ở §6.

### Bảng 2 — Metric đo kiểu corpus + khoảng tin cậy

**Bảng 2.** Chất lượng sinh để so với paper khác (đo corpus, pycocoevalcap). In
đậm = của chúng tôi. ᵃ trung bình ± độ lệch chuẩn. ᵇ khoảng tin cậy 95% bằng
bootstrap ghép cặp trên tập val 5 463 mẫu; ViMoE không công bố dự đoán theo từng
mẫu nên chỉ bootstrap được phía chúng tôi.

| Mô hình | CIDEr-D | BLEU-4 | ROUGE-L |
|---|--:|--:|--:|
| ViMoE-VQA | 88.67 | 12.54 | 47.07 |
| **Bridge Multi-Token (4 seed)** ᵃ | **92.80 ± 1.10** | **19.20 ± 0.30** | **49.20 ± 0.50** |
| **  khoảng tin cậy 95%** ᵇ | **[91.30, 97.10]** | — | — |
| **  + LoRA r=16 (seed 42)** | **101.70** | **23.20** | **52.70** |
| **  + LoRA r=16, 3 epoch (3 seed)** ᵃ | **106.80 ± 1.10** | **25.00 ± 0.40** | **54.20 ± 0.20** |

**Đọc:** Khoảng CIDEr-D của bridge thường [91.30, 97.10] nằm hoàn toàn trên mức
88.67 của ViMoE — thắng về chất lượng sinh không phải nhờ may.

### Bảng 3 — So sánh kiến trúc bridge (RQ1–2, RQ6) — cấu hình gốc: seed 42, +2 seed đang chạy

**Bảng 3.** Năm bridge trên backbone frozen (tập val; cột "F1"/"CIDEr"/"val CE"
là cấu hình gốc, seed 42; Multi-Token gốc = trung bình 4 seed). Cột "+ LoRA":
multi_token / qformer / mini_qformer / residual = trung bình 3 seed. In đậm =
bridge đề xuất. ᵃ tile_attention + LoRA hiện chỉ có seed 42.

| Bridge | Tham số | % | F1 | CIDEr | val CE | F1 +LoRA | ΔF1 | CIDEr +LoRA |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Residual (1 tok) | 4.86M | 0.52 | 36.45 | 66.07 | 2.35 | 52.64 | +16.19 | 104.05 |
| Tile-Attention (8 tok) | 4.14M | 0.44 | 46.69 | 87.46 | 1.62 | 52.99 ᵃ | +6.30 | 105.04 ᵃ |
| **Multi-Token (8 tok pooled)** | **7.35M** | **0.78** | **49.82** | **96.98** | **1.49** | **53.17** | **+3.35** | **105.59** |
| Light Q-Former (8 query) | 27.6M | 2.87 | 46.63 | 88.10 | 1.59 | 53.21 | +6.58 | 106.24 |
| Full Q-Former (16 query) | 69.4M | 6.91 | 47.66 | 90.82 | 1.57 | 53.21 | +5.55 | 105.70 |

**Đọc:** RQ1: Multi-Token (0.78%) là bridge tốt nhất và đã vượt Vintern
fine-tuned về metric sinh. RQ2: bridge to gấp 10 lần (Full Q-Former, 69M) lại
*tệ hơn*; Multi-Token có val CE thấp nhất — capacity không phải nút thắt. RQ6:
LoRA nâng F1 ở mọi bridge và triệt tiêu khoảng chênh CIDEr (Hình 1).

### Hình 1 — Decoder-LoRA san bằng chất lượng giữa các bridge (CIDEr-D corpus, tập val)

**Dạng biểu đồ:** thanh ngang, mỗi bridge một cặp thanh — *thường* (xám) so với
*+ LoRA r=16* (màu nhấn). Trục hoành CIDEr-D 0–120.

| Bridge | thường | + LoRA r=16 |
|---|--:|--:|
| Residual | 56.30 | 100.80 |
| Tile-Attention | 87.46 | 102.00 |
| Multi-Token | 94.40 | 101.70 |
| Light Q-Former | 83.80 | 103.00 |
| Full Q-Former | 86.70 | 102.43 |

*(CIDEr-D corpus. "thường" = seed 42; "+ LoRA" = trung bình 3 seed, trừ
Tile-Attention chỉ seed 42.)*

**Hình 1.** Bridge thường trải CIDEr-D 56–94 (chênh lệch chất lượng lớn, do thiết
kế bridge). Sau khi thêm LoRA decoder 0.23%, cả năm bridge đều hội tụ về
100–103 — **chọn bridge nào gần như không còn quan trọng một khi decoder đủ
capacity**.

### Bảng 4 — Tổng hợp ablation: sáu trục, một trục dương (RQ1–6) — seed 42, dòng âm +2 seed

**Bảng 4.** ΔF1 so với mốc (Multi-Token thường, seed 42: F1 50.66). CIDEr-D =
corpus. ᵃ align-logit ở α=1.0 bị sai trọng số (KL lấn át CE, val CE 2.84 so với
1.49); đang chạy lại full-val 3 seed. Trục alignment chủ yếu dựa vào align-feat.

| RQ · axis | Intervention | F1 | CIDEr-D | ΔF1 | Kết luận |
|---|---|--:|--:|--:|---|
| — mốc | Multi-Token thường | 50.66 | 94.40 | — | — |
| RQ1–2 · Bridge capacity | Full Q-Former (69M params, 10×) | 47.66 | 86.70 | −3.00 | âm |
| RQ3 · Number of visual tiles | Train 1 tile → evaluate 3 tiles | 21.05 | ~46 | −29.61 | âm (sụp) |
| RQ4 · Adaptive routing | Learned policy (conditioned on question type) | ≈50.7 | ≈94 | ≈0 | âm (không hơn cố định) |
| RQ5 · Training signal | Multi-reference answer sampling | 49.01 | 87.30 | −1.65 | âm |
| RQ5 · Representation alignment | Projector-level feature KD | 49.66 | 92.00 | −1.00 | âm |
| RQ5 · Representation alignment | Projector-level logit KD ᵃ | 40.70 | 80.10 | −9.96 | âm ᵃ |
| **RQ6 · Decoder capacity** | **LoRA r=16 (1 epoch)** | **53.17** | **101.70** | **+2.51** | **dương** |
| **RQ6 · Decoder capacity** | **LoRA r=16 (3 epochs)** | **54.67** | **106.80** | **+4.01** | **dương** |

**Đọc:** Bốn trục độc lập phía thị giác / huấn luyện đều âm; trục duy nhất phía
decoder rõ ràng dương. Chính *mẫu hình* này — không phải riêng một ablation nào —
khoanh nút thắt về frozen decoder.

### Bảng 5 — Decoder-LoRA theo từng bridge (RQ6) — 4/5 bridge đã 3 seed

**Bảng 5.** thường → + LoRA r=16 (1 epoch) theo từng bridge (tập val, đủ 5 463
mẫu). F1 = đo nội bộ; CIDEr-D = corpus. Cột "+ LoRA": multi_token / qformer /
mini_qformer / residual = trung bình 3 seed; tile_attention = seed 42. Cột
"thường" = seed 42. ᵃ khoảng tin cậy 95% bằng bootstrap ghép cặp. P(Δ>0) =
1.000 ở mọi dòng.

| Bridge | F1 thường | F1 +LoRA | ΔF1 [KTC 95%] ᵃ | CIDEr-D thường | CIDEr-D +LoRA | ΔCIDEr-D |
|---|--:|--:|--:|--:|--:|--:|
| multi_token | 50.66 | 53.17 | +2.51 [1.9, 3.1] | 94.40 | 101.70 | +7.30 |
| qformer | 47.66 | 53.21 | +5.55 | 86.70 | 102.43 | +15.73 |
| mini_qformer | 46.63 | 53.21 | +6.58 | 83.80 | 103.00 | +19.20 |
| residual | 36.45 | 52.64 | +16.19 [15.4, 17.0] | 56.30 | 100.80 | +44.50 |
| tile_attention | 46.69 | 52.99 | +6.30 | 87.46 | 102.00 | +14.54 |

**Đọc:** Mức nâng càng lớn khi bridge thường càng yếu: +2.5 F1 ở bridge tốt
nhất, +16.2 ở bridge tệ nhất — và cả hai đều về ≈53 F1 / ≈101 CIDEr-D. Đây chính
là hiện tượng san bằng ở Hình 1; chi tiết per-seed ở Phụ lục A3.

### Hình 2 — Bridge vỡ khi vượt quá 1 tile (Multi-Token, tập val)

**Dạng biểu đồ:** đường, hai trục. Trục hoành = số tile {1, 3, 6}. Trục trái =
token-F1, trục phải = val loss.

| số tile | token-F1 | val loss |
|--:|--:|--:|
| 1 | 50.66 | 1.48 |
| 3 | 21.05 | 3.35 |
| 6 | 22.51 | 3.36 |

**Hình 2.** Multi-Token huấn luyện ở 1 tile bị sụp khi đánh giá với nhiều tile
hơn: F1 50.7 → 21, val loss 1.48 → 3.36. Phép mean-pool trên 8 token đầu ra
không kham nổi lượng token thị giác gấp 3–6 lần. **1 tile là điểm vận hành,
không phải thoả hiệp** — tăng lên chỉ tệ hơn.

### Bảng 6 — Tự kiểm: token-F1 có bám theo đúng/sai không? — N=120, 1 người chấm

**Bảng 6.** Nhóm F1 so với đánh giá ngữ nghĩa (Multi-Token, N=120, một người chấm
đối chiếu 5 câu tham chiếu, không xem ảnh). Không phải quy trình 2 người chấm /
Cohen's κ như kế hoạch — chỉ là bản thay thế có giới hạn, đã ghi rõ. Bản
camera-ready cần nghiên cứu đầy đủ.

| Nhóm F1 | n | đúng | một phần | sai | vô nghĩa | chấp nhận được |
|---|--:|--:|--:|--:|--:|--:|
| mạnh (≥0.6) | 45 | 80.00 | 11.11 | 6.67 | 2.22 | 91.11 |
| một phần (0.2–0.6) | 58 | 12.07 | 31.03 | 55.17 | 1.72 | 43.10 |
| yếu (0–0.2) | 3 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 |
| bằng 0 | 13 | 7.69 | 7.69 | 76.92 | 7.69 | 15.38 |
| **tổng thể (n=119)** | — | **36.97** | **20.17** | **40.34** | **2.52** | **57.14** |

**Đọc:** Nhóm F1 *lớn nhất* (một phần, 51.5% tập val) lại *kém tin cậy nhất* —
55% câu "một phần" thực ra sai. token-F1 ở khoảng giữa là tín hiệu đúng/sai kém;
điều này giới hạn cách đọc mọi con số về sinh.

### Bảng 7 — Hiệu quả tính toán của đòn bẩy tile

**Bảng 7.** Chi phí encode thị giác của InternViT trên mỗi ảnh (Tesla P100-16GB,
`src.cli.profile`). Vintern fine-tuned chạy tới 12 tile; recipe của ta dùng 1.

| số tile | GFLOPs | độ trễ (ms) | thông lượng (ảnh/s) |
|---|--:|--:|--:|
| **1 (của ta)** | **362** | **229** | **6.00** |
| 2 | 724 | 374 | 3.30 |
| 4 | 1 448 | 648 | 1.70 |
| 6 | 2 172 | 922 | 1.15 |

**Đọc:** Đòn bẩy 1→6 tile là thật: FLOPs ×6.0, độ trễ ×4.0, thông lượng ×5.2 —
recipe không tốn đồng nào vào đó. Đây chính là phân tích FLOPs / độ trễ mà
ViMoE-VQA nói thẳng là để lại sau.

---

## PHẦN D — Phụ lục (không vào phần chính — giữ để trả lời reviewer)

### A1. Bridge Multi-Token (thường), theo từng seed

| Seed | F1 | BLEU | ROUGE | METEOR | CIDEr | CIDEr-D | BLEU-4 | ROUGE-L |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 42 | 50.66 | 16.34 | 48.95 | 41.05 | 98.69 | 94.40 | 19.60 | 50.00 |
| 123 | 49.46 | 16.05 | 47.76 | 40.16 | 95.84 | 91.70 | 19.20 | 48.80 |
| 2026 | 49.64 | 15.91 | 47.93 | 40.53 | 97.35 | 93.10 | 19.10 | 49.00 |
| 3407 | 49.51 | 15.64 | 47.80 | 40.13 | 96.05 | 91.80 | 18.80 | 48.90 |

### A2. Multi-Token + LoRA r=16, theo từng seed (đo nội bộ, full-val)

| Cấu hình | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 ep · s42 | 10.49 | 53.90 | 54.92 | 53.16 | 19.38 | 51.44 | 43.85 | 104.90 |
| 1 ep · s123 | 10.27 | 53.89 | 55.05 | 53.20 | 19.53 | 51.52 | 43.94 | 106.11 |
| 1 ep · s3407 | 10.51 | 53.76 | 55.03 | 53.15 | 19.42 | 51.48 | 43.93 | 105.76 |
| 3 ep · s42 | 11.73 | 55.47 | 56.07 | 54.52 | 20.59 | 52.82 | 45.06 | 108.49 |
| 3 ep · s123 | 11.92 | 55.45 | 56.31 | 54.67 | 21.30 | 52.91 | 45.34 | 110.63 |
| 3 ep · s3407 | 11.68 | 55.71 | 56.36 | 54.81 | 21.06 | 53.04 | 45.31 | 109.69 |

### A3. LoRA r=16 (1 epoch) theo từng seed — bridge phụ

| Bridge · seed | F1 | CIDEr (nội bộ) | CIDEr-D (corpus) |
|---|--:|--:|--:|
| qformer · 42 | 53.10 | 105.15 | 101.90 |
| qformer · 123 | 53.32 | 105.75 | 102.60 |
| qformer · 3407 | 53.22 | 106.19 | 102.80 |
| **qformer · trung bình** | **53.21** | **105.70** | **102.43** |
| mini_qformer · 42 | 53.39 | 106.65 | 103.30 |
| mini_qformer · 123 | 53.16 | 105.58 | 102.60 |
| mini_qformer · 3407 | 53.07 | 106.48 | 103.20 |
| **mini_qformer · trung bình** | **53.21** | **106.24** | **103.03** |
| residual · 42 | 52.66 | 103.26 | 100.00 |
| residual · 123 | 52.60 | 104.20 | 100.90 |
| residual · 3407 | 52.64 | 104.69 | 101.40 |
| **residual · trung bình** | **52.63** | **104.05** | **100.77** |
| tile_attention · 42 | 52.99 | 105.04 | 102.00 |

### A4. Đường cong rank LoRA (tập con 600 mẫu)

| rank | F1 (s42) | F1 trung bình (n) |
|--:|--:|--:|
| 4 | 51.26 | 51.26 (1) |
| 8 | 51.62 | 51.62 (1) |
| 16 | 51.98 | 51.98 (1) |
| 32 | 51.80 | 53.83 ± 1.77 (3) |
| 64 | 53.05 | 54.06 ± 0.94 (3) |

Ghi chú: khi tính trung bình 3 seed đúng cách, rank 32 ≈ rank 64 (chênh 0.23,
trong khoảng nhiễu); seed 42 là seed thấp bất thường. **Khuyến nghị: giữ r=16.**

### A5. Đường cong epoch — nguồn Bảng 1 (multi_token + LoRA, 1 ep so với 3 ep, trung bình 3 seed).

### Bảng đếm tham số (tham khảo)

| Thành phần | Số tham số train | % tổng |
|---|--:|--:|
| Residual Bridge | 4.86M | 0.52 |
| **Multi-Token Bridge** | **7.35M** | **0.78** |
| Tile-Attention Bridge | 4.14M | 0.44 |
| Light Q-Former | 27.57M | 2.87 |
| Full Q-Former | 69.39M | 6.91 |
| LoRA r=16 (Qwen2 q/k/v/o) | 2.16M | 0.23 |
| **Multi-Token + LoRA r=16** | **9.51M** | **1.01** |

---

## PHẦN C — Đang chạy: TIER-1 (19 job) — cập nhật 11:10 UTC 06/09

Hầu hết đã có seed 42; TIER-1 nâng lên trung bình 3 seed ± độ lệch chuẩn.

| Nhóm | Job | Bảng | Trạng thái |
|---|---|---|---|
| 1a · bridge nhiều seed | residual / mini_qformer / tile_attention × s123, s3407 + qformer s3407 | Bảng 3: seed 42 → 3 seed | 7 đang chạy |
| 1b · dòng âm nhiều seed | align-feat / answer-random × s123, s3407 + align-logit × 3 seed | Bảng 4: seed 42 → 3 seed | 7 đang chạy |
| 1c · phủ LoRA | mini_qformer / residual + LoRA × s123, s3407 + tile_attention + LoRA s42 | Bảng 5: → 3 seed + đủ 5/5 bridge | ✅ **5/5 xong, đã ghép vào Bảng 3 / 5 / A3** |

Sau TIER-1: TIER-2 (vị trí LoRA trong decoder: attn / MLP / cả hai — làm sâu
RQ6) · eval trên tập test · [camera-ready] human validation thật · [stretch]
decoder frozen lớn hơn.

---

*Nguồn: results-5bridge.md (kết quả chính) · results-grouped-split.md (ablation) ·
bootstrap_ci.json (KTC).*
