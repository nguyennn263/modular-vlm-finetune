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
| **Bridge Multi-Token (0.78%, 1 tile)** ᵇ | **8.20** | **50.36** | **51.43** | **49.55** | **15.47** | **47.84** | **40.22** | **96.49** |
| **  + decoder LoRA r=16 (~1.0%)** ᵇ | **10.42** | **53.85** | **55.00** | **53.17** | **19.44** | **51.48** | **43.91** | **105.59** |
| **  + decoder LoRA r=16, 3 epoch** ᵇ | **11.78** | **55.54** | **56.25** | **54.67** | **20.98** | **52.92** | **45.24** | **109.60** |

**Đọc:** Recipe frozen-backbone vượt Vintern-1B fine-tuned ở mọi metric sinh
(BLEU +14.9, METEOR +10.0, CIDEr +36.8) với ~1% số tham số train, và thắng
ViMoE-VQA ở BLEU / ROUGE / METEOR / CIDEr. Vẫn kém ViMoE ở token-F1 (−6.1) và
kém Vintern ở Acc. → phần chẩn đoán ở §6.

**Đối chiếu tập test (n=5468, 4 seed):** Bridge Multi-Token F1 **49.20** / CIDEr
**93.24** — chênh so với val (49.55 / 96.49) là −0.35 / −3.25, nhỏ và không nhất
quán về chiều → **không overfit vào val**. (mini_qformer test F1 47.25 vs val
47.05; residual 45.49 vs 45.91; tile_attention 44.44 vs 44.50.)

### Bảng 2 — Metric đo kiểu corpus + khoảng tin cậy

**Bảng 2.** Chất lượng sinh để so với paper khác (đo corpus, pycocoevalcap). In
đậm = của chúng tôi. ᵃ trung bình ± độ lệch chuẩn. ᵇ khoảng tin cậy 95% bằng
bootstrap ghép cặp trên tập val 5 463 mẫu; ViMoE không công bố dự đoán theo từng
mẫu nên chỉ bootstrap được phía chúng tôi.

| Mô hình | CIDEr-D | BLEU-4 | ROUGE-L |
|---|--:|--:|--:|
| ViMoE-VQA | 88.67 | 12.54 | 47.07 |
| **Bridge Multi-Token (4 seed, 2 epoch)** ᵃ | **92.30 ± 0.60** | **18.90 ± 0.30** | **48.90 ± 0.10** |
| **  + LoRA r=16 (seed 42)** | **101.70** | **23.20** | **52.70** |
| **  + LoRA r=16, 3 epoch (3 seed)** ᵃ | **106.80 ± 1.10** | **25.00 ± 0.40** | **54.20 ± 0.20** |

**Đọc:** Khoảng CIDEr-D của bridge thường [91.30, 97.10] nằm hoàn toàn trên mức
88.67 của ViMoE — thắng về chất lượng sinh không phải nhờ may.

### Bảng 3 — So sánh kiến trúc bridge (RQ1–2, RQ6) — 2 epoch, 3-seed

**Bảng 3.** Năm bridge trên backbone frozen (tập val). Cột "F1"/"CIDEr"(in-house)/
"val CE" = cấu hình gốc, **trung bình 3 seed @ 2 epoch** (Multi-Token = 4 seed).
Cột "+ LoRA": r=16 1ep, mean 3 seed (tile_attention ᵃ = seed 42). In đậm =
bridge đề xuất.

| Bridge | Tham số | % | F1 | CIDEr | val CE | F1 +LoRA | ΔF1 | CIDEr +LoRA |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Residual (1 tok) | 4.86M | 0.52 | 45.64 | 86.25 | 1.67 | 52.64 | +7.0 | 104.05 |
| Tile-Attention (8 tok) | 4.14M | 0.44 | 45.17 | 84.21 | 1.67 | 52.99 ᵃ | +7.8 | 105.04 ᵃ |
| **Multi-Token (8 tok pooled)** | **7.35M** | **0.78** | **49.55** | **96.49** | **1.49** | **53.17** | **+3.6** | **105.59** |
| Light Q-Former (8 query) | 27.6M | 2.87 | 46.25 | 86.80 | 1.60 | 53.21 | +7.0 | 106.24 |
| Full Q-Former (16 query) | 69.4M | 6.91 | 47.36 | 88.31 | 1.57 | 53.21 | +5.9 | 105.70 |

**Đọc:** RQ1: Multi-Token (0.78%) là bridge tốt nhất và đã vượt Vintern
fine-tuned về metric sinh. RQ2: bridge to gấp 10 lần (Full Q-Former, 69M) lại
*tệ hơn*; Multi-Token có val CE thấp nhất — capacity không phải nút thắt. RQ6:
LoRA nâng F1 ở mọi bridge (mức nâng lớn hơn khi bridge yếu hơn: +3.6 → +7.8) và
san bằng khoảng chênh CIDEr (Hình 1). *(residual không còn là ngoại lai — số cũ
F1 36.45 là lần chạy seed-42 hỏng, val CE 2.35.)*

### Hình 1 — Decoder-LoRA san bằng chất lượng giữa các bridge (CIDEr-D corpus, tập val)

**Dạng biểu đồ:** thanh ngang, mỗi bridge một cặp thanh — *thường* (xám) so với
*+ LoRA r=16* (màu nhấn). Trục hoành CIDEr-D 0–120.

| Bridge | thường | + LoRA r=16 |
|---|--:|--:|
| Residual | 81.10 | 100.80 |
| Tile-Attention | 79.03 | 102.00 |
| Multi-Token | 92.30 | 101.70 |
| Light Q-Former | 83.73 | 103.00 |
| Full Q-Former | 86.93 | 102.43 |

*(CIDEr-D corpus, trung bình 3 seed @ 2 epoch. "+ LoRA" 1ep, trừ Tile-Attention
= seed 42.)*

**Hình 1.** Bridge thường trải CIDEr-D 79–92 (chênh lệch chất lượng do thiết kế
bridge, với 3 kiểu trộn token khác nhau). Sau khi thêm LoRA decoder 0.23%, cả
năm bridge đều hội tụ về 100.8–103.0 — **chọn bridge nào gần như không còn quan
trọng một khi decoder đủ capacity**.

### Bảng 4 — Tổng hợp ablation: sáu trục, một trục dương (RQ1–6) — 2 epoch, 3-seed

**Bảng 4.** ΔF1 so với mốc (Multi-Token thường, trung bình 4 seed @ 2ep: F1
49.55). CIDEr-D = corpus. Mọi số 3-seed trừ RQ3/RQ4 (seed 42). ᵃ align-logit ở
α=1.0 bị sai trọng số (KL lấn át CE, val CE ~2.05 so với 1.49) — dòng "sai trọng
số" đã ghi rõ; align-feat (đúng trọng số) là bằng chứng chính cho RQ5.

| RQ · axis | Intervention | F1 | CIDEr-D | ΔF1 | Kết luận |
|---|---|--:|--:|--:|---|
| — mốc | Multi-Token thường (4 seed) | 49.55 | 92.30 | — | — |
| RQ1–2 · Bridge capacity | Full Q-Former (69M params, 10×) | 47.36 | 88.31 | −2.19 | âm |
| RQ3 · Number of visual tiles | Train 1 tile → evaluate 3 tiles | 21.05 | ~46 | −28.50 | âm (sụp) |
| RQ4 · Adaptive routing | Learned policy (conditioned on question type) | ≈50.7 | ≈94 | ≈0 | âm (không hơn cố định) |
| RQ5 · Training signal | Multi-reference answer sampling | 48.08 | ~86.7 | −1.47 | âm |
| RQ5 · Representation alignment | Projector-level feature KD | 49.53 | 92.10 | **−0.03** | âm (null sạch) |
| RQ5 · Representation alignment | Projector-level logit KD ᵃ | 40.75 | ~70.7 | −8.80 | âm ᵃ |
| **RQ6 · Decoder capacity** | **LoRA r=16 attn (1 epoch)** | **53.17** | **101.70** | **+3.62** | **dương** |
| **RQ6 · Decoder capacity** | **LoRA r=16 attn (3 epochs)** | **54.67** | **106.80** | **+5.12** | **dương** |
| RQ6 · Decoder capacity | LoRA r=16 **MLP-only** | 20.24 | — | −29.31 | phân kỳ |
| RQ6 · Decoder capacity | LoRA r=16 **attn + MLP** | 37.51 | — | −12.04 | phân kỳ |

**Đọc:** Bốn trục độc lập phía thị giác / huấn luyện đều âm (align-feat = null
tuyệt đối); trục duy nhất phía decoder dương — và cụ thể là **attention** của
decoder (MLP-only / attn+MLP đều phân kỳ, xem Bảng 5b). Chính *mẫu hình* này
khoanh nút thắt về attention của frozen decoder.

### Bảng 5a — Decoder-LoRA theo từng bridge (RQ6) — 2 epoch plain, LoRA 1ep

**Bảng 5a.** thường → + LoRA r=16 attn (1 epoch) theo từng bridge (tập val,
5 463 mẫu). F1 = đo nội bộ; CIDEr-D = corpus. "thường" = trung bình 3 seed @ 2ep;
"+ LoRA": multi_token / qformer / mini_qformer / residual = trung bình 3 seed,
tile_attention = seed 42.

| Bridge | F1 thường | F1 +LoRA | ΔF1 | CIDEr-D thường | CIDEr-D +LoRA | ΔCIDEr-D |
|---|--:|--:|--:|--:|--:|--:|
| multi_token | 49.55 | 53.17 | +3.6 | 92.3 | 101.7 | +9.4 |
| qformer | 47.36 | 53.21 | +5.9 | 86.9 | 102.4 | +15.5 |
| mini_qformer | 46.25 | 53.21 | +7.0 | 83.7 | 103.0 | +19.3 |
| residual | 45.64 | 52.64 | +7.0 | 81.1 | 100.8 | +19.7 |
| tile_attention | 45.17 | 52.99 | +7.8 | 79.0 | 102.0 | +23.0 |

**Đọc:** 5 bridge plain trải F1 45.2–49.6 / CIDEr-D 79–92 → sau LoRA đều F1
52.6–53.2 / CIDEr-D 100.8–103.0. Mức nâng lớn hơn khi bridge yếu hơn (+3.6 →
+7.8). Đây là hiện tượng san bằng ở Hình 1; per-seed ở Phụ lục A3.

### Bảng 5b — Decoder-LoRA: vị trí trong decoder (TIER-2, RQ6 sâu) — 3-seed, multi_token

**Bảng 5b.** LoRA r=16, α=32, 1 epoch, multi_token. Thay module gắn LoRA.

| Target module | F1 (3-seed) | val loss | Kết luận |
|---|--:|--:|---|
| attention (q/k/v/o) — recipe | **53.17** | 1.37 | ✅ +3.6 vs plain, ổn định |
| MLP (gate/up/down_proj) | 20.24 ± 1.52 | ~3.7 | 💥 phân kỳ |
| attention + MLP (cả 7) | 37.51 ± 1.70 | ~2.08 | 💥 phân kỳ (attn cứu một phần) |

**Đọc:** Dư địa hữu ích của decoder nằm **cụ thể ở attention**. LoRA lên
feed-forward làm training phân kỳ (val loss 3–4 vs 1.37). *Caveat:* có thể là
hyperparameter artifact (α=32 quá mạnh cho MLP dim ~4864 vs attn 896) — claim
giới hạn ở cấu hình recipe.

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

### A1. Bridge Multi-Token (thường, 2 epoch), theo từng seed — val + test

| Seed | F1 (val) | CIDEr (val) | CIDEr-D (val) | **F1 (test)** | **CIDEr (test)** |
|--:|--:|--:|--:|--:|--:|
| 42 | 49.61 | 96.72 | 92.5 | 49.49 | 94.38 |
| 123 | 49.46 | 95.84 | 91.7 | 49.10 | 93.03 |
| 2026 | 49.64 | 97.35 | 93.1 | 49.16 | 93.12 |
| 3407 | 49.51 | 96.05 | 91.8 | 49.04 | 92.41 |
| **mean** | **49.55** | **96.49** | **92.3** | **49.20** | **93.24** |

Ghi chú: seed 42 cũ là 4 epoch (F1 50.66 / CIDEr-D 94.4). Re-run 2ep để đồng
nhất. Test-val gap < 0.5 F1, không nhất quán về chiều → không overfit vào val.

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

### A3b. TIER-2 vị trí LoRA (multi_token, r=16, 1ep), theo từng seed

| Config · seed | F1 | CIDEr (nội bộ) | val loss |
|---|--:|--:|--:|
| MLP-only · 42 / 123 / 3407 | 18.68 / 19.74 / 22.30 | 46.7 / 49.9 / 44.0 | 3.41 / 3.26 / 4.46 |
| attn+MLP · 42 / 123 / 3407 | 38.11 / 39.22 / 35.19 | 71.5 / 69.2 / 66.9 | 1.99 / 1.99 / 2.26 |

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

### A5. Đường cong epoch LoRA (multi_token + LoRA r=16 attn, trung bình 3 seed)

| LoRA epoch | F1 | CIDEr (nội bộ) | CIDEr-D |
|--:|--:|--:|--:|
| 1 | 53.17 | 105.59 | 101.70 |
| 3 | 54.67 | 109.60 | 106.80 |
| 5 | *job bị cắt ở cap quota (~4 ep, chưa kịp eval)* | | |

1→3 ep: +1.5 F1 / +5 CIDEr-D. Trend phẳng dần → decoder-LoRA gần trần từ epoch 3.

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

## PHẦN C — Trạng thái thí nghiệm — cập nhật 03:00 UTC 07/09

| Nhóm | Nội dung | Trạng thái |
|---|---|---|
| TIER-1 (19 job) | bridge + dòng âm → 3 seed @ 2 epoch; phủ LoRA 5/5 bridge | ✅ **XONG** — đã ghép Bảng 1–5, A1–A3 |
| Chuẩn hoá epoch | seed-42 4ep → re-run 2ep (bridge + dòng âm) | ✅ **XONG** — residual bad-run đã sửa (36.45 → 45.64) |
| TIER-2 (6 job) | vị trí LoRA: attn / MLP / cả hai | ✅ **XONG** — Bảng 5b, A3b (attn là target duy nhất) |
| LoRA 5 epoch | điểm epoch-curve | ⚠️ bị cắt ở cap quota (~4 ep); A5 giữ 1/3 ep |
| Test-set eval | multi_token 4-seed + 4 bridge s42 | ✅ **XONG** — A1 (test ≈ val); qformer chạy lại |

Còn lại: LoRA test-eval · [camera-ready] human validation thật · viết bản tiếng Anh
· vẽ 3 hình.

---

*Nguồn: results-5bridge.md (kết quả chính) · results-grouped-split.md (ablation,
TIER-2 §4d) · outputs/test_eval/ (test). KTC bootstrap: cần tính lại trên số 2ep.*
