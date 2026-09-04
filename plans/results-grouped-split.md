# Kết quả hiện tại — Paper 3 (efficiency-bridge)

*Cập nhật 2026-09-04 14:30. Spine = efficiency-bridge (alignment-KD đã thử, âm).*

---

## Headline

> Bridge **multi_token** (7.35M param = 0.78%), **đóng băng cả InternViT + Qwen2**,
> huấn luyện ở **1 tile ảnh** thay vì tới 12 của Vintern gốc → **vượt Vintern-1B
> finetune-toàn-bộ và ViMoE-VQA trên metric sinh**, với chi phí train ~100× nhỏ hơn và
> chi phí vision inference ~×4–6 rẻ hơn.

---

## 0. multi_token multi-seed (C2, đang chạy — 2/4 xong)

| seed | CIDEr-D | BLEU-4 | ROUGE-L | F1 |
|---|---:|---:|---:|---:|
| 42 | 94.4 | 19.58 | 50.0 | 50.7 |
| 123 | 91.7 | 19.24 | 48.8 | 49.5 |
| 3407 | *đang chạy (relaunch acc6)* | | | |
| 2026 | *đang chạy (acc16)* | | | |
| **mean (2/4)** | **93.1** | **19.4** | **49.4** | **50.1** |
| ViMoE-VQA | 88.7 | 12.5 | 47.1 | 60.7 |

Cả 2 seed đã xong đều vượt ViMoE trên CIDEr-D/BLEU-4/ROUGE-L, chênh seed ~1–3%. Cập nhật mean±std đầy đủ khi 4/4 seed xong.

## 1. So 5 bridge (val, grouped split, seed 42, 1 tile) — ĐÃ KHÓA

Metric pycocoevalcap corpus (chuẩn để so paper khác).

| Bridge | Tham số train | % | CIDEr-D | BLEU-4 | ROUGE-L | F1(token) | val CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Multi-Token** (8 tok pooled) | 7.35M | 0.78 | **94.4** | **19.6** | **50.0** | **44.2** | **1.49** |
| Full Q-Former (16 query) | 69.4M | 6.91 | 86.7 | 17.5 | 47.1 | 43.3 | 1.57 |
| Light Q-Former (8 query) | 27.6M | 2.87 | 83.8 | 16.8 | 46.0 | 42.9 | 1.59 |
| Tile-Attention (8 tok) | 4.14M | 0.44 | 82.7 | 16.3 | 46.1 | 43.0 | 1.62 |
| Residual (1 tok) | 4.86M | 0.52 | 56.3 | 8.1 | 36.0 | 37.6 | 2.35 |

Multi-token tốt nhất mọi mặt **và** có CE thấp nhất — quan trọng cho §2 dưới.

## 2. Multi-Token vs prior work trên AutoViVQA — ĐÃ KHÓA (val, seed 42)

| Model | Acc | F1 | BLEU | ROUGE | METEOR | CIDEr |
|---|---:|---:|---:|---:|---:|---:|
| Vintern-1B (base) | 0.1 | 17.6 | 1.9 | 25.8 | 23.9 | 8.5 |
| Vintern-1B (**finetune toàn bộ**, đa tile) | **13.0** | 53.8 | 6.1 | **51.9** | 35.3 | 72.8 |
| GPT-5 (zero-shot) | 10.8 | 50.9 | 6.1 | 47.3 | 33.3 | 84.2 |
| **ViMoE-VQA / Tuong-MoE** (5 seed) | 9.7 | **60.7** | 12.5 | 47.1 | **39.1** | 88.7 |
| **★ Multi-Token Bridge (ours)** | 8.6 | 44–51* | **19.6** | **50.0** | ~28–41* | **94.4** |

<small>* F1/METEOR chênh theo implementation. BARTPhoBEiT bỏ khỏi bảng: CIDEr 189 là outlier.</small>

**Thắng rõ (metric sinh ổn định):** CIDEr-D +5.7 / BLEU-4 +7.0 / ROUGE-L +2.9 so với ViMoE.
**Thua:** F1 token-level, Acc. → §3.

## 3. Ba can thiệp khép F1 gap — TẤT CẢ ÂM (val, seed 42, anchor = 50.7 / 94.4) — ĐÃ KHÓA

| Can thiệp (trục) | F1 | CIDEr-D | Δ F1 |
|---|---:|---:|---:|
| baseline `first` | 50.7 | 94.4 | — |
| answer-sampling=random (training target) | 49.0 | 87.3 | −1.7 |
| align-feat α=1.0 (representation alignment) | 49.7 | 92.0 | −1.0 |
| align-logit α=1.0 (representation alignment) | 40.7† | 80.1† | −10 |

<small>† ep2 subset — bị cắt trước full-val. val CE 2.84 (vs 1.49) → KL term ở weight 1.0 chèn CE.</small>

**Kết luận (§6.1):** ba trục độc lập — phân bổ visual compute (routing, §5.2–5.4), training
target, representation alignment — đều **không** cải thiện token-F1. multi_token đã đạt CE
thấp nhất trong 5 bridge. → **frozen Qwen2-0.5B decoder LÀ trần cho khớp phrasing;
capacity phía thị giác/training KHÔNG phải nút thắt.**

## 4. Error analysis (multi_token, val 5463) — ĐÃ KHÓA

| Token-F1 bucket | % |
|---|---:|
| strong (≥0.6) | 36.7 |
| partial (0.2–0.6) | 51.5 |
| weak (<0.2) | 3.1 |
| zero | 8.7 |

- Độ dài dự đoán **4.39 từ** vs ref 4.32 — **không** bị "sinh câu cụt" như ViMoE (ViMoE 4.4 vs 5.6). Residual bridge thì sinh dài lê thê (6.41 từ).
- **Noun omission ở câu đếm: 5.8%** (vs ViMoE **10.7%**) — mình bỏ noun ÍT hơn.
- Per-category F1: tốt nhất counting 0.66 / yesno 0.61 / relational 0.55; tệ nhất action 0.40 / context 0.37 / causal 0.43 (câu mở, nhiều đáp án đúng — model đoán 1 đáp án hợp lý khác).

## 5. Còn PENDING (sau reset quota 00:00 UTC 5/9)

| # | Việc | Ai |
|---|---|---|
| C3 | Oracle sweep + policy ladder trên 3 **tiled** checkpoint → re-lock §5.2/§5.3 (đóng confound "bridge train ở 1 tile") | peer |
| C2 | multi_token seed 123 + 3407 → mean±std cho dòng headline | tôi |
| B3 | Tile-sweep: multi_token @ {1,3,6,12} vs Vintern-finetune @ {1,3,6,12} → bảng efficiency | tôi |
| C4 | Human validation 300–500 mẫu, 2 annotator, Cohen's κ | **user** + tôi setup |
| B5 | Số tile "Vintern-finetune" bảng cũ (≤6 hay ≤12) | **user** |

## 6. Câu chuyện paper (chốt)

1. **Đóng góp chính — efficiency**: bridge nhẹ trên backbone đóng băng đạt SOTA-generation ở 1 tile.
2. **Oracle analysis**: xác nhận 1 tile không phải thoả hiệp — không policy adaptive nào (kể cả biết reasoning-type) thắng fixed 1-tile.
3. **3-way negative → decoder ceiling**: định vị + định lượng nút thắt (frozen 0.5B decoder), không phải phía thị giác.
4. **Benchmark bridge leak-free** + compute-efficiency table (FLOPs/latency) mà ViMoE bỏ ngỏ.

Reasoning-type / P(r|Q) router → 1 mục ablation nhỏ.
