# Review — báo cáo cho thầy (01/10) + draft LaTeX cũ (`paper/draft-v1-2026-09/`, trước là `paper/sections/`, 08/09)

Đối chiếu với: bản arXiv AutoViVQA (2603.09689), config Vintern-1B-v3_5, `data/splits/`,
file dự đoán trong `checkpoints/` và `outputs/ood_eval/`, code `src/` + `metrics/`.

## A. Nghiêm trọng — ảnh hưởng số liệu / claim chính

1. **Lệch input lúc sinh câu trả lời (phát hiện 04/10, chưa có kết quả).** Ở 1 tile,
   lúc generate model nhận **ô 448px góc trên-trái** của dynamic tiling, còn lúc train
   thì nhận toàn ảnh resize 336px. Bridge pooled đọc CLS ở 1 tile nhưng đọc mean mọi
   token ở T>1. → Mọi số F1/BLEU/CIDEr của mình, kết quả RQ3 "tile collapse", RQ4 và
   cột "Ours" ở bảng OOD đều phải kiểm lại khi thu 13 job `input_diag.py collect`.
   Trong paper mới đã gắn `\todo` ở §3.1 và RQ3.
2. **BLEU của BARTPhoBEiT — hai bài của nhóm mâu thuẫn nhau:** AutoViVQA (arXiv + bản
   ACIIDS 2026) ghi **0.4329 = 43.29**, ViMoE-VQA (KES 2026) ghi **4.33**. Báo cáo cho
   thầy lấy theo ViMoE. *Quyết định của user (04/10):* baseline lấy theo AutoViVQA →
   paper dùng 43.29, có dấu † "ngoại lai, không dùng để so sánh".
3. **Split của baseline:** báo cáo ghi bảng baseline là "tập validation", nhưng caption
   của AutoViVQA ghi **"test set"**, split **8:1:1**. Draft cũ lại ghi "80/20, không có
   test split công khai". Bản thân paper AutoViVQA tự mâu thuẫn (§3 ghi 80/20, §4 ghi
   8:1:1). Bạn là đồng tác giả AutoViVQA → nên chốt với nhóm rồi viết thống nhất.
   *Cập nhật 04/10:* dữ liệu phát hành trong `data/raw/texts/` đúng là chia 80/20
   (29,661 / 7,416). Paper mới ghi: dữ liệu phát hành 80/20, bảng baseline đo trên 8:1:1.
4. **Cấu hình "Vintern-1B (fine-tuned)" chưa rõ:**
   - báo cáo cho thầy: "theo cookbook", tức đóng băng ViT + MLP, LoRA-16, 6 tile;
   - draft cũ: "fine-tune toàn bộ ViT + projector, ~3M cặp, ≤12 tile".

   Hai nguồn mâu thuẫn và đều là suy đoán. Bản tái lập (`exp/vintern-ft-baseline`)
   bị cắt ở step 1318/1611, chưa có số. → Hỏi nhóm AutoViVQA cấu hình thật. Paper mới
   chỉ ghi "reported, not reproduced", có `\todo`.
5. **Draft cũ còn kết luận RQ6 đã bị lật:** "LoRA trên FFN làm phân kỳ → nút thắt nằm
   ở attention" (abstract, intro, §6, discussion, conclusion). Ngày 14/09 đã xác định
   kết luận này do bug không đóng băng bridge. Khi đóng băng bridge, cả 3 vị trí LoRA
   đều > anchor (53.17 / 51.60 / 50.66). Draft cũ cũng ghi sai "LoRA train trên bridge
   cố định" — recipe thật là co-train bridge + LoRA (9.51M = 1.01%). Paper mới đã viết
   lại đúng theo phát hiện này.
6. **Claim "F1 baseline = 2PR/(P+R)" không đúng cho mọi dòng:**
   - Vintern zero-shot: 2PR/(P+R) = 18.62, nhưng bảng ghi 17.55;
   - Gemini 2.5 Flash: 2PR/(P+R) = **37.05**, nhưng bảng ghi 24.75 (nhiều khả năng là
     typo trong AutoViVQA).

   Paper mới chỉ viết "matches most of the published baseline F1 values".
7. **Bug `metrics/compute_score.py::f1_token`:**
   - `" ".join(str(list))` biến F1 thành F1 trên **tập ký tự** (VD "bàn" vs "bản" →
     F1 0.71 trong khi P=R=0).
   - Đã kiểm: các bảng đã báo cáo **không** dùng hàm này. Trainer dùng F1 mức từ của
     `vqa_metrics`; F1 OOD khớp bản tính lại mức từ trong khoảng ~1 điểm.
   - **Nhưng** `experiments/vintern-ft/score_local.py` gọi `compute_all_data`, nên số
     F1 Vintern-FT tái lập sẽ sai nếu chấm bằng hàm này. Cần sửa trước khi chấm.

2b. **Số của dòng ViMoE-VQA lệch so với bài gốc (KES 2026, Table 2):** báo cáo ghi
   Acc 9.65 / Rec 58.65 / ROUGE 47.07 / CIDEr 88.67; bảng gốc là **9.55 / 58.60 /
   47.12 / 88.76** (riêng phần chữ của ViMoE lại viết CIDEr 88.67). Paper dùng số
   trong bảng gốc.

## B. Sai lệch / không nhất quán trong draft cũ

8. Ghi **Qwen2-0.5B**, nhưng config Vintern-1B-v3_5 là **Qwen2.5-0.5B-Instruct**
   (`fig_method` cũng sai).
9. Mô tả Multi-Token là "mean-pooled patches, 1 anchor + 7 semantic". Code thực tế là
   2 linear head trên **vector toàn ảnh** (CLS/pooler) của InternViT, không pool patch.
10. Bảng split sai: ghi 13,576 / 2,908 / 2,914 ảnh và ~25,900 câu. Thực tế là
    **13,579 / 2,909 / 2,910** ảnh và **25,776** câu train. Tổng 36,707 câu, so với
    37,077 công bố. *Đã giải thích (04/10):* 370 = 273 cặp (ảnh, câu hỏi) trùng +
    97 câu có nhãn category không chuẩn hóa được; 13 ảnh mất theo. Đã ghi vào §4.1.
11. Abstract ghi "band 0.6 điểm", thân bài ghi 0.9 (đúng là 0.88). Chỗ ghi "five
    levers", chỗ ghi "four independent levers".
12. Dòng bridge ghi BLEU 15.47 / Acc 8.20 (số cũ, seed 42). Trung bình 4 seed đúng là
    **15.72 / 8.17**.
13. `fig_bridge_equalizing` dùng CIDEr-D sơ bộ cho Light/Full Q-Former (84/87). Số
    đúng là 81.7/85.4. Paper mới đã vẽ hình mới bằng F1 (có ±std).
14. Bảng CI: điểm ΔF1 lấy trung bình 3 seed, còn CI lấy từ seed 42 → Tile-Attention
    hiện thành "+7.8 [7.8, 9.2]" (điểm nằm sát mép CI). Báo cáo cho thầy cũng trộn
    như vậy (TA+LoRA là seed 42, còn plain là trung bình 3 seed). Cần ghi rõ.
15. Self-check: lấy 120 mẫu, 1 mẫu bị loại vì các đáp án tham chiếu mâu thuẫn nhau
    → 119 mẫu được chấm. Người chấm là **assistant (LLM)** chứ không phải tác giả
    (`outputs/human_validation/report.md`) và không xem ảnh. Không được gọi là
    human validation. Form A/B (300 mẫu) cho 2 người chấm vẫn trống.
16. Draft cũ dài 16 trang. CFP ACIIDS 2027 cho **12–15 trang, tính cả tài liệu tham
    khảo**.
17. So "CIDEr-D corpus" của mình (92.3) với 88.67 của ViMoE là chưa chắc cùng metric:
    88.67 trùng hệt CIDEr in-house trong bảng của ViMoE. Paper mới chỉ so cùng cột
    in-house.
18. Bib dùng placeholder ("Anonymous", "to verify"). ViVQA-X thiếu trích dẫn.

## C. Điểm reviewer có thể hỏi (đã xử lý hoặc ghi chú trong paper mới)

- Sweep k: n=8 **không tối ưu** (k=14: +1.12), và các k mới chạy trên T4 thay vì
  P100 → nói thẳng trong §5.1.
- RQ3 dùng checkpoint 4 epoch (khác chuẩn 2 epoch). So sánh trong cùng checkpoint thì
  ổn, nhưng phải ghi rõ.
- "Full Q-Former ~10×" → đúng là 9.4×.
- Tác giả, email, grant: đang để tạm (Quoc + thầy Tung Le), cần xác nhận.

## Việc còn lại trước khi nộp (hạn 31/10/2026, đã gia hạn)

1. Thu kết quả `input_diag` → cập nhật số + RQ3 (mục A1).
2. Chốt split/cấu hình Vintern-FT với nhóm AutoViVQA (A3, A4).
3. Sửa `f1_token` rồi mới chấm Vintern-FT (A7).
4. Human eval 2 người + κ (tool đã có sẵn).
5. Bib ViMoE / ViVQA-X / AutoViVQA bản proceedings.
6. `\showtodofalse` trong `main.tex` khi build bản nộp.

## D. Bản viết lại 04/10 (theo khuôn AutoViVQA + ViMoE-VQA)

- Cấu trúc: Intro (3 đóng góp) → Related Work → Proposed Method (công thức Eq. 1–3)
  → Experiments: 4.1 Dataset & Metrics, 4.2 Setup, 4.3 Main Results (bảng **test**,
  chia nhóm Fine-tuned / Zero-shot / Ours như ViMoE), 4.4 Evaluation & Efficiency,
  4.5 Ablation, 4.6 theo loại suy luận, 4.7 OOD, 4.8 Qualitative & Error → Conclusion
  + Acknowledgement (grant CNTT 2025-02, cần xác nhận).
- **Đã bỏ:** RQ4 (router/oracle), sweep k, bảng cách gộp patch, đoạn vị trí LoRA,
  self-check chấm bằng LLM, câu "train 1 tile → eval 3 tile".
- **Đã thêm:**
  - Hình 2b: F1 theo loại suy luận. Causal +7.4, relational +4.8, counting +0.1; seed
    42, nhìn toàn ảnh.
  - Hình 3: 4 ví dụ định tính (VI/EN).
  - Quan sát: 3.7% câu hỏi AutoViVQA hỏi về *chú thích* chứ không phải ảnh.
  - Trích dẫn: AViVQA-TranConI và các metric.
- Bản trước được sao lưu ở scratchpad (`aciids2027_backup_v1/`).
- **Còn chờ:** thay toàn bộ số "Ours" bằng kết quả chấm lại nhìn toàn ảnh (26 job
  regen + 12 job OOD của session kia).
