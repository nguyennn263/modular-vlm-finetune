# Draft v1 (07–08/09/2026) — SUPERSEDED

Bản draft LaTeX đầu tiên (16 trang). **Không dùng để nộp.** Bản nộp ACIIDS 2027 là
`paper/aciids2027/` (ViBridge-VQA).

Các lỗi đã biết của bản này được liệt kê trong `paper/aciids2027/REVIEW.md` (mục B), ví dụ:
- ghi Qwen2-0.5B, đúng là Qwen2.5-0.5B-Instruct;
- mô tả Multi-Token là mean-pool patch, đúng là 2 linear head trên vector toàn ảnh;
- kết luận RQ6 "FFN-LoRA phân kỳ" đã bị lật ngày 14/09 (do bug không đóng băng bridge);
- số split sai;
- dài 16 trang, vượt giới hạn 15.

Build: `latexmk -pdf main.tex` trong thư mục này.
Hình: `python scripts/figures/make_figures.py` (đã trỏ về `paper/draft-v1-2026-09/figures`).
