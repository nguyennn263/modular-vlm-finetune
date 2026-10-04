"""Qualitative examples (Fig. 3). Predictions are seed-42 validation outputs with the
whole image at generation (input_diag: mt-s42-t1-full, l3ep-s42-t1-full).
Run from the repo root: .venv/bin/python paper/aciids2027/figures/make_qualitative.py"""
import sys
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.labeled_table import resolve_dirs  # noqa: E402

OUT = Path(__file__).parent
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 5.4, "pdf.fonttype": 42})
OK, BAD = "#2b7a3d", "#a33b3b"

EX = [  # image, Q (vi), Q (en), GT, bridge-only, +LoRA, correctness of (bridge, lora)
    ("000000362189.jpg", "Giỏ chuối được đặt ở đâu?", "Where is the banana basket placed?",
     "Trên bàn (on the table)", "Trên bàn", "Trên bàn", (True, True)),
    ("000000260639.jpg", "Tại sao có người thả diều màu hồng trên bãi biển?",
     "Why is someone flying a pink kite on the beach?",
     "Để vui chơi giải trí (for fun)", "Để tạo dáng (to pose)", "Để vui chơi giải trí", (False, True)),
    ("000000115087.jpg", "Phòng tắm có mấy bồn rửa?", "How many sinks does the bathroom have?",
     "Hai bồn rửa (two sinks)", "Một bồn rửa (one sink)", "Hai bồn rửa", (False, True)),
    ("000000401518.jpg", "Hai con gấu con có màu gì?", "What colour are the two bear cubs?",
     "Màu nâu (brown)", "Màu xanh (blue/green)", "Màu nâu", (False, True)),
]

_, img_dir = resolve_dirs()
def crop_border(im):
    """Drop the black letterbox added by the dataset preprocessing."""
    box = im.convert("L").point(lambda v: 255 if v > 12 else 0).getbbox()
    return im.crop(box) if box else im


def wrap(prefix, text, width=34):
    return textwrap.fill(f"{prefix} {text.replace(chr(10), ' ')}", width)


fig, axes = plt.subplots(2, 4, figsize=(6.6, 3.2), gridspec_kw={"height_ratios": [1, 1.1]})
for j, (name, qv, qe, gt, pb, pl, (okb, okl)) in enumerate(EX):
    ax = axes[0, j]
    im = crop_border(Image.open(img_dir / name).convert("RGB"))
    w, h = im.size
    side = min(w, h * 4 // 3)                 # uniform 4:3 crop so all panels match
    ch = min(h, side * 3 // 4)
    top = (h - ch) // 2                       # centre crop, keeps the beach in panel (2)
    im = im.crop(((w - side) // 2, top, (w - side) // 2 + side, top + ch))
    ax.imshow(im)
    ax.axis("off")
    ax.set_title(f"({j + 1})", fontsize=7)
    t = axes[1, j]
    t.axis("off")
    t.text(0, 1.0, wrap("Q (VI):", qv), va="top", fontweight="bold")
    t.text(0, 0.72, wrap("Q (EN):", qe), va="top", style="italic", color="#444444")
    t.text(0, 0.46, wrap("GT:", gt), va="top")
    t.text(0, 0.30, wrap("Bridge:", pb), va="top", color=OK if okb else BAD)
    t.text(0, 0.14, wrap("+LoRA:", pl), va="top", color=OK if okl else BAD)
fig.tight_layout(h_pad=0.2, w_pad=0.6)
fig.savefig(OUT / "fig_qualitative.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_qualitative.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig_qualitative.pdf")
