"""Qualitative examples (Fig. 3). Predictions are seed-42 validation outputs with the
whole image at generation (exp/eval-input-diagnostic: outputs/input_diag/mt-s42-t1-full,
l3ep-s42-t1-full; outputs/train_gl/gl-g14-k36-s42_eval/out/val).
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

EX = [  # image, Q (vi), Q (en), GT, Multi-Token 8, + LoRA (3 ep), ours (g=14, k=36), plausible?
    # seed-42 validation predictions with the whole image; (1)-(3) are categories where the
    # local tokens help most (recognition, spatial, action), (4) a causal question where only
    # the decoder-LoRA model is right.
    ("000000487804.jpg", "Chiếc máy tính xách tay này của hãng nào?",
     "Which brand is this laptop?", "Dell", "Apple", "Apple", "Dell", (False, False, True)),
    ("000000231466.jpg", "Chiếc thuyền đang đi qua thành phố nào?",
     "Which city is the boat passing through?", "Thành phố Venice (Venice)",
     "Hà Nội (Hanoi)", "Thành phố New Orleans (New Orleans)", "Thành phố Venice (Venice)",
     (False, False, True)),
    ("000000477067.jpg", "Con voi đang làm hành động gì với chiếc vòi của nó?",
     "What is the elephant doing with its trunk?", "Phun nước (spraying water)",
     "Đang bơi (swimming)", "Đang dùng vòi (using its trunk)", "Đang phun nước (spraying water)",
     (False, False, True)),
    ("000000396159.jpg", "Xe buýt dừng lại để làm gì?", "Why has the bus stopped?",
     "Để đón hành khách (to pick up passengers)", "Đi đến đón khách (going to pick up passengers)",
     "Để đón khách (to pick up passengers)", "Đi dọc đường (going along the road)",
     (True, True, False)),
]

_, img_dir = resolve_dirs()
def crop_border(im):
    """Drop the black letterbox added by the dataset preprocessing."""
    box = im.convert("L").point(lambda v: 255 if v > 12 else 0).getbbox()
    return im.crop(box) if box else im


def wrap(prefix, text, width=34):
    return textwrap.fill(f"{prefix} {text.replace(chr(10), ' ')}", width)


fig, axes = plt.subplots(2, 4, figsize=(6.6, 3.8), gridspec_kw={"height_ratios": [1, 1.6]})
for j, (name, qv, qe, gt, pb, pl, po, (okb, okl, oko)) in enumerate(EX):
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
    t.text(0, 0.80, wrap("Q (EN):", qe), va="top", style="italic", color="#444444")
    t.text(0, 0.60, wrap("GT:", gt), va="top")
    t.text(0, 0.43, wrap("Global:", pb), va="top", color=OK if okb else BAD)
    t.text(0, 0.25, wrap("+LoRA:", pl), va="top", color=OK if okl else BAD)
    t.text(0, 0.07, wrap("Ours:", po), va="top", color=OK if oko else BAD, fontweight="bold")
fig.tight_layout(h_pad=0.2, w_pad=0.6)
fig.savefig(OUT / "fig_qualitative.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_qualitative.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig_qualitative.pdf")
