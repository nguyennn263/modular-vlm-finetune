"""Qualitative examples (Fig. 3). Predictions are seed-42 validation outputs with the
whole image at generation (exp/eval-input-diagnostic: outputs/input_diag/mt-s42-t1-full,
l3ep-s42-t1-full; outputs/train_gl/gl-g14-k36-s42_eval/out/val and
gl-g14-k36-lora1-s42_eval/out/val). Val indices: 2909, 5267, 248, 2159.
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
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 6.6, "pdf.fonttype": 42})
OK, BAD = "#2b7a3d", "#a33b3b"

EX = [  # image, Q (vi), Q (en), GT, global only, + LoRA (3 ep), ours (g=14, k=36), ours + LoRA (1 ep), correct?
    # seed-42 validation predictions with the whole image (images checked by eye). (1)-(3):
    # recognition, spatial and action questions where only the models with local tokens are
    # right; (4) a causal question answered from a local detail (the kites).
    ("000000346583.jpg", "Vật gì đang đậu trên gương chiếu hậu của xe?",
     "What is perched on the car's side mirror?", "Một con chim (a bird)",
     "Một chiếc xe máy (a motorbike)", "Một chiếc xe hơi (a car)", "Một con chim (a bird)",
     "Một con chim (a bird)", (False, False, True, True)),
    ("000000517830.jpg", "Biển báo dừng nằm cạnh vật thể nào trên góc phố?",
     "What is the stop sign next to on the corner?", "Cột đèn (a lamp post)",
     "Gần một chiếc xe tải (near a truck)", "Một chiếc xe hơi (a car)",
     "Một cột đèn (a lamp post)", "Một cột đèn (a lamp post)", (False, False, True, True)),
    ("000000451440.jpg", "Con mèo đang làm gì tại bàn ăn?", "What is the cat doing at the table?",
     "Ngồi trên ghế (sitting on a chair)", "Đang ăn (eating)", "Đang ăn (eating)",
     "Đang ngồi trên ghế (sitting on a chair)", "Đang ngồi trên ghế (sitting on a chair)",
     (False, False, True, True)),
    ("000000343307.jpg", "Tại sao có một đám đông lớn người dân trong công viên?",
     "Why is there a large crowd in the park?", "Để thả diều (to fly kites)",
     "Để chụp ảnh (to take photos)", "Để xem biểu diễn (to watch a show)",
     "Để thả diều (to fly kites)", "Để thả diều (to fly kites)", (False, False, True, True)),
]

_, img_dir = resolve_dirs()
def crop_border(im):
    """Drop the black letterbox added by the dataset preprocessing."""
    box = im.convert("L").point(lambda v: 255 if v > 12 else 0).getbbox()
    return im.crop(box) if box else im


def wrap(prefix, text, width=24):
    return textwrap.fill(f"{prefix} {text.replace(chr(10), ' ')}", width)


fig, axes = plt.subplots(2, 4, figsize=(4.82, 3.2), gridspec_kw={"height_ratios": [0.85, 1.85]})
for j, (name, qv, qe, gt, pb, pl, po, pol, oks) in enumerate(EX):
    ax = axes[0, j]
    im = crop_border(Image.open(img_dir / name).convert("RGB"))
    w, h = im.size
    side = min(w, h * 4 // 3)                 # uniform 4:3 crop so all panels match
    ch = min(h, side * 3 // 4)
    top = (h - ch) // 2
    im = im.crop(((w - side) // 2, top, (w - side) // 2 + side, top + ch))
    ax.imshow(im)
    ax.axis("off")
    ax.set_title(f"({j + 1})", fontsize=7, pad=2)
    t = axes[1, j]
    t.axis("off")
    LH, GAP = 0.054, 0.025                    # line height and block gap (axes units)
    y = 1.0
    def put(text, gap=GAP, x=0, **kw):
        global y
        t.text(x, y, text, va="top", **kw)
        y -= LH * (text.count("\n") + 1) + gap
    put(wrap("Q (VI):", qv), fontweight="bold", gap=0.026)
    put(wrap("Q (EN):", qe), style="italic", color="#444444")
    put(wrap("GT:", gt), gap=0.04)
    rows = (("Global:", pb), ("+LoRA:", pl), ("Ours:", po), ("Ours+LoRA:", pol))
    for (lab, pred), ok in zip(rows, oks):
        # tick / cross as a second, colour-independent cue (black-and-white print)
        t.text(-0.1, y, "\u2713" if ok else "\u2717", va="top", fontsize=6.8, family="DejaVu Sans",
               color=OK if ok else BAD, fontweight="bold")
        put(wrap(lab, pred), color=OK if ok else BAD, fontweight="bold" if lab.startswith("Ours") else "normal")
fig.subplots_adjust(left=0.035, right=0.995, top=0.95, bottom=0.0, wspace=0.16, hspace=0.05)
fig.savefig(OUT / "fig_qualitative.pdf", bbox_inches="tight", dpi=600)  # photos embedded sharply
fig.savefig(OUT / "fig_qualitative.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig_qualitative.pdf")
