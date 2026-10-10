"""Qualitative examples (Fig. 3). Predictions are seed-42 validation outputs with the
whole image at generation (exp/eval-input-diagnostic: outputs/input_diag/mt-s42-t1-full,
l3ep-s42-t1-full; outputs/train_gl/gl-g14-k36-s42_eval/out/val and
gl-g14-k36-lora1-s42_eval/out/val). Val indices: 2109, 50, 2133, 2241.
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

EX = [  # image, Q (vi), Q (en), GT, ours (g=14, k=36), ours + LoRA (1 ep)
    # seed-42 validation predictions (images checked by eye): ViBridge-VQA already finds the
    # relevant content, and the decoder LoRA makes the vague answer precise (action, causal and
    # yes/no questions; no relational questions by request).
    ("000000139944.jpg", "Chiếc xe đạp được dùng để làm gì với những chùm chuối?",
     "What is the bicycle used for with the bunches of bananas?",
     "Để chở những chùm chuối (to carry the bananas)", "Để đựng chuối (to hold the bananas)",
     "Để vận chuyển chuối (to transport the bananas)"),
    ("000000121792.jpg", "Tại sao những người phụ nữ này lại đội mũ bảo hiểm?",
     "Why are these women wearing helmets?", "Để bảo vệ đầu (to protect their heads)",
     "Để an toàn khi chạy (to be safe while running)", "Để bảo vệ đầu (to protect their heads)"),
    ("000000423962.jpg", "Tại sao người lớn lại nhìn vào khi trẻ em chơi bóng đá?",
     "Why are the adults watching while the children play football?",
     "Để theo dõi trận đấu (to follow the match)", "Để theo dõi tình huống (to follow the situation)",
     "Để theo dõi trận đấu (to follow the match)"),
    ("000000540656.jpg", "Cái máng ăn cho chim có treo trên cành cây không?",
     "Is the bird feeder hanging from a branch?", "Có, nó treo trên cành (yes, it hangs from a branch)",
     "Có (yes)", "Có, nó được treo (yes, it is hanging)"),
]

_, img_dir = resolve_dirs()
def crop_border(im):
    """Drop the black letterbox added by the dataset preprocessing."""
    box = im.convert("L").point(lambda v: 255 if v > 12 else 0).getbbox()
    return im.crop(box) if box else im


def wrap(prefix, text, width=24):
    return textwrap.fill(f"{prefix} {text.replace(chr(10), ' ')}", width)


fig, axes = plt.subplots(2, 4, figsize=(4.82, 2.45), gridspec_kw={"height_ratios": [0.85, 1.1]})
for j, (name, qv, qe, gt, po, pol) in enumerate(EX):
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
    LH, GAP = 0.091, 0.042                    # line height and block gap (axes units)
    y = 1.0
    def put(text, gap=GAP, x=0, **kw):
        global y
        t.text(x, y, text, va="top", **kw)
        y -= LH * (text.count("\n") + 1) + gap
    put(wrap("Q (VI):", qv), fontweight="bold", gap=0.044)
    put(wrap("Q (EN):", qe), style="italic", color="#444444")
    put(wrap("GT:", gt), gap=0.07)
    put(wrap("Ours:", po), color="#333333")
    put(wrap("Ours+LoRA:", pol), color=OK, fontweight="bold")
fig.subplots_adjust(left=0.035, right=0.995, top=0.95, bottom=0.0, wspace=0.16, hspace=0.05)
fig.savefig(OUT / "fig_qualitative.pdf", bbox_inches="tight", dpi=600)  # photos embedded sharply
fig.savefig(OUT / "fig_qualitative.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig_qualitative.pdf")
