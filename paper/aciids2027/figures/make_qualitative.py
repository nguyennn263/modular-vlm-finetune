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

EX = [  # image, Q (vi), Q (en), GT, bridge-only, +LoRA, plausibility of (bridge, lora)
    ("000000022929.jpg", "Em bé đang làm gì với con gấu bông?",
     "What is the baby doing with the teddy bear?",
     "Ôm gấu (hugging the bear)", "Đang chơi với nó (playing with it)",
     "Đang ôm con gấu bông (hugging the teddy bear)", (True, True)),
    ("000000124949.jpg", "Mục đích của việc hai người này ngồi dưới ô là gì?",
     "Why are these two people sitting under umbrellas?",
     "Để che nắng (to shade from the sun)", "Để nghỉ ngơi (to rest)",
     "Để tránh nắng (to avoid the sun)", (True, True)),
    ("000000565098.jpg", "Những chiếc máy bay chiến đấu này đang chuẩn bị cho hành động gì?",
     "What are these fighter jets preparing for?",
     "Có thể chuẩn bị cất cánh (possibly preparing to take off)",
     "Đang chuẩn bị bay (preparing to fly)", "Đang chuẩn bị bay (preparing to fly)", (True, True)),
    ("000000207058.jpg", "Tại sao năm người này lại cười khi tạo dáng cùng nhau?",
     "Why are these five people smiling while posing together?",
     "Họ đang vui vẻ (they are having fun)", "Vì họ rất vui vẻ (because they are very happy)",
     "Vì họ rất vui vẻ (because they are very happy)", (True, True)),
]

_, img_dir = resolve_dirs()
def crop_border(im):
    """Drop the black letterbox added by the dataset preprocessing."""
    box = im.convert("L").point(lambda v: 255 if v > 12 else 0).getbbox()
    return im.crop(box) if box else im


def wrap(prefix, text, width=34):
    return textwrap.fill(f"{prefix} {text.replace(chr(10), ' ')}", width)


fig, axes = plt.subplots(2, 4, figsize=(6.6, 3.5), gridspec_kw={"height_ratios": [1, 1.35]})
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
    t.text(0, 0.76, wrap("Q (EN):", qe), va="top", style="italic", color="#444444")
    t.text(0, 0.50, wrap("GT:", gt), va="top")
    t.text(0, 0.32, wrap("Bridge:", pb), va="top", color=OK if okb else BAD)
    t.text(0, 0.12, wrap("+LoRA:", pl), va="top", color=OK if okl else BAD)
fig.tight_layout(h_pad=0.2, w_pad=0.6)
fig.savefig(OUT / "fig_qualitative.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_qualitative.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig_qualitative.pdf")
