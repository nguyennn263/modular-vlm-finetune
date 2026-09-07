#!/usr/bin/env python3
"""Generate the three paper figures as vector PDFs into paper/figures/.

    python scripts/figures/make_figures.py

Numbers are the canonical 2-epoch / 3-seed set (see plans/SESSION-STATE.md).
No seaborn; matplotlib only, serif, colourblind-safe, no chartjunk.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[2] / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "figure.dpi": 200,
})

GREY = "#9aa3ae"
BLUE = "#2c5f7c"
RED = "#a0484a"
GREEN = "#3b7256"


# ---------------------------------------------------------------- Fig 1
def fig_bridge_equalizing():
    bridges = ["Residual", "Tile-Attn", "Multi-Token", "Light QF", "Full QF"]
    plain = [81.1, 79.0, 92.3, 83.7, 86.9]          # CIDEr-D, corpus, 3-seed mean @ 2ep
    lora = [100.8, 102.0, 101.7, 103.0, 102.4]      # + LoRA r=16 attn, 1 epoch

    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    x = range(len(bridges))
    w = 0.38
    ax.bar([i - w / 2 for i in x], plain, w, color=GREY, label="bridge only")
    ax.bar([i + w / 2 for i in x], lora, w, color=BLUE, label="+ decoder-attention LoRA")

    # equalisation band
    ax.axhspan(min(lora), max(lora), color=BLUE, alpha=0.10, zorder=0)

    ax.set_xticks(list(x))
    ax.set_xticklabels(bridges, fontsize=7.2, rotation=18, ha="right")
    ax.set_ylabel("corpus CIDEr-D")
    ax.set_ylim(0, 118); ax.set_xlim(-0.9, 4.7)
    ax.axhline(88.7, color=RED, lw=0.8, ls="--")
    ax.text(-0.55, 88.7, "ViMoE\n88.7", color=RED, fontsize=6.3, ha="right", va="center")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=7.5)
    for i, (p, l) in enumerate(zip(plain, lora)):
        ax.text(i - w/2, p + 2, f"{p:.0f}", ha="center", fontsize=6.0, color="#555")
        ax.text(i + w/2, l + 2, f"{l:.0f}", ha="center", fontsize=6.0, color=BLUE)
    fig.tight_layout()
    fig.savefig(OUT / "fig_bridge_equalizing.pdf", bbox_inches="tight"); fig.savefig(OUT / "fig_bridge_equalizing.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2
def fig_tile_collapse():
    tiles = [1, 3, 6]
    f1 = [50.66, 21.05, 22.51]     # multi_token trained at 1 tile, evaluated at n tiles
    vloss = [1.48, 3.35, 3.36]

    fig, ax1 = plt.subplots(figsize=(3.0, 2.2))
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)

    ax1.plot(tiles, f1, "-o", color=BLUE, lw=1.4, ms=4, label="token-F1")
    ax2.plot(tiles, vloss, "--s", color=RED, lw=1.4, ms=4, label="val. loss")

    ax1.set_xlabel("number of tiles at evaluation")
    ax1.set_xticks(tiles)
    ax1.set_ylabel("token-F1", color=BLUE)
    ax1.tick_params(axis="y", colors=BLUE)
    ax1.set_ylim(0, 60)
    ax2.set_ylabel("validation loss", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.set_ylim(0, 4)

    for t, v in zip(tiles, f1):
        ax1.annotate(f"{v:.1f}", (t, v), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=6.5, color=BLUE)
    ax1.text(2.2, 40, "trained at 1 tile;\npool washes out\nextra tokens", fontsize=6.8,
             color="#555", style="italic")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper center",
               bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tile_collapse.pdf", bbox_inches="tight"); fig.savefig(OUT / "fig_tile_collapse.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3
def fig_method():
    fig, ax = plt.subplots(figsize=(6.0, 1.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")

    def box(x, w, label, sub, fc, ec):
        ax.add_patch(FancyBboxPatch((x, 0.75), w, 1.5, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, 1.68, label, ha="center", va="center", fontsize=7.6, weight="bold")
        ax.text(x + w / 2, 1.28, sub, ha="center", va="center", fontsize=6.2, color="#444")

    def arrow(x0, x1, txt=""):
        ax.add_patch(FancyArrowPatch((x0, 1.5), (x1, 1.5), arrowstyle="-|>",
                                     mutation_scale=9, lw=1.0, color="#333"))
        if txt:
            ax.text((x0 + x1) / 2, 1.92, txt, ha="center", fontsize=5.8, color="#666")

    box(0.1, 2.7, "InternViT-300M", "frozen", "#eef0f2", GREY)
    arrow(2.9, 3.7, "T×256 tok")
    box(3.7, 2.2, "Bridge", "trainable · 0.78 %", "#e1eee6", GREEN)
    arrow(6.0, 6.8, "k tok")
    box(6.8, 3.4, "Qwen2-0.5B", "frozen + rank-16\nLoRA on attention · 0.23 %", "#eef0f2", GREY)
    arrow(10.3, 11.1)
    ax.text(11.2, 1.5, "answer", va="center", fontsize=7.6, style="italic")

    ax.text(1.45, 0.4, "image (1 tile, 448²)", ha="center", fontsize=6.2, color="#666")
    ax.text(6.0, 0.32, "the only ~1 % of parameters that trains", ha="center", fontsize=6.2, color=GREEN)
    ax.plot([3.7, 9.0], [0.62, 0.62], color=GREEN, lw=0.8)
    fig.savefig(OUT / "fig_method.pdf", bbox_inches="tight"); fig.savefig(OUT / "fig_method.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_bridge_equalizing()
    fig_tile_collapse()
    fig_method()
    print("wrote:", *(p.name for p in sorted(OUT.glob("*.pdf"))))
