"""Fig. 2 of the ACIIDS 2027 paper: (a) from analysis/gl_results.json (analysis/gl_results.py),
(b) from analysis/per_category.json (analysis/per_category.py; every model averaged over all
of its seeds), both computed from the runs on exp/eval-input-diagnostic.
Run: .venv/bin/python paper/aciids2027/figures/make_figures.py

Palette checked with the dataviz validator (blue/orange pass CVD and normal-vision
separation); the Multi-Token references are grey with a second encoding (dashes,
hollow bars) and direct labels, so they never rely on colour alone."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
D = json.loads((OUT.parent / "analysis" / "gl_results.json").read_text())
P = json.loads((OUT.parent / "analysis" / "per_category.json").read_text())
plt.rcParams.update({"font.family": "serif", "font.size": 6.8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
BLUE, ORANGE, GREY, INK = "#2a6fb0", "#c96a1c", "#6b7078", "#333333"
MT8, MT8_LORA = 50.74, 54.78           # Multi-Token 8, val F1: bridge only (4 seeds), + LoRA 3 ep (3 seeds)

fig, (a, b) = plt.subplots(1, 2, figsize=(4.6, 2.2), gridspec_kw={"width_ratios": [1.0, 1.25]})
T = json.loads((OUT.parent / "analysis" / "full_metric_tables.json").read_text())

# (a) choosing g: the bridge alone (Global-only, no local tokens) for g = 4..18
G = [int(g) for g in T["t4"] if int(g) <= 18]   # g=20 has two seeds only; not reported
F = [T["t4"][str(g)]["f1"][0] for g in G]
E = [T["t4"][str(g)]["f1"][1] for g in G]
a.errorbar(G, F, yerr=E, color=GREY, lw=1.4, marker="o", ms=3.5, capsize=1.5, elinewidth=0.6, zorder=2)
i14 = G.index(14)
a.plot([14], [F[i14]], marker="o", ms=6, color=BLUE, zorder=3)
a.annotate("best: $g=14$", (14, F[i14]), textcoords="offset points", xytext=(0, 8), fontsize=6.2,
           color=BLUE, ha="center")
i8 = G.index(8)
a.annotate("Global-only\n($g=8$)", (8, F[i8]), textcoords="offset points", xytext=(6, -16),
           fontsize=5.6, color=GREY, ha="left")
a.set_xticks(G)
a.set_xlabel("global tokens $g$ (no local tokens)")
a.set_ylabel("F1 (val)")
a.set_ylim(48.2, 52.4)
a.set_title("(a) Choosing $g$", fontsize=7)

# (b) choosing (g, k): knee of the accuracy--token front (linear token cost, as in Section 3.5)
C = D["configs"]
pts = {c: (v["tokens"], v["val"]["f1"][0]) for c, v in C.items()}
for g, colour in ((8, ORANGE), (14, BLUE)):
    ks = (1, 9, 36, 144)
    xs = [pts[f"g{g}-k{k}"][0] for k in ks]; ys = [pts[f"g{g}-k{k}"][1] for k in ks]
    b.plot(xs, ys, color=colour, lw=1.2, marker="o", ms=3.5, label=f"$g={g}$", zorder=3)
    if g == 14:
        for x, y, k in zip(xs, ys, ks):
            b.annotate(f"$k$={k}", (x, y), textcoords="offset points", xytext=(5, -9), fontsize=5.6,
                       color=BLUE, ha="left")
lo = min(pts.values()); hi = max(pts.values(), key=lambda p: p[1])
b.plot([lo[0], hi[0]], [lo[1], hi[1]], color=INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
kx, ky = pts["g14-k36"]
chord = lo[1] + (hi[1] - lo[1]) * (kx - lo[0]) / (hi[0] - lo[0])
b.annotate("", xy=(kx, ky), xytext=(kx, chord), arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
b.scatter([kx], [ky], s=70, facecolors="none", edgecolors=INK, lw=0.9, zorder=4)
b.annotate("knee: $g=14$, $k=36$", (kx, ky), textcoords="offset points", xytext=(-6, 9), fontsize=6,
           color=INK, ha="center")
xl = 112; yl = lo[1] + (hi[1] - lo[1]) * (xl - lo[0]) / (hi[0] - lo[0])
b.text(xl, yl - 0.25, "cheapest–best line", fontsize=5.4, color=INK, rotation=15, ha="center", va="top")
b.set_xlabel("visual tokens $g+k$")
b.set_ylabel("F1 (val)")
b.set_xlim(0, 170); b.set_ylim(50.5, 57.3)
b.legend(frameon=False, fontsize=5.8, loc="lower right", handlelength=1.4)
b.set_title("(b) Choosing $(g, k)$", fontsize=7)

fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / "fig_results.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_results.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig_results.pdf")
