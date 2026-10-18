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

fig, (a, b) = plt.subplots(1, 2, figsize=(4.6, 2.35), gridspec_kw={"width_ratios": [1.0, 1.15]})
T = json.loads((OUT.parent / "analysis" / "full_metric_tables.json").read_text())

# (a) ViBridge-VQA (g=14): val F1 against the number of local tokens k (k=0: the 14-token bridge alone)
KS = [0, 1, 9, 36, 144]
pts = [T["t3"]["g14-k0"]] + [D["configs"][f"g14-k{k}"]["val"] for k in KS[1:]]
xs = list(range(len(KS)))
ys = [p["f1"][0] for p in pts]
sd = [p["f1"][1] for p in pts]
a.errorbar(xs, ys, yerr=sd, color=BLUE, lw=1.6, marker="o", ms=4, capsize=1.5, elinewidth=0.6, zorder=3)
a.axhline(MT8_LORA, color=GREY, lw=0.9, ls=(0, (1, 1.5)), zorder=1)
a.text(-0.15, MT8_LORA + 0.25, "Global-only + LoRA", fontsize=5.6, color=GREY, ha="left", va="bottom")
sel = KS.index(36)
a.scatter([xs[sel]], [ys[sel]], s=80, facecolors="none", edgecolors=INK, lw=0.9, zorder=4)
a.annotate("selected\n(50 tokens)", (xs[sel], ys[sel]), textcoords="offset points", xytext=(7, -16),
           fontsize=5.6, color=INK, ha="left")
L = D["g14-k36+lora1"]["val"]["f1"]
a.errorbar([xs[sel]], [L[0]], yerr=[L[1]], color=BLUE, marker="*", ms=9, ls="none", capsize=1.5,
           elinewidth=0.6, zorder=4)
a.annotate("+ LoRA", (xs[sel], L[0]), textcoords="offset points", xytext=(7, -2), fontsize=5.6,
           color=BLUE, ha="left", va="center")
a.set_xticks(xs); a.set_xticklabels([str(k) for k in KS])
a.set_xlabel("local tokens $k$ (with $g=14$)")
a.set_ylabel("F1 (val)")
a.set_ylim(50.5, 59.3); a.set_yticks([51, 53, 55, 57, 59])
a.set_title("(a) ViBridge-VQA vs. local tokens", fontsize=7)

# (b) val F1 by reasoning type: one row per type, the four models as markers on a common line
label = {"yesno": "yes/no"}
MODELS = (("global_only", dict(marker="o", mfc="white", mec=GREY, color=GREY), "Global-only"),
          ("global_only+lora3ep", dict(marker="s", mfc=GREY, mec=GREY, color=GREY), "Global-only + LoRA"),
          ("g14-k36", dict(marker="o", mfc=BLUE, mec=BLUE, color=BLUE), "ViBridge-VQA"),
          ("g14-k36+lora1", dict(marker="*", mfc=BLUE, mec=BLUE, color=BLUE, ms=7), "ViBridge-VQA + LoRA"))
cats = [c for c in P["g14-k36"] if c != "ALL" and isinstance(P["g14-k36"][c], dict)]
order = sorted(cats, key=lambda c: P["g14-k36"][c]["f1"] - P["global_only"][c]["f1"])   # largest gain on top
for y, c in enumerate(order):
    vals = [P[m][c]["f1"] for m, _, _ in MODELS]
    b.plot([min(vals), max(vals)], [y, y], color="#d0d3d8", lw=2.2, zorder=1, solid_capstyle="round")
    for (m, kw, lab), v in zip(MODELS, vals):
        style = {"ms": 4.2} | kw                       # copy: the star keeps its own size on every row
        b.plot([v], [y], ls="none", zorder=3, label=lab if y == 0 else None, **style)
b.set_yticks(range(len(order)))
b.set_yticklabels([f"{label.get(c, c)} ({P['g14-k36'][c]['n']})" for c in order], fontsize=6.2)
b.set_xlabel("F1 (val)")
b.grid(axis="x", color="#eeeeee", lw=0.6); b.set_axisbelow(True)
b.legend(frameon=False, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=2,
         handlelength=1.0, columnspacing=0.8)
b.set_title("(b) F1 by reasoning type", fontsize=7)

fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / "fig_results.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_results.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig_results.pdf")
