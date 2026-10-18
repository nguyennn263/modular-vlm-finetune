"""Fig. 3 of the ACIIDS 2027 paper: validation F1 by reasoning type for Global-only, Global-only + LoRA
(three epochs), ViBridge-VQA and ViBridge-VQA + LoRA, from analysis/per_category.json (every model
averaged over all of its seeds). Run: .venv/bin/python paper/aciids2027/figures/make_types.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
P = json.loads((OUT.parent / "analysis" / "per_category.json").read_text())
plt.rcParams.update({"font.family": "serif", "font.size": 6.8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
BLUE, GREY = "#2a6fb0", "#6b7078"
MODELS = (("global_only", dict(marker="o", mfc="white", mec=GREY, ms=4.5), "Global-only"),
          ("global_only+lora3ep", dict(marker="s", mfc=GREY, mec=GREY, ms=4.2), "Global-only + LoRA"),
          ("g14-k36", dict(marker="o", mfc=BLUE, mec=BLUE, ms=4.8), "ViBridge-VQA"),
          ("g14-k36+lora1", dict(marker="*", mfc=BLUE, mec=BLUE, ms=8), "ViBridge-VQA + LoRA"))
label = {"yesno": "yes/no"}
cats = [c for c in P["g14-k36"] if c != "ALL" and isinstance(P["g14-k36"][c], dict)]
order = sorted(cats, key=lambda c: P["g14-k36"][c]["f1"] - P["global_only"][c]["f1"])   # largest gain on top

fig, ax = plt.subplots(figsize=(4.6, 1.75))
for y, c in enumerate(order):
    vals = [P[m][c]["f1"] for m, _, _ in MODELS]
    ax.plot([min(vals), max(vals)], [y, y], color="#d0d3d8", lw=2.4, zorder=1, solid_capstyle="round")
    for (m, kw, lab), v in zip(MODELS, vals):
        ax.plot([v], [y], ls="none", zorder=3, label=lab if y == 0 else None, **kw)
    gain = P["g14-k36"][c]["f1"] - P["global_only"][c]["f1"]
    ax.text(max(vals) + 0.8, y, f"{gain:+.1f}", va="center", fontsize=5.8, color=BLUE)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{label.get(c, c)} ({P['g14-k36'][c]['n']})" for c in order], fontsize=6.2)
ax.set_xlabel("F1 (val)")
ax.set_xlim(35, 76)
ax.grid(axis="x", color="#eeeeee", lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=5.8, loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.0)
fig.tight_layout()
fig.savefig(OUT / "fig_types.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_types.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig_types.pdf")
