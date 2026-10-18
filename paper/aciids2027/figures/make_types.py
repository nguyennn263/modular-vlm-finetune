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
MODELS = (("global_only", dict(facecolor="white", edgecolor=GREY, lw=0.8), "Global-only"),
          ("global_only+lora3ep", dict(color="#a9adb3"), "Global-only + LoRA"),
          ("g14-k36", dict(color="#8fb7de"), "ViBridge-VQA"),
          ("g14-k36+lora1", dict(color=BLUE), "ViBridge-VQA + LoRA"))
label = {"yesno": "yes/no"}
cats = [c for c in P["g14-k36"] if c != "ALL" and isinstance(P["g14-k36"][c], dict)]
order = sorted(cats, key=lambda c: P["g14-k36"][c]["f1"] - P["global_only"][c]["f1"])   # largest gain on top

fig, ax = plt.subplots(figsize=(4.6, 2.35))
h = 0.2
for i, (m, kw, lab) in enumerate(MODELS):
    ax.barh([y + (1.5 - i) * h for y in range(len(order))], [P[m][c]["f1"] for c in order], h,
            label=lab, **kw)
GX = 86                               # a separate column, outside the bars: gain of ViBridge-VQA over Global-only
for y, c in enumerate(order):
    gain = P["g14-k36"][c]["f1"] - P["global_only"][c]["f1"]
    ax.text(GX, y, f"{gain:+.1f}", va="center", ha="center", fontsize=6, color=BLUE, clip_on=False)
ax.text(GX, len(order) - 0.35, "gain vs.\nGlobal-only", va="bottom", ha="center", fontsize=5.6,
        color=BLUE, clip_on=False)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{label.get(c, c)} ({P['g14-k36'][c]['n']})" for c in order], fontsize=6.2)
ax.set_xlabel("F1 (val)")
ax.set_xlim(0, 80)
ax.grid(axis="x", color="#eeeeee", lw=0.6); ax.set_axisbelow(True)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=5.8, loc="center left",
          bbox_to_anchor=(1.17, 0.5), handlelength=1.2)
fig.tight_layout()
fig.savefig(OUT / "fig_types.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_types.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig_types.pdf")
