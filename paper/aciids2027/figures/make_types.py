"""Fig. 3 of the ACIIDS 2027 paper, for Global-only, Global-only + LoRA (three epochs), ViBridge-VQA and
ViBridge-VQA + LoRA on validation: (a) F1 by reasoning type, from analysis/per_category.json; (b) share of
answers with F1 = 1 and F1 = 0, from analysis/error_analysis.json (every model averaged over its seeds). Run: .venv/bin/python paper/aciids2027/figures/make_types.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
P = json.loads((OUT.parent / "analysis" / "per_category.json").read_text())
E = json.loads((OUT.parent / "analysis" / "error_analysis.json").read_text())
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

fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.8, 2.45), gridspec_kw=dict(width_ratios=[2.3, 1], wspace=0.55))
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
ax.set_xlabel("F1")
ax.set_xlim(0, 80)
ax.grid(axis="x", color="#eeeeee", lw=0.6); ax.set_axisbelow(True)
ax.set_title("(a) F1 by reasoning type", fontsize=6.8, loc="left", x=-0.3)

# (b) answers that match a reference exactly (F1 = 1) or share no word with any (F1 = 0)
w = 0.2
for i, (m, kw, lab) in enumerate(MODELS):
    vals = [E[m]["full_f1_pct"], E[m]["zero_f1_pct"]]
    xs = [x + (i - 1.5) * w for x in range(2)]
    bx.bar(xs, vals, w, **kw)
    for x, v in zip(xs, vals):
        bx.text(x, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=4.6, rotation=90)
bx.set_xticks(range(2))
bx.set_xticklabels(["exact\n(F1 = 1)", "no overlap\n(F1 = 0)"], fontsize=6.2)
bx.set_ylabel("answers (%)")
bx.set_ylim(0, 20); bx.set_yticks([0, 5, 10, 15, 20])
bx.grid(axis="y", color="#eeeeee", lw=0.6); bx.set_axisbelow(True)
bx.set_title("(b) Answer overlap", fontsize=6.8, loc="left")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=5.8, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.13), handlelength=1.2, columnspacing=1.0)
fig.savefig(OUT / "fig_types.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_types.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig_types.pdf")
