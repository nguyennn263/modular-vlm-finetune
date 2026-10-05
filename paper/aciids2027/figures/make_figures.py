"""Figures for the ACIIDS 2027 paper (Fig. 2). Numbers hard-coded from
plans/paper-status-for-advisor.md (01/10/2026); update here if they change.
Run: .venv/bin/python paper/aciids2027/figures/make_figures.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
GREY, BLUE, RED = "#9aa1ab", "#2b5f7d", "#a33b3b"

fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})

# (a) decoder LoRA equalises the five bridges (in-house token-F1, val)
names = ["Residual", "Tile-\nAttention", "Multi-\nToken", "Light\nQ-Former", "Full\nQ-Former"]
# whole-image re-evaluation (plans/paper-tables-full-image.md, 2026-10-05)
plain = [46.02, 45.14, 50.74, 47.06, 46.84]
plain_sd = [0.36, 0.92, 0.17, 0.66, 0.47]
lora = [52.69, 52.82, 53.47, 53.21, 53.29]
lora_sd = [0.14, 0.0, 0.17, 0.14, 0.29]
x = range(len(names)); w = 0.38
a.bar([i - w/2 for i in x], plain, w, yerr=plain_sd, color=GREY, label="bridge only", capsize=1.5, error_kw={"lw": 0.6})
a.bar([i + w/2 for i in x], lora, w, yerr=lora_sd, color=BLUE, label="+ decoder LoRA (1 ep)", capsize=1.5, error_kw={"lw": 0.6})
for i in x:
    a.text(i + w/2, lora[i] + 0.25, f"+{lora[i]-plain[i]+1e-9:.1f}", ha="center", va="bottom", fontsize=6.5, color=BLUE)
a.set_xticks(list(x)); a.set_xticklabels(names, fontsize=6.5)
a.set_ylim(42, 58); a.set_ylabel("token-F1 (val)")
a.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
a.set_title("(a) Bridge spread 45.1–50.7 → 52.7–53.5", fontsize=8)

# (b) F1 by reasoning type, bridge only vs + LoRA (3 ep); seed 42, whole image at generation
cats = ["relational", "recognition", "spatial", "causal", "counting", "action", "context", "yes/no"]
ncat = [1662, 1016, 802, 692, 689, 391, 145, 66]
pc = [55.71, 45.32, 47.57, 41.26, 66.57, 44.38, 35.10, 64.41]
lc = [60.49, 49.65, 51.95, 48.63, 66.69, 47.21, 37.26, 69.40]
y = range(len(cats))[::-1]
b.barh([k + 0.2 for k in y], pc, 0.4, color=GREY, label="bridge only")
b.barh([k - 0.2 for k in y], lc, 0.4, color=BLUE, label="+ LoRA (3 ep)")
for k, p_, l_ in zip(y, pc, lc):
    b.text(max(p_, l_) + 0.6, k, f"{l_ - p_:+.1f}", va="center", fontsize=6.5, color=BLUE)
b.set_yticks(list(y)); b.set_yticklabels([f"{c} ({n})" for c, n in zip(cats, ncat)], fontsize=6.5)
b.set_xlim(30, 76); b.set_xlabel("token-F1 (val)")
b.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2)
b.set_title("(b) Gain by reasoning type", fontsize=8)

fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / "fig_results.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_results.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig_results.pdf")
