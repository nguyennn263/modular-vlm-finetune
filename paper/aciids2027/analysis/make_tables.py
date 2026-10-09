#!/usr/bin/env python3
"""LaTeX tables with all eight metrics from analysis/full_tables.json -> scratch file."""
import json, sys
T = json.load(open(sys.argv[1]))
M = ["accuracy", "precision", "recall", "f1", "bleu", "rouge_l", "meteor", "cider"]
HEAD = r"Acc & Prec & Rec & F1 & BLEU & ROUGE-L & METEOR & CIDEr"


def cells(rows, std=True, ce=True, bold=True):
    """rows: list of agg dicts -> list of cell lists; best value per column in bold."""
    keys = M + (["ce"] if ce else [])
    best = {k: (min if k == "ce" else max)(r[k][0] for r in rows) for k in keys} if bold and len(rows) > 1 else {}
    out = []
    for r in rows:
        c = []
        for k in keys:
            m, s = r[k]
            v = f"{m:.3f}" if k == "ce" else f"{m:.2f}"
            if k in best and round(m, 3 if k == "ce" else 2) == round(best[k], 3 if k == "ce" else 2):
                v = rf"\textbf{{{v}}}"
            if std and r["n"] > 1 and k != "ce":
                sd = f"{s:.2f}"
                v = rf"\pms{{{v}}}{{{sd[1:] if sd.startswith('0.') else sd}}}"
            c.append(v)
        out.append(c)
    return out


def table(label, caption, colspec, head, body, size=r"\scriptsize"):
    return "\n".join([r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      size, r"\setlength{\tabcolsep}{2.5pt}", r"\resizebox{\textwidth}{!}{%",
                      rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", head + r" \\", r"\midrule", *body,
                      r"\bottomrule", r"\end{tabular}}", r"\end{table}"])


O = {}
# ---- stability
S = T["stab"]
names = [("bridge only", "bridge"), ("+ LoRA, 1 ep.", "l1"), ("+ LoRA, 3 ep.", "l3")]
body = []
for split in ("val", "test"):
    rows = [S[f"{k}_{split}"] for _, k in names]
    for (n, _), c in zip(names, cells(rows)):
        body.append((rf"\multirow{{3}}{{*}}{{{split}}}" if n == names[0][0] else "") + f" & {n} & " + " & ".join(c) + r" \\")
    if split == "val":
        body.append(r"\midrule")
O["tab:stability"] = table("tab:stability",
    r"ViBridge-VQA on the validation and test splits (mean $\pm$ std over seeds: bridge only 4 seeds, "
    r"+ LoRA 3 seeds; $\times100$). CE is the answer cross-entropy on the split. Best value per split in bold.",
    "llrrrrrrrrr", r"Split & Config. & " + HEAD + " & CE", body)
# ---- bridges
B = T["bridges"]
BR = [("Residual", "res", 1, "4.86M"), ("Tile-Attn.", "ta", 8, "4.14M"), ("Multi-Token", "mt", 8, "7.35M"),
      ("Light Q-F.", "mq", 8, "27.6M"), ("Full Q-F.", "qfx", 16, "69.4M")]
body = []
for j, grp in enumerate(("Bridge only (2 epochs)", r"+ LoRA (joint, 1 epoch)")):
    body.append(rf"\multicolumn{{13}}{{l}}{{\emph{{{grp}}}}} \\")
    rows = [B[t][j] for _, t, _, _ in BR]
    for (n, t, k, p), c, r in zip(BR, cells(rows), rows):
        d = f"${r['f1'][0] - B[t][0]['f1'][0]:+.2f}$" if j else "--"
        nm = n + (r"$^\ast$" if j and t == "ta" else "")
        body.append(f"{nm} & {k} & {p} & " + " & ".join(c) + f" & {d}" + r" \\")
    if j == 0:
        body.append(r"\midrule")
O["tab:bridges"] = table("tab:bridges",
    r"Bridge architectures before and after decoder LoRA (validation, $\times100$; mean $\pm$ std over 3 seeds, "
    r"Multi-Token bridge only 4 seeds; $^\ast$seed 42 only). Params: bridge parameters (LoRA adds 2.16M). "
    r"$\Delta$F1: gain of + LoRA over bridge only. Best value per group in bold.",
    "lrrrrrrrrrrrr", r"Bridge & $k$ & Params & " + HEAD + r" & CE & $\Delta$F1", body)
# ---- levers
base = B["mt"][0]
LV = [("Reference: bridge only", base, False),
      ("Random reference answer", T["levers"]["random"], False),
      ("Feature distillation", T["levers"]["feat"], False),
      (r"Logit distillation, $\lambda{=}0.1$", T["levers"]["logit01"], False),
      (r"Logit distillation, $\lambda{=}1$", T["levers"]["logit1"], False),
      (r"Light Q-Former ($3.8\times$)", B["mq"][0], False),
      (r"Full Q-Former ($9.4\times$)", B["qfx"][0], False),
      ("Decoder LoRA, 1 epoch", S["l1_val"], True),
      ("Decoder LoRA, 3 epochs", S["l3_val"], True)]
body = []
for i, ((n, r, hi), c) in enumerate(zip(LV, cells([r for _, r, _ in LV]))):
    d = "--" if i == 0 else f"${r['f1'][0] - base['f1'][0]:+.2f}$"
    if hi:
        n, d = rf"\textbf{{{n}}}", rf"$\mathbf{{{r['f1'][0] - base['f1'][0]:+.2f}}}$"
    body.append(f"{n} & " + " & ".join(c) + f" & {d}" + r" \\")
    if i in (0, 4, 6):
        body.append(r"\midrule")
O["tab:levers"] = table("tab:levers",
    r"Interventions on the Multi-Token bridge (validation, $\times100$, mean $\pm$ std over 3 seeds; reference: "
    r"4 seeds). $\Delta$F1 is relative to the bridge-only reference.",
    "lrrrrrrrrrr", r"Intervention & " + HEAD + r" & CE & $\Delta$F1", body)
# ---- design: token count + pooling
K = [4, 6, 8, 10, 12, 14, 16, 18, 20]
rows = [T["ntok"][str(k)] for k in K]
body = [r"\multicolumn{11}{l}{\emph{(a) Number of output tokens $k$ (Multi-Token bridge)}} \\"]
for k, c in zip(K, cells(rows)):
    nm = rf"\textbf{{$k={k}$ (ours)}}" if k == 8 else f"$k={k}$"
    body.append(f"{nm} & {k * 0.9184:.2f}M & " + " & ".join(c) + r" \\")
body += [r"\midrule", r"\multicolumn{11}{l}{\emph{(b) Pooling operator over the image (seed 42)}} \\"]
PL = [(r"\textbf{Multi-Token (ours)}", "7.35M", "mt"), ("Patch-Pool, mean", "0.92M", "mean"),
      ("Patch-Pool, max", "0.92M", "max"), ("Tile-Attention", "4.14M", "ta"),
      ("Conv-Abstractor", "19.87M", "conv")]
for (n, p, t_), c in zip(PL, cells([T["pool"][t_] for *_, t_ in PL])):
    body.append(f"{n} & {p} & " + " & ".join(c) + r" \\")
O["tab:design"] = table("tab:design",
    r"Bridge design (bridge only, validation, $\times100$). (a) Mean $\pm$ std over 3 seeds ($k=8$: 4 seeds). "
    r"(b) Multi-Token reads the global \texttt{[CLS]} vector; the other bridges pool the patch tokens with a fixed mean or max, learned attention, or convolution and pooling; the Conv-Abstractor uses $k=9$ because it needs a square grid. Both studies use an earlier evaluation "
    r"setting in which generation reads a $448\times448$ crop; all rows share this setting. Best value per panel in bold.",
    "lrrrrrrrrrr", r"Bridge & Params & " + HEAD + " & CE", body)
# ---- OOD
body = []
for ds, nm in (("vivqa", r"ViVQA~\cite{tran2021vivqa}"), ("vivqax", r"ViVQA-X~\cite{vivqax2025}")):
    rows = [T["ood"][ds]["vintern"], T["ood"][ds]["ours_full"]]
    for j, (mn, c) in enumerate(zip(("Vintern-1B, zero-shot", "ViBridge-VQA (ours)"), cells(rows, ce=False))):
        body.append((rf"\multirow{{2}}{{*}}{{{nm}}}" if j == 0 else "") + f" & {mn} & " + " & ".join(c) + r" \\")
    if ds == "vivqa":
        body.append(r"\midrule")
O["tab:ood"] = table("tab:ood",
    r"Generalisation to two general-domain Vietnamese VQA datasets not seen in training ($\times100$; Vintern-1B uses 6 tiles; about "
    r"1{,}000 sampled test questions per seed, 505--519 for ViVQA; mean $\pm$ std over 3 sampling seeds).",
    "llrrrrrrrr", r"Dataset & Model & " + HEAD, body)
json.dump(O, open(sys.argv[2], "w"), indent=1)
