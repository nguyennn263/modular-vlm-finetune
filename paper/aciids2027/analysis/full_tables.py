#!/usr/bin/env python3
"""All eight metrics (mean ± std over seeds, ddof=0) for every table of the paper.

    python analysis/full_tables.py DIAG_ROOT MAIN_ROOT > analysis/full_tables.json

DIAG_ROOT: checkout of exp/eval-input-diagnostic (outputs/regen_full, input_diag,
train_qfx, token_sweep, pool_ablation, conv_abstractor, ood_full).
MAIN_ROOT: main checkout (checkpoints/expA/seed42/*/eval_val.json, crop setting).
The k=8 crop row of the token sweep uses the per-seed values of the advisor report
(plans/paper-status-for-advisor.md @3bfdcae, Sect. 3.2); seed 42 matches its eval file.
"""
import json, statistics as st, sys
from pathlib import Path

D, MAIN = Path(sys.argv[1]) / "outputs", Path(sys.argv[2])
M = ["accuracy", "precision", "recall", "f1", "bleu", "rouge_l", "meteor", "cider"]


def find(d: Path, name="eval_val.json"):
    return next(iter(sorted(d.rglob(name))), None) if d.exists() else None


def load(kind, tag, split="val"):
    for t in (tag, tag + "-r2"):
        f = find(D / kind / t, f"eval_{split}.json")
        if f:
            return json.loads(f.read_text())
    raise FileNotFoundError(f"{kind}/{tag} {split}")


def agg(rows, scale=100):
    out = {k: [st.mean(scale * r[k] for r in rows), st.pstdev(scale * r[k] for r in rows)] for k in M}
    if all("loss" in r for r in rows):
        out["ce"] = [st.mean(r["loss"] for r in rows), st.pstdev(r["loss"] for r in rows)]
    out["n"] = len(rows)
    return out


S3 = (42, 123, 3407)
T = {}
# stability (val/test)
T["stab"] = {
    "bridge_val": agg([load("input_diag", f"mt-s{s}-t1-full") for s in (42, 123, 2026, 3407)]),
    "bridge_test": agg([load("regen_full", f"mt-s{s}", "test") for s in (42, 123, 2026, 3407)]),
    "l1_val": agg([load("regen_full", f"l1ep-s{s}") for s in S3]),
    "l1_test": agg([load("regen_full", f"l1ep-s{s}", "test") for s in S3]),
    "l3_val": agg([load("input_diag", f"l3ep-s{s}-t1-full") for s in S3]),
    "l3_test": agg([load("regen_full", f"l3ep-s{s}", "test") for s in S3]),
}
# bridges, plain and + LoRA (val)
qfx = lambda kind: agg([json.loads(find(D / "train_qfx" / f"qfx-{kind}-s{s}_eval").read_text()) for s in S3])
T["bridges"] = {
    "res": [agg([load("regen_full", f"res-s{s}") for s in S3]), agg([load("regen_full", f"l1ep-res-s{s}") for s in S3])],
    "ta": [agg([load("regen_full", f"ta-s{s}") for s in S3]), agg([load("regen_full", "l1ep-ta-s42")])],
    "mt": [T["stab"]["bridge_val"], T["stab"]["l1_val"]],
    "mq": [agg([load("regen_full", f"mq-s{s}") for s in S3]), agg([load("regen_full", f"l1ep-mq-s{s}") for s in S3])],
    "qfx": [qfx("plain"), qfx("lora")],
}
# levers (val)
T["levers"] = {t: agg([load("regen_full", f"rq5-{t}-s{s}") for s in S3]) for t in ("random", "feat", "logit01", "logit1")}
# token sweep (crop setting)
k8 = [[8.20, 50.36, 51.43, 49.61, 15.28, 47.86, 40.05, 96.72], [8.00, 50.10, 51.40, 49.46, 16.05, 47.76, 40.16, 95.84],
      [8.24, 50.20, 51.70, 49.64, 15.91, 47.93, 40.53, 97.35], [8.24, 50.20, 51.47, 49.51, 15.64, 47.80, 40.13, 96.05]]
T["ntok"] = {}
for k in (4, 6, 10, 12, 14, 16, 18, 20):
    T["ntok"][k] = agg([json.loads(find(D / "token_sweep" / f"tok{k}{sfx}").read_text()) for sfx in ("", "-s123", "-s3407")])
T["ntok"][8] = agg([dict(zip(M, r)) for r in k8], scale=1)
T["ntok"][8]["ce"] = [1.49, 0.0]
# pooling (crop setting, seed 42)
one = lambda p: agg([json.loads(p.read_text())])
T["pool"] = {
    "mt": one(MAIN / "checkpoints/expA/seed42/multi_token/eval_val.json"),
    "mean": one(find(D / "pool_ablation/mean")), "max": one(find(D / "pool_ablation/max")),
    "ta": one(MAIN / "checkpoints/expA/seed42/tile_attention/eval_val.json"),
    "conv": one(find(D / "conv_abstractor/m9")),
}
# OOD (already x100)
ood = json.loads((D / "ood_full/table.json").read_text())
T["ood"] = {ds: {m: agg([r[m] for r in ood[ds]["seeds"]], scale=1) for m in ("vintern", "ours_full")}
            for ds in ("vivqa", "vivqax")}
print(json.dumps(T, indent=1))
