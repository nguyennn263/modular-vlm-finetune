"""Validation F1 per reasoning category (Fig. 2b and Sec. 4.6) -> analysis/per_category.json.

Every model is averaged over all of its seeds, so that the figure and the text compare like
with like: global only (Multi-Token, g=8; seeds 42/123/2026/3407, outputs/input_diag/mt-*),
global only + LoRA 3 ep (seeds 42/123/3407, outputs/input_diag/l3ep-*), ViBridge-VQA g14-k36,
g14-k144 and g14-k36 + LoRA 1 ep (seeds 42/123/3407, outputs/train_gl/). Scores use
metrics.vqa_metrics.score_answers on the questions of each category. Run from a checkout of
exp/eval-input-diagnostic:

    .venv/bin/python <paper>/analysis/per_category.py --repo . --out <paper>/analysis/per_category.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

SEEDS = [42, 123, 3407]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.repo))
    from metrics.vqa_metrics import score_answers

    val = [json.loads(l) for l in open(a.repo / "data/splits/val.jsonl")]
    cats = [v["category"] for v in val]
    diag, gl = a.repo / "outputs/input_diag", a.repo / "outputs/train_gl"
    pred = "text_predictions_epoch_1.json"
    files = {
        "global_only": [diag / f"mt-s{s}-t1-full/out/mt-s{s}-t1-full/{pred}" for s in (42, 123, 2026, 3407)],
        "global_only+lora3ep": [diag / f"l3ep-s{s}-t1-full/out/l3ep-s{s}-t1-full/{pred}" for s in SEEDS],
        "g14-k36": [gl / f"gl-g14-k36-s{s}_eval/out/val/{pred}" for s in SEEDS],
        "g14-k144": [next(gl.glob(f"gl-g14-k144*-s{s}_eval")) / f"out/val/{pred}" for s in SEEDS],
        "g14-k36+lora1": [gl / f"gl-g14-k36-lora1-s{s}_eval/out/val/{pred}" for s in SEEDS],
    }

    def by_cat(path: Path) -> dict:
        s = json.loads(path.read_text())["samples"]
        assert len(s) == len(val) and all(x["question"] == v["question"] for x, v in zip(s, val))
        out = {}
        for c in sorted(set(cats)) + ["ALL"]:
            idx = [i for i, x in enumerate(cats) if c in (x, "ALL")]
            avg, _ = score_answers([s[i]["prediction"] for i in idx], [s[i]["ground_truths"] for i in idx])
            out[c] = 100 * avg["f1"]
        return out

    res = {"provenance": {"split": "val", "metric": "F1 (score_answers), x100, mean over seeds",
                          "files": {k: [str(f.relative_to(a.repo)) for f in v] for k, v in files.items()}}}
    for name, fs in files.items():
        runs = [by_cat(f) for f in fs]
        res[name] = {c: {"n": sum(1 for x in cats if c in (x, "ALL")), "f1": st.mean(r[c] for r in runs),
                         "std": st.pstdev(r[c] for r in runs)} for c in runs[0]}
        res[name]["n_seeds"] = len(runs)
        print(name, {c: round(v["f1"], 2) for c, v in res[name].items() if isinstance(v, dict)})
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
