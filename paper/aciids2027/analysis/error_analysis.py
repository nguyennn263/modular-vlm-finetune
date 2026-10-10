"""Answer-length and error statistics for Sec. 4 (qualitative and error analysis).

Validation split, all seeds of every model: Multi-Token 8 (seeds 42/123/2026/3407) and
Multi-Token 8 + LoRA 3 ep (seeds 42/123/3407, outputs/input_diag/), ViBridge-VQA g14-k36, g14-k144
and g14-k36 + LoRA 1 ep (seeds 42/123/3407, outputs/train_gl/). Per-sample F1 is the
best-reference word-overlap F1 of metrics.vqa_metrics.PrecisionRecallF1, i.e. the F1 in
every table. Run from a checkout of exp/eval-input-diagnostic:

    .venv/bin/python <paper>/analysis/error_analysis.py --repo . \
        --out <paper>/analysis/error_analysis.json
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
    from metrics.vqa_metrics import PrecisionRecallF1

    val = [json.loads(l) for l in open(a.repo / "data/splits/val.jsonl")]
    cats = [v["category"] for v in val]
    diag, gl = a.repo / "outputs/input_diag", a.repo / "outputs/train_gl"
    files = {
        "global_only": [diag / f"mt-s{s}-t1-full/out/mt-s{s}-t1-full/text_predictions_epoch_1.json"
                        for s in (42, 123, 2026, 3407)],
        "global_only+lora3ep": [diag / f"l3ep-s{s}-t1-full/out/l3ep-s{s}-t1-full/text_predictions_epoch_1.json"
                                for s in SEEDS],
        "g14-k36": [gl / f"gl-g14-k36-s{s}_eval/out/val/text_predictions_epoch_1.json" for s in SEEDS],
        "g14-k144": [next(gl.glob(f"gl-g14-k144*-s{s}_eval")) / "out/val/text_predictions_epoch_1.json"
                     for s in SEEDS],
        "g14-k36+lora1": [gl / f"gl-g14-k36-lora1-s{s}_eval/out/val/text_predictions_epoch_1.json"
                          for s in SEEDS],
    }

    def words(t: str) -> int:
        return len(t.split())

    def one(path: Path) -> dict:
        s = json.loads(path.read_text())["samples"]
        assert len(s) == len(val) and all(x["question"] == v["question"] for x, v in zip(s, val))
        m = PrecisionRecallF1()
        m.update([x["prediction"] for x in s], [x["ground_truths"] for x in s])
        f1 = m.f1s
        zero_by_cat = {c: 100 * st.mean(f == 0 for f, cc in zip(f1, cats) if cc == c) for c in sorted(set(cats))}
        return {
            "f1": 100 * st.mean(f1),
            "pred_words": st.mean(words(x["prediction"]) for x in s),
            "ref_words": st.mean(st.mean(words(r) for r in x["ground_truths"]) for x in s),
            "zero_f1_pct": 100 * st.mean(f == 0 for f in f1),
            "full_f1_pct": 100 * st.mean(f == 1 for f in f1),
            "partial_pct": 100 * st.mean(0 < f < 1 for f in f1),
            "zero_f1_pct_by_cat": zero_by_cat,
        }

    out = {"provenance": {"split": "val", "files": {k: [str(f.relative_to(a.repo)) for f in v]
                                                     for k, v in files.items()}}}
    for name, fs in files.items():
        runs = [one(f) for f in fs]
        agg = {k: st.mean(r[k] for r in runs) for k in runs[0] if k != "zero_f1_pct_by_cat"}
        agg["zero_f1_pct_by_cat"] = {c: st.mean(r["zero_f1_pct_by_cat"][c] for r in runs)
                                     for c in runs[0]["zero_f1_pct_by_cat"]}
        agg["n_seeds"] = len(runs)
        out[name] = agg
        print(f"{name:22s} F1 {agg['f1']:.2f} pred {agg['pred_words']:.2f}w ref {agg['ref_words']:.2f}w "
              f"zero {agg['zero_f1_pct']:.1f}% partial {agg['partial_pct']:.1f}% full {agg['full_f1_pct']:.1f}%")
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
