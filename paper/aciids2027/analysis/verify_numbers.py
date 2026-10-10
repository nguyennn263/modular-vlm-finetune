"""Re-derive every ViBridge-VQA number in the paper from the raw prediction files.

For each run (outputs/train_gl/*_eval/out/{val,test}/) it re-scores
text_predictions_epoch_1.json with metrics.vqa_metrics.score_answers, compares the
result with the eval_{split}.json written by the Kaggle eval and with
analysis/gl_results.json (mean, population std over seeds), and reports the largest
absolute difference per metric. Run from a checkout of exp/eval-input-diagnostic:

    .venv/bin/python <paper>/analysis/verify_numbers.py --repo . \
        --gl <paper>/analysis/gl_results.json --out <paper>/analysis/verify_numbers.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

M = ["accuracy", "precision", "recall", "f1", "bleu", "rouge_l", "meteor", "cider"]
SEEDS = [42, 123, 3407]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--gl", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.repo))
    from metrics.vqa_metrics import score_answers

    gl = json.loads(a.gl.read_text())
    runs = a.repo / "outputs/train_gl"
    cfgs = {n: (c["g"], c["k"], False) for n, c in gl["configs"].items()}
    cfgs["g14-k36+lora1"] = (14, 36, True)
    report, worst_eval, worst_agg = {}, 0.0, 0.0
    for name, (g, k, lora) in cfgs.items():
        ref = gl["g14-k36+lora1"] if lora else gl["configs"][name]
        report[name] = {}
        for split in ("val", "test"):
            per_seed = []
            for s in SEEDS:
                cands = ([runs / f"gl-g{g}-k{k}-lora1-s{s}_eval"] if lora else
                         [runs / f"gl-g{g}-k{k}-s{s}_eval", runs / f"gl-g{g}-k{k}r2-s{s}_eval"])
                d = next(c for c in cands if c.exists()) / "out" / split
                samples = json.loads((d / "text_predictions_epoch_1.json").read_text())["samples"]
                avg, _ = score_answers([x["prediction"] for x in samples],
                                       [x["ground_truths"] for x in samples])
                ev = json.loads((d / f"eval_{split}.json").read_text())
                diff = max(abs(100 * avg[m] - 100 * ev[m]) for m in M)
                worst_eval = max(worst_eval, diff)
                per_seed.append({m: 100 * avg[m] for m in M} | {"n": len(samples)})
            agg = {m: (st.mean(r[m] for r in per_seed), st.pstdev(r[m] for r in per_seed)) for m in M}
            dev = max(max(abs(agg[m][0] - ref[split][m][0]), abs(agg[m][1] - ref[split][m][1])) for m in M)
            worst_agg = max(worst_agg, dev)
            report[name][split] = {"mean_std": agg, "n": [r["n"] for r in per_seed],
                                   "max_abs_diff_vs_gl_results": dev}
            print(f"{name:15s} {split:4s} F1 {agg['f1'][0]:.2f}±{agg['f1'][1]:.2f} "
                  f"CIDEr {agg['cider'][0]:.2f}  n={per_seed[0]['n']}  max|diff| vs gl_results {dev:.4f}")
    report["max_abs_diff_rescored_vs_eval_json"] = worst_eval
    report["max_abs_diff_vs_gl_results"] = worst_agg
    print(f"max |rescored - eval json| = {worst_eval:.4f}; max |aggregate - gl_results| = {worst_agg:.4f}")
    a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
