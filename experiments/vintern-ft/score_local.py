"""Score fetched Vintern-FT predictions with the SAME metric stack as Table 1.

    python experiments/vintern-ft/score_local.py experiments/vintern-ft/out/val/results/text_predictions_epoch_1.json

Runs metrics.compute_score.compute_all_data (the function behind
plans/results-5bridge.md and the AutoViVQA baseline table -> in-house 8 metrics,
per-sample max-over-refs then averaged) + scripts/rescore_corpus.py-style corpus
CIDEr-D / BLEU-4 / ROUGE-L. Prints both, x100.
"""
import argparse
import json
import string
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metrics.compute_score import compute_all_data  # noqa: E402
from metrics.cider.cider import Cider  # noqa: E402
from metrics.bleu.bleu import Bleu  # noqa: E402
from metrics.rouge.rouge import Rouge  # noqa: E402


def _norm(s):
    s = str(s).translate(str.maketrans("", "", string.punctuation)).lower().strip()
    return " ".join(unicodedata.normalize("NFC", s).split())


def corpus(preds, refs):
    gts = {str(i): [_norm(r) for r in rs] for i, rs in enumerate(refs)}
    res = {str(i): [_norm(p)] for i, p in enumerate(preds)}
    cd, _ = Cider().compute_score(gts, res)
    bl, _ = Bleu(4).compute_score(gts, res)
    rg, _ = Rouge().compute_score(gts, res)
    return dict(cider_d=cd * 100, bleu_4=bl[3] * 100, rouge_l=rg * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    a = ap.parse_args()
    pf = Path(a.pred)
    d = json.loads(pf.read_text())
    samples = d["samples"] if isinstance(d, dict) else d
    gen = [s["prediction"] for s in samples]
    gts = [s["ground_truths"] for s in samples]

    print(f"{pf}  (n={len(samples)}, split={d.get('split','?')})")
    ih = compute_all_data(gts, gen)
    ih = {k: float(v["average"] if isinstance(v, dict) else v) for k, v in ih.items()}
    cols = ["accuracy", "precision", "recall", "f1_token", "bleu", "rouge", "meteor", "cider"]
    print("  in-house (x100): " + "  ".join(f"{c}={ih[c]*100:.2f}" for c in cols if c in ih))

    cp = corpus(gen, gts)
    print(f"  corpus (x100):   CIDEr-D={cp['cider_d']:.1f}  BLEU-4={cp['bleu_4']:.1f}  ROUGE-L={cp['rouge_l']:.1f}")

    out = {"n": len(samples), "in_house_x100": {c: round(ih[c] * 100, 2) for c in cols if c in ih},
           "corpus_x100": {k: round(v, 2) for k, v in cp.items()}}
    (pf.parent / "scored.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"  -> {pf.parent / 'scored.json'}")


if __name__ == "__main__":
    main()
