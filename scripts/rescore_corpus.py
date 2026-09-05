"""Corpus-level pycocoevalcap rescore (CIDEr-D + BLEU-4 + ROUGE-L) for an
arbitrary text_predictions_epoch_*.json file — same convention as
plans/results-grouped-split.md §1/§2 (the cross-paper-comparable numbers),
generalized from scripts/rescore_expA.py (CIDEr-only) to also cover BLEU/ROUGE
so LoRA checkpoints (or anything outside checkpoints/expA/) can be scored.

    python scripts/rescore_corpus.py --pred checkpoints/expA-lora16/seed42/qformer/results/text_predictions_epoch_1.json
    -> prints CIDEr-D / BLEU-4 / ROUGE-L, writes alongside as *_corpus.json
"""
from __future__ import annotations

import argparse
import json
import string
import unicodedata
from pathlib import Path

from metrics.cider.cider import Cider
from metrics.bleu.bleu import Bleu
from metrics.rouge.rouge import Rouge


def _norm(s: str) -> str:
    s = str(s).translate(str.maketrans("", "", string.punctuation)).lower().strip()
    s = unicodedata.normalize("NFC", s)
    return " ".join(s.split())


def load_samples(pred_path: Path) -> list[dict]:
    d = json.loads(pred_path.read_text())
    return d["samples"] if isinstance(d, dict) and "samples" in d else d


def corpus_score(preds: list[str], refs: list[list[str]]) -> dict:
    gts = {str(i): [_norm(r) for r in rs] for i, rs in enumerate(refs)}
    res = {str(i): [_norm(p)] for i, p in enumerate(preds)}
    cider, _ = Cider().compute_score(gts, res)
    bleu, _ = Bleu(4).compute_score(gts, res)
    rouge, _ = Rouge().compute_score(gts, res)
    return {"cider_d": float(cider) * 100, "bleu_4": float(bleu[3]) * 100, "rouge_l": float(rouge) * 100}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="path to text_predictions_epoch_*.json")
    ap.add_argument("--label", default=None, help="name to print/store (default: pred path)")
    a = ap.parse_args()

    pf = Path(a.pred)
    rows = load_samples(pf)
    preds = [r["prediction"] for r in rows]
    refs = [r["ground_truths"] for r in rows]
    sc = corpus_score(preds, refs)
    sc["n"] = len(rows)
    sc["source"] = str(pf)
    label = a.label or pf.parent.parent.name

    print(f"{label}  (n={sc['n']})")
    print(f"  CIDEr-D  {sc['cider_d']:.1f}")
    print(f"  BLEU-4   {sc['bleu_4']:.1f}")
    print(f"  ROUGE-L  {sc['rouge_l']:.1f}")

    dst = pf.with_name(pf.stem + "_corpus.json")
    dst.write_text(json.dumps(sc, indent=2, ensure_ascii=False))
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
