"""G10: paired-bootstrap 95% CIs for the headline comparisons.

For each bridge, seed 42, full val (n=5463), plain vs +LoRA r=16 (1 epoch):
  - token-F1  (metrics.vqa_metrics.PrecisionRecallF1 convention: set word overlap,
    best-F1 over the 5 refs, mean over samples) — per-sample, direct bootstrap.
  - corpus CIDEr-D (metrics.cider.Cider, x100 — the cross-paper number in §5.1) —
    resample sample indices, recompute corpus score on the resampled multiset.

Paired: the SAME resampled indices score both models each iteration, so the CI on
the difference accounts for the shared sample draw. Also reports P(Δ>0) across
resamples (a bootstrap one-sided p-value proxy).

vs ViMoE-VQA: no per-sample data is published, so only a one-sample CI on our
own estimate is possible — we report where ViMoE's point value falls relative
to it, and state that a paired test is not possible.

    python scripts/bootstrap_ci.py [--iters 2000] [--cider-iters 1000] [--seed 0]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics.cider.cider import Cider  # noqa: E402

VIMOE = {"cider_d": 88.7, "f1": 60.7, "bleu_4": 12.5, "rouge_l": 47.1}

# seed-42 full-val prediction files (5463 samples), plain vs +LoRA r=16 1-epoch
PAIRS = {
    "multi_token": (
        "checkpoints/expA/seed42/multi_token/results/text_predictions_epoch_1.json",
        "checkpoints/expA-lora16/seed42/multi_token_full/results/text_predictions_epoch_1.json",
    ),
    "qformer": (
        "checkpoints/expA/seed42/qformer/results/text_predictions_epoch_1.json",
        "checkpoints/expA-lora16/seed42/qformer/results/text_predictions_epoch_1.json",
    ),
    "mini_qformer": (
        "checkpoints/expA/seed42/mini_qformer/results/text_predictions_epoch_1.json",
        "checkpoints/expA-lora16/seed42/mini_qformer/results/text_predictions_epoch_1.json",
    ),
    "residual": (
        "checkpoints/expA/seed42/residual/results/text_predictions_epoch_1.json",
        "checkpoints/expA-lora16/seed42/residual/results/text_predictions_epoch_1.json",
    ),
}


def _norm(s: str) -> set:
    return set(re.sub(r"[^\w\s]", "", str(s).lower()).split())


def _norm_str(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", str(s).lower()).split())


def load(path: str):
    d = json.loads((ROOT / path).read_text())
    rows = d["samples"] if isinstance(d, dict) and "samples" in d else d
    preds = [r["prediction"] for r in rows]
    refs = [r["ground_truths"] for r in rows]
    return preds, refs


def per_sample_f1(preds, refs) -> np.ndarray:
    out = np.empty(len(preds))
    for i, (p, rs) in enumerate(zip(preds, refs)):
        pw = _norm(p)
        best = 0.0
        for r in rs:
            rw = _norm(r)
            if not pw or not rw:
                continue
            ov = len(pw & rw)
            if not ov:
                continue
            prec, rec = ov / len(pw), ov / len(rw)
            f1 = 2 * prec * rec / (prec + rec)
            best = max(best, f1)
        out[i] = best
    return out


def corpus_cider(preds, refs, idx=None) -> float:
    if idx is None:
        idx = range(len(preds))
    gts = {str(k): [_norm_str(x) for x in refs[j]] for k, j in enumerate(idx)}
    res = {str(k): [_norm_str(preds[j])] for k, j in enumerate(idx)}
    score, _ = Cider().compute_score(gts, res)
    return float(score) * 100


def ci(arr, lo=2.5, hi=97.5):
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--cider-iters", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    out = ROOT / "outputs" / "bootstrap_ci.json"
    report = {"iters_f1": a.iters, "iters_cider": a.cider_iters, "n_refs": 5, "bridges": {}}
    _p = lambda *x: print(*x, flush=True)  # noqa: E731

    for bridge, (plain_p, lora_p) in PAIRS.items():
        pp, pr = load(plain_p)
        lp, lr = load(lora_p)
        n = len(pp)
        assert len(lp) == n, f"{bridge}: plain n={n} vs lora n={len(lp)}"
        # refs are identical across the two runs (same val set); use plain's
        print(f"\n=== {bridge}  (n={n}) ===")

        f1_plain = per_sample_f1(pp, pr)
        f1_lora = per_sample_f1(lp, lr)
        pt = {"f1_plain": float(f1_plain.mean() * 100), "f1_lora": float(f1_lora.mean() * 100)}

        # paired F1 bootstrap
        bp, bl, bd = [], [], []
        for _ in range(a.iters):
            s = rng.integers(0, n, n)
            mp, ml = f1_plain[s].mean() * 100, f1_lora[s].mean() * 100
            bp.append(mp); bl.append(ml); bd.append(ml - mp)
        bp, bl, bd = map(np.asarray, (bp, bl, bd))
        f1_res = {
            "plain": [pt["f1_plain"], *ci(bp)],
            "lora": [pt["f1_lora"], *ci(bl)],
            "delta_lora_minus_plain": [pt["f1_lora"] - pt["f1_plain"], *ci(bd)],
            "p_delta_gt_0": float((bd > 0).mean()),
        }

        # paired corpus-CIDEr-D bootstrap
        c_plain = corpus_cider(pp, pr)
        c_lora = corpus_cider(lp, lr)
        cp, cl, cd = [], [], []
        for _ in range(a.cider_iters):
            s = rng.integers(0, n, n)
            xp, xl = corpus_cider(pp, pr, s), corpus_cider(lp, lr, s)
            cp.append(xp); cl.append(xl); cd.append(xl - xp)
        cp, cl, cd = map(np.asarray, (cp, cl, cd))
        cider_res = {
            "plain": [c_plain, *ci(cp)],
            "lora": [c_lora, *ci(cl)],
            "delta_lora_minus_plain": [c_lora - c_plain, *ci(cd)],
            "p_delta_gt_0": float((cd > 0).mean()),
        }

        report["bridges"][bridge] = {"f1": f1_res, "cider_d": cider_res}
        out.write_text(json.dumps(report, indent=2))  # incremental — survives a kill

        def fmt(x):
            return f"{x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]"
        _p(f"  F1     plain {fmt(f1_res['plain'])}   lora {fmt(f1_res['lora'])}")
        _p(f"         Δ {fmt(f1_res['delta_lora_minus_plain'])}   P(Δ>0)={f1_res['p_delta_gt_0']:.3f}")
        _p(f"  CIDEr-D plain {fmt(cider_res['plain'])}   lora {fmt(cider_res['lora'])}")
        _p(f"         Δ {fmt(cider_res['delta_lora_minus_plain'])}   P(Δ>0)={cider_res['p_delta_gt_0']:.3f}")

    # multi_token vs ViMoE — one-sample CI only (no ViMoE per-sample data)
    mt = report["bridges"]["multi_token"]
    print("\n=== multi_token vs ViMoE-VQA (one-sample CI; paired test not possible) ===")
    for metric, key in (("F1", "f1"), ("CIDEr-D", "cider_d")):
        plain_ci = mt[key]["plain"]
        lora_ci = mt[key]["lora"]
        v = VIMOE[key if key != "cider_d" else "cider_d"]
        print(f"  {metric}: ViMoE {v}  |  multi_token-plain {plain_ci[0]:.2f} "
              f"[{plain_ci[1]:.2f}, {plain_ci[2]:.2f}]  |  +LoRA {lora_ci[0]:.2f} "
              f"[{lora_ci[1]:.2f}, {lora_ci[2]:.2f}]")
    report["vimoe_note"] = ("ViMoE-VQA publishes corpus point values only; no per-sample "
                            "predictions, so a paired bootstrap vs multi_token is not possible. "
                            "CIs above are one-sample (our estimate's own sampling variability).")

    out.write_text(json.dumps(report, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
