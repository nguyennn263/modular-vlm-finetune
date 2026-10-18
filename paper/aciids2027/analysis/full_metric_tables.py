"""Rows with all eight metrics for Tables 2-5 (Accuracy, Precision, Recall, F1, BLEU, ROUGE-L,
METEOR, CIDEr; mean and population std over seeds, x100), read from the stored results.

    python analysis/full_metric_tables.py --repo <exp/eval-input-diagnostic checkout> --out analysis/full_metric_tables.json
"""
from __future__ import annotations
import argparse, glob, json, re, statistics as st
from pathlib import Path

M = ["accuracy", "precision", "recall", "f1", "bleu", "rouge_l", "meteor", "cider"]
HERE = Path(__file__).parent


def agg(dicts, scale=100):
    return {m: [st.mean(v := [scale * d[m] for d in dicts]), st.pstdev(v)] for m in M} | {"n": len(dicts)}


def from_pair(r):            # {metric: [mean, std]} already x100
    return {m: r[m] for m in M} | {"n": r.get("n", 3)}


def evals(repo, pattern):
    fs = sorted(glob.glob(str(repo / pattern), recursive=True))
    ds = [json.loads(Path(f).read_text()) for f in fs]
    for d in ds:
        assert d.get("gen_image") == "full", d.get("checkpoint")
    return ds


def md_rows(repo, prefix, split):
    md = (repo / "plans" / "results-full-image.md").read_text()
    out = []
    for line in md.splitlines():
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) > 14 and re.fullmatch(rf"{prefix}-s(42|123|3407)", c[0]) and c[1] == split:
            out.append({m: float(v) / 100 for m, v in zip(M, c[6:14])} | {"gen_image": "full"})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    ft = json.loads((HERE / "full_tables.json").read_text())["bridges"]
    gl = json.loads((HERE / "gl_results.json").read_text())
    R = a.repo / "outputs"
    t2 = {
        "Residual": from_pair(ft["res"][0]), "Global-only": from_pair(ft["mt"][0]),
        "Tile-Attention": from_pair(ft["ta"][0]), "Light Q-Former": from_pair(ft["mq"][0]),
        "Full Q-Former": from_pair(ft["qfx"][0]),
        "C-Abstractor": agg(evals(a.repo, "outputs/bridge_full/convabs-s42/**/eval_val.json")
                            + evals(a.repo, "outputs/train_xb/xb-cab-s123d_eval/**/eval_val.json")
                            + evals(a.repo, "outputs/bridge_full/cab-s3407-st6400-ds/**/eval_val.json")),
        "Patch-pool": agg(evals(a.repo, "outputs/bridge_full/poolmean-s42/**/eval_val.json")
                          + evals(a.repo, "outputs/bridge_full/ppm-s*-st6400-ds/**/eval_val.json")),
        "Local untrained": agg(evals(a.repo, "outputs/bridge_full/g0k36-notrain-r2/**/eval_val.json")),
        "ViBridge": from_pair(gl["configs"]["g14-k36"]["val"]),
    }
    t3 = {"g8-k0": from_pair(ft["mt"][0]),
          "g14-k0": agg(evals(a.repo, "outputs/tok_full/g14-*/**/val/eval_val.json"))}
    t3 |= {c: from_pair(v["val"]) for c, v in gl["configs"].items()}
    t4 = {"8": from_pair(ft["mt"][0])}
    by = {}
    for f in sorted(glob.glob(str(R / "tok_full/*/**/val/eval_val.json"), recursive=True)):
        g, s = re.match(r"g(\d+)-s(\d+)", Path(f).relative_to(R / "tok_full").parts[0]).groups()
        by.setdefault(g, {})[s] = json.loads(Path(f).read_text())
    t4 |= {g: agg(list(v.values())) for g, v in by.items()}
    t5 = {"ViBridge frozen": from_pair(gl["configs"]["g14-k36"]["test"]),
          "Global-only + LoRA": agg(md_rows(a.repo, "l1ep", "test")),
          "Global-only + LoRA 3ep": agg(md_rows(a.repo, "l3ep", "test")),
          "ViBridge + LoRA": from_pair(gl["g14-k36+lora1"]["test"]),
          "g0 + LoRA": agg(evals(a.repo, "outputs/train_gl/gl-g0-k36-lora1-s*_eval/out/test/eval_test.json"))}
    out = {"t2": t2, "t3": t3, "t4": dict(sorted(t4.items(), key=lambda kv: int(kv[0]))), "t5": t5}
    a.out.write_text(json.dumps(out, indent=1))
    for t, rows in out.items():
        for k, r in rows.items():
            print(t, k, r["n"], " ".join(f"{r[m][0]:.2f}" for m in M))


if __name__ == "__main__":
    main()
