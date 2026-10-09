"""Numbers for the global-local bridge (Sec. 4) -> analysis/gl_results.json.

Reads the eval runs of branch exp/eval-input-diagnostic (outputs/train_gl/, 24 runs:
g in {8, 14} x k in {1, 9, 36, 144} x seeds {42, 123, 3407}) and the seed-42 Multi-Token
predictions (outputs/input_diag/), scored with metrics.vqa_metrics.score_answers -- the
same function behind every other table. Run from a checkout of that branch:

    .venv/bin/python <paper>/analysis/gl_results.py --repo <eval-branch checkout> \
        --out <paper>/analysis/gl_results.json [--splits <dir with val.jsonl>]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import subprocess
import sys
from datetime import datetime
from pathlib import Path

M = ["accuracy", "precision", "recall", "f1", "bleu", "rouge_l", "meteor", "cider"]
SEEDS = [42, 123, 3407]
G, K = [8, 14], [1, 9, 36, 144]
BASE_VAL = {"f1": 50.74, "cider": 99.07, "meteor": 41.38}   # Multi-Token 8, val, 4 seeds


def _log_text(path: Path) -> str:
    try:
        return "".join(x.get("data", "") for x in json.loads(path.read_text()))
    except Exception:
        return path.read_text()


def _eval_minutes(run_dir: Path) -> float | None:
    logs = list(run_dir.glob("*.log"))
    if not logs:
        return None
    ts = [datetime.fromisoformat(x) for x in
          re.findall(r"(2026-10-\d\d \d\d:\d\d:\d\d)", _log_text(logs[0]))]
    return (ts[-1] - ts[0]).total_seconds() / 60 if len(ts) > 1 else None


def _run_dir(gl: Path, g: int, k: int, s: int) -> Path:
    for name in (f"gl-g{g}-k{k}-s{s}_eval", f"gl-g{g}-k{k}r2-s{s}_eval"):
        if (gl / name).exists():
            return gl / name
    raise FileNotFoundError(f"no eval run for g{g} k{k} s{s}")


def _knee(points: list[tuple[float, float, str]]) -> str:
    """Kneedle on the Pareto front: the point highest above the cheapest-best chord."""
    front, best = [], -1e9
    for x, y, lab in sorted(points):
        if y > best:
            front.append((x, y, lab))
            best = y
    (x0, y0, _), (x1, y1, _) = front[0], front[-1]
    score = [(y - y0) / (y1 - y0) - (x - x0) / (x1 - x0) for x, y, _ in front]
    return front[score.index(max(score))][2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=None, help="data/splits dir (default: <repo>/data/splits)")
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo))
    from metrics.vqa_metrics import score_answers  # noqa: E402

    gl = a.repo / "outputs" / "train_gl"
    cfg = {}
    for g in G:
        for k in K:
            runs = {sp: [] for sp in ("val", "test")}
            minutes = []
            for s in SEEDS:
                d = _run_dir(gl, g, k, s)
                for sp in runs:
                    e = json.loads((d / "out" / sp / f"eval_{sp}.json").read_text())
                    runs[sp].append([100 * e[m] for m in M] + [e["loss"]])
                minutes.append(_eval_minutes(d))
            cfg[f"g{g}-k{k}"] = {
                "g": g, "k": k, "tokens": g + k,
                "eval_minutes_val_test": st.mean(m for m in minutes if m),
                **{sp: {name: [st.mean(c), st.pstdev(c)] for name, c in zip(M + ["ce"], zip(*rows))}
                   for sp, rows in runs.items()},
            }

    # selection: knee per metric x split x cost axis
    names = list(cfg)
    knees = {}
    for sp in ("val", "test"):
        for m in M:
            for axis, cost in (("tokens", lambda c: c["tokens"]),
                               ("log_tokens", lambda c: math.log(c["tokens"])),
                               ("eval_time", lambda c: c["eval_minutes_val_test"])):
                knees[f"{sp}/{m}/{axis}"] = _knee([(cost(cfg[n]), cfg[n][sp][m][0], n) for n in names])

    # TOPSIS: benefits = gain over Multi-Token 8 (val F1, CIDEr, METEOR);
    # costs = extra tokens, extra eval time, trainable params, F1 std
    params = {8: 7.35, 14: 12.86}
    tmin = min(c["eval_minutes_val_test"] for c in cfg.values())

    def topsis(w_acc: float) -> list[list]:
        crit = [(+1, w_acc / 3), (+1, w_acc / 3), (+1, w_acc / 3)] + [(-1, (1 - w_acc) / 4)] * 4
        X = {n: [c["val"]["f1"][0] - BASE_VAL["f1"], c["val"]["cider"][0] - BASE_VAL["cider"],
                 c["val"]["meteor"][0] - BASE_VAL["meteor"], c["tokens"] - 8,
                 c["eval_minutes_val_test"] - tmin + 1, params[c["g"]], c["val"]["f1"][1]]
             for n, c in cfg.items()}
        norm = [math.sqrt(sum(X[n][j] ** 2 for n in X)) for j in range(len(crit))]
        V = {n: [X[n][j] / norm[j] * crit[j][1] for j in range(len(crit))] for n in X}
        best = [(max if crit[j][0] > 0 else min)(V[n][j] for n in V) for j in range(len(crit))]
        worst = [(min if crit[j][0] > 0 else max)(V[n][j] for n in V) for j in range(len(crit))]
        S = {n: math.dist(V[n], worst) / (math.dist(V[n], best) + math.dist(V[n], worst)) for n in V}
        return sorted([[n, round(s, 3)] for n, s in S.items()], key=lambda x: -x[1])

    # per reasoning category (val): Multi-Token 8 (seed 42) vs the global-local configs (3 seeds)
    cats = [json.loads(l)["category"] for l in open((a.splits or a.repo / "data" / "splits") / "val.jsonl")]

    def by_cat(pred_file: Path) -> dict:
        samples = json.loads(pred_file.read_text())["samples"]
        assert len(samples) == len(cats)
        out = {}
        for c in sorted(set(cats)):
            idx = [i for i, x in enumerate(cats) if x == c]
            avg, _ = score_answers([samples[i]["prediction"] for i in idx],
                                   [samples[i]["ground_truths"] for i in idx])
            out[c] = {"n": len(idx), "f1": 100 * avg["f1"]}
        return out

    diag = a.repo / "outputs" / "input_diag"
    per_cat = {
        "multi_token_s42": by_cat(diag / "mt-s42-t1-full/out/mt-s42-t1-full/text_predictions_epoch_1.json"),
        "multi_token_lora3ep_s42": by_cat(diag / "l3ep-s42-t1-full/out/l3ep-s42-t1-full/text_predictions_epoch_1.json"),
    }
    for n in ("g14-k36", "g14-k144"):
        seeds = [by_cat(_run_dir(gl, cfg[n]["g"], cfg[n]["k"], s) / "out/val/text_predictions_epoch_1.json")
                 for s in SEEDS]
        per_cat[n] = {c: {"n": seeds[0][c]["n"], "f1": st.mean(x[c]["f1"] for x in seeds)} for c in seeds[0]}

    commit = subprocess.run(["git", "-C", str(a.repo), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    a.out.write_text(json.dumps({
        "provenance": {"branch": "exp/eval-input-diagnostic", "commit": commit,
                       "runs": "outputs/train_gl/gl-g{g}-k{k}[r2]-s{seed}_eval/out/{val,test}/eval_*.json",
                       "metric": "metrics.vqa_metrics.score_answers, x100; std ddof=0",
                       "generated": datetime.now().isoformat(timespec="seconds")},
        "configs": cfg, "knee": knees,
        "topsis_val": {f"{w:.1f}": topsis(w) for w in (0.3, 0.4, 0.5, 0.6, 0.7)},
        "per_category_val_f1": per_cat,
    }, indent=1, ensure_ascii=False))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
