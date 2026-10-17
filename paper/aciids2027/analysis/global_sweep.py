"""Global-only (Multi-Token, k = 0) sweep over g = 4..20 on the validation set, full image.

Reads outputs/tok_full/g{g}-s{seed}[-r2]/**/val/eval_val.json from exp/eval-input-diagnostic
(scripts/parallel/tok_full.py; -r2 relaunches are the same checkpoint and give identical scores,
so each seed counts once). g = 8 is the 4-seed Global-only model of Table 2 (gl_results.json
baseline). Mean and population std over seeds, x100.

    python analysis/global_sweep.py --repo <exp/eval-input-diagnostic checkout> --out analysis/global_sweep.json
"""
from __future__ import annotations
import argparse, json, re, statistics as st
from pathlib import Path

G8 = {"f1": (50.74, 0.17), "cider": (99.07, 0.47), "n": 4}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    by: dict[int, dict[str, dict]] = {}
    for f in sorted((a.repo / "outputs" / "tok_full").glob("*/**/val/eval_val.json")):
        m = re.match(r"g(\d+)-s(\d+)", f.relative_to(a.repo / "outputs" / "tok_full").parts[0])
        d = json.loads(f.read_text())
        assert d.get("gen_image") == "full", f
        by.setdefault(int(m.group(1)), {})[m.group(2)] = d
    res = {"8": {"f1": list(G8["f1"]), "cider": list(G8["cider"]), "n": G8["n"]}}
    for g, runs in by.items():
        r = {"n": len(runs), "seeds": sorted(runs)}
        for k in ("f1", "cider"):
            v = [100 * x[k] for x in runs.values()]
            r[k] = [round(st.mean(v), 2), round(st.pstdev(v), 2)]
        res[str(g)] = r
    res = dict(sorted(res.items(), key=lambda kv: int(kv[0])))
    a.out.write_text(json.dumps(res, indent=1))
    for g, r in res.items():
        print(g, r["n"], r["f1"], r["cider"])


if __name__ == "__main__":
    main()
