"""Answer-generation time per budget configuration (Table 3, column Time).

For every eval kernel: the elapsed time of the generation pass over the 5,463 validation
questions (greedy, batch 2, one T4), read from its final tqdm line
"Epoch 1 metrics (batch=2, n=5463): ... [mm:ss<..]". This is the inference cost. The eval also runs
a teacher-forced loss pass first, which is not inference and was slower for the global-only runs
(29-33 vs 22 min). The old column took the span between the first and last logger timestamps:
validation loss + generation + one model load, while the caption said "validation and test".

    python analysis/eval_time.py --ledger <exp/eval-input-diagnostic>/outputs/parallel/ledger.json \
        --logs <dir with downloaded *.log> --out analysis/eval_time.json
"""
from __future__ import annotations
import argparse, json, re, statistics as st
from pathlib import Path

PAT = {  # config -> ledger keys of its eval kernels
    "g8-k0": [r"input-diag:mt-s(42|123|2026|3407)-t1-full$"],
    "g14-k0": [r"tokfull:g14-s(42|3407|123-r2)$"],
    **{f"g{g}-k{k}": [rf"gl-g{g}-k{k}(r2)?-s\d+:eval$"] for g in (8, 14) for k in (1, 9, 36, 144)},
}


def val_minutes(log: Path) -> float | None:
    text = "".join(e.get("data", "") for e in json.loads(log.read_text()))
    best = None
    for m in re.finditer(r"Epoch 1 metrics \(batch=\d+, n=5463\):[^\[]*\[(?:(\d+):)?(\d+):(\d+)<", text):
        h, mi, se = (int(x) if x else 0 for x in m.groups())
        t = h * 60 + mi + se / 60
        best = t if best is None or t > best else best
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    jobs = json.loads(a.ledger.read_text())["jobs"]
    res = {}
    for cfg, pats in PAT.items():
        runs = {}
        for key, j in jobs.items():
            if any(re.match(x, key) for x in pats):
                log = a.logs / Path(j["kernel"]).name / f"{Path(j['kernel']).name}.log"
                if log.exists() and (m := val_minutes(log)) is not None:
                    runs[key] = round(m, 2)
        if runs:
            res[cfg] = {"val_minutes_mean": round(st.mean(runs.values()), 1), "runs": runs}
    a.out.write_text(json.dumps(res, indent=1))
    for c, v in res.items():
        print(c, v["val_minutes_mean"], v["runs"])


if __name__ == "__main__":
    main()
