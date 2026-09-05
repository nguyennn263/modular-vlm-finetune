"""C4 -- compute the cross-tab (F1 bucket x semantic judgment) from the
assistant self-check and write the report.

    python scripts/human_validation_report.py
    -> outputs/human_validation/report.md

Reads outputs/human_validation/selfcheck_judgments.json (the judgment file,
hand-written by the assistant reading each of the 120 sampled predictions
against its 5 references -- see that file's "method" field for exactly what
this is and is not a substitute for).
"""
from __future__ import annotations

import json
from collections import defaultdict

from src.config.loader import repo_root


def main() -> None:
    root = repo_root()
    d = json.loads((root / "outputs/human_validation/selfcheck_judgments.json").read_text())
    rows = [r for r in d["judgments"] if r["judgment"] != "ambiguous_gt"]
    n_excluded = len(d["judgments"]) - len(rows)

    buckets = ["strong", "partial", "weak", "zero"]
    judgments = ["correct", "partial", "incorrect", "nonsensical"]

    by_bucket: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r["judgment"])

    lines = [
        "# C4 self-check: F1 bucket vs. semantic judgment",
        "",
        f"Single-rater (assistant) semantic-plausibility check, n={len(rows)} "
        f"scored (+{n_excluded} excluded: references disagreed with each other). "
        "**Not** the originally-scoped 300-500-sample / 2-annotator / Cohen's-kappa "
        "protocol -- no image access, no second rater, no inter-rater statistic. "
        "See outputs/human_validation/selfcheck_judgments.json for the method note "
        "and every individual judgment + reasoning.",
        "",
        "| F1 bucket | n | correct | partial | incorrect | nonsensical | semantic-acceptable (correct+partial) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    tot = defaultdict(int)
    n_tot = 0
    for b in buckets:
        js = by_bucket.get(b, [])
        n = len(js)
        if n == 0:
            continue
        n_tot += n
        counts = {j: js.count(j) for j in judgments}
        for j in judgments:
            tot[j] += counts[j]
        acc = counts["correct"] + counts["partial"]
        lines.append(
            f"| {b} | {n} | {counts['correct']} ({100*counts['correct']/n:.1f}%) | "
            f"{counts['partial']} ({100*counts['partial']/n:.1f}%) | "
            f"{counts['incorrect']} ({100*counts['incorrect']/n:.1f}%) | "
            f"{counts['nonsensical']} ({100*counts['nonsensical']/n:.1f}%) | "
            f"{acc} ({100*acc/n:.1f}%) |"
        )
    acc_tot = tot["correct"] + tot["partial"]
    lines.append(
        f"| **overall** | **{n_tot}** | **{tot['correct']} ({100*tot['correct']/n_tot:.1f}%)** | "
        f"**{tot['partial']} ({100*tot['partial']/n_tot:.1f}%)** | "
        f"**{tot['incorrect']} ({100*tot['incorrect']/n_tot:.1f}%)** | "
        f"**{tot['nonsensical']} ({100*tot['nonsensical']/n_tot:.1f}%)** | "
        f"**{acc_tot} ({100*acc_tot/n_tot:.1f}%)** |"
    )
    (root / "outputs/human_validation/report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
