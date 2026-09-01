"""Merge sharded oracle outputs into the final tables.

    python -m src.analysis.merge oracle --in outputs/oracle    # table.shard*.parquet -> table.parquet + labels.parquet
    python -m src.analysis.merge expB   --in outputs/expA      # */eval_val_samples.jsonl -> combined.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

from src.analysis.oracle import oracle_labels
from src.config.loader import repo_root


def _abs(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else repo_root() / p


def merge_oracle(in_dir: str) -> None:
    import pandas as pd

    d = _abs(in_dir)
    shards = sorted(d.glob("table.shard*.parquet"))
    if not shards:
        raise SystemExit(f"no table.shard*.parquet in {d}")
    table = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    table = table.drop_duplicates(["sample_id", "action"])
    table.to_parquet(d / "table.parquet", index=False)
    oracle_labels(table).to_parquet(d / "labels.parquet", index=False)
    n_s, n_a = table.sample_id.nunique(), table.action.nunique()
    print(f"[merge] {len(shards)} shards -> {len(table)} rows ({n_s} samples x {n_a} actions) "
          f"-> {d}/table.parquet + labels.parquet")


def merge_expB(in_dir: str) -> None:
    d = _abs(in_dir)
    files = glob.glob(str(d / "**" / "eval_val_samples.jsonl"), recursive=True)
    if not files:
        raise SystemExit(f"no eval_val_samples.jsonl under {d}")
    seen, out = set(), []
    for f in files:
        m = re.search(r"seed(\d+)", f)
        seed = m.group(1) if m else None
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r.setdefault("seed", seed)
            key = (r.get("bridge"), r.get("seed"), r.get("image_id"), r.get("question"))
            if key not in seen:
                seen.add(key)
                out.append(r)
    dst = d / "eval_val_samples.combined.jsonl"
    dst.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
    bridges = sorted({r.get("bridge") for r in out})
    print(f"[merge] {len(files)} files -> {len(out)} rows, bridges={bridges} -> {dst}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m src.analysis.merge", description=__doc__)
    p.add_argument("what", choices=["oracle", "expB"])
    p.add_argument("--in", dest="in_dir", required=True)
    a = p.parse_args(argv)
    (merge_oracle if a.what == "oracle" else merge_expB)(a.in_dir)


if __name__ == "__main__":
    main()
