"""Profile FLOPs + wall-clock across n_tiles  (final-plan P1).

    python -m src.cli.profile --n-tiles 1 2 3 4 6 --samples 200 --bridge tile_attention

STATUS: skeleton. Arg-parsing works; the measurement body is P1 work.
Writes ``outputs/profile/pipeline_cost.json`` with, per n_tiles:
FLOPs, single-sample latency, batched throughput.
"""
from __future__ import annotations

import argparse


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.cli.profile", description=__doc__)
    p.add_argument("--n-tiles", type=int, nargs="+", default=[1, 2, 3, 4, 6], dest="n_tiles")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--bridge", default="tile_attention")
    p.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    p.add_argument("--out", default="outputs/profile/pipeline_cost.json")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    raise NotImplementedError(
        "P1: run the multi-tile pipeline at each n_tiles, measure FLOPs "
        "(fvcore / torch.profiler) + latency + throughput, dump to "
        f"{args.out}. Requires the multi-tile forward path (see final-plan P1)."
    )


if __name__ == "__main__":
    main()
