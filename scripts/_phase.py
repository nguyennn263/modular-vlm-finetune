"""Shared helper for the phase orchestrators (final-plan section 6).

Each phase script defines a list of Step(...) and calls ``run_phase``. Steps marked
``done=False`` are printed as TODO and (unless --force) stop the run there, so the
phases stay honest about what is actually wired up.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class Step:
    desc: str
    cmd: list[str] = field(default_factory=list)
    done: bool = True


def run_phase(name: str, steps: list[Step], argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog=f"python scripts/{name}.py")
    ap.add_argument("--dry-run", action="store_true", help="Print steps, run nothing.")
    ap.add_argument("--force", action="store_true", help="Run past not-yet-implemented steps.")
    args = ap.parse_args(argv)

    print(f"=== {name} ===")
    for i, step in enumerate(steps, 1):
        tag = "  " if step.done else "TODO"
        cmd = " ".join(shlex.quote(c) for c in step.cmd) if step.cmd else "(manual)"
        print(f"[{tag}] {i}. {step.desc}\n       $ {cmd}")

    if args.dry_run:
        return

    for step in steps:
        if not step.done and not args.force:
            print(f"\n[stop] '{step.desc}' is not implemented yet. See plans/final-plan.md.")
            sys.exit(1)
        if step.cmd:
            print(f"\n$ {' '.join(step.cmd)}")
            subprocess.run(step.cmd, check=True)
    print(f"\n[{name}] complete.")
