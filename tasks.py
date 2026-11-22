"""Cross-platform mirror of the Makefile, for machines without `make` (e.g. Windows).

Usage:
    uv run python tasks.py <target> [--config configs/baseline.yaml] [extra args...]
    uv run python tasks.py eval --config configs/ollama_llama.yaml --tasks e01,m03 --resume
    uv run python tasks.py --list

Extra arguments after the target are forwarded to `eval` (e.g. --tasks, --limit, --resume)
and to `reclassify` (e.g. --dry-run).
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def _run(*cmds: list[str]) -> int:
    for cmd in cmds:
        print("$", " ".join(cmd), flush=True)
        code = subprocess.call(cmd)
        if code != 0:
            return code
    return 0


def _todo(phase: int, extra: str = "") -> int:
    print(f"not implemented (Phase {phase}){' - ' + extra if extra else ''}")
    return 0


TARGETS = {
    "setup": lambda a: _run(["uv", "sync"]),
    "db": lambda a: _run(["uv", "run", "python", "-m", "eval_harness.db", "build"]),
    "verify-tasks": lambda a: _run(["uv", "run", "python", "-m", "eval_harness.golden", "verify"]),
    "eval": lambda a: _run(
        ["uv", "run", "python", "-m", "eval_harness.runner", "--config", a.config, *a.extra]
    ),
    "reclassify": lambda a: _run(
        ["uv", "run", "python", "-m", "eval_harness.classify", "reclassify", *a.extra]
    ),
    "sweep": lambda a: _todo(7),
    "report": lambda a: _todo(6),
    "dashboard": lambda a: _todo(6),
    "lint": lambda a: _run(
        ["uv", "run", "ruff", "check", "."], ["uv", "run", "ruff", "format", "--check", "."]
    ),
    "format": lambda a: _run(
        ["uv", "run", "ruff", "format", "."], ["uv", "run", "ruff", "check", "--fix", "."]
    ),
    "test": lambda a: _run(["uv", "run", "pytest"]),
    "smoke": lambda a: _todo(8),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("target", nargs="?", choices=sorted(TARGETS))
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--list", action="store_true", help="list targets")
    args, args.extra = parser.parse_known_args()
    if args.list or args.target is None:
        print("\n".join(sorted(TARGETS)))
        return 0
    return TARGETS[args.target](args)


if __name__ == "__main__":
    sys.exit(main())
