"""Run formatting, linting, type checks, and tests."""
from __future__ import annotations

import argparse
import subprocess
from typing import Sequence


def run_command(args: Sequence[str]) -> None:
    """Run a command and raise on failure."""
    subprocess.run(list(args), check=True)


def main() -> int:
    """Execute quality checks in a consistent order."""
    parser = argparse.ArgumentParser(
        description="Run format, lint, typecheck, and tests with uv."
    )
    parser.add_argument(
        "--skip-format", action="store_true", help="Skip ruff formatting."
    )
    parser.add_argument(
        "--skip-lint", action="store_true", help="Skip ruff lint checks."
    )
    parser.add_argument(
        "--skip-typecheck", action="store_true", help="Skip pyright checks."
    )
    parser.add_argument(
        "--skip-tests", action="store_true", help="Skip pytest."
    )
    args = parser.parse_args()

    if not args.skip_format:
        run_command(["uv", "run", "ruff", "format", "."])
    if not args.skip_lint:
        run_command(["uv", "run", "ruff", "check", "."])
    if not args.skip_typecheck:
        run_command(["uv", "run", "pyright"])
    if not args.skip_tests:
        run_command(["uv", "run", "pytest"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
