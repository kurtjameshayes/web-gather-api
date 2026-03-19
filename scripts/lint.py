"""Run ruff lint checks."""
from __future__ import annotations

import subprocess
from typing import Sequence


def run_command(args: Sequence[str]) -> None:
    """Run a command and raise on failure."""
    subprocess.run(list(args), check=True)


def main() -> int:
    """Check the codebase with ruff."""
    run_command(["uv", "run", "ruff", "check", "."])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
