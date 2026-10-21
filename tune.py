"""Optuna hyper-parameter search on the first fold's validation block."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from stlf.cli.tune import main

if __name__ == "__main__":
    raise SystemExit(main())
