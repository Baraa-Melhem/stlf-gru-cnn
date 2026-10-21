"""Train a model on the full history and save a deployable artifact."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from stlf.cli.train import main

if __name__ == "__main__":
    raise SystemExit(main())
