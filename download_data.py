"""Download and checksum-verify the public Panama load dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from stlf.cli.download import main

if __name__ == "__main__":
    raise SystemExit(main())
