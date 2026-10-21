"""``download_data.py`` — fetch and checksum-verify the public dataset."""

from __future__ import annotations

import argparse

from stlf.data.download import download_dataset
from stlf.logging_utils import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="data/raw", help="target directory (default: data/raw)")
    parser.add_argument("--force", action="store_true", help="re-download even if files are valid")
    args = parser.parse_args(argv)
    setup_logging()
    download_dataset(args.dest, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
