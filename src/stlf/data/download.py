"""Download the public Panama load dataset from Mendeley Data with checksum verification.

Source: Aguilar Madrid, E. (2021). *Short-term electricity load forecasting (Panama case
study)*. Mendeley Data, V1. https://doi.org/10.17632/byx7sztj59.1 (CC BY 4.0). This is the
same data behind the Kaggle "Electricity Load Forecasting" dataset used by the original notebook.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from stlf.logging_utils import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://data.mendeley.com/public-files/datasets/byx7sztj59/files"


@dataclass(frozen=True)
class RemoteFile:
    """A file in the Mendeley dataset, pinned by id and SHA-256."""

    name: str
    file_id: str
    sha256: str
    size: int

    @property
    def url(self) -> str:
        return f"{_BASE_URL}/{self.file_id}/file_downloaded"


DATASET_FILES: tuple[RemoteFile, ...] = (
    RemoteFile(
        "continuous dataset.csv",
        "94ef3285-7ce4-43e2-9fd3-9cb3f592b89c",
        "ef52f68fc8989d02ff2f2565134014eaf9707318a8389d4022cd38d44656fbce",
        10_524_179,
    ),
    RemoteFile(
        "weekly pre-dispatch forecast.csv",
        "1339c5ea-adbc-4c6e-8e64-c64c1ba20d54",
        "002a40a01b7d7209ae791bb85d7ad2f9c9b5a49754e30e09258e56805b805dac",
        929_119,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(destination: str | Path, *, force: bool = False) -> list[Path]:
    """Fetch every dataset file into ``destination`` and verify its checksum.

    Files already present with the right hash are skipped unless ``force`` is set. A file
    whose hash does not match is deleted and an error is raised, so a corrupted or tampered
    download never reaches the training pipeline.
    """
    target_dir = Path(destination)
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for remote in DATASET_FILES:
        path = target_dir / remote.name
        if path.exists() and not force and _sha256(path) == remote.sha256:
            logger.info("%s already present (checksum OK)", remote.name)
            paths.append(path)
            continue
        logger.info("Downloading %s (%.1f MB)", remote.name, remote.size / 1e6)
        partial = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(remote.url, timeout=120) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        actual = _sha256(partial)
        if actual != remote.sha256:
            partial.unlink(missing_ok=True)
            raise OSError(f"checksum mismatch for {remote.name}: {actual} != {remote.sha256}")
        partial.replace(path)
        paths.append(path)
    return paths
