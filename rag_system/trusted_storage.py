"""Trusted local storage helpers (path confinement for index files)."""

import pickle
from pathlib import Path


def trusted_path(base_dir: Path, filename: str) -> Path:
    base = base_dir.resolve()
    path = (base / filename).resolve()
    if base not in path.parents and path != base:
        raise ValueError(f"Blocked path outside trusted directory: {filename}")
    return path


def load_trusted_pickle(base_dir: Path, filename: str):
    path = trusted_path(base_dir, filename)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        return pickle.load(f)
