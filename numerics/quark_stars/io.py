"""Shared I/O helpers for quark-star data products."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_table(path: Path, columns: list[str], data: np.ndarray, metadata: dict[str, object]) -> None:
    ensure_directory(path.parent)
    header_lines = [f"{key}={value}" for key, value in metadata.items()]
    header_lines.append(f"columns={' '.join(columns)}")
    np.savetxt(path, data, header="\n".join(header_lines), fmt="%.3e")


def output_directories(root: Path, pipeline: str) -> tuple[Path, Path]:
    base = ensure_directory(root / pipeline)
    return ensure_directory(base / "data"), ensure_directory(base / "plots")


def output_directory(root: Path, pipeline: str) -> Path:
    return ensure_directory(root / pipeline)
