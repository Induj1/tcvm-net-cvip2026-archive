"""I/O helpers used across scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def iter_files(root: str | Path, extensions: Iterable[str]) -> Iterable[Path]:
    root = Path(root)
    allowed = {ext.lower() for ext in extensions}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path


def read_image(path: str | Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image
