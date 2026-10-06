"""Video frame extraction and writing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoFrame:
    video_path: Path
    frame_id: int
    timestamp_ms: float
    image: np.ndarray


def iter_video_frames(
    video_path: str | Path,
    stride: int = 1,
    resize: tuple[int, int] | None = None,
    max_frames: int | None = None,
) -> Iterator[VideoFrame]:
    """Yield BGR frames from a video file."""

    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    emitted = 0
    frame_id = -1
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_id += 1
        if frame_id % stride != 0:
            continue
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        yield VideoFrame(video_path=video_path, frame_id=frame_id, timestamp_ms=timestamp_ms, image=frame)
        emitted += 1
        if max_frames is not None and emitted >= max_frames:
            break
    capture.release()


class VideoWriter:
    """Small context-manager wrapper around cv2.VideoWriter."""

    def __init__(self, path: str | Path, fps: float, frame_size: tuple[int, int], codec: str = "mp4v"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(str(self.path), fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
