"""Fallback IoU tracker when ByteTrack is unavailable."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from advtraffic.detection.types import Detection
from advtraffic.utils.geometry import box_iou


@dataclass
class _Track:
    track_id: int
    class_id: int
    bbox: np.ndarray
    confidence: float
    missed: int = 0


class SimpleIoUTracker:
    """A deterministic IoU tracker used as a reproducible fallback.

    For final experiments use YOLOv8Engine.track(..., tracker="bytetrack.yaml").
    This class is useful for tests, ablations, and environments where the
    tracker dependency is unavailable.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 8):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: dict[int, _Track] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        unmatched_track_ids = set(self.tracks)
        assigned: list[Detection] = []

        for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
            best_id = None
            best_iou = 0.0
            for track_id in list(unmatched_track_ids):
                track = self.tracks[track_id]
                if track.class_id != det.class_id:
                    continue
                iou = box_iou(track.bbox, det.xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_id = track_id

            if best_id is None or best_iou < self.iou_threshold:
                best_id = self.next_id
                self.next_id += 1
            else:
                unmatched_track_ids.remove(best_id)

            self.tracks[best_id] = _Track(
                track_id=best_id,
                class_id=det.class_id,
                bbox=det.xyxy.copy(),
                confidence=det.confidence,
                missed=0,
            )
            assigned.append(det.copy(track_id=best_id, metadata={**det.metadata, "tracker": "iou"}))

        for track_id in list(unmatched_track_ids):
            track = self.tracks[track_id]
            track.missed += 1
            if track.missed > self.max_missed:
                del self.tracks[track_id]

        return assigned
