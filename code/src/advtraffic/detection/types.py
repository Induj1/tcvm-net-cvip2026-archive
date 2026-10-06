"""Shared detection data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Detection:
    """Single object detection in absolute xyxy pixel coordinates."""

    frame_id: int
    class_id: int
    class_name: str
    confidence: float
    xyxy: np.ndarray
    track_id: int | None = None
    feature: np.ndarray | None = None
    anomaly_score: float = 0.0
    is_adversarial: bool = False
    is_recovered: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self, **updates: Any) -> "Detection":
        data = {
            "frame_id": self.frame_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "xyxy": self.xyxy.copy(),
            "track_id": self.track_id,
            "feature": None if self.feature is None else self.feature.copy(),
            "anomaly_score": self.anomaly_score,
            "is_adversarial": self.is_adversarial,
            "is_recovered": self.is_recovered,
            "metadata": dict(self.metadata),
        }
        data.update(updates)
        return Detection(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "xyxy": [float(v) for v in self.xyxy.tolist()],
            "track_id": self.track_id,
            "anomaly_score": float(self.anomaly_score),
            "is_adversarial": bool(self.is_adversarial),
            "is_recovered": bool(self.is_recovered),
            "metadata": self.metadata,
        }
