"""Temporal Consistency Verification Module (TCVM)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from advtraffic.detection.features import cosine_similarity
from advtraffic.detection.types import Detection
from advtraffic.utils.geometry import box_iou


@dataclass
class TCVMConfig:
    """Configuration for temporal adversarial anomaly scoring."""

    window: int = 8
    ema_alpha: float = 0.35
    min_history: int = 3
    anomaly_threshold: float = 0.62
    update_threshold: float = 0.75
    confidence_tau: float = 2.5
    confidence_sigma_floor: float = 0.05
    motion_tau: float = 0.7
    feature_tau: float = 0.45
    disappearance_patience: int = 3
    min_recovery_confidence: float = 0.2
    collapse_recovery_threshold: float = 0.25
    collapse_min_missing: int = 2
    collapse_max_detection_ratio: float = 0.35
    weight_confidence: float = 0.34
    weight_motion: float = 0.28
    weight_feature: float = 0.26
    weight_disappearance: float = 0.12

    def normalized_weights(self) -> tuple[float, float, float, float]:
        total = (
            self.weight_confidence
            + self.weight_motion
            + self.weight_feature
            + self.weight_disappearance
        )
        if total <= 0:
            return 0.34, 0.28, 0.26, 0.12
        return (
            self.weight_confidence / total,
            self.weight_motion / total,
            self.weight_feature / total,
            self.weight_disappearance / total,
        )


@dataclass
class TCVMScore:
    """Component-wise anomaly score for one detection or missing-track event."""

    confidence_instability: float = 0.0
    motion_inconsistency: float = 0.0
    feature_inconsistency: float = 0.0
    disappearance: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "confidence_instability": float(self.confidence_instability),
            "motion_inconsistency": float(self.motion_inconsistency),
            "feature_inconsistency": float(self.feature_inconsistency),
            "disappearance": float(self.disappearance),
            "total": float(self.total),
        }


@dataclass
class _TrackHistory:
    track_id: int
    class_id: int
    class_name: str
    boxes: deque[np.ndarray] = field(default_factory=deque)
    confidences: deque[float] = field(default_factory=deque)
    features: deque[np.ndarray] = field(default_factory=deque)
    ema_confidence: float | None = None
    confidence_var: float = 0.0
    ema_feature: np.ndarray | None = None
    last_frame_id: int = -1
    missed: int = 0

    def __len__(self) -> int:
        return len(self.confidences)

    @property
    def last_box(self) -> np.ndarray | None:
        return None if not self.boxes else self.boxes[-1]

    def predicted_box(self) -> np.ndarray | None:
        if not self.boxes:
            return None
        if len(self.boxes) == 1:
            return self.boxes[-1].copy()
        velocity = self.boxes[-1] - self.boxes[-2]
        return self.boxes[-1] + velocity

    def update(self, detection: Detection, cfg: TCVMConfig) -> None:
        if self.ema_confidence is None:
            self.ema_confidence = detection.confidence
            self.confidence_var = cfg.confidence_sigma_floor**2
        else:
            delta = detection.confidence - self.ema_confidence
            self.ema_confidence = (1.0 - cfg.ema_alpha) * self.ema_confidence + cfg.ema_alpha * detection.confidence
            self.confidence_var = (1.0 - cfg.ema_alpha) * self.confidence_var + cfg.ema_alpha * delta * delta

        if detection.feature is not None:
            feature = detection.feature.astype(np.float32)
            norm = np.linalg.norm(feature)
            if norm > 1e-8:
                feature = feature / norm
            if self.ema_feature is None:
                self.ema_feature = feature
            else:
                self.ema_feature = (1.0 - cfg.ema_alpha) * self.ema_feature + cfg.ema_alpha * feature
                self.ema_feature = self.ema_feature / max(np.linalg.norm(self.ema_feature), 1e-8)
            self.features.append(feature)

        self.boxes.append(detection.xyxy.astype(float).copy())
        self.confidences.append(float(detection.confidence))
        self.last_frame_id = detection.frame_id
        self.missed = 0

        while len(self.boxes) > cfg.window:
            self.boxes.popleft()
        while len(self.confidences) > cfg.window:
            self.confidences.popleft()
        while len(self.features) > cfg.window:
            self.features.popleft()


class TemporalConsistencyVerifier:
    """Detect adversarial anomalies by comparing detections with track history."""

    def __init__(self, config: TCVMConfig | None = None):
        self.config = config or TCVMConfig()
        self.histories: dict[int, _TrackHistory] = {}

    def verify(self, detections: list[Detection], frame_id: int) -> tuple[list[Detection], list[Detection]]:
        """Score detections and return (scored detections, missing-track events)."""

        if any(det.track_id is None for det in detections):
            raise ValueError("TCVM requires stable track_id values. Use ByteTrack or SimpleIoUTracker first.")

        seen_track_ids = {int(det.track_id) for det in detections if det.track_id is not None}
        scored = [self._score_and_update(det) for det in detections]
        missing_events = self._score_missing_tracks(seen_track_ids, frame_id)
        return scored, missing_events

    def get_history(self, track_id: int | None) -> _TrackHistory | None:
        if track_id is None:
            return None
        return self.histories.get(track_id)

    def _score_and_update(self, detection: Detection) -> Detection:
        assert detection.track_id is not None
        history = self.histories.get(detection.track_id)
        if history is None:
            history = _TrackHistory(
                track_id=detection.track_id,
                class_id=detection.class_id,
                class_name=detection.class_name,
            )
            self.histories[detection.track_id] = history

        score = self._score_detection(detection, history)
        is_adversarial = len(history) >= self.config.min_history and score.total >= self.config.anomaly_threshold
        scored = detection.copy(
            anomaly_score=score.total,
            is_adversarial=is_adversarial,
            metadata={**detection.metadata, "tcvm": score.to_dict()},
        )

        if score.total <= self.config.update_threshold:
            history.update(detection, self.config)
        else:
            history.missed += 1
        return scored

    def _score_detection(self, detection: Detection, history: _TrackHistory) -> TCVMScore:
        if len(history) < self.config.min_history:
            return TCVMScore()

        confidence_score = self._confidence_score(detection, history)
        motion_score = self._motion_score(detection, history)
        feature_score = self._feature_score(detection, history)
        wc, wm, wf, wd = self.config.normalized_weights()
        total = wc * confidence_score + wm * motion_score + wf * feature_score
        return TCVMScore(
            confidence_instability=confidence_score,
            motion_inconsistency=motion_score,
            feature_inconsistency=feature_score,
            disappearance=0.0,
            total=float(np.clip(total, 0.0, 1.0)),
        )

    def _confidence_score(self, detection: Detection, history: _TrackHistory) -> float:
        mean_conf = history.ema_confidence if history.ema_confidence is not None else np.mean(history.confidences)
        sigma = float(np.sqrt(max(history.confidence_var, self.config.confidence_sigma_floor**2)))
        z_score = abs(detection.confidence - float(mean_conf)) / max(sigma, 1e-6)
        return float(np.clip(z_score / self.config.confidence_tau, 0.0, 1.0))

    def _motion_score(self, detection: Detection, history: _TrackHistory) -> float:
        predicted = history.predicted_box()
        if predicted is None:
            return 0.0
        flow_predicted = detection.metadata.get("flow_predicted_box")
        if flow_predicted is not None:
            flow_predicted = np.asarray(flow_predicted, dtype=float)
            iou = max(box_iou(predicted, detection.xyxy), box_iou(flow_predicted, detection.xyxy))
        else:
            iou = box_iou(predicted, detection.xyxy)
        return float(np.clip((1.0 - iou) / self.config.motion_tau, 0.0, 1.0))

    def _feature_score(self, detection: Detection, history: _TrackHistory) -> float:
        if detection.feature is None or history.ema_feature is None:
            return 0.0
        dissimilarity = (1.0 - cosine_similarity(detection.feature, history.ema_feature)) / 2.0
        return float(np.clip(dissimilarity / self.config.feature_tau, 0.0, 1.0))

    def _score_missing_tracks(self, seen_track_ids: set[int], frame_id: int) -> list[Detection]:
        _, _, _, wd = self.config.normalized_weights()
        events: list[Detection] = []
        for track_id, history in list(self.histories.items()):
            if track_id in seen_track_ids:
                continue
            if history.last_frame_id < 0 or frame_id <= history.last_frame_id:
                continue
            history.missed += 1
            if history.missed > self.config.disappearance_patience:
                continue
            if len(history) < self.config.min_history:
                continue
            if (history.ema_confidence or 0.0) < self.config.min_recovery_confidence:
                continue

            disappearance_score = float(np.clip(history.missed / self.config.disappearance_patience, 0.0, 1.0))
            total = float(np.clip(wd * disappearance_score + 0.5 * (history.ema_confidence or 0.0), 0.0, 1.0))
            predicted = history.predicted_box()
            if predicted is None:
                continue
            score = TCVMScore(disappearance=disappearance_score, total=total)
            events.append(
                Detection(
                    frame_id=frame_id,
                    class_id=history.class_id,
                    class_name=history.class_name,
                    confidence=float(history.ema_confidence or 0.0),
                    xyxy=predicted,
                    track_id=track_id,
                    anomaly_score=total,
                    is_adversarial=total >= self.config.anomaly_threshold,
                    is_recovered=False,
                    metadata={"tcvm": score.to_dict(), "event": "missing_track"},
                )
            )
        return events
