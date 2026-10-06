"""Final robust prediction layer for TCVM-scored detections."""

from __future__ import annotations

import numpy as np

from advtraffic.defense.tcvm import TCVMConfig, TemporalConsistencyVerifier
from advtraffic.detection.types import Detection


class RobustPredictionLayer:
    """Reject anomalous detections and recover short adversarial disappearances."""

    def __init__(self, tcvm: TemporalConsistencyVerifier | None = None, config: TCVMConfig | None = None):
        self.tcvm = tcvm or TemporalConsistencyVerifier(config)
        self.last_scored: list[Detection] = []
        self.last_missing_events: list[Detection] = []

    def process(self, detections: list[Detection], frame_id: int) -> list[Detection]:
        scored, missing_events = self.tcvm.verify(detections, frame_id)
        missing_events = self._promote_global_collapse_events(scored, missing_events)
        self.last_scored = scored
        self.last_missing_events = missing_events
        robust: list[Detection] = []

        for detection in scored:
            if detection.is_adversarial:
                recovered = self._recover_detection(detection)
                if recovered is not None:
                    robust.append(recovered)
            else:
                robust.append(detection)

        for event in missing_events:
            if event.is_adversarial:
                recovered = self._recover_missing_event(event)
                if recovered is not None:
                    robust.append(recovered)

        return robust

    def _promote_global_collapse_events(self, scored: list[Detection], missing_events: list[Detection]) -> list[Detection]:
        """Promote short missing-track events when the detector count collapses."""

        cfg = self.tcvm.config
        active_count = len(scored) + len(missing_events)
        if active_count == 0 or len(missing_events) < cfg.collapse_min_missing:
            return missing_events

        detection_ratio = len(scored) / max(active_count, 1)
        if detection_ratio > cfg.collapse_max_detection_ratio:
            return missing_events

        promoted: list[Detection] = []
        for event in missing_events:
            if event.anomaly_score >= cfg.collapse_recovery_threshold:
                promoted.append(
                    event.copy(
                        anomaly_score=max(event.anomaly_score, cfg.anomaly_threshold),
                        is_adversarial=True,
                        metadata={**event.metadata, "collapse_gate": "frame_level_detector_count_drop"},
                    )
                )
            else:
                promoted.append(event)
        return promoted

    def _recover_detection(self, detection: Detection) -> Detection | None:
        history = self.tcvm.get_history(detection.track_id)
        if history is None or history.predicted_box() is None:
            return None
        confidence = float(max(history.ema_confidence or 0.0, detection.confidence) * 0.85)
        return detection.copy(
            confidence=confidence,
            xyxy=history.predicted_box().astype(float),
            is_recovered=True,
            metadata={**detection.metadata, "recovery": "constant_velocity_temporal_prior"},
        )

    def _recover_missing_event(self, event: Detection) -> Detection | None:
        history = self.tcvm.get_history(event.track_id)
        if history is None:
            return None
        predicted = history.predicted_box()
        if predicted is None or not np.isfinite(predicted).all():
            return None
        return event.copy(
            xyxy=predicted.astype(float),
            confidence=float((history.ema_confidence or event.confidence) * 0.8),
            is_recovered=True,
            metadata={**event.metadata, "recovery": "short_gap_temporal_interpolation"},
        )
