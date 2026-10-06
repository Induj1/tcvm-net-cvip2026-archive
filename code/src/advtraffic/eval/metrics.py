"""Metrics for robustness, temporal anomaly detection, and speed."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np


def attack_success_rate(clean_detected: list[bool], adv_detected: list[bool]) -> float:
    """Fraction of originally detected targets that are missed after attack."""

    clean = np.asarray(clean_detected, dtype=bool)
    adv = np.asarray(adv_detected, dtype=bool)
    eligible = clean.sum()
    if eligible == 0:
        return 0.0
    return float(np.logical_and(clean, ~adv).sum() / eligible)


@dataclass
class BinaryMetrics:
    """Binary anomaly metrics computed from labels and scores."""

    threshold: float = 0.62
    y_true: list[int] = field(default_factory=list)
    y_score: list[float] = field(default_factory=list)

    def update(self, label: int, score: float) -> None:
        self.y_true.append(int(label))
        self.y_score.append(float(score))

    def compute(self) -> dict[str, float]:
        if not self.y_true:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "fpr": 0.0}
        y_true = np.asarray(self.y_true, dtype=int)
        y_pred = (np.asarray(self.y_score) >= self.threshold).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return {
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fp / max(fp + tn, 1),
        }


class LatencyMeter:
    """Collect per-frame latency and summarize FPS percentiles."""

    def __init__(self):
        self.samples_ms: list[float] = []

    def time(self):
        return _LatencyContext(self)

    def update_ms(self, elapsed_ms: float) -> None:
        self.samples_ms.append(float(elapsed_ms))

    def summary(self) -> dict[str, float]:
        if not self.samples_ms:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}
        arr = np.asarray(self.samples_ms, dtype=float)
        mean_ms = float(arr.mean())
        return {
            "mean_ms": mean_ms,
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "fps": 1000.0 / max(mean_ms, 1e-8),
        }


class _LatencyContext:
    def __init__(self, meter: LatencyMeter):
        self.meter = meter
        self.start = 0.0

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = (perf_counter() - self.start) * 1000.0
        self.meter.update_ms(elapsed)
