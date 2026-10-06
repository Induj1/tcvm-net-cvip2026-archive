"""PyTorch implementation of TCVM anomaly scoring components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TCVMNetWeights:
    confidence: float = 0.34
    motion: float = 0.28
    feature: float = 0.26
    disappearance: float = 0.12

    def tensor(self, device: torch.device) -> torch.Tensor:
        weights = torch.tensor([self.confidence, self.motion, self.feature, self.disappearance], device=device)
        return weights / weights.sum().clamp_min(1e-8)


class ConfidenceStabilityModule(nn.Module):
    def __init__(self, tau: float = 2.5, sigma_floor: float = 0.05):
        super().__init__()
        self.tau = tau
        self.sigma_floor = sigma_floor

    def forward(self, confidence: torch.Tensor, ema: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        sigma = variance.clamp_min(self.sigma_floor**2).sqrt()
        z_score = (confidence - ema).abs() / sigma.clamp_min(1e-8)
        return (z_score / self.tau).clamp(0.0, 1.0)


class MotionContinuityModule(nn.Module):
    def __init__(self, tau: float = 0.7):
        super().__init__()
        self.tau = tau

    def forward(self, boxes_xyxy: torch.Tensor, predicted_xyxy: torch.Tensor) -> torch.Tensor:
        iou = box_iou_torch(boxes_xyxy, predicted_xyxy)
        return ((1.0 - iou) / self.tau).clamp(0.0, 1.0)


class FeatureSimilarityModule(nn.Module):
    def __init__(self, tau: float = 0.45):
        super().__init__()
        self.tau = tau

    def forward(self, features: torch.Tensor, ema_features: torch.Tensor) -> torch.Tensor:
        similarity = torch.nn.functional.cosine_similarity(features, ema_features, dim=-1)
        dissimilarity = (1.0 - similarity) / 2.0
        return (dissimilarity / self.tau).clamp(0.0, 1.0)


class TemporalAnomalyScoringModule(nn.Module):
    def __init__(self, weights: TCVMNetWeights | None = None):
        super().__init__()
        self.weights = weights or TCVMNetWeights()

    def forward(
        self,
        confidence_score: torch.Tensor,
        motion_score: torch.Tensor,
        feature_score: torch.Tensor,
        disappearance_score: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if disappearance_score is None:
            disappearance_score = torch.zeros_like(confidence_score)
        components = torch.stack([confidence_score, motion_score, feature_score, disappearance_score], dim=-1)
        weights = self.weights.tensor(components.device)
        return (components * weights).sum(dim=-1).clamp(0.0, 1.0)


class TCVMNet(nn.Module):
    """Differentiable TCVM score module for ablations and batched analysis."""

    def __init__(
        self,
        confidence_tau: float = 2.5,
        motion_tau: float = 0.7,
        feature_tau: float = 0.45,
        weights: TCVMNetWeights | None = None,
    ):
        super().__init__()
        self.confidence = ConfidenceStabilityModule(tau=confidence_tau)
        self.motion = MotionContinuityModule(tau=motion_tau)
        self.feature = FeatureSimilarityModule(tau=feature_tau)
        self.scorer = TemporalAnomalyScoringModule(weights=weights)

    def forward(
        self,
        confidence: torch.Tensor,
        confidence_ema: torch.Tensor,
        confidence_var: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        predicted_xyxy: torch.Tensor,
        features: torch.Tensor,
        feature_ema: torch.Tensor,
        disappearance: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        s_conf = self.confidence(confidence, confidence_ema, confidence_var)
        s_motion = self.motion(boxes_xyxy, predicted_xyxy)
        s_feature = self.feature(features, feature_ema)
        total = self.scorer(s_conf, s_motion, s_feature, disappearance)
        return {
            "confidence_instability": s_conf,
            "motion_inconsistency": s_motion,
            "feature_inconsistency": s_feature,
            "anomaly_score": total,
        }


def box_iou_torch(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(boxes_a[..., 0], boxes_b[..., 0])
    y1 = torch.maximum(boxes_a[..., 1], boxes_b[..., 1])
    x2 = torch.minimum(boxes_a[..., 2], boxes_b[..., 2])
    y2 = torch.minimum(boxes_a[..., 3], boxes_b[..., 3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area_a = (boxes_a[..., 2] - boxes_a[..., 0]).clamp_min(0) * (boxes_a[..., 3] - boxes_a[..., 1]).clamp_min(0)
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]).clamp_min(0) * (boxes_b[..., 3] - boxes_b[..., 1]).clamp_min(0)
    return inter / (area_a + area_b - inter).clamp_min(1e-8)
