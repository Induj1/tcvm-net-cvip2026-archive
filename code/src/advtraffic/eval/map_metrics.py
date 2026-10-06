"""Detection mAP helpers backed by TorchMetrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from advtraffic.detection.types import Detection
from advtraffic.utils.geometry import yolo_to_xyxy


def labels_to_target(label_path: str | Path, width: int, height: int) -> dict[str, torch.Tensor]:
    boxes = []
    labels = []
    path = Path(label_path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            row = np.asarray([float(v) for v in parts], dtype=float)
            boxes.append(yolo_to_xyxy(row, width, height))
            labels.append(int(row[0]))
    return {
        "boxes": torch.tensor(np.asarray(boxes, dtype=np.float32)).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def detections_to_prediction(detections: list[Detection]) -> dict[str, torch.Tensor]:
    boxes = np.asarray([det.xyxy for det in detections], dtype=np.float32).reshape(-1, 4)
    scores = np.asarray([det.confidence for det in detections], dtype=np.float32)
    labels = np.asarray([det.class_id for det in detections], dtype=np.int64)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def compute_map(predictions: list[dict[str, torch.Tensor]], targets: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError as exc:
        raise ImportError("Install torchmetrics[detection] and pycocotools to compute mAP.") from exc

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(predictions, targets)
    result = metric.compute()
    return {
        "map50": float(result["map_50"]),
        "map50_95": float(result["map"]),
        "mar100": float(result["mar_100"]),
    }
