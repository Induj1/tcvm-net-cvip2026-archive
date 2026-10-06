"""Geometry utilities for detections and tracks."""

from __future__ import annotations

import numpy as np


def xyxy_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box.astype(float)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection over union for two xyxy boxes."""

    ax1, ay1, ax2, ay2 = box_a.astype(float)
    bx1, by1, bx2, by2 = box_b.astype(float)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = xyxy_area(np.array([ix1, iy1, ix2, iy2], dtype=float))
    union = xyxy_area(box_a) + xyxy_area(box_b) - inter
    return 0.0 if union <= 0 else float(inter / union)


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = box.astype(float).copy()
    clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0, max(0, width - 1))
    clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0, max(0, height - 1))
    if clipped[2] < clipped[0]:
        clipped[0], clipped[2] = clipped[2], clipped[0]
    if clipped[3] < clipped[1]:
        clipped[1], clipped[3] = clipped[3], clipped[1]
    return clipped


def box_center(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(float)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)


def normalized_center_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ca = box_center(box_a)
    cb = box_center(box_b)
    diag = np.linalg.norm([max(box_a[2] - box_a[0], 1.0), max(box_a[3] - box_a[1], 1.0)])
    return float(np.linalg.norm(ca - cb) / max(diag, 1e-6))


def yolo_to_xyxy(label: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert one YOLO label row [cls, xc, yc, w, h] to xyxy pixels."""

    _, xc, yc, bw, bh = label.astype(float)
    x1 = (xc - bw / 2.0) * width
    y1 = (yc - bh / 2.0) * height
    x2 = (xc + bw / 2.0) * width
    y2 = (yc + bh / 2.0) * height
    return clip_box(np.array([x1, y1, x2, y2], dtype=float), width, height)


def xyxy_to_yolo(box: np.ndarray, class_id: int, width: int, height: int) -> str:
    x1, y1, x2, y2 = clip_box(box, width, height)
    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = max(0.0, x2 - x1) / width
    bh = max(0.0, y2 - y1) / height
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
