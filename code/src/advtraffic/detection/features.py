"""Lightweight ROI feature extraction for temporal consistency checks."""

from __future__ import annotations

import cv2
import numpy as np

from advtraffic.utils.geometry import clip_box


class ROIFeatureExtractor:
    """Extract compact color-gradient descriptors from object crops.

    The default descriptor is intentionally lightweight and does not require
    downloading an external embedding model. It combines HSV histograms with a
    Sobel orientation histogram, which is adequate for detecting abrupt
    appearance shifts caused by stickers, reflective patches, and occlusions.
    """

    def __init__(self, hsv_bins: tuple[int, int, int] = (12, 8, 8), orientation_bins: int = 8):
        self.hsv_bins = hsv_bins
        self.orientation_bins = orientation_bins

    def extract(self, frame: np.ndarray, boxes_xyxy: list[np.ndarray]) -> list[np.ndarray]:
        height, width = frame.shape[:2]
        features: list[np.ndarray] = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = clip_box(box, width, height).astype(int)
            crop = frame[y1:y2, x1:x2]
            features.append(self._describe_crop(crop))
        return features

    def _describe_crop(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros(sum(self.hsv_bins) + self.orientation_bins, dtype=np.float32)

        crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [self.hsv_bins[0]], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [self.hsv_bins[1]], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [self.hsv_bins[2]], [0, 256]).flatten()

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        orient_hist, _ = np.histogram(
            angle,
            bins=self.orientation_bins,
            range=(0.0, 360.0),
            weights=magnitude,
        )

        feature = np.concatenate([h_hist, s_hist, v_hist, orient_hist.astype(np.float32)]).astype(np.float32)
        norm = np.linalg.norm(feature)
        return feature / max(norm, 1e-8)


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 1.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 1.0
    return float(np.dot(a, b) / denom)
