"""Image-only defense baselines for comparison with TCVM-Net."""

from __future__ import annotations

import cv2
import numpy as np


def jpeg_compression(image: np.ndarray, quality: int = 75) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return image if decoded is None else decoded


def median_smoothing(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return cv2.medianBlur(image, max(3, int(kernel_size) | 1))


def brightness_normalization(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def image_only_defense(image: np.ndarray, mode: str = "jpeg_median_clahe") -> np.ndarray:
    if mode == "jpeg":
        return jpeg_compression(image)
    if mode == "median":
        return median_smoothing(image)
    if mode == "clahe":
        return brightness_normalization(image)
    if mode == "jpeg_median_clahe":
        return brightness_normalization(median_smoothing(jpeg_compression(image)))
    raise ValueError(f"Unknown image-only defense mode: {mode}")
