"""Physical attack simulators for traffic surveillance frames."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from advtraffic.utils.geometry import clip_box


def _region_from_box(image: np.ndarray, box: np.ndarray, scale: float = 1.0) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box.astype(float)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    scaled = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dtype=float)
    return tuple(clip_box(scaled, width, height).astype(int))


def apply_patch(
    image: np.ndarray,
    box: np.ndarray,
    patch: np.ndarray,
    scale: float = 0.45,
    alpha: float = 0.95,
    vertical_offset: float = -0.18,
) -> np.ndarray:
    """Overlay a printable patch near the helmet/rider region."""

    out = image.copy()
    x1, y1, x2, y2 = _region_from_box(out, box, scale=scale)
    if x2 <= x1 or y2 <= y1:
        return out
    dy = int((y2 - y1) * vertical_offset)
    y1 = max(0, y1 + dy)
    y2 = min(out.shape[0] - 1, y2 + dy)
    patch_resized = cv2.resize(patch, (max(1, x2 - x1), max(1, y2 - y1)), interpolation=cv2.INTER_LINEAR)
    if patch_resized.shape[2] == 4:
        mask = patch_resized[:, :, 3:4].astype(float) / 255.0
        patch_rgb = patch_resized[:, :, :3]
    else:
        mask = np.ones_like(patch_resized[:, :, :1], dtype=float)
        patch_rgb = patch_resized
    blend = alpha * mask
    out[y1:y2, x1:x2] = (blend * patch_rgb[: y2 - y1, : x2 - x1] + (1 - blend) * out[y1:y2, x1:x2]).astype(np.uint8)
    return out


def make_printable_patch(size: int = 96, palette: str = "high_contrast") -> np.ndarray:
    """Create a high-contrast, printer-friendly synthetic patch."""

    patch = np.zeros((size, size, 3), dtype=np.uint8)
    if palette == "traffic":
        colors = [(255, 255, 255), (0, 0, 0), (0, 255, 255), (0, 0, 255)]
    else:
        colors = [(255, 255, 255), (0, 0, 0), (255, 0, 255), (0, 255, 255)]
    cell = max(4, size // 8)
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            patch[y : y + cell, x : x + cell] = colors[((x // cell) + 2 * (y // cell)) % len(colors)]
    cv2.circle(patch, (size // 2, size // 2), size // 4, colors[-1], thickness=-1)
    cv2.line(patch, (0, size - 1), (size - 1, 0), colors[1], thickness=max(2, size // 16))
    return patch


def load_or_make_patch(path: str | Path | None = None, size: int = 96) -> np.ndarray:
    if path is not None:
        patch = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if patch is None:
            raise FileNotFoundError(f"Could not read patch: {path}")
        return patch
    return make_printable_patch(size=size)


def apply_sticker_attack(image: np.ndarray, box: np.ndarray, scale: float = 0.38) -> np.ndarray:
    patch = make_printable_patch(size=96, palette="traffic")
    return apply_patch(image, box, patch, scale=scale, alpha=0.98, vertical_offset=-0.24)


def apply_reflective_pattern_attack(
    image: np.ndarray,
    box: np.ndarray,
    intensity: float = 0.82,
    stripe_width: int = 8,
    scale: float = 0.9,
) -> np.ndarray:
    """Simulate reflective tape or glare-like adversarial patterns."""

    out = image.copy()
    x1, y1, x2, y2 = _region_from_box(out, box, scale=scale)
    if x2 <= x1 or y2 <= y1:
        return out
    roi = out[y1:y2, x1:x2].astype(np.float32)
    overlay = np.zeros_like(roi)
    h, w = roi.shape[:2]
    for offset in range(-h, w, stripe_width * 3):
        cv2.line(overlay, (offset, h), (offset + h, 0), (255, 255, 255), thickness=stripe_width)
    glare = cv2.GaussianBlur(overlay, (0, 0), sigmaX=3)
    out[y1:y2, x1:x2] = np.clip((1 - intensity) * roi + intensity * glare, 0, 255).astype(np.uint8)
    return out


def apply_occlusion_attack(
    image: np.ndarray,
    box: np.ndarray,
    occlusion_ratio: float = 0.35,
    color: tuple[int, int, int] = (24, 24, 24),
) -> np.ndarray:
    out = image.copy()
    x1, y1, x2, y2 = _region_from_box(out, box, scale=1.0)
    w, h = x2 - x1, y2 - y1
    ow, oh = int(w * math.sqrt(occlusion_ratio)), int(h * math.sqrt(occlusion_ratio))
    ox1 = x1 + max(0, (w - ow) // 2)
    oy1 = y1 + max(0, (h - oh) // 2)
    cv2.rectangle(out, (ox1, oy1), (min(x2, ox1 + ow), min(y2, oy1 + oh)), color, thickness=-1)
    return out


def apply_motion_blur(image: np.ndarray, kernel_size: int = 17, angle: float = 0.0) -> np.ndarray:
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))
    kernel = kernel / max(kernel.sum(), 1e-8)
    return cv2.filter2D(image, -1, kernel)


def apply_low_light(image: np.ndarray, gamma: float = 1.8, gain: float = 0.72) -> np.ndarray:
    normalized = np.clip(image.astype(np.float32) / 255.0, 0, 1)
    adjusted = gain * np.power(normalized, gamma)
    return np.clip(adjusted * 255.0, 0, 255).astype(np.uint8)


def apply_named_physical_attack(image: np.ndarray, box: np.ndarray, attack_name: str, **kwargs) -> np.ndarray:
    if attack_name == "sticker":
        return apply_sticker_attack(image, box, **kwargs)
    if attack_name == "reflective":
        return apply_reflective_pattern_attack(image, box, **kwargs)
    if attack_name == "occlusion":
        return apply_occlusion_attack(image, box, **kwargs)
    if attack_name == "patch":
        patch = load_or_make_patch(kwargs.pop("patch_path", None), size=kwargs.pop("patch_size", 96))
        return apply_patch(image, box, patch, **kwargs)
    raise ValueError(f"Unknown physical attack: {attack_name}")
