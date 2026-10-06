"""Optical-flow motion prediction for TCVM."""

from __future__ import annotations

import cv2
import numpy as np

from advtraffic.detection.types import Detection
from advtraffic.utils.geometry import clip_box


class OpticalFlowMotionEstimator:
    """Estimate short-term box displacement using dense Farneback flow."""

    def __init__(self, scale: float = 1.0):
        if scale <= 0:
            raise ValueError("Optical-flow scale must be positive.")
        self.scale = min(float(scale), 1.0)
        self.prev_gray: np.ndarray | None = None

    def annotate(self, frame_bgr: np.ndarray, detections: list[Detection]) -> list[Detection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        flow_gray = self._resize_for_flow(gray)
        if self.prev_gray is None or self.prev_gray.shape != flow_gray.shape:
            self.prev_gray = flow_gray
            return detections

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            flow_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        height, width = gray.shape[:2]
        flow_height, flow_width = flow_gray.shape[:2]
        annotated: list[Detection] = []
        for det in detections:
            x1, y1, x2, y2 = clip_box(det.xyxy, width, height).astype(int)
            fx1, fy1, fx2, fy2 = self._box_to_flow_coords((x1, y1, x2, y2), flow_width, flow_height)
            roi_flow = flow[fy1:fy2, fx1:fx2]
            if roi_flow.size == 0:
                annotated.append(det)
                continue
            displacement = np.median(roi_flow.reshape(-1, 2), axis=0)
            if self.scale != 1.0:
                displacement = displacement / self.scale
            predicted = clip_box(det.xyxy + np.array([*displacement, *displacement], dtype=float), width, height)
            metadata = {**det.metadata, "flow_predicted_box": predicted.tolist(), "flow_displacement": displacement.tolist()}
            annotated.append(det.copy(metadata=metadata))
        self.prev_gray = flow_gray
        return annotated

    def _resize_for_flow(self, gray: np.ndarray) -> np.ndarray:
        if self.scale == 1.0:
            return gray
        width = max(int(round(gray.shape[1] * self.scale)), 16)
        height = max(int(round(gray.shape[0] * self.scale)), 16)
        return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    def _box_to_flow_coords(self, box: tuple[int, int, int, int], flow_width: int, flow_height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        if self.scale == 1.0:
            return x1, y1, x2, y2
        fx1 = int(np.clip(round(x1 * self.scale), 0, flow_width - 1))
        fy1 = int(np.clip(round(y1 * self.scale), 0, flow_height - 1))
        fx2 = int(np.clip(round(x2 * self.scale), fx1 + 1, flow_width))
        fy2 = int(np.clip(round(y2 * self.scale), fy1 + 1, flow_height))
        return fx1, fy1, fx2, fy2
