"""Sequential TCVM-Net video inference pipeline."""

from __future__ import annotations

from advtraffic.defense.optical_flow import OpticalFlowMotionEstimator
from advtraffic.defense.robust_predictor import RobustPredictionLayer
from advtraffic.detection.types import Detection


class TCVMNetPipeline:
    """Apply optical flow, temporal verification, rejection, and recovery."""

    def __init__(
        self,
        robust_layer: RobustPredictionLayer | None = None,
        use_optical_flow: bool = True,
        flow_scale: float = 1.0,
    ):
        self.robust_layer = robust_layer or RobustPredictionLayer()
        self.flow = OpticalFlowMotionEstimator(scale=flow_scale) if use_optical_flow else None

    def process_frame(self, frame_bgr, detections: list[Detection], frame_id: int) -> list[Detection]:
        if self.flow is not None:
            detections = self.flow.annotate(frame_bgr, detections)
        return self.robust_layer.process(detections, frame_id)
