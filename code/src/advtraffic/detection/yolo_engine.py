"""YOLOv8 inference and tracking wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from advtraffic.detection.features import ROIFeatureExtractor
from advtraffic.detection.types import Detection


class YOLOv8Engine:
    """Thin wrapper around Ultralytics YOLOv8 for detection and ByteTrack IDs."""

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        device: str | int | None = None,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        classes: Sequence[int] | None = None,
        tracker: str = "bytetrack.yaml",
        extract_features: bool = True,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Install ultralytics to use YOLOv8Engine: pip install ultralytics") from exc

        self.model = YOLO(str(model_path))
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.classes = list(classes) if classes is not None else None
        self.tracker = tracker
        self.feature_extractor = ROIFeatureExtractor() if extract_features else None
        self.names = self.model.names if hasattr(self.model, "names") else {}

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> list[Detection]:
        result = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )[0]
        return self._result_to_detections(result, frame, frame_id)

    def track(self, frame: np.ndarray, frame_id: int = 0, persist: bool = True) -> list[Detection]:
        result = self.model.track(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            tracker=self.tracker,
            persist=persist,
            verbose=False,
        )[0]
        return self._result_to_detections(result, frame, frame_id)

    def raw_torch_model(self):
        """Return the underlying PyTorch module used by gradient attacks."""

        return self.model.model

    def _result_to_detections(self, result, frame: np.ndarray, frame_id: int) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        ids = None
        if getattr(boxes, "id", None) is not None:
            ids = boxes.id.detach().cpu().numpy().astype(int)

        features = None
        if self.feature_extractor is not None:
            features = self.feature_extractor.extract(frame, [box for box in xyxy])

        detections: list[Detection] = []
        for index, box in enumerate(xyxy):
            class_id = int(classes[index])
            detections.append(
                Detection(
                    frame_id=frame_id,
                    class_id=class_id,
                    class_name=str(self.names.get(class_id, class_id)),
                    confidence=float(confs[index]),
                    xyxy=box.astype(float),
                    track_id=None if ids is None else int(ids[index]),
                    feature=None if features is None else features[index],
                )
            )
        return detections
