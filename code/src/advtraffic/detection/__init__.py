"""Detection backbones and detection data structures."""

from .types import Detection
from .yolo_engine import YOLOv8Engine

__all__ = ["Detection", "YOLOv8Engine"]
