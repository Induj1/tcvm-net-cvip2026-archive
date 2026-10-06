import numpy as np

from advtraffic.utils.geometry import box_iou, xyxy_to_yolo, yolo_to_xyxy


def test_box_iou_identity():
    box = np.array([10, 20, 50, 80], dtype=float)
    assert box_iou(box, box) == 1.0


def test_yolo_roundtrip():
    box = np.array([10, 20, 50, 80], dtype=float)
    label = xyxy_to_yolo(box, class_id=2, width=100, height=100)
    row = np.asarray([float(v) for v in label.split()])
    restored = yolo_to_xyxy(row, width=100, height=100)
    assert np.allclose(restored, box)
