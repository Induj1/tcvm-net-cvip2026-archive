"""Run robust TCVM inference on a traffic video and render an annotated video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from advtraffic.defense import TCVMNetPipeline
from advtraffic.detection import YOLOv8Engine
from advtraffic.tracking import SimpleIoUTracker
from advtraffic.utils.video import VideoWriter, iter_video_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8+TCVM video inference.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="results/annotated_tcvm.mp4")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fallback-iou-tracker", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def draw_detections(frame, detections):
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.xyxy.astype(int)
        color = (0, 180, 0)
        if det.is_recovered:
            color = (0, 215, 255)
        if det.is_adversarial and not det.is_recovered:
            color = (0, 0, 255)
        label = f"{det.class_name}#{det.track_id} {det.confidence:.2f} A:{det.anomaly_score:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    engine = YOLOv8Engine(model_path=args.model, imgsz=args.imgsz, conf=args.conf, device=args.device)
    robust = TCVMNetPipeline(use_optical_flow=True)
    tracker = SimpleIoUTracker()

    first_frame = next(iter_video_frames(args.video, max_frames=1))
    height, width = first_frame.image.shape[:2]
    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    capture.release()

    with VideoWriter(args.output, fps=fps, frame_size=(width, height)) as writer:
        for frame in iter_video_frames(args.video, max_frames=args.max_frames):
            if args.fallback_iou_tracker:
                detections = tracker.update(engine.detect(frame.image, frame.frame_id))
            else:
                detections = engine.track(frame.image, frame.frame_id)
            final = robust.process_frame(frame.image, detections, frame.frame_id)
            writer.write(draw_detections(frame.image, final))
    print(f"Wrote annotated video: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
