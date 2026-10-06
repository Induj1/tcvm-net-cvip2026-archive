"""Video-level evaluation routines for baseline and TCVM pipelines."""

from __future__ import annotations

from pathlib import Path

from advtraffic.defense import TCVMNetPipeline
from advtraffic.detection import YOLOv8Engine
from advtraffic.eval.metrics import LatencyMeter
from advtraffic.tracking import SimpleIoUTracker
from advtraffic.utils.io import write_json
from advtraffic.utils.video import iter_video_frames


def evaluate_video(
    video_path: str | Path,
    model_path: str | Path,
    output_json: str | Path,
    use_tcvm: bool = True,
    use_bytetrack: bool = True,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str | None = None,
    max_frames: int | None = None,
) -> dict:
    engine = YOLOv8Engine(model_path=model_path, imgsz=imgsz, conf=conf, device=device)
    fallback_tracker = SimpleIoUTracker()
    robust_layer = TCVMNetPipeline(use_optical_flow=True) if use_tcvm else None
    latency = LatencyMeter()
    frames = []

    for frame in iter_video_frames(video_path, max_frames=max_frames):
        with latency.time():
            if use_bytetrack:
                detections = engine.track(frame.image, frame_id=frame.frame_id)
            else:
                detections = fallback_tracker.update(engine.detect(frame.image, frame_id=frame.frame_id))
            final = robust_layer.process_frame(frame.image, detections, frame.frame_id) if robust_layer else detections

        frames.append(
            {
                "frame_id": frame.frame_id,
                "timestamp_ms": frame.timestamp_ms,
                "detections": [det.to_dict() for det in final],
            }
        )

    report = {"video": str(video_path), "use_tcvm": use_tcvm, "latency": latency.summary(), "frames": frames}
    write_json(output_json, report)
    return report
