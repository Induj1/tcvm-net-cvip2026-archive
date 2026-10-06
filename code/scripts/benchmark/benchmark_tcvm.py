"""Sequential TCVM-Net benchmark over clean or attacked image sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from advtraffic.defense import TCVMConfig, TCVMNetPipeline
from advtraffic.defense.robust_predictor import RobustPredictionLayer
from advtraffic.defense.tcvm import TemporalConsistencyVerifier
from advtraffic.detection import YOLOv8Engine
from advtraffic.eval.map_metrics import compute_map, detections_to_prediction, labels_to_target
from advtraffic.eval.metrics import BinaryMetrics, LatencyMeter
from advtraffic.tracking import SimpleIoUTracker
from advtraffic.utils.io import IMAGE_EXTENSIONS, iter_files, read_image, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TCVM-Net on sequential frames.")
    parser.add_argument("--sequence-root", required=True, help="Root containing images/ and labels/ directories.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="outputs/results/tcvm")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--classes", nargs="*", type=int, default=None)
    parser.add_argument("--adversarial-label", type=int, default=1, help="1 for attacked sequence, 0 for clean sequence.")
    parser.add_argument("--frame-labels", default=None, help="Optional CSV with frame_id/adversarial_label columns.")
    parser.add_argument("--anomaly-threshold", type=float, default=0.62)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--fallback-iou-tracker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optical-flow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flow-scale", type=float, default=1.0, help="Scale factor for dense optical flow.")
    return parser.parse_args()


def load_frame_metadata(path: str | None, default_label: int) -> dict[int, dict]:
    if path is None:
        return {}
    df = pd.read_csv(path)
    if "frame_id" not in df.columns:
        raise ValueError("Frame-label CSV must contain a frame_id column.")
    label_col = "adversarial_label" if "adversarial_label" in df.columns else "attacked"
    if label_col not in df.columns:
        raise ValueError("Frame-label CSV must contain adversarial_label or attacked.")
    metadata = {}
    for row in df.itertuples(index=False):
        frame_id = int(getattr(row, "frame_id"))
        item = {"adversarial_label": int(getattr(row, label_col))}
        if "clip_id" in df.columns:
            item["clip_id"] = int(getattr(row, "clip_id"))
        if "clip" in df.columns:
            item["clip"] = str(getattr(row, "clip"))
        metadata[frame_id] = item
    return metadata


def build_pipeline(threshold: float, use_optical_flow: bool, flow_scale: float) -> TCVMNetPipeline:
    cfg = TCVMConfig(anomaly_threshold=threshold)
    return TCVMNetPipeline(
        RobustPredictionLayer(TemporalConsistencyVerifier(cfg)),
        use_optical_flow=use_optical_flow,
        flow_scale=flow_scale,
    )


def main() -> None:
    args = parse_args()
    root = Path(args.sequence_root)
    image_paths = list(iter_files(root / "images", IMAGE_EXTENSIONS))
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(args.anomaly_threshold, args.optical_flow, args.flow_scale)
    engine = YOLOv8Engine(model_path=args.model, imgsz=args.imgsz, conf=args.conf, device=args.device, classes=args.classes)
    tracker = SimpleIoUTracker()
    latency = LatencyMeter()
    anomaly_metrics = BinaryMetrics(threshold=args.anomaly_threshold)
    frame_metadata = load_frame_metadata(args.frame_labels, args.adversarial_label)
    predictions = []
    targets = []
    frame_rows = []
    previous_clip_id = None

    for frame_id, image_path in enumerate(tqdm(image_paths, desc="TCVM")):
        meta = frame_metadata.get(frame_id, {"adversarial_label": args.adversarial_label})
        clip_id = meta.get("clip_id")
        if clip_id is not None and previous_clip_id is not None and clip_id != previous_clip_id:
            tracker = SimpleIoUTracker()
            pipeline = build_pipeline(args.anomaly_threshold, args.optical_flow, args.flow_scale)
        previous_clip_id = clip_id

        image = read_image(image_path)
        height, width = image.shape[:2]
        with latency.time():
            if args.fallback_iou_tracker:
                detections = tracker.update(engine.detect(image, frame_id=frame_id))
            else:
                detections = engine.track(image, frame_id=frame_id)
            robust = pipeline.process_frame(image, detections, frame_id)

        predictions.append(detections_to_prediction(robust))
        targets.append(labels_to_target(root / "labels" / f"{image_path.stem}.txt", width, height))
        scored_events = pipeline.robust_layer.last_scored + pipeline.robust_layer.last_missing_events
        max_anomaly = max((det.anomaly_score for det in scored_events), default=0.0)
        frame_label = int(meta.get("adversarial_label", args.adversarial_label))
        anomaly_metrics.update(frame_label, max_anomaly)
        frame_rows.append(
            {
                "frame_id": frame_id,
                "clip_id": clip_id,
                "clip": meta.get("clip"),
                "image": str(image_path),
                "adversarial_label": frame_label,
                "detections": len(detections),
                "robust_detections": len(robust),
                "mean_raw_confidence": sum(det.confidence for det in detections) / max(len(detections), 1),
                "max_raw_confidence": max((det.confidence for det in detections), default=0.0),
                "max_anomaly_score": max_anomaly,
                "missing_events": len(pipeline.robust_layer.last_missing_events),
                "mean_confidence": sum(det.confidence for det in robust) / max(len(robust), 1),
                "recovered": sum(int(det.is_recovered) for det in robust),
            }
        )

    map_metrics = compute_map(predictions, targets) if predictions else {"map50": 0.0, "map50_95": 0.0, "mar100": 0.0}
    summary = {
        "method": "TCVM-Net",
        "sequence_root": str(root),
        "frames": len(image_paths),
        "classes": args.classes,
        "optical_flow": args.optical_flow,
        "flow_scale": args.flow_scale if args.optical_flow else 0.0,
        "map": map_metrics,
        "latency": latency.summary(),
        "anomaly": anomaly_metrics.compute(),
    }
    pd.DataFrame(frame_rows).to_csv(output_dir / "tcvm_frame_metrics.csv", index=False)
    write_json(output_dir / "tcvm_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
