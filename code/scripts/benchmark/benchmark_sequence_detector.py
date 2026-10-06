"""Evaluate a YOLO detector on an image sequence with YOLO labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from advtraffic.detection import YOLOv8Engine
from advtraffic.eval.map_metrics import compute_map, detections_to_prediction, labels_to_target
from advtraffic.eval.metrics import LatencyMeter
from advtraffic.utils.io import IMAGE_EXTENSIONS, iter_files, read_image, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 on a labeled image sequence.")
    parser.add_argument("--sequence-root", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--classes", nargs="*", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.sequence_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list(iter_files(root / "images", IMAGE_EXTENSIONS))
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    engine = YOLOv8Engine(model_path=args.model, imgsz=args.imgsz, conf=args.conf, device=args.device, classes=args.classes)
    latency = LatencyMeter()
    predictions = []
    targets = []
    rows = []

    for frame_id, image_path in enumerate(tqdm(image_paths, desc="YOLO sequence")):
        image = read_image(image_path)
        height, width = image.shape[:2]
        with latency.time():
            detections = engine.detect(image, frame_id=frame_id)
        predictions.append(detections_to_prediction(detections))
        targets.append(labels_to_target(root / "labels" / f"{image_path.stem}.txt", width, height))
        rows.append(
            {
                "frame_id": frame_id,
                "image": str(image_path),
                "detections": len(detections),
                "mean_confidence": sum(det.confidence for det in detections) / max(len(detections), 1),
                "max_confidence": max((det.confidence for det in detections), default=0.0),
            }
        )

    map_metrics = compute_map(predictions, targets) if predictions else {"map50": 0.0, "map50_95": 0.0, "mar100": 0.0}
    summary = {
        "method": "YOLOv8",
        "sequence_root": str(root),
        "frames": len(image_paths),
        "classes": args.classes,
        "map": map_metrics,
        "latency": latency.summary(),
    }
    pd.DataFrame(rows).to_csv(output_dir / "frame_metrics.csv", index=False)
    write_json(output_dir / "metrics.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
