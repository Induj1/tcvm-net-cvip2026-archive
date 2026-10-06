"""ADAV-style reference-frame consistency baseline for temporal object defense.

This is not a reproduction of ADAV. It implements a fair lightweight baseline
inspired by ADAV's previous/reference-frame consistency idea: detect abrupt
object-vanishing events from detector output changes, then recover missing
objects from the last stable reference frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from advtraffic.detection import YOLOv8Engine
from advtraffic.detection.types import Detection
from advtraffic.eval.map_metrics import compute_map, detections_to_prediction, labels_to_target
from advtraffic.eval.metrics import BinaryMetrics, LatencyMeter
from advtraffic.tracking import SimpleIoUTracker
from advtraffic.utils.geometry import box_iou
from advtraffic.utils.io import IMAGE_EXTENSIONS, iter_files, read_image, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ADAV-style reference-frame baseline.")
    parser.add_argument("--sequence-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="outputs/results_50_complete/adav_style_reference")
    parser.add_argument("--frame-labels", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--device", default=None)
    parser.add_argument("--classes", nargs="*", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.30)
    parser.add_argument("--min-reference-confidence", type=float, default=0.25)
    parser.add_argument("--recovery-confidence-scale", type=float, default=0.90)
    parser.add_argument("--max-recovered", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def load_frame_metadata(path: str | None) -> dict[int, dict]:
    if path is None:
        return {}
    df = pd.read_csv(path)
    label_col = "adversarial_label" if "adversarial_label" in df.columns else "attacked"
    if "frame_id" not in df.columns or label_col not in df.columns:
        raise ValueError("Frame-label CSV must contain frame_id and adversarial_label or attacked.")
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


def mean_confidence(detections: list[Detection]) -> float:
    return float(sum(det.confidence for det in detections) / max(len(detections), 1))


def match_reference(reference: list[Detection], current: list[Detection], match_iou: float) -> tuple[list[Detection], list[Detection]]:
    unmatched_current = set(range(len(current)))
    matched: list[Detection] = []
    missing: list[Detection] = []
    for ref in sorted(reference, key=lambda det: det.confidence, reverse=True):
        best_idx = None
        best_iou = 0.0
        for idx in list(unmatched_current):
            det = current[idx]
            if det.class_id != ref.class_id:
                continue
            iou = box_iou(ref.xyxy, det.xyxy)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is None or best_iou < match_iou:
            missing.append(ref)
        else:
            unmatched_current.remove(best_idx)
            matched.append(ref)
    return matched, missing


def recover_missing(missing: list[Detection], frame_id: int, score: float, scale: float, max_recovered: int) -> list[Detection]:
    recovered = []
    for det in sorted(missing, key=lambda item: item.confidence, reverse=True)[:max_recovered]:
        recovered.append(
            det.copy(
                frame_id=frame_id,
                confidence=max(0.01, min(1.0, det.confidence * scale)),
                anomaly_score=score,
                is_adversarial=True,
                is_recovered=True,
                metadata={**det.metadata, "baseline": "adav_style_reference_recovery"},
            )
        )
    return recovered


def main() -> None:
    args = parse_args()
    root = Path(args.sequence_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list(iter_files(root / "images", IMAGE_EXTENSIONS))
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    engine = YOLOv8Engine(model_path=args.model, imgsz=args.imgsz, conf=args.conf, device=args.device, classes=args.classes)
    tracker = SimpleIoUTracker()
    latency = LatencyMeter()
    anomaly = BinaryMetrics(threshold=args.threshold)
    metadata = load_frame_metadata(args.frame_labels)
    previous_clip_id = None
    reference: list[Detection] = []
    reference_count = 0
    reference_confidence = 0.0
    predictions = []
    targets = []
    rows = []

    for frame_id, image_path in enumerate(tqdm(image_paths, desc="ADAV-style reference")):
        meta = metadata.get(frame_id, {})
        clip_id = meta.get("clip_id")
        if clip_id is not None and previous_clip_id is not None and clip_id != previous_clip_id:
            tracker = SimpleIoUTracker()
            reference = []
            reference_count = 0
            reference_confidence = 0.0
        previous_clip_id = clip_id

        image = read_image(image_path)
        height, width = image.shape[:2]
        with latency.time():
            detections = tracker.update(engine.detect(image, frame_id=frame_id))
            usable_reference = [det for det in reference if det.confidence >= args.min_reference_confidence]
            _, missing = match_reference(usable_reference, detections, args.match_iou) if usable_reference else ([], [])

            count_drop = max(0.0, (reference_count - len(detections)) / max(reference_count, 1))
            confidence_drop = max(0.0, (reference_confidence - mean_confidence(detections)) / max(reference_confidence, 1e-6))
            missing_fraction = len(missing) / max(len(usable_reference), 1)
            score = max(count_drop, confidence_drop, missing_fraction)
            is_anomalous = bool(score >= args.threshold and usable_reference)
            recovered = recover_missing(missing, frame_id, score, args.recovery_confidence_scale, args.max_recovered) if is_anomalous else []
            robust = detections + recovered

            if (not is_anomalous) and detections:
                reference = [det.copy() for det in detections]
                reference_count = len(detections)
                reference_confidence = mean_confidence(detections)

        predictions.append(detections_to_prediction(robust))
        targets.append(labels_to_target(root / "labels" / f"{image_path.stem}.txt", width, height))
        label = int(meta.get("adversarial_label", 0))
        anomaly.update(label, score)
        rows.append(
            {
                "frame_id": frame_id,
                "clip_id": clip_id,
                "clip": meta.get("clip"),
                "image": str(image_path),
                "adversarial_label": label,
                "detections": len(detections),
                "robust_detections": len(robust),
                "reference_count": reference_count,
                "count_drop_score": count_drop,
                "confidence_drop_score": confidence_drop,
                "missing_fraction_score": missing_fraction,
                "max_anomaly_score": score,
                "missing_reference_objects": len(missing),
                "recovered": len(recovered),
                "mean_confidence": mean_confidence(robust),
            }
        )

    map_metrics = compute_map(predictions, targets) if predictions else {"map50": 0.0, "map50_95": 0.0, "mar100": 0.0}
    summary = {
        "method": "ADAV-style reference-frame recovery",
        "sequence_root": str(root),
        "frames": len(image_paths),
        "classes": args.classes,
        "threshold": args.threshold,
        "match_iou": args.match_iou,
        "min_reference_confidence": args.min_reference_confidence,
        "recovery_confidence_scale": args.recovery_confidence_scale,
        "map": map_metrics,
        "latency": latency.summary(),
        "anomaly": anomaly.compute(),
    }
    pd.DataFrame(rows).to_csv(output_dir / "adav_frame_metrics.csv", index=False)
    write_json(output_dir / "adav_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
