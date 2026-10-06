"""Run TCVM-Net ablation studies on a sequential image split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

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


ABLATIONS = {
    "full_tcvm": {},
    "track_recovery_only": {
        "weight_confidence": 0.0,
        "weight_motion": 0.0,
        "weight_feature": 0.0,
        "weight_disappearance": 1.0,
    },
    "no_confidence": {"weight_confidence": 0.0},
    "no_motion": {"weight_motion": 0.0},
    "no_feature": {"weight_feature": 0.0},
    "no_temporal_smoothing": {"ema_alpha": 1.0},
    "no_recovery": {"disable_recovery": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TCVM ablation study.")
    parser.add_argument("--sequence-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="outputs/results/ablations")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--classes", nargs="*", type=int, default=None)
    parser.add_argument("--adversarial-label", type=int, default=1)
    parser.add_argument("--frame-labels", default=None)
    parser.add_argument("--anomaly-threshold", type=float, default=0.62)
    parser.add_argument("--flow-scale", type=float, default=1.0, help="Resolution scale for dense optical flow.")
    parser.add_argument(
        "--optical-flow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dense optical flow in the motion consistency term.",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        choices=list(ABLATIONS),
        help="Optional subset of ablation variants. Defaults to all variants.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows in output-dir/ablation_summary.csv.")
    return parser.parse_args()


def load_frame_metadata(path: str | None, default_label: int) -> dict[int, dict]:
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
        if "local_frame" in df.columns:
            item["local_frame"] = int(getattr(row, "local_frame"))
        metadata[frame_id] = item
    return metadata


def build_config(params: dict, anomaly_threshold: float) -> TCVMConfig:
    params = {k: v for k, v in params.items() if k != "disable_recovery"}
    params["anomaly_threshold"] = anomaly_threshold
    return TCVMConfig(**params)


def evaluate_variant(args: argparse.Namespace, variant: str, params: dict) -> dict:
    root = Path(args.sequence_root)
    image_paths = list(iter_files(root / "images", IMAGE_EXTENSIONS))
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    cfg = build_config(params, args.anomaly_threshold)
    verifier = TemporalConsistencyVerifier(cfg)
    pipeline = TCVMNetPipeline(
        RobustPredictionLayer(verifier),
        use_optical_flow=args.optical_flow,
        flow_scale=args.flow_scale,
    )
    engine = YOLOv8Engine(model_path=args.model, imgsz=args.imgsz, conf=args.conf, device=args.device, classes=args.classes)
    tracker = SimpleIoUTracker()
    latency = LatencyMeter()
    anomaly = BinaryMetrics(threshold=cfg.anomaly_threshold)
    predictions = []
    targets = []
    frame_rows = []
    frame_metadata = load_frame_metadata(args.frame_labels, args.adversarial_label)
    previous_clip_id = None

    for frame_id, image_path in enumerate(tqdm(image_paths, desc=variant)):
        meta = frame_metadata.get(frame_id, {"adversarial_label": args.adversarial_label})
        clip_id = meta.get("clip_id")
        if clip_id is not None and previous_clip_id is not None and clip_id != previous_clip_id:
            verifier = TemporalConsistencyVerifier(cfg)
            pipeline = TCVMNetPipeline(
                RobustPredictionLayer(verifier),
                use_optical_flow=args.optical_flow,
                flow_scale=args.flow_scale,
            )
            tracker = SimpleIoUTracker()
        previous_clip_id = clip_id

        image = read_image(image_path)
        height, width = image.shape[:2]
        with latency.time():
            detections = tracker.update(engine.detect(image, frame_id=frame_id))
            robust = pipeline.process_frame(image, detections, frame_id)
            if params.get("disable_recovery"):
                robust = [det for det in robust if not det.is_recovered]
        predictions.append(detections_to_prediction(robust))
        targets.append(labels_to_target(root / "labels" / f"{image_path.stem}.txt", width, height))
        scored_events = pipeline.robust_layer.last_scored + pipeline.robust_layer.last_missing_events
        max_anomaly_score = max((det.anomaly_score for det in scored_events), default=0.0)
        adversarial_label = int(meta.get("adversarial_label", args.adversarial_label))
        anomaly.update(adversarial_label, max_anomaly_score)
        frame_rows.append(
            {
                "frame_id": frame_id,
                "clip_id": clip_id,
                "clip": meta.get("clip"),
                "local_frame": meta.get("local_frame"),
                "image": str(image_path),
                "adversarial_label": adversarial_label,
                "detections": len(detections),
                "robust_detections": len(robust),
                "max_anomaly_score": max_anomaly_score,
                "missing_events": len(pipeline.robust_layer.last_missing_events),
                "recovered": sum(det.is_recovered for det in robust),
            }
        )

    map_metrics = compute_map(predictions, targets) if predictions else {"map50": 0.0, "map50_95": 0.0, "mar100": 0.0}
    pd.DataFrame(frame_rows).to_csv(Path(args.output_dir) / f"{variant}_frame_metrics.csv", index=False)
    return {
        "variant": variant,
        "frames": len(image_paths),
        "map50": map_metrics["map50"],
        "map50_95": map_metrics["map50_95"],
        "fps": latency.summary()["fps"],
        "mean_latency_ms": latency.summary()["mean_ms"],
        **{f"anomaly_{k}": v for k, v in anomaly.compute().items()},
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation_summary.csv"
    rows = []
    completed = set()
    if args.resume and csv_path.exists():
        existing = pd.read_csv(csv_path)
        rows = existing.to_dict(orient="records")
        completed = set(existing["variant"].astype(str).tolist())

    selected = args.variants or list(ABLATIONS)
    for variant in selected:
        if variant in completed:
            print(f"Skipping completed variant: {variant}")
            continue
        row = evaluate_variant(args, variant, ABLATIONS[variant])
        rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        write_json(output_dir / "ablation_summary.json", {"rows": rows})
        print(f"Wrote checkpoint after {variant}: {csv_path.resolve()}")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    write_json(output_dir / "ablation_summary.json", {"rows": rows})
    print(df.to_string(index=False))
    print(f"Wrote {csv_path.resolve()}")


if __name__ == "__main__":
    main()
