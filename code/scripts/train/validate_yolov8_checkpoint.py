"""Validate a YOLOv8 checkpoint and persist publication-ready metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained YOLOv8 checkpoint.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="yolov8")
    parser.add_argument("--epochs-requested", type=int, default=None)
    parser.add_argument("--epochs-completed", type=int, default=None)
    parser.add_argument("--status", default="completed")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        split=args.split,
        project=str(output_dir),
        name="validation",
        exist_ok=True,
        plots=True,
    )
    summary = {
        "data": args.data,
        "model": args.model_name,
        "best_weights": str(Path(args.weights).resolve()),
        "device": str(device),
        "epochs_requested": args.epochs_requested,
        "epochs_completed": args.epochs_completed,
        "completion_status": args.status,
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "metrics": {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "fitness": float(metrics.fitness),
        },
        "speed_ms": dict(getattr(metrics, "speed", {}) or {}),
        "validation_save_dir": str(getattr(metrics, "save_dir", output_dir / "validation")),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([{**summary["metrics"], "epochs_completed": args.epochs_completed}]).to_csv(
        output_dir / "metrics.csv", index=False
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
