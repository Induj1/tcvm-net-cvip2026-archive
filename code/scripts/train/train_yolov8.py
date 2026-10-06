"""Train and validate YOLOv8 baselines for TCVM-Net experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO

from advtraffic.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publication-grade YOLOv8 baseline training.")
    parser.add_argument("--data", required=True, help="YOLO dataset YAML.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model or checkpoint.")
    parser.add_argument("--name", default="helmet_yolov8n")
    parser.add_argument("--project", default="outputs/baselines")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable mixed precision.")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export", choices=["onnx", "torchscript", "engine"], default=None)
    return parser.parse_args()


def metrics_to_dict(metrics) -> dict[str, float]:
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "fitness": float(metrics.fitness),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    project_dir = Path(args.project).resolve()
    run_dir = project_dir / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    train_result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        amp=args.amp,
        cache=args.cache,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        plots=True,
        save=True,
        save_period=max(args.epochs // 5, 1),
    )

    actual_run_dir = Path(getattr(train_result, "save_dir", run_dir)).resolve()
    best_weights = actual_run_dir / "weights" / "best.pt"
    validator = YOLO(str(best_weights if best_weights.exists() else args.model))
    val_metrics = validator.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(actual_run_dir),
        name="validation",
        exist_ok=True,
        plots=True,
    )
    summary = {
        "data": args.data,
        "model": args.model,
        "best_weights": str(best_weights),
        "device": str(device),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "amp": args.amp,
        "metrics": metrics_to_dict(val_metrics),
        "train_save_dir": str(actual_run_dir),
    }
    (actual_run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary["metrics"]]).to_csv(actual_run_dir / "metrics.csv", index=False)

    if args.export:
        validator.export(format=args.export, imgsz=args.imgsz, device=device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
