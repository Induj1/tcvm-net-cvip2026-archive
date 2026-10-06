"""Train YOLOv8 on AdvTraffic-26."""

from __future__ import annotations

import argparse

from ultralytics import YOLO

from advtraffic.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for AdvTraffic-26.")
    parser.add_argument("--data", default="data/AdvTraffic-26/advtraffic26.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="advtraffic26_yolov8n")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        workers=args.workers,
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
