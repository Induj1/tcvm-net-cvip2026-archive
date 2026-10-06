"""Evaluate YOLOv8 and TCVM robustness."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from advtraffic.eval.evaluator import evaluate_video
from advtraffic.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AdvTraffic-26 experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    val = subparsers.add_parser("yolo-val", help="Run Ultralytics validation and export metrics.")
    val.add_argument("--model", required=True)
    val.add_argument("--data", default="data/AdvTraffic-26/advtraffic26.yaml")
    val.add_argument("--imgsz", type=int, default=640)
    val.add_argument("--batch", type=int, default=16)
    val.add_argument("--device", default=None)
    val.add_argument("--output", default="results/yolo_val_metrics.json")

    video = subparsers.add_parser("video", help="Evaluate baseline or TCVM on a video.")
    video.add_argument("--video", required=True)
    video.add_argument("--model", required=True)
    video.add_argument("--output", default="results/video_eval.json")
    video.add_argument("--imgsz", type=int, default=640)
    video.add_argument("--conf", type=float, default=0.25)
    video.add_argument("--device", default=None)
    video.add_argument("--no-tcvm", action="store_true")
    video.add_argument("--fallback-iou-tracker", action="store_true")
    video.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "yolo-val":
        model = YOLO(args.model)
        metrics = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device, plots=True)
        report = {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        write_json(args.output, report)
        print(report)
    elif args.command == "video":
        report = evaluate_video(
            video_path=args.video,
            model_path=args.model,
            output_json=args.output,
            use_tcvm=not args.no_tcvm,
            use_bytetrack=not args.fallback_iou_tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            max_frames=args.max_frames,
        )
        print(f"Wrote video report: {Path(args.output).resolve()}")
        print(report["latency"])


if __name__ == "__main__":
    main()
