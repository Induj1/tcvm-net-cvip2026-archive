"""Benchmark per-frame latency for edge deployment reports."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import torch

from advtraffic.eval.evaluator import evaluate_video
from advtraffic.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8+TCVM edge latency.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="results/edge_benchmark.json")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_video(
        video_path=args.video,
        model_path=args.model,
        output_json=args.output,
        use_tcvm=True,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        max_frames=args.max_frames,
    )
    report["hardware"] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    write_json(args.output, report)
    print(f"Wrote benchmark: {Path(args.output).resolve()}")
    print(report["latency"])


if __name__ == "__main__":
    main()
