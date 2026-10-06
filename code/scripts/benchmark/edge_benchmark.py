"""Measure FPS, latency, memory, and GPU utilization for YOLOv8 and TCVM-Net."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import psutil
import torch
from tqdm import tqdm

from advtraffic.defense import TCVMNetPipeline
from advtraffic.detection import YOLOv8Engine
from advtraffic.eval.metrics import LatencyMeter
from advtraffic.tracking import SimpleIoUTracker
from advtraffic.utils.io import IMAGE_EXTENSIONS, iter_files, read_image, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge deployment benchmark for TCVM-Net.")
    parser.add_argument("--sequence-root", required=True, help="Root containing images/ directory.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="outputs/results/edge_benchmark.json")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def gpu_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    snap = {
        "cuda_available": True,
        "cuda_name": torch.cuda.get_device_name(0),
        "torch_allocated_mb": torch.cuda.memory_allocated(0) / (1024**2),
        "torch_reserved_mb": torch.cuda.memory_reserved(0) / (1024**2),
    }
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        snap.update({"gpu_util_percent": util.gpu, "gpu_mem_used_mb": mem.used / (1024**2)})
    except Exception:
        pass
    return snap


def run_mode(mode: str, engine: YOLOv8Engine, image_paths: list[Path], warmup: int) -> dict:
    tracker = SimpleIoUTracker()
    tcvm = TCVMNetPipeline(use_optical_flow=True)
    meter = LatencyMeter()
    proc = psutil.Process()
    peak_rss = proc.memory_info().rss

    for idx, image_path in enumerate(tqdm(image_paths, desc=mode)):
        image = read_image(image_path)
        if idx < warmup:
            detections = tracker.update(engine.detect(image, frame_id=idx))
            if mode == "tcvm":
                _ = tcvm.process_frame(image, detections, idx)
            continue
        with meter.time():
            detections = tracker.update(engine.detect(image, frame_id=idx))
            if mode == "tcvm":
                _ = tcvm.process_frame(image, detections, idx)
        peak_rss = max(peak_rss, proc.memory_info().rss)

    summary = meter.summary()
    summary["peak_rss_mb"] = peak_rss / (1024**2)
    summary.update(gpu_snapshot())
    return summary


def main() -> None:
    args = parse_args()
    image_paths = list(iter_files(Path(args.sequence_root) / "images", IMAGE_EXTENSIONS))[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {Path(args.sequence_root) / 'images'}")
    engine = YOLOv8Engine(model_path=args.model, imgsz=args.imgsz, conf=args.conf, device=args.device)
    report = {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "sequence_root": args.sequence_root,
        "frames": len(image_paths),
        "yolov8": run_mode("yolov8", engine, image_paths, args.warmup),
        "tcvm": run_mode("tcvm", engine, image_paths, args.warmup),
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
