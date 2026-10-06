"""Profile detector, tracking, and TCVM overhead on a sequential image set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
import psutil
import torch
from tqdm import tqdm

from advtraffic.defense import TCVMNetPipeline
from advtraffic.detection import YOLOv8Engine
from advtraffic.eval.metrics import LatencyMeter
from advtraffic.tracking import SimpleIoUTracker
from advtraffic.utils.io import IMAGE_EXTENSIONS, iter_files, read_image, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile TCVM-Net runtime overhead.")
    parser.add_argument("--sequence-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-json", default="outputs/results_50_complete/tcvm_overhead_profile.json")
    parser.add_argument("--output-csv", default="outputs/results_50_complete/tcvm_overhead_profile.csv")
    parser.add_argument("--output-tex", default="paper/tables/tcvm_overhead_profile_auto.tex")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--classes", nargs="*", type=int, default=None)
    parser.add_argument("--frame-labels", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--flow-scale", type=float, default=1.0)
    return parser.parse_args()


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(meter: LatencyMeter, fn):
    cuda_sync()
    start = perf_counter()
    result = fn()
    cuda_sync()
    meter.update_ms((perf_counter() - start) * 1000.0)
    return result


def load_clip_ids(path: str | None) -> dict[int, int]:
    if path is None:
        return {}
    df = pd.read_csv(path)
    if "frame_id" not in df.columns or "clip_id" not in df.columns:
        return {}
    return {int(row.frame_id): int(row.clip_id) for row in df.itertuples(index=False)}


def row(stage: str, meter: LatencyMeter, peak_rss_mb: float) -> dict:
    summary = meter.summary()
    return {
        "Stage": stage,
        "Mean ms": summary["mean_ms"],
        "p95 ms": summary["p95_ms"],
        "FPS": summary["fps"],
        "Peak RSS MB": peak_rss_mb,
    }


def write_latex(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in ["Mean ms", "p95 ms", "FPS", "Peak RSS MB"]:
        out[col] = out[col].map(lambda value: f"{float(value):.3f}")
    latex = out.to_latex(
        index=False,
        escape=False,
        caption="RTX 4060 runtime overhead profile on the public HELMET 10-clip sequence.",
        label="tab:tcvm_overhead_profile_auto",
        position="ht",
    )
    latex = latex.replace("{lllll}", "{lrrrr}", 1)
    latex = latex.replace(
        "\\begin{tabular}",
        "\\small\n\\setlength{\\tabcolsep}{4pt}\n\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}",
        1,
    )
    latex = latex.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.sequence_root)
    image_paths = list(iter_files(root / "images", IMAGE_EXTENSIONS))
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {root / 'images'}")

    proc = psutil.Process()
    clip_ids = load_clip_ids(args.frame_labels)

    baseline_engine = YOLOv8Engine(
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        classes=args.classes,
        extract_features=False,
    )
    baseline_meter = LatencyMeter()
    peak_baseline = proc.memory_info().rss / (1024**2)
    for idx, image_path in enumerate(tqdm(image_paths, desc="YOLOv8 no-features")):
        image = read_image(image_path)
        if idx < args.warmup:
            baseline_engine.detect(image, frame_id=idx)
            continue
        timed_call(baseline_meter, lambda image=image, idx=idx: baseline_engine.detect(image, frame_id=idx))
        peak_baseline = max(peak_baseline, proc.memory_info().rss / (1024**2))

    del baseline_engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tcvm_engine = YOLOv8Engine(
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        classes=args.classes,
        extract_features=True,
    )
    tracker = SimpleIoUTracker()
    pipeline = TCVMNetPipeline(use_optical_flow=True, flow_scale=args.flow_scale)
    detect_meter = LatencyMeter()
    track_meter = LatencyMeter()
    verify_meter = LatencyMeter()
    total_meter = LatencyMeter()
    peak_tcvm = proc.memory_info().rss / (1024**2)
    previous_clip_id = None

    for idx, image_path in enumerate(tqdm(image_paths, desc="TCVM profile")):
        clip_id = clip_ids.get(idx)
        if clip_id is not None and previous_clip_id is not None and clip_id != previous_clip_id:
            tracker = SimpleIoUTracker()
            pipeline = TCVMNetPipeline(use_optical_flow=True, flow_scale=args.flow_scale)
        previous_clip_id = clip_id

        image = read_image(image_path)
        if idx < args.warmup:
            detections = tracker.update(tcvm_engine.detect(image, frame_id=idx))
            pipeline.process_frame(image, detections, idx)
            continue

        cuda_sync()
        total_start = perf_counter()
        detections = timed_call(detect_meter, lambda image=image, idx=idx: tcvm_engine.detect(image, frame_id=idx))
        detections = timed_call(track_meter, lambda detections=detections: tracker.update(detections))
        timed_call(verify_meter, lambda image=image, detections=detections, idx=idx: pipeline.process_frame(image, detections, idx))
        cuda_sync()
        total_meter.update_ms((perf_counter() - total_start) * 1000.0)
        peak_tcvm = max(peak_tcvm, proc.memory_info().rss / (1024**2))

    rows = [
        row("YOLOv8 detector only", baseline_meter, peak_baseline),
        row("YOLOv8 detector + ROI features", detect_meter, peak_tcvm),
        row("IoU tracking association", track_meter, peak_tcvm),
        row("TCVM verification + optical flow", verify_meter, peak_tcvm),
        row("Full YOLOv8 + TCVM pipeline", total_meter, peak_tcvm),
    ]
    report = {
        "sequence_root": str(root),
        "frames": len(image_paths),
        "warmup": args.warmup,
        "device": args.device,
        "flow_scale": args.flow_scale,
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rows": rows,
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, report)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    write_latex(df, Path(args.output_tex))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
