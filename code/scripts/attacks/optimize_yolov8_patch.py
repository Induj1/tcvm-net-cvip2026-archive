"""Optimize and evaluate a detector-specific printable patch for YOLOv8.

This script is intentionally scoped as a reproducible pilot rather than a full
physical-world claim. It optimizes one EOT-style patch against YOLOv8 target
class confidence, applies it to YOLO-format images, and reports attack success
against detections that were present in the clean image.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from advtraffic.attacks.physical import apply_patch
from advtraffic.attacks.patch_optimizer import EOTPatchOptimizer, PatchOptimizerConfig
from advtraffic.detection import Detection, YOLOv8Engine
from advtraffic.utils.geometry import box_iou, yolo_to_xyxy
from advtraffic.utils.io import IMAGE_EXTENSIONS, ensure_dir, iter_files, read_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a YOLOv8-specific printable adversarial patch.")
    parser.add_argument("--dataset-root", required=True, help="YOLO-style root with images/ and labels/.")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--output-root", default="outputs/attacks_detector_specific/yolov8_patch")
    parser.add_argument("--classes", type=int, nargs="+", default=[3])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--device", default=None)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--max-opt-images", type=int, default=24)
    parser.add_argument(
        "--optimization-clips",
        nargs="*",
        type=int,
        default=None,
        help="Optional clip IDs used for patch optimization; evaluation still covers all input images.",
    )
    parser.add_argument("--max-eval-images", type=int, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=0.45)
    return parser.parse_args()


def read_labels(label_path: Path, width: int, height: int, target_classes: set[int]) -> list[tuple[int, np.ndarray]]:
    if not label_path.exists():
        return []
    labels: list[tuple[int, np.ndarray]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        row = np.asarray([float(value) for value in parts], dtype=float)
        cls = int(row[0])
        if cls in target_classes:
            labels.append((cls, yolo_to_xyxy(row, width, height)))
    return labels


def resize_box(box: np.ndarray, width: int, height: int, image_size: int) -> np.ndarray:
    scaled = box.astype(np.float32).copy()
    scaled[[0, 2]] *= image_size / max(width, 1)
    scaled[[1, 3]] *= image_size / max(height, 1)
    return scaled


def clean_detected(detections: list[Detection], cls: int, box: np.ndarray, iou_threshold: float) -> bool:
    return any(det.class_id == cls and box_iou(det.xyxy, box) >= iou_threshold for det in detections)


def collect_optimization_batch(
    image_paths: list[Path],
    label_root: Path,
    target_classes: set[int],
    image_size: int,
    max_images: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    images: list[torch.Tensor] = []
    boxes: list[np.ndarray] = []
    for image_path in image_paths:
        image = read_image(image_path)
        height, width = image.shape[:2]
        labels = read_labels(label_root / f"{image_path.stem}.txt", width, height, target_classes)
        if not labels:
            continue
        cls, box = labels[0]
        _ = cls
        resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        images.append(torch.from_numpy(rgb).permute(2, 0, 1))
        boxes.append(resize_box(box, width, height, image_size))
        if len(images) >= max_images:
            break
    if not images:
        raise RuntimeError("No target boxes found for patch optimization.")
    return torch.stack(images, dim=0), torch.from_numpy(np.stack(boxes, axis=0).astype(np.float32))


def select_clip_disjoint_optimization_paths(
    image_paths: list[Path],
    clip_ids: list[int] | None,
    limit: int,
) -> list[Path]:
    if not clip_ids:
        return image_paths
    grouped: dict[int, list[Path]] = {clip_id: [] for clip_id in sorted(set(clip_ids))}
    for image_path in image_paths:
        match = re.match(r"clip(\d+)_", image_path.stem)
        if match:
            clip_id = int(match.group(1))
            if clip_id in grouped:
                grouped[clip_id].append(image_path)
    missing = [clip_id for clip_id, paths in grouped.items() if not paths]
    if missing:
        raise ValueError(f"No optimization images found for clip IDs: {missing}")
    selected = []
    offset = 0
    while len(selected) < limit:
        added = False
        for clip_id in sorted(grouped):
            paths = grouped[clip_id]
            if offset < len(paths):
                selected.append(paths[offset])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    image_root = dataset_root / "images"
    label_root = dataset_root / "labels"
    output_root = Path(args.output_root)
    out_images = ensure_dir(output_root / "images")
    out_labels = ensure_dir(output_root / "labels")
    out_vis = ensure_dir(output_root / "visualizations")
    target_classes = set(args.classes)

    image_paths = list(iter_files(image_root, IMAGE_EXTENSIONS))
    if args.max_eval_images is not None:
        image_paths = image_paths[: args.max_eval_images]

    optimization_limit = min(args.max_opt_images, args.batch)
    optimization_paths = select_clip_disjoint_optimization_paths(
        image_paths,
        args.optimization_clips,
        optimization_limit,
    )
    opt_images, opt_boxes = collect_optimization_batch(
        optimization_paths,
        label_root,
        target_classes,
        args.imgsz,
        optimization_limit,
    )

    engine = YOLOv8Engine(model_path=args.model, imgsz=args.imgsz, conf=args.conf, device=args.device, classes=args.classes, extract_features=False)
    optimizer = EOTPatchOptimizer(
        engine.raw_torch_model(),
        PatchOptimizerConfig(patch_size=args.patch_size, steps=args.steps, lr=args.lr),
        device=args.device,
    )
    patch_rgb = optimizer.optimize(opt_images, opt_boxes, target_classes=args.classes)
    patch_np = (patch_rgb.permute(1, 2, 0).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    patch_bgr = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_root / "optimized_patch.png"), patch_bgr)

    rows = []
    clean_detected_total = 0
    successful_attacks = 0
    target_total = 0
    for image_path in tqdm(image_paths, desc="YOLO patch attack"):
        image = read_image(image_path)
        height, width = image.shape[:2]
        label_path = label_root / f"{image_path.stem}.txt"
        labels = read_labels(label_path, width, height, target_classes)
        patched = image.copy()
        for _, box in labels:
            patched = apply_patch(patched, box, patch_bgr, scale=args.scale, alpha=0.98, vertical_offset=-0.18)

        clean_dets = engine.detect(image, frame_id=0)
        patched_dets = engine.detect(patched, frame_id=0)

        per_image_targets = 0
        per_image_success = 0
        for cls, box in labels:
            if clean_detected(clean_dets, cls, box, args.iou_threshold):
                clean_detected_total += 1
                per_image_targets += 1
                if not clean_detected(patched_dets, cls, box, args.iou_threshold):
                    successful_attacks += 1
                    per_image_success += 1
            target_total += 1

        cv2.imwrite(str(out_images / image_path.name), patched)
        if label_path.exists():
            shutil.copy2(label_path, out_labels / label_path.name)
        else:
            (out_labels / f"{image_path.stem}.txt").touch()

        preview = np.concatenate([image, patched], axis=1)
        cv2.imwrite(str(out_vis / image_path.name), preview)
        rows.append(
            {
                "image": str(image_path),
                "targets": len(labels),
                "clean_detected_targets": per_image_targets,
                "successful_attacks": per_image_success,
            }
        )

    frame_labels = dataset_root / "frame_labels.csv"
    if frame_labels.exists():
        shutil.copy2(frame_labels, output_root / "frame_labels.csv")

    summary = {
        "attack": "yolov8_detector_specific_patch",
        "model": args.model,
        "dataset_root": str(dataset_root),
        "images": len(image_paths),
        "optimization_images": int(opt_images.shape[0]),
        "optimization_clips": args.optimization_clips,
        "target_classes": args.classes,
        "target_objects": target_total,
        "clean_detected_targets": clean_detected_total,
        "successful_attacks": successful_attacks,
        "attack_success_rate": successful_attacks / max(clean_detected_total, 1),
        "patch_size": args.patch_size,
        "steps": args.steps,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "per_image_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "attack_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
