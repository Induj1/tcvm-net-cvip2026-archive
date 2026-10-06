"""Build a pseudo-labeled real-video temporal attack sequence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from advtraffic.attacks import apply_low_light, apply_motion_blur, apply_named_physical_attack
from advtraffic.detection import YOLOv8Engine
from advtraffic.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a real-video temporal attack sequence with pseudo labels.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--attack", default="reflective", choices=["patch", "sticker", "reflective", "occlusion", "motion_blur", "low_light"])
    parser.add_argument("--attack-start", type=int, default=24)
    parser.add_argument("--attack-end", type=int, default=36)
    parser.add_argument("--target-classes", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--max-targets", type=int, default=12)
    return parser.parse_args()


def xyxy_to_yolo(box: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box.astype(float)
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = max(0.0, x2 - x1) / width
    bh = max(0.0, y2 - y1) / height
    return cx, cy, bw, bh


def apply_sequence_attack(frame: np.ndarray, boxes: list[np.ndarray], attack: str) -> np.ndarray:
    if attack == "motion_blur":
        return apply_motion_blur(frame)
    if attack == "low_light":
        return apply_low_light(frame)
    attacked = frame.copy()
    for box in boxes:
        attacked = apply_named_physical_attack(attacked, box, attack)
    return attacked


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    image_dir = ensure_dir(output_root / "images")
    label_dir = ensure_dir(output_root / "labels")
    target_classes = set(args.target_classes)

    engine = YOLOv8Engine(model_path=args.model, imgsz=args.imgsz, conf=args.conf, device=args.device, extract_features=False)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    frame_rows = []
    pseudo_objects = 0
    saved = 0
    raw_frame_id = -1
    pbar = tqdm(total=args.max_frames, desc="video-sequence")
    while saved < args.max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        raw_frame_id += 1
        if raw_frame_id % args.stride != 0:
            continue

        height, width = frame.shape[:2]
        clean_detections = [det for det in engine.detect(frame, frame_id=saved) if det.class_id in target_classes]
        clean_detections = sorted(clean_detections, key=lambda det: det.confidence, reverse=True)
        target_boxes = [det.xyxy for det in clean_detections[: args.max_targets]]
        attacked = args.attack_start <= saved < args.attack_end
        out_frame = apply_sequence_attack(frame, target_boxes, args.attack) if attacked else frame

        image_name = f"frame_{saved:04d}.jpg"
        label_name = f"frame_{saved:04d}.txt"
        cv2.imwrite(str(image_dir / image_name), out_frame)
        with (label_dir / label_name).open("w", encoding="utf-8") as handle:
            for det in clean_detections:
                cx, cy, bw, bh = xyxy_to_yolo(det.xyxy, width, height)
                if bw <= 0 or bh <= 0:
                    continue
                handle.write(f"{det.class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n")
                pseudo_objects += 1

        frame_rows.append(
            {
                "frame_id": saved,
                "raw_frame_id": raw_frame_id,
                "image": image_name,
                "adversarial_label": int(attacked),
                "attack": args.attack,
                "pseudo_labels": len(clean_detections),
                "attacked_targets": len(target_boxes) if attacked else 0,
            }
        )
        saved += 1
        pbar.update(1)
    pbar.close()
    capture.release()

    with (output_root / "frame_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0].keys()) if frame_rows else ["frame_id"])
        writer.writeheader()
        writer.writerows(frame_rows)

    yaml_text = "\n".join(
        [
            f"path: {output_root.resolve().as_posix()}",
            "train: images",
            "val: images",
            "test: images",
            "names:",
            "  0: helmet",
            "  1: no_helmet",
            "  2: rider",
        ]
    )
    (output_root / "sequence.yaml").write_text(yaml_text + "\n", encoding="utf-8")

    metadata = {
        "source_video": str(Path(args.video).resolve()),
        "model": args.model,
        "frames": saved,
        "stride": args.stride,
        "attack": args.attack,
        "attack_start": args.attack_start,
        "attack_end": args.attack_end,
        "target_classes": args.target_classes,
        "max_targets": args.max_targets,
        "pseudo_objects": pseudo_objects,
        "label_type": "clean_yolov8_pseudo_labels",
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
