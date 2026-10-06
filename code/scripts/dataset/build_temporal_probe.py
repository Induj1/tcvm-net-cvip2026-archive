"""Build a controlled temporal attack probe from a real YOLO-labeled frame."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from advtraffic.attacks import apply_low_light, apply_motion_blur, apply_named_physical_attack
from advtraffic.utils.geometry import yolo_to_xyxy
from advtraffic.utils.io import ensure_dir, read_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reproducible temporal probe sequence.")
    parser.add_argument("--image", required=True, help="Source held-out image.")
    parser.add_argument("--label", required=True, help="YOLO label for the source image.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--attack", default="reflective", choices=["patch", "sticker", "reflective", "occlusion", "motion_blur", "low_light"])
    parser.add_argument("--attack-start", type=int, default=8)
    parser.add_argument("--attack-end", type=int, default=11, help="Exclusive end frame.")
    parser.add_argument("--target-classes", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--names", nargs="*", default=["helmet", "no_helmet", "rider"])
    return parser.parse_args()


def read_boxes(label_path: Path, width: int, height: int, target_classes: set[int]) -> list[tuple[int, np.ndarray]]:
    boxes: list[tuple[int, np.ndarray]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        if cls not in target_classes:
            continue
        row = np.asarray([float(v) for v in parts], dtype=float)
        boxes.append((cls, yolo_to_xyxy(row, width, height)))
    return boxes


def apply_probe_attack(image: np.ndarray, boxes: list[tuple[int, np.ndarray]], attack: str) -> np.ndarray:
    if attack == "motion_blur":
        return apply_motion_blur(image)
    if attack == "low_light":
        return apply_low_light(image)
    attacked = image.copy()
    for _, box in boxes:
        attacked = apply_named_physical_attack(attacked, box, attack)
    return attacked


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    label_path = Path(args.label)
    output_root = Path(args.output_root)
    image_dir = ensure_dir(output_root / "images")
    label_dir = ensure_dir(output_root / "labels")

    image = read_image(image_path)
    height, width = image.shape[:2]
    boxes = read_boxes(label_path, width, height, set(args.target_classes))
    rows = []

    for frame_id in range(args.frames):
        attacked = args.attack_start <= frame_id < args.attack_end
        frame = apply_probe_attack(image, boxes, args.attack) if attacked else image.copy()
        frame_name = f"frame_{frame_id:04d}{image_path.suffix.lower()}"
        cv2.imwrite(str(image_dir / frame_name), frame)
        shutil.copy2(label_path, label_dir / f"frame_{frame_id:04d}.txt")
        rows.append({"frame_id": frame_id, "image": frame_name, "adversarial_label": int(attacked), "attack": args.attack})

    with (output_root / "frame_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_id", "image", "adversarial_label", "attack"])
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "source_image": str(image_path),
        "source_label": str(label_path),
        "frames": args.frames,
        "attack": args.attack,
        "attack_start": args.attack_start,
        "attack_end": args.attack_end,
        "target_classes": args.target_classes,
        "target_boxes": len(boxes),
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    names = {index: name for index, name in enumerate(args.names)}
    yaml_lines = [
        f"path: {output_root.resolve().as_posix()}",
        "train: images",
        "val: images",
        "test: images",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in names.items())
    (output_root / "sequence.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), **metadata}, indent=2))


if __name__ == "__main__":
    main()
