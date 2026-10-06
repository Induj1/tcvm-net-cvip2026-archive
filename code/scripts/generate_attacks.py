"""Generate adversarial variants of AdvTraffic-26 images."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from advtraffic.attacks import (
    apply_low_light,
    apply_motion_blur,
    apply_named_physical_attack,
    fgsm_attack,
    pgd_attack,
)
from advtraffic.detection import YOLOv8Engine
from advtraffic.utils.geometry import yolo_to_xyxy
from advtraffic.utils.io import IMAGE_EXTENSIONS, ensure_dir, iter_files, read_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create adversarial image splits for AdvTraffic-26.")
    parser.add_argument("--dataset-root", default="data/AdvTraffic-26")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--attack",
        required=True,
        choices=["fgsm", "pgd", "patch", "sticker", "reflective", "occlusion", "motion_blur", "low_light"],
    )
    parser.add_argument("--model", default=None, help="YOLOv8 weights for FGSM/PGD.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--target-classes", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def read_yolo_labels(label_path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    rows = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            values = np.asarray([float(v) for v in parts], dtype=float)
            rows.append((int(values[0]), yolo_to_xyxy(values, width, height)))
    return rows


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    image_root = dataset_root / "images" / args.split
    label_root = dataset_root / "labels" / args.split
    out_image_root = ensure_dir(dataset_root / "adversarial" / args.attack / "images" / args.split)
    out_label_root = ensure_dir(dataset_root / "adversarial" / args.attack / "labels" / args.split)
    metadata_path = dataset_root / "metadata" / f"{args.attack}_{args.split}.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    engine = None
    if args.attack in {"fgsm", "pgd"}:
        if not args.model:
            raise ValueError("--model is required for FGSM/PGD generation")
        engine = YOLOv8Engine(model_path=args.model, imgsz=args.imgsz, extract_features=False)

    images = list(iter_files(image_root, IMAGE_EXTENSIONS))
    if args.max_images is not None:
        images = images[: args.max_images]

    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "attack", "targets", "output"])
        writer.writeheader()

        for image_path in images:
            image = read_image(image_path)
            height, width = image.shape[:2]
            label_path = label_root / f"{image_path.stem}.txt"
            labels = read_yolo_labels(label_path, width, height)
            targets = [(cls, box) for cls, box in labels if cls in set(args.target_classes)]

            if args.attack == "fgsm":
                assert engine is not None
                adv = fgsm_attack(
                    engine.raw_torch_model(),
                    image,
                    eps=args.eps,
                    image_size=args.imgsz,
                    target_classes=args.target_classes,
                )
            elif args.attack == "pgd":
                assert engine is not None
                adv = pgd_attack(
                    engine.raw_torch_model(),
                    image,
                    eps=args.eps,
                    steps=args.pgd_steps,
                    image_size=args.imgsz,
                    target_classes=args.target_classes,
                )
            elif args.attack == "motion_blur":
                adv = apply_motion_blur(image)
            elif args.attack == "low_light":
                adv = apply_low_light(image)
            else:
                adv = image.copy()
                for _, box in targets:
                    adv = apply_named_physical_attack(adv, box, args.attack)

            out_image = out_image_root / image_path.name
            out_label = out_label_root / label_path.name
            cv2.imwrite(str(out_image), adv)
            if label_path.exists():
                shutil.copyfile(label_path, out_label)
            else:
                out_label.touch()
            writer.writerow(
                {
                    "image": str(image_path),
                    "attack": args.attack,
                    "targets": len(targets),
                    "output": str(out_image),
                }
            )
    print(f"Wrote adversarial split: {out_image_root}")
    print(f"Wrote attack metadata: {metadata_path}")


if __name__ == "__main__":
    main()
