"""Prepare BDD100K detection images for YOLOv8 experiments.

BDD100K requires registration and manual download from the official site. Place
the source files in this layout before running this converter:

source_dir/
  labels/bdd100k_labels_images_train.json
  labels/bdd100k_labels_images_val.json
  images/100k/train/*.jpg
  images/100k/val/*.jpg
  images/100k/test/*.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from advtraffic.datasets import BDD100K_CLASSES, BDD_TO_ADVTRAFFIC, convert_bdd100k_detection_split
from advtraffic.datasets.conversion import write_dataset_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BDD100K detection annotations to YOLO format.")
    parser.add_argument("--source-dir", required=True, help="Manually downloaded BDD100K source directory.")
    parser.add_argument("--output-dir", default="data/bdd100k-yolo", help="YOLO output directory.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    parser.add_argument("--advtraffic-classes", action="store_true", help="Map BDD rider/motor classes into AdvTraffic-26 classes.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--link-images", action="store_true", help="Symlink images instead of copying when possible.")
    parser.add_argument("--write-manifest", action="store_true", help="Write source metadata JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.advtraffic_classes:
        classes = ["helmet", "no_helmet", "rider", "motorcycle", "license_plate", "violation"]
        class_map = BDD_TO_ADVTRAFFIC
        yaml_name = "bdd100k_advtraffic.yaml"
    else:
        classes = BDD100K_CLASSES
        class_map = {name: name for name in classes}
        yaml_name = "bdd100k.yaml"

    summary = {}
    for split in args.splits:
        summary[split] = convert_bdd100k_detection_split(
            source_dir=args.source_dir,
            output_dir=output_dir,
            split=split,
            classes=classes,
            class_map=class_map,
            max_images=args.max_images,
            link_images=args.link_images,
        )
    yaml_path = write_dataset_yaml(output_dir, classes, filename=yaml_name)
    if args.write_manifest:
        manifest = {
            "dataset": "BDD100K",
            "source_dir": str(Path(args.source_dir).resolve()),
            "output_dir": str(output_dir.resolve()),
            "splits": args.splits,
            "class_mode": "advtraffic" if args.advtraffic_classes else "bdd100k",
            "summary": summary,
        }
        (output_dir / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"yaml": str(yaml_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
