"""Prepare a public helmet dataset for YOLOv8.

The default converter supports the Kaggle hard-hat / safety helmet VOC layout:

source_dir/
  annotations/*.xml
  images/*.png or images/*.jpg

If the dataset is already in YOLO format, use --source-yolo to copy/symlink it
into the expected project layout and write the dataset YAML.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from advtraffic.datasets import HELMET_CLASSES, convert_voc_helmet_dataset
from advtraffic.datasets.conversion import ensure_yolo_split_dirs, write_dataset_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare helmet detection dataset for YOLOv8.")
    parser.add_argument("--source-dir", default=None, help="Local extracted dataset directory.")
    parser.add_argument("--output-dir", default="data/helmet-yolo")
    parser.add_argument("--source-yolo", action="store_true", help="Source already contains images/ and labels/ splits.")
    parser.add_argument("--kaggle-dataset", default=None, help="Optional Kaggle slug, e.g. andrewmvd/hard-hat-detection.")
    parser.add_argument("--download-dir", default="data/downloads/helmet", help="Kaggle download/extract directory.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-images", action="store_true")
    return parser.parse_args()


def maybe_download_kaggle(slug: str | None, download_dir: str | Path) -> Path | None:
    if not slug:
        return None
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(download_dir), "--unzip"]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Kaggle CLI is not installed. Install it and configure kaggle.json credentials.") from exc
    return download_dir


def copy_yolo_dataset(source_dir: Path, output_dir: Path, link_images: bool) -> dict[str, int]:
    ensure_yolo_split_dirs(output_dir)
    stats = {"images": 0, "labels": 0}
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            src = source_dir / kind / split
            dst = output_dir / kind / split
            if not src.exists():
                continue
            for path in src.iterdir():
                if not path.is_file():
                    continue
                target = dst / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if kind == "images" and link_images:
                    try:
                        target.symlink_to(path.resolve())
                    except OSError:
                        shutil.copy2(path, target)
                else:
                    shutil.copy2(path, target)
                stats[kind] += 1
    write_dataset_yaml(output_dir, HELMET_CLASSES, filename="helmet.yaml")
    return stats


def main() -> None:
    args = parse_args()
    downloaded = maybe_download_kaggle(args.kaggle_dataset, args.download_dir)
    source_dir = Path(args.source_dir or downloaded or "")
    if not source_dir.exists():
        raise FileNotFoundError("Provide --source-dir or --kaggle-dataset with configured Kaggle credentials.")
    output_dir = Path(args.output_dir)
    if args.source_yolo:
        summary = copy_yolo_dataset(source_dir, output_dir, link_images=args.link_images)
    else:
        summary = convert_voc_helmet_dataset(
            source_dir=source_dir,
            output_dir=output_dir,
            classes=HELMET_CLASSES,
            seed=args.seed,
            max_images=args.max_images,
            link_images=args.link_images,
        )
        generated_yaml = output_dir / f"{output_dir.name}.yaml"
        if generated_yaml.exists() and generated_yaml.name != "helmet.yaml":
            (output_dir / "helmet.yaml").write_text(generated_yaml.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"yaml": str(output_dir / "helmet.yaml"), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
