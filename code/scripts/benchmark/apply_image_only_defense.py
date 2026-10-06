"""Apply image-only preprocessing defenses to attacked images."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import cv2
from tqdm import tqdm

from advtraffic.defense.image_only import image_only_defense
from advtraffic.utils.io import IMAGE_EXTENSIONS, ensure_dir, iter_files, read_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply image-only defense baseline to an image split.")
    parser.add_argument("--source-root", required=True, help="Root with images/ and labels/ directories.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", default="jpeg_median_clahe", choices=["jpeg", "median", "clahe", "jpeg_median_clahe"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.source_root)
    dst = Path(args.output_root)
    out_images = ensure_dir(dst / "images")
    out_labels = ensure_dir(dst / "labels")
    for image_path in tqdm(list(iter_files(src / "images", IMAGE_EXTENSIONS)), desc=args.mode):
        image = image_only_defense(read_image(image_path), mode=args.mode)
        cv2.imwrite(str(out_images / image_path.name), image)
        label = src / "labels" / f"{image_path.stem}.txt"
        if label.exists():
            shutil.copy2(label, out_labels / label.name)
    print(f"Wrote image-only defense split: {dst.resolve()}")


if __name__ == "__main__":
    main()
