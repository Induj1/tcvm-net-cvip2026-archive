"""Merge prepared BDD100K and helmet YOLO datasets into AdvTraffic-26."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from advtraffic.datasets import ADVTRAFFIC26_CLASSES, BDD_TO_ADVTRAFFIC, HELMET_TO_ADVTRAFFIC, merge_yolo_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AdvTraffic-26 from prepared source datasets.")
    parser.add_argument("--bdd-yolo", default="data/bdd100k-yolo", help="Prepared BDD100K YOLO dataset.")
    parser.add_argument("--helmet-yolo", default="data/helmet-yolo", help="Prepared helmet YOLO dataset.")
    parser.add_argument("--output-dir", default="data/AdvTraffic-26")
    parser.add_argument("--link-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = []
    if Path(args.bdd_yolo).exists():
        sources.append((args.bdd_yolo, BDD_TO_ADVTRAFFIC))
    if Path(args.helmet_yolo).exists():
        sources.append((args.helmet_yolo, HELMET_TO_ADVTRAFFIC))
    if not sources:
        raise FileNotFoundError("No prepared source datasets found. Run prepare_bdd100k.py and/or prepare_helmet_dataset.py.")
    stats = merge_yolo_datasets(
        sources=sources,
        output_dir=args.output_dir,
        classes=ADVTRAFFIC26_CLASSES,
        link_images=args.link_images,
    )
    manifest = {
        "dataset": "AdvTraffic-26",
        "sources": [str(Path(src).resolve()) for src, _ in sources],
        "classes": ADVTRAFFIC26_CLASSES,
        "stats": stats,
    }
    manifest_path = Path(args.output_dir) / "metadata" / "advtraffic26_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"yaml": str(Path(args.output_dir) / "advtraffic26.yaml"), "manifest": str(manifest_path), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
