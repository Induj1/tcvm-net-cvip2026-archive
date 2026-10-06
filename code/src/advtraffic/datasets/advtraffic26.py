"""AdvTraffic-26 dataset preparation utilities."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import cv2

from advtraffic.config import write_yaml
from advtraffic.utils.io import VIDEO_EXTENSIONS, ensure_dir, iter_files
from advtraffic.utils.video import iter_video_frames


ADVTRAFFIC_CLASSES = ["helmet", "no_helmet", "rider", "motorcycle", "license_plate", "violation"]

SUBSETS = ["train", "val", "test"]
ATTACK_SPLITS = ["clean", "fgsm", "pgd", "patch", "sticker", "reflective", "occlusion", "motion_blur", "low_light"]


def create_advtraffic_structure(root: str | Path) -> dict[str, Path]:
    """Create the AdvTraffic-26 folder structure and return key paths."""

    root = Path(root)
    paths: dict[str, Path] = {"root": root}
    for subset in SUBSETS:
        paths[f"images_{subset}"] = ensure_dir(root / "images" / subset)
        paths[f"labels_{subset}"] = ensure_dir(root / "labels" / subset)
    for attack in ATTACK_SPLITS:
        for subset in SUBSETS:
            ensure_dir(root / "adversarial" / attack / "images" / subset)
            ensure_dir(root / "adversarial" / attack / "labels" / subset)
    for folder in ["raw/videos/normal", "raw/videos/physical", "metadata", "exports/cvat", "exports/labelstudio"]:
        ensure_dir(root / folder)

    write_yolo_data_yaml(root)
    return paths


def stable_split(video_path: Path, train_ratio: float = 0.7, val_ratio: float = 0.15) -> str:
    digest = hashlib.sha1(str(video_path).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def extract_video_frames(
    raw_video_root: str | Path,
    dataset_root: str | Path,
    stride: int = 5,
    resize: tuple[int, int] | None = None,
    max_frames_per_video: int | None = None,
) -> Path:
    """Extract sampled video frames and write metadata/frame_index.csv."""

    raw_video_root = Path(raw_video_root)
    dataset_root = Path(dataset_root)
    create_advtraffic_structure(dataset_root)
    metadata_path = dataset_root / "metadata" / "frame_index.csv"

    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_path", "subset", "frame_id", "timestamp_ms", "image_path", "label_path"],
        )
        writer.writeheader()
        for video_path in iter_files(raw_video_root, VIDEO_EXTENSIONS):
            subset = stable_split(video_path)
            video_stem = video_path.stem.replace(" ", "_")
            for frame in iter_video_frames(video_path, stride=stride, resize=resize, max_frames=max_frames_per_video):
                image_name = f"{video_stem}_f{frame.frame_id:06d}.jpg"
                image_path = dataset_root / "images" / subset / image_name
                label_path = dataset_root / "labels" / subset / f"{image_path.stem}.txt"
                cv2.imwrite(str(image_path), frame.image)
                label_path.touch(exist_ok=True)
                writer.writerow(
                    {
                        "video_path": str(video_path),
                        "subset": subset,
                        "frame_id": frame.frame_id,
                        "timestamp_ms": f"{frame.timestamp_ms:.3f}",
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                    }
                )
    return metadata_path


def write_yolo_data_yaml(dataset_root: str | Path) -> Path:
    dataset_root = Path(dataset_root)
    yaml_path = dataset_root / "advtraffic26.yaml"
    data = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(ADVTRAFFIC_CLASSES)},
    }
    write_yaml(yaml_path, data)
    return yaml_path
