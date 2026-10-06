"""Prepare a small annotated public-video pilot from the Kaggle HELMET dataset.

The script downloads one HELMET clip frame-by-frame with the Kaggle API,
converts its annotations to YOLO format, and optionally injects a localized
perturbation window. It is intentionally small so the paper can report a
reproducible public-video stress pilot without downloading the full dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from advtraffic.attacks.physical import apply_occlusion_attack, apply_reflective_pattern_attack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Kaggle HELMET public-video TCVM pilot.")
    parser.add_argument("--dataset", default="hozngvan/helmet-detection")
    parser.add_argument("--clip", default="Bago_highway_10")
    parser.add_argument("--split", default="test")
    parser.add_argument("--start-frame", type=int, default=48, help="First source frame index to include.")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--source-root", default="data/source/helmet-detection-kaggle")
    parser.add_argument("--output-root", default="outputs/tcvm_analysis/public_helmet_bago_highway10_shortgap_full_occlusion")
    parser.add_argument("--class-id", type=int, default=3, help="YOLO label id; COCO motorcycle is 3.")
    parser.add_argument("--attack-type", default="occlusion", choices=["reflective", "occlusion"])
    parser.add_argument("--attack-start", type=int, default=52)
    parser.add_argument("--attack-end", type=int, default=54)
    parser.add_argument("--max-attack-boxes", type=int, default=64)
    parser.add_argument("--reflective-intensity", type=float, default=0.82)
    parser.add_argument("--reflective-scale", type=float, default=0.92)
    parser.add_argument("--stripe-width", type=int, default=9)
    parser.add_argument("--occlusion-ratio", type=float, default=1.0)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def download_file(dataset: str, file_name: str, target_dir: Path, no_download: bool) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(file_name).name
    if target.exists() or no_download:
        return
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(dataset, file_name, path=str(target_dir), force=True, quiet=True)


def read_annotation(path: Path) -> list[tuple[str, str, float, float, float, float]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 6:
            continue
        label, track_id, cx, cy, w, h = parts
        rows.append((label, track_id, float(cx), float(cy), float(w), float(h)))
    return rows


def write_yolo_labels(rows: list[tuple[str, str, float, float, float, float]], out_path: Path, class_id: int, width: int, height: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for _, _, cx, cy, w, h in rows:
        if w <= 1 or h <= 1:
            continue
        lines.append(f"{class_id} {cx / width:.6f} {cy / height:.6f} {w / width:.6f} {h / height:.6f}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def row_to_xyxy(row: tuple[str, str, float, float, float, float]) -> tuple[float, float, float, float]:
    _, _, cx, cy, w, h = row
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    clip_root = source_root / args.split / args.clip
    output_root = Path(args.output_root)
    clean_root = output_root / "clean"
    attack_root = output_root / args.attack_type

    source_frames = list(range(args.start_frame, args.start_frame + args.frames))

    for frame_idx in tqdm(source_frames, desc="Downloading HELMET pilot"):
        stem = f"{frame_idx:02d}" if frame_idx < 100 else str(frame_idx)
        download_file(
            args.dataset,
            f"{args.split}/{args.clip}/images/{stem}.jpg",
            clip_root / "images",
            args.no_download,
        )
        download_file(
            args.dataset,
            f"{args.split}/{args.clip}/annotations/{stem}.txt",
            clip_root / "annotations",
            args.no_download,
        )

    frame_rows = []
    for root in (clean_root, attack_root):
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)

    for zero_idx, frame_idx in enumerate(tqdm(source_frames, desc="Preparing sequences")):
        stem = f"{frame_idx:02d}" if frame_idx < 100 else str(frame_idx)
        image_path = clip_root / "images" / f"{stem}.jpg"
        anno_path = clip_root / "annotations" / f"{stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Missing downloaded frame: {image_path}")
        height, width = image.shape[:2]
        rows = read_annotation(anno_path)
        out_name = f"frame_{zero_idx:04d}.jpg"
        label_name = f"frame_{zero_idx:04d}.txt"

        shutil.copy2(image_path, clean_root / "images" / out_name)
        write_yolo_labels(rows, clean_root / "labels" / label_name, args.class_id, width, height)

        attacked = image.copy()
        adversarial_label = int(args.attack_start <= frame_idx <= args.attack_end)
        if adversarial_label:
            boxes = sorted((row_to_xyxy(row) for row in rows), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            for box in boxes[: args.max_attack_boxes]:
                if args.attack_type == "reflective":
                    attacked = apply_reflective_pattern_attack(
                        attacked,
                        box=np.asarray(box),
                        intensity=args.reflective_intensity,
                        stripe_width=args.stripe_width,
                        scale=args.reflective_scale,
                    )
                elif args.attack_type == "occlusion":
                    attacked = apply_occlusion_attack(
                        attacked,
                        box=np.asarray(box),
                        occlusion_ratio=args.occlusion_ratio,
                    )
                else:
                    raise ValueError(f"Unsupported attack type: {args.attack_type}")
        cv2.imwrite(str(attack_root / "images" / out_name), attacked)
        write_yolo_labels(rows, attack_root / "labels" / label_name, args.class_id, width, height)
        frame_rows.append(
            {
                "frame_id": zero_idx,
                "source_frame": frame_idx,
                "adversarial_label": adversarial_label,
                "objects": len(rows),
            }
        )

    pd.DataFrame(frame_rows).to_csv(attack_root / "frame_labels.csv", index=False)
    metadata = {
        "dataset": args.dataset,
        "dataset_url": f"https://www.kaggle.com/datasets/{args.dataset}",
        "clip": args.clip,
        "split": args.split,
        "start_frame": args.start_frame,
        "frames": args.frames,
        "source_resolution": [width, height],
        "class_id": args.class_id,
        "class_semantics": "COCO motorcycle / HELMET motorcycle-box pilot target",
        "attack": f"{args.attack_type} perturbation applied to annotated motorcycle boxes",
        "attack_window_source_frames": [args.attack_start, args.attack_end],
        "attack_parameters": {
            "max_attack_boxes": args.max_attack_boxes,
            "reflective_intensity": args.reflective_intensity,
            "reflective_scale": args.reflective_scale,
            "stripe_width": args.stripe_width,
            "occlusion_ratio": args.occlusion_ratio,
        },
        "note": "Uses public real traffic frames and human annotations; perturbation itself is synthetic and should not be described as a printed physical attack.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
