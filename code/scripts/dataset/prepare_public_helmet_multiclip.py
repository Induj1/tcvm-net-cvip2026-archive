"""Prepare a multi-clip annotated HELMET benchmark for temporal evaluation.

This script uses the public Kaggle HELMET split file and per-clip CSV
annotations. It selects several held-out clips, extracts a short window from
each, applies one or more synthetic short-gap attacks, and writes combined
YOLO-style sequences with clip-boundary metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from advtraffic.attacks.physical import (
    apply_motion_blur,
    apply_named_physical_attack,
    apply_occlusion_attack,
    apply_reflective_pattern_attack,
)


ATTACK_CHOICES = ["occlusion", "reflective", "sticker", "patch", "detector_patch", "motion_blur"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare multi-clip public HELMET temporal benchmark.")
    parser.add_argument("--dataset", default="hozngvan/helmet-detection")
    parser.add_argument("--source-root", default="data/source/helmet-detection-kaggle")
    parser.add_argument("--output-root", default="outputs/tcvm_analysis/public_helmet_multiclip_occlusion")
    parser.add_argument("--split-name", default="test", choices=["training", "validation", "test"])
    parser.add_argument("--clips", nargs="*", default=None, help="Optional explicit clip names.")
    parser.add_argument("--num-clips", type=int, default=5)
    parser.add_argument("--frames-per-clip", type=int, default=24)
    parser.add_argument("--attack-offset", type=int, default=4, help="0-based offset within each selected window.")
    parser.add_argument("--attack-length", type=int, default=3)
    parser.add_argument("--class-id", type=int, default=3, help="YOLO class id; COCO motorcycle is 3.")
    parser.add_argument("--attack-type", default="occlusion", choices=ATTACK_CHOICES)
    parser.add_argument(
        "--attack-types",
        nargs="+",
        choices=ATTACK_CHOICES,
        default=None,
        help="Generate several attack sequences over identical clip windows.",
    )
    parser.add_argument("--max-attack-boxes", type=int, default=64)
    parser.add_argument("--occlusion-ratio", type=float, default=1.0)
    parser.add_argument("--reflective-intensity", type=float, default=0.82)
    parser.add_argument("--reflective-scale", type=float, default=0.92)
    parser.add_argument("--stripe-width", type=int, default=9)
    parser.add_argument("--sticker-scale", type=float, default=0.38)
    parser.add_argument("--patch-scale", type=float, default=0.45)
    parser.add_argument("--patch-path", default=None)
    parser.add_argument("--motion-blur-kernel", type=int, default=17)
    parser.add_argument("--motion-blur-angle", type=float, default=0.0)
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
    last_error: Exception | None = None
    for attempt in range(1, 7):
        try:
            api.dataset_download_file(dataset, file_name, path=str(target_dir), force=True, quiet=True)
            if target.exists():
                return
        except Exception as exc:
            last_error = exc
        if attempt < 6:
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"Failed to download {file_name} after retries.") from last_error


def load_split_file(args: argparse.Namespace, source_root: Path) -> pd.DataFrame:
    download_file(args.dataset, "data_split.csv", source_root, args.no_download)
    split_path = source_root / "data_split.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing HELMET split file: {split_path}")
    return pd.read_csv(split_path)


def load_annotation_csv(args: argparse.Namespace, source_root: Path, clip: str) -> pd.DataFrame:
    target_dir = source_root / "annotation" / "annotation"
    download_file(args.dataset, f"annotation/annotation/{clip}.csv", target_dir, args.no_download)
    path = target_dir / f"{clip}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing HELMET annotation CSV: {path}")
    df = pd.read_csv(path)
    required = {"track_id", "frame_id", "x", "y", "w", "h", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def choose_window(df: pd.DataFrame, frames_per_clip: int, attack_offset: int, attack_length: int) -> tuple[int, float]:
    counts = df.groupby("frame_id").size().to_dict()
    max_frame = int(df["frame_id"].max())
    best_start, best_score = 1, -1.0
    for start in range(1, max_frame - frames_per_clip + 2):
        frames = list(range(start, start + frames_per_clip))
        attack_frames = frames[attack_offset : attack_offset + attack_length]
        if not attack_frames:
            continue
        attack_min = min(counts.get(frame, 0) for frame in attack_frames)
        history_sum = sum(counts.get(frame, 0) for frame in frames[:attack_offset])
        future_sum = sum(counts.get(frame, 0) for frame in frames[attack_offset + attack_length :])
        score = 8.0 * attack_min + 0.25 * history_sum + 0.05 * future_sum
        if attack_min > 0 and score > best_score:
            best_start, best_score = start, score
    if best_score < 0:
        raise ValueError("Could not find a window with annotated objects in the attack interval.")
    return best_start, best_score


def rows_for_frame(df: pd.DataFrame, frame_id: int) -> list[tuple[str, str, float, float, float, float]]:
    rows = []
    for row in df[df["frame_id"] == frame_id].itertuples(index=False):
        x1, y1, w, h = float(row.x), float(row.y), float(row.w), float(row.h)
        rows.append((str(row.label), str(row.track_id), x1, y1, w, h))
    return rows


def write_yolo_labels(rows: list[tuple[str, str, float, float, float, float]], out_path: Path, class_id: int, width: int, height: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for _, _, x1, y1, w, h in rows:
        if w <= 1 or h <= 1:
            continue
        cx, cy = x1 + w / 2, y1 + h / 2
        lines.append(f"{class_id} {cx / width:.6f} {cy / height:.6f} {w / width:.6f} {h / height:.6f}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def row_to_xyxy(row: tuple[str, str, float, float, float, float]) -> tuple[float, float, float, float]:
    _, _, x1, y1, w, h = row
    return (x1, y1, x1 + w, y1 + h)


def apply_attack(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    args: argparse.Namespace,
    attack_type: str | None = None,
) -> np.ndarray:
    attack_type = attack_type or args.attack_type
    if attack_type == "motion_blur":
        return apply_motion_blur(
            image,
            kernel_size=args.motion_blur_kernel,
            angle=args.motion_blur_angle,
        )

    attacked = image.copy()
    for box in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[: args.max_attack_boxes]:
        box_array = np.asarray(box)
        if attack_type == "occlusion":
            attacked = apply_occlusion_attack(attacked, box_array, occlusion_ratio=args.occlusion_ratio)
        elif attack_type == "reflective":
            attacked = apply_reflective_pattern_attack(
                attacked,
                box_array,
                intensity=args.reflective_intensity,
                stripe_width=args.stripe_width,
                scale=args.reflective_scale,
            )
        elif attack_type == "sticker":
            attacked = apply_named_physical_attack(
                attacked,
                box_array,
                "sticker",
                scale=args.sticker_scale,
            )
        elif attack_type in {"patch", "detector_patch"}:
            attacked = apply_named_physical_attack(
                attacked,
                box_array,
                "patch",
                scale=args.patch_scale,
                patch_path=args.patch_path,
            )
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}")
    return attacked


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    clean_root = output_root / "clean"
    attack_types = list(dict.fromkeys(args.attack_types or [args.attack_type]))
    attack_roots = {attack_type: output_root / attack_type for attack_type in attack_types}
    for root in (clean_root, *attack_roots.values()):
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)

    split_df = load_split_file(args, source_root)
    if args.clips:
        candidate_clips = args.clips
    else:
        candidate_clips = split_df[split_df["Set"] == args.split_name]["video_id"].tolist()

    candidates = []
    for clip in tqdm(candidate_clips, desc="Scoring HELMET clips"):
        try:
            anno = load_annotation_csv(args, source_root, clip)
            start_frame, score = choose_window(anno, args.frames_per_clip, args.attack_offset, args.attack_length)
            candidates.append((score, clip, start_frame, anno))
        except Exception as exc:
            print(f"Skipping {clip}: {exc}")

    selected = sorted(candidates, reverse=True)[: args.num_clips]
    if not selected:
        raise RuntimeError("No HELMET clips could be selected.")

    frame_rows = []
    clip_rows = []
    global_frame = 0
    for clip_idx, (score, clip, start_frame, anno) in enumerate(tqdm(selected, desc="Preparing HELMET multi-clip")):
        clip_split = split_df.loc[split_df["video_id"] == clip, "Set"].iloc[0]
        clip_root = source_root / clip_split / clip
        attack_start = start_frame + args.attack_offset
        attack_end = attack_start + args.attack_length - 1
        clip_rows.append(
            {
                "clip_id": clip_idx,
                "clip": clip,
                "split": clip_split,
                "start_frame": start_frame,
                "attack_start": attack_start,
                "attack_end": attack_end,
                "selection_score": score,
            }
        )

        for local_idx, source_frame in enumerate(range(start_frame, start_frame + args.frames_per_clip)):
            stem = f"{source_frame:02d}" if source_frame < 100 else str(source_frame)
            download_file(
                args.dataset,
                f"{clip_split}/{clip}/images/{stem}.jpg",
                clip_root / "images",
                args.no_download,
            )
            image_path = clip_root / "images" / f"{stem}.jpg"
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Missing downloaded frame: {image_path}")
            height, width = image.shape[:2]
            rows = rows_for_frame(anno, source_frame)
            out_name = f"clip{clip_idx:02d}_frame_{local_idx:04d}.jpg"
            label_name = f"clip{clip_idx:02d}_frame_{local_idx:04d}.txt"

            shutil.copy2(image_path, clean_root / "images" / out_name)
            write_yolo_labels(rows, clean_root / "labels" / label_name, args.class_id, width, height)

            adversarial_label = int(attack_start <= source_frame <= attack_end)
            boxes = [row_to_xyxy(row) for row in rows]
            for attack_type, attack_root in attack_roots.items():
                attacked = apply_attack(image, boxes, args, attack_type) if adversarial_label else image
                cv2.imwrite(str(attack_root / "images" / out_name), attacked)
                write_yolo_labels(rows, attack_root / "labels" / label_name, args.class_id, width, height)
            frame_rows.append(
                {
                    "frame_id": global_frame,
                    "clip_id": clip_idx,
                    "clip": clip,
                    "local_frame": local_idx,
                    "source_frame": source_frame,
                    "adversarial_label": adversarial_label,
                    "objects": len(rows),
                }
            )
            global_frame += 1

    frame_df = pd.DataFrame(frame_rows)
    for attack_root in attack_roots.values():
        frame_df.to_csv(attack_root / "frame_labels.csv", index=False)
    pd.DataFrame(clip_rows).to_csv(output_root / "clip_windows.csv", index=False)
    metadata = {
        "dataset": args.dataset,
        "dataset_url": f"https://www.kaggle.com/datasets/{args.dataset}",
        "split_name": args.split_name,
        "selected_clips": [row["clip"] for row in clip_rows],
        "num_clips": len(selected),
        "frames_per_clip": args.frames_per_clip,
        "total_frames": int(len(frame_df)),
        "total_objects": int(frame_df["objects"].sum()),
        "class_id": args.class_id,
        "class_semantics": "COCO motorcycle / HELMET motorcycle-box pilot target",
        "attacks": attack_types,
        "attack": "short-gap perturbations applied over identical annotated windows",
        "attack_parameters": {
            "attack_offset": args.attack_offset,
            "attack_length": args.attack_length,
            "max_attack_boxes": args.max_attack_boxes,
            "occlusion_ratio": args.occlusion_ratio,
            "reflective_intensity": args.reflective_intensity,
            "reflective_scale": args.reflective_scale,
            "stripe_width": args.stripe_width,
            "sticker_scale": args.sticker_scale,
            "patch_scale": args.patch_scale,
            "motion_blur_kernel": args.motion_blur_kernel,
            "motion_blur_angle": args.motion_blur_angle,
        },
        "note": "Uses public real traffic frames and human annotations; perturbation itself is synthetic.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
