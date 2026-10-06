"""Validate a YOLO dataset and generate statistics, plots, and sample grids."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from advtraffic.utils.io import IMAGE_EXTENSIONS, ensure_dir, iter_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and visualize a YOLO-format dataset.")
    parser.add_argument("--data", required=True, help="YOLO data YAML.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-split", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data_yaml(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = Path(data["path"])
    if not root.is_absolute():
        root = (Path(path).parent / root).resolve() if not Path(data["path"]).exists() else Path(data["path"]).resolve()
    data["_root"] = root
    names = data.get("names", {})
    if isinstance(names, list):
        data["_names"] = {idx: name for idx, name in enumerate(names)}
    else:
        data["_names"] = {int(idx): name for idx, name in names.items()}
    return data


def resolve_split_paths(data: dict, split: str) -> tuple[Path, Path]:
    """Resolve common YOLO layouts: images/train, train/images, or absolute split paths."""
    root = data["_root"]
    split_value = data.get(split)
    image_root = Path(split_value) if split_value else root / "images" / split
    if not image_root.is_absolute():
        image_root = root / image_root
    image_root = image_root.resolve()

    label_parts = list(Path(split_value).parts) if split_value else ["images", split]
    if "images" in label_parts:
        label_parts[label_parts.index("images")] = "labels"
        label_root = root.joinpath(*label_parts).resolve()
    else:
        label_root = root / "labels" / split
    return image_root, label_root


def iter_split_images(image_root: Path) -> list[Path]:
    if not image_root.exists():
        return []
    return sorted(path for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def validate_label_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(float(parts[0]))
        xc, yc, bw, bh = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1 and bw > 0 and bh > 0):
        return None
    return class_id, xc, yc, bw, bh


def analyze_split(data: dict, split: str, names: dict[int, str]) -> tuple[dict, pd.DataFrame]:
    image_root, label_root = resolve_split_paths(data, split)
    images = iter_split_images(image_root)
    rows = []
    invalid = 0
    missing_labels = 0
    empty_labels = 0
    class_counts: Counter[str] = Counter()
    box_areas = []
    for image_path in images:
        label_path = label_root / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
        for line in lines:
            parsed = validate_label_line(line)
            if parsed is None:
                invalid += 1
                continue
            class_id, xc, yc, bw, bh = parsed
            class_name = names.get(class_id, str(class_id))
            class_counts[class_name] += 1
            area = bw * bh
            box_areas.append(area)
            rows.append(
                {
                    "split": split,
                    "image": str(image_path),
                    "class_id": class_id,
                    "class_name": class_name,
                    "xc": xc,
                    "yc": yc,
                    "w": bw,
                    "h": bh,
                    "area": area,
                }
            )
    stats = {
        "split": split,
        "images": len(images),
        "objects": int(sum(class_counts.values())),
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "invalid_labels": invalid,
        "class_counts": dict(class_counts),
        "mean_box_area": float(np.mean(box_areas)) if box_areas else 0.0,
        "median_box_area": float(np.median(box_areas)) if box_areas else 0.0,
    }
    return stats, pd.DataFrame(rows)


def draw_sample_grid(data: dict, split: str, df: pd.DataFrame, output_dir: Path, samples_per_split: int, seed: int) -> None:
    names = data["_names"]
    image_root, label_root = resolve_split_paths(data, split)
    images = iter_split_images(image_root)
    if not images:
        return
    rng = random.Random(seed)
    sample_paths = rng.sample(images, k=min(samples_per_split, len(images)))
    cols = 3
    rows = int(np.ceil(len(sample_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.1))
    axes = np.asarray(axes).reshape(-1)
    for ax, image_path in zip(axes, sample_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            ax.axis("off")
            continue
        h, w = image.shape[:2]
        label_path = label_root / f"{image_path.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []:
            parsed = validate_label_line(line)
            if parsed is None:
                continue
            class_id, xc, yc, bw, bh = parsed
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 210, 80), 2)
            cv2.putText(image, names.get(class_id, str(class_id)), (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 80), 1, cv2.LINE_AA)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(image_path.name, fontsize=8)
        ax.axis("off")
    for ax in axes[len(sample_paths) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"samples_{split}.png", dpi=220)
    plt.close(fig)


def plot_class_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    counts = df.groupby(["split", "class_name"]).size().reset_index(name="count")
    pivot = counts.pivot(index="class_name", columns="split", values="count").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(7.5, 4.0), width=0.76)
    ax.set_xlabel("")
    ax.set_ylabel("Object count")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "class_distribution.png", dpi=300)
    ax.figure.savefig(output_dir / "class_distribution.pdf")
    plt.close(ax.figure)


def plot_box_area(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for split, split_df in df.groupby("split"):
        ax.hist(split_df["area"], bins=50, alpha=0.55, label=split)
    ax.set_xlabel("Normalized bounding-box area")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "box_area_distribution.png", dpi=300)
    fig.savefig(output_dir / "box_area_distribution.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_data_yaml(args.data)
    output_dir = ensure_dir(args.output_dir)
    all_stats = {}
    frames = []
    for split in ("train", "val", "test"):
        stats, df = analyze_split(data, split, data["_names"])
        all_stats[split] = stats
        print(f"{split}: {stats['images']} images, {stats['objects']} objects, {stats['invalid_labels']} invalid labels", flush=True)
        if not df.empty:
            frames.append(df)
        draw_sample_grid(data, split, df, output_dir, args.samples_per_split, args.seed)
    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    all_df.to_csv(output_dir / "annotations.csv", index=False)
    (output_dir / "dataset_stats.json").write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    plot_class_distribution(all_df, output_dir)
    plot_box_area(all_df, output_dir)
    print(json.dumps(all_stats, indent=2))


if __name__ == "__main__":
    main()
