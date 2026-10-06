"""Fast BDD100K YOLO validation for large OneDrive-backed checkouts.

The script computes full image, label-file, and class-count statistics, then
uses a deterministic label sample for box-area distributions and visual QA.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from advtraffic.utils.io import IMAGE_EXTENSIONS, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast large-scale BDD100K YOLO analysis.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-sample", type=int, default=5000)
    parser.add_argument("--samples-per-split", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-only", action="store_true", help="Skip full class counting and report sampled label statistics.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    yaml_path = Path(path)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(data["path"])
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve() if (Path.cwd() / root).exists() else (yaml_path.parent / root).resolve()
    data["_root"] = root
    names = data["names"]
    data["_names"] = {idx: name for idx, name in enumerate(names)} if isinstance(names, list) else {int(k): v for k, v in names.items()}
    return data


def resolve_paths(data: dict, split: str) -> tuple[Path, Path]:
    root = data["_root"]
    candidates = [root / "images" / split, root / split / "images"]
    split_value = data.get(split)
    if split_value:
        split_path = Path(split_value)
        candidates.insert(0, split_path if split_path.is_absolute() else root / split_path)
    image_root = next(path for path in candidates if path.exists())
    parts = list(image_root.parts)
    parts[parts.index("images")] = "labels"
    return image_root, Path(*parts)


def list_files(root: Path, extensions: set[str] | None = None) -> list[Path]:
    out = []
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            path = Path(entry.path)
            if extensions is None or path.suffix.lower() in extensions:
                out.append(path)
    return sorted(out)


def full_class_counts(label_root: Path, names: dict[int, str]) -> Counter[str]:
    try:
        proc = subprocess.run(
            ["rg", "--no-filename", "-o", r"^[0-9]+", str(label_root)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        ids = [int(line) for line in proc.stdout.splitlines() if line.strip().isdigit()]
    except FileNotFoundError:
        ids = []
        for label_path in list_files(label_root, {".txt"}):
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if parts and parts[0].isdigit():
                    ids.append(int(parts[0]))
    return Counter(names.get(class_id, str(class_id)) for class_id in ids)


def parse_label_sample(label_paths: list[Path], names: dict[int, str], sample_size: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    sampled = rng.sample(label_paths, k=min(sample_size, len(label_paths)))
    rows = []
    for label_path in sampled:
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) != 5:
                rows.append({"label": str(label_path), "valid": False})
                continue
            try:
                class_id = int(float(parts[0]))
                xc, yc, bw, bh = [float(v) for v in parts[1:]]
                valid = 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1
            except ValueError:
                rows.append({"label": str(label_path), "valid": False})
                continue
            rows.append(
                {
                    "label": str(label_path),
                    "valid": valid,
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)),
                    "area": bw * bh if valid else np.nan,
                }
            )
    return pd.DataFrame(rows)


def draw_samples(image_paths: list[Path], label_root: Path, names: dict[int, str], output: Path, seed: int, count: int) -> None:
    rng = random.Random(seed)
    sample = rng.sample(image_paths, k=min(count, len(image_paths)))
    if not sample:
        return
    cols = 3
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.0))
    axes = np.asarray(axes).reshape(-1)
    for ax, image_path in zip(axes, sample):
        image = cv2.imread(str(image_path))
        if image is None:
            ax.axis("off")
            continue
        h, w = image.shape[:2]
        label_path = label_root / f"{image_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(float(parts[0]))
                xc, yc, bw, bh = [float(v) for v in parts[1:]]
                x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
                x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 210, 80), 2)
                cv2.putText(image, names.get(class_id, str(class_id)), (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 80), 1, cv2.LINE_AA)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(image_path.name, fontsize=8)
        ax.axis("off")
    for ax in axes[len(sample) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_yaml(args.data)
    output_dir = ensure_dir(args.output_dir)
    names = data["_names"]
    stats = {}
    sample_frames = []

    for split in ("train", "val", "test"):
        image_root, label_root = resolve_paths(data, split)
        images = list_files(image_root, IMAGE_EXTENSIONS)
        labels = list_files(label_root, {".txt"})
        sample_df = parse_label_sample(labels, names, args.label_sample, args.seed)
        valid_sample = sample_df[sample_df.get("valid", False) == True] if not sample_df.empty else sample_df
        if args.sample_only:
            class_counts = Counter(valid_sample["class_name"]) if not valid_sample.empty else Counter()
            object_basis = "sampled_labels"
        else:
            class_counts = full_class_counts(label_root, names)
            object_basis = "full_labels"
        stats[split] = {
            "split": split,
            "analysis_mode": "full image/label-file counts; sampled annotation validation" if args.sample_only else "full image/label/class counts; sampled bbox validation",
            "images": len(images),
            "label_files": len(labels),
            "objects": int(sum(class_counts.values())),
            "object_count_basis": object_basis,
            "class_counts_are_full": not args.sample_only,
            "class_counts": dict(class_counts),
            "sampled_label_files": min(args.label_sample, len(labels)),
            "sampled_objects": int(len(sample_df)),
            "sampled_invalid_labels": int((~sample_df["valid"]).sum()) if not sample_df.empty else 0,
            "sampled_mean_box_area": float(valid_sample["area"].mean()) if not valid_sample.empty else 0.0,
            "sampled_median_box_area": float(valid_sample["area"].median()) if not valid_sample.empty else 0.0,
        }
        sample_df["split"] = split
        sample_frames.append(sample_df)
        draw_samples(images, label_root, names, output_dir / f"samples_{split}.png", args.seed, args.samples_per_split)
        print(f"{split}: {len(images)} images, {sum(class_counts.values())} {object_basis} objects", flush=True)

    all_samples = pd.concat(sample_frames, ignore_index=True)
    all_samples.to_csv(output_dir / "annotations_sample.csv", index=False)
    (output_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    counts = []
    for split, split_stats in stats.items():
        for class_name, count in split_stats["class_counts"].items():
            counts.append({"split": split, "class_name": class_name, "count": count})
    counts_df = pd.DataFrame(counts)
    pivot = counts_df.pivot(index="class_name", columns="split", values="count").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(8.5, 4.2), width=0.76)
    ax.set_xlabel("")
    ax.set_ylabel("Object count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "class_distribution.png", dpi=300)
    ax.figure.savefig(output_dir / "class_distribution.pdf")
    plt.close(ax.figure)

    valid = all_samples[all_samples["valid"] == True]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for split, split_df in valid.groupby("split"):
        ax.hist(split_df["area"], bins=50, alpha=0.55, label=split)
    ax.set_xlabel("Normalized bounding-box area")
    ax.set_ylabel("Sampled object count")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "box_area_distribution.png", dpi=300)
    fig.savefig(output_dir / "box_area_distribution.pdf")
    plt.close(fig)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
