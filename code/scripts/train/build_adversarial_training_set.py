"""Build a YOLO dataset for adversarial training from clean and attacked samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from advtraffic.datasets.conversion import load_yolo_names, write_dataset_yaml
from advtraffic.utils.io import IMAGE_EXTENSIONS, ensure_dir, iter_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create adversarial-training dataset.")
    parser.add_argument("--clean-root", required=True, help="Clean YOLO dataset root.")
    parser.add_argument("--attacks-root", default="outputs/attacks")
    parser.add_argument("--output-root", default="data/AdvTraffic-26-advtrain")
    parser.add_argument("--attacks", nargs="+", default=["fgsm", "pgd", "sticker", "reflective", "occlusion", "low_light"])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--copy-val-test", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def copy_split(source_root: Path, output_root: Path, split: str, prefix: str = "") -> int:
    count = 0
    image_root = source_root / "images" / split if (source_root / "images" / split).exists() else source_root / "images"
    label_root = source_root / "labels" / split if (source_root / "labels" / split).exists() else source_root / "labels"
    for image_path in iter_files(image_root, IMAGE_EXTENSIONS):
        name = f"{prefix}{image_path.name}"
        dst_image = ensure_dir(output_root / "images" / split) / name
        dst_label = ensure_dir(output_root / "labels" / split) / f"{Path(name).stem}.txt"
        shutil.copy2(image_path, dst_image)
        label_path = label_root / f"{image_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, dst_label)
        else:
            dst_label.touch()
        count += 1
    return count


def main() -> None:
    args = parse_args()
    clean_root = Path(args.clean_root)
    output_root = Path(args.output_root)
    stats = {"clean_train": copy_split(clean_root, output_root, args.train_split, prefix="clean_")}
    for attack in args.attacks:
        attack_root = Path(args.attacks_root) / attack / args.train_split
        if attack_root.exists():
            stats[f"{attack}_train"] = copy_split(attack_root, output_root, args.train_split, prefix=f"{attack}_")
    if args.copy_val_test:
        for split in ("val", "test"):
            if (clean_root / "images" / split).exists():
                stats[f"clean_{split}"] = copy_split(clean_root, output_root, split, prefix="clean_")
    source_yaml = next(clean_root.glob("*.yaml"), None)
    names = load_yolo_names(source_yaml) if source_yaml else {}
    classes = [names[idx] for idx in sorted(names)]
    write_dataset_yaml(output_root, classes, filename="advtrain.yaml")
    print(stats)
    print(f"Wrote {output_root / 'advtrain.yaml'}")


if __name__ == "__main__":
    main()
