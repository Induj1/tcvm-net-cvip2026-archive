"""Dataset conversion helpers for BDD100K, helmet data, and AdvTraffic-26."""

from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from advtraffic.config import write_yaml
from advtraffic.utils.geometry import clip_box, xyxy_to_yolo
from advtraffic.utils.io import IMAGE_EXTENSIONS, ensure_dir, iter_files


BDD100K_CLASSES = ["bike", "bus", "car", "motor", "person", "rider", "traffic light", "traffic sign", "train", "truck"]
HELMET_CLASSES = ["helmet", "no_helmet", "rider"]
ADVTRAFFIC26_CLASSES = ["helmet", "no_helmet", "rider", "motorcycle", "license_plate", "violation"]

BDD_TO_ADVTRAFFIC = {
    "rider": "rider",
    "person": "rider",
    "motor": "motorcycle",
    "bike": "motorcycle",
}

HELMET_TO_ADVTRAFFIC = {
    "helmet": "helmet",
    "hardhat": "helmet",
    "hat": "helmet",
    "head": "no_helmet",
    "no_helmet": "no_helmet",
    "no helmet": "no_helmet",
    "person": "rider",
    "rider": "rider",
}


@dataclass
class YoloObject:
    class_name: str
    xyxy: np.ndarray


def read_image_shape(image_path: str | Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def write_yolo_label(label_path: str | Path, objects: Iterable[YoloObject], classes: list[str], width: int, height: int) -> int:
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    count = 0
    with label_path.open("w", encoding="utf-8") as handle:
        for obj in objects:
            if obj.class_name not in class_to_id:
                continue
            line = xyxy_to_yolo(obj.xyxy, class_to_id[obj.class_name], width, height)
            handle.write(line + "\n")
            count += 1
    return count


def copy_or_link_image(src: str | Path, dst: str | Path, link: bool = False) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            pass
        try:
            dst.hardlink_to(src.resolve())
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def write_dataset_yaml(root: str | Path, classes: list[str], filename: str = "data.yaml") -> Path:
    root = Path(root)
    yaml_path = root / filename
    write_yaml(
        yaml_path,
        {
            "path": str(root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {idx: name for idx, name in enumerate(classes)},
        },
    )
    return yaml_path


def ensure_yolo_split_dirs(root: str | Path, splits: Iterable[str] = ("train", "val", "test")) -> None:
    root = Path(root)
    for split in splits:
        ensure_dir(root / "images" / split)
        ensure_dir(root / "labels" / split)


def convert_bdd100k_detection_split(
    source_dir: str | Path,
    output_dir: str | Path,
    split: str,
    classes: list[str] | None = None,
    class_map: dict[str, str] | None = None,
    max_images: int | None = None,
    link_images: bool = False,
) -> dict[str, int]:
    """Convert one BDD100K detection split to YOLO format.

    Expected source layout follows the official/FiftyOne structure:
    labels/bdd100k_labels_images_train.json and images/100k/train/*.jpg.
    """

    classes = classes or BDD100K_CLASSES
    class_map = class_map or {name: name for name in classes}
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    label_json = source_dir / "labels" / f"bdd100k_labels_images_{split}.json"
    image_root = source_dir / "images" / "100k" / split
    if not label_json.exists():
        raise FileNotFoundError(f"BDD100K label JSON not found: {label_json}")
    if not image_root.exists():
        raise FileNotFoundError(f"BDD100K image root not found: {image_root}")

    ensure_yolo_split_dirs(output_dir, splits=[split])
    records = json.loads(label_json.read_text(encoding="utf-8"))
    if max_images is not None:
        records = records[:max_images]

    written_images = 0
    written_objects = 0
    skipped_missing = 0
    for record in records:
        image_name = record["name"]
        image_path = image_root / image_name
        if not image_path.exists():
            skipped_missing += 1
            continue
        width, height = read_image_shape(image_path)
        objects: list[YoloObject] = []
        for label in record.get("labels", []):
            raw_class = label.get("category")
            target_class = class_map.get(raw_class)
            box = label.get("box2d")
            if target_class is None or target_class not in classes or box is None:
                continue
            xyxy = clip_box(np.array([box["x1"], box["y1"], box["x2"], box["y2"]], dtype=float), width, height)
            if (xyxy[2] - xyxy[0]) < 2 or (xyxy[3] - xyxy[1]) < 2:
                continue
            objects.append(YoloObject(class_name=target_class, xyxy=xyxy))

        dst_image = output_dir / "images" / split / image_name
        dst_label = output_dir / "labels" / split / f"{Path(image_name).stem}.txt"
        copy_or_link_image(image_path, dst_image, link=link_images)
        written_objects += write_yolo_label(dst_label, objects, classes, width, height)
        written_images += 1

    write_dataset_yaml(output_dir, classes, filename=f"{Path(output_dir).name}.yaml")
    return {"images": written_images, "objects": written_objects, "missing_images": skipped_missing}


def parse_voc_xml(xml_path: str | Path, class_map: dict[str, str], classes: list[str]) -> tuple[str, list[YoloObject]]:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename") or f"{Path(xml_path).stem}.jpg"
    objects: list[YoloObject] = []
    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip().lower().replace("-", "_")
        target = class_map.get(raw_name)
        bbox = obj.find("bndbox")
        if target not in classes or bbox is None:
            continue
        xyxy = np.array(
            [
                float(bbox.findtext("xmin", "0")),
                float(bbox.findtext("ymin", "0")),
                float(bbox.findtext("xmax", "0")),
                float(bbox.findtext("ymax", "0")),
            ],
            dtype=float,
        )
        objects.append(YoloObject(target, xyxy))
    return filename, objects


def convert_voc_helmet_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    classes: list[str] | None = None,
    class_map: dict[str, str] | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_images: int | None = None,
    link_images: bool = False,
) -> dict[str, int]:
    """Convert a VOC-style helmet dataset such as Kaggle hard-hat detection."""

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    classes = classes or HELMET_CLASSES
    class_map = class_map or HELMET_TO_ADVTRAFFIC
    annotation_dir = source_dir / "annotations"
    image_dirs = [source_dir / "images", source_dir]
    if not annotation_dir.exists():
        raise FileNotFoundError(f"VOC annotations directory not found: {annotation_dir}")

    xml_files = sorted(annotation_dir.glob("*.xml"))
    if max_images is not None:
        xml_files = xml_files[:max_images]
    random.Random(seed).shuffle(xml_files)
    ensure_yolo_split_dirs(output_dir)

    stats = {"train": 0, "val": 0, "test": 0, "objects": 0, "missing_images": 0}
    for idx, xml_path in enumerate(xml_files):
        fraction = idx / max(len(xml_files), 1)
        split = "train" if fraction < train_ratio else "val" if fraction < train_ratio + val_ratio else "test"
        filename, objects = parse_voc_xml(xml_path, class_map=class_map, classes=classes)
        image_path = find_image_for_annotation(filename, image_dirs)
        if image_path is None:
            stats["missing_images"] += 1
            continue
        width, height = read_image_shape(image_path)
        objects = [YoloObject(obj.class_name, clip_box(obj.xyxy, width, height)) for obj in objects]
        dst_image = output_dir / "images" / split / image_path.name
        dst_label = output_dir / "labels" / split / f"{image_path.stem}.txt"
        copy_or_link_image(image_path, dst_image, link=link_images)
        stats["objects"] += write_yolo_label(dst_label, objects, classes, width, height)
        stats[split] += 1

    write_dataset_yaml(output_dir, classes, filename=f"{Path(output_dir).name}.yaml")
    return stats


def find_image_for_annotation(filename: str, image_dirs: Iterable[Path]) -> Path | None:
    candidates = [filename]
    stem = Path(filename).stem
    candidates.extend(f"{stem}{ext}" for ext in IMAGE_EXTENSIONS)
    for image_dir in image_dirs:
        for candidate in candidates:
            path = image_dir / candidate
            if path.exists():
                return path
    return None


def merge_yolo_datasets(
    sources: list[tuple[str | Path, dict[str, str]]],
    output_dir: str | Path,
    classes: list[str] | None = None,
    link_images: bool = False,
) -> dict[str, int]:
    """Merge YOLO datasets into AdvTraffic-26 using class-name mappings."""

    output_dir = Path(output_dir)
    classes = classes or ADVTRAFFIC26_CLASSES
    ensure_yolo_split_dirs(output_dir)
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    stats = {"images": 0, "objects": 0}

    for source_root, mapping in sources:
        source_root = Path(source_root)
        source_yaml = next(source_root.glob("*.yaml"), None)
        source_names = load_yolo_names(source_yaml) if source_yaml else {}
        for split in ("train", "val", "test"):
            image_root, label_root = resolve_yolo_split_roots(source_root, source_yaml, split)
            if not image_root.exists():
                continue
            for image_path in iter_files(image_root, IMAGE_EXTENSIONS):
                label_path = label_root / f"{image_path.stem}.txt"
                remapped_lines = []
                if label_path.exists():
                    for line in label_path.read_text(encoding="utf-8").splitlines():
                        parts = line.split()
                        if len(parts) != 5:
                            continue
                        src_class_id = int(float(parts[0]))
                        src_class = source_names.get(src_class_id, str(src_class_id))
                        target = mapping.get(src_class, src_class if src_class in class_to_id else None)
                        if target not in class_to_id:
                            continue
                        xc, yc, bw, bh = [float(v) for v in parts[1:]]
                        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                            continue
                        remapped_lines.append(f"{class_to_id[target]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                prefix = source_root.name.replace("-", "_")
                dst_image = output_dir / "images" / split / f"{prefix}_{image_path.name}"
                dst_label = output_dir / "labels" / split / f"{dst_image.stem}.txt"
                copy_or_link_image(image_path, dst_image, link=link_images)
                dst_label.parent.mkdir(parents=True, exist_ok=True)
                dst_label.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")
                stats["objects"] += len(remapped_lines)
                stats["images"] += 1

    write_dataset_yaml(output_dir, classes, filename="advtraffic26.yaml")
    return stats


def resolve_yolo_split_roots(source_root: Path, source_yaml: Path | None, split: str) -> tuple[Path, Path]:
    split_value = None
    if source_yaml and source_yaml.exists():
        import yaml

        with source_yaml.open("r", encoding="utf-8") as handle:
            split_value = (yaml.safe_load(handle) or {}).get(split)

    candidates = []
    if split_value:
        split_path = Path(split_value)
        candidates.append(split_path)
        if not split_path.is_absolute():
            candidates.append(source_root / split_path)
    candidates.extend([source_root / "images" / split, source_root / split / "images"])
    image_root = next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[-1].resolve())

    parts = list(image_root.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        label_root = Path(*parts)
    elif image_root.name == "images":
        label_root = image_root.parent / "labels"
    else:
        label_root = source_root / "labels" / split
    return image_root, label_root.resolve()


def load_yolo_names(yaml_path: str | Path) -> dict[int, str]:
    import yaml

    with Path(yaml_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return {idx: name for idx, name in enumerate(names)}
    return {int(idx): name for idx, name in names.items()}
