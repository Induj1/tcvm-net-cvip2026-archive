"""Benchmark clean, attacked, and image-only YOLOv8 robustness with mAP and ASR."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

from advtraffic.utils.io import IMAGE_EXTENSIONS, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 robustness benchmark over generated attack splits.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--clean-data", required=True, help="Clean YOLO dataset YAML.")
    parser.add_argument("--attacks-root", default="outputs/attacks")
    parser.add_argument("--split", default="test")
    parser.add_argument("--attacks", nargs="+", default=["fgsm", "pgd", "patch", "sticker", "reflective", "motion_blur", "occlusion", "low_light"])
    parser.add_argument("--output-dir", default="outputs/results")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--method-name", default="YOLOv8", help="Method label written to the result CSV.")
    parser.add_argument(
        "--match-clean-subset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate clean mAP on the exact filenames present in the attack splits.",
    )
    return parser.parse_args()


def load_names(clean_yaml: str | Path):
    data = yaml.safe_load(Path(clean_yaml).read_text(encoding="utf-8"))
    return data.get("names", {})


def make_eval_yaml(root: Path, names, split_name: str) -> Path:
    config = {"path": str(root.resolve()), "train": "images", "val": "images", "test": "images", "names": names}
    path = root / f"{split_name}_eval.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _resolve_clean_split(clean_yaml: str | Path, split: str) -> tuple[dict, list[Path]]:
    yaml_path = Path(clean_yaml).resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    dataset_root = Path(config.get("path", yaml_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = (yaml_path.parent / dataset_root).resolve()

    entries = config.get(split)
    if entries is None:
        raise KeyError(f"Split '{split}' is missing from {yaml_path}")
    if not isinstance(entries, list):
        entries = [entries]

    images: list[Path] = []
    for entry in entries:
        source = Path(entry)
        if not source.is_absolute():
            source = dataset_root / source
        if source.is_dir():
            images.extend(
                path.absolute()
                for path in sorted(source.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif source.suffix.lower() == ".txt":
            for line in source.read_text(encoding="utf-8").splitlines():
                image_path = Path(line.strip())
                if not image_path.is_absolute():
                    image_path = source.parent / image_path
                images.append(image_path.absolute())
        elif source.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(source.absolute())
    return config, images


def make_matched_clean_yaml(
    clean_yaml: str | Path,
    attacks_root: str | Path,
    attacks: list[str],
    split: str,
    output_dir: Path,
) -> Path:
    reference_names: set[str] | None = None
    for attack in attacks:
        image_root = Path(attacks_root) / attack / split / "images"
        if not image_root.exists():
            continue
        names = {
            path.name
            for path in image_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise ValueError(f"Attack split '{attack}' does not contain the same paired image set.")

    if not reference_names:
        raise FileNotFoundError("No attack images were found to define the matched clean subset.")

    config, clean_images = _resolve_clean_split(clean_yaml, split)
    clean_by_name = {path.name: path for path in clean_images}
    if len(clean_by_name) != len(clean_images):
        raise ValueError("Clean split contains duplicate image basenames; paired matching is ambiguous.")
    missing = sorted(reference_names - clean_by_name.keys())
    if missing:
        raise FileNotFoundError(f"Clean split is missing {len(missing)} attacked images, including {missing[0]}.")

    list_path = output_dir / "clean_matched_images.txt"
    selected = [clean_by_name[name] for name in sorted(reference_names)]
    list_path.write_text("\n".join(path.as_posix() for path in selected) + "\n", encoding="utf-8")
    config[split] = str(list_path.resolve())
    matched_yaml = output_dir / "clean_matched_eval.yaml"
    matched_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return matched_yaml


def val_model(
    model: YOLO,
    data_yaml: str | Path,
    imgsz: int,
    batch: int,
    device: str | None,
    project: Path,
    name: str,
    split: str,
) -> dict[str, float]:
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
        plots=True,
    )
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    names = load_names(args.clean_data)
    rows = []

    clean_data = args.clean_data
    if args.match_clean_subset:
        clean_data = make_matched_clean_yaml(
            args.clean_data,
            args.attacks_root,
            args.attacks,
            args.split,
            output_dir,
        )
    clean = val_model(model, clean_data, args.imgsz, args.batch, args.device, output_dir, "clean_yolo", args.split)
    rows.append({"attack": "clean", "method": args.method_name, "attack_success_rate": 0.0, "robust_accuracy": 1.0, **clean})

    for attack in args.attacks:
        attack_root = Path(args.attacks_root) / attack / args.split
        if not (attack_root / "images").exists():
            continue
        eval_yaml = make_eval_yaml(attack_root, names, split_name=attack)
        metrics = val_model(model, eval_yaml, args.imgsz, args.batch, args.device, output_dir, f"{attack}_yolo", args.split)
        summary_path = attack_root / "attack_summary.json"
        asr = read_json(summary_path).get("attack_success_rate", None) if summary_path.exists() else None
        rows.append(
            {
                "attack": attack,
                "method": args.method_name,
                "attack_success_rate": asr,
                "robust_accuracy": None if asr is None else 1.0 - asr,
                **metrics,
            }
        )

        image_def_root = Path(args.attacks_root) / f"{attack}_image_defense" / args.split
        if (image_def_root / "images").exists():
            image_def_yaml = make_eval_yaml(image_def_root, names, split_name=f"{attack}_image_defense")
            image_def_metrics = val_model(
                model,
                image_def_yaml,
                args.imgsz,
                args.batch,
                args.device,
                output_dir,
                f"{attack}_image_defense",
                args.split,
            )
            rows.append({"attack": attack, "method": "Image-only defense", "attack_success_rate": asr, "robust_accuracy": None if asr is None else 1.0 - asr, **image_def_metrics})

    df = pd.DataFrame(rows)
    csv_path = output_dir / "robustness_summary.csv"
    json_path = output_dir / "robustness_summary.json"
    df.to_csv(csv_path, index=False)
    write_json(json_path, {"rows": rows})
    print(df.to_string(index=False))
    print(f"Wrote {csv_path.resolve()}")


if __name__ == "__main__":
    main()
