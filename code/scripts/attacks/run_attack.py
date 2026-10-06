"""Batch-generate adversarial attacks and compute attack success rate."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from advtraffic.attacks import apply_low_light, apply_motion_blur, apply_named_physical_attack, fgsm_attack, pgd_attack
from advtraffic.attacks.toolbox_adapters import foolbox_linf_pgd, make_art_classifier
from advtraffic.detection import Detection, YOLOv8Engine
from advtraffic.utils.geometry import box_iou, yolo_to_xyxy
from advtraffic.utils.io import IMAGE_EXTENSIONS, ensure_dir, iter_files, read_image, write_json
from advtraffic.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adversarial samples and attack metrics.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--attack",
        required=True,
        choices=["fgsm", "pgd", "patch", "sticker", "reflective", "motion_blur", "occlusion", "low_light"],
    )
    parser.add_argument("--model", required=True, help="YOLOv8 checkpoint used for gradient attacks and ASR.")
    parser.add_argument("--output-root", default="outputs/attacks")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--gradient-backend", choices=["native", "art", "foolbox"], default="native")
    parser.add_argument("--target-classes", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def read_labels(label_path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    if not label_path.exists():
        return []
    labels = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        row = np.asarray([float(v) for v in parts], dtype=float)
        labels.append((int(row[0]), yolo_to_xyxy(row, width, height)))
    return labels


def target_detected(detections: list[Detection], target_class: int, target_box: np.ndarray, iou_threshold: float) -> bool:
    for det in detections:
        if det.class_id == target_class and box_iou(det.xyxy, target_box) >= iou_threshold:
            return True
    return False


def apply_attack(args: argparse.Namespace, engine: YOLOv8Engine, image: np.ndarray, target_boxes: list[np.ndarray]) -> np.ndarray:
    if args.attack == "fgsm":
        if args.gradient_backend == "art":
            return art_fgsm(engine.raw_torch_model(), image, args.eps, args.imgsz, args.target_classes, args.device)
        return fgsm_attack(
            engine.raw_torch_model(),
            image,
            eps=args.eps,
            image_size=args.imgsz,
            target_classes=args.target_classes,
            device=args.device,
        )
    if args.attack == "pgd":
        if args.gradient_backend == "foolbox":
            return foolbox_pgd(engine.raw_torch_model(), image, args.eps, args.pgd_steps, args.imgsz, args.target_classes, args.device)
        return pgd_attack(
            engine.raw_torch_model(),
            image,
            eps=args.eps,
            steps=args.pgd_steps,
            image_size=args.imgsz,
            target_classes=args.target_classes,
            device=args.device,
        )
    if args.attack == "motion_blur":
        return apply_motion_blur(image)
    if args.attack == "low_light":
        return apply_low_light(image)

    attacked = image.copy()
    for box in target_boxes:
        attacked = apply_named_physical_attack(attacked, box, args.attack)
    return attacked


def art_fgsm(
    model: torch.nn.Module,
    image: np.ndarray,
    eps: float,
    imgsz: int,
    target_classes: list[int],
    device: str | None,
) -> np.ndarray:
    classifier = make_art_classifier(model, input_shape=(3, imgsz, imgsz), target_classes=target_classes, device=device)
    resized = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(rgb, (2, 0, 1))[None]
    y = np.zeros((1,), dtype=np.int64)
    from art.attacks.evasion import FastGradientMethod

    attack = FastGradientMethod(estimator=classifier, eps=eps)
    adv = attack.generate(x=x, y=y)[0]
    bgr = cv2.cvtColor((np.transpose(adv, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)


def foolbox_pgd(
    model: torch.nn.Module,
    image: np.ndarray,
    eps: float,
    steps: int,
    imgsz: int,
    target_classes: list[int],
    device: str | None,
) -> np.ndarray:
    resized = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(np.transpose(rgb, (2, 0, 1))[None])
    adv = foolbox_linf_pgd(model, x, eps=eps, steps=steps, target_classes=target_classes, device=device)[0]
    adv_rgb = (adv.permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(adv_rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)


def draw_panel(clean: np.ndarray, adv: np.ndarray, clean_dets: list[Detection], adv_dets: list[Detection]) -> np.ndarray:
    left = draw_detections(clean, clean_dets, title="clean")
    right = draw_detections(adv, adv_dets, title="attacked")
    return np.concatenate([left, right], axis=1)


def draw_detections(image: np.ndarray, detections: list[Detection], title: str) -> np.ndarray:
    out = image.copy()
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 1, cv2.LINE_AA)
    for det in detections:
        x1, y1, x2, y2 = det.xyxy.astype(int)
        color = (0, 200, 80)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{det.class_name} {det.confidence:.2f}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    image_root = dataset_root / "images" / args.split
    label_root = dataset_root / "labels" / args.split
    output_root = Path(args.output_root) / args.attack / args.split
    out_images = ensure_dir(output_root / "images")
    out_labels = ensure_dir(output_root / "labels")
    out_vis = ensure_dir(output_root / "visualizations")
    engine = YOLOv8Engine(model_path=args.model, imgsz=args.imgsz, conf=args.conf, device=args.device, extract_features=False)
    set_seed(args.seed)

    image_paths = list(iter_files(image_root, IMAGE_EXTENSIONS))
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    rows = []
    clean_detected_total = 0
    successful_attacks = 0
    target_total = 0
    for image_path in tqdm(image_paths, desc=f"{args.attack}:{args.split}"):
        image = read_image(image_path)
        height, width = image.shape[:2]
        labels = read_labels(label_root / f"{image_path.stem}.txt", width, height)
        source_label = label_root / f"{image_path.stem}.txt"
        targets = [(cls, box) for cls, box in labels if cls in set(args.target_classes)]
        target_boxes = [box for _, box in targets]

        clean_dets = engine.detect(image, frame_id=0)
        adv = apply_attack(args, engine, image, target_boxes)
        adv_dets = engine.detect(adv, frame_id=0)
        cv2.imwrite(str(out_images / image_path.name), adv)
        if source_label.exists():
            shutil.copy2(source_label, out_labels / source_label.name)
        else:
            (out_labels / f"{image_path.stem}.txt").touch()

        per_image_targets = 0
        per_image_success = 0
        for cls, box in targets:
            clean_hit = target_detected(clean_dets, cls, box, args.iou_threshold)
            adv_hit = target_detected(adv_dets, cls, box, args.iou_threshold)
            if clean_hit:
                clean_detected_total += 1
                per_image_targets += 1
                if not adv_hit:
                    successful_attacks += 1
                    per_image_success += 1
            target_total += 1

        if args.save_visualizations:
            panel = draw_panel(image, adv, clean_dets, adv_dets)
            cv2.imwrite(str(out_vis / image_path.name), panel)
        rows.append(
            {
                "image": str(image_path),
                "targets": len(targets),
                "clean_detected_targets": per_image_targets,
                "successful_attacks": per_image_success,
                "output_image": str(out_images / image_path.name),
            }
        )

    asr = successful_attacks / max(clean_detected_total, 1)
    summary = {
        "attack": args.attack,
        "split": args.split,
        "dataset_root": str(dataset_root),
        "model": args.model,
        "images": len(image_paths),
        "target_objects": target_total,
        "clean_detected_targets": clean_detected_total,
        "successful_attacks": successful_attacks,
        "attack_success_rate": asr,
        "epsilon": args.eps,
        "gradient_backend": args.gradient_backend,
        "seed": args.seed,
    }
    with (output_root / "per_image_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_root / "attack_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
