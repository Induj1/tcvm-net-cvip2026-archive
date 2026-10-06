"""Generate Grad-CAM overlays before and after attack."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from advtraffic.detection import YOLOv8Engine
from advtraffic.explain import YOLOGradCAM
from advtraffic.utils.io import read_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate YOLOv8 Grad-CAM overlays.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--clean-image", required=True)
    parser.add_argument("--attacked-image", default=None)
    parser.add_argument("--output-dir", default="outputs/figures/gradcam")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--target-classes", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def save_overlay(cam: YOLOGradCAM, image_path: str, output_path: Path, imgsz: int, target_classes: list[int]) -> None:
    image = read_image(image_path)
    heatmap = cam.generate(image, image_size=imgsz, target_classes=target_classes)
    overlay = cam.overlay(image, heatmap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def main() -> None:
    args = parse_args()
    engine = YOLOv8Engine(model_path=args.model, imgsz=args.imgsz, device=args.device, extract_features=False)
    cam = YOLOGradCAM(engine.raw_torch_model(), device=args.device)
    output_dir = Path(args.output_dir)
    save_overlay(cam, args.clean_image, output_dir / "gradcam_clean.png", args.imgsz, args.target_classes)
    if args.attacked_image:
        save_overlay(cam, args.attacked_image, output_dir / "gradcam_attacked.png", args.imgsz, args.target_classes)
    print(f"Wrote Grad-CAM overlays to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
