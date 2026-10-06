"""Create a compact before/after attack gallery from visualization panels."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from advtraffic.utils.io import IMAGE_EXTENSIONS, iter_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create attack gallery figure.")
    parser.add_argument("--attacks-root", default="outputs/attacks")
    parser.add_argument("--split", default="test")
    parser.add_argument("--attacks", nargs="+", default=["fgsm", "pgd", "sticker", "reflective", "occlusion", "motion_blur"])
    parser.add_argument("--output", default="outputs/figures/attack_gallery.png")
    parser.add_argument("--samples-per-attack", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panels = []
    labels = []
    for attack in args.attacks:
        vis_root = Path(args.attacks_root) / attack / args.split / "visualizations"
        for path in list(iter_files(vis_root, IMAGE_EXTENSIONS))[: args.samples_per_attack]:
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            panels.append(image)
            labels.append(attack)
    if not panels:
        raise FileNotFoundError("No attack visualization panels found.")

    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    columns = 2 if len(panels) > 1 else 1
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(8.2, 2.45 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    pretty = {
        "fgsm": "FGSM",
        "pgd": "PGD",
        "patch": "Patch",
        "sticker": "Sticker",
        "reflective": "Reflective",
        "occlusion": "Occlusion",
        "motion_blur": "Motion blur",
        "low_light": "Low light",
    }
    for ax, image, label in zip(axes, panels, labels):
        ax.imshow(image)
        ax.set_title(pretty.get(label, label), loc="left", fontweight="bold", pad=4)
        ax.axis("off")
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.4)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
