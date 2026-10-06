"""Create a compact Grad-CAM comparison figure for the LNCS paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a two-panel Grad-CAM comparison.")
    parser.add_argument("--clean", required=True)
    parser.add_argument("--attacked", required=True)
    parser.add_argument("--output", default="outputs/figures/gradcam_reflective_probe/gradcam_comparison.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panels = [
        ("Clean frame", Image.open(args.clean).convert("RGB")),
        ("Reflective attack frame", Image.open(args.attacked).convert("RGB")),
    ]
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.9))
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title, pad=4)
        ax.axis("off")
    fig.tight_layout(pad=0.3, w_pad=0.4)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
