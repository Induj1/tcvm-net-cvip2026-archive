"""Create a compact clean/attack gallery for the public HELMET pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create public HELMET pilot gallery.")
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--attack-root", required=True)
    parser.add_argument("--attack-name", default="Attack")
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 4, 5, 7])
    parser.add_argument("--output", default="outputs/figures/public_helmet_pilot_gallery.pdf")
    return parser.parse_args()


def read_rgb(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def frame_path(root: Path, frame: int) -> Path:
    exact = root / f"frame_{frame:04d}.jpg"
    if exact.exists():
        return exact
    matches = sorted(root.glob(f"*frame_{frame:04d}.jpg"))
    if matches:
        return matches[0]
    raise FileNotFoundError(exact)


def main() -> None:
    args = parse_args()
    clean_root = Path(args.clean_root) / "images"
    attack_root = Path(args.attack_root) / "images"
    fig, axes = plt.subplots(2, len(args.frames), figsize=(2.2 * len(args.frames), 4.0), constrained_layout=True)
    for col, frame in enumerate(args.frames):
        for row, (root, title) in enumerate(((clean_root, "Clean"), (attack_root, args.attack_name))):
            ax = axes[row, col] if len(args.frames) > 1 else axes[row]
            ax.imshow(read_rgb(frame_path(root, frame)))
            ax.set_title(f"{title} F{frame}", fontsize=8)
            ax.set_axis_off()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    png_output = output.with_suffix(".png")
    fig.savefig(png_output, dpi=300, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
