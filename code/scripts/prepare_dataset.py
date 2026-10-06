"""Prepare the AdvTraffic-26 dataset structure and sampled frames."""

from __future__ import annotations

import argparse
from pathlib import Path

from advtraffic.datasets import create_advtraffic_structure, extract_video_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AdvTraffic-26 from raw traffic surveillance videos.")
    parser.add_argument("--dataset-root", default="data/AdvTraffic-26", help="Output dataset root.")
    parser.add_argument("--raw-video-root", default=None, help="Folder containing raw videos to sample.")
    parser.add_argument("--stride", type=int, default=5, help="Sample every Nth frame.")
    parser.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    create_advtraffic_structure(dataset_root)
    if args.raw_video_root:
        metadata = extract_video_frames(
            raw_video_root=args.raw_video_root,
            dataset_root=dataset_root,
            stride=args.stride,
            resize=tuple(args.resize) if args.resize else None,
            max_frames_per_video=args.max_frames_per_video,
        )
        print(f"Wrote frame metadata: {metadata}")
    print(f"Dataset scaffold ready: {dataset_root.resolve()}")


if __name__ == "__main__":
    main()
