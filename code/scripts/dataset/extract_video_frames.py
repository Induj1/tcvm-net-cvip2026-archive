"""Extract frames from traffic videos into a YOLO-ready image split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from advtraffic.utils.io import VIDEO_EXTENSIONS, ensure_dir, iter_files
from advtraffic.utils.video import iter_video_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract video frames for attack-ready sequence evaluation.")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-root", default="data/custom-video-yolo")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    image_root = ensure_dir(output_root / "images" / args.split)
    label_root = ensure_dir(output_root / "labels" / args.split)
    metadata_path = ensure_dir(output_root / "metadata") / "video_frames.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "frame_id", "timestamp_ms", "image", "label"])
        writer.writeheader()
        for video_path in iter_files(args.video_root, VIDEO_EXTENSIONS):
            for frame in iter_video_frames(video_path, stride=args.stride, max_frames=args.max_frames_per_video):
                name = f"{video_path.stem}_f{frame.frame_id:06d}.jpg"
                image_path = image_root / name
                label_path = label_root / f"{image_path.stem}.txt"
                cv2.imwrite(str(image_path), frame.image)
                label_path.touch(exist_ok=True)
                writer.writerow(
                    {
                        "video": str(video_path),
                        "frame_id": frame.frame_id,
                        "timestamp_ms": frame.timestamp_ms,
                        "image": str(image_path),
                        "label": str(label_path),
                    }
                )
    print(f"Wrote frame metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
