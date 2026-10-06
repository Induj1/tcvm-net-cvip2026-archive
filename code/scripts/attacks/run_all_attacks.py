"""Run all configured attacks for one split."""

from __future__ import annotations

import argparse
import subprocess
import sys


ATTACKS = ["fgsm", "pgd", "patch", "sticker", "reflective", "motion_blur", "occlusion", "low_light"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all TCVM-Net attack generators.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", default="outputs/attacks")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for attack in ATTACKS:
        cmd = [
            sys.executable,
            "scripts/attacks/run_attack.py",
            "--dataset-root",
            args.dataset_root,
            "--model",
            args.model,
            "--split",
            args.split,
            "--attack",
            attack,
            "--output-root",
            args.output_root,
        ]
        if args.max_images is not None:
            cmd.extend(["--max-images", str(args.max_images)])
        if args.device is not None:
            cmd.extend(["--device", args.device])
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
