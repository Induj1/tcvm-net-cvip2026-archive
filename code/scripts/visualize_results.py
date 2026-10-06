"""Generate publication figures from result CSV files."""

from __future__ import annotations

import argparse

from advtraffic.explain.plots import plot_confidence_trajectory, plot_robustness_bars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready plots.")
    parser.add_argument("--confidence-csv", default=None)
    parser.add_argument("--robustness-csv", default=None)
    parser.add_argument("--confidence-output", default="paper/figures/confidence_trajectory.png")
    parser.add_argument("--robustness-output", default="paper/figures/robustness_bars.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.confidence_csv:
        plot_confidence_trajectory(args.confidence_csv, args.confidence_output)
        print(f"Wrote {args.confidence_output}")
    if args.robustness_csv:
        plot_robustness_bars(args.robustness_csv, args.robustness_output)
        print(f"Wrote {args.robustness_output}")


if __name__ == "__main__":
    main()
