"""Measure TCVM threshold sensitivity across repeated clip-disjoint splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .calibrate_tcvm_threshold import (
        binary_metrics,
        find_runs,
        make_split,
        select_threshold,
        split_df,
    )
except ImportError:  # Direct script execution.
    from calibrate_tcvm_threshold import (
        binary_metrics,
        find_runs,
        make_split,
        select_threshold,
        split_df,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeat clip-disjoint TCVM calibration over fixed seeds.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--thresholds", type=float, nargs="*", default=None)
    parser.add_argument("--calibration-clip-count", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=2026)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--max-calibration-fpr", type=float, default=0.10)
    parser.add_argument("--min-calibration-recall", type=float, default=0.70)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tex", default=None)
    return parser.parse_args()


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def evaluate_seed(
    runs: dict[float, Path],
    calibration_clip_count: int,
    seed: int,
    max_fpr: float,
    min_recall: float,
) -> dict:
    first_df = pd.read_csv(next(iter(runs.values())))
    split = make_split(first_df, calibration_clip_count, seed)
    calibration_rows = []
    frames_by_threshold: dict[float, pd.DataFrame] = {}

    for threshold, frame_metrics in runs.items():
        df = pd.read_csv(frame_metrics)
        frames_by_threshold[threshold] = df
        metrics = binary_metrics(split_df(df, split.calibration_clips), threshold)
        calibration_rows.append(
            {
                "threshold": threshold,
                "split": "calibration",
                "clips": len(split.calibration_clips),
                **metrics,
            }
        )

    selected = select_threshold(calibration_rows, max_fpr, min_recall)
    selected_threshold = float(selected["threshold"])
    selected_df = frames_by_threshold[selected_threshold]
    heldout = binary_metrics(split_df(selected_df, split.heldout_clips), selected_threshold)
    return {
        "seed": seed,
        "selected_threshold": selected_threshold,
        "calibration_clips": " ".join(map(str, split.calibration_clips)),
        "heldout_clips": " ".join(map(str, split.heldout_clips)),
        **{f"calibration_{key}": value for key, value in selected.items() if key not in {"split", "threshold", "clips"}},
        **{f"heldout_{key}": value for key, value in heldout.items()},
    }


def write_latex(summary: dict, path: Path) -> None:
    labels = [
        ("Selected $\\theta$", "selected_threshold"),
        ("Held-out F1", "heldout_f1"),
        ("Held-out recall", "heldout_recall"),
        ("Held-out FPR", "heldout_fpr"),
    ]
    lines = [
        "\\begin{table}[tbp]",
        "\\caption{Repeated clip-disjoint calibration sensitivity over 20 fixed splits.}",
        "\\label{tab:repeated_split_sensitivity}",
        "\\small",
        "\\centering",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Metric & Median & IQR & Range \\\\",
        "\\midrule",
    ]
    for label, key in labels:
        stats = summary[key]
        lines.append(
            f"{label} & {stats['median']:.3f} & "
            f"[{stats['q1']:.3f}, {stats['q3']:.3f}] & "
            f"[{stats['min']:.3f}, {stats['max']:.3f}] \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    seeds = args.seeds or list(range(args.seed_start, args.seed_start + args.num_seeds))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")

    runs = find_runs(Path(args.run_root), args.thresholds)
    rows = [
        evaluate_seed(
            runs,
            args.calibration_clip_count,
            seed,
            args.max_calibration_fpr,
            args.min_calibration_recall,
        )
        for seed in seeds
    ]
    frame = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)

    metrics = ["selected_threshold", "heldout_f1", "heldout_recall", "heldout_fpr"]
    summary = {metric: describe(frame[metric].astype(float).tolist()) for metric in metrics}
    threshold_counts = {
        f"{float(threshold):.2f}": int(count)
        for threshold, count in frame["selected_threshold"].value_counts().sort_index().items()
    }
    result = {
        "protocol": "repeated clip-disjoint threshold calibration",
        "seeds": seeds,
        "calibration_clip_count": args.calibration_clip_count,
        "heldout_clip_count": int(frame["heldout_clips"].iloc[0].count(" ") + 1),
        "selection_constraints": {
            "max_calibration_fpr": args.max_calibration_fpr,
            "min_calibration_recall": args.min_calibration_recall,
        },
        "threshold_counts": threshold_counts,
        "summary": summary,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.output_tex:
        write_latex(summary, Path(args.output_tex))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
