"""Calibrate a TCVM anomaly threshold on disjoint video clips.

The protocol is intentionally simple and audit-friendly:

1. Run TCVM once per candidate threshold.
2. Split clips into calibration and held-out sets.
3. Select the threshold only from calibration clips.
4. Report the selected threshold on held-out clips without further tuning.

This avoids using the final evaluation clips to choose the temporal anomaly
threshold, which is important because TCVM recovery decisions are threshold
dependent.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


THRESHOLD_RE = re.compile(r"(?:thr|threshold)[_-]?([0-9]+(?:[._][0-9]+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    calibration_clip_count: int
    calibration_clips: list[int]
    heldout_clips: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clip-disjoint TCVM threshold calibration.")
    parser.add_argument(
        "--run-root",
        required=True,
        help="Directory containing one subdirectory per threshold with tcvm_frame_metrics.csv.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=None,
        help="Optional candidate thresholds. If omitted, thresholds are inferred from subdirectory names.",
    )
    parser.add_argument("--calibration-clip-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-calibration-fpr", type=float, default=0.25)
    parser.add_argument("--min-calibration-recall", type=float, default=0.70)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--output-csv", default="outputs/results_50_complete/tcvm_calibration_protocol.csv")
    parser.add_argument("--output-json", default="outputs/results_50_complete/tcvm_calibration_protocol.json")
    parser.add_argument("--output-tex", default=None)
    return parser.parse_args()


def infer_threshold(path: Path) -> float | None:
    for candidate in [path.name, path.parent.name]:
        match = THRESHOLD_RE.search(candidate)
        if match:
            return float(match.group(1).replace("_", "."))
    summary_path = path / "tcvm_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            anomaly = summary.get("anomaly", {})
            if "threshold" in anomaly:
                return float(anomaly["threshold"])
        except json.JSONDecodeError:
            return None
    return None


def find_runs(run_root: Path, thresholds: list[float] | None) -> dict[float, Path]:
    candidates = sorted(run_root.glob("*/tcvm_frame_metrics.csv"))
    runs: dict[float, Path] = {}
    for frame_metrics in candidates:
        threshold = infer_threshold(frame_metrics.parent)
        if threshold is None:
            continue
        if thresholds is not None and not any(math.isclose(threshold, target, abs_tol=1e-9) for target in thresholds):
            continue
        runs[threshold] = frame_metrics
    if not runs:
        raise FileNotFoundError(f"No TCVM runs found under {run_root}")
    missing = []
    for threshold in thresholds or []:
        if not any(math.isclose(threshold, found, abs_tol=1e-9) for found in runs):
            missing.append(threshold)
    if missing:
        raise FileNotFoundError(f"Missing threshold runs: {missing}")
    return dict(sorted(runs.items()))


def binary_metrics(df: pd.DataFrame, threshold: float) -> dict[str, float]:
    y_true = df["adversarial_label"].astype(int).to_numpy()
    scores = df["max_anomaly_score"].astype(float).to_numpy()
    y_pred = (scores >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    accuracy = (tp + tn) / max(len(y_true), 1)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "frames": int(len(df)),
        "positive_frames": int(y_true.sum()),
    }


def make_split(df: pd.DataFrame, calibration_clip_count: int, seed: int) -> SplitConfig:
    if "clip_id" not in df.columns:
        raise ValueError("Frame metrics must include clip_id for clip-disjoint calibration.")
    clips = sorted(int(clip) for clip in df["clip_id"].dropna().unique())
    if calibration_clip_count <= 0 or calibration_clip_count >= len(clips):
        raise ValueError("Calibration clip count must be between 1 and total_clips - 1.")
    rng = np.random.default_rng(seed)
    shuffled = np.array(clips, dtype=int)
    rng.shuffle(shuffled)
    calibration = sorted(int(clip) for clip in shuffled[:calibration_clip_count])
    heldout = sorted(int(clip) for clip in shuffled[calibration_clip_count:])
    return SplitConfig(
        seed=seed,
        calibration_clip_count=calibration_clip_count,
        calibration_clips=calibration,
        heldout_clips=heldout,
    )


def split_df(df: pd.DataFrame, clips: list[int]) -> pd.DataFrame:
    return df[df["clip_id"].astype(int).isin(clips)].copy()


def clip_bootstrap_ci(df: pd.DataFrame, threshold: float, samples: int, seed: int) -> tuple[float, float]:
    clips = sorted(int(clip) for clip in df["clip_id"].dropna().unique())
    if len(clips) < 2:
        return (float("nan"), float("nan"))
    per_clip_counts = []
    for clip in clips:
        clip_df = df[df["clip_id"].astype(int) == clip]
        y_true = clip_df["adversarial_label"].astype(int).to_numpy()
        y_pred = (clip_df["max_anomaly_score"].astype(float).to_numpy() >= threshold).astype(int)
        per_clip_counts.append(
            np.array(
                [
                    ((y_true == 1) & (y_pred == 1)).sum(),
                    ((y_true == 0) & (y_pred == 1)).sum(),
                    ((y_true == 0) & (y_pred == 0)).sum(),
                    ((y_true == 1) & (y_pred == 0)).sum(),
                ],
                dtype=float,
            )
        )
    counts = np.vstack(per_clip_counts)
    rng = np.random.default_rng(seed)
    f1_values = []
    for _ in range(samples):
        sampled_idx = rng.integers(0, len(clips), size=len(clips))
        tp, fp, _tn, fn = counts[sampled_idx].sum(axis=0)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1_values.append(2 * precision * recall / max(precision + recall, 1e-12))
    return tuple(np.percentile(f1_values, [2.5, 97.5]).tolist())


def select_threshold(rows: list[dict], max_fpr: float, min_recall: float) -> dict:
    calibration_rows = [row for row in rows if row["split"] == "calibration"]
    feasible = [
        row
        for row in calibration_rows
        if row["fpr"] <= max_fpr and row["recall"] >= min_recall
    ]
    if feasible:
        return sorted(feasible, key=lambda row: (-row["f1"], row["fpr"], -row["threshold"]))[0]
    return sorted(
        calibration_rows,
        key=lambda row: (-(row["f1"] - max(0.0, row["fpr"] - max_fpr)), row["fpr"], -row["threshold"]),
    )[0]


def write_latex(selected: dict, heldout: dict, path: Path) -> None:
    out = pd.DataFrame(
        [
            {
                "Split": "Calibration",
                "Clips": selected["clips"],
                "Threshold": selected["threshold"],
                "F1": selected["f1"],
                "FPR": selected["fpr"],
                "Recall": selected["recall"],
                "Precision": selected["precision"],
            },
            {
                "Split": "Held-out",
                "Clips": heldout["clips"],
                "Threshold": heldout["threshold"],
                "F1": heldout["f1"],
                "FPR": heldout["fpr"],
                "Recall": heldout["recall"],
                "Precision": heldout["precision"],
            },
        ]
    )
    for col in ["Threshold", "F1", "FPR", "Recall", "Precision"]:
        out[col] = out[col].map(lambda value: f"{float(value):.3f}")
    latex = out.to_latex(
        index=False,
        escape=False,
        caption="Clip-disjoint TCVM threshold calibration on the public HELMET temporal benchmark.",
        label="tab:tcvm_calibration_protocol",
        position="t",
    )
    latex = latex.replace("\\begin{tabular}", "\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    runs = find_runs(run_root, args.thresholds)

    first_df = pd.read_csv(next(iter(runs.values())))
    split = make_split(first_df, args.calibration_clip_count, args.seed)
    rows = []

    for threshold, frame_metrics in runs.items():
        df = pd.read_csv(frame_metrics)
        for split_name, clips in [
            ("calibration", split.calibration_clips),
            ("heldout", split.heldout_clips),
            ("all", split.calibration_clips + split.heldout_clips),
        ]:
            subset = split_df(df, clips)
            metrics = binary_metrics(subset, threshold)
            ci_low, ci_high = clip_bootstrap_ci(
                subset,
                threshold,
                args.bootstrap_samples,
                args.seed + int(round(threshold * 1000)) + (0 if split_name == "calibration" else 10000),
            )
            rows.append(
                {
                    "threshold": threshold,
                    "split": split_name,
                    "clips": len(clips),
                    **metrics,
                    "f1_ci_low": ci_low,
                    "f1_ci_high": ci_high,
                    "frame_metrics": str(frame_metrics),
                }
            )

    selected = select_threshold(rows, args.max_calibration_fpr, args.min_calibration_recall)
    heldout = next(
        row
        for row in rows
        if row["split"] == "heldout" and math.isclose(row["threshold"], selected["threshold"], abs_tol=1e-9)
    )
    all_selected = next(
        row
        for row in rows
        if row["split"] == "all" and math.isclose(row["threshold"], selected["threshold"], abs_tol=1e-9)
    )

    result = {
        "protocol": "clip-disjoint threshold calibration",
        "selection_rule": {
            "primary": "maximize calibration F1",
            "constraints": {
                "max_calibration_fpr": args.max_calibration_fpr,
                "min_calibration_recall": args.min_calibration_recall,
            },
            "tie_break": "lower FPR, then higher threshold",
        },
        "split": asdict(split),
        "selected_threshold": selected["threshold"],
        "selected_calibration_metrics": selected,
        "heldout_metrics": heldout,
        "all_clip_metrics_at_selected_threshold": all_selected,
    }

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.output_tex:
        write_latex(selected, heldout, Path(args.output_tex))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
