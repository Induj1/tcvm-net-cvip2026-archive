"""Evaluate simple previous-frame consistency baselines from saved frame logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute reference-frame consistency baselines.")
    parser.add_argument("--frame-labels", required=True, help="CSV with frame_id, clip_id, local_frame, adversarial_label.")
    parser.add_argument("--yolo-frame-metrics", required=True, help="YOLO frame_metrics.csv.")
    parser.add_argument("--tcvm-frame-metrics", required=True, help="TCVM tcvm_frame_metrics.csv.")
    parser.add_argument("--output-dir", default="outputs/results_50_complete/reference_consistency_baseline")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.62, 0.70])
    parser.add_argument("--tcvm-threshold", type=float, default=0.45)
    return parser.parse_args()


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y = y_true.to_numpy(dtype=int)
    pred = y_pred.to_numpy(dtype=int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / len(y) if len(y) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "accuracy": accuracy,
    }


def add_reference_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    sort_cols = ["clip_id", "local_frame"] if "local_frame" in df.columns else ["clip_id", "frame_id"]
    for _, group in df.sort_values(sort_cols).groupby("clip_id"):
        prev_count: float | None = None
        prev_conf: float | None = None
        for row in group.to_dict(orient="records"):
            curr_count = float(row["detections"])
            curr_conf = float(row["mean_confidence"])
            if prev_count is None:
                count_score = 0.0
                confidence_score = 0.0
            else:
                count_score = max(0.0, (prev_count - curr_count) / max(prev_count, 1.0))
                confidence_score = max(0.0, (prev_conf - curr_conf) / max(prev_conf, 1e-6))
            row["count_collapse_score"] = count_score
            row["confidence_drop_score"] = confidence_score
            rows.append(row)
            prev_count = curr_count
            prev_conf = curr_conf
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.frame_labels)
    yolo = pd.read_csv(args.yolo_frame_metrics)
    tcvm = pd.read_csv(args.tcvm_frame_metrics)
    df = labels.merge(yolo[["frame_id", "detections", "mean_confidence", "max_confidence"]], on="frame_id", how="left")
    df = df.merge(tcvm[["frame_id", "max_anomaly_score", "recovered", "missing_events"]], on="frame_id", how="left")
    scored = add_reference_scores(df)

    rows = []
    definitions = [
        ("Previous-frame count collapse", "count_collapse_score", args.thresholds),
        ("Previous-frame confidence drop", "confidence_drop_score", args.thresholds),
        ("TCVM-Net", "max_anomaly_score", [args.tcvm_threshold]),
    ]
    for method, score_col, thresholds in definitions:
        for threshold in thresholds:
            pred = scored[score_col] >= threshold
            rows.append({"method": method, "threshold": threshold, **binary_metrics(scored["adversarial_label"], pred)})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_dir / "frame_scores.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "reference_consistency_summary.csv", index=False)
    best = summary.sort_values(["method", "f1"], ascending=[True, False]).groupby("method").head(1)
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
