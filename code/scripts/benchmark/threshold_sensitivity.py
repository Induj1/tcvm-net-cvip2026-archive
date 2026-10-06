"""Compute frame-level TCVM threshold sensitivity from saved frame metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TCVM anomaly threshold sensitivity.")
    parser.add_argument("--frame-metrics", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.40, 0.45, 0.50, 0.62])
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-csv", default="outputs/results_50_complete/tcvm_threshold_sensitivity.csv")
    parser.add_argument("--output-tex", default="paper/tables/tcvm_threshold_sensitivity_auto.tex")
    return parser.parse_args()


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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
    }


def clip_bootstrap_ci(df: pd.DataFrame, threshold: float, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    if "clip_id" not in df.columns or df["clip_id"].nunique() < 2:
        return (float("nan"), float("nan"))
    clips = sorted(df["clip_id"].unique().tolist())
    per_clip_counts = []
    for clip in clips:
        clip_df = df[df["clip_id"] == clip]
        y_true = clip_df["adversarial_label"].astype(int).to_numpy()
        y_pred = (clip_df["max_anomaly_score"].to_numpy() >= threshold).astype(int)
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
    f1_values = []
    for _ in range(samples):
        sampled_idx = rng.integers(0, len(clips), size=len(clips))
        tp, fp, _tn, fn = counts[sampled_idx].sum(axis=0)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1_values.append(2 * precision * recall / max(precision + recall, 1e-12))
    return tuple(np.percentile(f1_values, [2.5, 97.5]).tolist())


def write_latex(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in ["Threshold", "Accuracy", "Precision", "Recall", "F1", "FPR", "F1 CI Low", "F1 CI High"]:
        out[col] = out[col].map(lambda value: "--" if pd.isna(value) else f"{float(value):.3f}")
    latex = out.to_latex(
        index=False,
        escape=False,
        caption="TCVM anomaly-threshold sensitivity on the public HELMET multi-clip benchmark.",
        label="tab:tcvm_threshold_sensitivity_auto",
        position="ht",
    )
    latex = latex.replace("{llllllll}", "{lrrrrrrr}", 1)
    latex = latex.replace("\\begin{tabular}", "\\small\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}", 1)
    latex = latex.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.frame_metrics)
    if "adversarial_label" not in df.columns or "max_anomaly_score" not in df.columns:
        raise ValueError("Frame metrics must contain adversarial_label and max_anomaly_score columns.")
    rng = np.random.default_rng(args.seed)
    rows = []
    y_true = df["adversarial_label"].astype(int).to_numpy()
    scores = df["max_anomaly_score"].to_numpy()
    for threshold in args.thresholds:
        metrics = binary_metrics(y_true, (scores >= threshold).astype(int))
        ci_low, ci_high = clip_bootstrap_ci(df, threshold, args.bootstrap_samples, rng)
        rows.append(
            {
                "Threshold": threshold,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "FPR": metrics["fpr"],
                "F1 CI Low": ci_low,
                "F1 CI High": ci_high,
            }
        )
    out = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    write_latex(out, Path(args.output_tex))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
