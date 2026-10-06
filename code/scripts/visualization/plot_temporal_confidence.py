"""Create publication-quality temporal confidence/anomaly plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TCVM temporal confidence and anomaly trajectories.")
    parser.add_argument("--csv", required=True, help="CSV from benchmark_tcvm.py or custom frame metrics.")
    parser.add_argument("--output", default="outputs/figures/temporal_confidence.png")
    parser.add_argument("--title", default=None)
    parser.add_argument("--confidence-column", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    frame = df["frame_id"]
    if args.confidence_column is not None:
        confidence_col = args.confidence_column
    elif "mean_raw_confidence" in df.columns:
        confidence_col = "mean_raw_confidence"
    elif "mean_confidence" in df.columns:
        confidence_col = "mean_confidence"
    else:
        confidence_col = "confidence"
    anomaly_col = "max_anomaly_score" if "max_anomaly_score" in df.columns else "anomaly_score"

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(frame, df[confidence_col], color="#4C78A8", linewidth=2.0, label="Detector confidence")
    if anomaly_col in df.columns:
        ax.plot(frame, df[anomaly_col], color="#E15759", linewidth=2.0, label="TCVM anomaly score")
        spike_idx = df[anomaly_col].idxmax()
        ax.scatter(
            [df.loc[spike_idx, "frame_id"]],
            [df.loc[spike_idx, anomaly_col]],
            color="#E15759",
            edgecolor="#333333",
            linewidth=0.4,
            s=38,
            zorder=4,
        )
    if "adversarial_label" in df.columns:
        attacked = df["adversarial_label"].astype(int).to_numpy()
        frame_values = frame.to_numpy()
        span_label = "Attack window"
        start = None
        previous = None
        for idx, is_attacked in enumerate(attacked):
            if is_attacked and start is None:
                start = frame_values[idx]
            if start is not None:
                last_row = idx == len(attacked) - 1
                next_is_clean = (not last_row) and attacked[idx + 1] == 0
                next_is_new_clip = (not last_row) and "clip_id" in df.columns and df.loc[idx + 1, "clip_id"] != df.loc[idx, "clip_id"]
                if last_row or next_is_clean or next_is_new_clip:
                    previous = frame_values[idx]
                    ax.axvspan(
                        start,
                        previous,
                        color="#E15759",
                        alpha=0.10,
                        label=span_label,
                    )
                    span_label = None
                    start = None
    if "clip_id" in df.columns:
        boundary_rows = df.index[df["clip_id"].ne(df["clip_id"].shift())].tolist()
        for boundary in boundary_rows[1:]:
            ax.axvline(df.loc[boundary, "frame_id"], color="#8C8C8C", linewidth=0.6, alpha=0.35)
    if "recovered" in df.columns:
        recovered = df[df["recovered"] > 0]
        if not recovered.empty:
            ax.scatter(
                recovered["frame_id"],
                recovered[confidence_col],
                marker="D",
                color="#F2A900",
                edgecolor="#333333",
                linewidth=0.35,
                s=28,
                label="Recovered",
            )
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.05)
    if args.title:
        ax.set_title(args.title, fontsize=12, pad=10)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="#D0D0D0", ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    ax.grid(alpha=0.22)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
