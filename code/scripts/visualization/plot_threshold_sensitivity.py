"""Plot TCVM threshold sensitivity for publication/supplementary material."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot anomaly threshold sensitivity from CSV output.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="outputs/figures/tcvm_threshold_sensitivity.png")
    parser.add_argument("--title", default="TCVM threshold sensitivity")
    return parser.parse_args()


def apply_lncs_style() -> None:
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


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    apply_lncs_style()

    fig, ax = plt.subplots(figsize=(6.9, 3.9))
    x = df["Threshold"].to_numpy()
    ax.plot(x, df["F1"], color="#4C78A8", marker="o", linewidth=2.0, label="F1")
    if {"F1 CI Low", "F1 CI High"}.issubset(df.columns):
        ax.fill_between(x, df["F1 CI Low"], df["F1 CI High"], color="#4C78A8", alpha=0.16, label="95% clip bootstrap CI")
    ax.plot(x, df["Recall"], color="#59A14F", marker="s", linewidth=1.7, label="Recall")
    ax.plot(x, df["FPR"], color="#E15759", marker="^", linewidth=1.7, label="FPR")
    best_idx = df["F1"].idxmax()
    ax.axvline(df.loc[best_idx, "Threshold"], color="#555555", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.annotate(
        f"best F1={df.loc[best_idx, 'F1']:.3f}",
        xy=(df.loc[best_idx, "Threshold"], df.loc[best_idx, "F1"]),
        xytext=(6, -18),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "linewidth": 0.6, "color": "#555555"},
    )
    ax.set_xlabel("Anomaly threshold")
    ax.set_ylabel("Frame-level score")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(args.title, fontsize=11, pad=8)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="#D0D0D0", ncol=2, loc="upper right")
    ax.grid(alpha=0.22)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
