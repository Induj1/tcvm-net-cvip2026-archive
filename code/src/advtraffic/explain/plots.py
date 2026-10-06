"""Plotting helpers for temporal trajectories and robustness reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_confidence_trajectory(csv_path: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(df["frame_id"], df["confidence"], label="YOLO confidence", linewidth=2)
    if "anomaly_score" in df:
        ax.plot(df["frame_id"], df["anomaly_score"], label="TCVM anomaly", linewidth=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_robustness_bars(csv_path: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    attacks = df["attack"].astype(str)
    width = 0.35
    x = range(len(attacks))
    ax.bar([i - width / 2 for i in x], df["baseline_map50"], width=width, label="YOLOv8")
    ax.bar([i + width / 2 for i in x], df["tcvm_map50"], width=width, label="YOLOv8+TCVM")
    ax.set_xticks(list(x))
    ax.set_xticklabels(attacks, rotation=25, ha="right")
    ax.set_ylabel("mAP@50")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
