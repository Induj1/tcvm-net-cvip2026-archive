"""Create robustness, FPS, and ablation charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot robustness and ablation summaries.")
    parser.add_argument("--robustness-csv", default=None)
    parser.add_argument("--advtrain-csv", default=None)
    parser.add_argument("--ablation-csv", default=None)
    parser.add_argument("--edge-json", default=None)
    parser.add_argument("--tcvm-baseline-json", default=None)
    parser.add_argument("--tcvm-summary-json", default=None)
    parser.add_argument("--output-dir", default="outputs/figures")
    return parser.parse_args()


def apply_lNCS_style() -> None:
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


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")


def plot_robustness(df: pd.DataFrame, output_dir: Path) -> None:
    attacked = df[df["attack"] != "clean"].copy()
    if attacked.empty:
        return
    apply_lNCS_style()
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    pivot = attacked.pivot_table(index="attack", columns="method", values="map50", aggfunc="mean")
    colors = ["#4C78A8", "#59A14F", "#E15759"][: len(pivot.columns)]
    pivot.plot(kind="bar", ax=ax, width=0.74, color=colors, edgecolor="#333333", linewidth=0.35)
    ax.set_ylabel("mAP@50")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#D0D0D0", loc="upper right")
    ax.grid(axis="y", alpha=0.24)
    save_figure(fig, output_dir, "robustness_comparison")

    if "attack_success_rate" in attacked:
        fig, ax = plt.subplots(figsize=(8.2, 3.7))
        asr = attacked[attacked["method"] == "YOLOv8"].dropna(subset=["attack_success_rate"])
        ax.bar(asr["attack"], asr["attack_success_rate"], color="#E15759", edgecolor="#333333", linewidth=0.35)
        ax.set_ylabel("Attack success rate")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.24)
        save_figure(fig, output_dir, "attack_success_rate")


def plot_ablation(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    apply_lNCS_style()
    fig, ax1 = plt.subplots(figsize=(8.2, 4.0))
    x = range(len(df))
    ax1.bar(x, df["map50"], color="#4C78A8", edgecolor="#333333", linewidth=0.35, label="mAP@50")
    ax1.set_ylabel("mAP@50")
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["variant"], rotation=25, ha="right")
    ax2 = ax1.twinx()
    if "anomaly_fpr" in df:
        ax2.plot(x, df["anomaly_fpr"], color="#E15759", marker="o", linewidth=1.7, label="FPR")
        ax2.set_ylabel("False positive rate")
        ax2.set_ylim(0, 1.0)
    ax1.grid(axis="y", alpha=0.22)
    save_figure(fig, output_dir, "ablation_chart")


def plot_advtrain_comparison(clean_df: pd.DataFrame, adv_df: pd.DataFrame, output_dir: Path) -> None:
    base = clean_df[(clean_df["attack"] != "clean") & (clean_df["method"] == "YOLOv8")].copy()
    adv = adv_df[(adv_df["attack"] != "clean") & (adv_df["method"] == "AdvAug-YOLOv8n")].copy()
    if base.empty or adv.empty:
        return
    attack_order = ["fgsm", "pgd", "patch", "sticker", "reflective", "motion_blur", "occlusion", "low_light"]
    pretty = {
        "fgsm": "FGSM",
        "pgd": "PGD",
        "patch": "Patch",
        "sticker": "Sticker",
        "reflective": "Reflective",
        "motion_blur": "Motion blur",
        "occlusion": "Occlusion",
        "low_light": "Low light",
    }
    merged = base.merge(adv, on="attack", suffixes=("_base", "_adv"))
    merged["attack"] = pd.Categorical(merged["attack"], categories=attack_order, ordered=True)
    merged = merged.sort_values("attack")

    apply_lNCS_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    x = range(len(merged))
    width = 0.38
    ax.bar([i - width / 2 for i in x], merged["map50_base"], width=width, color="#4C78A8", edgecolor="#333333", linewidth=0.35, label="YOLOv8n")
    ax.bar([i + width / 2 for i in x], merged["map50_adv"], width=width, color="#59A14F", edgecolor="#333333", linewidth=0.35, label="AdvAug-YOLOv8n")
    ax.set_ylabel("mAP@50")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels([pretty.get(str(a), str(a)) for a in merged["attack"]], rotation=25, ha="right")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#D0D0D0", loc="upper left")
    ax.grid(axis="y", alpha=0.24)
    save_figure(fig, output_dir, "advtrain_comparison")


def plot_speed_robustness(edge_json: Path, yolo_json: Path, tcvm_json: Path, output_dir: Path) -> None:
    edge = json.loads(edge_json.read_text())
    yolo = json.loads(yolo_json.read_text())
    tcvm = json.loads(tcvm_json.read_text())
    points = pd.DataFrame(
        [
            {
                "method": "YOLOv8n",
                "fps": float(edge["yolov8"]["fps"]),
                "map50": float(yolo["metrics"]["map50"]),
            },
            {
                "method": "YOLOv8n+TCVM-Net",
                "fps": float(edge["tcvm"]["fps"]),
                "map50": float(tcvm["map"]["map50"]),
            },
        ]
    )
    apply_lNCS_style()
    fig, ax = plt.subplots(figsize=(5.1, 3.5))
    colors = {"YOLOv8n": "#4C78A8", "YOLOv8n+TCVM-Net": "#59A14F"}
    markers = {"YOLOv8n": "o", "YOLOv8n+TCVM-Net": "s"}
    for _, row in points.iterrows():
        ax.scatter(
            row["fps"],
            row["map50"],
            s=72,
            color=colors[row["method"]],
            marker=markers[row["method"]],
            edgecolor="#333333",
            linewidth=0.5,
            label=row["method"],
            zorder=3,
        )
        ax.annotate(
            row["method"],
            (row["fps"], row["map50"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )
    ax.plot(points["fps"], points["map50"], color="#777777", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_xlabel("Throughput (FPS)")
    ax.set_ylabel("Reflective probe mAP@50")
    ax.set_xlim(0, max(points["fps"]) + 1.2)
    ax.set_ylim(0.78, 0.96)
    ax.grid(alpha=0.22)
    save_figure(fig, output_dir, "fps_robustness_tradeoff")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    robustness_df = None
    if args.robustness_csv:
        robustness_df = pd.read_csv(args.robustness_csv)
        plot_robustness(robustness_df, output_dir)
    if args.advtrain_csv and robustness_df is not None:
        plot_advtrain_comparison(robustness_df, pd.read_csv(args.advtrain_csv), output_dir)
    if args.ablation_csv:
        plot_ablation(pd.read_csv(args.ablation_csv), output_dir)
    if args.edge_json and args.tcvm_baseline_json and args.tcvm_summary_json:
        plot_speed_robustness(
            Path(args.edge_json),
            Path(args.tcvm_baseline_json),
            Path(args.tcvm_summary_json),
            output_dir,
        )
    print(f"Wrote figures to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
