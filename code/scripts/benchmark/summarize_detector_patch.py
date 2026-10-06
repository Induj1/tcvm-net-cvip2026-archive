"""Summarize detector-specific patch pilot metrics for supplementary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create YOLOv8 detector-specific patch summary table.")
    parser.add_argument("--clean-yolo", required=True)
    parser.add_argument("--patch-yolo", required=True)
    parser.add_argument("--patch-tcvm", default=None)
    parser.add_argument("--patch-summary", required=True)
    parser.add_argument("--output-csv", default="outputs/results_50_complete/yolov8_patch_public_summary.csv")
    parser.add_argument("--output-tex", default="paper/tables/yolov8_patch_public_auto.tex")
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fps(metrics: dict) -> float:
    latency = metrics.get("latency", {})
    if latency.get("fps"):
        return float(latency["fps"])
    if latency.get("mean_ms"):
        return 1000.0 / float(latency["mean_ms"])
    return 0.0


def write_latex(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in ["mAP@50", "mAP@50:95", "ASR", "Anom. R", "Anom. F1", "FPS"]:
        out[col] = out[col].map(lambda value: "--" if pd.isna(value) else f"{float(value):.3f}")
    latex = out.to_latex(
        index=False,
        escape=False,
        caption="Detector-specific YOLOv8 patch pilot on the public HELMET multi-clip sequence.",
        label="tab:yolov8_patch_public_auto",
        position="ht",
    )
    latex = latex.replace("{lrllllll}", "{lrrrrrrr}", 1)
    latex = latex.replace("\\begin{tabular}", "\\small\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}", 1)
    latex = latex.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")


def main() -> None:
    args = parse_args()
    clean = load_json(args.clean_yolo)
    patch = load_json(args.patch_yolo)
    attack = load_json(args.patch_summary)
    rows = [
        {
            "Setting": "Clean public video (YOLOv8n)",
            "Frames": clean.get("frames", 0),
            "mAP@50": clean.get("map", {}).get("map50", 0.0),
            "mAP@50:95": clean.get("map", {}).get("map50_95", 0.0),
            "ASR": None,
            "Anom. R": None,
            "Anom. F1": None,
            "FPS": fps(clean),
        },
        {
            "Setting": "YOLOv8-specific printable patch",
            "Frames": patch.get("frames", 0),
            "mAP@50": patch.get("map", {}).get("map50", 0.0),
            "mAP@50:95": patch.get("map", {}).get("map50_95", 0.0),
            "ASR": attack.get("attack_success_rate", 0.0),
            "Anom. R": None,
            "Anom. F1": None,
            "FPS": fps(patch),
        },
    ]
    if args.patch_tcvm:
        tcvm = load_json(args.patch_tcvm)
        rows.append(
            {
                "Setting": "YOLOv8-specific patch + TCVM",
                "Frames": tcvm.get("frames", 0),
                "mAP@50": tcvm.get("map", {}).get("map50", 0.0),
                "mAP@50:95": tcvm.get("map", {}).get("map50_95", 0.0),
                "ASR": None,
                "Anom. R": tcvm.get("anomaly", {}).get("recall", 0.0),
                "Anom. F1": tcvm.get("anomaly", {}).get("f1", 0.0),
                "FPS": fps(tcvm),
            }
        )
    df = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    write_latex(df, Path(args.output_tex))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
