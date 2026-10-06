"""Summarize the annotated public-video HELMET pilot for paper tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create public HELMET pilot summary table.")
    parser.add_argument("--clean-yolo", required=True)
    parser.add_argument("--attack-yolo", required=True)
    parser.add_argument("--attack-tcvm", required=True)
    parser.add_argument("--attack-name", default="Occlusion")
    parser.add_argument("--output-csv", default="outputs/results_50_complete/public_helmet_pilot_summary.csv")
    parser.add_argument("--output-tex", default="paper/tables/public_helmet_pilot_auto.tex")
    parser.add_argument("--caption", default="Annotated public-video HELMET pilot on real traffic frames.")
    parser.add_argument("--label", default="tab:public_helmet_pilot_auto")
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def row(name: str, data: dict, anomaly: dict | None = None) -> dict:
    latency = data.get("latency", {})
    fps = latency.get("fps", 0.0)
    if not fps and latency.get("mean_ms", 0.0):
        fps = 1000.0 / latency["mean_ms"]
    return {
        "Setting": name,
        "Frames": data.get("frames", 0),
        "mAP@50": data.get("map", {}).get("map50", 0.0),
        "mAP@50:95": data.get("map", {}).get("map50_95", 0.0),
        "Anom. F1": None if anomaly is None else anomaly.get("f1", 0.0),
        "FPR": None if anomaly is None else anomaly.get("false_positive_rate", anomaly.get("fpr", 0.0)),
        "FPS": fps,
    }


def write_latex(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    out = df.copy()
    for col in ["mAP@50", "mAP@50:95", "Anom. F1", "FPR", "FPS"]:
        out[col] = out[col].map(lambda value: "--" if pd.isna(value) else f"{float(value):.3f}")
    latex = out.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        position="ht",
    )
    latex = latex.replace("{lrlllll}", "{lrrrrrr}", 1)
    latex = latex.replace("\\begin{tabular}", "\\small\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}", 1)
    latex = latex.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")


def main() -> None:
    args = parse_args()
    clean = load_json(args.clean_yolo)
    attack = load_json(args.attack_yolo)
    tcvm = load_json(args.attack_tcvm)
    df = pd.DataFrame(
        [
            row("Clean public video (YOLOv8n)", clean),
            row(f"{args.attack_name} public video (YOLOv8n)", attack),
            row(f"{args.attack_name} public video (TCVM-Net)", tcvm, tcvm.get("anomaly", {})),
        ]
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    write_latex(df, Path(args.output_tex), args.caption, args.label)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
