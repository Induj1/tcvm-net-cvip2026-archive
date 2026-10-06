"""Summarize TCVM optical-flow speed/accuracy tradeoffs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create TCVM speed/robustness mode table.")
    parser.add_argument("--full-json", required=True)
    parser.add_argument("--scaled-jsons", nargs="*", default=None)
    parser.add_argument("--fast-json", required=True)
    parser.add_argument("--output-csv", default="outputs/results_50_complete/tcvm_speed_modes.csv")
    parser.add_argument("--output-tex", default="paper/tables/tcvm_speed_modes_auto.tex")
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def row(name: str, data: dict) -> dict:
    anomaly = data.get("anomaly", {})
    latency = data.get("latency", {})
    metrics = data.get("map", {})
    return {
        "Mode": name,
        "Flow Scale": f"{float(data.get('flow_scale', 1.0)):.2f}" if data.get("optical_flow", True) else "none",
        "mAP@50": metrics.get("map50", 0.0),
        "mAP@50:95": metrics.get("map50_95", 0.0),
        "Anom. F1": anomaly.get("f1", 0.0),
        "FPR": anomaly.get("fpr", 0.0),
        "FPS": latency.get("fps", 0.0),
    }


def write_latex(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in ["mAP@50", "mAP@50:95", "Anom. F1", "FPR", "FPS"]:
        out[col] = out[col].map(lambda value: f"{float(value):.3f}")
    latex = out.to_latex(
        index=False,
        escape=False,
        caption="TCVM optical-flow speed and anomaly-detection tradeoff on the public HELMET 10-clip benchmark.",
        label="tab:tcvm_speed_modes_auto",
        position="ht",
    )
    latex = latex.replace("{lllllll}", "{llrrrrr}", 1)
    latex = latex.replace(
        "\\begin{tabular}",
        "\\small\n\\setlength{\\tabcolsep}{3.5pt}\n\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}",
        1,
    )
    latex = latex.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = pd.DataFrame(
        [
            row("Full TCVM", load(args.full_json)),
            *[
                row(f"Scaled-flow TCVM ({load(path).get('flow_scale', 1.0):.2f})", load(path))
                for path in (args.scaled_jsons or [])
            ],
            row("Fast TCVM", load(args.fast_json)),
        ]
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    write_latex(df, Path(args.output_tex))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
