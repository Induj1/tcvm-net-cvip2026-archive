"""Compare frozen TCVM, tracker-only, and reference-frame temporal baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .calibrate_tcvm_threshold import binary_metrics
except ImportError:  # Direct script execution.
    from calibrate_tcvm_threshold import binary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize held-out cross-attack temporal baselines.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--tracker-root", required=True)
    parser.add_argument("--adav-root", required=True)
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["occlusion", "reflective", "detector_patch", "sticker", "motion_blur"],
    )
    parser.add_argument("--attack-root-override", action="append", default=[], metavar="ATTACK=PATH")
    parser.add_argument("--tcvm-run-name", default="tcvm_frozen_thr070_flow025")
    parser.add_argument("--tcvm-threshold", type=float, default=0.70)
    parser.add_argument("--tracker-threshold", type=float, default=0.70)
    parser.add_argument("--adav-threshold", type=float, default=0.85)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tex", required=True)
    return parser.parse_args()


def parse_overrides(values: list[str]) -> dict[str, Path]:
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid attack override: {value}")
        attack, path = value.split("=", 1)
        overrides[attack] = Path(path)
    return overrides


def heldout_metrics(path: Path, score_column: str, threshold: float, heldout_clips: list[int]) -> dict[str, float]:
    frame = pd.read_csv(path)
    required = {"clip_id", "adversarial_label", score_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    subset = frame[frame["clip_id"].astype(int).isin(heldout_clips)].copy()
    if len(subset) != len(heldout_clips) * 24:
        raise ValueError(f"Expected {len(heldout_clips) * 24} held-out frames in {path}, found {len(subset)}")
    normalized = subset.rename(columns={score_column: "max_anomaly_score"})
    return binary_metrics(normalized, threshold)


def write_latex(frame: pd.DataFrame, path: Path) -> None:
    labels = {
        "occlusion": "Occlusion",
        "reflective": "Reflective",
        "detector_patch": "Detector patch",
        "sticker": "Sticker",
        "motion_blur": "Motion blur",
    }
    lines = [
        "\\begin{table}[tbp]",
        "\\caption{Frozen temporal-baseline transfer on the 20 held-out HELMET clips. "
        "Entries are anomaly F1/FPR; no attack-specific threshold tuning is used.}",
        "\\label{tab:crossattack_temporal_baselines}",
        "\\scriptsize",
        "\\centering",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Attack & TCVM-Net & Track recovery & Reference recovery \\\\",
        "\\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{labels.get(row.attack, row.attack)} & "
            f"{row.tcvm_f1:.3f}/{row.tcvm_fpr:.3f} & "
            f"{row.tracker_f1:.3f}/{row.tracker_fpr:.3f} & "
            f"{row.adav_f1:.3f}/{row.adav_fpr:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    protocol = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    heldout_clips = [int(clip) for clip in protocol["split"]["heldout_clips"]]
    overrides = parse_overrides(args.attack_root_override)
    rows = []

    for attack in args.attacks:
        attack_root = overrides.get(attack, Path(args.run_root) / attack)
        tracker_path = Path(args.tracker_root) / attack / "track_recovery_only_frame_metrics.csv"
        adav_path = Path(args.adav_root) / attack / "adav_frame_metrics.csv"
        tcvm_path = attack_root / args.tcvm_run_name / "tcvm_frame_metrics.csv"
        tcvm = heldout_metrics(tcvm_path, "max_anomaly_score", args.tcvm_threshold, heldout_clips)
        tracker = heldout_metrics(tracker_path, "max_anomaly_score", args.tracker_threshold, heldout_clips)
        adav = heldout_metrics(adav_path, "max_anomaly_score", args.adav_threshold, heldout_clips)
        rows.append(
            {
                "attack": attack,
                **{f"tcvm_{key}": value for key, value in tcvm.items()},
                **{f"tracker_{key}": value for key, value in tracker.items()},
                **{f"adav_{key}": value for key, value in adav.items()},
            }
        )

    frame = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    result = {
        "protocol": "frozen temporal baselines on held-out clips",
        "heldout_clips": heldout_clips,
        "thresholds": {
            "tcvm": args.tcvm_threshold,
            "tracker_recovery": args.tracker_threshold,
            "reference_recovery": args.adav_threshold,
        },
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_latex(frame, Path(args.output_tex))
    print(frame[["attack", "tcvm_f1", "tracker_f1", "adav_f1", "tcvm_fpr", "tracker_fpr", "adav_fpr"]].to_string(index=False))


if __name__ == "__main__":
    main()
