"""Summarize frozen-threshold cross-attack TCVM evaluation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from calibrate_tcvm_threshold import binary_metrics, clip_bootstrap_ci, split_df


ATTACK_NAMES = {
    "occlusion": "Occlusion",
    "reflective": "Reflective",
    "sticker": "Sticker",
    "patch": "Patch",
    "detector_patch": "Detector patch",
    "motion_blur": "Motion blur",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize frozen-threshold public HELMET attack transfer.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--clean-metrics", required=True)
    parser.add_argument("--attacks", nargs="+", default=list(ATTACK_NAMES))
    parser.add_argument(
        "--attack-root-override",
        action="append",
        default=[],
        metavar="ATTACK=PATH",
        help="Override the run directory for an attack.",
    )
    parser.add_argument(
        "--detector-patch-metrics",
        default=None,
        help="Optional per-image detector-patch metrics used to report held-out ASR.",
    )
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--tcvm-run-name", default="tcvm_frozen_thr070_flow025")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tex", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_latex(df: pd.DataFrame, path: Path) -> None:
    display = df[
        [
            "Attack",
            "YOLO mAP@50",
            "TCVM mAP@50",
            "$\\Delta$ mAP",
            "Held-out Rec.",
            "Held-out F1",
            "Held-out FPR",
        ]
    ]
    latex = display.to_latex(
        index=False,
        escape=False,
        float_format=lambda value: f"{value:.3f}",
        caption=(
            "Frozen-threshold cross-attack evaluation on 20 held-out HELMET clips. "
            "$\\theta=0.70$ is selected only on disjoint occlusion calibration clips; "
            "mAP uses all 720 frames and anomaly metrics use the 480 held-out frames."
        ),
        label="tab:public_helmet_crossattack_auto",
        position="t",
    )
    latex = latex.replace(
        "\\begin{tabular}",
        "\\small\n\\setlength{\\tabcolsep}{3.2pt}\n"
        "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}",
        1,
    )
    latex = latex.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    protocol = read_json(Path(args.calibration_json))
    heldout_clips = [int(value) for value in protocol["split"]["heldout_clips"]]
    clean = read_json(Path(args.clean_metrics))
    clean_map50 = float(clean["map"]["map50"])
    root_overrides = {}
    for item in args.attack_root_override:
        attack, path = item.split("=", 1)
        root_overrides[attack] = Path(path)

    detector_patch_heldout_asr = None
    if args.detector_patch_metrics:
        patch_df = pd.read_csv(args.detector_patch_metrics)
        patch_df["clip_id"] = patch_df["image"].map(
            lambda value: int(re.search(r"clip(\d+)_", Path(value).stem).group(1))
        )
        patch_heldout = patch_df[patch_df["clip_id"].isin(heldout_clips)]
        detector_patch_heldout_asr = float(
            patch_heldout["successful_attacks"].sum()
            / max(patch_heldout["clean_detected_targets"].sum(), 1)
        )

    rows = []
    for attack in args.attacks:
        attack_root = root_overrides.get(attack, run_root / attack)
        yolo = read_json(attack_root / "yolo_baseline" / "metrics.json")
        tcvm_root = attack_root / args.tcvm_run_name
        tcvm = read_json(tcvm_root / "tcvm_summary.json")
        frame_df = pd.read_csv(tcvm_root / "tcvm_frame_metrics.csv")
        heldout_df = split_df(frame_df, heldout_clips)
        heldout = binary_metrics(heldout_df, args.threshold)
        ci_low, ci_high = clip_bootstrap_ci(
            heldout_df,
            args.threshold,
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
        yolo_map50 = float(yolo["map"]["map50"])
        tcvm_map50 = float(tcvm["map"]["map50"])
        rows.append(
            {
                "attack": attack,
                "Attack": ATTACK_NAMES.get(attack, attack),
                "frames": int(tcvm["frames"]),
                "clean_map50": clean_map50,
                "YOLO mAP@50": yolo_map50,
                "attack_map_drop": yolo_map50 - clean_map50,
                "TCVM mAP@50": tcvm_map50,
                "$\\Delta$ mAP": tcvm_map50 - yolo_map50,
                "Held-out Prec.": heldout["precision"],
                "Held-out Rec.": heldout["recall"],
                "Held-out F1": heldout["f1"],
                "Held-out FPR": heldout["fpr"],
                "f1_ci_low": ci_low,
                "f1_ci_high": ci_high,
                "fps": float(tcvm["latency"]["fps"]),
                "threshold": args.threshold,
                "heldout_clips": len(heldout_clips),
                "heldout_frames": int(heldout["frames"]),
                "heldout_attack_success_rate": (
                    detector_patch_heldout_asr if attack == "detector_patch" else None
                ),
            }
        )

    result = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    output_json.write_text(
        json.dumps(
            {
                "protocol": "cross-attack transfer with frozen occlusion-calibrated threshold",
                "threshold": args.threshold,
                "calibration_source": args.calibration_json,
                "heldout_clips": heldout_clips,
                "clean_map50": clean_map50,
                "rows": result.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_latex(result, Path(args.output_tex))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
