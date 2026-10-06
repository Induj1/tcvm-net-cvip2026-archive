"""Fail-fast quality gate for paper-ready experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_TABLES = [
    "baseline_models_auto.tex",
    "robustness_auto.tex",
    "adversarial_training_auto.tex",
    "real_video_auto.tex",
    "public_helmet_crossattack_frozen070_auto.tex",
    "crossattack_temporal_baselines_frozen_auto.tex",
    "ablation_auto.tex",
    "edge_auto.tex",
]
REQUIRED_FIGURES = [
    "outputs/figures/robustness_comparison.pdf",
    "outputs/figures/advtrain_comparison.pdf",
    "outputs/figures/attack_gallery.pdf",
    "outputs/figures/temporal_confidence_reflective_probe.pdf",
    "outputs/figures/temporal_confidence_real_video_reflective.pdf",
    "outputs/figures/ablation_chart.pdf",
    "outputs/figures/fps_robustness_tradeoff.pdf",
    "outputs/figures/gradcam_reflective_probe/gradcam_comparison.pdf",
    "outputs/figures/gradcam_reflective_probe/gradcam_clean.png",
    "outputs/figures/gradcam_reflective_probe/gradcam_attacked.png",
]
REQUIRED_SUPPLEMENTARY = [
    "supplementary/supplementary_material.tex",
    "supplementary/supplementary_material.pdf",
    "docs/reviewer_defense_matrix.md",
    "docs/physical_world_validation_protocol.md",
    "docs/submission_package_checklist.md",
]
FORBIDDEN_TEXT = ["placeholder", "to be populated", "Replace placeholders", " -- ", "& --"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate that paper outputs are real and non-placeholder.")
    parser.add_argument("--paper", default="paper/main.tex")
    parser.add_argument("--tables-dir", default="paper/tables")
    parser.add_argument("--robustness-csv", default="outputs/results_50_complete_v2/robustness_summary.csv")
    parser.add_argument(
        "--ablation-csv",
        default="outputs/results_50_complete/ablations_public_helmet_multiclip20/ablation_summary.csv",
    )
    return parser.parse_args()


def assert_csv_has_rows(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required result CSV is missing: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Result CSV is empty: {path}")


def main() -> None:
    args = parse_args()
    assert_csv_has_rows(args.robustness_csv)
    assert_csv_has_rows(args.ablation_csv)
    paper_text = Path(args.paper).read_text(encoding="utf-8")
    table_dir = Path(args.tables_dir)
    for table in REQUIRED_TABLES:
        path = table_dir / table
        if not path.exists():
            raise FileNotFoundError(f"Missing generated table: {path}")
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_TEXT:
            if forbidden.lower() in text.lower():
                raise ValueError(f"Placeholder-like text found in {path}: {forbidden}")
    for figure in REQUIRED_FIGURES:
        path = Path(figure)
        if not path.exists():
            raise FileNotFoundError(f"Missing generated figure: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Generated figure is empty: {path}")
    for artifact in REQUIRED_SUPPLEMENTARY:
        path = Path(artifact)
        if not path.exists():
            raise FileNotFoundError(f"Missing supplementary/reviewer artifact: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Supplementary/reviewer artifact is empty: {path}")
    for forbidden in ["resultplaceholder", "to be populated"]:
        if forbidden.lower() in paper_text.lower():
            raise ValueError(f"Placeholder-like text found in paper: {forbidden}")
    print("Publication output validation passed.")


if __name__ == "__main__":
    main()
