"""Bootstrap uncertainty estimates for image-level attack success rates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap attack success-rate confidence intervals.")
    parser.add_argument("--attack-root", required=True, help="Directory containing <attack>/<split>/per_image_metrics.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-csv", default="outputs/results_50_complete/attack_asr_bootstrap_ci.csv")
    parser.add_argument("--output-tex", default="paper/tables/attack_ci_auto.tex")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def bootstrap_asr(df: pd.DataFrame, samples: int, rng: np.random.Generator) -> dict[str, float]:
    clean = df["clean_detected_targets"].to_numpy(dtype=float)
    success = df["successful_attacks"].to_numpy(dtype=float)
    n = len(df)
    if n == 0 or clean.sum() <= 0:
        return {"asr": 0.0, "ci_low": 0.0, "ci_high": 0.0, "images": n, "clean_targets": 0}

    values = []
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        denom = clean[idx].sum()
        values.append(0.0 if denom <= 0 else success[idx].sum() / denom)
    arr = np.asarray(values)
    return {
        "asr": float(success.sum() / clean.sum()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "images": int(n),
        "clean_targets": int(clean.sum()),
    }


def write_latex(df: pd.DataFrame, path: Path) -> None:
    pretty = df.copy()
    pretty["ASR 95\\% CI"] = pretty.apply(
        lambda row: f"{row['asr']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
        axis=1,
    )
    pretty = pretty.rename(columns={"attack": "Attack", "images": "Images", "clean_targets": "Clean targets"})
    pretty = pretty[["Attack", "Images", "Clean targets", "ASR 95\\% CI"]]
    latex = pretty.to_latex(
        index=False,
        escape=False,
        caption="Bootstrap 95\\% confidence intervals for image-level attack success rate.",
        label="tab:attack_ci_auto",
        position="ht",
    )
    latex = latex.replace("\\begin{tabular}", "\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    root = Path(args.attack_root)
    for metrics_path in sorted(root.glob(f"*/{args.split}/per_image_metrics.csv")):
        attack = metrics_path.parents[1].name
        df = pd.read_csv(metrics_path)
        rows.append({"attack": attack, **bootstrap_asr(df, args.samples, rng)})

    out = pd.DataFrame(rows)
    attack_order = {
        name: i
        for i, name in enumerate(["fgsm", "pgd", "patch", "sticker", "reflective", "motion_blur", "occlusion", "low_light"])
    }
    attack_names = {
        "fgsm": "FGSM",
        "pgd": "PGD",
        "patch": "Patch",
        "sticker": "Sticker",
        "reflective": "Reflective",
        "motion_blur": "Motion blur",
        "occlusion": "Occlusion",
        "low_light": "Low light",
    }
    out = out.sort_values("attack", key=lambda col: col.map(attack_order).fillna(99))
    out["attack"] = out["attack"].map(attack_names).fillna(out["attack"])

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    write_latex(out, Path(args.output_tex))
    print(out.to_string(index=False))
    print(f"Wrote {output_csv.resolve()}")


if __name__ == "__main__":
    main()
