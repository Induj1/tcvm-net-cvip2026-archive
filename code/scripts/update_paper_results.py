"""Convert result CSV files into LaTeX tables used by the LNCS draft."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from experiment CSV files.")
    parser.add_argument("--robustness-csv", required=True)
    parser.add_argument("--ablation-csv", required=False)
    parser.add_argument("--edge-json", required=False)
    parser.add_argument("--edge-csv", required=False)
    parser.add_argument(
        "--advtrain-robustness-csv",
        required=False,
        help="Optional robustness CSV from an adversarial-augmentation baseline.",
    )
    parser.add_argument(
        "--baseline-jsons",
        nargs="*",
        default=None,
        help="Optional validation/test metric JSON files for detector baseline comparison.",
    )
    parser.add_argument("--output-dir", default="paper/tables")
    return parser.parse_args()


def dataframe_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    resize: bool = False,
    position: str = "t",
) -> str:
    latex = df.to_latex(
        index=False,
        escape=False,
        float_format=lambda x: f"{x:.3f}",
        caption=caption,
        label=label,
        position=position,
    )
    latex = latex.replace("\\begin{tabular}", "\\small\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}")
    if resize:
        latex = latex.replace(
            "\\begin{tabular}",
            "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}",
            1,
        )
        latex = latex.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
    return latex


def pretty_robustness(df: pd.DataFrame) -> pd.DataFrame:
    attack_names = {
        "clean": "Clean",
        "fgsm": "FGSM",
        "pgd": "PGD",
        "patch": "Patch",
        "sticker": "Sticker",
        "reflective": "Reflective",
        "motion_blur": "Motion blur",
        "occlusion": "Occlusion",
        "low_light": "Low light",
    }
    method_names = {"Image-only defense": "Image-only", "YOLOv8": "YOLOv8", "AdvAug-YOLOv8n": "AdvAug-YOLOv8n"}
    out = df.copy()
    out["attack"] = out["attack"].map(attack_names).fillna(out["attack"])
    out["method"] = out["method"].map(method_names).fillna(out["method"])
    out = out.rename(
        columns={
            "attack": "Attack",
            "method": "Method",
            "attack_success_rate": "Base ASR",
            "robust_accuracy": "Base RAcc.",
            "map50": "mAP@50",
            "map50_95": "mAP@50:95",
            "precision": "Precision",
            "recall": "Recall",
        }
    )
    return out[["Attack", "Method", "Base ASR", "Base RAcc.", "mAP@50", "mAP@50:95"]]


def pretty_ablation(df: pd.DataFrame) -> pd.DataFrame:
    variant_names = {
        "full_tcvm": "Full TCVM",
        "track_recovery_only": "Track recovery",
        "no_confidence": "No confidence",
        "no_motion": "No motion",
        "no_feature": "No feature",
        "no_temporal_smoothing": "No smoothing",
        "no_recovery": "No recovery",
    }
    out = df.copy()
    out["variant"] = out["variant"].map(variant_names).fillna(out["variant"])
    out = out.rename(
        columns={
            "variant": "Variant",
            "frames": "Frames",
            "map50": "mAP@50",
            "map50_95": "mAP@50:95",
            "fps": "FPS",
            "mean_latency_ms": "Latency (ms)",
            "anomaly_accuracy": "Anom. Acc.",
            "anomaly_precision": "Anom. Prec.",
            "anomaly_recall": "Anom. Rec.",
            "anomaly_f1": "Anom. F1",
            "anomaly_fpr": "FPR",
        }
    )
    return out[["Variant", "mAP@50", "mAP@50:95", "Anom. F1", "FPR", "FPS"]]


def pretty_baselines(metric_paths: list[str]) -> pd.DataFrame:
    import json

    model_names = {
        "yolov8n_50_complete": "YOLOv8n",
        "yolov8n": "YOLOv8n",
        "yolov8s": "YOLOv8s",
        "yolov8n_advtrain300": "AdvAug-YOLOv8n",
    }
    rows = []
    for metric_path in metric_paths:
        data = json.loads(Path(metric_path).read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        speed = data.get("speed_ms", {})
        model = data.get("model", Path(metric_path).parent.name)
        rows.append(
            {
                "Model": model_names.get(model, model),
                "Epochs": data.get("epochs_completed", data.get("epochs", "")),
                "mAP@50": metrics.get("map50", 0.0),
                "mAP@50:95": metrics.get("map50_95", 0.0),
                "Precision": metrics.get("precision", 0.0),
                "Recall": metrics.get("recall", 0.0),
                "Inference (ms)": speed.get("inference", 0.0),
            }
        )
    return pd.DataFrame(rows)


def pretty_advtrain_comparison(clean_csv: str, advtrain_csv: str) -> pd.DataFrame:
    clean = pd.read_csv(clean_csv)
    adv = pd.read_csv(advtrain_csv)
    clean = clean[clean["method"] == "YOLOv8"].copy()
    adv = adv[adv["method"] == "AdvAug-YOLOv8n"].copy()
    attack_names = {
        "clean": "Clean",
        "fgsm": "FGSM",
        "pgd": "PGD",
        "patch": "Patch",
        "sticker": "Sticker",
        "reflective": "Reflective",
        "motion_blur": "Motion blur",
        "occlusion": "Occlusion",
        "low_light": "Low light",
    }
    merged = clean.merge(adv, on="attack", suffixes=("_clean", "_adv"))
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "Attack": attack_names.get(row["attack"], row["attack"]),
                "Base @50": row["map50_clean"],
                "AdvAug @50": row["map50_adv"],
                "$\\Delta$ @50": row["map50_adv"] - row["map50_clean"],
                "Base @50:95": row["map50_95_clean"],
                "AdvAug @50:95": row["map50_95_adv"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.baseline_jsons:
        baseline = pretty_baselines(args.baseline_jsons)
        (output_dir / "baseline_models_auto.tex").write_text(
            dataframe_to_latex(
                baseline,
                "Clean held-out test performance of trained YOLOv8 helmet baselines.",
                "tab:baseline_models_auto",
                resize=True,
            ),
            encoding="utf-8",
        )
    robust = pretty_robustness(pd.read_csv(args.robustness_csv))
    (output_dir / "robustness_auto.tex").write_text(
        dataframe_to_latex(
            robust,
            "Robustness on the paired 50-image subset. Base ASR and robust target accuracy (RAcc.) quantify the unmitigated attack and are repeated across methods; mitigation is compared by mAP.",
            "tab:robustness_auto",
            resize=True,
        ),
        encoding="utf-8",
    )
    if args.ablation_csv:
        ablation = pretty_ablation(pd.read_csv(args.ablation_csv))
        (output_dir / "ablation_auto.tex").write_text(
            dataframe_to_latex(
                ablation,
                "Component ablation on the 20-clip public HELMET occlusion benchmark.",
                "tab:ablation_auto",
                resize=True,
                position="ht",
            ),
            encoding="utf-8",
        )
    if args.advtrain_robustness_csv:
        advtrain = pretty_advtrain_comparison(args.robustness_csv, args.advtrain_robustness_csv)
        (output_dir / "adversarial_training_auto.tex").write_text(
            dataframe_to_latex(
                advtrain,
                "Adversarial augmentation on the paired 50-image attack splits; entries are mAP.",
                "tab:adversarial_training_auto",
                resize=True,
            ),
            encoding="utf-8",
        )
    if args.edge_csv:
        edge = pd.read_csv(args.edge_csv)
        (output_dir / "edge_auto.tex").write_text(
            dataframe_to_latex(
                edge,
                "Edge deployment latency and throughput benchmark.",
                "tab:edge_auto",
                resize=True,
                position="H",
            ),
            encoding="utf-8",
        )
    elif args.edge_json:
        import json

        data = json.loads(Path(args.edge_json).read_text(encoding="utf-8"))
        device_label = data.get("hardware", {}).get("platform", "local")
        for metrics in (data.get("yolov8", {}), data.get("tcvm", {})):
            cuda_name = metrics.get("cuda_name")
            if cuda_name:
                device_label = cuda_name.replace("NVIDIA GeForce ", "").replace(" GPU", "")
                break
        rows = []
        for method in ("yolov8", "tcvm"):
            metrics = data.get(method, {})
            rows.append(
                {
                    "Method": "YOLOv8" if method == "yolov8" else "YOLOv8+TCVM-Net",
                    "Mean Latency (ms)": metrics.get("mean_ms", 0.0),
                    "p95 Latency (ms)": metrics.get("p95_ms", 0.0),
                    "FPS": metrics.get("fps", 0.0),
                    "Peak RSS (MB)": metrics.get("peak_rss_mb", 0.0),
                }
            )
        edge = pd.DataFrame(rows)
        (output_dir / "edge_auto.tex").write_text(
            dataframe_to_latex(
                edge,
                "Edge deployment latency, throughput, and memory usage.",
                "tab:edge_auto",
                resize=True,
                position="H",
            ),
            encoding="utf-8",
        )
    print(f"Wrote LaTeX tables to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
