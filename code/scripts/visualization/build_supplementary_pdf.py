"""Build a lightweight supplementary-material PDF from generated artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from PIL import Image as PILImage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build supplementary_material.pdf.")
    parser.add_argument("--output", default="supplementary/supplementary_material.pdf")
    return parser.parse_args()


def add_heading(story: list, styles, text: str) -> None:
    story.append(Paragraph(text, styles["Heading2"]))
    story.append(Spacer(1, 0.08 * inch))


def add_paragraph(story: list, styles, text: str) -> None:
    story.append(Paragraph(text, styles["BodyText"]))
    story.append(Spacer(1, 0.08 * inch))


def add_image(story: list, path: str, width: float = 6.4 * inch) -> None:
    image_path = Path(path)
    if image_path.exists():
        with PILImage.open(image_path) as image:
            image_width, image_height = image.size
        height = width * image_height / image_width
        max_height = 7.0 * inch
        if height > max_height:
            scale = max_height / height
            width *= scale
            height = max_height
        story.append(Image(str(image_path), width=width, height=height))
        story.append(Spacer(1, 0.12 * inch))
    else:
        story.append(Paragraph(f"Missing generated figure: {path}", getSampleStyleSheet()["BodyText"]))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    story: list = []
    story.append(Paragraph("Supplementary Material: TCVM-Net", styles["Title"]))
    add_paragraph(
        story,
        styles,
        "This supplement summarizes additional visual, threshold, hardware, and reproducibility material "
        "derived from generated TCVM-Net outputs. It does not introduce manually estimated metrics.",
    )

    add_heading(story, styles, "S1. Extended Attack Examples")
    add_paragraph(story, styles, "Representative physical-style perturbation panels from the attack pipeline.")
    add_image(story, "outputs/figures/attack_gallery.png", width=6.5 * inch)

    add_heading(story, styles, "S2. Temporal Confidence and Anomaly Trajectories")
    add_paragraph(story, styles, "Controlled reflective probe and real-video stress-test confidence trajectories.")
    add_image(story, "outputs/figures/temporal_confidence_reflective_probe.png", width=6.4 * inch)
    add_image(story, "outputs/figures/temporal_confidence_real_video_reflective.png", width=6.4 * inch)

    story.append(PageBreak())
    add_heading(story, styles, "S3. Grad-CAM Explanation")
    add_paragraph(story, styles, "Grad-CAM comparison before and during the reflective temporal attack.")
    add_image(story, "outputs/figures/gradcam_reflective_probe/gradcam_comparison.png", width=5.8 * inch)

    add_heading(story, styles, "S4. Threshold Sensitivity")
    threshold_table = Table(
        [
            ["Threshold", "mAP@50", "mAP@50:95", "Anom. Acc.", "F1", "Interpretation"],
            ["0.30", "0.519", "0.504", "0.413", "0.405", "High recall, many false positives"],
            ["0.45", "0.522", "0.512", "0.713", "0.258", "Balanced paper setting"],
            ["0.62", "0.516", "0.509", "0.800", "0.000", "Low FPR, misses attack window"],
        ],
        colWidths=[0.78 * inch, 0.72 * inch, 0.78 * inch, 0.82 * inch, 0.55 * inch, 2.5 * inch],
    )
    threshold_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(threshold_table)
    story.append(Spacer(1, 0.16 * inch))

    add_heading(story, styles, "S5. Hardware and Reproducibility")
    add_paragraph(
        story,
        styles,
        "Edge benchmark: YOLOv8 8.49 FPS and YOLOv8+TCVM-Net 3.98 FPS on the dense reflective probe "
        "using an RTX 4060 Laptop GPU. Peak resident memory increased from 1102 MB to 1120 MB.",
    )
    add_paragraph(
        story,
        styles,
        "Reproduction commands, dataset validation, and quality gates are documented in README.md, "
        "docs/experimental_execution.md, and docs/reproducibility_checklist.md.",
    )
    doc.build(story)
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
