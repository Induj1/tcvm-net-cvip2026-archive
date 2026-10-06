"""Create a beginner-friendly end-to-end explainer PDF for TCVM-Net."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "TCVM-Net_Beginner_Friendly_End_to_End_Explanation.pdf"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | str | None, digits: int = 3) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=29,
            alignment=TA_CENTER,
            spaceAfter=16,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=17,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#334155"),
            spaceAfter=8,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=10,
            spaceAfter=8,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=8,
            spaceAfter=5,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.8,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "Small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.3,
            leading=11,
            spaceAfter=4,
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.2,
            leading=10.5,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=8,
        ),
        "Code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#0f172a"),
        ),
    }


S = styles()


def P(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, S[style])


def H1(text: str) -> Paragraph:
    return P(escape(text), "H1")


def H2(text: str) -> Paragraph:
    return P(escape(text), "H2")


def bullets(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(P(item), leftIndent=12) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=18,
    )


def note(title: str, text: str) -> Table:
    body = P(f"<b>{escape(title)}</b><br/>{escape(text)}", "Small")
    table = Table([[body]], colWidths=[6.6 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef6ff")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#7aa6d8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def make_table(rows: list[list[str]], col_widths: list[float] | None = None, font_size: float = 7.8) -> Table:
    data = [[P(str(cell), "Small") for cell in row] for row in rows]
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def add_image(story: list, path: Path, caption: str, max_width: float = 6.4 * inch, max_height: float = 3.4 * inch) -> None:
    if not path.exists():
        story.append(note("Missing visual", f"Expected image was not found: {path}"))
        return
    with PILImage.open(path) as img:
        width, height = img.size
    scale = min(max_width / width, max_height / height)
    flow = Image(str(path), width=width * scale, height=height * scale)
    flow.hAlign = "CENTER"
    story.append(flow)
    story.append(P(escape(caption), "Caption"))


def code_block(text: str) -> Table:
    block = Preformatted(text.strip(), S["Code"])
    table = Table([[block]], colWidths=[6.6 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(0.7 * inch, 0.45 * inch, "TCVM-Net beginner-friendly explainer")
    canvas.drawRightString(A4[0] - 0.7 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_story() -> list:
    robustness = read_csv(ROOT / "outputs" / "results_50_complete" / "robustness_summary.csv")
    public = read_csv(ROOT / "outputs" / "results_50_complete" / "public_helmet_multiclip30_calibrated_summary.csv")
    calibration = read_json(ROOT / "outputs" / "results_50_complete" / "tcvm_calibration_protocol_public_helmet30.json")
    edge = read_json(ROOT / "outputs" / "results_50_complete" / "edge_benchmark_reflective_probe.json")
    patch = read_csv(ROOT / "outputs" / "results_50_complete" / "yolov8_patch_public_summary.csv")

    heldout = calibration["heldout_metrics"]
    selected = calibration["selected_calibration_metrics"]
    all_clip = calibration["all_clip_metrics_at_selected_threshold"]

    story: list = []
    story.append(P("TCVM-Net", "Title"))
    story.append(P("Temporal Consistency Verification for Robust Traffic Surveillance", "Subtitle"))
    story.append(P("Beginner-Friendly End-to-End Explanation Guide", "Subtitle"))
    story.append(Spacer(1, 0.25 * inch))
    story.append(
        P(
            "This guide explains the complete project in plain language: what problem it solves, how the system works, "
            "what each module does, how attacks and datasets are used, how experiments are evaluated, what the final "
            "results mean, and how to reproduce the main outputs. It is a companion document, not a replacement for the LNCS paper.",
            "Body",
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        make_table(
            [
                ["Item", "Current final value"],
                ["Submission paper", "TCVM-Net: Temporal Consistency Verification for Robust Traffic Surveillance Against Physical Adversarial Attacks"],
                ["Primary detector", "YOLOv8n, with YOLOv8s used as a stronger detector comparison"],
                ["Core defense idea", "Use video-frame consistency to detect short adversarial failures"],
                ["Main calibrated public HELMET result", f"Held-out F1 {fmt(heldout['f1'])}, FPR {fmt(heldout['fpr'])}, threshold {fmt(calibration['selected_threshold'], 2)}"],
                ["Final LNCS manuscript", "14 pages, Springer LNCS format"],
            ],
            col_widths=[1.7 * inch, 4.9 * inch],
        )
    )
    story.append(PageBreak())

    story.append(H1("1. One-Page Summary"))
    story.append(
        P(
            "The project studies adversarial attacks on traffic surveillance systems. The practical setting is a camera "
            "watching riders, helmets, motorcycles, and traffic scenes. A normal YOLO detector looks at each frame and "
            "predicts boxes and classes. The problem is that physical-style perturbations such as stickers, reflective "
            "patterns, occlusion, blur, or low light can make the detector suddenly lose confidence or miss an object.",
        )
    )
    story.append(
        P(
            "TCVM-Net adds a lightweight video verification layer after YOLOv8 and tracking. It asks a simple question: "
            "if an object was visible and stable in recent frames, is the current detection behavior still temporally plausible? "
            "If confidence suddenly collapses, a box jumps, a feature changes abruptly, or a tracked object disappears for a short gap, "
            "TCVM-Net raises an anomaly score and can recover the short missing detection from track history.",
        )
    )
    story.append(
        note(
            "Beginner translation",
            "YOLO answers: what is in this frame? TCVM-Net answers: does this frame make sense compared with the last few frames?",
        )
    )
    story.append(H2("What changed in the final revision"))
    story.append(
        bullets(
            [
                "The public HELMET video stress test was expanded to 30 clips and 720 annotated frames.",
                "The anomaly threshold is no longer just a fixed guess: it is selected on 10 calibration clips.",
                "The selected threshold is then frozen and evaluated on 20 unseen held-out clips.",
                f"The held-out result is F1 {fmt(heldout['f1'])}, recall {fmt(heldout['recall'])}, precision {fmt(heldout['precision'])}, and FPR {fmt(heldout['fpr'])}.",
            ]
        )
    )
    story.append(PageBreak())

    story.append(H1("2. Key Vocabulary"))
    vocab = [
        ["Term", "Meaning in simple words"],
        ["Object detection", "Finding objects in an image and drawing bounding boxes around them."],
        ["YOLOv8", "The detector backbone used to detect helmets, riders, motorcycles, and traffic objects."],
        ["Confidence", "The detector's belief that a predicted object is real."],
        ["Bounding box", "The rectangle around the detected object."],
        ["Tracking", "Connecting the same object across multiple frames."],
        ["Temporal consistency", "The idea that nearby frames in a video should change smoothly."],
        ["Adversarial attack", "A carefully designed or simulated perturbation that makes a model fail."],
        ["Physical-style attack", "A perturbation inspired by real-world problems such as stickers, reflection, occlusion, blur, or low light."],
        ["Anomaly score", "A number saying how suspicious the current frame is compared with recent history."],
        ["FPR", "False positive rate: how often normal frames are incorrectly flagged as attacked."],
        ["F1", "A balanced score combining precision and recall for anomaly detection."],
        ["mAP@50", "Detection accuracy metric using 50 percent box-overlap threshold."],
    ]
    story.append(make_table(vocab, col_widths=[1.55 * inch, 5.05 * inch]))
    story.append(PageBreak())

    story.append(H1("3. Why This Problem Matters"))
    story.append(
        P(
            "Traffic surveillance systems are increasingly used for helmet compliance, rider detection, and traffic analytics. "
            "These systems often run automatically and may feed enforcement or safety dashboards. A short detector failure can be enough "
            "to miss a violation or hide a vulnerable road user.",
        )
    )
    story.append(
        P(
            "Most adversarial defenses are image-only. They take one image, denoise it, compress it, or train the model on adversarial examples. "
            "That can help in some cases, but a traffic camera sees video. Video provides free extra evidence: the same rider or motorcycle usually "
            "moves smoothly from one frame to the next. TCVM-Net uses this evidence.",
        )
    )
    story.append(
        bullets(
            [
                "Normal traffic video: confidence changes gradually, boxes move smoothly, and object appearance is stable.",
                "Adversarial or physical-style failure: confidence can collapse suddenly, boxes can jump, or objects can vanish briefly.",
                "TCVM-Net detects the suspicious change rather than trying to identify the exact attack texture.",
            ]
        )
    )
    add_image(
        story,
        ROOT / "paper" / "figures" / "architecture_graphviz.png",
        "Figure: Complete YOLOv8 + tracking + TCVM-Net verification pipeline.",
        max_height=2.5 * inch,
    )
    story.append(PageBreak())

    story.append(H1("4. Complete System Pipeline"))
    pipeline_rows = [
        ["Step", "Component", "What happens"],
        ["1", "Traffic video input", "The system receives a video stream from a traffic camera."],
        ["2", "Frame extraction", "The video is processed frame by frame."],
        ["3", "YOLOv8 detector", "YOLOv8 predicts object boxes, classes, and confidence scores."],
        ["4", "Object tracking", "ByteTrack or an IoU tracker links detections across frames using track IDs."],
        ["5", "TCVM-Net", "The temporal verifier compares current detections with recent track history."],
        ["6", "Anomaly scoring", "Confidence, motion, feature, and disappearance scores are combined."],
        ["7", "Robust prediction", "Stable detections are accepted; suspicious gaps can be recovered; unstable detections are flagged."],
    ]
    story.append(make_table(pipeline_rows, col_widths=[0.45 * inch, 1.55 * inch, 4.6 * inch]))
    story.append(
        note(
            "Important point",
            "TCVM-Net does not replace YOLOv8. It wraps around YOLOv8 as a verification layer. This makes the method modular and easier to deploy.",
        )
    )
    story.append(H2("Frame-by-frame intuition"))
    story.append(
        make_table(
            [
                ["Frame", "YOLO behavior", "TCVM interpretation"],
                ["1", "Helmet confidence 0.95", "Stable object history begins."],
                ["2", "Helmet confidence 0.94", "Still normal; confidence is stable."],
                ["3", "Helmet confidence 0.07 or object disappears", "Suspicious collapse; possible attack window."],
                ["4", "Helmet confidence returns to 0.95", "Short disappearance confirms temporal inconsistency."],
            ],
            col_widths=[0.8 * inch, 2.4 * inch, 3.4 * inch],
        )
    )
    story.append(PageBreak())

    story.append(H1("5. TCVM-Net Explained Module by Module"))
    story.append(H2("5.1 Confidence stability"))
    story.append(
        P(
            "For every tracked object, TCVM-Net remembers the recent average confidence. If the current confidence is very different from the recent average, "
            "the confidence instability score increases. This is useful because many attacks do not move the object physically, but they make the detector suddenly lose confidence.",
        )
    )
    story.append(code_block("confidence_score = abs(current_confidence - recent_average_confidence) / allowed_variation"))
    story.append(H2("5.2 Motion continuity"))
    story.append(
        P(
            "Objects in video do not teleport. A motorcycle can move fast, but from one frame to the next its bounding box usually changes smoothly. "
            "TCVM-Net predicts where the object should be using recent box motion and compares that prediction with the new box.",
        )
    )
    story.append(code_block("motion_score = 1 - IoU(current_box, predicted_box_from_track_history)"))
    story.append(H2("5.3 Feature similarity"))
    story.append(
        P(
            "A helmet or rider should look broadly similar across nearby frames. TCVM-Net uses lightweight visual descriptors such as color and edge patterns. "
            "If the current object's descriptor differs sharply from its recent descriptor, the feature inconsistency score rises.",
        )
    )
    story.append(code_block("feature_score = 1 - cosine_similarity(current_feature, recent_average_feature)"))
    story.append(H2("5.4 Disappearance consistency"))
    story.append(
        P(
            "Some attacks make an object vanish from detector output for one or two frames. TCVM-Net checks whether a recently stable object is missing. "
            "It promotes missing-track events only when the frame also shows detector-count collapse, reducing false recovery during ordinary tracking noise.",
        )
    )
    story.append(H2("5.5 Final anomaly score"))
    story.append(
        P(
            "The final anomaly score is a weighted combination of all four signals. If the score crosses a threshold, TCVM-Net flags the detection or missing event as suspicious.",
        )
    )
    story.append(
        code_block(
            """
anomaly_score =
    wc * confidence_score
  + wm * motion_score
  + wf * feature_score
  + wd * disappearance_score

if anomaly_score >= threshold:
    flag as adversarially inconsistent
else:
    accept as normal
            """
        )
    )
    story.append(PageBreak())

    story.append(H1("6. Attacks Used in the Project"))
    attack_rows = [
        ["Attack", "Beginner explanation", "Purpose"],
        ["FGSM", "One-step digital perturbation.", "Tests simple gradient-based vulnerability."],
        ["PGD", "Multi-step digital perturbation.", "Stronger digital baseline than FGSM."],
        ["Adversarial patch", "A visible patch intended to hide or confuse objects.", "Simulates localized physical attacks."],
        ["Sticker attack", "Sticker-like pattern near target regions.", "Models printed stickers on helmets or riders."],
        ["Reflective attack", "Bright stripe/reflection pattern.", "Models reflective tape or glare."],
        ["Motion blur", "Blur caused by motion or camera exposure.", "Tests natural surveillance degradation."],
        ["Occlusion", "Blocks part or all of the object.", "Tests short disappearance failures."],
        ["Low-light", "Darkens the image.", "Tests night or poor lighting conditions."],
    ]
    story.append(make_table(attack_rows, col_widths=[1.25 * inch, 3.0 * inch, 2.35 * inch]))
    add_image(
        story,
        ROOT / "outputs" / "figures" / "attack_gallery.png",
        "Figure: Examples of physical-style perturbations generated by the benchmark pipeline.",
        max_height=4.5 * inch,
    )
    story.append(PageBreak())

    story.append(H1("7. Datasets and Data Flow"))
    story.append(
        P(
            "The project uses public datasets and converts them into YOLO-compatible format. The core helmet detector is trained on a helmet dataset, "
            "while traffic-scene context and public video stress checks are used for broader surveillance realism.",
        )
    )
    data_rows = [
        ["Dataset or protocol", "Role in project"],
        ["Helmet detection dataset", "Main training and held-out testing for helmet/no-helmet/rider detection."],
        ["BDD100K", "Traffic-scene context and dataset validation."],
        ["AdvTraffic-26 protocol", "Unified attack-ready benchmark structure for clean and adversarial samples."],
        ["Public HELMET video clips", "Human-annotated public traffic frames for multi-clip temporal stress testing."],
    ]
    story.append(make_table(data_rows, col_widths=[2.0 * inch, 4.6 * inch]))
    story.append(H2("How data becomes usable"))
    story.append(
        bullets(
            [
                "Download or place public dataset files.",
                "Convert annotations into YOLO text files.",
                "Split into train, validation, and test sets.",
                "Generate clean and attacked copies.",
                "Run YOLOv8 and TCVM-Net on the same sequences.",
                "Save metrics, plots, tables, and logs under outputs/.",
            ]
        )
    )
    story.append(PageBreak())

    story.append(H1("8. Training and Baseline Models"))
    story.append(
        P(
            "The detector is trained first, before temporal defense is evaluated. YOLOv8n is the main model because it is lightweight. "
            "YOLOv8s is included to check whether a larger model changes the baseline behavior. An adversarial-augmentation model is also trained for comparison.",
        )
    )
    story.append(
        make_table(
            [
                ["Model", "Epochs", "mAP@50", "mAP@50:95", "Precision", "Recall", "Inference ms"],
                ["YOLOv8n", "50", "0.628", "0.415", "0.610", "0.617", "3.025"],
                ["YOLOv8s", "50", "0.634", "0.417", "0.957", "0.593", "5.908"],
                ["AdvAug-YOLOv8n", "44", "0.623", "0.404", "0.610", "0.586", "3.473"],
            ],
            col_widths=[1.7 * inch, 0.7 * inch, 0.8 * inch, 0.9 * inch, 0.8 * inch, 0.8 * inch, 0.9 * inch],
        )
    )
    story.append(
        note(
            "How to read this table",
            "YOLOv8s is slightly better in mAP but costs more time. YOLOv8n is chosen for the main defense because the paper targets real-time surveillance.",
        )
    )
    story.append(PageBreak())

    story.append(H1("9. Main Attack Robustness Results"))
    wanted_attacks = ["clean", "patch", "sticker", "reflective", "motion_blur", "occlusion", "low_light"]
    rows = [["Attack", "Method", "ASR", "Robust acc.", "mAP@50", "mAP@50:95"]]
    for row in robustness:
        if row["attack"] in wanted_attacks and row["method"] == "YOLOv8":
            rows.append(
                [
                    row["attack"].replace("_", " "),
                    row["method"],
                    fmt(row["attack_success_rate"]),
                    fmt(row["robust_accuracy"]),
                    fmt(row["map50"]),
                    fmt(row["map50_95"]),
                ]
            )
    story.append(make_table(rows, col_widths=[1.2 * inch, 1.2 * inch, 0.9 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch]))
    story.append(
        P(
            "The strongest physical-style attacks in the completed benchmark are reflective, occlusion, and motion blur. "
            "Reflective attack has very high attack success rate, which explains why temporal consistency is useful: the detector can fail for a short interval even though the object history was stable.",
        )
    )
    add_image(
        story,
        ROOT / "outputs" / "figures" / "robustness_comparison.png",
        "Figure: Robustness comparison across attacks and defenses.",
        max_height=3.2 * inch,
    )
    story.append(PageBreak())

    story.append(H1("10. Public HELMET Calibration Protocol"))
    story.append(
        P(
            "This is the most important final strengthening step. Instead of using one fixed threshold everywhere, the project uses a clip-disjoint calibration protocol. "
            "This means the threshold is chosen on one set of clips and evaluated on different clips. That is more honest and more reviewer-friendly.",
        )
    )
    story.append(
        make_table(
            [
                ["Split", "Clips", "Frames", "Threshold", "F1", "Precision", "Recall", "FPR"],
                [
                    "Calibration",
                    str(selected["clips"]),
                    str(selected["frames"]),
                    fmt(selected["threshold"], 2),
                    fmt(selected["f1"]),
                    fmt(selected["precision"]),
                    fmt(selected["recall"]),
                    fmt(selected["fpr"]),
                ],
                [
                    "Held-out",
                    str(heldout["clips"]),
                    str(heldout["frames"]),
                    fmt(heldout["threshold"], 2),
                    fmt(heldout["f1"]),
                    fmt(heldout["precision"]),
                    fmt(heldout["recall"]),
                    fmt(heldout["fpr"]),
                ],
                [
                    "All clips",
                    str(all_clip["clips"]),
                    str(all_clip["frames"]),
                    fmt(all_clip["threshold"], 2),
                    fmt(all_clip["f1"]),
                    fmt(all_clip["precision"]),
                    fmt(all_clip["recall"]),
                    fmt(all_clip["fpr"]),
                ],
            ],
            col_widths=[1.0 * inch, 0.6 * inch, 0.7 * inch, 0.85 * inch, 0.7 * inch, 0.8 * inch, 0.8 * inch, 0.7 * inch],
        )
    )
    story.append(
        note(
            "Why this matters",
            "The held-out clips are unseen during threshold selection. So the result is stronger than simply finding a threshold that works on the same clips.",
        )
    )
    story.append(H2("All-clip detection result after calibration"))
    public_rows = [["Setting", "Frames", "mAP@50", "mAP@50:95", "Anom. F1", "FPR", "FPS"]]
    for row in public:
        public_rows.append(
            [
                row["Setting"],
                row["Frames"],
                fmt(row["mAP@50"]),
                fmt(row["mAP@50:95"]),
                fmt(row["Anom. F1"]),
                fmt(row["FPR"]),
                fmt(row["FPS"]),
            ]
        )
    story.append(make_table(public_rows, col_widths=[2.35 * inch, 0.65 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.6 * inch, 0.6 * inch]))
    story.append(PageBreak())

    story.append(H1("11. Temporal Confidence and Recovery"))
    story.append(
        P(
            "The temporal confidence plot is the easiest way to understand TCVM-Net. During the attack window, raw detector confidence drops or objects disappear. "
            "TCVM-Net produces anomaly spikes and recovers short missing detections using track history.",
        )
    )
    add_image(
        story,
        ROOT / "outputs" / "figures" / "temporal_confidence_reflective_probe.png",
        "Figure: Detector confidence drops during the attack window, while TCVM anomaly score spikes.",
        max_height=3.4 * inch,
    )
    story.append(
        bullets(
            [
                "Blue line: detector confidence.",
                "Red line: TCVM anomaly score.",
                "Shaded region: attack window.",
                "Recovered markers: frames where temporal recovery supplies a short-gap prediction.",
            ]
        )
    )
    story.append(PageBreak())

    story.append(H1("12. Ablation Study: Which Part Matters?"))
    story.append(
        P(
            "An ablation study disables one component at a time. This shows whether the method works because of the full design or only because of one trick.",
        )
    )
    ablation_rows = [
        ["Variant", "mAP@50", "Anom. F1", "FPR", "Meaning"],
        ["Full TCVM", "0.355", "0.591", "0.181", "Complete design."],
        ["Track recovery", "0.365", "0.307", "0.645", "Simple recovery creates too many false alarms."],
        ["No motion", "0.344", "0.435", "0.345", "Motion continuity helps reduce false positives."],
        ["No feature", "0.331", "0.355", "0.486", "Feature similarity is important on natural video."],
        ["No smoothing", "0.354", "0.567", "0.200", "Smoothing stabilizes decisions."],
        ["No recovery", "0.346", "0.591", "0.181", "Detection of anomaly remains, but recovery mAP falls."],
    ]
    story.append(make_table(ablation_rows, col_widths=[1.2 * inch, 0.75 * inch, 0.75 * inch, 0.65 * inch, 3.25 * inch]))
    add_image(
        story,
        ROOT / "outputs" / "figures" / "ablation_chart.png",
        "Figure: Ablation chart showing robustness changes when TCVM components are removed.",
        max_height=3.2 * inch,
    )
    story.append(PageBreak())

    story.append(H1("13. Hard Failure Case: Detector-Specific Patch"))
    story.append(
        P(
            "The paper intentionally includes a negative result: a YOLOv8-specific printable-style EOT patch. This matters because strong papers do not pretend a method solves every attack. "
            "A persistent optimized patch can remain temporally smooth, so a temporal consistency defense may not flag it strongly.",
        )
    )
    patch_rows = [["Setting", "Frames", "mAP@50", "mAP50:95", "ASR", "Anom. recall", "Anom. F1"]]
    for row in patch:
        patch_rows.append(
            [
                row["Setting"],
                row["Frames"],
                fmt(row["mAP@50"]),
                fmt(row["mAP@50:95"]),
                fmt(row["ASR"]),
                fmt(row["Anom. R"]),
                fmt(row["Anom. F1"]),
            ]
        )
    story.append(make_table(patch_rows, col_widths=[2.2 * inch, 0.55 * inch, 0.7 * inch, 0.85 * inch, 0.55 * inch, 0.8 * inch, 0.75 * inch], font_size=6.8))
    add_image(
        story,
        ROOT / "outputs" / "figures" / "public_helmet_yolov8_patch_gallery.png",
        "Figure: YOLOv8-specific patch overlay examples.",
        max_height=3.3 * inch,
    )
    story.append(
        note(
            "Reviewer-friendly interpretation",
            "This failure case strengthens the paper because it clearly states the threat boundary: TCVM-Net is best for abrupt temporal inconsistency, not persistent smooth adaptive attacks.",
        )
    )
    story.append(PageBreak())

    story.append(H1("14. Explainability: Grad-CAM and Attention"))
    story.append(
        P(
            "Grad-CAM is used to visualize where the detector focuses before and during an attack. In this project it is only an analysis tool; it is not part of runtime inference. "
            "The key observation is that attention shifts under reflective perturbation while detections collapse, matching the temporal anomaly behavior.",
        )
    )
    add_image(
        story,
        ROOT / "outputs" / "figures" / "gradcam_reflective_probe" / "gradcam_comparison.png",
        "Figure: Grad-CAM comparison before and during the reflective attack.",
        max_height=3.6 * inch,
    )
    story.append(
        bullets(
            [
                "Clean frame: attention aligns better with target objects.",
                "Attacked frame: reflective pattern shifts attention and detections collapse.",
                "TCVM-Net then observes the sequence-level failure through confidence and disappearance signals.",
            ]
        )
    )
    story.append(PageBreak())

    story.append(H1("15. Edge Deployment and Runtime"))
    story.append(
        P(
            "The system is designed for surveillance deployment, so speed and memory matter. TCVM-Net adds temporal logic, optical flow, and feature comparison, so it is slower than YOLO alone. "
            "However, it remains practical as a selective verification layer for high-risk streams or lower-throughput deployments.",
        )
    )
    story.append(
        make_table(
            [
                ["Method", "Mean latency ms", "p95 latency ms", "FPS", "Peak RSS MB"],
                ["YOLOv8", fmt(edge["yolov8"]["mean_ms"]), fmt(edge["yolov8"]["p95_ms"]), fmt(edge["yolov8"]["fps"]), fmt(edge["yolov8"]["peak_rss_mb"])],
                ["YOLOv8 + TCVM-Net", fmt(edge["tcvm"]["mean_ms"]), fmt(edge["tcvm"]["p95_ms"]), fmt(edge["tcvm"]["fps"]), fmt(edge["tcvm"]["peak_rss_mb"])],
            ],
            col_widths=[1.8 * inch, 1.2 * inch, 1.2 * inch, 0.8 * inch, 1.0 * inch],
        )
    )
    add_image(
        story,
        ROOT / "outputs" / "figures" / "fps_robustness_tradeoff.png",
        "Figure: FPS versus robustness tradeoff.",
        max_height=3.3 * inch,
    )
    story.append(
        note(
            "Practical deployment idea",
            "Do not necessarily run full TCVM on every stream at maximum settings. Use scaled optical flow or enable TCVM selectively on sensitive cameras or suspicious intervals.",
        )
    )
    story.append(PageBreak())

    story.append(H1("16. How to Reproduce the Main Results"))
    story.append(
        P(
            "The repository is structured so that experiments write outputs into predictable folders. The paper reads tables and figures from those generated files. "
            "Below is a simplified reproduction path for the final calibrated public HELMET benchmark.",
        )
    )
    story.append(
        code_block(
            r"""
python scripts/dataset/prepare_public_helmet_multiclip.py ^
  --output-root outputs/tcvm_analysis/public_helmet_multiclip30_occlusion ^
  --split test --num-clips 30 --frames-per-clip 24 ^
  --attack-type occlusion --attack-offset 4 --attack-length 3 ^
  --class-id 3 --max-attack-boxes 64 --occlusion-ratio 1.0

python scripts/benchmark/benchmark_sequence_detector.py ^
  --sequence-root outputs/tcvm_analysis/public_helmet_multiclip30_occlusion/clean ^
  --model yolov8n.pt --output-dir outputs/tcvm_analysis/public_helmet_multiclip30_occlusion/clean/yolo_baseline ^
  --classes 3 --device 0 --conf 0.15

python scripts/benchmark/benchmark_sequence_detector.py ^
  --sequence-root outputs/tcvm_analysis/public_helmet_multiclip30_occlusion/occlusion ^
  --model yolov8n.pt --output-dir outputs/tcvm_analysis/public_helmet_multiclip30_occlusion/occlusion/yolo_baseline ^
  --classes 3 --device 0 --conf 0.15

python scripts/benchmark/calibrate_tcvm_threshold.py ^
  --run-root outputs/tcvm_analysis/public_helmet_multiclip30_occlusion/occlusion/tcvm_calibration_grid ^
  --thresholds 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 ^
  --calibration-clip-count 10 --seed 2026 ^
  --max-calibration-fpr 0.10 --min-calibration-recall 0.70
            """
        )
    )
    story.append(PageBreak())

    story.append(H1("17. How to Explain the Paper in a Viva or Review Discussion"))
    qa_rows = [
        ["Question", "Strong beginner-friendly answer"],
        [
            "What is the novelty?",
            "The novelty is not a new detector. It is a temporal verification layer that uses confidence, motion, feature, and disappearance consistency to detect physical-style adversarial failures in traffic video.",
        ],
        [
            "Why not just adversarial training?",
            "Adversarial training helps known synthetic families, but it is expensive and can overfit to attack types. TCVM-Net is complementary because it detects sequence-level inconsistency after inference.",
        ],
        [
            "Why does temporal consistency help?",
            "Traffic objects usually change smoothly across adjacent frames. Sudden confidence collapse or disappearance is suspicious when the object history was stable.",
        ],
        [
            "What is the strongest result?",
            f"On the public HELMET 30-clip protocol, threshold calibration on 10 clips generalizes to 20 held-out clips with F1 {fmt(heldout['f1'])} and FPR {fmt(heldout['fpr'])}.",
        ],
        [
            "What is the main limitation?",
            "Persistent adaptive patches can be temporally smooth, so TCVM-Net may not detect them. Real printed physical validation is future work.",
        ],
    ]
    story.append(make_table(qa_rows, col_widths=[1.75 * inch, 4.85 * inch], font_size=7.4))
    story.append(PageBreak())

    story.append(H1("18. Limitations and Honest Claim Boundary"))
    story.append(
        bullets(
            [
                "The method is a practical robustness layer, not a certified defense.",
                "It is strongest for abrupt temporal failures such as confidence collapse, box jumps, short disappearance, and reflective/occlusion-like events.",
                "It is weaker against persistent adaptive patches that remain smooth over time.",
                "The public HELMET benchmark uses real frames and human annotations, but the attack perturbation is synthetic.",
                "The paper does not claim full real-world deployment readiness without printed physical testing.",
                "Thresholds should be calibrated per camera, class, or deployment setting.",
            ]
        )
    )
    story.append(
        note(
            "Why limitations help",
            "Honest limitations increase reviewer trust. They show that the paper understands its boundary and is not overclaiming.",
        )
    )
    story.append(PageBreak())

    story.append(H1("19. Final Takeaway"))
    story.append(
        P(
            "TCVM-Net is best understood as a measured, deployment-aware temporal robustness framework. It does not try to prove that all attacks are solved. "
            "Instead, it shows that many physical-style traffic-surveillance failures are visible not only in pixels, but in time. By comparing the current frame with recent history, "
            "the system can flag suspicious detector behavior, recover short gaps, and provide a practical defense layer around YOLOv8.",
        )
    )
    story.append(
        make_table(
            [
                ["Final message", "Meaning"],
                ["For computer vision", "Video continuity is useful robustness evidence."],
                ["For adversarial ML", "Physical-style attacks should be evaluated beyond isolated images."],
                ["For traffic surveillance", "A lightweight verification layer can improve reliability without replacing the detector."],
                ["For reviewers", "The claims are bounded, the experiments are reproducible, and the final threshold is calibrated on disjoint clips."],
            ],
            col_widths=[1.8 * inch, 4.8 * inch],
        )
    )
    return story


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.65 * inch,
        title="TCVM-Net Beginner-Friendly End-to-End Explanation",
        author="TCVM-Net project",
    )
    story = build_story()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    main()
