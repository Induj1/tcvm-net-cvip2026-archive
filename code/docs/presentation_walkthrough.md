# TCVM-Net Presentation Walkthrough

This is a compact repository tour for a five-to-seven-minute technical demonstration.

## 1. Problem

Framewise traffic detectors can suffer abrupt confidence collapse, localization drift, or disappearance under physical-style perturbations. Image-only preprocessing and matched adversarial training do not use the ordered video stream that surveillance systems already provide.

Show: [`outputs/figures/attack_gallery.png`](../outputs/figures/attack_gallery.png)

## 2. Proposed Method

TCVM-Net wraps YOLOv8 and ByteTrack-style association without modifying detector weights. It combines confidence stability, motion continuity, compact appearance similarity, detector-count collapse, and short-gap recovery.

Show: [`outputs/figures/architecture_graphviz.png`](../outputs/figures/architecture_graphviz.png)

Key implementation: [`src/advtraffic/defense/tcvm.py`](../src/advtraffic/defense/tcvm.py)

## 3. Attack and Evaluation Pipeline

- Digital attacks: FGSM and PGD.
- Physical-style overlays: patch, sticker, reflective pattern, occlusion, motion blur, and low light.
- Detector-specific patch optimization uses calibration images only.
- Video thresholds are calibrated on clip-disjoint data and frozen for held-out attacks.

Key code:

- [`scripts/attacks/run_attack.py`](../scripts/attacks/run_attack.py)
- [`scripts/attacks/optimize_yolov8_patch.py`](../scripts/attacks/optimize_yolov8_patch.py)
- [`scripts/benchmark/calibrate_tcvm_threshold.py`](../scripts/benchmark/calibrate_tcvm_threshold.py)

## 4. Main Evidence

- Clean helmet YOLOv8n: 0.628 mAP@50 and 0.415 mAP@50:95.
- PGD: 0.988 ASR on the paired 50-image evaluation.
- Public 30-clip HELMET benchmark: 0.887 held-out anomaly F1 at 0.021 FPR.
- Twenty clip-disjoint resplits: median F1 0.889 and median FPR 0.020.
- Frozen reflective transfer: TCVM-Net reaches 0.887/0.021 F1/FPR, compared with 0.839/0.012 for reference recovery.
- Quarter-resolution optical flow: 14.10 FPS on the calibrated benchmark.

Show:

- [`outputs/figures/temporal_confidence_reflective_probe.png`](../outputs/figures/temporal_confidence_reflective_probe.png)
- [`outputs/figures/robustness_comparison.png`](../outputs/figures/robustness_comparison.png)
- [`outputs/figures/fps_robustness_tradeoff.png`](../outputs/figures/fps_robustness_tradeoff.png)

## 5. Ablation and Comparators

The repository includes tracker-only recovery, reference-frame recovery, image preprocessing, adversarial augmentation, and TCVM component ablations. Tracker-only recovery has high false-positive rates; reference recovery is strong for disappearance but transfers less effectively to reflection.

Show: [`outputs/figures/ablation_chart.png`](../outputs/figures/ablation_chart.png)

## 6. Honest Boundary

TCVM-Net is effective when attacks create abrupt temporal failure. It is not a certified defense and does not reliably identify temporally smooth optimized patches. Printed physical validation, adaptive verifier-aware attacks, and multi-camera evaluation remain future work.

This limitation is important: the project demonstrates a practical, measurable robustness layer rather than claiming universal adversarial robustness.

## Suggested Live Commands

```powershell
python -m pytest -q
python scripts/benchmark/repeat_split_sensitivity.py --help
python scripts/benchmark/summarize_crossattack_temporal_baselines.py --help
```

The compact result files used for discussion are under `outputs/results_50_complete/` and `outputs/results_50_complete_v2/`.
