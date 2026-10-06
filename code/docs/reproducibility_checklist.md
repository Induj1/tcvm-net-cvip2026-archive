# Reproducibility Checklist

## Environment

- Record Python, PyTorch, CUDA, Ultralytics, OpenCV, and GPU versions.
- Preserve `requirements.txt`, `environment.yml`, and the executed command line.
- Run `python -m compileall src scripts` and `python -m pytest -q`.

## Data

- Record dataset source URLs, versions, checksums, and license terms.
- Keep datasets and credentials outside Git.
- Use image-disjoint splits for detector training and clip-disjoint splits for video calibration/testing.
- Generate attacks only after splitting.
- Validate labels, class mappings, and out-of-frame boxes.

## Training

- Record model checkpoint, image size, batch size, epochs, early stopping, optimizer settings, seed, and precision mode.
- Preserve clean-test metrics separately from attacked metrics.
- Do not select checkpoints using final test data.

## Attacks

- Report epsilon, step size, steps, random start, patch size, scale, placement, alpha, EOT transforms, and target classes.
- State whether each attack is digital, simulated physical, or printed physical.
- Measure attack strength before interpreting defense performance.

## Temporal Evaluation

- Preserve source-frame order and clip identifiers.
- Select thresholds on calibration clips only.
- Report held-out precision, recall, F1, and false-positive rate.
- Report box-level mAP separately from frame-level anomaly metrics.
- Include tracker-only and reference-frame baselines.

## Deployment

- Use batch size one for streaming latency.
- Report mean and p95 latency, FPS, peak memory, GPU model, and enabled optical-flow mode.
- Separate detector time from optional temporal overhead when possible.

## Integrity

- Keep generated CSV/JSON summaries immutable after a run.
- Update reported values only by rerunning the pipeline.
- Report failure cases, pseudo-label use, synthetic overlays, and unsupported deployment claims explicitly.
