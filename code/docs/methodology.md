# TCVM-Net Methodology

## Inputs

For frame `t`, the detector emits bounding boxes, classes, confidence scores, and compact region descriptors. The tracker associates detections with active track identifiers and stores a fixed-length history.

## Temporal Signals

TCVM-Net combines four normalized signals:

1. Confidence instability: deviation from the track-level exponential moving mean and variance.
2. Motion inconsistency: disagreement between the observed box and a constant-velocity prediction.
3. Feature discontinuity: cosine distance between the current HSV/Sobel descriptor and the track history.
4. Disappearance evidence: consecutive misses for a recently confident track, gated by frame-level detector-count collapse.

The weighted anomaly score is compared with a calibrated threshold. Suspicious detections can be rejected; short missing intervals can be recovered from the motion prior with discounted confidence.

## Conservative Recovery

Recovered boxes do not aggressively update the normal track history. This limits feedback poisoning and prevents a synthesized prediction from becoming equivalent to a detector observation.

## Causality and Cost

The verifier uses current and past frames only. Scalar track updates are linear in the number of active tracks for a fixed history window. Descriptor extraction is restricted to detected regions. Dense optical flow is optional and can be disabled or evaluated at reduced resolution.

## Threat Boundary

The method targets attacks and corruptions that cause abrupt detector inconsistency, including confidence collapse, localization jumps, feature changes, and short disappearance. A temporally persistent adaptive patch may remain consistent across frames and evade the verifier. TCVM-Net therefore provides empirical detection and recovery, not certified robustness.

## Calibration

Thresholds must be selected on video clips that are disjoint from the final test clips. Calibration should report the selected threshold, objective, constraints, and held-out false-positive rate. Adjacent frames from one source video must never be split across calibration and test sets.
