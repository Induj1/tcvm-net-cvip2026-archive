import numpy as np

from advtraffic.defense import RobustPredictionLayer, TCVMConfig, TemporalConsistencyVerifier
from advtraffic.detection.types import Detection


def make_detection(frame_id, conf=0.9, box=None, feature=None):
    return Detection(
        frame_id=frame_id,
        class_id=0,
        class_name="helmet",
        confidence=conf,
        xyxy=np.array(box if box is not None else [10 + frame_id, 10, 50 + frame_id, 50], dtype=float),
        track_id=1,
        feature=np.ones(28, dtype=np.float32) if feature is None else feature,
    )


def test_tcvm_scores_confidence_drop():
    cfg = TCVMConfig(min_history=2, anomaly_threshold=0.3, update_threshold=0.9)
    verifier = TemporalConsistencyVerifier(cfg)
    for frame_id in range(3):
        scored, _ = verifier.verify([make_detection(frame_id, conf=0.9)], frame_id)
        assert not scored[0].is_adversarial

    scored, _ = verifier.verify([make_detection(3, conf=0.1)], 3)
    assert scored[0].anomaly_score > 0.0
    assert scored[0].is_adversarial


def test_robust_predictor_recovers_anomalous_detection():
    cfg = TCVMConfig(min_history=2, anomaly_threshold=0.4, update_threshold=0.9)
    layer = RobustPredictionLayer(config=cfg)
    for frame_id in range(3):
        output = layer.process([make_detection(frame_id, conf=0.9)], frame_id)
        assert output

    output = layer.process([make_detection(3, conf=0.1, box=[200, 200, 240, 240])], 3)
    assert output
    assert output[0].is_recovered
