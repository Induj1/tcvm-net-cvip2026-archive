import torch

from advtraffic.defense.tcvm_net import TCVMNet


def test_tcvm_net_outputs_scores():
    module = TCVMNet()
    output = module(
        confidence=torch.tensor([0.2]),
        confidence_ema=torch.tensor([0.9]),
        confidence_var=torch.tensor([0.01]),
        boxes_xyxy=torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
        predicted_xyxy=torch.tensor([[11.0, 10.0, 21.0, 20.0]]),
        features=torch.ones(1, 8),
        feature_ema=torch.ones(1, 8),
    )
    assert 0.0 <= float(output["anomaly_score"][0]) <= 1.0
    assert float(output["confidence_instability"][0]) > 0.0
