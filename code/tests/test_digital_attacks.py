import numpy as np
import torch

from advtraffic.attacks.digital import _bgr_to_tensor, _resize_for_model, _tensor_to_bgr, detector_confidence_loss


def test_detector_resize_preserves_native_attack_grid_and_gradient():
    image = np.zeros((415, 416, 3), dtype=np.uint8)
    native = _bgr_to_tensor(image, "cpu").requires_grad_(True)

    detector_input = _resize_for_model(native, 640)
    detector_input.sum().backward()

    assert native.shape == (1, 3, 415, 416)
    assert detector_input.shape == (1, 3, 640, 640)
    assert native.grad is not None
    assert native.grad.shape == native.shape
    assert np.all(np.asarray(native.grad) > 0)
    assert _tensor_to_bgr(native).shape == image.shape


def test_detector_confidence_loss_does_not_double_sigmoid_probabilities():
    predictions = torch.zeros((1, 7, 100))
    predictions[:, 4, 0] = 0.9
    predictions[:, 5, 1] = 0.8

    loss = detector_confidence_loss(predictions, target_classes=[0, 1, 2])

    assert torch.isclose(loss, torch.tensor(0.9))
