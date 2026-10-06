"""Optional ART and Foolbox adapters for detector-confidence attacks."""

from __future__ import annotations

import torch


def normalize_torch_device(device: str | int | torch.device | None = None) -> torch.device:
    """Normalize device strings shared by Ultralytics, ART/Foolbox, and PyTorch."""

    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    device_text = str(device).strip()
    if device_text.isdigit():
        return torch.device(f"cuda:{device_text}" if torch.cuda.is_available() else "cpu")
    return torch.device(device_text)


def refresh_detection_head_tensors(model: torch.nn.Module) -> None:
    """Clone cached YOLOv8 inference tensors before ART/Foolbox autograd calls."""

    for module in model.modules():
        for name in ("anchors", "strides"):
            value = getattr(module, name, None)
            if isinstance(value, torch.Tensor):
                setattr(module, name, value.detach().clone())


def batch_detector_confidence(preds, target_classes: list[int] | None = None) -> torch.Tensor:
    """Return one YOLO confidence proxy score per batch item."""

    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    if preds.ndim != 3:
        raise ValueError(f"Unsupported YOLO prediction shape: {tuple(preds.shape)}")
    if preds.shape[1] < preds.shape[2]:
        class_scores = preds[:, 4:, :]
    else:
        class_scores = preds[:, :, 4:].transpose(1, 2)
    if target_classes is not None:
        class_scores = class_scores[:, target_classes, :]
    # Suppress the top-k confident anchors (near detections) per image, not just
    # the single global maximum: minimizing only the max anchor lets another
    # anchor take over and leaves mAP essentially unchanged.
    # The decoded Ultralytics eval output already contains probabilities.
    scores = class_scores.amax(dim=1)  # [B, N]
    k = max(1, int(round(0.01 * scores.shape[-1])))
    return scores.topk(k, dim=-1).values.mean(dim=-1)  # [B]


class DetectorConfidenceProxy(torch.nn.Module):
    """Classification-style proxy that lets ART/Foolbox attack YOLO confidence.

    The proxy exposes two logits: class 0 means "detector remains confident" and
    class 1 means "detector is suppressed". Untargeted classification attacks
    against label 0 therefore seek perturbations that reduce detector confidence.
    """

    def __init__(self, yolo_model: torch.nn.Module, target_classes: list[int] | None = None):
        super().__init__()
        self.yolo_model = yolo_model
        self.target_classes = target_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refresh_detection_head_tensors(self.yolo_model)
        confidence = batch_detector_confidence(self.yolo_model(x), self.target_classes).clamp(0, 1)
        return torch.stack([confidence, 1.0 - confidence], dim=1)


def make_art_classifier(
    yolo_model: torch.nn.Module,
    input_shape: tuple[int, int, int] = (3, 640, 640),
    target_classes: list[int] | None = None,
    device: str | None = None,
):
    """Create an ART PyTorchClassifier over the YOLO confidence proxy."""

    try:
        from art.estimators.classification import PyTorchClassifier
    except ImportError as exc:
        raise ImportError("Install adversarial-robustness-toolbox to use ART adapters") from exc

    device = normalize_torch_device(device)
    proxy = DetectorConfidenceProxy(yolo_model, target_classes=target_classes).to(device).eval()
    device_text = str(device)
    return PyTorchClassifier(
        model=proxy,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=input_shape,
        nb_classes=2,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device_text.startswith("cuda") else "cpu",
    )


def foolbox_linf_pgd(
    yolo_model: torch.nn.Module,
    images: torch.Tensor,
    eps: float = 8 / 255,
    steps: int = 10,
    target_classes: list[int] | None = None,
    device: str | None = None,
) -> torch.Tensor:
    """Run Foolbox Linf PGD against the detector-confidence proxy."""

    try:
        import foolbox as fb
    except ImportError as exc:
        raise ImportError("Install foolbox to use Foolbox adapters") from exc

    device = normalize_torch_device(device)
    proxy = DetectorConfidenceProxy(yolo_model, target_classes=target_classes).to(device).eval()
    fmodel = fb.PyTorchModel(proxy, bounds=(0, 1), device=device)
    labels = torch.zeros(images.size(0), dtype=torch.long, device=device)
    attack = fb.attacks.LinfPGD(steps=steps)
    _, clipped, _ = attack(fmodel, images.to(device), labels, epsilons=eps)
    return clipped.detach()
