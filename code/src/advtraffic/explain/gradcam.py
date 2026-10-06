"""Grad-CAM utilities for YOLOv8 raw models."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from advtraffic.attacks.digital import detector_confidence_loss, normalize_torch_device, refresh_detection_head_tensors


class YOLOGradCAM:
    """Best-effort Grad-CAM for the final convolutional layer of a YOLO model."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None, device: str | None = None):
        self.model = model.eval()
        self.device = normalize_torch_device(device)
        self.model.to(self.device)
        self.target_layer = target_layer or self._find_last_conv(self.model)
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def generate(self, image_bgr: np.ndarray, image_size: int = 640, target_classes: list[int] | None = None) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(image_bgr, (image_size, image_size)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        x.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        refresh_detection_head_tensors(self.model)
        loss = detector_confidence_loss(self.model(x), target_classes)
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True).relu()
        cam = torch.nn.functional.interpolate(cam, size=(image_size, image_size), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-8)
        return cv2.resize(cam, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    def overlay(self, image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42) -> np.ndarray:
        colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(image_bgr, 1 - alpha, colored, alpha, 0)

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    @staticmethod
    def _find_last_conv(model: torch.nn.Module) -> torch.nn.Module:
        convs = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
        if not convs:
            raise ValueError("Could not find a Conv2d layer for Grad-CAM")
        return convs[-1]
