"""Expectation-over-transformation adversarial patch optimizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from advtraffic.attacks.digital import detector_confidence_loss, normalize_torch_device, refresh_detection_head_tensors


PRINTABLE_RGB = torch.tensor(
    [
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ],
    dtype=torch.float32,
) / 255.0


@dataclass
class PatchOptimizerConfig:
    patch_size: int = 96
    steps: int = 250
    lr: float = 0.05
    tv_weight: float = 0.002
    nps_weight: float = 0.01
    scale_min: float = 0.24
    scale_max: float = 0.48
    minimize_confidence: bool = True


class EOTPatchOptimizer:
    """Optimize a printable patch under random brightness and scale transforms."""

    def __init__(self, model: torch.nn.Module, config: PatchOptimizerConfig | None = None, device: str | None = None):
        self.model = model.eval()
        self.config = config or PatchOptimizerConfig()
        self.device = normalize_torch_device(device)
        self.model.to(self.device)

    def optimize(self, images: torch.Tensor, boxes_xyxy: torch.Tensor, target_classes: list[int] | None = None) -> torch.Tensor:
        """Return optimized patch tensor [3, P, P] in RGB range [0, 1].

        images must be [B, 3, H, W] RGB tensors in [0, 1]. boxes_xyxy are pixel
        coordinates [B, 4] identifying helmet/rider regions.
        """

        cfg = self.config
        images = images.to(self.device)
        boxes_xyxy = boxes_xyxy.to(self.device)
        patch = torch.rand(1, 3, cfg.patch_size, cfg.patch_size, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([patch], lr=cfg.lr)

        for _ in range(cfg.steps):
            patched = self._paste_patch(images, boxes_xyxy, patch.clamp(0, 1))
            brightness = torch.empty((patched.size(0), 1, 1, 1), device=self.device).uniform_(0.75, 1.25)
            transformed = (patched * brightness).clamp(0, 1)
            refresh_detection_head_tensors(self.model)
            confidence = torch.nan_to_num(detector_confidence_loss(self.model(transformed), target_classes), nan=0.0, posinf=1.0, neginf=0.0)
            attack_loss = confidence if cfg.minimize_confidence else -confidence
            regularizer = cfg.tv_weight * total_variation(patch) + cfg.nps_weight * non_printability_score(patch)
            loss = torch.nan_to_num(attack_loss + regularizer, nan=0.0, posinf=1.0, neginf=0.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([patch], max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                patch.nan_to_num_(nan=0.5, posinf=1.0, neginf=0.0)
                patch.clamp_(0, 1)
        return patch.detach().squeeze(0).clamp(0, 1)

    def _paste_patch(self, images: torch.Tensor, boxes_xyxy: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        out = images.clone()
        batch, _, height, width = images.shape
        for i in range(batch):
            x1, y1, x2, y2 = boxes_xyxy[i]
            bw = torch.clamp(x2 - x1, min=1)
            bh = torch.clamp(y2 - y1, min=1)
            scale = torch.empty((), device=images.device).uniform_(cfg.scale_min, cfg.scale_max)
            side = torch.clamp(torch.minimum(bw, bh) * scale, min=8).round().long()
            patch_resized = F.interpolate(patch, size=(int(side.item()), int(side.item())), mode="bilinear", align_corners=False)
            cx = ((x1 + x2) / 2).round().long()
            cy = (y1 + 0.25 * (y2 - y1)).round().long()
            px1 = torch.clamp(cx - side // 2, min=0, max=width - 1).item()
            py1 = torch.clamp(cy - side // 2, min=0, max=height - 1).item()
            px2 = min(width, px1 + int(side.item()))
            py2 = min(height, py1 + int(side.item()))
            out[i : i + 1, :, py1:py2, px1:px2] = patch_resized[:, :, : py2 - py1, : px2 - px1]
        return out


def total_variation(patch: torch.Tensor) -> torch.Tensor:
    return (patch[:, :, 1:, :] - patch[:, :, :-1, :]).abs().mean() + (
        patch[:, :, :, 1:] - patch[:, :, :, :-1]
    ).abs().mean()


def non_printability_score(patch: torch.Tensor) -> torch.Tensor:
    colors = PRINTABLE_RGB.to(patch.device).view(1, -1, 3, 1, 1)
    pixels = patch.unsqueeze(1)
    distances = ((pixels - colors) ** 2).sum(dim=2).sqrt()
    return distances.min(dim=1).values.mean()
