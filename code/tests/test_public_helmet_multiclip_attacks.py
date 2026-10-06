from types import SimpleNamespace

import numpy as np

from scripts.dataset.prepare_public_helmet_multiclip import apply_attack


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        attack_type="occlusion",
        max_attack_boxes=4,
        occlusion_ratio=1.0,
        reflective_intensity=0.82,
        reflective_scale=0.92,
        stripe_width=9,
        sticker_scale=0.38,
        patch_scale=0.45,
        patch_path=None,
        motion_blur_kernel=17,
        motion_blur_angle=0.0,
    )


def test_cross_attack_families_preserve_shape_and_change_pixels() -> None:
    grid = np.indices((128, 192)).sum(axis=0) % 2
    image = np.repeat((grid * 255).astype(np.uint8)[..., None], 3, axis=2)
    boxes = [(24.0, 20.0, 152.0, 108.0)]

    for attack_type in ("occlusion", "reflective", "sticker", "patch", "detector_patch", "motion_blur"):
        attacked = apply_attack(image, boxes, _args(), attack_type)
        assert attacked.shape == image.shape
        assert attacked.dtype == np.uint8
        assert np.any(attacked != image), attack_type
