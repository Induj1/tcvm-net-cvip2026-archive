"""Adversarial attack generation modules."""

from .digital import fgsm_attack, pgd_attack
from .physical import (
    apply_occlusion_attack,
    apply_reflective_pattern_attack,
    apply_sticker_attack,
    apply_motion_blur,
    apply_low_light,
    apply_named_physical_attack,
)
from .toolbox_adapters import DetectorConfidenceProxy, foolbox_linf_pgd, make_art_classifier

__all__ = [
    "fgsm_attack",
    "pgd_attack",
    "apply_occlusion_attack",
    "apply_reflective_pattern_attack",
    "apply_sticker_attack",
    "apply_motion_blur",
    "apply_low_light",
    "apply_named_physical_attack",
    "DetectorConfidenceProxy",
    "foolbox_linf_pgd",
    "make_art_classifier",
]
