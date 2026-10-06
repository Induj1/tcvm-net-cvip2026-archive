"""Base classes and shared helpers for adversarial attacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AttackResult:
    clean_image: np.ndarray
    adversarial_image: np.ndarray
    attack_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAttack:
    name = "base"

    def __call__(self, image: np.ndarray, **kwargs) -> AttackResult:
        raise NotImplementedError
