"""The shared chi2 / machine-learning loss.

``Y = A * chi2_pos + B * chi2_mag + penalty`` where the penalty adds
``LOSS_PENALTY_PL * delta`` for every image whose positional residual exceeds its
1-sigma tolerance. This is identical across all legacy models (CPU and GPU).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..format.config import GladeConfig


@dataclass
class LossConfig:
    coef_a: float = 4.0       # LOSS_COEF_A : weight on positional chi2
    coef_b: float = 1.0       # LOSS_COEF_B : weight on magnification chi2
    penalty_pl: float = 10000.0  # LOSS_PENALTY_PL : per-image over-tolerance penalty

    @classmethod
    def from_cfg(cls, cfg: GladeConfig) -> "LossConfig":
        a = cfg.algorithm
        return cls(
            coef_a=float(a.get("LOSS_COEF_A", 4.0)),
            coef_b=float(a.get("LOSS_COEF_B", 1.0)),
            penalty_pl=float(a.get("LOSS_PENALTY_PL", 10000.0)),
        )


def ml_loss(delta_pos_mas: np.ndarray,
            pred_mag: np.ndarray,
            obs_mag: np.ndarray,
            obs_mag_err: np.ndarray,
            obs_pos_sigma_mas: np.ndarray,
            cfg: LossConfig) -> float:
    delta = np.asarray(delta_pos_mas, dtype=float)
    sigma = np.asarray(obs_pos_sigma_mas, dtype=float)

    chi2_pos = float(np.sum((delta / sigma) ** 2))
    chi2_mag = float(np.sum(((np.asarray(pred_mag, dtype=float) - obs_mag)
                             / obs_mag_err) ** 2))
    over = delta > sigma
    penalty = float(np.sum(cfg.penalty_pl * delta[over])) if np.any(over) else 0.0

    return cfg.coef_a * chi2_pos + cfg.coef_b * chi2_mag + penalty
