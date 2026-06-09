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
    # missing_img_penalty : per-missing-image penalty. When > 0 a candidate that
    # forms FEWER images than observed is scored on its available images plus
    # (n_obs - n_pred) * this, instead of being hard-rejected (see
    # objective.point_source_loss). 0.0 keeps the historical hard reject.
    missing_img_penalty: float = 0.0

    @classmethod
    def from_cfg(cls, cfg: GladeConfig) -> "LossConfig":
        a = cfg.algorithm
        return cls(
            coef_a=float(a.get("LOSS_COEF_A", 4.0)),
            coef_b=float(a.get("LOSS_COEF_B", 1.0)),
            penalty_pl=float(a.get("LOSS_PENALTY_PL", 10000.0)),
            missing_img_penalty=float(a.get("missing_img_penalty", 0.0)),
        )


@dataclass
class ExtendLossConfig:
    """Per-component weights for the extended-source weighted loss.

    The loss is ``Σ w_k · chi2_k`` over glafic's c2calc components. With every
    weight at 1.0 the loss is exactly glafic's ``c2calc``; the legacy point-only
    behaviour corresponds to ``w_pos, w_flux`` non-zero and the rest zero.
    """

    w_pos: float = 1.0     # W_POS   : image-position chi2
    w_flux: float = 1.0    # W_FLUX  : flux / magnitude chi2
    w_td: float = 1.0      # W_TD    : time-delay chi2
    w_ext: float = 1.0     # W_EXT   : extended-source pixel chi2
    w_prior: float = 1.0   # W_PRIOR : all parameter priors (point + ext + lens + map)
    # missing_img_penalty : per-missing-image penalty for the extend path. When
    # > 0 and the SN point source forms FEWER images than observed, glafic's flat
    # chi2pen_nimg reject is replaced by the still-valid extended-pixel + prior
    # chi2 plus (n_obs - n_pred) * this (see extend.ExtendObjective.evaluate_one).
    missing_img_penalty: float = 0.0

    @classmethod
    def from_cfg(cls, cfg: GladeConfig) -> "ExtendLossConfig":
        a = cfg.algorithm
        return cls(
            w_pos=float(a.get("W_POS", 1.0)),
            w_flux=float(a.get("W_FLUX", 1.0)),
            w_td=float(a.get("W_TD", 1.0)),
            w_ext=float(a.get("W_EXT", 1.0)),
            w_prior=float(a.get("W_PRIOR", 1.0)),
            missing_img_penalty=float(a.get("missing_img_penalty", 0.0)),
        )

    def combine(self, comp) -> float:
        """Weight a c2calc_each tuple [pos,flux,td,prior_pt,pixel,prior_ext,
        prior_lens,penalty] into a scalar loss. The penalty term (physical-range
        violation) is always added unweighted."""
        pos, flux, td, prior_pt, pixel, prior_ext, prior_lens, penalty = comp
        return (self.w_pos * pos + self.w_flux * flux + self.w_td * td
                + self.w_ext * pixel
                + self.w_prior * (prior_pt + prior_ext + prior_lens)
                + penalty)


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
