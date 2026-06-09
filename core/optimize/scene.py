"""Concrete (fully-numeric) model scene and observation data.

A :class:`Scene` is what gets fed to an engine backend: cosmology + grid +
source + a flat list of concrete lens components. It is produced by
:class:`~core.optimize.problem.OptProblem` from a candidate parameter vector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..format.config import GladeConfig


@dataclass
class SceneComponent:
    glafic_type: str          # keyword passed to set_lens (engine model name)
    z: float                  # redshift
    params: list[float]       # p1..pk in glafic order (padded to 7 on emit)


@dataclass
class Scene:
    omega: float
    lam: float
    weos: float
    hubble: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    pix_ext: float
    pix_poi: float
    maxlev: int
    source_z: float
    source_x: float
    source_y: float
    components: list[SceneComponent] = field(default_factory=list)
    # Extended-source components (glafic set_extend); empty in the point-only path.
    extends: list[SceneComponent] = field(default_factory=list)


@dataclass
class ObsData:
    """Observation constraints, already converted to arcsec / engine frame."""

    positions: np.ndarray        # (n, 2) arcsec, x-flip applied
    magnifications: np.ndarray   # (n,)
    mag_errors: np.ndarray       # (n,)
    pos_sigma_mas: np.ndarray    # (n,) milliarcsec
    center_offset: tuple[float, float] = (0.0, 0.0)

    @property
    def n(self) -> int:
        return len(self.positions)


def _as_float(v, default=0.0) -> float:
    # scalar values in a GladeConfig are plain floats once locked; Bounds are
    # handled by OptProblem, never here.
    if isinstance(v, (int, float)):
        return float(v)
    return float(default)


def build_obs(cfg: GladeConfig) -> ObsData:
    """Build :class:`ObsData` from a merged config, applying the mas->arcsec
    conversion and the ``obs_x_flip`` sign exactly as the legacy scripts do."""
    obs = cfg.obs
    positions_mas = np.array(obs["obs_positions_mas_list"], dtype=float)
    x_sign = -1.0 if obs.get("obs_x_flip", False) else 1.0

    positions = np.zeros_like(positions_mas, dtype=float)
    positions[:, 0] = x_sign * positions_mas[:, 0] / 1000.0
    positions[:, 1] = positions_mas[:, 1] / 1000.0

    # The x-flip applies to BOTH the observed positions and the x center offset
    # (they live in the same observation frame); only then do the engine's
    # predictions line up with the observations. center_offset_y is unchanged.
    return ObsData(
        positions=positions,
        magnifications=np.array(obs["obs_magnifications_list"], dtype=float),
        mag_errors=np.array(obs["obs_mag_errors_list"], dtype=float),
        pos_sigma_mas=np.array(obs["obs_pos_sigma_mas_list"], dtype=float),
        center_offset=(x_sign * _as_float(obs.get("center_offset_x")),
                       _as_float(obs.get("center_offset_y"))),
    )
