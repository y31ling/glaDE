"""Turn a validated :class:`GladeConfig` into a Differential-Evolution problem.

Every ``{lo, hi}`` parameter (on the source position or on any component) becomes
one search dimension. Mass-like parameters (flagged in the schema) are searched
in log10 space, so the candidate value is ``10 ** x``; all others are linear.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..format import schema
from ..format.config import GladeConfig
from ..format.values import Bounds, Fixed
from .scene import Scene, SceneComponent


@dataclass
class Dim:
    """One optimizable dimension.

    ``target`` identifies where the value is injected:
        ('source', 'source_x' | 'source_y')
        ('cosmo', 'hubble')
        ('comp_z', comp_index)
        ('comp_param', comp_index, param_index)
    ``lo``/``hi`` are bounds in *search* space (already log10 for mass-like).
    """

    target: tuple
    lo: float
    hi: float
    log: bool
    label: str

    def to_value(self, x: float) -> float:
        return 10.0 ** x if self.log else x

    def midpoint_value(self) -> float:
        mid = 0.5 * (self.lo + self.hi)
        return 10.0 ** mid if self.log else mid


def _glafic_key(comp_type: str) -> str:
    spec = schema.model(comp_type)
    return spec.glafic_key if spec else comp_type


class OptProblem:
    """Optimizable-dimension model built from a config."""

    def __init__(self, cfg: GladeConfig, extend_mode: bool = False):
        self.cfg = cfg
        # In extend mode the point-source position is solved internally by glafic
        # (it is glafic's fast inner parameter), so source_x / source_y are NOT
        # outer DE dimensions; everything else (lens, extend, hubble) still is.
        self.extend_mode = extend_mode
        self.dims: list[Dim] = []
        self._build_dims()

    # -- dimension extraction ------------------------------------------------
    def _build_dims(self) -> None:
        if not self.extend_mode:
            src = self.cfg.source
            for axis in ("source_x", "source_y"):
                v = src.get(axis)
                if isinstance(v, Bounds):
                    self.dims.append(Dim(("source", axis), v.lo, v.hi, False, axis))

        # Hubble may be an optimizable dimension (e.g. time-delay cosmography).
        h = self.cfg.cosmology.get("hubble")
        if isinstance(h, Bounds):
            self.dims.append(Dim(("cosmo", "hubble"), h.lo, h.hi, False, "hubble"))

        for comp in self.cfg.components:
            spec = schema.model(comp.type)
            if isinstance(comp.z, Bounds):
                self.dims.append(
                    Dim(("comp_z", comp.index), comp.z.lo, comp.z.hi, False,
                        f"{comp.name}.z"))
            for j, p in enumerate(comp.params):
                if not isinstance(p, Bounds):
                    continue
                pname = spec.params[j].name if (spec and j < len(spec.params)) else f"p{j+1}"
                is_mass = bool(spec and j < len(spec.params) and spec.params[j].is_mass)
                if is_mass:
                    lo, hi = math.log10(p.lo), math.log10(p.hi)
                else:
                    lo, hi = p.lo, p.hi
                self.dims.append(
                    Dim(("comp_param", comp.index, j), lo, hi, is_mass,
                        f"{comp.name}.{pname}"))

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(d.lo, d.hi) for d in self.dims]

    @property
    def ndim(self) -> int:
        return len(self.dims)

    # -- scene reconstruction ------------------------------------------------
    def _fixed_scalar(self, section: dict, name: str, default: float) -> float:
        v = section.get(name, default)
        return float(v) if isinstance(v, (int, float)) else float(default)

    def _overrides(self, candidate) -> dict:
        return {d.target: d.to_value(candidate[i]) for i, d in enumerate(self.dims)}

    def make_scene(self, candidate) -> Scene:
        ov = self._overrides(candidate)
        return self._scene_from_overrides(ov)

    def baseline_scene(self) -> Scene:
        """Scene with optimizable params at their (geometric/arithmetic) midpoint."""
        ov = {d.target: d.midpoint_value() for d in self.dims}
        return self._scene_from_overrides(ov)

    def _scene_from_overrides(self, ov: dict) -> Scene:
        cfg = self.cfg
        cos, grid, rs, src = cfg.cosmology, cfg.grid, cfg.redshifts, cfg.source

        source_x = ov.get(("source", "source_x"),
                          self._fixed_scalar(src, "source_x", 0.0))
        source_y = ov.get(("source", "source_y"),
                          self._fixed_scalar(src, "source_y", 0.0))
        hubble = ov.get(("cosmo", "hubble"),
                        self._fixed_scalar(cos, "hubble", 0.7))

        components: list[SceneComponent] = []
        extends: list[SceneComponent] = []
        for comp in cfg.components:
            z = ov.get(("comp_z", comp.index),
                       comp.z.value if isinstance(comp.z, Fixed) else float("nan"))
            params: list[float] = []
            for j, p in enumerate(comp.params):
                if isinstance(p, Fixed):
                    params.append(p.value)
                else:  # Bounds -> from candidate
                    params.append(ov[("comp_param", comp.index, j)])
            sc = SceneComponent(_glafic_key(comp.type), float(z), params)
            (extends if schema.is_extend_model(comp.type) else components).append(sc)

        return Scene(
            omega=self._fixed_scalar(cos, "omega", 0.3),
            lam=self._fixed_scalar(cos, "lambda_cosmo", 0.7),
            weos=self._fixed_scalar(cos, "weos", -1.0),
            hubble=hubble,
            xmin=self._fixed_scalar(grid, "xmin", -0.5),
            ymin=self._fixed_scalar(grid, "ymin", -0.5),
            xmax=self._fixed_scalar(grid, "xmax", 0.5),
            ymax=self._fixed_scalar(grid, "ymax", 0.5),
            pix_ext=self._fixed_scalar(grid, "pix_ext", 0.01),
            pix_poi=self._fixed_scalar(grid, "pix_poi", 0.2),
            maxlev=int(self._fixed_scalar(grid, "maxlev", 5)),
            source_z=self._fixed_scalar(rs, "source_z", 0.409),
            source_x=source_x,
            source_y=source_y,
            components=components,
            extends=extends,
        )

    # -- result decoding -----------------------------------------------------
    def decode(self, candidate) -> dict:
        """Human-readable fitted values keyed by dimension label."""
        return {d.label: d.to_value(candidate[i]) for i, d in enumerate(self.dims)}
