"""The DE objective: candidate vector -> scene -> images -> match -> loss.

Implemented as a *picklable* callable so scipy's process-pool ``workers`` can
fan it out. The (unpicklable) engine module is rebuilt lazily per worker from the
backend name; when constructed with an explicit backend object (e.g. a fake in
tests) it is used directly and the optimizer must run single-process.
"""
from __future__ import annotations

from typing import Union

import numpy as np

from .backends import Backend, make_backend
from .loss import LossConfig, ml_loss
from .matching import assign_images, select_images
from .problem import OptProblem
from .scene import ObsData

# returned when a candidate is physically invalid (wrong image count, engine error)
INVALID_LOSS = 1e15


def point_source_loss(images, obs: ObsData, loss_cfg: LossConfig) -> float:
    """Scalar point-source loss for one candidate's predicted images.

    Shared by the CPU :class:`Objective`, the batched GPU objective and the MCMC
    log-prob so they score a candidate identically.

    Returns :data:`INVALID_LOSS` for an unusable candidate (engine error, or --
    when ``missing_img_penalty`` is disabled -- an image count that does not match
    the observations). When ``loss_cfg.missing_img_penalty > 0`` a candidate that
    forms *fewer* images than observed is no longer rejected: its available images
    are Hungarian-matched to the best-fitting observed subset, scored with the
    usual weighted chi2, and ``(n_obs - n_pred) * missing_img_penalty`` is added
    so DE sees a gradient toward configurations that reproduce every image. An
    over-imaged candidate (more than ``n_obs + 1``) is still rejected.
    """
    if images is None:
        return INVALID_LOSS
    allow_partial = loss_cfg.missing_img_penalty > 0.0
    sel = select_images(images, obs.n, allow_partial=allow_partial)
    if sel is None:
        return INVALID_LOSS

    n_pred = len(sel)
    if n_pred == 0:
        base = 0.0                          # nothing to match; only the penalty
    else:
        pred_pos = np.array([[im[0], im[1]] for im in sel], dtype=float)
        pred_mag = np.array([im[2] for im in sel], dtype=float)
        _, matched_mag, delta, obs_idx = assign_images(
            obs.positions, pred_pos, pred_mag, obs.center_offset)
        base = ml_loss(delta, matched_mag,
                       obs.magnifications[obs_idx], obs.mag_errors[obs_idx],
                       obs.pos_sigma_mas[obs_idx], loss_cfg)
    n_missing = obs.n - n_pred
    return float(base + n_missing * loss_cfg.missing_img_penalty)


class Objective:
    def __init__(self, problem: OptProblem, obs: ObsData, loss_cfg: LossConfig,
                 backend: Union[str, Backend], auto_check: bool = False):
        self.problem = problem
        self.obs = obs
        self.loss_cfg = loss_cfg
        # auto_check: in-loop micro-image protection (plan §5b). When True and
        # a compact perturber sits within its trigger radius of a matched
        # image, that image's magnification is replaced by the local cluster
        # Sigma|mu| via a second zoomed engine cycle. False (or no trigger)
        # keeps the historical code path bit-identical.
        self.auto_check = bool(auto_check)
        if isinstance(backend, str):
            self.backend_name: str | None = backend
            self._backend: Backend | None = None
        else:
            self.backend_name = None
            self._backend = backend

    # -- pickling: drop the engine handle when we can rebuild it from a name --
    def __getstate__(self):
        state = self.__dict__.copy()
        if self.backend_name is not None:
            state["_backend"] = None
        return state

    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = make_backend(self.backend_name)
        return self._backend

    def _checked_loss(self, images, scene) -> float:
        """The auto_check loss: triggered images get their cluster Sigma|mu|
        from a second zoomed cycle of the SAME engine module (after the main
        cycle's quit; init/quit never nest). Only reachable when the backend
        is an EngineBackend-style object exposing its module as ``_m``."""
        from ..micro_audit import (checked_point_source_loss,
                                   find_compact_perturbers,
                                   make_binding_solver)
        if not find_compact_perturbers(scene):
            return point_source_loss(images, self.obs, self.loss_cfg)
        solver = make_binding_solver(self.backend()._m)
        return checked_point_source_loss(images, self.obs, self.loss_cfg,
                                         scene, solver)

    def evaluate_one(self, candidate) -> float:
        scene = self.problem.make_scene(np.asarray(candidate, dtype=float))
        try:
            images = self.backend().compute_images(scene)
        except Exception:
            return INVALID_LOSS
        if self.auto_check and getattr(self.backend(), "_m", None) is not None:
            try:
                return self._checked_loss(images, scene)
            except Exception:  # noqa: BLE001 — fail safe to the plain loss
                pass
        return point_source_loss(images, self.obs, self.loss_cfg)

    def __call__(self, candidate) -> float:
        return self.evaluate_one(candidate)
