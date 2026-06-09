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
                 backend: Union[str, Backend]):
        self.problem = problem
        self.obs = obs
        self.loss_cfg = loss_cfg
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

    def evaluate_one(self, candidate) -> float:
        scene = self.problem.make_scene(np.asarray(candidate, dtype=float))
        try:
            images = self.backend().compute_images(scene)
        except Exception:
            return INVALID_LOSS
        return point_source_loss(images, self.obs, self.loss_cfg)

    def __call__(self, candidate) -> float:
        return self.evaluate_one(candidate)
