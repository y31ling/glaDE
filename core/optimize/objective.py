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
from .matching import match_images, select_images
from .problem import OptProblem
from .scene import ObsData

# returned when a candidate is physically invalid (wrong image count, engine error)
INVALID_LOSS = 1e15


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
        if images is None:
            return INVALID_LOSS
        sel = select_images(images, self.obs.n)
        if sel is None:
            return INVALID_LOSS

        pred_pos = np.array([[im[0], im[1]] for im in sel], dtype=float)
        pred_mag = np.array([im[2] for im in sel], dtype=float)
        _, matched_mag, delta = match_images(
            self.obs.positions, pred_pos, pred_mag, self.obs.center_offset)
        return float(ml_loss(delta, matched_mag, self.obs.magnifications,
                             self.obs.mag_errors, self.obs.pos_sigma_mas,
                             self.loss_cfg))

    def __call__(self, candidate) -> float:
        return self.evaluate_one(candidate)
