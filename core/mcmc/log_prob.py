"""Log-probability callables for emcee.

Prior: uniform box over the OptProblem bounds (in search space; mass dims are
log10). Likelihood: ``-0.5 * point_source_loss`` reusing the exact DE objective
helper (including ``missing_img_penalty``) so the posterior matches the DE
optimization.
"""
from __future__ import annotations

from typing import Union

import numpy as np

from ..optimize.backends import Backend, make_backend
from ..optimize.loss import LossConfig
from ..optimize.objective import INVALID_LOSS, point_source_loss
from ..optimize.problem import OptProblem
from ..optimize.scene import ObsData


class LogProbability:
    """Picklable per-sample log-prob (for CPU/glafic or per-candidate GPU)."""

    def __init__(self, problem: OptProblem, obs: ObsData, loss_cfg: LossConfig,
                 backend: Union[str, Backend] = "cpu"):
        self.problem = problem
        self.obs = obs
        self.loss_cfg = loss_cfg
        b = problem.bounds
        self.lo = np.array([x[0] for x in b], dtype=float)
        self.hi = np.array([x[1] for x in b], dtype=float)
        if isinstance(backend, str):
            self.backend_name = backend
            self._backend = None
        else:
            self.backend_name = None
            self._backend = backend

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.backend_name is not None:
            state["_backend"] = None
        return state

    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = make_backend(self.backend_name)
        return self._backend

    def __call__(self, theta) -> float:
        theta = np.asarray(theta, dtype=float)
        if np.any(theta < self.lo) or np.any(theta > self.hi):
            return -np.inf
        scene = self.problem.make_scene(theta)
        try:
            images = self.backend().compute_images(scene)
        except Exception:  # noqa: BLE001
            return -np.inf
        # point_source_loss applies the same missing_img_penalty as the DE
        # objective, so the posterior stays consistent with the optimization.
        loss = point_source_loss(images, self.obs, self.loss_cfg)
        if loss >= INVALID_LOSS or not np.isfinite(loss):
            return -np.inf
        return -0.5 * loss


class ExtendLogProbability:
    """Picklable per-sample log-prob for the extended-source CPU path.

    Prior: uniform box over the OptProblem bounds (extend mode skips the source
    position, which glafic solves internally). Likelihood: ``-0.5 * weighted
    loss`` where the weighted loss is exactly the one the extend DE minimizes
    (glafic ``c2calc`` components with the W_* weights), so the posterior is
    consistent with the extend DE result. Reuses :class:`ExtendObjective`, which
    rebuilds the glafic engine lazily per worker (fork pool safe).
    """

    def __init__(self, problem: OptProblem, spec, loss_cfg):
        from ..optimize.extend import ExtendObjective  # noqa: PLC0415
        self.obj = ExtendObjective(problem, spec, loss_cfg)
        b = problem.bounds
        self.lo = np.array([x[0] for x in b], dtype=float)
        self.hi = np.array([x[1] for x in b], dtype=float)

    def __call__(self, theta) -> float:
        theta = np.asarray(theta, dtype=float)
        if np.any(theta < self.lo) or np.any(theta > self.hi):
            return -np.inf
        loss = self.obj.evaluate_one(theta)
        if loss >= INVALID_LOSS or not np.isfinite(loss):
            return -np.inf
        return -0.5 * float(loss)


class BatchedGPULogProbability:
    """Vectorized log-prob over walkers using the batched GPU objective.

    Used with ``emcee.EnsembleSampler(..., vectorize=True)``; receives a
    ``(nwalkers, ndim)`` array and returns ``(nwalkers,)`` log-probs.
    """

    def __init__(self, problem: OptProblem, obs: ObsData, loss_cfg: LossConfig):
        from ..optimize.batched import BatchedGPUObjective
        self.obj = BatchedGPUObjective(problem, obs, loss_cfg)
        b = problem.bounds
        self.lo = np.array([x[0] for x in b], dtype=float)
        self.hi = np.array([x[1] for x in b], dtype=float)

    def __call__(self, theta_batch) -> np.ndarray:
        arr = np.atleast_2d(np.asarray(theta_batch, dtype=float))  # (nwalkers, ndim)
        losses = np.asarray(self.obj(arr.T), dtype=float)          # (nwalkers,)
        in_box = np.all((arr >= self.lo) & (arr <= self.hi), axis=1)
        good = in_box & np.isfinite(losses) & (losses < INVALID_LOSS)
        return np.where(good, -0.5 * losses, -np.inf)
