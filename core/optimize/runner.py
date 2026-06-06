"""High-level entry point: optimize a config on a chosen backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ..format.config import GladeConfig
from .backends import Backend
from .de import DEConfig, DEResult, IterCallback, run_de
from .loss import LossConfig
from .objective import Objective
from .problem import OptProblem
from .scene import Scene, build_obs


@dataclass
class OptResult:
    x: np.ndarray          # best candidate (search space; mass dims are log10)
    loss: float
    fitted: dict           # dimension label -> fitted physical value
    scene: Scene           # best-fit concrete scene
    problem: OptProblem
    de: DEResult
    backend: str


def optimize(cfg: GladeConfig,
             backend: Union[str, Backend] = "cpu",
             on_iteration: IterCallback = None,
             de_overrides: Optional[dict] = None,
             record_population: bool = True) -> OptResult:
    """Run Differential Evolution on ``cfg`` using ``backend``.

    ``backend`` is normally ``'cpu' | 'glafic' | 'gpu'`` (rebuilt per worker for
    process parallelism); an explicit :class:`Backend` object may be passed for
    testing, in which case the run is single-process.
    """
    problem = OptProblem(cfg)
    if problem.ndim == 0:
        raise ValueError(
            "configuration has no optimizable {lo, hi} parameters to search")

    obs = build_obs(cfg)
    loss_cfg = LossConfig.from_cfg(cfg)
    objective = Objective(problem, obs, loss_cfg, backend)

    de_cfg = DEConfig.from_cfg(cfg)
    is_gpu = isinstance(backend, str) and backend.lower() == "gpu"
    if is_gpu or not isinstance(backend, str):
        de_cfg.workers = 1   # GPU/torch and explicit-object backends run single-process
    if de_overrides:
        for k, v in de_overrides.items():
            setattr(de_cfg, k, v)

    result = run_de(objective, problem.bounds, de_cfg,
                    on_iteration=on_iteration, record_population=record_population)

    return OptResult(
        x=result.x,
        loss=result.fun,
        fitted=problem.decode(result.x),
        scene=problem.make_scene(result.x),
        problem=problem,
        de=result,
        backend=backend if isinstance(backend, str) else getattr(backend, "name", "custom"),
    )
