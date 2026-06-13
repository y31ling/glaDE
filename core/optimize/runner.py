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
    # Extended-source runs only (None for the point-only path):
    extend_spec: object = None        # core.optimize.extend.ExtendSpec
    extend_components: object = None   # best-fit c2calc_each tuple
    mode: str = "point"               # 'point' | 'extend'


def optimize(cfg: GladeConfig,
             backend: Union[str, Backend] = "cpu",
             on_iteration: IterCallback = None,
             de_overrides: Optional[dict] = None,
             record_population: bool = True,
             base_dir: Optional[str] = None) -> OptResult:
    """Run Differential Evolution on ``cfg`` using ``backend``.

    ``backend`` is normally ``'cpu' | 'glafic' | 'gpu'`` (rebuilt per worker for
    process parallelism); an explicit :class:`Backend` object may be passed for
    testing, in which case the run is single-process.

    If ``cfg`` is an extended-source configuration (a FITS ``extended_file`` or
    any extend-model component), the extended-source CPU path is used instead;
    ``base_dir`` resolves relative FITS / constraint / prior file paths.
    """
    from ..format.validate import is_extend_mode
    if is_extend_mode(cfg):
        return _optimize_extend(cfg, backend, on_iteration, de_overrides,
                                record_population, base_dir)

    problem = OptProblem(cfg)
    if problem.ndim == 0:
        raise ValueError(
            "configuration has no optimizable {lo, hi} parameters to search")

    obs = build_obs(cfg)
    loss_cfg = LossConfig.from_cfg(cfg)
    de_cfg = DEConfig.from_cfg(cfg)

    is_gpu = isinstance(backend, str) and backend.lower() == "gpu"
    objective = None
    if is_gpu:
        from .batched import BatchedGPUObjective, can_batch_gpu
        ok, _reason = can_batch_gpu(cfg)
        if ok:
            objective = BatchedGPUObjective(problem, obs, loss_cfg)
            de_cfg.gpu_vectorized = True   # whole population in one batched CUDA pass
    if objective is None:
        objective = Objective(problem, obs, loss_cfg, backend)
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
        mode="point",
    )


def _optimize_extend(cfg: GladeConfig,
                     backend: Union[str, Backend],
                     on_iteration: IterCallback,
                     de_overrides: Optional[dict],
                     record_population: bool,
                     base_dir: Optional[str]) -> OptResult:
    """Extended-source path: DE over a c2calc_each weighted loss.

    ``backend='cpu'|'glafic'`` drives glafic per candidate (process pool);
    ``backend='gpu'`` drives Rhongomyniad — batched over the whole population
    when the configuration allows it, else per candidate (single process).
    """
    from .extend import ExtendObjective, build_extend_spec
    from .loss import ExtendLossConfig

    name = str(backend if isinstance(backend, str)
               else getattr(backend, "name", "cpu")).lower()
    is_gpu = name == "gpu"

    problem = OptProblem(cfg, extend_mode=True)
    if problem.ndim == 0:
        raise ValueError(
            "configuration has no optimizable {lo, hi} parameters to search")

    spec = build_extend_spec(cfg, base_dir=base_dir)
    loss_cfg = ExtendLossConfig.from_cfg(cfg)
    de_cfg = DEConfig.from_cfg(cfg)

    objective = None
    if is_gpu:
        from .batched_extend import BatchedExtendGPUObjective, can_batch_extend_gpu
        ok, _reason = can_batch_extend_gpu(cfg)
        if ok:
            objective = BatchedExtendGPUObjective(problem, spec, loss_cfg)
            de_cfg.gpu_vectorized = True
        de_cfg.workers = 1     # CUDA is single-process either way
    if objective is None:
        objective = ExtendObjective(problem, spec, loss_cfg,
                                    backend=("gpu" if is_gpu else name))
    if de_overrides:
        for k, v in de_overrides.items():
            setattr(de_cfg, k, v)

    result = run_de(objective, problem.bounds, de_cfg,
                    on_iteration=on_iteration,
                    record_population=record_population)
    best_components = objective.components_for(result.x)
    # NB: a temp point file (glade-arrays path) is intentionally left on disk so
    # the result figure / verification can re-read it; the caller owns cleanup.

    return OptResult(
        x=result.x,
        loss=result.fun,
        fitted=problem.decode(result.x),
        scene=problem.make_scene(result.x),
        problem=problem,
        de=result,
        backend=("gpu" if is_gpu else "cpu"),
        extend_spec=spec,
        extend_components=best_components,
        mode="extend",
    )
