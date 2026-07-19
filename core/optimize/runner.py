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
    de: DEResult           # optimizer result (DEResult-shaped for every algorithm)
    backend: str
    # Extended-source runs only (None for the point-only path):
    extend_spec: object = None        # core.optimize.extend.ExtendSpec
    extend_components: object = None   # best-fit c2calc_each tuple
    mode: str = "point"               # 'point' | 'extend'
    algorithm: str = "DE"             # 'DE' | 'BIPOP-CMA-ES' | 'JSO'


# accepted spellings of the OPTIMIZER .dat key / the ``algorithm`` argument
_ALGO_ALIASES = {
    "DE": "DE", "DIFFERENTIAL-EVOLUTION": "DE",
    "BIPOP-CMA-ES": "BIPOP-CMA-ES", "BIPOPCMAES": "BIPOP-CMA-ES",
    "CMA-ES": "BIPOP-CMA-ES", "CMAES": "BIPOP-CMA-ES",
    "JSO": "JSO",
}


def normalize_algorithm(name) -> str:
    """Canonical optimizer name for the OPTIMIZER key (case-insensitive)."""
    key = str(name).strip().upper().replace("_", "-")
    key = _ALGO_ALIASES.get(key.replace("-", ""), _ALGO_ALIASES.get(key))
    if key is None:
        raise ValueError(
            f"unknown OPTIMIZER '{name}'; expected one of DE, BIPOP-CMA-ES, jSO")
    return key


def optimize(cfg: GladeConfig,
             backend: Union[str, Backend] = "cpu",
             on_iteration: IterCallback = None,
             de_overrides: Optional[dict] = None,
             record_population: bool = True,
             base_dir: Optional[str] = None,
             algorithm: Optional[str] = None) -> OptResult:
    """Optimize ``cfg`` on ``backend`` with the selected point-source algorithm.

    ``algorithm`` (or the ``OPTIMIZER`` .dat key; default ``'DE'``) selects
    Differential Evolution, ``'BIPOP-CMA-ES'`` or ``'jSO'``. All three drive
    the same objective machinery: multi-process glafic on the CPU backends,
    the batched Rhongomyniad objective on ``'gpu'`` (per-candidate fallback
    when the config is not batchable).

    ``backend`` is normally ``'cpu' | 'glafic' | 'gpu'`` (rebuilt per worker for
    process parallelism); an explicit :class:`Backend` object may be passed for
    testing, in which case the run is single-process.

    If ``cfg`` is an extended-source configuration (a FITS ``extended_file`` or
    any extend-model component), the extended-source CPU path is used instead
    (DE only); ``base_dir`` resolves relative FITS / constraint / prior paths.
    """
    from ..format.validate import is_extend_mode
    if "fine_tuning" in cfg.algorithm:
        # optimize() is single-stage by contract; an ACTIVE fine_tuning key
        # would otherwise be silently ignored on the library path (the staged
        # pipeline is run_fine_tuning / the WebUI runjob dispatch).
        from .fine_tuning import resolve_fine_tuning
        if resolve_fine_tuning(cfg)[0] is not None:
            import warnings
            warnings.warn(
                "the active fine_tuning key is not executed by optimize(); "
                "call glade.run_fine_tuning(cfg, backend=...) for the staged "
                "macro -> substructure -> polish pipeline", stacklevel=2)
    algo = normalize_algorithm(
        algorithm if algorithm is not None
        else cfg.algorithm.get("OPTIMIZER", "DE"))
    if is_extend_mode(cfg):
        if algo != "DE":
            raise ValueError(
                f"{algo} is point-source only; extended-source runs use DE")
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
    vectorized = False
    if is_gpu:
        from .batched import BatchedGPUObjective, can_batch_gpu
        ok, _reason = can_batch_gpu(cfg)
        if ok:
            objective = BatchedGPUObjective(problem, obs, loss_cfg)
            vectorized = True
            de_cfg.gpu_vectorized = True   # whole population in one batched CUDA pass
    if objective is None:
        objective = Objective(
            problem, obs, loss_cfg, backend,
            auto_check=bool(cfg.algorithm.get("auto_check", True)))
    single_process = is_gpu or not isinstance(backend, str)
    if single_process:
        de_cfg.workers = 1   # GPU/torch and explicit-object backends run single-process

    if algo == "BIPOP-CMA-ES":
        from .cmaes import CMAESConfig, run_cmaes
        a_cfg = CMAESConfig.from_cfg(cfg)
        a_cfg.vectorized = vectorized
        if single_process:
            a_cfg.workers = 1
        result = run_cmaes(objective, problem.bounds, a_cfg,
                           on_iteration=on_iteration,
                           record_population=record_population)
    elif algo == "JSO":
        from .jso import JSOConfig, run_jso
        a_cfg = JSOConfig.from_cfg(cfg)
        a_cfg.vectorized = vectorized
        if single_process:
            a_cfg.workers = 1
        result = run_jso(objective, problem.bounds, a_cfg,
                         on_iteration=on_iteration,
                         record_population=record_population)
    else:
        if de_overrides:
            for k, v in de_overrides.items():
                setattr(de_cfg, k, v)
        result = run_de(objective, problem.bounds, de_cfg,
                        on_iteration=on_iteration,
                        record_population=record_population)

    return OptResult(
        x=result.x,
        loss=result.fun,
        fitted=problem.decode(result.x),
        scene=problem.make_scene(result.x),
        problem=problem,
        de=result,
        backend=backend if isinstance(backend, str) else getattr(backend, "name", "custom"),
        mode="point",
        algorithm=algo,
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


def calcimage(cfg: GladeConfig,
              backend: Union[str, Backend] = "cpu") -> OptResult:
    """Find images at the representative parameter values WITHOUT optimizing.

    Every ``{lo, hi}`` parameter takes its search-space midpoint — the
    geometric mean for mass-like (log10) dimensions, the arithmetic mean
    otherwise, exactly the convention of the glade->glafic export — and the
    images are computed once: glafic on the CPU backends, Rhongomyniad on
    ``'gpu'``. Fully-locked configurations (no optimizable parameters) are
    the primary use case and are fine here. Returns an OptResult (so the
    triptych / verification / glade_output machinery applies unchanged) whose
    ``loss`` is the informative point-source loss at those values.
    """
    from ..format.validate import is_extend_mode
    if is_extend_mode(cfg):
        raise ValueError("calcimage is point-source only")
    from .backends import make_backend
    from .objective import point_source_loss

    problem = OptProblem(cfg)          # ndim == 0 (all locked) is allowed
    obs = build_obs(cfg)
    loss_cfg = LossConfig.from_cfg(cfg)
    x = np.array([0.5 * (d.lo + d.hi) for d in problem.dims], dtype=float)
    scene = problem.make_scene(x)
    b = backend if not isinstance(backend, str) else make_backend(backend)
    images = b.compute_images(scene)
    loss = float(point_source_loss(images, obs, loss_cfg))
    de = DEResult(x=x, fun=loss, nit=0, converged=True, history=[])
    return OptResult(
        x=x, loss=loss, fitted=problem.decode(x), scene=scene,
        problem=problem, de=de,
        backend=backend if isinstance(backend, str) else getattr(backend, "name", "custom"),
        mode="point", algorithm="CALCIMAGE")
