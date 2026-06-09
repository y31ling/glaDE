"""Backend-agnostic Differential Evolution optimizer for GLADE.

Extracted from the duplicated legacy ``version_*.py`` scripts. The same code
optimizes any combination of lens / sub-structure components against either the
glafic (CPU) engine or the Rhongomyniad (GPU) engine -- both expose the same
``init / startup_setnum / set_lens / set_point / model_init / point_solve``
primitives, so a single :class:`~core.optimize.backends.EngineBackend` drives
both.

Pipeline::

    cfg (core.format.GladeConfig)
      -> OptProblem (optimizable dimensions, log10 for mass-like params)
      -> Objective (scene -> backend images -> Hungarian match -> ML loss)
      -> run_de (scipy DifferentialEvolutionSolver + early stopping)
      -> OptResult (best scene, fitted params, history)
"""
from __future__ import annotations

from .backends import EngineBackend, make_backend
from .batched import BatchedGPUObjective, can_batch_gpu
from .de import DEConfig, DEResult, run_de
from .extend import ExtendObjective, ExtendSpec, build_extend_spec
from .loss import ExtendLossConfig, LossConfig, ml_loss
from .matching import match_images, select_images
from .objective import Objective
from .problem import Dim, OptProblem
from .runner import OptResult, optimize
from .scene import ObsData, Scene, SceneComponent, build_obs

__all__ = [
    "optimize", "OptResult",
    "OptProblem", "Dim",
    "Objective",
    "run_de", "DEConfig", "DEResult",
    "ml_loss", "LossConfig",
    "ExtendObjective", "ExtendSpec", "build_extend_spec", "ExtendLossConfig",
    "match_images", "select_images",
    "Scene", "SceneComponent", "ObsData", "build_obs",
    "EngineBackend", "make_backend",
    "BatchedGPUObjective", "can_batch_gpu",
]
