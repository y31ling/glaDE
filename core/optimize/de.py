"""The Differential Evolution loop with early stopping.

Faithfully reproduces the legacy ``solver.__next__()`` loop: per-iteration best
tracking, an early-stop counter that needs ``EARLY_STOP_PATIENCE`` consecutive
within-tolerance iterations, and an optional per-iteration callback (used for
plotting / history).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

from ..format.config import GladeConfig


@dataclass
class DEConfig:
    maxiter: int = 650
    popsize: int = 64
    atol: float = 1e-4
    tol: float = 1e-4
    seed: int = 42
    polish: bool = True
    workers: int = 1
    early_stopping: bool = True
    early_stop_patience: int = 30
    gpu_vectorized: bool = False   # objective handles the whole population (batched GPU)

    @classmethod
    def from_cfg(cls, cfg: GladeConfig) -> "DEConfig":
        a = cfg.algorithm
        return cls(
            maxiter=int(a.get("DE_MAXITER", 650)),
            popsize=int(a.get("DE_POPSIZE", 64)),
            atol=float(a.get("DE_ATOL", 1e-4)),
            tol=float(a.get("DE_TOL", 1e-4)),
            seed=int(a.get("DE_SEED", 42)),
            polish=bool(a.get("DE_POLISH", True)),
            workers=int(a.get("DE_WORKERS", 1)),
            early_stopping=bool(a.get("EARLY_STOPPING", True)),
            early_stop_patience=int(a.get("EARLY_STOP_PATIENCE", 30)),
        )


@dataclass
class DEResult:
    x: np.ndarray
    fun: float
    nit: int
    converged: bool
    history: list  # list of dicts: {iteration, best_energy, population}


# callback(iteration, population, best_energy, population_energies) -> None
# `population` is in real search space (mass dims are log10); energies align row-wise.
IterCallback = Optional[Callable[[int, np.ndarray, float, np.ndarray], None]]


def run_de(objective, bounds, cfg: DEConfig,
           on_iteration: IterCallback = None,
           record_population: bool = True) -> DEResult:
    # Pin updating='deferred' on EVERY path (GPU-batched, GPU/CPU per-candidate,
    # and multi-worker). scipy's default for a single-process solver is
    # 'immediate', which advances the population mid-generation and therefore
    # produces a DIFFERENT same-seed DE trajectory than the batched-GPU and
    # multi-worker paths (both of which require 'deferred'). Forcing 'deferred'
    # everywhere preserves the project's same-seed cross-backend parity invariant:
    # the same config + seed walks an identical trajectory regardless of backend
    # or worker count. 'deferred' is valid for workers==1 (serial, one population
    # update per generation).
    if cfg.gpu_vectorized:
        workers, updating, vectorized = 1, "deferred", True
    else:
        workers = cfg.workers if cfg.workers else 1
        updating = "deferred"
        vectorized = False

    solver = DifferentialEvolutionSolver(
        objective,
        list(bounds),
        maxiter=cfg.maxiter,
        popsize=cfg.popsize,
        atol=cfg.atol,
        tol=cfg.tol,
        rng=np.random.default_rng(cfg.seed),
        polish=cfg.polish,
        disp=False,
        workers=workers,
        updating=updating,
        vectorized=vectorized,
    )

    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)

    def _real_population() -> np.ndarray:
        """Population in real search space. scipy stores it normalized to [0,1]."""
        pop = solver.population.copy()
        if pop.size and pop.max() <= 1.0 + 1e-9 and pop.min() >= -1e-9:
            return lb + pop * (ub - lb)
        return pop

    history: list = []

    def _emit(iteration: int, best: float) -> None:
        pop = _real_population()
        energies = np.array(solver.population_energies, dtype=float).copy()
        if record_population:
            history.append({"iteration": iteration, "best_energy": best,
                            "population": pop})
        else:
            history.append({"iteration": iteration, "best_energy": best})
        if on_iteration is not None:
            on_iteration(iteration, pop, best, energies)

    best_energy = float(np.min(solver.population_energies))
    _emit(0, best_energy)

    iteration = 1
    previous = best_energy
    converged_count = 0
    converged = False

    while True:
        next_gen = solver.__next__()
        best_energy = float(np.min(solver.population_energies))

        abs_change = abs(best_energy - previous)
        if abs(previous) > 1e-10 and math.isfinite(previous):
            rel_change = abs_change / abs(previous)
        else:
            rel_change = float("inf")
        within_tol = (abs_change < cfg.atol) or (rel_change < cfg.tol)

        if cfg.early_stopping:
            if within_tol:
                converged_count += 1
                if converged_count >= cfg.early_stop_patience:
                    converged = True
                    _emit(iteration, best_energy)
                    break
            else:
                converged_count = 0

        _emit(iteration, best_energy)
        previous = best_energy

        if next_gen is None:        # solver's own convergence
            converged = True
            break
        iteration += 1
        if iteration > cfg.maxiter:
            break

    return DEResult(
        x=np.asarray(solver.x, dtype=float),
        fun=float(np.min(solver.population_energies)),
        nit=iteration,
        converged=converged,
        history=history,
    )
