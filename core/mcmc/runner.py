"""Run emcee MCMC, reusing the OptProblem bounds as the prior."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..optimize.loss import LossConfig
from ..optimize.problem import OptProblem
from ..optimize.scene import ObsData
from .config import MCMCConfig
from .log_prob import BatchedGPULogProbability, LogProbability


@dataclass
class MCMCResult:
    samples: np.ndarray            # (nsamples, ndim) flat, post burn-in + thin
    chain: np.ndarray              # (nsteps, nwalkers, ndim) raw
    acceptance_fraction: float
    param_names: list              # Dim labels
    is_log: list                   # per-dim: searched in log10 (mass-like)?
    burnin: int
    de_truth: Optional[np.ndarray] = None   # DE best (search space); None for MCMC-only
    summary: dict = field(default_factory=dict)


def run_mcmc(problem: OptProblem, obs: ObsData, loss_cfg: LossConfig,
             backend: str = "cpu",
             best_x: Optional[np.ndarray] = None,
             mcmc_cfg: Optional[MCMCConfig] = None,
             on_step: Optional[Callable[[int, object], None]] = None) -> MCMCResult:
    import emcee

    cfg = mcmc_cfg or MCMCConfig()
    ndim = problem.ndim
    if ndim == 0:
        raise ValueError("no optimizable {lo,hi} parameters to sample")
    nwalkers = max(int(cfg.nwalkers), 2 * ndim + 2)
    rng = np.random.default_rng(cfg.seed)

    bounds = problem.bounds
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    # walker initialization
    if best_x is not None:                                   # DE + MCMC: ball around DE best
        width = hi - lo
        init = np.asarray(best_x, dtype=float) + rng.normal(
            0.0, cfg.perturbation * width, size=(nwalkers, ndim))
        init = np.clip(init, lo, hi)
    else:                                                    # MCMC-only: uniform over the box
        init = np.column_stack([rng.uniform(lo[i], hi[i], nwalkers) for i in range(ndim)])

    # sampler + likelihood
    pool = None
    is_gpu = backend == "gpu"
    use_batched = False
    if is_gpu:
        from ..optimize.batched import can_batch_gpu
        use_batched = can_batch_gpu(problem.cfg)[0]
    if use_batched:
        log_prob = BatchedGPULogProbability(problem, obs, loss_cfg)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, vectorize=True)
    else:
        eng = backend if backend in ("cpu", "glafic", "gpu") else "cpu"
        log_prob = LogProbability(problem, obs, loss_cfg, eng)
        if eng != "gpu" and cfg.workers != 1:
            import multiprocessing as mp
            n = mp.cpu_count() if cfg.workers in (-1, 0) else cfg.workers
            pool = mp.get_context("fork").Pool(n)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)

    try:
        if on_step is not None:
            every = max(1, cfg.nsteps // 50)
            for k, _state in enumerate(sampler.sample(init, iterations=cfg.nsteps), start=1):
                if k == 1 or k % every == 0 or k == cfg.nsteps:
                    on_step(k, sampler)
        else:
            sampler.run_mcmc(init, cfg.nsteps, progress=cfg.progress)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    samples = sampler.get_chain(discard=cfg.burnin, thin=cfg.thin, flat=True)
    chain = sampler.get_chain()
    accept = float(np.mean(sampler.acceptance_fraction))

    is_log = [d.log for d in problem.dims]
    summary = {}
    for i, d in enumerate(problem.dims):
        col = samples[:, i]
        p16, p50, p84 = np.percentile(col, [16, 50, 84])
        entry = {"p16": float(p16), "p50": float(p50), "p84": float(p84)}
        if d.log:  # report mass-like in linear units too
            entry["p50_linear"] = float(10.0 ** p50)
        summary[d.label] = entry

    return MCMCResult(samples=samples, chain=chain, acceptance_fraction=accept,
                      param_names=[d.label for d in problem.dims], is_log=is_log,
                      burnin=cfg.burnin,
                      de_truth=(np.asarray(best_x, dtype=float) if best_x is not None else None),
                      summary=summary)
