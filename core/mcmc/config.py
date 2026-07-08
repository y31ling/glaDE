"""MCMC configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..format.config import GladeConfig


@dataclass
class MCMCConfig:
    nwalkers: int = 32
    nsteps: int = 2000
    burnin: int = 300
    thin: int = 2
    perturbation: float = 0.01    # walker init spread as a fraction of bound width (DE+MCMC)
    # CPU pool size for the likelihood. Default 1 (single process) to match
    # DE_WORKERS: a fork pool left at -1 in a detached/backgrounded terminal can
    # orphan its workers and spin at ~full CPU after the parent is killed mid-run.
    # Set MCMC_WORKERS=-1 explicitly (in a real foreground terminal) to use all cores.
    workers: int = 1
    progress: bool = True
    seed: Optional[int] = None    # reuse DE_SEED when None

    @classmethod
    def from_cfg(cls, cfg: GladeConfig) -> "MCMCConfig":
        a = cfg.algorithm
        return cls(
            nwalkers=int(a.get("MCMC_NWALKERS", 32)),
            nsteps=int(a.get("MCMC_NSTEPS", 2000)),
            burnin=int(a.get("MCMC_BURNIN", 300)),
            thin=int(a.get("MCMC_THIN", 2)),
            perturbation=float(a.get("MCMC_PERTURBATION", 0.01)),
            workers=int(a.get("MCMC_WORKERS", 1)),
            progress=bool(a.get("MCMC_PROGRESS", True)),
            seed=int(a["DE_SEED"]) if "DE_SEED" in a else None,
        )

    @property
    def enabled_default(self) -> bool:  # convenience for the runjob
        return True
