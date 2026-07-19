"""BIPOP-CMA-ES (Hansen 2009) for the point-source optimization rail.

A faithful self-contained implementation — no external CMA library — of the
(mu/mu_w, lambda)-CMA-ES core (Hansen's tutorial, arXiv:1604.00772, incl. the
exact hsig indicator and the (1-hsig)*c_c*(2-c_c)*C covariance correction)
under the BI-POPulation restart strategy: interlaced "large" (IPOP doubling,
lambda = 2^k * lambda_def, fixed sigma0) and "small" (lambda in
[lambda_def, lambda_large/2] with U^2-distributed exponent, sigma0 scaled by
10^(-2U)) regimes, each restart taking whichever regime has consumed fewer
evaluations so far; the first (default) run is charged to the small budget;
the whole search stops after 9 large restarts or when the evaluation budget
is exhausted.

The search runs in normalized [0,1]^n coordinates (mass-like dimensions are
already log10 in the OptProblem search space, exactly as for DE). Box
constraints are handled by clip-repair + a quadratic penalty on the repaired
distance (Hansen's recommended simple scheme): candidates are RANKED by the
penalized value while the best-so-far tracks the raw loss of the repaired
(feasible) point.

Evaluation goes through :class:`core.optimize.evaluator.PopulationEvaluator`,
so the same loop drives the multi-process CPU/glafic objective and the
batched Rhongomyniad GPU objective (``vectorized=True``).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..format.config import GladeConfig
from .evaluator import PopulationEvaluator

IterCallback = Optional[Callable[[int, np.ndarray, float, np.ndarray], None]]


@dataclass
class CMAESConfig:
    maxevals: int = 0            # 0 -> 10000 * ndim (BBOB/CEC convention)
    seed: int = 42
    sigma0: float = 0.3          # initial step in normalized [0,1] coordinates
    popsize: int = 0             # 0 -> 4 + floor(3 ln n)  (lambda_def)
    max_restarts: int = 9        # BIPOP large-restart limit
    tolfun: float = 1e-10
    tolx: float = 1e-12          # relative to sigma0
    workers: int = 1
    vectorized: bool = False     # objective takes (ndim, N) (batched GPU)

    @classmethod
    def from_cfg(cls, cfg: GladeConfig) -> "CMAESConfig":
        a = cfg.algorithm
        return cls(
            maxevals=int(a.get("CMAES_MAXEVALS", 0)),
            seed=int(a.get("CMAES_SEED", a.get("DE_SEED", 42))),
            sigma0=float(a.get("CMAES_SIGMA0", 0.3)),
            popsize=int(a.get("CMAES_POPSIZE", 0)),
            max_restarts=int(a.get("CMAES_RESTARTS", 9)),
            tolfun=float(a.get("CMAES_TOLFUN", 1e-10)),
            tolx=float(a.get("CMAES_TOLX", 1e-12)),
            workers=int(a.get("CMAES_WORKERS", a.get("DE_WORKERS", 1))),
        )


@dataclass
class CMAESResult:
    x: np.ndarray
    fun: float
    nit: int
    converged: bool
    history: list = field(default_factory=list)
    n_evals: int = 0
    restarts: int = 0


class _SingleRun:
    """One CMA-ES run in normalized [0,1]^n coordinates."""

    def __init__(self, n: int, lam: int, sigma0: float, x0: np.ndarray,
                 rng: np.random.Generator, tolfun: float, tolx_abs: float):
        self.n, self.lam = n, max(int(lam), 4)
        self.rng = rng
        self.sigma0 = float(sigma0)
        self.sigma = float(sigma0)
        self.m = np.asarray(x0, dtype=float).copy()
        self.tolfun, self.tolx_abs = tolfun, tolx_abs

        lam, mu = self.lam, self.lam // 2
        w = np.log((lam + 1.0) / 2.0) - np.log(np.arange(1, mu + 1))
        self.w = w / w.sum()
        self.mu = mu
        self.mueff = 1.0 / np.sum(self.w ** 2)
        me, nn = self.mueff, float(n)
        self.cc = (4.0 + me / nn) / (nn + 4.0 + 2.0 * me / nn)
        self.cs = (me + 2.0) / (nn + me + 5.0)
        self.c1 = 2.0 / ((nn + 1.3) ** 2 + me)
        self.cmu = min(1.0 - self.c1,
                       2.0 * (me - 2.0 + 1.0 / me) / ((nn + 2.0) ** 2 + me))
        self.ds = 1.0 + 2.0 * max(0.0, math.sqrt((me - 1.0) / (nn + 1.0)) - 1.0) + self.cs
        self.chiN = math.sqrt(nn) * (1.0 - 1.0 / (4.0 * nn) + 1.0 / (21.0 * nn * nn))

        self.C = np.eye(n)
        self.B = np.eye(n)
        self.Dv = np.ones(n)
        self.invsqrtC = np.eye(n)
        self.ps = np.zeros(n)
        self.pc = np.zeros(n)
        self.counteval = 0
        self.eigeneval = 0
        self.gen = 0
        hist_len = 10 + int(math.ceil(30.0 * n / lam))
        self.hist = deque(maxlen=hist_len)
        self.best_hist: list[float] = []
        self.maxiter = int(100 + 50 * (n + 3) ** 2 / math.sqrt(lam))
        self.stop_reason: Optional[str] = None

    def ask(self) -> np.ndarray:
        Z = self.rng.standard_normal((self.lam, self.n))
        Y = Z @ (self.B * self.Dv).T          # y_k = B diag(D) z_k
        return self.m + self.sigma * Y

    def tell(self, X: np.ndarray, F_rank: np.ndarray, best_gen_f: float) -> None:
        idx = np.argsort(F_rank, kind="stable")
        self.counteval += self.lam
        self.gen += 1
        xold = self.m
        elite = X[idx[:self.mu]]
        self.m = self.w @ elite
        yw = (self.m - xold) / self.sigma
        zw = self.invsqrtC @ yw

        self.ps = ((1.0 - self.cs) * self.ps
                   + math.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * zw)
        denom = math.sqrt(1.0 - (1.0 - self.cs) ** (2.0 * self.gen))
        hsig = (np.linalg.norm(self.ps) / denom
                < (1.4 + 2.0 / (self.n + 1.0)) * self.chiN)
        self.pc = ((1.0 - self.cc) * self.pc
                   + (math.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * yw
                      if hsig else 0.0))

        artmp = (elite - xold) / self.sigma
        rank_mu = (artmp.T * self.w) @ artmp
        corr = 0.0 if hsig else self.cc * (2.0 - self.cc)
        self.C = ((1.0 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + corr * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= math.exp((self.cs / self.ds)
                               * (np.linalg.norm(self.ps) / self.chiN - 1.0))

        # flat-fitness escape (INVALID_LOSS plateaus): inflate sigma
        k70 = min(self.lam - 1, int(math.ceil(0.7 * self.lam)))
        if F_rank[idx[0]] == F_rank[idx[k70]]:
            self.sigma *= math.exp(0.2 + self.cs / self.ds)

        # lazy eigendecomposition
        if (self.counteval - self.eigeneval
                > self.lam / ((self.c1 + self.cmu) * self.n * 10.0)):
            self.eigeneval = self.counteval
            self.C = (self.C + self.C.T) / 2.0
            vals, vecs = np.linalg.eigh(self.C)
            vals = np.maximum(vals, 1e-30)
            self.B = vecs
            self.Dv = np.sqrt(vals)
            self.invsqrtC = vecs @ np.diag(1.0 / self.Dv) @ vecs.T

        self.hist.append(best_gen_f)
        self.best_hist.append(best_gen_f)
        self._check_stop(F_rank)

    def _check_stop(self, F_rank: np.ndarray) -> None:
        if self.stop_reason is not None:
            return
        n = self.n
        dmax, dmin = float(self.Dv.max()), float(self.Dv.min())
        if (dmax / max(dmin, 1e-300)) ** 2 > 1e14:
            self.stop_reason = "ConditionCov"
            return
        if self.sigma * dmax > 1e4 * self.sigma0:
            self.stop_reason = "TolXUp"
            return
        stds = self.sigma * np.sqrt(np.maximum(np.diag(self.C), 0.0))
        if np.all(stds < self.tolx_abs) and np.all(
                np.abs(self.sigma * self.pc) < self.tolx_abs):
            self.stop_reason = "TolX"
            return
        if len(self.hist) == self.hist.maxlen:
            lo, hi = min(self.hist), max(self.hist)
            gen_rng = float(np.max(F_rank) - np.min(F_rank))
            if hi - lo < self.tolfun and gen_rng < self.tolfun:
                self.stop_reason = "TolFun"
                return
        # NoEffectAxis (one principal axis per generation, cycled)
        i = self.gen % n
        step = 0.1 * self.sigma * self.Dv[i] * self.B[:, i]
        if np.all(self.m == self.m + step):
            self.stop_reason = "NoEffectAxis"
            return
        # NoEffectCoord
        if np.any(self.m == self.m + 0.2 * stds):
            self.stop_reason = "NoEffectCoord"
            return
        # Stagnation (simplified per Hansen: medians of the best-history)
        min_len = int(120 + 30.0 * n / self.lam)
        if len(self.best_hist) > max(min_len, 40):
            win = self.best_hist[-max(min_len, int(0.2 * len(self.best_hist))):]
            third = max(len(win) // 3, 1)
            if np.median(win[-third:]) >= np.median(win[:third]):
                self.stop_reason = "Stagnation"
        if self.gen >= self.maxiter:
            self.stop_reason = "MaxIter"


def run_cmaes(objective, bounds, cfg: CMAESConfig,
              on_iteration: IterCallback = None,
              record_population: bool = True) -> CMAESResult:
    """BIPOP-CMA-ES over ``bounds`` (search space; mass dims already log10)."""
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    span = np.where(ub > lb, ub - lb, 1.0)
    n = len(bounds)
    budget = cfg.maxevals if cfg.maxevals > 0 else 10000 * n
    lam_def = cfg.popsize if cfg.popsize > 0 else 4 + int(3 * math.log(n))
    rng = np.random.default_rng(cfg.seed)
    tolx_abs = cfg.tolx * cfg.sigma0

    history: list = []
    it_global = 0
    best_x = None
    best_f = float("inf")
    converged = False

    def to_real(Xn: np.ndarray) -> np.ndarray:
        return lb + Xn * span

    with PopulationEvaluator(objective, vectorized=cfg.vectorized,
                             workers=cfg.workers) as evalf:
        budget_small = 0
        budget_large = 0
        runs_small = 0
        irun = 0
        while evalf.n_evals < budget:
            n_large_done = irun - runs_small
            if irun == 0:
                lam, s0, regime = lam_def, cfg.sigma0, "small"
                x0 = np.full(n, 0.5)
                cap = budget
            elif budget_small < budget_large:
                regime = "small"
                runs_small += 1
                lam_large_next = lam_def * (2 ** n_large_done)
                u = rng.uniform()
                lam = max(lam_def, int(math.floor(
                    lam_def * (0.5 * lam_large_next / lam_def) ** (u * u))))
                s0 = cfg.sigma0 * 10.0 ** (-2.0 * rng.uniform())
                x0 = rng.uniform(0.0, 1.0, n)
                cap = max(lam * 10, int(0.5 * budget_large))
            else:
                regime = "large"
                if n_large_done + 1 > cfg.max_restarts:
                    break
                lam = lam_def * (2 ** (n_large_done + 1))
                s0 = cfg.sigma0
                x0 = rng.uniform(0.0, 1.0, n)
                cap = budget
            run_budget = min(cap, budget - evalf.n_evals)
            if run_budget < lam:
                break

            run = _SingleRun(n, lam, s0, x0, rng, cfg.tolfun, tolx_abs)
            used0 = evalf.n_evals
            while (run.stop_reason is None
                   and evalf.n_evals - used0 + run.lam <= run_budget):
                X = run.ask()
                Xc = np.clip(X, 0.0, 1.0)
                f_raw = evalf(to_real(Xc))
                dist2 = ((X - Xc) ** 2).sum(axis=1)
                finite = f_raw[np.isfinite(f_raw)]
                scale = 100.0 * (1.0 + (abs(float(np.median(finite)))
                                        if finite.size else 1.0))
                f_rank = f_raw + scale * dist2

                k = int(np.argmin(f_raw))
                if f_raw[k] < best_f:
                    best_f = float(f_raw[k])
                    best_x = to_real(Xc[k]).copy()
                gen_best = float(np.min(f_raw))
                run.tell(X, f_rank, gen_best)

                it_global += 1
                if record_population:
                    history.append({"iteration": it_global,
                                    "best_energy": best_f,
                                    "population": to_real(Xc)})
                else:
                    history.append({"iteration": it_global,
                                    "best_energy": best_f})
                if on_iteration is not None:
                    on_iteration(it_global, to_real(Xc), best_f,
                                 np.asarray(f_raw, dtype=float))
            used = evalf.n_evals - used0
            if regime == "small":
                budget_small += used
            else:
                budget_large += used
            if run.stop_reason in ("TolFun", "TolX"):
                converged = True
            irun += 1
        total_evals = evalf.n_evals

    if best_x is None:
        best_x = to_real(np.full(n, 0.5))
    return CMAESResult(x=np.asarray(best_x, dtype=float), fun=best_f,
                       nit=it_global, converged=converged, history=history,
                       n_evals=total_evals, restarts=irun - 1)
