"""jSO (Brest, Sepesy Maucec, Boskovic — CEC 2017) for the point-source rail.

A faithful implementation of the jSO algorithm (the winner-class iL-SHADE /
L-SHADE descendant; pseudo-code Algorithm 1 of the paper, vendored at
exception/jSO_CEC2017.pdf, cross-checked against Brest's reference C++):

* NP_init = round(25 * ln(D) * sqrt(D))  (NATURAL log), NP_min = 4, linear
  population size reduction (LPSR) with NP-truncation of the worst;
* success-history memories M_F (init 0.3) / M_CR (init 0.8) of size H = 5;
  reading the LAST slot always yields the fixed terminal pair (0.9, 0.9);
  updates use the weighted Lehmer mean for BOTH F and CR, averaged with the
  old slot value (iL-SHADE), with the CR terminal (⊥ = -1) rule;
* DE/current-to-pBest-w/1/bin: v = x + Fw*(x_pbest - x) + F*(x_r1 - x_r2),
  Fw = 0.7F / 0.8F / 1.2F for nfes < 0.2/0.4/otherwise of max_nfes; r2 drawn
  from population ∪ archive (|A| = NP, random replacement when full);
* p_best rate decays linearly 0.25 -> 0.125 with nfes; p_num >= 2;
* jSO clamps: CR >= 0.7 (nfes < 0.25 max), CR >= 0.6 (< 0.5 max);
  F <= 0.7 (nfes < 0.6 max); F ~ Cauchy(M_F, 0.1) redrawn while <= 0,
  clipped at 1; CR ~ N(M_CR, 0.1) clipped to [0, 1];
* bound repair by midpoint reflection toward the parent;
* selection: child replaces parent on <=; success sets / archive on < only.

Evaluation goes through :class:`core.optimize.evaluator.PopulationEvaluator`
(multi-process CPU/glafic objective, or the batched Rhongomyniad GPU
objective with ``vectorized=True``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..format.config import GladeConfig
from .evaluator import PopulationEvaluator

IterCallback = Optional[Callable[[int, np.ndarray, float, np.ndarray], None]]


@dataclass
class JSOConfig:
    maxevals: int = 0            # 0 -> 10000 * ndim (CEC convention)
    seed: int = 42
    np_init: int = 0             # 0 -> round(25 ln(D) sqrt(D))
    np_min: int = 4
    h: int = 5
    arc_rate: float = 1.0
    pbest_max: float = 0.25
    workers: int = 1
    vectorized: bool = False

    @classmethod
    def from_cfg(cls, cfg: GladeConfig) -> "JSOConfig":
        a = cfg.algorithm
        return cls(
            maxevals=int(a.get("JSO_MAXEVALS", 0)),
            seed=int(a.get("JSO_SEED", a.get("DE_SEED", 42))),
            np_init=int(a.get("JSO_NP_INIT", 0)),
            np_min=int(a.get("JSO_NP_MIN", 4)),
            h=int(a.get("JSO_H", 5)),
            arc_rate=float(a.get("JSO_ARC_RATE", 1.0)),
            pbest_max=float(a.get("JSO_PBEST_MAX", 0.25)),
            workers=int(a.get("JSO_WORKERS", a.get("DE_WORKERS", 1))),
        )


@dataclass
class JSOResult:
    x: np.ndarray
    fun: float
    nit: int
    converged: bool
    history: list = field(default_factory=list)
    n_evals: int = 0


def _lehmer(values: np.ndarray, weights: np.ndarray) -> float:
    num = float(np.sum(weights * values * values))
    den = float(np.sum(weights * values))
    return num / den if den != 0.0 else 0.0


def run_jso(objective, bounds, cfg: JSOConfig,
            on_iteration: IterCallback = None,
            record_population: bool = True) -> JSOResult:
    """jSO over ``bounds`` (search space; mass dims already log10)."""
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    D = len(bounds)
    rng = np.random.default_rng(cfg.seed)
    max_nfes = cfg.maxevals if cfg.maxevals > 0 else 10000 * D
    np_init = (cfg.np_init if cfg.np_init > 0
               else int(round(25.0 * math.log(D) * math.sqrt(D))) if D > 1
               else 25)
    np_init = max(np_init, cfg.np_min, 4)
    H = max(int(cfg.h), 1)

    NP = np_init
    X = lb + rng.uniform(size=(NP, D)) * (ub - lb)
    history: list = []
    it = 0

    M_F = np.full(H, 0.3)
    M_CR = np.full(H, 0.8)
    k_pos = 0
    archive = np.empty((0, D))
    arc_size = int(round(cfg.arc_rate * NP))

    with PopulationEvaluator(objective, vectorized=cfg.vectorized,
                             workers=cfg.workers) as evalf:
        F_pop = evalf(X)
        nfes = NP

        def emit(pop, fvals):
            nonlocal it
            it += 1
            best = float(np.min(fvals))
            if record_population:
                history.append({"iteration": it, "best_energy": best,
                                "population": pop.copy()})
            else:
                history.append({"iteration": it, "best_energy": best})
            if on_iteration is not None:
                on_iteration(it, pop.copy(), best,
                             np.asarray(fvals, dtype=float).copy())

        emit(X, F_pop)

        while nfes < max_nfes:
            order = np.argsort(F_pop, kind="stable")
            frac = nfes / max_nfes
            p_rate = cfg.pbest_max * (1.0 - 0.5 * frac)
            p_num = max(2, int(round(NP * p_rate)))
            p_num = min(p_num, NP)

            # --- per-individual F / CR from the success-history memory ------
            r = rng.integers(0, H, NP)
            mu_F = M_F[r].copy()
            mu_CR = M_CR[r].copy()
            terminal_slot = (r == H - 1)
            mu_F[terminal_slot] = 0.9      # fixed last-slot pair (iL-SHADE)
            mu_CR[terminal_slot] = 0.9

            CR = rng.normal(mu_CR, 0.1)
            CR = np.clip(CR, 0.0, 1.0)
            CR[mu_CR < 0.0] = 0.0          # terminal (⊥) memory -> CR = 0
            if frac < 0.25:
                CR = np.maximum(CR, 0.7)
            elif frac < 0.50:
                CR = np.maximum(CR, 0.6)

            Fs = np.empty(NP)
            for i in range(NP):
                f = mu_F[i] + 0.1 * math.tan(math.pi * (rng.uniform() - 0.5))
                while f <= 0.0:
                    f = mu_F[i] + 0.1 * math.tan(math.pi * (rng.uniform() - 0.5))
                Fs[i] = min(f, 1.0)
            if frac < 0.60:
                Fs = np.minimum(Fs, 0.7)

            if frac < 0.20:
                Fw = 0.7 * Fs
            elif frac < 0.40:
                Fw = 0.8 * Fs
            else:
                Fw = 1.2 * Fs

            # --- mutation: DE/current-to-pBest-w/1 --------------------------
            n_arch = archive.shape[0]
            pb_idx = order[rng.integers(0, p_num, NP)]
            r1 = np.empty(NP, dtype=int)
            r2 = np.empty(NP, dtype=int)   # index into pop (0..NP-1) or archive (NP..)
            for i in range(NP):
                a = int(rng.integers(0, NP))
                while a == i:
                    a = int(rng.integers(0, NP))
                b = int(rng.integers(0, NP + n_arch))
                while b == i or b == a:
                    b = int(rng.integers(0, NP + n_arch))
                r1[i], r2[i] = a, b
            X_r2 = np.where((r2 < NP)[:, None], X[np.minimum(r2, NP - 1)],
                            archive[np.maximum(r2 - NP, 0)] if n_arch
                            else X[np.minimum(r2, NP - 1)])
            V = (X + Fw[:, None] * (X[pb_idx] - X)
                 + Fs[:, None] * (X[r1] - X_r2))

            # --- binomial crossover with guaranteed jrand -------------------
            cross = rng.uniform(size=(NP, D)) <= CR[:, None]
            jrand = rng.integers(0, D, NP)
            cross[np.arange(NP), jrand] = True
            U = np.where(cross, V, X)

            # --- bound repair: midpoint toward the parent -------------------
            low = U < lb[None, :]
            high = U > ub[None, :]
            U = np.where(low, (lb[None, :] + X) / 2.0, U)
            U = np.where(high, (ub[None, :] + X) / 2.0, U)

            # --- evaluate + select ------------------------------------------
            F_trial = evalf(U)
            nfes += NP

            improved = F_trial < F_pop           # strict: success sets + archive
            accepted = F_trial <= F_pop          # ties also replace the parent
            if np.any(improved):
                S_F = Fs[improved]
                S_CR = CR[improved]
                d_f = np.abs(F_pop[improved] - F_trial[improved])
                # archive the replaced parents (random overwrite when full)
                for parent in X[improved]:
                    if archive.shape[0] < arc_size:
                        archive = np.vstack([archive, parent[None, :]])
                    elif arc_size > 0:
                        archive[int(rng.integers(0, arc_size))] = parent

                # --- memory update (weighted Lehmer + iL-SHADE averaging) --
                wsum = float(np.sum(d_f))
                if wsum > 0.0:
                    w = d_f / wsum
                    new_F = _lehmer(S_F, w)
                    M_F[k_pos] = (new_F + M_F[k_pos]) / 2.0
                    if M_CR[k_pos] < 0.0 or float(np.max(S_CR)) <= 0.0:
                        M_CR[k_pos] = -1.0          # terminal ⊥ (sticky)
                    else:
                        new_CR = _lehmer(S_CR, w)
                        M_CR[k_pos] = (new_CR + M_CR[k_pos]) / 2.0
                    k_pos = (k_pos + 1) % H

            X = np.where(accepted[:, None], U, X)
            F_pop = np.where(accepted, F_trial, F_pop)

            # --- LPSR --------------------------------------------------------
            np_next = int(round((cfg.np_min - np_init) / max_nfes * nfes
                          + np_init))
            np_next = max(np_next, cfg.np_min)
            if np_next < NP:
                keep = np.argsort(F_pop, kind="stable")[:np_next]
                X = X[keep]
                F_pop = F_pop[keep]
                NP = np_next
                arc_size = int(round(cfg.arc_rate * NP))
                if archive.shape[0] > arc_size:
                    sel = rng.permutation(archive.shape[0])[:arc_size]
                    archive = archive[sel]

            emit(X, F_pop)

        total = evalf.n_evals

    k = int(np.argmin(F_pop))
    return JSOResult(x=X[k].copy(), fun=float(F_pop[k]), nit=it,
                     converged=True, history=history, n_evals=total)
