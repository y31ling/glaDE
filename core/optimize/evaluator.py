"""Population evaluation for the non-DE optimizers (BIPOP-CMA-ES, jSO).

scipy's DifferentialEvolutionSolver brings its own worker pool / vectorized
dispatch; the self-implemented optimizers use this small helper instead so the
three evaluation modes look identical from the algorithm loop:

* ``vectorized`` — the objective accepts an ``(ndim, N)`` array and returns
  ``(N,)`` losses (the batched GPU objective);
* ``workers != 1`` — a platform-aware process pool maps the picklable
  per-candidate objective over the population (the CPU/glafic path; workers
  follow the DE convention: -1/0 = all cores);
* otherwise — a plain serial loop.
"""
from __future__ import annotations

import numpy as np


class PopulationEvaluator:
    """Evaluate an ``(N, ndim)`` candidate matrix -> ``(N,)`` losses."""

    def __init__(self, objective, vectorized: bool = False, workers: int = 1):
        self.objective = objective
        self.vectorized = bool(vectorized)
        self.workers = int(workers)
        self._pool = None
        self.n_evals = 0

    def _ensure_pool(self):
        if self._pool is None:
            import multiprocessing as mp

            from ..parallel import get_pool_context
            n = mp.cpu_count() if self.workers in (-1, 0) else self.workers
            self._pool = get_pool_context().Pool(n)
        return self._pool

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        self.n_evals += X.shape[0]
        if self.vectorized:
            return np.asarray(self.objective(X.T), dtype=float).reshape(-1)
        if self.workers != 1:
            pool = self._ensure_pool()
            return np.asarray(pool.map(self.objective, list(X)), dtype=float)
        return np.asarray([self.objective(x) for x in X], dtype=float)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
