"""Platform-aware multiprocessing helpers.

GLADE parallelises both Differential Evolution (via scipy's worker pool) and
MCMC (via an emcee pool). Historically both used a ``fork`` start method, which
is the fast, zero-copy default on Linux.

On macOS forking a process that has *already* loaded native libraries (the
Accelerate/vecLib BLAS that NumPy uses, plus the compiled ``glafic`` extension)
is unsafe: the CoreFoundation fork-safety guard aborts the worker
(``__THE_PROCESS_HAS_FORKED_AND_YOU_CANNOT_USE_THIS_COREFOUNDATION_...``). Since
Python 3.8 Apple ships ``spawn`` as the macOS default for exactly this reason.

GLADE's pool payloads (:class:`~core.optimize.objective.Objective`,
:class:`~core.mcmc.log_prob.LogProbability`,
:class:`~core.optimize.extend.ExtendObjective`) are all pickle-safe and rebuild
their engine lazily per worker, so ``spawn`` is *functionally identical* to
``fork`` -- only worker start-up is marginally slower. Results are unchanged
(DE uses a fixed seed with ``updating="deferred"``; emcee's RNG/walker moves
live in the parent and the pool only fans out likelihood evaluations).

Linux behaviour is intentionally left exactly as before (``fork``).
"""
from __future__ import annotations

import multiprocessing as mp
import sys


def preferred_start_method() -> str:
    """The multiprocessing start method GLADE should use on this platform.

    ``"fork"`` on Linux (unchanged historical default); ``"spawn"`` everywhere
    else (macOS, where fork-after-native-load is unsafe).
    """
    return "fork" if sys.platform.startswith("linux") else "spawn"


def get_pool_context():
    """A :mod:`multiprocessing` context using :func:`preferred_start_method`."""
    return mp.get_context(preferred_start_method())
