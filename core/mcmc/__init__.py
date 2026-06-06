"""MCMC sampling for GLADE (emcee), reusing the DE optimization machinery.

The MCMC prior is ALWAYS the ``{lower, upper}`` bounds of the optimizable
parameters (the same :class:`~core.optimize.problem.OptProblem` bounds DE uses;
mass dims are sampled in log10). Two modes:

* MCMC-only   -- walkers initialized uniformly across the bounds; no DE truth.
* DE + MCMC   -- walkers seeded around the DE best (``best_x``); the DE best is
                 overlaid on the corner plot as red truth lines.

The likelihood is ``-0.5 * ml_loss`` (the same weighted chi2 + penalty as the DE
objective), so the posterior is consistent with the DE result.
"""
from __future__ import annotations

from .config import MCMCConfig
from .plots import plot_corner, plot_mcmc, plot_trace
from .runner import MCMCResult, run_mcmc

__all__ = ["MCMCConfig", "MCMCResult", "run_mcmc", "plot_mcmc", "plot_corner", "plot_trace"]
