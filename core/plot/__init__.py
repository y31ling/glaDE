"""Unified result plotting for GLADE (self-contained; no legacy imports).

Consolidates the duplicated ``plot_paper_style.py`` / ``gpu_postprocess.py``
plotting from the legacy scripts into one module:

* :func:`read_critical_curves` -- parse a glafic ``*_crit.dat`` file.
* :func:`plot_triptych` -- the 3-panel result figure (position residuals,
  magnifications, image plane with critical curves + sub-halo markers).
* :func:`plot_triptych_compare` -- baseline vs optimized variant.
* :func:`plot_iteration_corner` -- legacy-format full-parameter corner of the
  DE population for an iteration frame.
* :func:`plot_iteration` -- single-axes (x, y) population scatter (legacy
  compatibility).
* :func:`subhalo_label` -- per-model marker label formatter.
"""
from __future__ import annotations

from .crit import read_critical_curves
from .extend_fig import plot_extend_result
from .iteration import plot_iteration, plot_iteration_corner
from .labels import subhalo_label
from .triptych import plot_triptych, plot_triptych_compare

__all__ = [
    "read_critical_curves",
    "plot_triptych",
    "plot_triptych_compare",
    "plot_extend_result",
    "plot_iteration",
    "plot_iteration_corner",
    "subhalo_label",
]
