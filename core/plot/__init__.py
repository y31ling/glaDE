"""Unified result plotting for GLADE (self-contained; no legacy imports).

Consolidates the duplicated ``plot_paper_style.py`` / ``gpu_postprocess.py``
plotting from the legacy scripts into one module:

* :func:`read_critical_curves` -- parse a glafic ``*_crit.dat`` file.
* :func:`plot_triptych` -- the 3-panel result figure (position residuals,
  magnifications, image plane with critical curves + sub-halo markers).
* :func:`plot_triptych_compare` -- baseline vs optimized variant.
* :func:`plot_iteration` -- DE population scatter for an iteration frame.
* :func:`subhalo_label` -- per-model marker label formatter.
"""
from __future__ import annotations

from .crit import read_critical_curves
from .iteration import plot_iteration
from .labels import subhalo_label
from .triptych import plot_triptych, plot_triptych_compare

__all__ = [
    "read_critical_curves",
    "plot_triptych",
    "plot_triptych_compare",
    "plot_iteration",
    "subhalo_label",
]
