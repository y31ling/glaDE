"""Translate between glafic input files and glade ``.dat`` files.

* ``glafic_to_glade`` -- import a glafic model (+ optional opt matrix / obs) and
  emit glade ``.dat`` text, splitting model and observation data into separate
  documents. A parameter flagged for optimization in the opt matrix becomes
  ``name = {v, v}`` so the user can widen the bounds by hand.
* ``glade_to_glafic`` -- export a glade config to a runnable glafic input file
  plus a separate observation file. ``{lo, hi}`` collapses to a single value: the
  geometric mean for mass-like parameters (so ``{1E5, 1E7}`` -> ``1E6``), the
  arithmetic mean otherwise.
"""
from __future__ import annotations

from .convert import glade_to_glafic, glafic_to_glade
from .glafic_io import (
    GlaficLens,
    GlaficModel,
    GlaficObs,
    parse_glafic_input,
    render_glafic_input,
    render_glafic_obs,
)

__all__ = [
    "glafic_to_glade", "glade_to_glafic",
    "GlaficModel", "GlaficLens", "GlaficObs",
    "parse_glafic_input", "render_glafic_input", "render_glafic_obs",
]
