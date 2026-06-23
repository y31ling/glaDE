"""Translate between glafic input files and glade ``.dat`` files.

* ``glafic_to_glade`` -- import a glafic model (+ optional opt matrix / obs) and
  emit glade ``.dat`` text, splitting model and observation data into separate
  documents. A parameter flagged for optimization in the opt matrix becomes
  ``name = {v, v}`` so the user can widen the bounds by hand.
* ``glade_to_glafic`` -- export a glade config to a runnable glafic input file
  plus separate observation / prior files. ``{lo, hi}`` collapses to a single
  starting value (geometric mean for mass-like parameters, so ``{1E5, 1E7}`` ->
  ``1E6``; arithmetic mean otherwise). When ANY ``{lo, hi}`` is present the input
  also gains a ``start_setopt`` matrix and an ``optimize`` command (glafic's
  amoeba) plus the matching ``readobs_point`` / ``parprior`` files.
"""
from __future__ import annotations

from .convert import glade_to_glafic, glafic_to_glade
from .glafic_io import (
    GlaficLens,
    GlaficModel,
    GlaficObs,
    looks_like_glafic_input,
    parse_glafic_input,
    render_glafic_input,
    render_glafic_obs,
    render_glafic_point_constraint,
    render_glafic_prior,
)

__all__ = [
    "glafic_to_glade", "glade_to_glafic",
    "GlaficModel", "GlaficLens", "GlaficObs",
    "parse_glafic_input", "render_glafic_input", "render_glafic_obs",
    "render_glafic_point_constraint", "render_glafic_prior",
    "looks_like_glafic_input",
]
