"""GLADE as an importable library (V0.6).

Make any script able to ``import glade`` the same way ``import glafic`` works,
with no manual ``sys.path`` boilerplate::

    import glade

    cfg, issues = glade.load_config(["constants.dat", "lens.dat"], backend="cpu")
    assert not glade.has_errors(issues)
    result = glade.optimize(cfg, backend="gpu")
    glade.make_triptych(result, glade.build_obs(cfg), "triptych.png")

Importing this package bootstraps ``sys.path`` for the whole GLADE tree (repo
root, glafic's python bindings, Rhongomyniad), so ``import core`` /
``import glafic`` / ``import rhongomyniad`` all work afterwards too.

Importability: the repo root must be on ``sys.path`` — true automatically when
your CWD is the repo root, and from anywhere after ``source env.sh`` (which
exports it in PYTHONPATH).

Heavy dependencies stay lazy: ``import glade`` itself pulls in neither
matplotlib, emcee, torch nor the glafic C extension — those load on first use
of the corresponding function (``make_triptych``, ``run_mcmc``, engines).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.7.0"

# ── path bootstrap: make core / glafic / rhongomyniad importable ────────────
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT),
           str(_ROOT / "glafic2" / "python"),
           str(_ROOT / "Rhongomyniad")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("GLADE_ROOT", str(_ROOT))

# ── eager exports (light: no engines, no plotting) ──────────────────────────
from core.format import (  # noqa: E402
    Bounds,
    Fixed,
    GladeConfig,
    has_errors,
    lint_text,
    load_config,
)
from core.optimize import (  # noqa: E402
    OptProblem,
    OptResult,
    build_obs,
    make_backend,
    optimize,
)

# ── lazy exports (pull in matplotlib / emcee / engines on first use) ───────
_LAZY = {
    "make_triptych":      ("core.report", "make_triptych"),
    "write_glade_output": ("core.report", "write_glade_output"),
    "make_extend_figure": ("core.report", "make_extend_figure"),
    "run_mcmc":           ("core.mcmc", "run_mcmc"),
    "MCMCConfig":         ("core.mcmc", "MCMCConfig"),
    "MCMCResult":         ("core.mcmc", "MCMCResult"),
    "plot_mcmc":          ("core.mcmc", "plot_mcmc"),
    "plot_corner":        ("core.mcmc", "plot_corner"),
    "plot_trace":         ("core.mcmc", "plot_trace"),
    "verify_with_glafic": ("core.verify", "verify_with_glafic"),
    "verify_extend":      ("core.verify", "verify_extend"),
    "reference_check":    ("core.verify", "reference_check"),
}

__all__ = [
    "__version__",
    # format
    "load_config", "lint_text", "has_errors", "GladeConfig", "Bounds", "Fixed",
    # optimize
    "optimize", "OptResult", "build_obs", "OptProblem", "make_backend",
    # lazy: report / mcmc / verify
    *_LAZY.keys(),
    # low-level engine access
    "engine",
]


def __getattr__(name):  # PEP 562
    if name in _LAZY:
        import importlib
        module_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value          # cache for subsequent lookups
        return value
    if name == "core":
        import core
        return core
    raise AttributeError(f"module 'glade' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


def engine(name: str = "cpu"):
    """Return the low-level engine module, glafic-style.

    ``glade.engine("cpu")`` / ``("glafic")`` → the ``glafic`` C-extension module;
    ``glade.engine("gpu")`` / ``("rhongomyniad")`` → the ``rhongomyniad`` module.
    Both expose the same imperative API (``init`` / ``startup_setnum`` /
    ``set_lens`` / ``set_point`` / ``model_init`` / ``point_solve`` / ``quit``).
    """
    import importlib
    key = {"cpu": "glafic", "glafic": "glafic",
           "gpu": "rhongomyniad", "rhongomyniad": "rhongomyniad"}.get(name)
    if key is None:
        raise ValueError(f"unknown engine {name!r} (use 'cpu'/'glafic' or 'gpu'/'rhongomyniad')")
    return importlib.import_module(key)
