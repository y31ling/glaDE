"""Engine backends.

Both glafic (CPU) and Rhongomyniad (GPU) expose the same module-level API, so a
single :class:`EngineBackend` drives either by importing the right module. The
backend's only job is: given a :class:`Scene`, return the predicted images as a
list of ``(x, y, magnification)`` (or ``None`` if the engine produced none).
"""
from __future__ import annotations

import os
from typing import Optional, Protocol

from .scene import Scene


class Backend(Protocol):
    name: str

    def compute_images(self, scene: Scene) -> Optional[list[tuple[float, float, float]]]:
        ...


def _pad7(params) -> list[float]:
    return ([float(p) for p in params] + [0.0] * 7)[:7]


class EngineBackend:
    """Drives a glafic-compatible engine module (glafic or rhongomyniad)."""

    def __init__(self, module, name: str, *, unique_prefix: bool = True):
        self._m = module
        self.name = name
        self.unique_prefix = unique_prefix

    def compute_images(self, scene: Scene):
        m = self._m
        prefix = f"temp_glade_{os.getpid()}" if self.unique_prefix else "out"
        m.init(scene.omega, scene.lam, scene.weos, scene.hubble, prefix,
               scene.xmin, scene.ymin, scene.xmax, scene.ymax,
               scene.pix_ext, scene.pix_poi, scene.maxlev, verb=0)
        try:
            m.startup_setnum(len(scene.components), 0, 1)
            for k, comp in enumerate(scene.components, start=1):
                m.set_lens(k, comp.glafic_type, comp.z, *_pad7(comp.params))
            m.set_point(1, scene.source_z, scene.source_x, scene.source_y)
            m.model_init(verb=0)
            result = m.point_solve(scene.source_z, scene.source_x, scene.source_y,
                                   verb=0)
        finally:
            m.quit()
        if not result:
            return None
        return [(float(im[0]), float(im[1]), float(im[2])) for im in result]


def _import_glafic():
    import multiprocessing
    if multiprocessing.get_start_method(allow_none=True) != "fork":
        try:
            multiprocessing.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    import glafic  # noqa: PLC0415
    return glafic


def _import_rhongomyniad():
    import rhongomyniad  # noqa: PLC0415
    return rhongomyniad


# user-facing backend name -> engine importer
_ENGINES = {
    "cpu": _import_glafic,
    "glafic": _import_glafic,
    "gpu": _import_rhongomyniad,
}


def make_backend(name: str) -> EngineBackend:
    """Construct an :class:`EngineBackend` for ``'cpu' | 'glafic' | 'gpu'``."""
    key = name.lower()
    if key not in _ENGINES:
        raise ValueError(f"unknown backend '{name}'; expected one of "
                         f"{sorted(_ENGINES)}")
    module = _ENGINES[key]()
    return EngineBackend(module, key)
