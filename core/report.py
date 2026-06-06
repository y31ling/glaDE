"""Glue between the DE-core result and the plotting module.

Produces the standard triptych for an :class:`~core.optimize.OptResult`: it
recomputes the best-fit images through a backend, gathers critical curves and
sub-halo markers, and renders the figure. Imports both ``optimize`` and ``plot``
but neither imports this, so package imports stay light.
"""
from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np

from .format import schema
from .optimize.backends import Backend, make_backend
from .optimize.matching import match_images, select_images
from .optimize.runner import OptResult
from .optimize.scene import ObsData
from .optimize.backends import _pad7
from .plot import plot_triptych, read_critical_curves, subhalo_label


def compute_crit_curves(scene, prefix: str):
    """Drive glafic.writecrit for *scene* and read back the critical/caustic
    segments. Always uses glafic (the engine that emits crit curves); returns
    ``([], [])`` if glafic is unavailable or the call fails."""
    try:
        import glafic  # noqa: PLC0415
    except Exception:
        return [], []
    try:
        glafic.init(scene.omega, scene.lam, scene.weos, scene.hubble, prefix,
                    scene.xmin, scene.ymin, scene.xmax, scene.ymax,
                    scene.pix_ext, scene.pix_poi, scene.maxlev, verb=0)
        glafic.startup_setnum(len(scene.components), 0, 1)
        for k, comp in enumerate(scene.components, start=1):
            glafic.set_lens(k, comp.glafic_type, comp.z, *_pad7(comp.params))
        glafic.set_point(1, scene.source_z, scene.source_x, scene.source_y)
        glafic.model_init(verb=0)
        glafic.writecrit(scene.source_z)
        crit, caus = read_critical_curves(f"{prefix}_crit.dat")
        glafic.quit()
        return crit, caus
    except Exception:
        try:
            glafic.quit()
        except Exception:
            pass
        return [], []


def _subhalo_markers(opt_result: OptResult, center_offset) -> list:
    markers = []
    cfg_comps = opt_result.problem.cfg.components
    scene_comps = opt_result.scene.components
    for i, (gc, sc) in enumerate(zip(cfg_comps, scene_comps), start=1):
        spec = schema.model(gc.type)
        is_sub = (spec and spec.category == "substructure") or gc.is_optimizable()
        if not is_sub:
            continue
        if len(sc.params) < 3:
            continue
        x = sc.params[1] + center_offset[0]
        y = sc.params[2] + center_offset[1]
        markers.append((x, y, subhalo_label(i, gc.type, sc.params)))
    return markers


def make_triptych(opt_result: OptResult,
                  obs: ObsData,
                  output_file: str = "triptych.png",
                  backend: Optional[Union[str, Backend]] = None,
                  crit_file: Optional[str] = None,
                  suptitle: str = "GLADE result",
                  show_2sigma: bool = False) -> str:
    """Render the result triptych for ``opt_result``; returns ``output_file``."""
    # Choose the image engine for the figure:
    #  * explicit Backend object (e.g. a test fake) -> used as-is;
    #  * a GPU run -> the SAME batched solver that drove the optimization, so the
    #    figure matches what the optimizer saw;
    #  * otherwise -> glafic (the reference finder, also used for crit curves).
    be_obj = backend if (backend is not None and not isinstance(backend, str)) else None
    images = None
    if be_obj is not None:
        images = be_obj.compute_images(opt_result.scene)
    elif opt_result.backend == "gpu":
        from .optimize.batched import BatchedGPUObjective, can_batch_gpu
        if can_batch_gpu(opt_result.problem.cfg)[0]:
            from .optimize.loss import LossConfig
            bobj = BatchedGPUObjective(opt_result.problem, obs,
                                       LossConfig.from_cfg(opt_result.problem.cfg))
            images = bobj.images_for(opt_result.x)
    if images is None:
        try:
            images = make_backend("cpu").compute_images(opt_result.scene)
        except Exception:  # noqa: BLE001 - glafic unavailable; last-resort run backend
            fb = opt_result.backend if opt_result.backend in ("cpu", "gpu", "glafic") else "cpu"
            images = make_backend(fb).compute_images(opt_result.scene)

    sel = select_images(images, obs.n) if images else None
    if sel is None:
        n = 0 if not images else len(images)
        raise ValueError(
            f"best-fit model produced {n} image(s) (expected {obs.n}); "
            f"triptych skipped")
    pred_pos = np.array([[im[0], im[1]] for im in sel], dtype=float)
    pred_mag = np.array([im[2] for im in sel], dtype=float)

    matched_pos, matched_mag, delta = match_images(
        obs.positions, pred_pos, pred_mag, obs.center_offset)

    if crit_file:
        crit_segments, caus_segments = read_critical_curves(crit_file)
    else:
        prefix = os.path.join(os.path.dirname(os.path.abspath(output_file)) or ".",
                              "best")
        crit_segments, caus_segments = compute_crit_curves(opt_result.scene, prefix)

    markers = _subhalo_markers(opt_result, obs.center_offset)
    img_numbers = list(range(1, obs.n + 1))

    return plot_triptych(
        img_numbers=img_numbers,
        delta_pos_mas=delta,
        sigma_pos_mas=obs.pos_sigma_mas,
        mu_obs=obs.magnifications,
        mu_obs_err=obs.mag_errors,
        mu_pred=matched_mag,
        mu_at_obs_pred=matched_mag,   # stand-in until calcimage-at-obs is wired
        obs_positions_arcsec=obs.positions,
        pred_positions_arcsec=matched_pos,
        crit_segments=crit_segments,
        caus_segments=caus_segments or None,
        subhalos=markers,
        output_file=output_file,
        suptitle=suptitle,
        show_2sigma=show_2sigma,
    )
