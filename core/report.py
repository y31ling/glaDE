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
from .plot import (
    plot_extend_result,
    plot_triptych,
    read_critical_curves,
    subhalo_label,
)


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


def mag_at_points(scene, points, prefix: str):
    """Model magnification ``mu = 1/det(A)`` at each image-plane point, via
    ``glafic.calcimage``.

    ``points`` are ``(x, y)`` pairs in the model/glafic frame (arcsec). Returns a
    list of signed magnifications aligned with *points*, or ``None`` if glafic is
    unavailable or the call fails (the caller then falls back to the predicted-
    image magnifications). Uses glafic for consistency with the critical curves.

    This feeds the triptych's ``mu@obs`` diagnostic: the model's magnification
    exactly at the *observed* image positions, which differs from the
    magnification at the model's own *predicted* images near critical curves,
    where mu is hypersensitive to position.
    """
    try:
        import glafic  # noqa: PLC0415
    except Exception:
        return None
    try:
        glafic.init(scene.omega, scene.lam, scene.weos, scene.hubble, prefix,
                    scene.xmin, scene.ymin, scene.xmax, scene.ymax,
                    scene.pix_ext, scene.pix_poi, scene.maxlev, verb=0)
        glafic.startup_setnum(len(scene.components), 0, 1)
        for k, comp in enumerate(scene.components, start=1):
            glafic.set_lens(k, comp.glafic_type, comp.z, *_pad7(comp.params))
        glafic.set_point(1, scene.source_z, scene.source_x, scene.source_y)
        glafic.model_init(verb=0)
        mags = []
        for (x, y) in points:
            # calcimage returns [ax, ay, tdelay, kappa, gam1, gam2, ...]
            out = glafic.calcimage(scene.source_z, float(x), float(y), verb=0)
            kappa, gam1, gam2 = out[3], out[4], out[5]
            mu_inv = (1.0 - kappa) ** 2 - (gam1 ** 2 + gam2 ** 2)
            mags.append(1.0 / mu_inv if mu_inv != 0.0 else float("inf"))
        glafic.quit()
        return mags
    except Exception:
        try:
            glafic.quit()
        except Exception:
            pass
        return None


def _subhalo_markers(opt_result: OptResult, center_offset) -> list:
    markers = []
    cfg_comps = opt_result.problem.cfg.components
    scene_comps = opt_result.scene.components
    for i, (gc, sc) in enumerate(zip(cfg_comps, scene_comps), start=1):
        spec = schema.model(gc.type)
        # an index suffix ('3l' / '3s') in the .dat overrides the default
        # classification (schema category, or "optimizable => sub-structure")
        override = getattr(gc, "category_override", None)
        if override is not None:
            is_sub = override == "substructure"
        else:
            is_sub = (spec and spec.category == "substructure") or gc.is_optimizable()
        if not is_sub:
            continue
        if len(sc.params) < 3:
            continue
        x = sc.params[1] + center_offset[0]
        y = sc.params[2] + center_offset[1]
        markers.append((x, y, subhalo_label(i, gc.type, sc.params)))
    return markers


def make_extend_figure(opt_result: OptResult,
                       output_file: str = "extend_result.png",
                       suptitle: str = "GLADE extended-source result") -> str:
    """Render the observed / model / residual figure for an extended-source run.

    Drives glafic on the best-fit scene to render the lensed model image
    (``writeimage``), reads the observed FITS, and plots both plus the residual,
    overlaying the critical curve. Returns ``output_file``.
    """
    if opt_result.mode != "extend" or opt_result.extend_spec is None:
        raise ValueError("make_extend_figure requires an extended-source OptResult")

    from .optimize.backends import _ENGINES
    from .optimize.extend import render_images

    scene = opt_result.scene
    # render via the engine that produced the result (Rhongomyniad mirrors
    # glafic's writeimage); critical curves below still come from glafic.
    engine_key = opt_result.backend if opt_result.backend in _ENGINES else "cpu"
    rendered = render_images(_ENGINES[engine_key](), scene, opt_result.extend_spec,
                             prefix="temp_glade_extfig")
    if rendered is None:
        raise ValueError("could not render the best-fit extended model image")
    model, obs, _nx, _ny = rendered

    extent = (scene.xmin, scene.xmax, scene.ymin, scene.ymax)
    prefix = os.path.join(os.path.dirname(os.path.abspath(output_file)) or ".",
                          "best_ext")
    crit_segments, _caus = compute_crit_curves(scene, prefix)

    comps = None
    if opt_result.extend_components is not None:
        c = opt_result.extend_components
        comps = {"pos": c[0], "flux": c[1], "td": c[2],
                 "pixel": c[4], "loss": float(opt_result.loss)}

    return plot_extend_result(
        obs, model, output_file=output_file, extent=extent,
        crit_segments=crit_segments, suptitle=suptitle, components=comps)


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

    # mu@obs: the model's magnification AT the observed image positions. obs.positions
    # live in the observation frame (predicted = model frame; obs = model +
    # center_offset), so the model-frame coordinate of each observed image is
    # obs.position - center_offset. Falls back to the predicted-image mags
    # (matched_mag) when glafic is unavailable. select_images above guarantees
    # n_pred == obs.n, so this is row-aligned with mu_pred / img_numbers.
    co = np.asarray(obs.center_offset, dtype=float)
    obs_model_xy = np.asarray(obs.positions, dtype=float) - co
    _muobs_prefix = os.path.join(
        os.path.dirname(os.path.abspath(output_file)) or ".", "best_muobs")
    mu_at_obs = mag_at_points(opt_result.scene, obs_model_xy, _muobs_prefix)
    mu_at_obs_pred = (np.asarray(mu_at_obs, dtype=float)
                      if mu_at_obs is not None else matched_mag)
    # the magnification panel follows the loss convention (abs_mag, default
    # True: |mu| bars upward from 0; False keeps the signed values)
    abs_mag = bool(opt_result.problem.cfg.algorithm.get("abs_mag", True))

    return plot_triptych(
        img_numbers=img_numbers,
        delta_pos_mas=delta,
        sigma_pos_mas=obs.pos_sigma_mas,
        mu_obs=obs.magnifications,
        mu_obs_err=obs.mag_errors,
        mu_pred=matched_mag,
        mu_at_obs_pred=mu_at_obs_pred,
        obs_positions_arcsec=obs.positions,
        pred_positions_arcsec=matched_pos,
        crit_segments=crit_segments,
        caus_segments=caus_segments or None,
        subhalos=markers,
        output_file=output_file,
        suptitle=suptitle,
        show_2sigma=show_2sigma,
        abs_mag=abs_mag,
    )


# --------------------------------------------------------------------------- #
# glade_output_<runfolder>.dat: the result as a complete, re-runnable .dat
# --------------------------------------------------------------------------- #

def write_glade_output(opt_result: OptResult, output_dir: str,
                       filename: Optional[str] = None) -> str:
    """Write the best-fit model back out as a COMPLETE glade input file.

    Every ``{lo, hi}`` (and shared-variable) parameter is pinned at its fitted
    value; everything else (cosmology, grid, observations, algorithm settings)
    is carried over verbatim, so the file can be dropped straight back into
    GLADE (e.g. for a Calcimage pass or as the seed of a refined search). The
    file is named ``glade_output_<runfolder>.dat`` after the run directory.

    Returns the written path. Works for both point and extend results.
    """
    from .format.values import Bounds, Fixed, SharedBounds
    from .translate.convert import _num

    problem = opt_result.problem
    cfg = problem.cfg
    x = np.asarray(opt_result.x, dtype=float)
    ov = {d.target: d.to_value(x[i]) for i, d in enumerate(problem.dims)}

    run_name = os.path.basename(os.path.normpath(os.path.abspath(output_dir)))
    if filename is None:
        filename = f"glade_output_{run_name}.dat"

    def render_scalar(name: str, val) -> Optional[str]:
        if isinstance(val, SharedBounds):
            fitted = ov.get(("var", val.name))
            if fitted is None:
                return None
            return f"{name} = {_num(fitted)}"
        if isinstance(val, Bounds):
            for target in (("source", name), ("cosmo", name)):
                if target in ov:
                    return f"{name} = {_num(ov[target])}"
            if ("var", name) in ov:
                return f"{name} = {_num(ov[('var', name)])}"
            return None                     # unused optimizable scalar
        if isinstance(val, Fixed):
            return f"{name} = {_num(val.value)}"
        if isinstance(val, bool):
            return f"{name} = {val}"
        if isinstance(val, (int, float)):
            return f"{name} = {_num(val)}"
        if isinstance(val, str):
            return f"{name} = {val!r}"
        if isinstance(val, list):
            return f"{name} = {val!r}"
        return None

    lines = [
        f"# GLADE result: {run_name}",
        f"# algorithm={getattr(opt_result, 'algorithm', 'DE')}"
        f"  backend={opt_result.backend}  loss={opt_result.loss:.10g}",
        "# Every optimizable parameter is FIXED at its best-fit value; feed",
        "# this file back to GLADE (Calcimage / verification / re-search).",
        "",
    ]
    section_titles = (
        ("cosmology", "cosmology"), ("grid", "grid window"),
        ("redshifts", "redshifts"), ("source", "source"),
        ("obs", "observations"), ("algorithm", "algorithm parameters"),
        ("other", "other"),
    )
    for sec, title in section_titles:
        entries = []
        for name, val in getattr(cfg, sec).items():
            if name == "UnitSetting":
                continue     # values below are already in engine units
            line = render_scalar(name, val)
            if line is not None:
                entries.append(line)
        if entries:
            lines.append(f"# --- {title} ---")
            lines.extend(entries)
            lines.append("")

    def param_value(comp, j, p):
        if isinstance(p, SharedBounds):
            scales = getattr(comp, "unit_scales", None)
            sc = scales[j] if scales is not None else 1.0
            return ov[("var", p.name)] * sc
        if isinstance(p, Bounds):
            return ov[("comp_param", comp.index, j)]
        if isinstance(p, Fixed):
            return p.value
        return 0.0

    lines.append("# --- components (all parameters fixed at best fit) ---")
    for comp in cfg.components:
        if isinstance(comp.z, SharedBounds):
            z = ov[("var", comp.z.name)]
        elif isinstance(comp.z, Bounds):
            z = ov.get(("comp_z", comp.index), float("nan"))
        else:
            z = comp.z.value if isinstance(comp.z, Fixed) else float(comp.z)
        params = [param_value(comp, j, p) for j, p in enumerate(comp.params)]
        suffix = {"lens": "l", "substructure": "s"}.get(
            getattr(comp, "category_override", None) or "", "")
        idx = f"{comp.index}{suffix}"
        nums = ", ".join(_num(v) for v in params)
        lines.append(f"'{comp.name}': ({idx}, '{comp.type}', {_num(z)}, {nums})")
    lines.append("")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path
