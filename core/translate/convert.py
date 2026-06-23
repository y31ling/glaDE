"""glafic <-> glade conversion logic."""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

from ..format import schema
from ..format.config import GladeConfig
from ..format.values import Bounds, Fixed, SharedBounds, Unfilled
import os

from .glafic_io import (
    GlaficLens,
    GlaficModel,
    GlaficObs,
    parse_glafic_any,
    parse_glafic_input,
    render_glafic_input,
    render_glafic_obs,
    render_glafic_point_constraint,
    render_glafic_prior,
)


def _num(v: float) -> str:
    """Round-trip-exact number formatting for glade .dat output.

    ``repr`` of a float is the shortest string that parses back to the same
    value, so glafic -> glade -> glafic preserves parameters faithfully.
    """
    v = float(v)
    if v == 0:
        return "0.0"
    return repr(v)


def _midpoint(pv, is_mass: bool) -> float:
    if isinstance(pv, Fixed):
        return pv.value
    if isinstance(pv, Bounds):
        if is_mass and pv.lo > 0 and pv.hi > 0:
            return math.sqrt(pv.lo * pv.hi)          # geometric mean for masses
        return 0.5 * (pv.lo + pv.hi)                  # arithmetic mean otherwise
    if isinstance(pv, Unfilled):
        return 0.0
    if isinstance(pv, (int, float)):
        return float(pv)
    return 0.0


# --------------------------------------------------------------------------- #
# glafic -> glade
# --------------------------------------------------------------------------- #

def glafic_to_glade(glafic_text: str) -> dict[str, str]:
    """Translate a glafic input file *or* python driver script into glade ``.dat``.

    Returns ``{"model": <text>, "obs": <text or "">}`` -- model and observation
    data go into separate documents. Extended-source scripts (set_extend /
    readobs_extend) produce an extend-mode glade config.
    """
    model = parse_glafic_any(glafic_text)
    if model.extends or model.extended_file:
        return _render_glade_extend(model)
    return {
        "model": _render_glade_model(model),
        "obs": _render_glade_obs(model.obs) if model.obs else "",
    }


def _opt_param(val: float, opt_flag: bool) -> str:
    """A locked value, or a degenerate ``{v, v}`` for an optimizable param.

    Per the import convention, an optimizable glafic parameter becomes a
    degenerate ``{v, v}`` bound the user widens into a real search range.
    """
    return ("{%s, %s}" % (_num(val), _num(val))) if opt_flag else _num(val)


def _render_glade_extend(model: GlaficModel) -> dict[str, str]:
    """Render an extended-source glafic model into glade model + obs documents."""
    p = model.primary
    sec = dict(model.secondary)
    hvary = int(sec.pop("hvary", 0) or 0)          # -> optimizable hubble
    # ran_seed/hvary aside, forward only the chi2/noise engine settings glade uses
    forward = ("chi2_splane", "chi2_checknimg", "chi2_restart", "chi2_usemag",
               "ran_seed", "obs_gain", "obs_ncomb", "obs_readnoise", "flag_extnorm")

    src_z = model.source_z if model.source_z is not None else 1.0
    lens_z_val = None
    if model.lenses:
        lens_z_val = Counter(round(ll.z, 9) for ll in model.lenses).most_common(1)[0][0]

    # ---- model document -----------------------------------------------------
    m: list[str] = [
        "# Translated from a glafic extended-source script by GLADE.",
        "# Optimizable parameters were written as degenerate {v, v}; WIDEN them",
        "# into real search ranges before running a DE fit.", "",
        "# --- constants ---",
        f"omega = {p.get('omega', 0.3)}",
        f"lambda_cosmo = {p.get('lambda', 0.7)}",
        f"weos = {p.get('weos', -1.0)}",
    ]
    hub = float(p.get("hubble", 0.7))
    m.append(f"hubble = {{{_num(hub)}, {_num(hub)}}}" if hvary else f"hubble = {_num(hub)}")
    for k in ("xmin", "ymin", "xmax", "ymax", "pix_ext", "pix_poi", "maxlev"):
        if k in p:
            m.append(f"{k} = {p[k]}")
    if lens_z_val is not None:
        m.append(f"lens_z = {_num(lens_z_val)}")
    m.append(f"source_z = {_num(src_z)}")

    # source (point) position: free flags from setopt_point
    px = model.source_x if model.source_x is not None else 0.0
    py = model.source_y if model.source_y is not None else 0.0
    po = model.point_opt or [0, 0, 0]
    if model.source_x is not None:
        m += ["", "# --- point source (SN); {v,v} = solved by glafic ---",
              f"source_x = {_opt_param(px, len(po) > 1 and po[1] == 1)}",
              f"source_y = {_opt_param(py, len(po) > 2 and po[2] == 1)}"]

    m += ["", "# --- lens / sub-structure components ---"]
    counts: Counter = Counter()
    idx = 0
    for lens in model.lenses:
        idx += 1
        counts[lens.type] += 1
        spec = schema.model(lens.type)
        z_tok = ("lens_z" if (lens_z_val is not None and round(lens.z, 9) == lens_z_val)
                 else _num(lens.z))
        parts = [str(idx), repr(lens.type), z_tok]
        for j, val in enumerate(lens.params):
            opt = bool(lens.opt and j + 1 < len(lens.opt) and lens.opt[j + 1] == 1)
            parts.append(_opt_param(val, opt))
        m.append(f"'{lens.type}{counts[lens.type]}': ({', '.join(parts)})")

    m += ["", "# --- extended-source components (host galaxy) ---"]
    ecounts: Counter = Counter()
    for ext in model.extends:
        idx += 1
        key = f"ext{ext.type}"                      # e.g. 'extsersic'
        if schema.model(key) is None:
            key = ext.type                          # fall back to raw name
        ecounts[key] += 1
        parts = [str(idx), repr(key), "source_z"]
        for j, val in enumerate(ext.params):
            opt = bool(ext.opt and j + 1 < len(ext.opt) and ext.opt[j + 1] == 1)
            parts.append(_opt_param(val, opt))
        m.append(f"'{key}{ecounts[key]}': ({', '.join(parts)})")

    m += ["", "# --- per-component loss weights (all 1.0 = exact glafic c2calc) ---",
          "W_POS = 1.0", "W_FLUX = 1.0", "W_TD = 1.0", "W_EXT = 1.0", "W_PRIOR = 1.0",
          "# per-missing-image penalty (0 = glafic's flat wrong-count reject)",
          "missing_img_penalty = 0.0", ""]

    # ---- obs document -------------------------------------------------------
    o: list[str] = ["# Extended-source observation data translated by GLADE.",
                    "# File paths are basenames; place the files next to this .dat.", ""]
    if model.extended_file:
        o.append(f"extended_file = {_path(model.extended_file)}")
    if model.mask_file:
        o.append(f"extend_mask_file = {_path(model.mask_file)}")
    if model.noise_file:
        o.append(f"noise_file = {_path(model.noise_file)}")
    if model.constraint_file:
        o.append(f"constraint_file = {_path(model.constraint_file)}")
    if model.prior_file:
        o.append(f"prior_file = {_path(model.prior_file)}")
    o.append("")
    o.append("# --- glafic chi2 / noise engine settings ---")
    for k in forward:
        if k in sec:
            v = sec[k]
            o.append(f"{k} = {int(v) if float(v).is_integer() else v}")
    o.append("")
    return {"model": "\n".join(m), "obs": "\n".join(o)}


def _path(p: str) -> str:
    """Render a file path as a quoted basename (place files next to the .dat)."""
    return repr(os.path.basename(str(p)))


def _render_glade_model(model: GlaficModel) -> str:
    p = model.primary
    out: list[str] = ["# Translated from glafic by GLADE", "",
                      "# --- constants ---"]
    out.append(f"omega = {p.get('omega', 0.3)}")
    out.append(f"lambda_cosmo = {p.get('lambda', 0.7)}")
    out.append(f"weos = {p.get('weos', -1.0)}")
    out.append(f"hubble = {p.get('hubble', 0.7)}")
    for k in ("xmin", "ymin", "xmax", "ymax", "pix_ext", "pix_poi", "maxlev"):
        if k in p:
            out.append(f"{k} = {p[k]}")

    # define lens_z from the most common lens redshift for reuse
    lens_z_val: Optional[float] = None
    if model.lenses:
        lens_z_val = Counter(round(ll.z, 9) for ll in model.lenses).most_common(1)[0][0]
        out.append(f"lens_z = {_num(lens_z_val)}")
    if model.source_z is not None:
        out.append(f"source_z = {_num(model.source_z)}")

    if model.source_x is not None:
        out.append("")
        out.append("# --- source (point) ---")
        out.append(f"source_x = {_num(model.source_x)}")
        out.append(f"source_y = {_num(model.source_y)}")

    out.append("")
    out.append("# --- lens / sub-structure components ---")
    counts: Counter = Counter()
    for k, lens in enumerate(model.lenses, start=1):
        counts[lens.type] += 1
        name = f"{lens.type}{counts[lens.type]}"
        spec = schema.model(lens.type)
        z_tok = "lens_z" if (lens_z_val is not None
                             and round(lens.z, 9) == lens_z_val) else _num(lens.z)
        parts = [str(k), repr(lens.type), z_tok]
        for j, val in enumerate(lens.params):
            optimize = bool(lens.opt and j + 1 < len(lens.opt) and lens.opt[j + 1] == 1)
            if optimize:
                parts.append("{%s, %s}" % (_num(val), _num(val)))
            else:
                parts.append(_num(val))
        out.append(f"'{name}': ({', '.join(parts)})")
    out.append("")
    return "\n".join(out)


def _render_glade_obs(obs: GlaficObs) -> str:
    # glafic observation positions are arcsec; glade lists are milliarcsec.
    pos = [[round(x * 1000.0, 6), round(y * 1000.0, 6)]
           for (x, y, *_rest) in obs.images]
    mags = [m for (_x, _y, m, *_r) in obs.images]
    spos = [round(s * 1000.0, 6) for (_x, _y, _m, s, *_r) in obs.images]
    smag = [sm for (_x, _y, _m, _s, sm, _f) in obs.images]

    out = ["# Translated observation data from glafic by GLADE", ""]
    out.append(f"obs_positions_mas_list = {pos}")
    out.append(f"obs_magnifications_list = {mags}")
    out.append(f"obs_mag_errors_list = {smag}")
    out.append(f"obs_pos_sigma_mas_list = {spos}")
    out.append("center_offset_x = 0.0")
    out.append("center_offset_y = 0.0")
    out.append("obs_x_flip = False")
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# glade -> glafic
# --------------------------------------------------------------------------- #

def glade_to_glafic(cfg: GladeConfig, command: bool = True,
                    base_name: str = "glade_export") -> dict:
    """Translate a merged glade config into a runnable glafic input bundle.

    Returns ``{"model", "obs", "constraint", "prior", "optimize"}``:

    * ``model``      -- the glafic ``.input`` text. ``{lo, hi}`` parameters
      collapse to a representative starting value (geometric mean for mass-like,
      arithmetic otherwise); when ANY ``{lo, hi}`` is present the model also gets
      a ``start_setopt`` matrix and an ``optimize`` command, plus
      ``readobs_point``/``parprior`` lines referencing the files below.
    * ``obs``        -- legacy ``start_obs`` round-trip artifact (always emitted
      when observations exist; glafic itself reads ``constraint`` instead).
    * ``constraint`` -- ``readobs_point`` file (only when optimizing + obs given).
    * ``prior``      -- ``parprior`` ``range`` rows (only when optimizing).
    * ``optimize``   -- whether the model fits via glafic's amoeba.

    The ``base_name`` ties the in-model filenames to what the caller writes:
    ``readobs_point <base_name>_obs.dat`` and ``parprior <base_name>_prior.dat``.
    """
    model = _glade_to_model(cfg)
    has_obs = "obs_positions_mas_list" in cfg.obs
    do_opt = bool(model.ranges) or bool(model.hvary)

    obs_text = render_glafic_obs(_glade_to_obs(cfg)) if has_obs else ""
    constraint_text = ""
    prior_text = ""
    if do_opt:
        model.optimize = True
        if model.ranges or model.hubble_range:
            prior_text = render_glafic_prior(model.ranges, model.hubble_range,
                                             model.matches)
            model.prior_file_name = f"{base_name}_prior.dat"
        if has_obs:
            constraint_text = render_glafic_point_constraint(
                _glade_to_obs(cfg), center_offset=_center_offset(cfg))
            model.point_constraint_file = f"{base_name}_obs.dat"

    return {"model": render_glafic_input(model, command=command),
            "obs": obs_text, "constraint": constraint_text,
            "prior": prior_text, "optimize": do_opt}


def _center_offset(cfg: GladeConfig) -> tuple:
    """Observation-frame center offset (x-flip applied), matching scene.build_obs."""
    obs = cfg.obs
    x_sign = -1.0 if obs.get("obs_x_flip", False) else 1.0
    return (x_sign * _scalar(obs, "center_offset_x", 0.0),
            _scalar(obs, "center_offset_y", 0.0))


def _scalar(section: dict, name: str, default: float) -> float:
    v = section.get(name, default)
    return float(v) if isinstance(v, (int, float)) else float(default)


def _glade_to_model(cfg: GladeConfig) -> GlaficModel:
    cos, grid, rs = cfg.cosmology, cfg.grid, cfg.redshifts
    hub = cos.get("hubble")
    primary = {
        "omega": _scalar(cos, "omega", 0.3),
        "lambda": _scalar(cos, "lambda_cosmo", 0.7),
        "weos": _scalar(cos, "weos", -1.0),
        "hubble": _midpoint(hub, False) if isinstance(hub, Bounds)
        else _scalar(cos, "hubble", 0.7),
        "xmin": _scalar(grid, "xmin", -0.5),
        "ymin": _scalar(grid, "ymin", -0.5),
        "xmax": _scalar(grid, "xmax", 0.5),
        "ymax": _scalar(grid, "ymax", 0.5),
        "pix_ext": _scalar(grid, "pix_ext", 0.01),
        "pix_poi": _scalar(grid, "pix_poi", 0.2),
        "maxlev": int(_scalar(grid, "maxlev", 5)),
    }

    # ranges: parprior `range` rows for every {lo, hi}. param_no 1 = z, 2.. = p1..
    # matches: parprior `match` rows tying every extra reference of a shared
    # user-variable to its first occurrence (glafic has no shared-dimension
    # concept, but `match lens i j ii jj 1.0 0.0` is a hard 1:1 tie -> the GLADE
    # "one shared search dimension" semantics are preserved exactly).
    ranges: list = []
    matches: list = []
    var_primary: dict = {}                            # var name -> (lens_id, param_no)

    def _classify(pv, ci, opt, opt_index):
        """Flag an optimizable value: free dim + range, a tied match, or nothing."""
        param_no = opt_index + 1                       # glafic param numbering (z=1)
        if isinstance(pv, SharedBounds):
            if pv.name in var_primary:                 # tie to the first occurrence
                pi, pj = var_primary[pv.name]
                matches.append(("lens", ci, param_no, pi, pj))
            else:                                      # first occurrence is the free dim
                var_primary[pv.name] = (ci, param_no)
                opt[opt_index] = 1
                ranges.append(("lens", ci, param_no, pv.lo, pv.hi))
        elif isinstance(pv, Bounds):
            opt[opt_index] = 1
            ranges.append(("lens", ci, param_no, pv.lo, pv.hi))

    lenses: list[GlaficLens] = []
    for ci, comp in enumerate(cfg.components, start=1):
        spec = schema.model(comp.type)
        gtype = spec.glafic_key if spec else comp.type
        opt = [0] * 8                                 # [z, p1..p7]
        _classify(comp.z, ci, opt, 0)
        z = comp.z.value if isinstance(comp.z, Fixed) else _midpoint(comp.z, False)
        params: list[float] = []
        for j, pv in enumerate(comp.params):
            is_mass = bool(spec and j < len(spec.params) and spec.params[j].is_mass)
            params.append(_midpoint(pv, is_mass))
            if j < 7:
                _classify(pv, ci, opt, j + 1)
        params = (params + [0.0] * 7)[:7]
        lenses.append(GlaficLens(type=gtype, z=z, params=params, opt=opt))

    src_x = cfg.source.get("source_x")
    src_y = cfg.source.get("source_y")
    point_opt = [0, 0, 0]                             # [zs, xs, ys]
    if isinstance(src_x, Bounds):
        point_opt[1] = 1
        ranges.append(("point", 1, 2, src_x.lo, src_x.hi))
    if isinstance(src_y, Bounds):
        point_opt[2] = 1
        ranges.append(("point", 1, 3, src_y.lo, src_y.hi))

    hvary = 1 if isinstance(hub, Bounds) else 0
    hubble_range = (hub.lo, hub.hi) if isinstance(hub, Bounds) else None

    return GlaficModel(
        primary=primary,
        prefix=str(cfg.algorithm.get("OUTPUT_PREFIX", "out")),
        lenses=lenses,
        source_z=_scalar(rs, "source_z", 0.409),
        source_x=_midpoint(src_x, False) if src_x is not None else 0.0,
        source_y=_midpoint(src_y, False) if src_y is not None else 0.0,
        point_opt=point_opt,
        ranges=ranges,
        matches=matches,
        hvary=hvary,
        hubble_range=hubble_range,
    )


def _glade_to_obs(cfg: GladeConfig) -> GlaficObs:
    obs = cfg.obs
    positions_mas = obs["obs_positions_mas_list"]
    mags = obs.get("obs_magnifications_list", [0.0] * len(positions_mas))
    merr = obs.get("obs_mag_errors_list", [0.0] * len(positions_mas))
    spos = obs.get("obs_pos_sigma_mas_list", [0.0] * len(positions_mas))
    x_sign = -1.0 if obs.get("obs_x_flip", False) else 1.0

    images = []
    for i, (xm, ym) in enumerate(positions_mas):
        images.append((
            x_sign * float(xm) / 1000.0,   # arcsec
            float(ym) / 1000.0,
            float(mags[i]) if i < len(mags) else 0.0,
            float(spos[i]) / 1000.0 if i < len(spos) else 0.0,  # arcsec
            float(merr[i]) if i < len(merr) else 0.0,
            0,
        ))
    return GlaficObs(zs=_scalar(cfg.redshifts, "source_z", 0.409), images=images)
