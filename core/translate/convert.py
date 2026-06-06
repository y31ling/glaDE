"""glafic <-> glade conversion logic."""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

from ..format import schema
from ..format.config import GladeConfig
from ..format.values import Bounds, Fixed, Unfilled
from .glafic_io import (
    GlaficLens,
    GlaficModel,
    GlaficObs,
    parse_glafic_input,
    render_glafic_input,
    render_glafic_obs,
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
    """Translate a glafic input file into glade ``.dat`` text.

    Returns ``{"model": <text>, "obs": <text or "">}`` -- model and observation
    data go into separate documents.
    """
    model = parse_glafic_input(glafic_text)
    return {
        "model": _render_glade_model(model),
        "obs": _render_glade_obs(model.obs) if model.obs else "",
    }


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

def glade_to_glafic(cfg: GladeConfig, command: bool = True) -> dict[str, str]:
    """Translate a merged glade config into glafic input + observation text.

    Returns ``{"model": <input text>, "obs": <obs text or "">}``. ``{lo, hi}``
    parameters collapse to a representative value (geometric mean for mass-like,
    arithmetic otherwise).
    """
    model = _glade_to_model(cfg)
    obs_text = ""
    if "obs_positions_mas_list" in cfg.obs:
        obs_text = render_glafic_obs(_glade_to_obs(cfg))
    return {"model": render_glafic_input(model, command=command), "obs": obs_text}


def _scalar(section: dict, name: str, default: float) -> float:
    v = section.get(name, default)
    return float(v) if isinstance(v, (int, float)) else float(default)


def _glade_to_model(cfg: GladeConfig) -> GlaficModel:
    cos, grid, rs = cfg.cosmology, cfg.grid, cfg.redshifts
    primary = {
        "omega": _scalar(cos, "omega", 0.3),
        "lambda": _scalar(cos, "lambda_cosmo", 0.7),
        "weos": _scalar(cos, "weos", -1.0),
        "hubble": _scalar(cos, "hubble", 0.7),
        "xmin": _scalar(grid, "xmin", -0.5),
        "ymin": _scalar(grid, "ymin", -0.5),
        "xmax": _scalar(grid, "xmax", 0.5),
        "ymax": _scalar(grid, "ymax", 0.5),
        "pix_ext": _scalar(grid, "pix_ext", 0.01),
        "pix_poi": _scalar(grid, "pix_poi", 0.2),
        "maxlev": int(_scalar(grid, "maxlev", 5)),
    }

    lenses: list[GlaficLens] = []
    for comp in cfg.components:
        spec = schema.model(comp.type)
        gtype = spec.glafic_key if spec else comp.type
        z = comp.z.value if isinstance(comp.z, Fixed) else _midpoint(comp.z, False)
        params: list[float] = []
        for j, pv in enumerate(comp.params):
            is_mass = bool(spec and j < len(spec.params) and spec.params[j].is_mass)
            params.append(_midpoint(pv, is_mass))
        params = (params + [0.0] * 7)[:7]
        lenses.append(GlaficLens(type=gtype, z=z, params=params))

    src_x = cfg.source.get("source_x")
    src_y = cfg.source.get("source_y")
    return GlaficModel(
        primary=primary,
        prefix=str(cfg.algorithm.get("OUTPUT_PREFIX", "out")),
        lenses=lenses,
        source_z=_scalar(rs, "source_z", 0.409),
        source_x=_midpoint(src_x, False) if src_x is not None else 0.0,
        source_y=_midpoint(src_y, False) if src_y is not None else 0.0,
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
