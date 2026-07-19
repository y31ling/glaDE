"""Unit profiles (the ``UnitSetting`` key) — V0.7.

GLADE's engine convention (glafic's, and the historical GLADE default) is:

* component masses in **h^-1 Msun** (glafic manual: "All mass scales M is in
  units of h^-1 Msun"; the schema comments claimed plain Msun before V0.7 —
  that was a documentation bug, the numbers always went to glafic unchanged);
* observed image positions and their sigmas in **mas**
  (``obs_positions_mas_list`` / ``obs_pos_sigma_mas_list``);
* component centres and angular scale lengths (re / rc / a / rb / rco ...)
  and the source position in **arcsec**.

A unit profile lets a ``.dat`` be AUTHORED in different units; values are
converted to the engine convention once, at load time, so everything
downstream (optimizers, translate, verification, glade_output) keeps seeing
canonical units. The profile is a small JSON file (default location:
``InputFiles/<name>.units.json``) referenced from the ``.dat`` via::

    UnitSetting = 'myunits'        # or omitted / 'default'

Profile format::

    {"format": "glade-units-v1",
     "units": {"mass": "msun", "obs_pos": "arcsec",
               "comp_pos": "arcsec", "src_pos": "mas"}}

Categories and options (first = engine default):

* ``mass``     : ``hinv_msun`` | ``msun``  — component mass params; ``msun``
  multiplies by ``hubble`` (which must then be a fixed number);
* ``obs_pos``  : ``mas`` | ``arcsec``      — obs positions + position sigmas;
* ``comp_pos`` : ``arcsec`` | ``mas``      — component centres + angular
  lengths (every ``[arcsec]``-documented tuple parameter);
* ``src_pos``  : ``arcsec`` | ``mas``      — ``source_x`` / ``source_y``.

Everything else (velocity dispersions km/s, angles deg, grid arcsec, time
delays days, center_offset arcsec) is fixed.

User-defined ``{lo, hi}`` variables stay DIMENSIONLESS: the insertion slot
decides the unit, so ``xxx = 0.1`` referenced from ``source_x`` under a mas
profile means 0.1 mas while ``lens(..., xxx, ...)`` under an arcsec profile
means 0.1 arcsec. For a shared (optimizable) variable the conversion factor
is applied at scene-injection time per reference site
(``Component.unit_scales``); the shared search dimension itself is untouched.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Sequence

from .diagnostics import ERROR, Issue

PROFILE_SUFFIX = ".units.json"
PROFILE_FORMAT = "glade-units-v1"

# category -> (engine default, allowed options)
CATEGORIES = {
    "mass": ("hinv_msun", ("hinv_msun", "msun")),
    "obs_pos": ("mas", ("mas", "arcsec")),
    "comp_pos": ("arcsec", ("arcsec", "mas")),
    "src_pos": ("arcsec", ("arcsec", "mas")),
}

DEFAULT_UNITS = {k: v[0] for k, v in CATEGORIES.items()}

# fixed-unit rows the UnitSetting dialog displays but cannot change
FIXED_UNITS = {
    "velocity": "km/s",
    "angle": "deg",
    "grid": "arcsec",
    "center_offset": "arcsec",
    "time_delay": "days",
    "redshift": "-",
}


def default_units() -> dict:
    return dict(DEFAULT_UNITS)


def param_unit_kind(pspec) -> Optional[str]:
    """The unit category of one schema ParamSpec: 'mass' | 'ang' | None.

    Driven by the unit tag in the schema description — the single source of
    truth ('[h^-1 Msun]' masses, '[arcsec]' angular lengths; '[km/s]' and
    '[deg]' are fixed-unit and return None).
    """
    desc = getattr(pspec, "desc", "") or ""
    if "Msun" in desc:
        return "mass"
    if "[arcsec]" in desc:
        return "ang"
    return None


def is_default(units: Optional[dict]) -> bool:
    if not units:
        return True
    return all(units.get(k, d) == d for k, d in DEFAULT_UNITS.items())


def resolve_profile(name, search_dirs: Sequence[str]) -> tuple[Optional[dict], list[Issue]]:
    """Resolve a ``UnitSetting`` value to a units dict.

    ``name`` may be None / 'default' (engine defaults), a profile name
    (``<dir>/<name>.units.json`` looked up along ``search_dirs``), or a
    path ending in ``.units.json``. Returns ``(units | None, issues)``;
    ``None`` means default units.
    """
    issues: list[Issue] = []
    if name is None:
        return None, issues
    name = str(name).strip()
    if name in ("", "default"):
        return None, issues

    candidates = []
    if name.endswith(PROFILE_SUFFIX) or name.endswith(".json"):
        if os.path.isabs(name):
            candidates.append(name)
        else:
            candidates.extend(os.path.join(d, name) for d in search_dirs)
    else:
        candidates.extend(os.path.join(d, name + PROFILE_SUFFIX)
                          for d in search_dirs)

    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        issues.append(Issue(
            ERROR, "unit_profile_missing",
            f"UnitSetting = {name!r}: no profile file found "
            f"(looked for {name if name.endswith('.json') else name + PROFILE_SUFFIX} "
            f"in: {', '.join(search_dirs) or '<none>'})"))
        return None, issues

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        issues.append(Issue(
            ERROR, "unit_profile_bad",
            f"UnitSetting profile {path}: cannot read ({exc})"))
        return None, issues

    units = data.get("units", data) if isinstance(data, dict) else None
    if not isinstance(units, dict):
        issues.append(Issue(
            ERROR, "unit_profile_bad",
            f"UnitSetting profile {path}: expected a JSON object with a "
            f"'units' mapping"))
        return None, issues

    out = default_units()
    for key, val in units.items():
        if key not in CATEGORIES:
            issues.append(Issue(
                ERROR, "unit_profile_bad",
                f"UnitSetting profile {path}: unknown category {key!r} "
                f"(expected one of {', '.join(CATEGORIES)})"))
            continue
        allowed = CATEGORIES[key][1]
        if val not in allowed:
            issues.append(Issue(
                ERROR, "unit_profile_bad",
                f"UnitSetting profile {path}: {key} = {val!r} "
                f"(expected one of {', '.join(allowed)})"))
            continue
        out[key] = val
    out["__path__"] = path
    return out, issues


def scale_factors(units: Optional[dict], hubble: float) -> dict:
    """Multiplicative factors taking AUTHORED values to ENGINE values."""
    u = units or DEFAULT_UNITS
    return {
        # authored Msun -> engine h^-1 Msun: M_engine = M_phys * h
        "mass": (float(hubble) if u.get("mass") == "msun" else 1.0),
        # authored arcsec -> engine mas fields
        "obs_pos": (1000.0 if u.get("obs_pos") == "arcsec" else 1.0),
        # authored mas -> engine arcsec
        "comp_pos": (1.0e-3 if u.get("comp_pos") == "mas" else 1.0),
        "src_pos": (1.0e-3 if u.get("src_pos") == "mas" else 1.0),
    }


def unit_labels(units: Optional[dict]) -> dict:
    """Display strings for template comments under a profile."""
    u = units or DEFAULT_UNITS
    return {
        "mass": ("Msun" if u.get("mass") == "msun" else "h^-1 Msun"),
        "obs_pos": ("arcsec" if u.get("obs_pos") == "arcsec" else "mas"),
        "comp_pos": ("mas" if u.get("comp_pos") == "mas" else "arcsec"),
        "src_pos": ("mas" if u.get("src_pos") == "mas" else "arcsec"),
    }
