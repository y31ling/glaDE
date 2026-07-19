"""High-level merged configuration and the multi-file section-merge."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from . import schema
from .defaults import DEFAULTS
from .diagnostics import ERROR, Issue
from .expr import ExprError, build_env, evaluate
from .values import Bounds, Component, Expr, Fixed, ParsedFile, Ref, SharedBounds


@dataclass
class GladeConfig:
    """A merged, ready-to-validate configuration assembled from one or more files."""

    cosmology: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, Any] = field(default_factory=dict)
    redshifts: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    obs: dict[str, Any] = field(default_factory=dict)
    algorithm: dict[str, Any] = field(default_factory=dict)
    other: dict[str, Any] = field(default_factory=dict)
    components: list[Component] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)
    applied_defaults: list[str] = field(default_factory=list)
    # user-defined {lo, hi} variables actually referenced from component
    # tuples (name -> Bounds); every reference shares ONE search dimension.
    user_vars: dict[str, Any] = field(default_factory=dict)

    _SECTIONS = ("cosmology", "grid", "redshifts", "source", "obs", "algorithm", "other")

    def get(self, name: str, default=None):
        name = schema.SCALAR_ALIASES.get(name, name)
        for sec in self._SECTIONS:
            d = getattr(self, sec)
            if name in d:
                return d[name]
        return default

    def all_scalars(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for sec in self._SECTIONS:
            out.update(getattr(self, sec))
        return out


def _symbol_table(merged_scalars: dict[str, tuple[Any, str, str]]) -> dict[str, float]:
    """Numeric scalars usable as references in tuples / nested lists.

    Keyed by canonical name *and* any alias (so a reference to ``lambda``
    resolves to the value stored as ``lambda_cosmo``).
    """
    reverse: dict[str, list[str]] = {}
    for alias, canon in schema.SCALAR_ALIASES.items():
        reverse.setdefault(canon, []).append(alias)

    table: dict[str, float] = {}
    for canon, entry in merged_scalars.items():
        val = entry[0]
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            table[canon] = float(val)
            for alias in reverse.get(canon, ()):
                table[alias] = float(val)
    return table


def _resolve_nested(val: Any, symbols: dict[str, float]) -> Any:
    """Resolve ``Ref`` objects buried inside list/tuple scalar values.

    Known references become floats; unknown ones are left as ``Ref`` for the
    validation layer to report.
    """
    if isinstance(val, Ref):
        return symbols.get(val.name, val)
    if isinstance(val, list):
        return [_resolve_nested(v, symbols) for v in val]
    if isinstance(val, tuple):
        return tuple(_resolve_nested(v, symbols) for v in val)
    return val


def _scale_value(v, f: float):
    """Scale a scalar section value (float or literal Bounds) by ``f``."""
    if isinstance(v, SharedBounds):
        return v                       # dimensionless by definition
    if isinstance(v, Bounds):
        return Bounds(v.lo * f, v.hi * f)
    if isinstance(v, Fixed):
        return Fixed(v.value * f)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v) * f
    return v


def _scale_nested(v, f: float):
    if isinstance(v, list):
        return [_scale_nested(x, f) for x in v]
    if isinstance(v, tuple):
        return tuple(_scale_nested(x, f) for x in v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v) * f
    return v


def _convert_scalar_sections(cfg: GladeConfig, factors: dict) -> None:
    """Convert authored obs arrays / source scalars to engine units."""
    fo = factors["obs_pos"]
    if fo != 1.0:
        for key in ("obs_positions_mas_list", "obs_pos_sigma_mas_list"):
            if key in cfg.obs:
                cfg.obs[key] = _scale_nested(cfg.obs[key], fo)
    fs = factors["src_pos"]
    if fs != 1.0:
        for key in ("source_x", "source_y"):
            if key in cfg.source:
                cfg.source[key] = _scale_value(cfg.source[key], fs)


def _apply_component_units(comp: Component, orig_params: list,
                           factors: dict) -> None:
    """Convert one component's parameters to engine units.

    Literal Fixed/Bounds values were authored in the profile's units and are
    scaled in place; SHARED-variable references stay dimensionless and record
    their slot factor in ``comp.unit_scales`` (applied at injection); values
    that came from deferred obs-position expressions are already engine-frame
    and are left alone.
    """
    spec = schema.model(comp.type)
    if spec is None:
        return
    from .units import param_unit_kind
    scales: list = [1.0] * len(comp.params)
    changed = False
    for j, p in enumerate(comp.params):
        if j >= len(spec.params):
            break
        kind = param_unit_kind(spec.params[j])
        if kind == "mass":
            f = factors["mass"]
        elif kind == "ang":
            f = factors["comp_pos"]
        else:
            continue
        if f == 1.0:
            continue
        if isinstance(orig_params[j], Expr):
            continue                   # engine-frame expression result
        if isinstance(p, SharedBounds):
            scales[j] = f
            changed = True
        elif isinstance(p, Bounds):
            comp.params[j] = Bounds(p.lo * f, p.hi * f)
        elif isinstance(p, Fixed):
            comp.params[j] = Fixed(p.value * f)
    if changed:
        comp.unit_scales = scales


def merge(parsed_files: list[ParsedFile],
          units: dict | None = None) -> tuple[GladeConfig, list[Issue]]:
    """Section-merge several parsed files.

    Scalars may be defined in at most one file (a conflict is an error naming the
    variable and both files). Components concatenate in file order and are
    re-indexed globally 1-based.

    ``units`` is a resolved UnitSetting profile (see :mod:`core.format.units`);
    when given (and non-default) the authored values are converted to the
    engine convention here: obs arrays / source scalars right after the
    sections are built (so deferred expressions see canonical obs), literal
    component parameters after reference resolution, and shared-variable
    reference slots via ``Component.unit_scales`` (the shared dimension itself
    stays dimensionless). ``None`` leaves everything byte-identical.
    """
    issues: list[Issue] = []

    # 1) merge scalars with conflict detection (keyed by canonical name so that
    #    aliases such as 'lambda' / 'lambda_cosmo' collide as expected).
    merged: dict[str, tuple[Any, str, str]] = {}  # canon -> (value, file, raw_name)
    for pf in parsed_files:
        fname = pf.path or "<file>"
        for assign in pf.assignments:
            canon = schema.SCALAR_ALIASES.get(assign.name, assign.name)
            if canon in merged:
                prev_file = merged[canon][1]
                issues.append(Issue(
                    ERROR, "conflict",
                    f"'{assign.name}' is defined in both '{prev_file}' and "
                    f"'{fname}'; each scalar may be defined only once across the "
                    f"selected files",
                    source_file=fname, lineno=assign.lineno,
                ))
                continue
            merged[canon] = (assign.value, fname, assign.name)

    symbols = _symbol_table(merged)

    # user-defined {lo, hi} variables: any non-schema scalar holding Bounds is
    # referencable from component tuples; every reference shares ONE search
    # dimension. Schema scalars (source_x, ...) stay un-referencable when
    # optimizable — they already own their own dimension.
    bounds_vars: dict[str, Bounds] = {
        canon: entry[0] for canon, entry in merged.items()
        if isinstance(entry[0], Bounds) and schema.classify_scalar(canon) == "other"
    }

    # 2) build config sections (resolving any references nested inside lists)
    cfg = GladeConfig()
    for canon, (val, fname, _raw) in merged.items():
        cfg.provenance[canon] = fname
        sec = schema.classify_scalar(canon)
        target = getattr(cfg, sec) if sec != "other" else cfg.other
        target[canon] = _resolve_nested(val, symbols)

    # 2b) unit conversion of the scalar sections (before build_env, so the
    # deferred img-position expressions see canonical mas obs arrays).
    factors = None
    if units is not None:
        from .units import is_default, scale_factors
        if not is_default(units):
            h = cfg.cosmology.get("hubble", 0.7)
            if units.get("mass") == "msun" and not isinstance(h, (int, float)):
                issues.append(Issue(
                    ERROR, "unit_profile_bad",
                    "UnitSetting: mass = 'msun' needs a FIXED hubble to "
                    "convert to the engine's h^-1 Msun; make hubble a number "
                    "or keep mass = 'hinv_msun'"))
                h = 0.7
            factors = scale_factors(units, float(h) if isinstance(h, (int, float)) else 0.7)
            _convert_scalar_sections(cfg, factors)

    # 3) concatenate + resolve + re-index components
    # env for deferred expressions (arithmetic + obs-position references); built
    # now that cfg.obs carries the (merged) observation arrays + center offset.
    expr_env = build_env(cfg.obs, symbols)

    def _resolve(pv, comp: Component, what: str):
        if isinstance(pv, Expr):
            try:
                result = evaluate(pv.code, expr_env)
            except (ExprError, ArithmeticError, ValueError, TypeError) as exc:
                issues.append(Issue(
                    ERROR, "bad_expr",
                    f"component '{comp.name}' {what}: {exc}",
                    source_file=comp.source_file, lineno=comp.lineno,
                ))
                return pv
            if result[0] == "bounds":
                return Bounds(result[1], result[2])
            return Fixed(result[1])
        if isinstance(pv, Ref):
            canon = schema.SCALAR_ALIASES.get(pv.name, pv.name)
            if canon in bounds_vars:
                b = bounds_vars[canon]
                cfg.user_vars[canon] = b
                return SharedBounds(b.lo, b.hi, name=canon)
            if pv.name in symbols:
                return Fixed(symbols[pv.name])
            if canon in merged and isinstance(merged[canon][0], Bounds):
                issues.append(Issue(
                    ERROR, "unresolved_ref",
                    f"component '{comp.name}' {what} references the optimizable "
                    f"scalar '{pv.name}'; only fixed scalars or custom {{lo, hi}} "
                    f"variables can be referenced",
                    source_file=comp.source_file, lineno=comp.lineno,
                ))
                return pv
            if canon in merged:
                issues.append(Issue(
                    ERROR, "unresolved_ref",
                    f"component '{comp.name}' {what} references '{pv.name}', "
                    f"whose value is not a number or {{lo, hi}} bounds and "
                    f"cannot be referenced",
                    source_file=comp.source_file, lineno=comp.lineno,
                ))
                return pv
            issues.append(Issue(
                ERROR, "unresolved_ref",
                f"component '{comp.name}' {what} references unknown name "
                f"'{pv.name}'",
                source_file=comp.source_file, lineno=comp.lineno,
            ))
            return pv
        return pv

    index = 0
    for pf in parsed_files:
        for comp in pf.components:
            index += 1
            comp.index = index
            comp.z = _resolve(comp.z, comp, "redshift z")
            orig_params = list(comp.params)
            comp.params = [
                _resolve(p, comp, f"parameter p{i + 1}")
                for i, p in enumerate(comp.params)
            ]
            if factors is not None:
                _apply_component_units(comp, orig_params, factors)
            cfg.components.append(comp)

    return cfg, issues


def apply_defaults(cfg: GladeConfig) -> list[str]:
    """Fill missing basics from :data:`DEFAULTS`. Returns the names defaulted.

    The required observation arrays and components are never defaulted.
    """
    present = set(cfg.all_scalars())
    applied: list[str] = []
    for name, val in DEFAULTS.items():
        if name in present:
            continue
        sec = schema.classify_scalar(name)
        target = getattr(cfg, sec) if sec != "other" else cfg.other
        target[name] = val
        cfg.applied_defaults.append(name)
        applied.append(name)
    return applied
