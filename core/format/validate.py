"""Semantic validation of a merged :class:`GladeConfig`.

Returns a list of :class:`Issue`; the caller decides whether any ``ERROR`` blocks
the run. Backend-capability checks only apply when a backend is given.
"""
from __future__ import annotations

import math
from typing import Optional

from . import schema
from .config import GladeConfig
from .diagnostics import ERROR, WARNING, Issue
from .values import Bounds, Component, Fixed, Ref, SharedBounds, Unfilled


def _check_unfilled_scalars(cfg: GladeConfig, issues: list[Issue]) -> None:
    """Flag placeholders and unresolved references, including ones nested inside
    list values such as the observation arrays."""

    def scan(val, name, depth=0):
        if isinstance(val, Unfilled):
            where = f"'{name}'" if depth == 0 else f"an element of '{name}'"
            issues.append(Issue(
                ERROR, "unfilled",
                f"{where} is still a placeholder ({val}); fill in a value "
                f"before running", source_file=cfg.provenance.get(name)))
        elif isinstance(val, Ref):
            where = f"'{name}'" if depth == 0 else f"an element of '{name}'"
            if isinstance(cfg.get(val.name), Bounds):
                msg = (f"{where} references '{val.name}', a {{lo, hi}} "
                       f"variable; optimizable variables can only be "
                       f"referenced from component tuples")
            else:
                msg = f"{where} references unknown name '{val.name}'"
            issues.append(Issue(
                ERROR, "unresolved_ref", msg,
                source_file=cfg.provenance.get(name)))
        elif isinstance(val, (list, tuple)):
            for v in val:
                scan(v, name, depth + 1)

    for name, val in cfg.all_scalars().items():
        scan(val, name)
        if name in schema.DEPRECATED_KEYS:
            issues.append(Issue(
                WARNING, "deprecated",
                f"'{name}' was removed in V0.4.1 and is ignored; the MCMC prior is "
                f"always the DE {{lower, upper}} bounds",
                source_file=cfg.provenance.get(name)))


def _check_user_vars(cfg: GladeConfig, issues: list[Issue]) -> None:
    """A shared {lo, hi} variable is ONE search dimension: it must be used
    either only in mass-like (log10-searched) parameter slots or only in
    linear ones — a mixed usage cannot have a single consistent search space."""
    usage: dict[str, set[bool]] = {}
    for comp in cfg.components:
        spec = schema.model(comp.type)
        if isinstance(comp.z, SharedBounds):
            usage.setdefault(comp.z.name, set()).add(False)
        for j, p in enumerate(comp.params):
            if not isinstance(p, SharedBounds):
                continue
            is_mass = bool(spec and j < len(spec.params) and spec.params[j].is_mass)
            usage.setdefault(p.name, set()).add(is_mass)
    for name, kinds in usage.items():
        if len(kinds) > 1:
            issues.append(Issue(
                ERROR, "var_mixed_usage",
                f"variable '{name}' is referenced by both mass-like "
                f"(log10-searched) and linear parameters; one shared variable "
                f"cannot serve both — define two variables"))

    # a {lo, hi} variable defined but never referenced is inert: it gets no
    # search dimension and never appears in the fit output — warn loudly
    # (it is most likely a typo in either the definition or the references).
    defined = {n for n, v in cfg.other.items() if isinstance(v, Bounds)}
    for name in sorted(defined - set(usage)):
        issues.append(Issue(
            WARNING, "var_unused",
            f"variable '{name}' = {{lo, hi}} is defined but never referenced "
            f"by a component; it is ignored (no search dimension)",
            source_file=cfg.provenance.get(name)))


def is_extend_mode(cfg: GladeConfig) -> bool:
    """An extended-source run: a FITS is given or any component is an extend model."""
    if cfg.obs.get("extended_file") is not None:
        return True
    return any(schema.is_extend_model(c.type) for c in cfg.components)


def _check_obs(cfg: GladeConfig, issues: list[Issue]) -> None:
    extend = is_extend_mode(cfg)

    if extend:
        # Extended-source mode: the FITS is the primary constraint and is
        # required; the four point arrays are optional (point constraints may be
        # absent, in glade arrays, or in a glafic constraint_file).
        if cfg.obs.get("extended_file") is None:
            issues.append(Issue(
                ERROR, "missing_extended_file",
                "extended-source components are present but 'extended_file' "
                "(the observed FITS image) is not set"))
        elif not any(schema.is_extend_model(c.type) for c in cfg.components):
            issues.append(Issue(
                ERROR, "no_extend_component",
                "'extended_file' is set but the configuration has no extended-"
                "source component (e.g. an 'extsersic' tuple) to fit it with"))
        has_arrays = any(k in cfg.obs for k in schema.REQUIRED_OBS_KEYS)
        has_file = cfg.obs.get("constraint_file") is not None
        if has_arrays and has_file:
            issues.append(Issue(
                WARNING, "duplicate_point_obs",
                "both glade point-observation arrays and a 'constraint_file' are "
                "given; the constraint_file takes precedence"))
        if not has_arrays:
            return  # point obs via constraint_file (or none) -> array checks N/A
        # a partial set of glade point arrays would crash the constraint-file
        # writer at runtime -> require all four when any is present.
        for key in schema.REQUIRED_OBS_KEYS:
            if key not in cfg.obs:
                issues.append(Issue(
                    ERROR, "missing_obs",
                    f"point-observation array '{key}' is missing; in extend mode "
                    f"either give all four obs arrays or use a 'constraint_file'"))
    else:
        for key in schema.REQUIRED_OBS_KEYS:
            if key not in cfg.obs:
                issues.append(Issue(
                    ERROR, "missing_obs",
                    f"required observation array '{key}' is missing"))

    array_keys = schema.REQUIRED_OBS_KEYS + schema.OBS_EXTEND_ARRAY_KEYS
    arrays = {k: cfg.obs[k] for k in array_keys if k in cfg.obs}
    lengths = set()
    for k, v in arrays.items():
        if not isinstance(v, list):
            issues.append(Issue(ERROR, "obs_type",
                                f"'{k}' must be a list, got {type(v).__name__}"))
            continue
        lengths.add(len(v))
    if len(lengths) > 1:
        issues.append(Issue(
            ERROR, "obs_length",
            f"observation arrays have inconsistent lengths: "
            f"{ {k: len(v) for k, v in arrays.items() if isinstance(v, list)} }"))

    pos = arrays.get("obs_positions_mas_list")
    if isinstance(pos, list):
        for i, p in enumerate(pos):
            if not (isinstance(p, (list, tuple)) and len(p) == 2):
                issues.append(Issue(
                    ERROR, "obs_shape",
                    f"obs_positions_mas_list[{i}] must be a [x, y] pair"))
                break


def _check_component(comp: Component, backend: Optional[str],
                     distinct_z: set, issues: list[Issue],
                     known_scalars: frozenset = frozenset()) -> None:
    spec = schema.model(comp.type)
    if spec is None:
        issues.append(Issue(
            ERROR, "unknown_model",
            f"component '{comp.name}' uses unknown model type '{comp.type}'",
            source_file=comp.source_file, lineno=comp.lineno))
        return

    # redshift
    if isinstance(comp.z, Unfilled):
        issues.append(Issue(ERROR, "unfilled",
                            f"component '{comp.name}' redshift z is unfilled",
                            source_file=comp.source_file, lineno=comp.lineno))
    elif isinstance(comp.z, Ref):
        # a Ref to a KNOWN scalar already got a precise merge-time error
        # (e.g. "references the optimizable scalar ..."); don't double-report
        if schema.SCALAR_ALIASES.get(comp.z.name, comp.z.name) not in known_scalars:
            issues.append(Issue(ERROR, "unresolved_ref",
                                f"component '{comp.name}' redshift references unknown "
                                f"'{comp.z.name}'",
                                source_file=comp.source_file, lineno=comp.lineno))
    elif isinstance(comp.z, Fixed):
        # extend models carry the SOURCE redshift, not a lens plane, so they
        # must not feed the multi-plane check
        if not schema.is_extend_model(comp.type):
            distinct_z.add(round(comp.z.value, 9))

    # parameter count
    n = len(comp.params)
    if n < spec.required_min:
        issues.append(Issue(
            ERROR, "too_few_params",
            f"component '{comp.name}' ('{comp.type}') needs at least "
            f"{spec.required_min} parameters, got {n}",
            source_file=comp.source_file, lineno=comp.lineno))
    if n > schema.GLAFIC_NPARAM:
        issues.append(Issue(
            ERROR, "too_many_params",
            f"component '{comp.name}' ('{comp.type}') has {n} parameters; "
            f"glafic allows at most {schema.GLAFIC_NPARAM}",
            source_file=comp.source_file, lineno=comp.lineno))

    # per-parameter checks
    for i, p in enumerate(comp.params):
        pname = spec.params[i].name if i < len(spec.params) else f"p{i + 1}"
        if isinstance(p, Unfilled):
            issues.append(Issue(
                ERROR, "unfilled",
                f"component '{comp.name}' parameter '{pname}' is unfilled ({p})",
                source_file=comp.source_file, lineno=comp.lineno))
        elif isinstance(p, Ref):
            if schema.SCALAR_ALIASES.get(p.name, p.name) not in known_scalars:
                issues.append(Issue(
                    ERROR, "unresolved_ref",
                    f"component '{comp.name}' parameter '{pname}' references unknown "
                    f"'{p.name}'", source_file=comp.source_file, lineno=comp.lineno))

    # mass-like parameters must be positive to allow log10 search
    for mi in spec.mass_positions:
        if mi < len(comp.params):
            p = comp.params[mi]
            pname = spec.params[mi].name
            if isinstance(p, Bounds) and (p.lo <= 0 or p.hi <= 0):
                issues.append(Issue(
                    ERROR, "mass_nonpositive",
                    f"component '{comp.name}' mass-like parameter '{pname}' bounds "
                    f"{{{p.lo}, {p.hi}}} must be > 0 (searched in log10)",
                    source_file=comp.source_file, lineno=comp.lineno))

    # backend capability
    if backend == "gpu" and not spec.gpu:
        issues.append(Issue(
            ERROR, "gpu_unsupported",
            f"component '{comp.name}' model '{comp.type}' is not supported by the "
            f"GPU backend (Rhongomyniad); use CPU or Glafic",
            source_file=comp.source_file, lineno=comp.lineno))

    if spec.uncertain and comp.is_optimizable():
        issues.append(Issue(
            WARNING, "uncertain_labels",
            f"component '{comp.name}' model '{comp.type}' has best-effort "
            f"parameter labels; verify the parameter order against glafic docs",
            source_file=comp.source_file, lineno=comp.lineno))

    # an 'Nl'/'Ns' index suffix classifies deflectors only; an extended-source
    # profile is neither a lens nor a sub-structure marker
    if comp.category_override is not None and spec.category == schema.EXTEND_CATEGORY:
        issues.append(Issue(
            WARNING, "category_suffix_ignored",
            f"component '{comp.name}' ('{comp.type}') is an extended-source "
            f"profile; its index suffix "
            f"'{comp.raw_index}{ 'l' if comp.category_override == 'lens' else 's' }' "
            f"is ignored",
            source_file=comp.source_file, lineno=comp.lineno))


def validate(cfg: GladeConfig, backend: Optional[str] = None) -> list[Issue]:
    issues: list[Issue] = []

    if backend is not None and not schema.is_backend(backend):
        issues.append(Issue(ERROR, "bad_backend",
                            f"unknown backend '{backend}'; expected one of "
                            f"{schema.BACKENDS}"))
        backend = None

    _check_unfilled_scalars(cfg, issues)
    _check_obs(cfg, issues)
    _check_user_vars(cfg, issues)

    known_scalars = frozenset(cfg.all_scalars())
    distinct_z: set = set()
    for comp in cfg.components:
        _check_component(comp, backend, distinct_z, issues, known_scalars)

    if not cfg.components:
        issues.append(Issue(ERROR, "no_components",
                            "configuration has no lens/sub-structure components"))

    if backend == "gpu" and len(distinct_z) > 1:
        issues.append(Issue(
            WARNING, "multi_plane",
            "components span multiple lens redshifts; the GPU backend "
            "(Rhongomyniad) is single-plane and may reject this configuration"))

    gp = cfg.algorithm.get("gpu_precision")
    if gp is not None and not (isinstance(gp, (int, float))
                               and math.isfinite(gp)
                               and int(gp) in (64, 48, 32)):
        issues.append(Issue(
            ERROR, "bad_gpu_precision",
            f"gpu_precision = {gp!r}: expected 64 (fp64), 48 (mixed: fp32 "
            f"fields + fp64 Newton refine) or 32 (fp32)"))

    return issues
