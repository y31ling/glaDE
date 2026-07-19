"""The ``fine_tuning`` staged pipeline: macro -> substructure -> joint polish.

.. code-block:: text

    fine_tuning = (activate,
                   algo1, A1, B1,        # round 1: main lens + source only
                   algo2, A2, B2,        # round 2: substructure, macro frozen
                   algo3, perturb,       # round 3: joint polish in a narrow box
                   A3, B3)
    fine_tuning_top_k     = 3            # diverse round-1 basins kept (chains)
    fine_tuning_diversity = 0.1          # min normalized L_inf distance between basins

Round 1 drops every substructure component and globally searches the
user-optimizable macro (main-lens + source) parameters; the ``top_k``
mutually-diverse best candidates seed independent chains. Round 2 freezes each
chain's macro at its seed and optimizes only the substructure. Round 3 prunes
chains an order of magnitude worse than the best surviving one, then re-opens
EVERY deflector / source parameter -- user-fixed or optimizable alike -- in a
relative ``value*(1 +- perturb)`` box around the chain incumbent and polishes.
The winner is the surviving chain with the lowest final loss.

Per-round ``algoN`` / ``AN`` / ``BN`` override the ``OPTIMIZER`` /
``LOSS_COEF_A`` / ``LOSS_COEF_B`` keys for that round only (amoeba is the
glafic-binary rail and is not supported here). Component roles come from the
STRICT rule: an explicit ``Nl``/``Ns`` index suffix wins, else the model's
schema category -- never the report-marker "optimizable => substructure"
default, which would classify an optimizable main lens as substructure and
delete it in round 1.

Deliberate semantics (see SPEC.md):

* activation falls back to a plain run (with a warning) when the configuration
  has no main lens, no substructure, nothing searchable in round 1 or 2, is an
  extended-source (FITS) run, or a shared ``{lo, hi}`` variable spans both a
  lens and a substructure component;
* round 2 freezes the SOURCE as well as the lens (round 3 re-opens it);
* round 3 keeps redshifts and ``hubble`` at their original optimizability
  (a user-fixed z is a measurement, not a modelling choice), skips ``_unused``
  parameter slots, and keeps exact zeros of originally-fixed parameters fixed
  (an originally-optimizable zero falls back to ``perturb * (hi - lo)``);
* shared ``{lo, hi}`` variables stay ONE shared dimension in every round.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..format import schema
from ..format.config import GladeConfig
from ..format.values import Bounds, Fixed, SharedBounds
from .de import DEResult, IterCallback
from .runner import OptResult, normalize_algorithm, optimize

#: chains whose round-2 loss exceeds ``PRUNE_RATIO * best`` are dropped
PRUNE_RATIO = 10.0
DEFAULT_TOP_K = 3
DEFAULT_DIVERSITY = 0.1
#: per-generation top slice kept in the round-1 basin archive
_ARCHIVE_PER_GEN = 16
_ARCHIVE_CAP = 50000


@dataclass(frozen=True)
class RoundSpec:
    algorithm: str      # canonical: 'DE' | 'BIPOP-CMA-ES' | 'JSO'
    coef_a: float       # per-round LOSS_COEF_A override
    coef_b: float       # per-round LOSS_COEF_B override


@dataclass
class FineTuningSpec:
    activate: bool
    rounds: Optional[tuple] = None      # (RoundSpec, RoundSpec, RoundSpec)
    perturb: float = 0.01               # round-3 relative box half-width
    top_k: int = DEFAULT_TOP_K
    diversity: float = DEFAULT_DIVERSITY


# --------------------------------------------------------------------------- #
# parsing / activation
# --------------------------------------------------------------------------- #

def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _parse_algorithm(raw, slot: str, errors: list) -> str:
    if not isinstance(raw, str):
        errors.append(f"fine_tuning {slot} algorithm must be a string, got {raw!r}")
        return "DE"
    if "amoeba" in raw.strip().lower():
        errors.append(f"fine_tuning does not support amoeba ({slot}); "
                      f"use DE, BIPOP-CMA-ES or jSO")
        return "DE"
    try:
        return normalize_algorithm(raw)
    except ValueError:
        errors.append(f"fine_tuning {slot} algorithm {raw!r}: "
                      f"expected DE, BIPOP-CMA-ES or jSO")
        return "DE"


def _parse_coef(raw, slot: str, errors: list) -> float:
    if not _is_number(raw) or float(raw) < 0.0:
        errors.append(f"fine_tuning {slot} must be a number >= 0, got {raw!r}")
        return 0.0
    return float(raw)


_MISSING = object()


def _parse_aux(cfg: GladeConfig, errors: list) -> tuple[int, float]:
    """Validate fine_tuning_top_k / fine_tuning_diversity whenever present
    (regardless of the main key's value, so garbage never lints clean)."""
    top_k = cfg.algorithm.get("fine_tuning_top_k", DEFAULT_TOP_K)
    if isinstance(top_k, bool) or not _is_number(top_k) or \
            float(top_k) != int(top_k) or int(top_k) < 1:
        errors.append(f"fine_tuning_top_k must be an integer >= 1, got {top_k!r}")
        top_k = DEFAULT_TOP_K
    diversity = cfg.algorithm.get("fine_tuning_diversity", DEFAULT_DIVERSITY)
    if not _is_number(diversity) or not (0.0 < float(diversity) <= 1.0):
        errors.append(f"fine_tuning_diversity must be in (0, 1], got {diversity!r}")
        diversity = DEFAULT_DIVERSITY
    return int(top_k), float(diversity)


def parse_fine_tuning(cfg: GladeConfig) -> tuple[Optional[FineTuningSpec], list]:
    """Parse the ``fine_tuning`` family of keys.

    Returns ``(spec, errors)``. ``spec`` is None when the key is absent;
    format errors leave a best-effort spec so activation checks still run.
    """
    errors: list = []
    raw = cfg.algorithm.get("fine_tuning", _MISSING)
    if raw is _MISSING:
        for aux in ("fine_tuning_top_k", "fine_tuning_diversity"):
            if aux in cfg.algorithm:
                errors.append(
                    f"{aux} has no effect without the fine_tuning key; if it "
                    f"was one of your own variables in an older .dat, rename "
                    f"it -- the fine_tuning* names are reserved since V0.7.1")
        return None, errors

    top_k, diversity = _parse_aux(cfg, errors)

    if isinstance(raw, bool):
        if raw:
            errors.append(
                "fine_tuning = True needs the full 11-tuple (activate, algo1, "
                "A1, B1, algo2, A2, B2, algo3, perturb, A3, B3)")
            return FineTuningSpec(activate=False), errors
        return FineTuningSpec(activate=False), errors

    if not isinstance(raw, (tuple, list)) or len(raw) != 11:
        errors.append(
            f"fine_tuning must be the 11-tuple (activate, algo1, A1, B1, "
            f"algo2, A2, B2, algo3, perturb, A3, B3) or False, got {raw!r}; "
            f"if 'fine_tuning' was one of your own variables in an older .dat, "
            f"rename it -- fine_tuning/fine_tuning_top_k/fine_tuning_diversity "
            f"are reserved keys since V0.7.1")
        return FineTuningSpec(activate=False), errors

    act = raw[0]
    if isinstance(act, bool):
        activate = act
    elif _is_number(act) and float(act) in (0.0, 1.0):
        activate = bool(act)
    else:
        errors.append(f"fine_tuning activate must be True/False, got {act!r}")
        activate = False

    rounds = (
        RoundSpec(_parse_algorithm(raw[1], "round-1", errors),
                  _parse_coef(raw[2], "round-1 A", errors),
                  _parse_coef(raw[3], "round-1 B", errors)),
        RoundSpec(_parse_algorithm(raw[4], "round-2", errors),
                  _parse_coef(raw[5], "round-2 A", errors),
                  _parse_coef(raw[6], "round-2 B", errors)),
        RoundSpec(_parse_algorithm(raw[7], "round-3", errors),
                  _parse_coef(raw[9], "round-3 A", errors),
                  _parse_coef(raw[10], "round-3 B", errors)),
    )

    perturb = raw[8]
    if not _is_number(perturb) or not (0.0 < float(perturb) < 1.0):
        errors.append(f"fine_tuning perturb must be in (0, 1) "
                      f"(e.g. 0.01 = +-1%), got {perturb!r}")
        perturb = 0.01
    perturb = float(perturb)

    return FineTuningSpec(activate=activate, rounds=rounds, perturb=perturb,
                          top_k=top_k, diversity=diversity), errors


def component_role(comp) -> str:
    """``'lens' | 'substructure' | 'extend'`` under the STRICT partition rule.

    An explicit ``Nl``/``Ns`` index suffix wins, else the model's schema
    category. Unlike the report-marker default this NEVER uses optimizability
    (which would classify an optimizable main lens as substructure and delete
    it in round 1).
    """
    spec = schema.model(comp.type)
    if spec is not None and spec.category == schema.EXTEND_CATEGORY:
        return "extend"
    if comp.category_override is not None:
        return comp.category_override
    return spec.category if spec is not None else "lens"


def _shared_names(comp) -> set:
    names = set()
    if isinstance(comp.z, SharedBounds):
        names.add(comp.z.name)
    for p in comp.params:
        if isinstance(p, SharedBounds):
            names.add(p.name)
    return names


def _comp_searchable(comp) -> bool:
    return isinstance(comp.z, Bounds) or any(isinstance(p, Bounds) for p in comp.params)


def resolve_fine_tuning(cfg: GladeConfig) -> tuple[Optional[FineTuningSpec], list, list]:
    """Parse + activation-gate the fine_tuning keys against a merged config.

    Returns ``(active_spec_or_None, errors, warnings)``: the spec is non-None
    only when the pipeline should actually run; ``warnings`` are the fallback
    reasons when the user asked for activation but a precondition failed
    (the run then proceeds as a normal single optimization).
    """
    spec, errors = parse_fine_tuning(cfg)
    warnings: list = []
    if spec is None or not spec.activate or errors:
        return None, errors, warnings

    def _fallback(reason: str):
        warnings.append(f"fine_tuning falls back to a normal run: {reason}")
        return None, errors, warnings

    from ..format.validate import is_extend_mode
    if is_extend_mode(cfg):
        return _fallback("extended-source (FITS) runs are not supported")

    lens = [c for c in cfg.components if component_role(c) == "lens"]
    subs = [c for c in cfg.components if component_role(c) == "substructure"]
    if not lens:
        return _fallback("no main-lens component (use an 'Nl' index suffix to "
                         "mark one)")
    if not subs:
        return _fallback("no substructure component (use an 'Ns' index suffix "
                         "to mark one)")

    lens_vars = set().union(*(_shared_names(c) for c in lens)) if lens else set()
    sub_vars = set().union(*(_shared_names(c) for c in subs)) if subs else set()
    crossed = sorted(lens_vars & sub_vars)
    if crossed:
        return _fallback(
            f"shared {{lo, hi}} variable(s) {', '.join(crossed)} are referenced "
            f"by both a lens and a substructure component; split them so each "
            f"side owns its own variable")

    r1_searchable = (
        any(_comp_searchable(c) for c in lens)
        or isinstance(cfg.source.get("source_x"), Bounds)
        or isinstance(cfg.source.get("source_y"), Bounds)
        or isinstance(cfg.cosmology.get("hubble"), Bounds))
    if not r1_searchable:
        return _fallback("round 1 has nothing to optimize (no {lo, hi} on the "
                         "main lens or source)")
    if not any(_comp_searchable(c) for c in subs):
        return _fallback("round 2 has nothing to optimize (no {lo, hi} on any "
                         "substructure component)")

    return spec, errors, warnings


# --------------------------------------------------------------------------- #
# per-round config surgery (never mutates the caller's cfg)
# --------------------------------------------------------------------------- #

def _apply_round(cfg: GladeConfig, r: RoundSpec) -> None:
    cfg.algorithm["OPTIMIZER"] = r.algorithm
    cfg.algorithm["LOSS_COEF_A"] = float(r.coef_a)
    cfg.algorithm["LOSS_COEF_B"] = float(r.coef_b)
    # round configs are single-stage: drop the pipeline keys so per-stage
    # glade_output .dat files are directly re-runnable and optimize() never
    # sees a nested active fine_tuning
    for k in ("fine_tuning", "fine_tuning_top_k", "fine_tuning_diversity"):
        cfg.algorithm.pop(k, None)


def _stamp_roles(cfg: GladeConfig) -> None:
    """Pin the user-visible lens/substructure classification onto the derived
    round configs. Round surgery changes which parameters are optimizable, and
    the report default ("optimizable => substructure") would otherwise flip
    markers -- e.g. the round-3 winner's triptych marking the main lens as a
    subhalo. An explicit override renders (and re-exports via the Nl/Ns
    suffix) exactly like the original run."""
    for comp in cfg.components:
        role = component_role(comp)
        if comp.category_override is None and role in ("lens", "substructure"):
            comp.category_override = role


def build_round1_config(cfg: GladeConfig, spec: FineTuningSpec) -> GladeConfig:
    """Full config minus the substructure components (indexes preserved)."""
    c = copy.deepcopy(cfg)
    c.components = [k for k in c.components if component_role(k) != "substructure"]
    _stamp_roles(c)
    _apply_round(c, spec.rounds[0])
    return c


def _freeze_shared(cfg: GladeConfig, name: str, v: float) -> None:
    """Replace EVERY reference to shared variable *name* with its fitted value
    (per-slot unit factors applied, exactly as scene injection would)."""
    for comp in cfg.components:
        if isinstance(comp.z, SharedBounds) and comp.z.name == name:
            comp.z = Fixed(v)
        scales = comp.unit_scales
        for j, p in enumerate(comp.params):
            if isinstance(p, SharedBounds) and p.name == name:
                f = scales[j] if scales is not None else 1.0
                comp.params[j] = Fixed(v * f)
    cfg.user_vars.pop(name, None)


def freeze_dims(cfg: GladeConfig, dims, x) -> GladeConfig:
    """Copy of *cfg* with every dimension in *dims* locked at candidate *x*.

    Scalar-section freezes (source_x/source_y/hubble) are stored as plain
    floats -- ``problem._fixed_scalar`` / ``scene._as_float`` silently fall
    back to DEFAULTS for anything that is not an int/float.
    """
    c = copy.deepcopy(cfg)
    by_index = {comp.index: comp for comp in c.components}
    for i, d in enumerate(dims):
        v = float(d.to_value(x[i]))
        kind = d.target[0]
        if kind == "source":
            c.source[d.target[1]] = v
        elif kind == "cosmo":
            c.cosmology["hubble"] = v
        elif kind == "comp_z":
            by_index[d.target[1]].z = Fixed(v)
        elif kind == "comp_param":
            by_index[d.target[1]].params[d.target[2]] = Fixed(v)
        elif kind == "var":
            _freeze_shared(c, d.target[1], v)
    return c


def build_round2_config(cfg: GladeConfig, spec: FineTuningSpec,
                        r1_dims, seed_x) -> GladeConfig:
    """Full config with the chain's macro (all round-1 dims) frozen at its
    seed; substructure keeps the user's ``{lo, hi}`` bounds."""
    c = freeze_dims(cfg, r1_dims, seed_x)
    _stamp_roles(c)
    _apply_round(c, spec.rounds[1])
    return c


# Hard engine parameter domains for the round-3 box. glafic enforces these
# INSIDE its kappa/deflection code via checkmodelpar_* -> terminator() ->
# exit(EXIT_FAILURE) -- a C-level process exit that no Python except can
# catch -- so an emitted Bounds crossing them would kill the whole run.
# Keyed by (model_key, param_name) with a ('*', name) fallback; values are
# (lo, hi) clamps. Conservative: only well-known hard limits are listed.
_INF = float("inf")
_DOMAINS = {
    ("*", "e"): (0.0, 1.0 - 1e-6),          # ellipticity in [0, 1)
    ("sers", "n"): (0.06, 20.0),
    ("serspot", "n"): (0.06, 20.0),
    ("pow", "gamma"): (1.0 + 1e-6, 3.0 - 1e-6),
    ("powpot", "gamma"): (1.0 + 1e-6, 3.0 - 1e-6),
    ("ein", "alpha"): (1e-6, 2.0),
    ("einpot", "alpha"): (1e-6, 2.0),
    ("gnfw", "alpha"): (0.0, 2.0 - 1e-6),
    ("gnfwpot", "alpha"): (0.0, 2.0 - 1e-6),
    ("king", "c"): (0.0, _INF),             # log10(rt/rc) >= 0
    ("*", "c"): (1e-9, _INF),               # concentrations > 0
    ("*", "rcore"): (0.0, _INF),
    ("*", "rc"): (0.0, _INF),
    ("*", "re"): (0.0, _INF),
    ("*", "rb"): (0.0, _INF),
    ("*", "a"): (0.0, _INF),
    ("*", "rco"): (0.0, _INF),
    ("*", "radius"): (0.0, _INF),
    ("*", "rd"): (0.0, _INF),
    ("*", "t"): (0.0, _INF),
    ("*", "b"): (0.0, 100.0 - 1e-6),        # acnfw core (0 <= b < 100)
}


def _param_domain(model_key: str, pname: str) -> tuple:
    dom = _DOMAINS.get((model_key, pname))
    if dom is None:
        dom = _DOMAINS.get(("*", pname), (-_INF, _INF))
    return dom


def _perturbed(v: float, pct: float, original, is_mass: bool,
               domain: tuple = (-_INF, _INF)):
    """Round-3 box for one parameter: ``value*(1 +- pct)``, clamped.

    The raw box is intersected with (a) the original user ``{lo, hi}`` when
    the parameter was optimizable (rounds 1-2 already ran safely inside it)
    and (b) the engine's hard parameter *domain* (crossing it makes glafic
    ``exit()`` the whole process). A box that collapses under clamping
    returns ``Fixed`` at the (domain-clamped) incumbent, dropping the
    dimension. Exact zeros cannot scale multiplicatively: an
    originally-optimizable zero falls back to a ``pct * (hi - lo)``
    half-width around 0, an originally-fixed zero stays fixed. Mass-like
    values <= 0 stay fixed (their search dimension is log10).
    """
    v = float(v)
    if is_mass and v <= 0.0:
        return Fixed(v)
    if v != 0.0:
        lo, hi = sorted((v * (1.0 - pct), v * (1.0 + pct)))
    else:
        if not isinstance(original, Bounds):
            return Fixed(0.0)
        half = pct * (original.hi - original.lo)
        if half <= 0.0:
            return Fixed(0.0)
        lo, hi = -half, half
    if isinstance(original, Bounds):
        lo, hi = max(lo, original.lo), min(hi, original.hi)
    lo, hi = max(lo, domain[0]), min(hi, domain[1])
    if not lo < hi:
        return Fixed(min(max(v, domain[0]), domain[1]))
    return Bounds(lo, hi)


def build_round3_config(cfg: GladeConfig, spec: FineTuningSpec,
                        incumbents: dict) -> GladeConfig:
    """Copy of *cfg* with every deflector/source parameter -- user-fixed or
    optimizable alike -- re-opened in a narrow box around the chain incumbent.

    *incumbents* maps round-1/round-2 ``Dim.target`` tuples to fitted PHYSICAL
    values; parameters absent from it (always-fixed ones) perturb around their
    own fixed value. Redshifts / hubble keep their original optimizability;
    ``_unused`` slots and shared-variable references are handled separately
    (a shared variable stays ONE narrowed shared dimension).
    """
    c = copy.deepcopy(cfg)
    pct = spec.perturb

    for comp in c.components:
        mspec = schema.model(comp.type)
        if mspec is not None and mspec.category == schema.EXTEND_CATEGORY:
            continue
        # z: only if the user made it searchable (a fixed z is a measurement)
        if isinstance(comp.z, SharedBounds):
            pass                              # narrowed via the shared variable
        elif isinstance(comp.z, Bounds):
            v = incumbents.get(("comp_z", comp.index))
            if v is not None:
                comp.z = _perturbed(v, pct, comp.z, is_mass=False)
        for j, p in enumerate(comp.params):
            if isinstance(p, SharedBounds):
                continue                      # narrowed via the shared variable
            pname = (mspec.params[j].name
                     if mspec and j < len(mspec.params) else f"p{j+1}")
            if pname.startswith("_"):
                continue                      # engine-unused slots stay put
            is_mass = bool(mspec and j < len(mspec.params)
                           and mspec.params[j].is_mass)
            if isinstance(p, Bounds):
                v = incumbents.get(("comp_param", comp.index, j))
                if v is None:
                    continue
            elif isinstance(p, Fixed):
                v = p.value
            else:
                continue                      # Unfilled/Ref: validation blocks these
            comp.params[j] = _perturbed(v, pct, p, is_mass=is_mass,
                                        domain=_param_domain(comp.type, pname))

    # shared {lo, hi} variables: one narrowed SharedBounds per name
    for name, ub in list(c.user_vars.items()):
        v = incumbents.get(("var", name))
        if v is None:
            continue
        nb = _perturbed(v, pct, ub, is_mass=False)
        if isinstance(nb, Bounds):
            c.user_vars[name] = Bounds(nb.lo, nb.hi)
            new_ref = SharedBounds(nb.lo, nb.hi, name=name)
        else:
            c.user_vars.pop(name, None)
            new_ref = nb                       # Fixed 0.0
        for comp in c.components:
            if isinstance(comp.z, SharedBounds) and comp.z.name == name:
                comp.z = new_ref
            for j, p in enumerate(comp.params):
                if isinstance(p, SharedBounds) and p.name == name:
                    comp.params[j] = new_ref

    # source position: re-opened regardless of the user's fixed/optimizable
    # choice (it is part of the frozen-bias round 3 exists to relieve)
    for axis in ("source_x", "source_y"):
        orig = c.source.get(axis)
        if isinstance(orig, Bounds):
            v = incumbents.get(("source", axis))
        elif isinstance(orig, (int, float)) and not isinstance(orig, bool):
            v = float(orig)
        else:
            continue
        if v is None:
            continue
        nb = _perturbed(v, pct, orig if isinstance(orig, Bounds) else None,
                        is_mass=False)
        c.source[axis] = nb if isinstance(nb, Bounds) else float(nb.value)

    # hubble: original optimizability only
    h = c.cosmology.get("hubble")
    if isinstance(h, Bounds):
        v = incumbents.get(("cosmo", "hubble"))
        if v is not None:
            c.cosmology["hubble"] = _perturbed(v, pct, h, is_mass=False)

    _stamp_roles(c)
    _apply_round(c, spec.rounds[2])
    return c


# --------------------------------------------------------------------------- #
# round-1 basin archive + diverse selection
# --------------------------------------------------------------------------- #

class BasinArchive:
    """Bounded (x, loss) archive fed by the ``on_iteration`` callback.

    Keeps the top ``per_gen`` candidates of every generation so early-search
    secondary basins survive the population's later collapse onto the best one.
    """

    def __init__(self, per_gen: int = _ARCHIVE_PER_GEN, cap: int = _ARCHIVE_CAP):
        self.per_gen = per_gen
        self.cap = cap
        self.xs: list = []
        self.fs: list = []

    def __call__(self, it, pop, best, energies) -> None:
        if pop is None or energies is None:
            return
        e = np.asarray(energies, dtype=float)
        p = np.asarray(pop, dtype=float)
        if e.size == 0 or p.shape[0] != e.shape[0]:
            return
        k = min(self.per_gen, e.size)
        idx = np.argpartition(e, k - 1)[:k]
        for i in idx:
            if math.isfinite(e[i]):
                self.xs.append(p[i].copy())
                self.fs.append(float(e[i]))
        if len(self.fs) > self.cap:
            order = np.argsort(self.fs)[: self.cap // 2]
            self.xs = [self.xs[i] for i in order]
            self.fs = [self.fs[i] for i in order]

    def add(self, x, f: float) -> None:
        if math.isfinite(f):
            self.xs.append(np.asarray(x, dtype=float).copy())
            self.fs.append(float(f))


def select_diverse(xs, fs, bounds, top_k: int, diversity: float) -> list:
    """Greedy best-first pick of up to *top_k* mutually-diverse candidates.

    Two candidates are diverse when at least one search dimension differs by
    ``>= diversity`` of that dimension's bound width (normalized L_inf).
    Candidates at/above ``INVALID_LOSS`` (engine error / hard-rejected image
    count) never seed a chain. Returns indices into *xs* (may be fewer than
    *top_k*).
    """
    from .objective import INVALID_LOSS
    if not fs:
        return []
    width = np.array([hi - lo for lo, hi in bounds], dtype=float)
    width[width <= 0.0] = 1.0
    order = np.argsort(fs)
    picked: list = []
    for i in order:
        if fs[i] >= INVALID_LOSS:
            break                    # ascending order: the rest are invalid too
        xi = xs[i]
        if any(np.max(np.abs(xi - xs[p]) / width) < diversity for p in picked):
            continue
        picked.append(int(i))
        if len(picked) >= top_k:
            break
    return picked


def _incumbent_result(p3, incumbents: dict, backend, algorithm: str) -> OptResult:
    """Score the chain incumbent (the round-3 box centre) under the round-3
    objective and wrap it as an OptResult (calcimage-style synthesis).

    Used both as the no-regression guard for the polish and as the final
    result of a chain whose round-3 problem has no dimensions -- either way
    the returned loss is in the round-3 ``A3/B3`` convention, so every
    chain's ``final_loss`` compares like with like.
    """
    from .backends import make_backend
    from .loss import LossConfig
    from .objective import point_source_loss
    from .scene import build_obs

    xs = []
    for d in p3.dims:
        v = incumbents.get(d.target)
        if v is None or (d.log and v <= 0.0):
            x = 0.5 * (d.lo + d.hi)
        else:
            x = math.log10(v) if d.log else float(v)
        xs.append(min(max(x, d.lo), d.hi))
    x = np.asarray(xs, dtype=float)
    scene = p3.make_scene(x)
    try:
        b = backend if not isinstance(backend, str) else make_backend(backend)
        images = b.compute_images(scene)
        loss = float(point_source_loss(images, build_obs(p3.cfg),
                                       LossConfig.from_cfg(p3.cfg)))
    except Exception:                    # noqa: BLE001 - scoring is best-effort
        loss = float("inf")              # the optimizer's answer wins
    de = DEResult(x=x, fun=loss, nit=0, converged=True, history=[])
    return OptResult(
        x=x, loss=loss, fitted=p3.decode(x), scene=scene, problem=p3, de=de,
        backend=backend if isinstance(backend, str)
        else getattr(backend, "name", "custom"),
        mode="point", algorithm=algorithm)


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #

@dataclass
class ChainRecord:
    chain: int                          # 1-based
    seed_x: np.ndarray                  # round-1 search-space candidate
    seed_loss: float                    # its round-1 loss
    seed_fitted: dict = field(default_factory=dict)
    round2: Optional[OptResult] = None
    round3: Optional[OptResult] = None
    pruned: bool = False

    @property
    def final(self) -> Optional[OptResult]:
        return self.round3 if self.round3 is not None else self.round2

    @property
    def final_loss(self) -> float:
        r = self.final
        return float(r.loss) if r is not None else float("inf")


@dataclass
class FineTuningResult:
    winner: OptResult
    winner_chain: int                   # 1-based chain number
    round1: OptResult
    chains: list
    spec: FineTuningSpec
    # Dim.target -> fitted PHYSICAL value for the winning chain, covering every
    # round-1/2/3 dimension. Lets a caller re-express the winner in ANY problem
    # built from the original config (e.g. to seed MCMC over the user bounds).
    winner_values: dict = field(default_factory=dict)


# stage_hook(stage, chain_or_None, OptResult) with stage in
# {'round1', 'round2', 'round3'}; called right after each optimize() returns.
StageHook = Optional[Callable[[str, Optional[int], OptResult], None]]


def _noop(*_a, **_k) -> None:
    return None


def run_fine_tuning(cfg: GladeConfig,
                    backend="cpu",
                    spec: Optional[FineTuningSpec] = None,
                    on_iteration: IterCallback = None,
                    stage_hook: StageHook = None,
                    log: Callable[[str], None] = print) -> FineTuningResult:
    """Run the 3-round fine_tuning pipeline and return the winning chain.

    *cfg* must be a merged+validated point-source config for which
    :func:`resolve_fine_tuning` returns an active spec (passed via *spec*,
    or re-resolved here). The caller's *cfg* is never mutated.
    """
    if spec is None:
        spec, errors, warns = resolve_fine_tuning(cfg)
        if spec is None:
            raise ValueError("fine_tuning is not active for this configuration: "
                             + "; ".join(errors + warns))
    hook = stage_hook or _noop

    # ---- round 1: macro only, archive the basins --------------------------
    r1_cfg = build_round1_config(cfg, spec)
    archive = BasinArchive()

    def r1_cb(it, pop, best, energies):
        archive(it, pop, best, energies)
        if on_iteration is not None:
            on_iteration(it, pop, best, energies)

    log(f"[fine_tuning] round 1/3: macro only "
        f"({spec.rounds[0].algorithm}, A={spec.rounds[0].coef_a:g}, "
        f"B={spec.rounds[0].coef_b:g}; substructure removed)")
    r1 = optimize(r1_cfg, backend=backend, on_iteration=r1_cb,
                  record_population=False, algorithm=spec.rounds[0].algorithm)
    archive.add(r1.x, r1.loss)
    hook("round1", None, r1)
    log(f"[fine_tuning] round 1 best loss = {r1.loss:.6g}")

    seed_idx = select_diverse(archive.xs, archive.fs, r1.problem.bounds,
                              spec.top_k, spec.diversity)
    if not seed_idx:                     # degenerate; the best result always exists
        seed_idx = [len(archive.fs) - 1]
    if len(seed_idx) < spec.top_k:
        log(f"[fine_tuning] only {len(seed_idx)}/{spec.top_k} diverse basins "
            f"found (diversity >= {spec.diversity:g})")

    chains = [ChainRecord(chain=n + 1,
                          seed_x=np.asarray(archive.xs[i], dtype=float),
                          seed_loss=float(archive.fs[i]),
                          seed_fitted=r1.problem.decode(archive.xs[i]))
              for n, i in enumerate(seed_idx)]

    # ---- round 2: freeze macro per chain, fit substructure -----------------
    for ch in chains:
        log(f"[fine_tuning] round 2/3 chain {ch.chain}/{len(chains)}: "
            f"substructure ({spec.rounds[1].algorithm}, "
            f"A={spec.rounds[1].coef_a:g}, B={spec.rounds[1].coef_b:g}; "
            f"macro frozen at seed loss {ch.seed_loss:.6g})")
        c2 = build_round2_config(cfg, spec, r1.problem.dims, ch.seed_x)
        ch.round2 = optimize(c2, backend=backend, on_iteration=on_iteration,
                             record_population=False,
                             algorithm=spec.rounds[1].algorithm)
        hook("round2", ch.chain, ch.round2)
        log(f"[fine_tuning] round 2 chain {ch.chain} loss = {ch.round2.loss:.6g}")

    # ---- prune: an order of magnitude worse than the best is out -----------
    best2 = min(ch.round2.loss for ch in chains)
    threshold = PRUNE_RATIO * best2 + 1e-9
    for ch in chains:
        ch.pruned = ch.round2.loss > threshold
        if ch.pruned:
            log(f"[fine_tuning] chain {ch.chain} pruned "
                f"(loss {ch.round2.loss:.6g} > {PRUNE_RATIO:g}x best "
                f"{best2:.6g})")
    survivors = [ch for ch in chains if not ch.pruned]

    # ---- round 3: joint polish in a narrow box -----------------------------
    # Round 3 can only ever IMPROVE a chain: none of the optimizers evaluates
    # the box centre (the chain incumbent), so a budget-limited run could
    # otherwise return a point strictly worse than what the chain already
    # holds. The incumbent is therefore re-scored under the round-3 objective
    # and kept whenever it beats the optimizer's answer; a skipped (ndim==0)
    # chain gets the same re-scoring so every final_loss compares in the SAME
    # round-3 A/B convention.
    from .problem import OptProblem
    for ch in survivors:
        incumbents = {d.target: d.to_value(ch.seed_x[i])
                      for i, d in enumerate(r1.problem.dims)}
        r2p = ch.round2.problem
        incumbents.update({d.target: d.to_value(ch.round2.x[i])
                           for i, d in enumerate(r2p.dims)})
        c3 = build_round3_config(cfg, spec, incumbents)
        p3 = OptProblem(c3)
        incumbent = _incumbent_result(p3, incumbents, backend,
                                      spec.rounds[2].algorithm)
        if p3.ndim == 0:                 # nothing survived the zero/mass guards
            log(f"[fine_tuning] round 3 chain {ch.chain} skipped "
                f"(no perturbable dimensions); keeping the incumbent "
                f"(loss {incumbent.loss:.6g} in round-3 units)")
            ch.round3 = incumbent
            hook("round3", ch.chain, ch.round3)
            continue
        log(f"[fine_tuning] round 3/3 chain {ch.chain}: joint polish "
            f"({spec.rounds[2].algorithm}, +-{spec.perturb * 100:g}%, "
            f"A={spec.rounds[2].coef_a:g}, B={spec.rounds[2].coef_b:g})")
        polished = optimize(c3, backend=backend, on_iteration=on_iteration,
                            record_population=False,
                            algorithm=spec.rounds[2].algorithm)
        if math.isfinite(incumbent.loss) and incumbent.loss < polished.loss:
            log(f"[fine_tuning] round 3 chain {ch.chain} did not improve on "
                f"its incumbent ({polished.loss:.6g} > {incumbent.loss:.6g}); "
                f"keeping the incumbent")
            ch.round3 = incumbent
        else:
            ch.round3 = polished
        hook("round3", ch.chain, ch.round3)
        log(f"[fine_tuning] round 3 chain {ch.chain} loss = {ch.round3.loss:.6g}")

    winner = min(survivors, key=lambda ch: ch.final_loss)
    runner_up = sorted((ch for ch in survivors if ch is not winner),
                       key=lambda ch: ch.final_loss)
    for ch in runner_up:
        if ch.final_loss <= 2.0 * winner.final_loss + 1e-12:
            log(f"[fine_tuning] note: chain {ch.chain} "
                f"(loss {ch.final_loss:.6g}) is within 2x of the winner -- "
                f"the data may not distinguish these solutions")
    log(f"[fine_tuning] winner: chain {winner.chain} "
        f"loss = {winner.final_loss:.6g}")

    winner_values = {d.target: d.to_value(winner.seed_x[i])
                     for i, d in enumerate(r1.problem.dims)}
    winner_values.update({d.target: d.to_value(winner.round2.x[i])
                          for i, d in enumerate(winner.round2.problem.dims)})
    if winner.round3 is not None:
        winner_values.update({d.target: d.to_value(winner.round3.x[i])
                              for i, d in enumerate(winner.round3.problem.dims)})

    return FineTuningResult(winner=winner.final, winner_chain=winner.chain,
                            round1=r1, chains=chains, spec=spec,
                            winner_values=winner_values)
