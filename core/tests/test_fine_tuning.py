"""Tests for the fine_tuning staged pipeline (macro -> substructure -> polish).

Uses an analytic two-component FakeLensSub backend so the 3-round pipeline is
exercised end-to-end without the glafic or torch engines.

    python -m pytest core/tests/test_fine_tuning.py
    python core/tests/test_fine_tuning.py
"""
from __future__ import annotations

import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from core.format import lint_text  # noqa: E402
from core.format.values import Bounds, Fixed, SharedBounds  # noqa: E402
from core.optimize import OptProblem  # noqa: E402
from core.optimize.fine_tuning import (  # noqa: E402
    FineTuningSpec, build_round1_config, build_round2_config,
    build_round3_config, component_role, parse_fine_tuning,
    resolve_fine_tuning, run_fine_tuning, select_diverse)


_BASE = """
omega = 0.3
lambda_cosmo = 0.7
weos = -1.0
hubble = 0.7
xmin, ymin = -0.5, -0.5
xmax, ymax = 0.5, 0.5
pix_ext = 0.01
pix_poi = 0.2
maxlev = 5
source_z = 0.409
lens_z = 0.216
source_x = {-0.1, 0.1}
source_y = 0.024
obs_positions_mas_list = [[-266.0, 0.4], [118.8, -221.9], [238.3, 227.3], [-126.2, 319.7]]
obs_magnifications_list = [-35.6, 15.7, -7.5, 9.1]
obs_mag_errors_list = [2.1, 1.3, 1.0, 1.1]
obs_pos_sigma_mas_list = [0.41, 0.86, 2.23, 3.11]
center_offset_x = 0.0
center_offset_y = 0.0
obs_x_flip = False
DE_MAXITER = 120
DE_POPSIZE = 16
DE_SEED = 3
EARLY_STOP_PATIENCE = 10
"""

_FT_LINE = "fine_tuning = (True, 'DE', 4.0, 1.0, 'DE', 4.0, 1.0, 'DE', 0.01, 4.0, 1.0)\n"
_COMPS = """
'sie1': (1, 'sie', lens_z, {50, 300}, {-0.1, 0.1}, 0.0)
'point1': (2, 'point', lens_z, {1e5, 1e7}, {-0.30, -0.20}, {-0.05, 0.05})
"""

TRUTH = {"source_x": 0.03, "sigma": 200.0, "sie_x": 0.02,
         "log_mass": 6.0, "point_x": -0.25, "point_y": 0.0}


def _cfg(text, expect_errors=False):
    cfg, issues = lint_text(text, backend="cpu", with_defaults=True)
    if not expect_errors:
        assert not any(i.is_error for i in issues), [str(i) for i in issues]
    return cfg, issues


class FakeLensSub:
    """Separable synthetic model over a lens (sie) + substructure (point):
    each optimizable dim controls one image residual, so the global optimum
    sits exactly at TRUTH with loss 0 -- with or without the substructure."""

    name = "fake"

    def __init__(self, obs_positions, obs_mags):
        self.obs = np.asarray(obs_positions, dtype=float)
        self.mags = np.asarray(obs_mags, dtype=float)

    def compute_images(self, scene):
        sie = next(c for c in scene.components if c.glafic_type == "sie")
        pt = next((c for c in scene.components if c.glafic_type == "point"), None)
        d_src = scene.source_x - TRUTH["source_x"]
        d_sig = sie.params[0] / TRUTH["sigma"] - 1.0
        d_sx = sie.params[1] - TRUTH["sie_x"]
        if pt is not None:
            d_m = math.log10(max(pt.params[0], 1e-30)) - TRUTH["log_mass"]
            d_px = pt.params[1] - TRUTH["point_x"]
            d_py = pt.params[2] - TRUTH["point_y"]
        else:
            d_m = d_px = d_py = 0.0
        rx = [d_src, 0.1 * d_sig, d_sx, 0.05 * d_m]
        ry = [0.0, 0.0, d_py, d_px]
        return [(ox + rx[i], oy + ry[i], self.mags[i])
                for i, (ox, oy) in enumerate(self.obs)]


def _fake(cfg):
    from core.optimize import build_obs
    obs = build_obs(cfg)
    return FakeLensSub(obs.positions, obs.magnifications)


# --------------------------------------------------------------------------- #
# parsing / validation
# --------------------------------------------------------------------------- #

def test_key_parses_into_algorithm_section():
    cfg, _ = _cfg(_BASE + _FT_LINE + _COMPS)
    raw = cfg.algorithm.get("fine_tuning")
    assert isinstance(raw, tuple) and len(raw) == 11
    spec, errors = parse_fine_tuning(cfg)
    assert errors == []
    assert spec.activate is True
    assert [r.algorithm for r in spec.rounds] == ["DE", "DE", "DE"]
    assert spec.perturb == 0.01 and spec.top_k == 3


def test_upper_alias_and_aux_keys():
    text = (_BASE + _COMPS
            + "FINE_TUNING = (True, 'CMAES', 4, 0, 'jSO', 4, 1, 'DE', 0.05, 2, 1)\n"
            + "FINE_TUNING_TOP_K = 2\nFINE_TUNING_DIVERSITY = 0.2\n")
    cfg, _ = _cfg(text)
    spec, errors = parse_fine_tuning(cfg)
    assert errors == []
    assert [r.algorithm for r in spec.rounds] == ["BIPOP-CMA-ES", "JSO", "DE"]
    assert spec.top_k == 2 and spec.diversity == 0.2 and spec.perturb == 0.05
    assert spec.rounds[0].coef_b == 0.0 and spec.rounds[2].coef_a == 2.0


def test_validate_rejects_malformed_tuples():
    bad = [
        ("fine_tuning = (True, 'DE', 4, 1)\n", "11-tuple"),           # arity
        ("fine_tuning = True\n", "11-tuple"),                          # bare True
        ("fine_tuning = (True, 'amoeba', 4, 1, 'DE', 4, 1, 'DE', 0.01, 4, 1)\n",
         "amoeba"),
        ("fine_tuning = (True, 'DE', 4, 1, 'foo', 4, 1, 'DE', 0.01, 4, 1)\n",
         "round-2"),
        ("fine_tuning = (True, 'DE', 4, 1, 'DE', 4, 1, 'DE', 0.0, 4, 1)\n",
         "perturb"),
        ("fine_tuning = (True, 'DE', 4, 1, 'DE', 4, 1, 'DE', 0.01, -1, 1)\n",
         "round-3 A"),
    ]
    for line, needle in bad:
        _, issues = _cfg(_BASE + _COMPS + line, expect_errors=True)
        errs = [i.message for i in issues if i.is_error and i.code == "bad_fine_tuning"]
        assert errs, f"no bad_fine_tuning error for {line!r}"
        assert any(needle in m for m in errs), (needle, errs)


def test_validate_false_and_absent_are_silent():
    for extra in ("", "fine_tuning = False\n"):
        cfg, issues = _cfg(_BASE + _COMPS + extra)
        assert not any(i.code in ("bad_fine_tuning", "fine_tuning_inactive")
                       for i in issues)
        spec, errors, warns = resolve_fine_tuning(cfg)
        assert spec is None and errors == [] and warns == []


def test_aux_key_without_main_key_errors():
    _, issues = _cfg(_BASE + _COMPS + "fine_tuning_top_k = 2\n",
                     expect_errors=True)
    assert any(i.is_error and i.code == "bad_fine_tuning"
               and "no effect" in i.message for i in issues)


def test_activation_fallbacks_warn_and_deactivate():
    # no substructure component
    only_lens = "'sie1': (1, 'sie', lens_z, {50, 300}, {-0.1, 0.1}, 0.0)\n"
    cfg, issues = _cfg(_BASE + _FT_LINE + only_lens)
    assert any(i.code == "fine_tuning_inactive" and "substructure" in i.message
               for i in issues)
    spec, _, warns = resolve_fine_tuning(cfg)
    assert spec is None and warns

    # no main lens (point defaults to substructure)
    only_sub = "'point1': (1, 'point', lens_z, {1e5, 1e7}, {-0.3, -0.2}, 0.0)\n"
    cfg, issues = _cfg(_BASE + _FT_LINE + only_sub)
    assert any(i.code == "fine_tuning_inactive" and "main-lens" in i.message
               for i in issues)

    # nothing searchable on the substructure -> round 2 empty
    locked_sub = ("'sie1': (1, 'sie', lens_z, {50, 300}, {-0.1, 0.1}, 0.0)\n"
                  "'point1': (2, 'point', lens_z, 1e6, -0.25, 0.0)\n")
    cfg, issues = _cfg(_BASE + _FT_LINE + locked_sub)
    assert any(i.code == "fine_tuning_inactive" and "round 2" in i.message
               for i in issues)


def test_cross_category_shared_variable_falls_back():
    text = (_BASE + _FT_LINE + "shift = {-0.05, 0.05}\n"
            + "'sie1': (1, 'sie', lens_z, {50, 300}, shift, 0.0)\n"
            + "'point1': (2, 'point', lens_z, {1e5, 1e7}, shift, 0.0)\n")
    cfg, issues = _cfg(text)
    assert any(i.code == "fine_tuning_inactive" and "shift" in i.message
               for i in issues)
    spec, _, warns = resolve_fine_tuning(cfg)
    assert spec is None and any("shift" in w for w in warns)


# --------------------------------------------------------------------------- #
# partition / per-round configs
# --------------------------------------------------------------------------- #

def test_component_role_suffix_overrides_schema():
    text = (_BASE + _FT_LINE
            + "'nfw1': (1l, 'nfw', lens_z, {1e12, 1e14}, {-0.1, 0.1}, 0.0)\n"
            + "'sie1': (2s, 'sie', lens_z, {50, 300}, {-0.1, 0.1}, 0.0)\n")
    cfg, _ = _cfg(text)
    roles = {c.name: component_role(c) for c in cfg.components}
    assert roles == {"nfw1": "lens", "sie1": "substructure"}
    spec, _, warns = resolve_fine_tuning(cfg)
    assert spec is not None and warns == []


def test_round1_drops_substructure_and_keeps_indexes():
    # substructure listed FIRST so the surviving lens keeps global index 2
    text = (_BASE + _FT_LINE
            + "'point1': (1, 'point', lens_z, {1e5, 1e7}, {-0.3, -0.2}, 0.0)\n"
            + "'sie1': (2, 'sie', lens_z, {50, 300}, {-0.1, 0.1}, 0.0)\n")
    cfg, _ = _cfg(text)
    spec, _, _ = resolve_fine_tuning(cfg)
    c1 = build_round1_config(cfg, spec)
    assert [c.name for c in c1.components] == ["sie1"]
    assert c1.components[0].index == 2
    p1 = OptProblem(c1)
    assert [d.label for d in p1.dims] == ["source_x", "sie1.sigma", "sie1.x"]
    assert p1.dims[1].target == ("comp_param", 2, 0)
    assert c1.algorithm["OPTIMIZER"] == "DE"
    # the caller's cfg is untouched
    assert len(cfg.components) == 2


def test_round2_freezes_macro_at_seed():
    cfg, _ = _cfg(_BASE + _FT_LINE + _COMPS)
    spec, _, _ = resolve_fine_tuning(cfg)
    c1 = build_round1_config(cfg, spec)
    p1 = OptProblem(c1)
    # seed in SEARCH space: (source_x, log-ish? sigma is mass-like -> log10, x)
    seed = []
    truth_by_label = {"source_x": TRUTH["source_x"],
                      "sie1.sigma": TRUTH["sigma"], "sie1.x": TRUTH["sie_x"]}
    for d in p1.dims:
        v = truth_by_label[d.label]
        seed.append(math.log10(v) if d.log else v)
    seed = np.asarray(seed)

    c2 = build_round2_config(cfg, spec, p1.dims, seed)
    sie = next(c for c in c2.components if c.name == "sie1")
    assert isinstance(sie.params[0], Fixed)
    assert abs(sie.params[0].value - TRUTH["sigma"]) < 1e-9
    assert isinstance(sie.params[1], Fixed)
    # source frozen as a PLAIN float (Fixed silently reverts to defaults)
    assert type(c2.source["source_x"]) is float
    assert abs(c2.source["source_x"] - TRUTH["source_x"]) < 1e-12
    # substructure bounds untouched
    pt = next(c for c in c2.components if c.name == "point1")
    assert isinstance(pt.params[0], Bounds) and pt.params[0].lo == 1e5
    p2 = OptProblem(c2)
    assert [d.label for d in p2.dims] == ["point1.mass", "point1.x", "point1.y"]


def test_round2_freezes_shared_variable_every_reference():
    text = (_BASE + _FT_LINE + "off = {-0.1, 0.1}\n"
            + "'sie1': (1, 'sie', lens_z, {50, 300}, off, off)\n"
            + "'point1': (2, 'point', lens_z, {1e5, 1e7}, {-0.3, -0.2}, 0.0)\n")
    cfg, _ = _cfg(text)
    spec, _, _ = resolve_fine_tuning(cfg)
    c1 = build_round1_config(cfg, spec)
    p1 = OptProblem(c1)
    labels = [d.label for d in p1.dims]
    assert "off" in labels                       # ONE shared dimension
    seed = np.array([0.01 if d.label == "source_x"
                     else (math.log10(150.0) if d.log else 0.05)
                     for d in p1.dims])
    c2 = build_round2_config(cfg, spec, p1.dims, seed)
    sie = next(c for c in c2.components if c.name == "sie1")
    assert isinstance(sie.params[1], Fixed) and isinstance(sie.params[2], Fixed)
    assert abs(sie.params[1].value - 0.05) < 1e-12
    assert abs(sie.params[2].value - 0.05) < 1e-12
    assert "off" not in c2.user_vars


def test_round3_box_semantics():
    cfg, _ = _cfg(_BASE + _FT_LINE + _COMPS)
    spec, _, _ = resolve_fine_tuning(cfg)
    incumbents = {("source", "source_x"): 0.05,
                  ("comp_param", 1, 0): 210.0,      # sie sigma
                  ("comp_param", 1, 1): 0.03,       # sie x
                  ("comp_param", 2, 0): 1e6,        # point mass
                  ("comp_param", 2, 1): -0.25,      # point x
                  ("comp_param", 2, 2): 0.0}        # point y (zero, was Bounds)
    c3 = build_round3_config(cfg, spec, incumbents)
    sie = next(c for c in c3.components if c.name == "sie1")
    pt = next(c for c in c3.components if c.name == "point1")

    # optimizable nonzero -> value*(1 +- pct)
    assert isinstance(sie.params[0], Bounds)
    assert abs(sie.params[0].lo - 210.0 * 0.99) < 1e-9
    assert abs(sie.params[0].hi - 210.0 * 1.01) < 1e-9
    # originally-FIXED nonzero re-opens too (that is the point of round 3)
    assert isinstance(c3.source["source_y"], Bounds)
    # originally-fixed exact zero stays fixed (sie1.y = 0.0)
    assert isinstance(sie.params[2], Fixed) and sie.params[2].value == 0.0
    # originally-optimizable exact zero -> +- pct * (hi - lo) fallback
    assert isinstance(pt.params[2], Bounds)
    assert abs(pt.params[2].hi - 0.01 * 0.1) < 1e-12
    # negative incumbent: Bounds auto-normalizes
    assert isinstance(pt.params[1], Bounds)
    assert pt.params[1].lo < pt.params[1].hi < 0.0
    # fixed redshift stays fixed; per-round algorithm/coefs applied
    assert isinstance(sie.z, Fixed)
    assert c3.algorithm["LOSS_COEF_A"] == 4.0
    p3 = OptProblem(c3)
    assert p3.ndim >= 6


def test_select_diverse_greedy():
    xs = [np.array([0.0, 0.0]), np.array([0.01, 0.0]),
          np.array([1.0, 1.0]), np.array([0.99, 1.0])]
    fs = [0.0, 0.1, 0.2, 0.3]
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    assert select_diverse(xs, fs, bounds, top_k=3, diversity=0.1) == [0, 2]
    assert select_diverse(xs, fs, bounds, top_k=1, diversity=0.1) == [0]
    assert select_diverse([], [], bounds, top_k=3, diversity=0.1) == []


def test_select_diverse_never_seeds_invalid_candidates():
    from core.optimize.objective import INVALID_LOSS
    xs = [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5])]
    fs = [0.0, INVALID_LOSS, INVALID_LOSS]
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    # the two mutually-distant INVALID rows must NOT fill the top_k slots
    assert select_diverse(xs, fs, bounds, top_k=3, diversity=0.1) == [0]


def test_round3_boxes_clamped_to_engine_domains():
    # a user-FIXED ellipticity near the e < 1 hard limit must not produce a
    # box crossing 1.0 (glafic exit()s the whole process on e >= 1)
    text = (_BASE
            + "fine_tuning = (True, 'DE', 4, 1, 'DE', 4, 1, 'DE', 0.1, 4, 1)\n"
            + "'sie1': (1, 'sie', lens_z, {50, 300}, {-0.1, 0.1}, 0.0, 0.95, 30.0)\n"
            + "'point1': (2, 'point', lens_z, {1e5, 1e7}, {-0.3, -0.2}, {-0.05, 0.05})\n")
    cfg, _ = _cfg(text)
    spec, _, _ = resolve_fine_tuning(cfg)
    incumbents = {("source", "source_x"): 0.05,
                  ("comp_param", 1, 0): 210.0, ("comp_param", 1, 1): 0.03,
                  ("comp_param", 2, 0): 1e6, ("comp_param", 2, 1): -0.25,
                  ("comp_param", 2, 2): 0.0}
    c3 = build_round3_config(cfg, spec, incumbents)
    sie = next(c for c in c3.components if c.name == "sie1")
    e_box = sie.params[3]
    assert isinstance(e_box, Bounds) and e_box.hi < 1.0, e_box
    # an originally-optimizable param at an incumbent of exactly 0 must keep
    # its zero-fallback box INSIDE the user's own bounds (no negative e etc.)
    pt = next(c for c in c3.components if c.name == "point1")
    assert isinstance(pt.params[2], Bounds)
    assert pt.params[2].lo >= -0.05 and pt.params[2].hi <= 0.05
    # sigma box intersected with the user's {50, 300}
    assert sie.params[0].lo >= 50.0 and sie.params[0].hi <= 300.0


def test_round_configs_pin_classification_and_drop_pipeline_keys():
    cfg, _ = _cfg(_BASE + _FT_LINE + _COMPS)
    spec, _, _ = resolve_fine_tuning(cfg)
    c1 = build_round1_config(cfg, spec)
    assert all(c.category_override == "lens" for c in c1.components)
    incumbents = {("source", "source_x"): 0.05,
                  ("comp_param", 1, 0): 210.0, ("comp_param", 1, 1): 0.03,
                  ("comp_param", 2, 0): 1e6, ("comp_param", 2, 1): -0.25,
                  ("comp_param", 2, 2): 0.0}
    c3 = build_round3_config(cfg, spec, incumbents)
    roles = {c.name: c.category_override for c in c3.components}
    assert roles == {"sie1": "lens", "point1": "substructure"}
    # round configs are single-stage: the pipeline keys are stripped
    for c in (c1, c3):
        assert "fine_tuning" not in c.algorithm
    # the caller's cfg keeps both its key and its None overrides
    assert "fine_tuning" in cfg.algorithm
    assert all(c.category_override is None for c in cfg.components)


def test_aux_keys_validated_even_when_inactive():
    text = (_BASE + _COMPS + "fine_tuning = False\n"
            + "fine_tuning_top_k = 'garbage'\nfine_tuning_diversity = 99\n")
    _, issues = _cfg(text, expect_errors=True)
    msgs = [i.message for i in issues if i.is_error and i.code == "bad_fine_tuning"]
    assert any("fine_tuning_top_k" in m for m in msgs), msgs
    assert any("fine_tuning_diversity" in m for m in msgs), msgs


def test_none_is_a_malformed_value_not_absence():
    _, issues = _cfg(_BASE + _COMPS + "fine_tuning = None\n", expect_errors=True)
    assert any(i.is_error and i.code == "bad_fine_tuning" and "None" in i.message
               for i in issues)


def test_optimize_warns_on_ignored_active_key():
    import warnings

    from core.optimize import optimize
    cfg, _ = _cfg(_BASE + _FT_LINE + _COMPS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimize(cfg, backend=_fake(cfg), record_population=False)
    assert any("run_fine_tuning" in str(w.message) for w in caught)


# --------------------------------------------------------------------------- #
# end-to-end pipeline (FakeLensSub backend)
# --------------------------------------------------------------------------- #

def test_pipeline_end_to_end_converges():
    cfg, _ = _cfg(_BASE + _FT_LINE + _COMPS)
    spec, _, _ = resolve_fine_tuning(cfg)
    stages = []

    def hook(stage, chain, result):
        stages.append((stage, chain, float(result.loss)))

    ft = run_fine_tuning(cfg, backend=_fake(cfg), spec=spec,
                         stage_hook=hook, log=lambda m: None)
    assert ft.winner.loss < 1e-3, ft.winner.loss
    assert 1 <= len(ft.chains) <= spec.top_k
    assert any(s[0] == "round1" for s in stages)
    assert any(s[0] == "round2" for s in stages)
    survivors = [ch for ch in ft.chains if not ch.pruned]
    assert survivors and ft.winner_chain in [ch.chain for ch in survivors]
    # every chain ran round 2; every survivor has a final result
    assert all(ch.round2 is not None for ch in ft.chains)
    assert all(ch.final is not None for ch in survivors)
    # the fitted values recover TRUTH
    fitted = ft.winner.fitted
    for label, truth in (("point1.mass", 10 ** TRUTH["log_mass"]),
                         ("point1.x", TRUTH["point_x"])):
        assert abs(fitted[label] - truth) / max(abs(truth), 1e-12) < 0.02, \
            (label, fitted[label], truth)

    # winner_values covers every FULL-problem dimension (MCMC re-seeding)
    full = OptProblem(cfg)
    for d in full.dims:
        assert d.target in ft.winner_values, d.target


def test_round3_never_regresses_below_incumbent():
    # reviewer repro: a budget-starved round-3 jSO (random init, never
    # evaluates the box centre) used to return a ~1e4x WORSE point than the
    # chain already held; the incumbent guard must keep the better result.
    text = (_BASE + "JSO_MAXEVALS = 60\n"
            + "fine_tuning = (True, 'DE', 4, 1, 'DE', 4, 1, 'jSO', 0.01, 4, 1)\n"
            + _COMPS)
    cfg, _ = _cfg(text)
    spec, _, _ = resolve_fine_tuning(cfg)
    ft = run_fine_tuning(cfg, backend=_fake(cfg), spec=spec, log=lambda m: None)
    for ch in ft.chains:
        if not ch.pruned and ch.round3 is not None:
            # A2/B2 == A3/B3 here, so the losses compare directly
            assert ch.round3.loss <= ch.round2.loss + 1e-12, \
                (ch.chain, ch.round3.loss, ch.round2.loss)


def test_pipeline_is_deterministic():
    def run():
        cfg, _ = _cfg(_BASE + _FT_LINE + _COMPS)
        spec, _, _ = resolve_fine_tuning(cfg)
        return run_fine_tuning(cfg, backend=_fake(cfg), spec=spec,
                               log=lambda m: None)
    a, b = run(), run()
    assert a.winner_chain == b.winner_chain
    assert np.array_equal(a.winner.x, b.winner.x)
    assert a.winner.loss == b.winner.loss


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    failed = 0
    for name in names:
        try:
            globals()[name]()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"{len(names) - failed}/{len(names)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
