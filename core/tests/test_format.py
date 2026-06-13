"""Tests for the GLADE V0.4 ``.dat`` format package.

Runnable two ways::

    python -m pytest core/tests/test_format.py
    python core/tests/test_format.py          # no pytest needed

Each ``test_*`` function uses plain ``assert``.
"""
from __future__ import annotations

import os
import sys

# allow `python core/tests/test_format.py` from anywhere
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.format import (  # noqa: E402
    Bounds,
    Fixed,
    GladeSyntaxError,
    Unfilled,
    apply_defaults,
    has_errors,
    lint_text,
    merge,
    parse_text,
    validate,
)
from core.format.values import Ref  # noqa: E402


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def test_scalar_numbers_and_scientific():
    pf = parse_text("omega = 0.3\nhubble = 7E-1\nweos = -1.0\nmaxlev = 5")
    vals = {a.name: a.value for a in pf.assignments}
    assert vals["omega"] == 0.3
    assert vals["hubble"] == 0.7
    assert vals["weos"] == -1.0
    assert vals["maxlev"] == 5.0


def test_multi_target_assignment():
    pf = parse_text("xmin, ymin = -0.5, -0.5\nxmax, ymax = 0.5, 0.5")
    vals = {a.name: a.value for a in pf.assignments}
    assert vals == {"xmin": -0.5, "ymin": -0.5, "xmax": 0.5, "ymax": 0.5}


def test_bounds_is_optimizable():
    pf = parse_text("source_x = {-0.1, 0.1}\nsource_y = 0.0")
    vals = {a.name: a.value for a in pf.assignments}
    assert isinstance(vals["source_x"], Bounds)
    assert vals["source_x"].lo == -0.1 and vals["source_x"].hi == 0.1
    assert vals["source_y"] == 0.0


def test_bounds_normalises_order():
    pf = parse_text("m = {1e7, 1e5}")
    b = pf.assignments[0].value
    assert isinstance(b, Bounds) and b.lo == 1e5 and b.hi == 1e7


def test_obs_list_literals():
    text = (
        "obs_positions_mas_list = [[-266.035, 0.427], [118.835, -221.927]]\n"
        "obs_magnifications_list = [-35.6, 15.7]\n"
    )
    pf = parse_text(text)
    vals = {a.name: a.value for a in pf.assignments}
    assert vals["obs_positions_mas_list"] == [[-266.035, 0.427], [118.835, -221.927]]
    assert vals["obs_magnifications_list"] == [-35.6, 15.7]


def test_bool_flag():
    pf = parse_text("obs_x_flip = True\nDE_POLISH = False")
    vals = {a.name: a.value for a in pf.assignments}
    assert vals["obs_x_flip"] is True
    assert vals["DE_POLISH"] is False


def test_comments_stripped():
    text = "# full line comment\nomega = 0.3   # trailing comment\n"
    pf = parse_text(text)
    assert len(pf.assignments) == 1
    assert pf.assignments[0].value == 0.3


def test_component_basic_and_ref_resolution():
    text = (
        "lens_z = 0.216\n"
        "'sers1': (1, 'sers', lens_z, 9.9e9, 2.6e-3, 2.7e-2, 0.30, 112.0, 0.39, 1.05)\n"
    )
    pf = parse_text(text)
    assert len(pf.components) == 1
    c = pf.components[0]
    assert c.name == "sers1" and c.type == "sers" and c.raw_index == 1
    # lens_z defined before -> resolved in-file
    assert isinstance(c.z, Fixed) and c.z.value == 0.216
    assert all(isinstance(p, Fixed) for p in c.params)
    assert len(c.params) == 7


def test_component_multiline_with_bounds():
    text = (
        "'king1': (4, 'king', 0.216, {1e6, 1e9}, {0.10, 0.16},\n"
        "          {-0.24, -0.18}, 0.0, 0.0, {0.001, 0.05}, {0.8, 2.2})\n"
    )
    pf = parse_text(text)
    c = pf.components[0]
    assert c.type == "king"
    assert isinstance(c.params[0], Bounds)  # mass
    assert isinstance(c.params[3], Fixed) and c.params[3].value == 0.0
    assert c.is_optimizable()


def test_component_index_suffix_overrides_category():
    text = (
        "lens_z = 0.216\n"
        "'anfw1': (3l, 'anfw', lens_z, 3.6e11, -2.9e-3, 2.7e-2, 0.46, 26.6, 29.4)\n"
        "'sers1': (4s, 'sers', lens_z, 2.1e10, 0.0, 0.0)\n"
        "'point1': (5, 'point', lens_z, 1e6, -0.18, -0.33)\n"
    )
    pf = parse_text(text)
    anfw, sers, point = pf.components
    assert anfw.raw_index == 3 and anfw.category_override == "lens"
    assert sers.raw_index == 4 and sers.category_override == "substructure"
    assert point.raw_index == 5 and point.category_override is None
    # the suffix is classification-only: params parse exactly as before
    assert all(isinstance(p, Fixed) for p in anfw.params)
    assert len(anfw.params) == 6


def test_component_index_suffix_spacing_case_multiline():
    pf = parse_text("'a1': ( 3L ,\n  'anfw', 0.216, 1e11, 0.0, 0.0)")
    c = pf.components[0]
    assert c.raw_index == 3 and c.category_override == "lens"
    pf = parse_text("'b1': (7S, 'point', 0.216, 1e6, 0.0, 0.0)")
    assert pf.components[0].category_override == "substructure"


def test_component_index_unknown_suffix_rejected():
    assert _expect_syntax_error("'a1': (3x, 'point', 0.216, 1e6, 0.0, 0.0)")


def test_component_index_suffix_survives_merge_reindex():
    f1 = parse_text("'anfw1': (3l, 'anfw', 0.2, 1e11, 0.0, 0.0, 0.4, 20.0, 29.0)\n"
                    "'point1': (4, 'point', 0.2, 1e6, 0.1, 0.1)", path="f1.dat")
    cfg, issues = merge([f1])
    assert not has_errors(issues)
    assert [c.index for c in cfg.components] == [1, 2]
    assert cfg.components[0].category_override == "lens"
    assert cfg.components[1].category_override is None


def test_category_suffix_on_extend_model_warns():
    pf = parse_text(
        "'ext1': (1s, 'extsersic', 1.0, 1.0, 0.0, 0.0, 0.2, 10.0, 0.3, 1.0)")
    cfg, _ = merge([pf])
    issues = validate(cfg)
    assert any(i.code == "category_suffix_ignored" for i in issues)


def test_forward_or_cross_file_ref_is_deferred():
    # lens_z defined AFTER the tuple -> stays a Ref until merge
    text = (
        "'sie1': (1, 'sie', lens_z, 118.0, 0.0, 0.0, 0.1, 29.0)\n"
        "lens_z = 0.216\n"
    )
    pf = parse_text(text)
    assert isinstance(pf.components[0].z, Ref)
    cfg, issues = merge([pf])
    assert not has_errors(issues)
    assert isinstance(cfg.components[0].z, Fixed)
    assert cfg.components[0].z.value == 0.216


def test_placeholder_unfilled():
    pf = parse_text("source_x = $float{lower, upper}\nsource_y = $float\nmaxlev = $int")
    vals = {a.name: a.value for a in pf.assignments}
    assert isinstance(vals["source_x"], Unfilled) and vals["source_x"].optimizable
    assert isinstance(vals["source_y"], Unfilled) and not vals["source_y"].optimizable
    assert isinstance(vals["maxlev"], Unfilled) and vals["maxlev"].kind == "int"


def test_placeholder_in_component():
    pf = parse_text("'point1': (1, 'point', 0.216, $float{l,u}, $float, $float)")
    c = pf.components[0]
    assert all(isinstance(p, Unfilled) for p in c.params)


# --------------------------------------------------------------------------- #
# syntax errors
# --------------------------------------------------------------------------- #

def _expect_syntax_error(text):
    try:
        parse_text(text)
    except GladeSyntaxError:
        return True
    return False


def test_duplicate_scalar_in_file_errors():
    assert _expect_syntax_error("omega = 0.3\nomega = 0.4")


def test_arithmetic_rejected():
    assert _expect_syntax_error("omega = 0.3 + 0.1")


def test_call_rejected():
    assert _expect_syntax_error("omega = float(3)")


def test_bounds_wrong_arity_rejected():
    assert _expect_syntax_error("m = {1, 2, 3}")


def test_bare_expression_rejected():
    assert _expect_syntax_error("42")


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #

def test_merge_conflict_detected():
    a = parse_text("omega = 0.3", path="a.dat")
    b = parse_text("omega = 0.4", path="b.dat")
    cfg, issues = merge([a, b])
    assert has_errors(issues)
    assert any(i.code == "conflict" for i in issues)


def test_merge_cross_file_reference():
    consts = parse_text("lens_z = 0.216", path="consts.dat")
    lens = parse_text("'sie1': (1, 'sie', lens_z, 118.0, 0.0, 0.0, 0.1, 29.0)",
                      path="lens.dat")
    cfg, issues = merge([consts, lens])
    assert not has_errors(issues)
    assert isinstance(cfg.components[0].z, Fixed) and cfg.components[0].z.value == 0.216


def test_merge_global_reindex():
    f1 = parse_text("'sers1': (1, 'sers', 0.2, 1e9, 0, 0, 0, 0, 0.3, 1.0)\n"
                    "'sie1': (2, 'sie', 0.2, 100, 0, 0, 0.1, 29)", path="f1.dat")
    f2 = parse_text("'point1': (1, 'point', 0.2, 1e6, 0.1, 0.1)", path="f2.dat")
    cfg, issues = merge([f1, f2])
    assert [c.index for c in cfg.components] == [1, 2, 3]
    assert [c.name for c in cfg.components] == ["sers1", "sie1", "point1"]


def test_classification_into_sections():
    pf = parse_text(
        "omega = 0.3\nxmin = -0.5\nlens_z = 0.2\nsource_x = 0.0\n"
        "center_offset_x = 0.01\nDE_MAXITER = 650\nweird_key = 1.0",
        path="m.dat")
    cfg, _ = merge([pf])
    assert "omega" in cfg.cosmology
    assert "xmin" in cfg.grid
    assert "lens_z" in cfg.redshifts
    assert "source_x" in cfg.source
    assert "center_offset_x" in cfg.obs
    assert "DE_MAXITER" in cfg.algorithm
    assert "weird_key" in cfg.other


def test_lambda_alias():
    pf = parse_text("lambda = 0.7", path="m.dat")
    cfg, _ = merge([pf])
    assert cfg.cosmology.get("lambda_cosmo") == 0.7


# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #

def test_apply_defaults_fills_missing_not_present():
    pf = parse_text("omega = 0.25", path="m.dat")
    cfg, _ = merge([pf])
    applied = apply_defaults(cfg)
    assert cfg.cosmology["omega"] == 0.25      # not overridden
    assert "omega" not in applied
    assert cfg.cosmology["lambda_cosmo"] == 0.7  # defaulted
    assert "lambda_cosmo" in applied
    assert cfg.grid["maxlev"] == 5


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

_GOOD = """
omega = 0.3
lambda_cosmo = 0.7
weos = -1.0
hubble = 0.7
xmin, ymin = -0.5, -0.5
xmax, ymax = 0.5, 0.5
pix_ext = 0.01
pix_poi = 0.2
maxlev = 5
source_z = 0.4090
lens_z = 0.2160
source_x = {-0.1, 0.1}
source_y = 0.0244
obs_positions_mas_list = [[-266.0, 0.4], [118.8, -221.9], [238.3, 227.3], [-126.2, 319.7]]
obs_magnifications_list = [-35.6, 15.7, -7.5, 9.1]
obs_mag_errors_list = [2.1, 1.3, 1.0, 1.1]
obs_pos_sigma_mas_list = [0.41, 0.86, 2.23, 3.11]
center_offset_x = 0.01535
center_offset_y = 0.0322
obs_x_flip = True
'sers1': (1, 'sers', lens_z, 9.9e9, 2.6e-3, 2.7e-2, 0.30, 112.0, 0.39, 1.05)
'point1': (2, 'point', lens_z, {1e5, 1e7}, {-0.30, -0.20}, {-0.05, 0.05})
"""


def test_good_config_validates_cpu():
    cfg, issues = lint_text(_GOOD, backend="cpu", with_defaults=True)
    assert not has_errors(issues), [str(i) for i in issues if i.is_error]


def test_missing_obs_blocks():
    text = "'point1': (1, 'point', 0.2, 1e6, 0.1, 0.1)"
    cfg, issues = lint_text(text, backend="cpu", with_defaults=True)
    assert has_errors(issues)
    assert any(i.code == "missing_obs" for i in issues)


def test_no_components_blocks():
    cfg, issues = lint_text(_GOOD.split("'sers1'")[0], backend="cpu", with_defaults=True)
    assert any(i.code == "no_components" for i in issues)


def test_unknown_model_blocks():
    text = _GOOD + "'foo1': (3, 'banana', lens_z, 1e6, 0.1, 0.1)\n"
    cfg, issues = lint_text(text, backend="cpu", with_defaults=True)
    assert any(i.code == "unknown_model" for i in issues)


def test_gpu_unsupported_model_blocks_only_on_gpu():
    # V0.5.0: nearly every model runs on the GPU now; 'gals' (file-based
    # catalogue) is the remaining CPU-only one
    text = _GOOD + "'g1': (3, 'gals', lens_z, 1.0)\n"
    _, issues_gpu = lint_text(text, backend="gpu", with_defaults=True)
    _, issues_cpu = lint_text(text, backend="cpu", with_defaults=True)
    assert any(i.code == "gpu_unsupported" for i in issues_gpu)
    assert not any(i.code == "gpu_unsupported" for i in issues_cpu)


def test_unfilled_blocks():
    text = _GOOD.replace("source_y = 0.0244", "source_y = $float")
    cfg, issues = lint_text(text, backend="cpu", with_defaults=True)
    assert any(i.code == "unfilled" for i in issues)


def test_mass_nonpositive_bounds_blocks():
    text = _GOOD + "'p2': (3, 'point', lens_z, {-1e5, 1e7}, 0.1, 0.1)\n"
    cfg, issues = lint_text(text, backend="cpu", with_defaults=True)
    assert any(i.code == "mass_nonpositive" for i in issues)


def test_too_few_params_blocks():
    text = _GOOD + "'p2': (3, 'point', lens_z, {1e5, 1e7})\n"
    cfg, issues = lint_text(text, backend="cpu", with_defaults=True)
    assert any(i.code == "too_few_params" for i in issues)


# --------------------------------------------------------------------------- #
# regressions from adversarial verification
# --------------------------------------------------------------------------- #

def test_nested_unknown_ref_in_list_is_flagged():
    # a reference to an undefined name buried inside an obs array must error,
    # not silently leak a Ref object into the config.
    text = _GOOD.replace(
        "obs_magnifications_list = [-35.6, 15.7, -7.5, 9.1]",
        "obs_magnifications_list = [nope, 15.7, -7.5, 9.1]")
    cfg, issues = lint_text(text, backend="cpu", with_defaults=True)
    assert any(i.code == "unresolved_ref" for i in issues)


def test_nested_placeholder_in_list_is_flagged():
    text = _GOOD.replace(
        "obs_mag_errors_list = [2.1, 1.3, 1.0, 1.1]",
        "obs_mag_errors_list = [$float, 1.3, 1.0, 1.1]")
    cfg, issues = lint_text(text, backend="cpu", with_defaults=True)
    assert any(i.code == "unfilled" for i in issues)


def test_nested_known_ref_in_list_resolves():
    # a reference to a defined scalar inside a list should resolve to its value
    text = (
        "k = 9.1\n"
        "obs_positions_mas_list = [[1.0, 2.0]]\n"
        "obs_magnifications_list = [k]\n"
        "obs_mag_errors_list = [0.1]\n"
        "obs_pos_sigma_mas_list = [0.1]\n"
        "'p1': (1, 'point', 0.2, 1e6, 0.1, 0.1)\n"
    )
    cfg, issues = lint_text(text, backend="cpu", with_defaults=True)
    assert not has_errors(issues), [str(i) for i in issues if i.is_error]
    assert cfg.obs["obs_magnifications_list"] == [9.1]


def test_lambda_alias_conflict_cross_file():
    a = parse_text("lambda = 0.7", path="a.dat")
    b = parse_text("lambda_cosmo = 0.8", path="b.dat")
    cfg, issues = merge([a, b])
    assert any(i.code == "conflict" for i in issues)
    # first definition wins
    assert cfg.cosmology["lambda_cosmo"] == 0.7


def test_lambda_alias_conflict_same_file():
    assert _expect_syntax_error("lambda = 0.7\nlambda_cosmo = 0.8")


def test_pow_schema_has_zsfid_and_mass_on_re():
    from core.format import schema as S
    spec = S.model("pow")
    names = [p.name for p in spec.params]
    assert names == ["zs_fid", "x", "y", "e", "pa", "re", "gamma"]
    # 're' (index 5) is the mass-like / log-searched parameter
    assert spec.mass_positions == (5,)
    assert S.model("powpot").params == spec.params


# --------------------------------------------------------------------------- #
# user-defined shared variables
# --------------------------------------------------------------------------- #

_VAR_BASE = """
lens_z = 0.216
lens_x = {-0.1, 0.1}
'sers1': (1, 'sers', lens_z, {1e9,1e12}, lens_x, {-0.1,0.1}, 0.2, 30.0, 0.4, 1.0)
'sers2': (2, 'sers', lens_z, {1e9,1e12}, lens_x, {-0.05,0.05}, 0.2, 30.0, 0.4, 1.0)
"""


def test_user_var_resolves_to_shared_bounds():
    from core.format.values import SharedBounds
    cfg, issues = merge([parse_text(_VAR_BASE)])
    assert not any(i.is_error for i in issues), [str(i) for i in issues]
    p1 = cfg.components[0].params[1]
    p2 = cfg.components[1].params[1]
    assert isinstance(p1, SharedBounds) and isinstance(p2, SharedBounds)
    assert p1.name == p2.name == "lens_x"
    assert (p1.lo, p1.hi) == (-0.1, 0.1)
    assert isinstance(p1, Bounds)            # subclasses Bounds (is_optimizable etc.)
    assert cfg.user_vars == {"lens_x": Bounds(-0.1, 0.1)}


def test_user_var_fixed_value_still_inlines():
    cfg, issues = merge([parse_text(_VAR_BASE.replace("lens_x = {-0.1, 0.1}",
                                                      "lens_x = 0.05"))])
    assert not any(i.is_error for i in issues)
    assert cfg.components[0].params[1] == Fixed(0.05)
    assert cfg.components[1].params[1] == Fixed(0.05)
    assert cfg.user_vars == {}


def test_user_var_unknown_reference_still_errors():
    cfg, issues = merge([parse_text(_VAR_BASE.replace("lens_x = {-0.1, 0.1}\n", ""))])
    assert any(i.code == "unresolved_ref" for i in issues)


def test_user_var_cannot_reference_optimizable_schema_scalar():
    txt = _VAR_BASE.replace("lens_x = {-0.1, 0.1}", "source_x = {-0.1, 0.1}") \
                   .replace("lens_x,", "source_x,")
    cfg, issues = merge([parse_text(txt)])
    errs = [i for i in issues if i.code == "unresolved_ref"]
    assert errs and "optimizable scalar" in errs[0].message


def test_user_var_mixed_mass_linear_usage_is_error():
    txt = """
lens_z = 0.216
v1 = {0.01, 0.1}
'sers1': (1, 'sers', lens_z, v1, 0.0, 0.0, 0.2, 30.0, 0.4, 1.0)
'sers2': (2, 'sers', lens_z, 1e10, v1, 0.0, 0.2, 30.0, 0.4, 1.0)
"""
    cfg, issues = merge([parse_text(txt)])
    issues.extend(validate(cfg))
    assert any(i.code == "var_mixed_usage" for i in issues)


def test_user_var_unused_warns():
    cfg, issues = merge([parse_text("typo_x = {-0.1, 0.1}\nlens_z = 0.216\n"
                                    "'p1': (1, 'point', lens_z, 1e6, 0.0, 0.0)")])
    issues.extend(validate(cfg))
    assert any(i.code == "var_unused" and not i.is_error for i in issues)


def test_user_var_in_list_scalar_gets_clear_error():
    cfg, issues = merge([parse_text(
        "v = {1.0, 2.0}\nlens_z = 0.216\n"
        "obs_positions_mas_list = [[v, 0.4]]\n"
        "'p1': (1, 'point', lens_z, 1e6, v, 0.0)")])
    issues.extend(validate(cfg))
    msgs = [i.message for i in issues if i.code == "unresolved_ref"]
    assert msgs and "component tuples" in msgs[0]


def test_user_var_schema_scalar_ref_single_error():
    txt = _VAR_BASE.replace("lens_x = {-0.1, 0.1}", "source_x = {-0.1, 0.1}") \
                   .replace("lens_x,", "source_x,")
    cfg, issues = merge([parse_text(txt)])
    issues.extend(validate(cfg))
    errs = [i for i in issues if i.code == "unresolved_ref"]
    # one precise merge-time error per reference site, no contradictory
    # "unknown name" duplicate from the component check
    assert len(errs) == 2 and all("optimizable scalar" in e.message for e in errs)


def test_user_var_cross_file_reference():
    pf_vars = parse_text("lens_z = 0.216\nlens_x = {-0.1, 0.1}", path="vars.dat")
    pf_comp = parse_text(
        "'sers1': (1, 'sers', lens_z, 1e10, lens_x, 0.0, 0.2, 30.0, 0.4, 1.0)",
        path="comp.dat")
    from core.format.values import SharedBounds
    cfg, issues = merge([pf_vars, pf_comp])
    assert not any(i.is_error for i in issues), [str(i) for i in issues]
    assert isinstance(cfg.components[0].params[1], SharedBounds)


# --------------------------------------------------------------------------- #
# manual runner
# --------------------------------------------------------------------------- #

def _run_all() -> int:
    funcs = sorted(
        (n, f) for n, f in globals().items()
        if n.startswith("test_") and callable(f)
    )
    failures = 0
    for name, fn in funcs:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(funcs) - failures}/{len(funcs)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
