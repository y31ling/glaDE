"""Tests for the V0.7 UnitSetting unit-profile system.

    python -m pytest core/tests/test_units.py
    python core/tests/test_units.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from core.format import load_config  # noqa: E402
from core.format.values import Bounds, Fixed  # noqa: E402
from core.optimize import OptProblem, build_obs  # noqa: E402

BASE = """
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
obs_magnifications_list = [-35.6, 15.7, -7.5, 9.1]
obs_mag_errors_list = [2.1, 1.3, 1.0, 1.1]
center_offset_x = 0.0
center_offset_y = 0.0
obs_x_flip = False
"""

# canonical-units body (engine convention)
CANON = BASE + """
source_x = 0.03
source_y = 0.024
obs_positions_mas_list = [[-266.0, 0.4], [118.8, -221.9], [238.3, 227.3], [-126.2, 319.7]]
obs_pos_sigma_mas_list = [0.41, 0.86, 2.23, 3.11]
'sers1': (1, 'sers', lens_z, 2.0e10, -0.003, 0.027, 0.3, 112.0, 0.39, 1.06)
'point1': (2, 'point', lens_z, {1e5, 1e7}, {-0.30, -0.20}, 0.01)
"""


def _load(text: str, profile: dict | None, profile_name: str = "myunits"):
    with tempfile.TemporaryDirectory(prefix="glade_units_") as tmp:
        dat = os.path.join(tmp, "cfg.dat")
        body = text
        if profile is not None:
            body = f"UnitSetting = '{profile_name}'\n" + text
            with open(os.path.join(tmp, profile_name + ".units.json"), "w") as fh:
                json.dump({"format": "glade-units-v1", "units": profile}, fh)
        with open(dat, "w", encoding="utf-8") as fh:
            fh.write(body)
        return load_config([dat], backend="cpu", with_defaults=True)


def test_default_profile_is_identity():
    cfg_a, iss_a = _load(CANON, None)
    cfg_b, iss_b = _load(CANON, {})            # explicit empty profile
    assert not any(i.is_error for i in iss_a + iss_b)
    assert cfg_a.obs["obs_positions_mas_list"] == cfg_b.obs["obs_positions_mas_list"]
    pa = cfg_a.components[1].params
    pb = cfg_b.components[1].params
    assert pa[0].lo == pb[0].lo and pa[1].lo == pb[1].lo
    assert cfg_b.components[1].unit_scales is None


def test_obs_pos_arcsec_profile():
    arc = BASE + """
source_x = 0.03
source_y = 0.024
obs_positions_mas_list = [[-0.266, 0.0004], [0.1188, -0.2219], [0.2383, 0.2273], [-0.1262, 0.3197]]
obs_pos_sigma_mas_list = [0.00041, 0.00086, 0.00223, 0.00311]
'sers1': (1, 'sers', lens_z, 2.0e10, -0.003, 0.027, 0.3, 112.0, 0.39, 1.06)
'point1': (2, 'point', lens_z, {1e5, 1e7}, {-0.30, -0.20}, 0.01)
"""
    cfg, issues = _load(arc, {"obs_pos": "arcsec"})
    assert not any(i.is_error for i in issues), [str(i) for i in issues]
    obs = build_obs(cfg)
    assert abs(obs.positions[0, 0] - (-0.266)) < 1e-12
    assert abs(obs.pos_sigma_mas[0] - 0.41) < 1e-9


def test_mass_msun_profile():
    cfg, issues = _load(CANON, {"mass": "msun"})
    assert not any(i.is_error for i in issues)
    sers = cfg.components[0]
    # fixed literal mass authored as 2e10 Msun -> engine 2e10 * h
    assert abs(sers.params[0].value - 2.0e10 * 0.7) < 1e-3
    # optimizable mass bounds scale too (search happens in engine units)
    pm = cfg.components[1]
    assert abs(pm.params[0].lo - 1e5 * 0.7) < 1e-9
    assert abs(pm.params[0].hi - 1e7 * 0.7) < 1e-3
    # angular params untouched
    assert abs(sers.params[1].value - (-0.003)) < 1e-15
    assert abs(sers.params[5].value - 0.39) < 1e-15


def test_mass_msun_needs_fixed_hubble():
    text = CANON.replace("hubble = 0.7", "hubble = {0.6, 0.8}")
    cfg, issues = _load(text, {"mass": "msun"})
    assert any(i.code == "unit_profile_bad" and "hubble" in i.message
               for i in issues), [str(i) for i in issues]


def test_comp_pos_mas_profile():
    mas = BASE + """
source_x = 0.03
source_y = 0.024
obs_positions_mas_list = [[-266.0, 0.4], [118.8, -221.9], [238.3, 227.3], [-126.2, 319.7]]
obs_pos_sigma_mas_list = [0.41, 0.86, 2.23, 3.11]
'sers1': (1, 'sers', lens_z, 2.0e10, -3.0, 27.0, 0.3, 112.0, 390.0, 1.06)
'point1': (2, 'point', lens_z, {1e5, 1e7}, {-300.0, -200.0}, 10.0)
"""
    cfg, issues = _load(mas, {"comp_pos": "mas"})
    assert not any(i.is_error for i in issues), [str(i) for i in issues]
    sers = cfg.components[0]
    assert abs(sers.params[1].value - (-0.003)) < 1e-15   # x mas -> arcsec
    assert abs(sers.params[5].value - 0.39) < 1e-15       # re mas -> arcsec
    assert abs(sers.params[3].value - 0.3) < 1e-15        # e untouched
    assert abs(sers.params[4].value - 112.0) < 1e-12      # pa untouched
    pm = cfg.components[1]
    assert abs(pm.params[1].lo - (-0.30)) < 1e-15         # bounds scaled
    assert abs(pm.params[0].lo - 1e5) < 1e-9              # mass untouched


def test_src_pos_mas_profile():
    text = CANON.replace("source_x = 0.03", "source_x = 30.0") \
                .replace("source_y = 0.024", "source_y = 24.0")
    cfg, issues = _load(text, {"src_pos": "mas"})
    assert not any(i.is_error for i in issues)
    assert abs(float(cfg.source["source_x"]) - 0.03) < 1e-15
    assert abs(float(cfg.source["source_y"]) - 0.024) < 1e-15


def test_shared_variable_slot_units():
    """A dimensionless shared var referenced from a comp-position slot under a
    mas profile: the search dimension keeps the raw numbers, the scene
    injection applies the slot factor (roadmap: insertion decides the unit)."""
    text = BASE + """
source_x = 0.03
source_y = 0.024
obs_positions_mas_list = [[-266.0, 0.4], [118.8, -221.9], [238.3, 227.3], [-126.2, 319.7]]
obs_pos_sigma_mas_list = [0.41, 0.86, 2.23, 3.11]
shift = {-300.0, -200.0}
'sers1': (1, 'sers', lens_z, 2.0e10, -3.0, 27.0, 0.3, 112.0, 390.0, 1.06)
'point1': (2, 'point', lens_z, {1e5, 1e7}, shift, 10.0)
"""
    cfg, issues = _load(text, {"comp_pos": "mas"})
    assert not any(i.is_error for i in issues), [str(i) for i in issues]
    pm = cfg.components[1]
    assert pm.unit_scales is not None and abs(pm.unit_scales[1] - 1e-3) < 1e-18
    problem = OptProblem(cfg)
    labels = [d.label for d in problem.dims]
    assert "shift" in labels
    k = labels.index("shift")
    # the shared dim stays dimensionless (raw mas numbers)
    assert problem.dims[k].lo == -300.0 and problem.dims[k].hi == -200.0
    x = np.array([d.midpoint_value() for d in problem.dims])
    x[k] = -250.0
    # rebuild the candidate in SEARCH space (mass dims are log10 already)
    xs = []
    for i, d in enumerate(problem.dims):
        xs.append(np.log10(x[i]) if d.log else x[i])
    scene = problem.make_scene(np.array(xs))
    # point1 x = -250 (dimensionless) * 1e-3 (slot factor) = -0.25 arcsec
    assert abs(scene.components[1].params[1] - (-0.25)) < 1e-12


def test_expr_params_not_rescaled():
    text = BASE + """
source_x = 0.03
source_y = 0.024
obs_positions_mas_list = [[-266.0, 0.4], [118.8, -221.9], [238.3, 227.3], [-126.2, 319.7]]
obs_pos_sigma_mas_list = [0.41, 0.86, 2.23, 3.11]
'sers1': (1, 'sers', lens_z, 2.0e10, -3.0, 27.0, 0.3, 112.0, 390.0, 1.06)
'point1': (2, 'point', lens_z, {1e5, 1e7}, img1_x, img1_y)
"""
    cfg_mas, iss = _load(text, {"comp_pos": "mas"})
    assert not any(i.is_error for i in iss), [str(i) for i in iss]
    cfg_def, _ = _load(text.replace("-3.0, 27.0", "-0.003, 0.027")
                       .replace("390.0", "0.39"), None)
    # the img1_x expression is engine-frame: identical under both profiles
    pm_mas = cfg_mas.components[1].params[1]
    pm_def = cfg_def.components[1].params[1]
    assert isinstance(pm_mas, Fixed) and isinstance(pm_def, Fixed)
    assert abs(pm_mas.value - pm_def.value) < 1e-15


def test_missing_profile_is_an_error():
    cfg, issues = _load(CANON, None, "nope")
    # write UnitSetting but no profile file
    with tempfile.TemporaryDirectory(prefix="glade_units_") as tmp:
        dat = os.path.join(tmp, "cfg.dat")
        with open(dat, "w", encoding="utf-8") as fh:
            fh.write("UnitSetting = 'no_such_profile'\n" + CANON)
        cfg, issues = load_config([dat], backend="cpu")
    assert any(i.code == "unit_profile_missing" for i in issues), \
        [str(i) for i in issues]


def test_glade_output_written_canonical():
    from core.optimize import optimize
    from core.report import write_glade_output
    from test_optimize import FakeBackend

    mas = BASE + """
source_x = 0.03
source_y = 0.024
obs_positions_mas_list = [[-266.0, 0.4], [118.8, -221.9], [238.3, 227.3], [-126.2, 319.7]]
obs_pos_sigma_mas_list = [0.41, 0.86, 2.23, 3.11]
'point1': (1, 'point', lens_z, {1e5, 1e7}, {-300.0, -200.0}, {-50.0, 50.0})
"""
    cfg, issues = _load(mas, {"comp_pos": "mas"})
    assert not any(i.is_error for i in issues)
    obs = build_obs(cfg)
    res = optimize(cfg, backend=FakeBackend(obs.positions, obs.magnifications),
                   de_overrides={"maxiter": 5, "polish": False},
                   record_population=False)
    with tempfile.TemporaryDirectory(prefix="glade_out_units_") as tmp:
        path = write_glade_output(res, os.path.join(tmp, "run-u"))
        text = open(path, encoding="utf-8").read()
        assert "UnitSetting" not in text
        cfg2, iss2 = load_config([path], backend="cpu", with_defaults=True)
        assert not any(i.is_error for i in iss2), [str(i) for i in iss2]
        # positions written in engine arcsec (|x| < 1), not raw mas numbers
        assert abs(cfg2.components[0].params[1].value) < 1.0


def _run_all() -> int:
    funcs = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    failures = 0
    for name, fn in funcs:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(funcs) - failures}/{len(funcs)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
