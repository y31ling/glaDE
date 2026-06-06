"""Tests for glafic <-> glade translation.

    python -m pytest core/tests/test_translate.py
    python core/tests/test_translate.py
"""
from __future__ import annotations

import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.format import has_errors, lint_text, parse_text  # noqa: E402
from core.format.config import apply_defaults, merge  # noqa: E402
from core.format.values import Bounds, Fixed  # noqa: E402
from core.translate import (  # noqa: E402
    glade_to_glafic,
    glafic_to_glade,
    parse_glafic_input,
)


SN_BESTFIT = """
lens   sers    0.2160  9.896617e+09  2.656977e-03  2.758473e-02  2.986760e-01  1.124730e+02  3.939718e-01  1.057760e+00
lens   sers    0.2160  2.555580e+10  2.656977e-03  2.758473e-02  4.242340e-01  5.396370e+01  1.538855e+00  1.000000e+00
lens   sie     0.2160  1.183382e+02  2.656977e-03  2.758473e-02  1.571203e-01  2.920348e+01  0.000000e+00  0.000000e+00
point  0.4090  2.685497e-03  2.443616e-02
"""

GLAFIC_INPUT_WITH_STARTUP = """
omega      0.3
lambda     0.7
weos       -1.0
hubble     0.7
prefix     out
xmin       -0.5
ymin       -0.5
xmax        0.5
ymax        0.5
pix_ext     0.01
pix_poi     0.2
maxlev      5
startup 1 0 1
lens   point   0.2160  1.0e6  -0.25  0.0  0.0  0.0  0.0  0.0
point  0.4090  0.0027  0.0244
start_setopt
0 1 1 1 0 0 0 0
0 0 0
end_setopt
"""


def test_parse_glafic_canonical_lens_line():
    m = parse_glafic_input(SN_BESTFIT)
    assert len(m.lenses) == 3
    assert m.lenses[0].type == "sers"
    assert math.isclose(m.lenses[0].z, 0.216)
    assert math.isclose(m.lenses[0].params[0], 9.896617e9)   # sigma/mass
    assert len(m.lenses[0].params) == 7
    assert m.source_z == 0.409 and math.isclose(m.source_x, 2.685497e-03)


def test_parse_glafic_old_id_variant():
    # older 'lens TYPE id z p1..p7' (9 numbers) -> drop the leading id
    old = "lens sers 1 0.2160 9.9e9 0.0 0.0 0.3 112.0 0.39 1.05\npoint 0.409 0.0 0.0\n"
    m = parse_glafic_input(old)
    assert m.lenses[0].type == "sers"
    assert math.isclose(m.lenses[0].z, 0.216)
    assert math.isclose(m.lenses[0].params[0], 9.9e9)


def test_glafic_to_glade_produces_parseable_dat():
    out = glafic_to_glade(SN_BESTFIT)
    model_text = out["model"]
    # the translated model must parse cleanly with the glade parser
    pf = parse_text(model_text, path="translated.dat")
    assert len(pf.components) == 3
    assert pf.components[0].type == "sers"
    # all locked (no opt matrix) -> Fixed params
    assert all(isinstance(p, Fixed) for p in pf.components[0].params)
    # lens_z defined and referenced
    assert any(a.name == "lens_z" for a in pf.assignments)


def test_glafic_optmatrix_becomes_bounds():
    out = glafic_to_glade(GLAFIC_INPUT_WITH_STARTUP)
    pf = parse_text(out["model"], path="t.dat")
    comp = pf.components[0]
    assert comp.type == "point"
    # opt row "0 1 1 1 ..." => z fixed, p1(mass) p2(x) p3(y) optimizable
    assert isinstance(comp.params[0], Bounds)   # mass flagged
    assert isinstance(comp.params[1], Bounds)   # x flagged
    assert isinstance(comp.params[2], Bounds)   # y flagged
    # flagged value becomes {v, v}
    assert comp.params[0].lo == comp.params[0].hi == 1.0e6


def test_glade_to_glafic_midpoint_geometric_for_mass():
    glade = """
omega = 0.3
lambda_cosmo = 0.7
weos = -1.0
hubble = 0.7
source_z = 0.409
lens_z = 0.216
source_x = 0.0027
source_y = 0.0244
'point1': (1, 'point', lens_z, {1e5, 1e7}, {-0.30, -0.20}, 0.0)
"""
    cfg, _ = merge([parse_text(glade, path="g.dat")])
    apply_defaults(cfg)
    out = glade_to_glafic(cfg)
    m = parse_glafic_input(out["model"])
    assert len(m.lenses) == 1
    # mass {1e5,1e7} -> geometric mean 1e6
    assert math.isclose(m.lenses[0].params[0], 1.0e6, rel_tol=1e-9)
    # x {-0.30,-0.20} -> arithmetic mean -0.25
    assert math.isclose(m.lenses[0].params[1], -0.25, rel_tol=1e-9)
    assert m.source_z == 0.409


def test_round_trip_glafic_glade_glafic_preserves_model():
    m1 = parse_glafic_input(SN_BESTFIT)
    glade = glafic_to_glade(SN_BESTFIT)["model"]
    cfg, _ = merge([parse_text(glade, path="g.dat")])
    apply_defaults(cfg)
    m2 = parse_glafic_input(glade_to_glafic(cfg)["model"])
    assert len(m1.lenses) == len(m2.lenses) == 3
    for a, b in zip(m1.lenses, m2.lenses):
        assert a.type == b.type
        assert math.isclose(a.z, b.z, rel_tol=1e-6)
        for pa, pb in zip(a.params, b.params):
            assert math.isclose(pa, pb, rel_tol=1e-5, abs_tol=1e-9)
    assert math.isclose(m1.source_x, m2.source_x, rel_tol=1e-6)


def test_glade_to_glafic_obs_mas_to_arcsec():
    glade = """
omega = 0.3
source_z = 0.409
lens_z = 0.216
source_x = 0.0
source_y = 0.0
obs_positions_mas_list = [[-266.0, 0.4], [118.8, -221.9]]
obs_magnifications_list = [-35.6, 15.7]
obs_mag_errors_list = [2.1, 1.3]
obs_pos_sigma_mas_list = [0.41, 0.86]
center_offset_x = 0.0
center_offset_y = 0.0
obs_x_flip = False
'point1': (1, 'point', lens_z, 1e6, 0.1, 0.1)
"""
    cfg, _ = merge([parse_text(glade, path="g.dat")])
    apply_defaults(cfg)
    out = glade_to_glafic(cfg)
    assert out["obs"]
    m = parse_glafic_input(out["obs"])
    assert m.obs is not None and len(m.obs.images) == 2
    # -266 mas -> -0.266 arcsec
    assert math.isclose(m.obs.images[0][0], -0.266, rel_tol=1e-6)


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
