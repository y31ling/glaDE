"""Tests for extended-source (FITS) support.

    python core/tests/test_extend.py

Engine-dependent tests (c2calc_each parity, a live extend evaluation) are
skipped cleanly when glafic / the example FITS are unavailable.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.format import schema  # noqa: E402
from core.format.api import load_config  # noqa: E402
from core.format.parser import parse_text  # noqa: E402
from core.format.config import merge  # noqa: E402
from core.format.validate import is_extend_mode, validate  # noqa: E402
from core.format.values import Bounds, Fixed  # noqa: E402
from core.optimize.problem import OptProblem  # noqa: E402
from core.optimize.loss import ExtendLossConfig  # noqa: E402
from core.optimize.extend import (  # noqa: E402
    ExtendObjective,
    ExtendSpec,
    PointSource,
    build_extend_spec,
    parse_point_file_sources,
    write_point_file_from_arrays,
)
from core.translate.convert import glafic_to_glade  # noqa: E402
from core.translate.glafic_io import parse_glafic_python  # noqa: E402

_IVY = os.path.join(_ROOT, "InputFiles", "IvyProject")
_FITS = os.path.join(_IVY, "mock_HST_only_ER_32.fits")


def _glafic():
    try:
        import glafic  # noqa: PLC0415
        return glafic if hasattr(glafic, "c2calc_each") else None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #

def test_extend_models_registered():
    for k in ("extsersic", "extgauss", "exttophat", "extmoffat", "extjaffe"):
        spec = schema.model(k)
        assert spec is not None, k
        assert spec.category == schema.EXTEND_CATEGORY
        assert spec.gpu is False
        assert schema.is_extend_model(k)
    # extsersic glafic_key is the bare glafic name and params start with norm
    s = schema.model("extsersic")
    assert s.glafic_key == "sersic"
    assert s.params[0].name == "norm"
    assert [p.name for p in s.params] == ["norm", "x", "y", "e", "pa", "re", "n"]
    # a deflector model is NOT an extend model
    assert not schema.is_extend_model("sie")


def test_extend_blocked_on_gpu():
    assert "extsersic" not in schema.GPU_MODELS
    assert not schema.supports("gpu", "extsersic")
    assert schema.supports("cpu", "extsersic")


def test_new_scalar_keys_classify():
    for k in schema.EXTEND_FILE_KEYS + schema.OBS_EXTEND_ARRAY_KEYS:
        assert schema.classify_scalar(k) == "obs", k
    for k in schema.WEIGHT_KEYS + schema.SECONDARY_KEYS:
        assert schema.classify_scalar(k) == "algorithm", k


def test_missing_img_penalty_classifies_and_reads():
    # both spellings classify to the algorithm section
    assert schema.classify_scalar("missing_img_penalty") == "algorithm"
    assert schema.classify_scalar("MISSING_IMG_PENALTY") == "algorithm"
    cfg, _ = merge([parse_text(_EXTEND_DAT + "\nmissing_img_penalty = 42.0\n",
                               path="e.dat")])
    assert cfg.algorithm.get("missing_img_penalty") == 42.0
    assert ExtendLossConfig.from_cfg(cfg).missing_img_penalty == 42.0


class _FakeEngine:
    """A glafic stand-in: every driver call is a no-op except c2calc_each (which
    returns a fixed component tuple) and findimg_i (a fixed predicted count)."""

    def __init__(self, comp, n_pred):
        self._comp = comp
        self._n_pred = n_pred

    def __getattr__(self, _name):
        return lambda *a, **k: None          # init/set_*/model_init/readobs/quit

    def c2calc_each(self):
        return self._comp

    def findimg_i(self, _i, verb=0):
        return [(0.0, 0.0, 1.0, 0.0)] * self._n_pred


def test_extend_missing_penalty_graded():
    cfg, _ = merge([parse_text(_EXTEND_DAT, path="e.dat")])
    prob = OptProblem(cfg, extend_mode=True)
    mid = [0.5 * (d.lo + d.hi) for d in prob.dims]
    # one observed point source with 4 images; files are never opened (fake engine)
    spec = ExtendSpec(extended_file="x.fits", point_file="p.dat",
                      point_sources=[PointSource(zs=1.0, nimg=4)])
    PEN = 1.0e30
    good = (1.0, 2.0, 3.0, 0.5, 100.0, 1.0, 2.0, 0.0)       # all valid
    short = (PEN, PEN, 0.0, 0.0, 100.0, 1.0, 2.0, 0.0)      # under-imaged: pos/flux poisoned
    rng = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, PEN)          # physical-range violation

    def run(comp, n_pred, mp):
        lc = ExtendLossConfig(missing_img_penalty=mp)
        obj = ExtendObjective(prob, spec, lc)
        obj._engine = _FakeEngine(comp, n_pred)
        return obj.evaluate_one(mid)

    # full image count -> plain weighted c2calc sum (penalty irrelevant)
    assert math.isclose(run(good, 4, 100.0), ExtendLossConfig().combine(good),
                        rel_tol=1e-9)
    # under-imaged, penalty OFF -> glafic's flat 1e30 reject
    assert run(short, 3, 0.0) >= 1e15
    # under-imaged, penalty ON -> w_ext*pixel + w_prior*(prior_ext+prior_lens) + miss*mp
    base = 1.0 * 100.0 + 1.0 * (1.0 + 2.0)
    assert math.isclose(run(short, 3, 250.0), base + 1 * 250.0, rel_tol=1e-9)
    assert math.isclose(run(short, 2, 250.0), base + 2 * 250.0, rel_tol=1e-9)
    # a genuine physical-range violation is still a hard reject, even under-imaged
    assert run(rng, 3, 250.0) >= 1e15
    # a range penalty hiding in a PRIOR component (out[5]/out[6]=1e30, out[7]=0)
    # must ALSO hard-reject in the graded branch -- those components feed `base`.
    prior_poison_ext = (PEN, PEN, 0.0, 0.0, 100.0, PEN, 2.0, 0.0)
    prior_poison_lens = (PEN, PEN, 0.0, 0.0, 100.0, 1.0, PEN, 0.0)
    assert run(prior_poison_ext, 3, 250.0) >= 1e15
    assert run(prior_poison_lens, 3, 250.0) >= 1e15


# --------------------------------------------------------------------------- #
# format / validation
# --------------------------------------------------------------------------- #

_EXTEND_DAT = """
omega = 0.3
hubble = {0.6, 0.8}
xmin, ymin = -2.0, -2.0
xmax, ymax = 2.0, 2.0
pix_ext = 0.04
pix_poi = 0.04
maxlev = 1
source_z = 1.0
lens_z = 0.261
extended_file = 'img.fits'
constraint_file = 'pts.dat'
source_x = {-0.3, 0.3}
source_y = {-0.3, 0.3}
'sie1': (1, 'sie', lens_z, {120, 180}, 0.0, 0.0, {0.1, 0.5}, 180, 0.0)
'extsersic1': (2, 'extsersic', source_z, {20, 40}, {-0.1, 0.3}, 0.1, 0.132, 22.5, {0.1, 0.4}, 4.0)
"""


def test_extend_mode_detected_and_valid():
    cfg, _ = merge([parse_text(_EXTEND_DAT, path="e.dat")])
    assert is_extend_mode(cfg)
    issues = validate(cfg, backend="cpu")
    assert not any(i.is_error for i in issues), [str(i) for i in issues if i.is_error]


def test_extend_requires_fits():
    bad = _EXTEND_DAT.replace("extended_file = 'img.fits'\n", "")
    cfg, _ = merge([parse_text(bad, path="e.dat")])
    issues = validate(cfg, backend="cpu")
    assert any(i.code == "missing_extended_file" for i in issues)


def test_extend_blocks_gpu_backend():
    cfg, _ = merge([parse_text(_EXTEND_DAT, path="e.dat")])
    issues = validate(cfg, backend="gpu")
    assert any(i.code == "gpu_unsupported" for i in issues)


def test_point_arrays_not_required_in_extend_mode():
    # no obs arrays, no constraint_file -> still valid (pure extended fit)
    pure = _EXTEND_DAT.replace("constraint_file = 'pts.dat'\n", "")
    cfg, _ = merge([parse_text(pure, path="e.dat")])
    issues = validate(cfg, backend="cpu")
    assert not any(i.code == "missing_obs" for i in issues)


def test_partial_obs_arrays_blocked_in_extend_mode():
    # a partial set of glade point arrays must be rejected (would crash the
    # constraint-file writer, which needs all four).
    partial = _EXTEND_DAT.replace("constraint_file = 'pts.dat'\n",
                                  "obs_positions_mas_list = [[449.2, -276.9]]\n")
    cfg, _ = merge([parse_text(partial, path="e.dat")])
    issues = validate(cfg, backend="cpu")
    assert any(i.code == "missing_obs" for i in issues)


def test_extended_file_requires_extend_component():
    # extended_file set but only deflectors (no extend source) -> error.
    no_ext = """
omega = 0.3
source_z = 1.0
lens_z = 0.261
extended_file = 'img.fits'
constraint_file = 'pts.dat'
'sie1': (1, 'sie', lens_z, {120, 180}, 0.0, 0.0, {0.1, 0.5}, 180, 0.0)
"""
    cfg, _ = merge([parse_text(no_ext, path="e.dat")])
    assert is_extend_mode(cfg)
    issues = validate(cfg, backend="cpu")
    assert any(i.code == "no_extend_component" for i in issues)


# --------------------------------------------------------------------------- #
# problem (extend_mode)
# --------------------------------------------------------------------------- #

def test_problem_extend_mode_dims():
    cfg, _ = merge([parse_text(_EXTEND_DAT, path="e.dat")])
    prob = OptProblem(cfg, extend_mode=True)
    labels = [d.label for d in prob.dims]
    # source position is glafic-solved -> NOT a DE dimension in extend mode
    assert "source_x" not in labels and "source_y" not in labels
    # hubble (a Bounds) IS a dimension
    assert "hubble" in labels
    # component Bounds (lens + extend) are dimensions
    assert "sie1.sigma" in labels and "extsersic1.norm" in labels
    # the point-only problem WOULD include the source dims
    assert "source_x" in [d.label for d in OptProblem(cfg, extend_mode=False).dims]


def test_scene_routes_extend_components():
    cfg, _ = merge([parse_text(_EXTEND_DAT, path="e.dat")])
    prob = OptProblem(cfg, extend_mode=True)
    scene = prob.baseline_scene()
    types = [c.glafic_type for c in scene.components]
    ext_types = [c.glafic_type for c in scene.extends]
    assert "sie" in types and "sersic" not in types
    assert ext_types == ["sersic"]


# --------------------------------------------------------------------------- #
# loss weights
# --------------------------------------------------------------------------- #

def test_weighted_loss_combine():
    comp = (10.0, 20.0, 30.0, 1.0, 100.0, 2.0, 3.0, 0.0)  # pos,flux,td,pp,pix,pe,pl,pen
    # all weights 1 -> exactly the sum (== c2calc)
    assert math.isclose(ExtendLossConfig().combine(comp), sum(comp))
    # legacy point-only: only pos + flux
    legacy = ExtendLossConfig(w_pos=1, w_flux=1, w_td=0, w_ext=0, w_prior=0)
    assert math.isclose(legacy.combine(comp), 30.0)
    # penalty always added regardless of weights
    pen = (0.0,) * 7 + (1e30,)
    assert ExtendLossConfig(w_pos=0, w_flux=0, w_td=0, w_ext=0, w_prior=0).combine(pen) == 1e30


# --------------------------------------------------------------------------- #
# point-constraint file: parse + write-from-arrays round trip
# --------------------------------------------------------------------------- #

def test_point_file_roundtrip():
    text = "1 2 1.0 0.0\n0.45 -0.28 10.7 0.003 0.15 0.01 0.01 0\n-0.41 0.05 2.0 0.003 0.13 2.4 0.01 0\n"
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as f:
        f.write(text); p = f.name
    srcs = parse_point_file_sources(p)
    os.unlink(p)
    assert len(srcs) == 1 and math.isclose(srcs[0].zs, 1.0)
    assert srcs[0].nimg == 2


def test_read_point_file_positions():
    from core.optimize.extend import read_point_file_positions
    # the real IvyProject constraint file -> all 4 image (x, y) positions, in order
    pos = read_point_file_positions(os.path.join(_IVY, "pos+mag+td_point.dat"))
    assert pos.shape == (4, 2)
    assert math.isclose(pos[0, 0], 0.4492) and math.isclose(pos[0, 1], -0.2769)
    assert math.isclose(pos[1, 0], -0.4110) and math.isclose(pos[1, 1], 0.0526)
    assert math.isclose(pos[3, 0], 0.5820) and math.isclose(pos[3, 1], 0.0086)
    # tab-separated columns and a missing/short file degrade gracefully
    import tempfile as _t
    with _t.NamedTemporaryFile("w", suffix=".dat", delete=False) as f:
        f.write("2 1 0.4 0.0\n0.1\t0.2\t9 0.003 0.1 0 0 0\n"); q = f.name
    one = read_point_file_positions(q); os.unlink(q)
    assert one.shape == (1, 2) and math.isclose(one[0, 0], 0.1)


def test_write_point_file_from_arrays():
    glade = """
source_z = 1.0
obs_positions_mas_list = [[449.2, -276.9], [-411.0, 52.6]]
obs_magnifications_list = [10.766, 2.060]
obs_mag_errors_list = [0.155, 0.129]
obs_pos_sigma_mas_list = [3.0, 3.0]
obs_td_list = [0.0145, 2.4184]
obs_td_err_list = [0.01, 0.01]
obs_parity_list = [0, 0]
center_offset_x = 0.0
center_offset_y = 0.0
obs_x_flip = False
'sie1': (1, 'sie', 0.26, 150, 0, 0, 0.3, 180, 0)
extended_file = 'x.fits'
"""
    cfg, _ = merge([parse_text(glade, path="g.dat")])
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as f:
        out = f.name
    n = write_point_file_from_arrays(cfg, out)
    rows = [ln.split() for ln in open(out).read().splitlines() if ln.strip()]
    os.unlink(out)
    assert n == 1
    assert rows[0][0] == "1" and rows[0][1] == "2"        # header: 1 source, 2 images
    # first image: 449.2 mas -> 0.4492 arcsec, flux 10.766, pos_sig 3mas -> 0.003
    assert math.isclose(float(rows[1][0]), 0.4492, abs_tol=1e-9)
    assert math.isclose(float(rows[1][2]), 10.766, abs_tol=1e-9)
    assert math.isclose(float(rows[1][3]), 0.003, abs_tol=1e-9)
    assert math.isclose(float(rows[1][5]), 0.0145, abs_tol=1e-9)   # time delay


# --------------------------------------------------------------------------- #
# import: glafic python driver script -> glade extend config
# --------------------------------------------------------------------------- #

_PYSCRIPT = """
import glafic
extended_file = '/abs/path/host.fits'
constraint_file = '/abs/path/pts.dat'
prior_file = '/abs/path/prior.dat'
glafic.init(0.3, 0.7, -1.0, 0.7, 'out', -2.0, -2.0, 2.0, 2.0, 0.04, 0.04, 1, verb=0)
glafic.set_secondary('chi2_splane 1', verb=0)
glafic.set_secondary('hvary 1', verb=0)
glafic.set_secondary('obs_gain 1.6', verb=0)
glafic.startup_setnum(1, 2, 1)
glafic.set_lens(1, 'sie', 0.261, 150.0, 0.0, 0.0, 0.3, 180.0, 0.0, 0.0)
glafic.set_extend(1, 'sersic', 1.0, 30.0, 0.1, 0.1, 0.132, 22.5, 0.2, 4.0)
glafic.set_extend(2, 'sersic', 1.0, 15.0, 0.1, 0.1, 0.365, -13.28, 0.8, 1.0)
glafic.set_point(1, 1.0, 0.0, 0.0)
glafic.setopt_lens(1, 0, 1, 1, 1, 1, 1, 0, 0)
glafic.setopt_extend(1, 0, 1, 1, 1, 1, 1, 1, 1)
glafic.setopt_point(1, 0, 1, 1)
glafic.readobs_extend(extended_file)
glafic.readobs_point(constraint_file)
glafic.parprior(prior_file)
glafic.optimize()
glafic.quit()
"""


def test_parse_python_script():
    m = parse_glafic_python(_PYSCRIPT)
    assert len(m.lenses) == 1 and m.lenses[0].type == "sie"
    assert len(m.extends) == 2 and m.extends[0].type == "sersic"
    assert math.isclose(m.extends[0].params[0], 30.0)         # norm
    assert os.path.basename(m.extended_file) == "host.fits"
    assert m.constraint_file and m.prior_file
    assert m.secondary.get("chi2_splane") == 1
    assert m.secondary.get("hvary") == 1
    # opt flags captured (sie: sigma,x,y,e,pa free; rcore fixed)
    assert m.lenses[0].opt == [0, 1, 1, 1, 1, 1, 0, 0]


def test_import_python_to_glade_extend():
    out = glafic_to_glade(_PYSCRIPT)
    cfg, issues = merge([parse_text(out["model"], path="m.dat"),
                         parse_text(out["obs"], path="o.dat")])
    assert is_extend_mode(cfg)
    # hvary=1 -> hubble became an optimizable Bounds
    assert isinstance(cfg.cosmology.get("hubble"), Bounds)
    # two extend components + one lens, paths reduced to basenames
    types = [c.type for c in cfg.components]
    assert types.count("extsersic") == 2 and "sie" in types
    assert cfg.get("extended_file") == "host.fits"
    # free glafic params became degenerate {v, v} bounds
    sie = next(c for c in cfg.components if c.type == "sie")
    assert isinstance(sie.params[0], Bounds) and sie.params[0].lo == sie.params[0].hi
    assert isinstance(sie.params[5], Fixed)   # rcore was fixed (opt flag 0)


# --------------------------------------------------------------------------- #
# live (glafic) : c2calc_each parity + a real extend evaluation
# --------------------------------------------------------------------------- #

def test_c2calc_each_sums_to_c2calc():
    gl = _glafic()
    if gl is None or not os.path.exists(_FITS):
        print("    (skipped: glafic/c2calc_each or example FITS unavailable)")
        return
    gl.init(0.3, 0.7, -1.0, 0.7, "temp_test_each", -2.0, -2.0, 2.0, 2.0,
            0.04, 0.04, 1, verb=0)
    for s in ("chi2_splane 1", "chi2_usemag -1", "obs_gain 1.6", "obs_readnoise 3.08"):
        gl.set_secondary(s, verb=0)
    gl.startup_setnum(1, 2, 1)
    gl.set_lens(1, "sie", 0.261, 150.0, 0.0, 0.0, 0.3, 180.0, 0.0, 0.0)
    gl.set_extend(1, "sersic", 1.0, 30.0, 0.1, 0.1, 0.132, 22.5, 0.2, 4.0)
    gl.set_extend(2, "sersic", 1.0, 15.0, 0.1, 0.1, 0.365, -13.28, 0.8, 1.0)
    gl.set_point(1, 1.0, 0.0, 0.0)
    gl.setopt_point(1, 0, 1, 1)
    gl.model_init(verb=0)
    gl.readobs_extend(_FITS)
    gl.readobs_point(os.path.join(_IVY, "pos+mag+td_point.dat"))
    total = gl.c2calc()
    comp = gl.c2calc_each()
    gl.quit()
    for f in os.listdir("."):
        if f.startswith("temp_test_each"):
            os.remove(f)
    assert math.isclose(total, sum(comp), rel_tol=1e-6, abs_tol=1e-3)
    assert len(comp) == 8


def test_live_extend_objective_runs():
    gl = _glafic()
    if gl is None or not os.path.exists(_FITS):
        print("    (skipped: glafic/example FITS unavailable)")
        return
    from core.optimize.extend import ExtendObjective
    dat = """
omega = 0.3
hubble = 0.7
xmin, ymin = -2.0, -2.0
xmax, ymax = 2.0, 2.0
pix_ext = 0.04
pix_poi = 0.04
maxlev = 1
source_z = 1.0
lens_z = 0.261343256161012
extended_file = 'mock_HST_only_ER_32.fits'
constraint_file = 'pos+mag+td_point.dat'
chi2_splane = 1
chi2_usemag = -1
obs_gain = 1.6
obs_readnoise = 3.08
source_x = {-0.3, 0.3}
source_y = {-0.3, 0.3}
'sie1': (1, 'sie', lens_z, {145, 155}, 0.0, 0.0, 0.3, 180, 0.0)
'extsersic1': (2, 'extsersic', source_z, 30.0, 0.1, 0.1, 0.132, 22.5, 0.2, 4.0)
'extsersic2': (3, 'extsersic', source_z, 15.0, 0.1, 0.1, 0.365, -13.28, 0.8, 1.0)
"""
    p = os.path.join(_IVY, "_test_live.dat")
    open(p, "w").write(dat)
    cfg, issues = load_config([p], backend="cpu")
    assert not any(i.is_error for i in issues), [str(i) for i in issues if i.is_error]
    prob = OptProblem(cfg, extend_mode=True)
    spec = build_extend_spec(cfg, base_dir=_IVY)
    obj = ExtendObjective(prob, spec, ExtendLossConfig.from_cfg(cfg))
    # candidates are in SEARCH space (log10 for mass-like dims like sie sigma)
    mid = [0.5 * (d.lo + d.hi) for d in prob.dims]
    comp = obj.components_for(mid)
    val = obj.evaluate_one(mid)
    os.unlink(p)
    if spec._own_point_file and spec.point_file and os.path.exists(spec.point_file):
        os.unlink(spec.point_file)
    assert comp is not None and len(comp) == 8
    assert math.isfinite(val) and val > 0
    # a feasible candidate (no glafic range/image penalty) -> loss == weighted sum
    assert all(c < 1e29 for c in comp), f"unexpected penalty: {comp}"
    assert math.isclose(obj.loss_cfg.combine(comp), val, rel_tol=1e-6)


def test_extend_mcmc_log_prob_and_sampling():
    gl = _glafic()
    if gl is None or not os.path.exists(_FITS):
        print("    (skipped: glafic/example FITS unavailable)")
        return
    import numpy as np
    from core.mcmc import MCMCConfig, run_mcmc
    from core.mcmc.log_prob import ExtendLogProbability
    from core.optimize.extend import build_extend_spec
    dat = """
omega = 0.3
hubble = {0.6, 0.8}
xmin, ymin = -2.0, -2.0
xmax, ymax = 2.0, 2.0
pix_ext = 0.04
pix_poi = 0.04
maxlev = 1
source_z = 1.0
lens_z = 0.261343256161012
extended_file = 'mock_HST_only_ER_32.fits'
constraint_file = 'pos+mag+td_point.dat'
chi2_splane = 1
chi2_usemag = -1
obs_gain = 1.6
obs_readnoise = 3.08
source_x = {-0.3, 0.3}
source_y = {-0.3, 0.3}
'sie1': (1, 'sie', lens_z, {145, 155}, 0.0, 0.0, 0.3, 180, 0.0)
'extsersic1': (2, 'extsersic', source_z, 30.0, 0.1, 0.1, 0.132, 22.5, 0.2, 4.0)
'extsersic2': (3, 'extsersic', source_z, 15.0, 0.1, 0.1, 0.365, -13.28, 0.8, 1.0)
"""
    p = os.path.join(_IVY, "_test_mcmc.dat")
    open(p, "w").write(dat)
    cfg, _ = load_config([p], backend="cpu")
    prob = OptProblem(cfg, extend_mode=True)
    spec = build_extend_spec(cfg, base_dir=_IVY)
    lp = ExtendLogProbability(prob, spec, ExtendLossConfig.from_cfg(cfg))
    mid = [0.5 * (d.lo + d.hi) for d in prob.dims]
    val = lp(mid)
    assert val < 0 and math.isfinite(val)              # -0.5 * positive loss
    # outside the bounds -> -inf
    out = list(mid); out[0] = prob.dims[0].hi + 1.0
    assert lp(out) == float("-inf")
    # a tiny end-to-end sample (serial, no pool)
    mcfg = MCMCConfig(nwalkers=6, nsteps=3, burnin=1, thin=1, workers=1,
                      progress=False, seed=3)
    res = run_mcmc(prob, None, ExtendLossConfig.from_cfg(cfg), backend="cpu",
                   best_x=None, mcmc_cfg=mcfg, extend_spec=spec)
    os.unlink(p)
    if spec._own_point_file and spec.point_file and os.path.exists(spec.point_file):
        os.unlink(spec.point_file)
    assert res.samples.shape[1] == prob.ndim
    assert set(res.summary) == {"hubble", "sie1.sigma"}


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
