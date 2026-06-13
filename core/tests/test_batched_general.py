"""Tests for the generalized batched GPU objective (V0.50).

Covers the widened ``can_batch_gpu`` predicate (pure config logic, no torch)
and — when torch + Rhongomyniad are importable (CPU torch is enough) — the
mode selection inside ``BatchedGPUObjective``, batched-vs-per-candidate loss
parity for a free (non-point) lens model, the optimizable source position,
and chunking invariance.

    python -m pytest core/tests/test_batched_general.py
    python core/tests/test_batched_general.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# rhongomyniad lives at Rhongomyniad/rhongomyniad and is not pip-installed:
# add it like webui/runjob.py does, so the torch tests run without env.sh.
for _p in (_ROOT, os.path.join(_ROOT, "Rhongomyniad")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from core.format import lint_text  # noqa: E402
from core.optimize.batched import can_batch_gpu  # noqa: E402

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
source_x = 0.03
source_y = 0.024
obs_positions_mas_list = [[-266.0, 0.4], [118.8, -221.9], [238.3, 227.3], [-126.2, 319.7]]
obs_magnifications_list = [-35.6, 15.7, -7.5, 9.1]
obs_mag_errors_list = [2.1, 1.3, 1.0, 1.1]
obs_pos_sigma_mas_list = [0.41, 0.86, 2.23, 3.11]
center_offset_x = 0.0
center_offset_y = 0.0
obs_x_flip = False
"""

# a free main lens: every SIE parameter optimizable (no point masses at all).
# sigma is kept moderate so the Einstein radius stays well inside the
# [-0.5, 0.5] arcsec grid box (sigma=250 km/s would put it at ~0.8").
FREE_SIE = _BASE + """
missing_img_penalty = 50000
'sie1': (1, 'sie', lens_z, {80, 160}, {-0.05, 0.05}, {-0.05, 0.05}, {0.01, 0.3}, {0, 180})
"""

# locked main lens + one optimizable point mass (the legacy fast path)
LOCKED_PLUS_POINT = _BASE + """
'sie1': (1, 'sie', lens_z, 250.0, 0.0, 0.0, 0.2, 30.0)
'point1': (2, 'point', lens_z, {1e5, 1e7}, {-0.30, -0.20}, {-0.05, 0.05})
"""


def _cfg(text):
    cfg, issues = lint_text(text, backend="gpu", with_defaults=True)
    assert not any(i.is_error for i in issues), [str(i) for i in issues]
    return cfg


def _torch_ready() -> bool:
    try:
        import torch  # noqa: F401, PLC0415
        import rhongomyniad  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# can_batch_gpu (config-only, no torch)
# ---------------------------------------------------------------------------

def test_predicate_accepts_free_main_lens():
    ok, reason = can_batch_gpu(_cfg(FREE_SIE))
    assert ok, reason


def test_predicate_accepts_optimizable_source():
    ok, reason = can_batch_gpu(_cfg(
        FREE_SIE.replace("source_x = 0.03", "source_x = {-0.1, 0.1}")))
    assert ok, reason


def test_predicate_accepts_mixed_models():
    txt = _BASE + """
'sers1': (1, 'sers', lens_z, {1e9, 1e12}, 0.0, 0.0, {0.01, 0.5}, {0, 180}, {0.1, 0.5}, {0.5, 1.5})
'anfw1': (2, 'anfw', lens_z, {1e9, 1e13}, 0.0, 0.0, 0.2, 30.0, {0.5, 10})
'point1': (3, 'point', lens_z, {1e5, 1e7}, -0.25, 0.0)
"""
    ok, reason = can_batch_gpu(_cfg(txt))
    assert ok, reason


def test_predicate_keeps_point_only_configs():
    ok, reason = can_batch_gpu(_cfg(LOCKED_PLUS_POINT))
    assert ok, reason


def test_predicate_rejects_optimizable_hubble():
    ok, reason = can_batch_gpu(_cfg(
        FREE_SIE.replace("hubble = 0.7", "hubble = {0.5, 0.9}")))
    assert not ok and "hubble" in reason


def test_predicate_rejects_optimizable_component_z():
    txt = _BASE + """
'sie1': (1, 'sie', {0.1, 0.4}, {150, 350}, 0.0, 0.0, 0.2, 30.0)
"""
    ok, reason = can_batch_gpu(_cfg(txt))
    assert not ok and "redshift" in reason


def test_predicate_rejects_optimizable_zs_fid():
    # pert's p1 is the fiducial source redshift, a CPU-scalar in the kernels
    txt = _BASE + """
'sie1': (1, 'sie', lens_z, 250.0, 0.0, 0.0, 0.2, 30.0)
'pert1': (2, 'pert', lens_z, {0.3, 0.5}, 0.0, 0.0, {0.001, 0.1}, {0, 180})
"""
    ok, reason = can_batch_gpu(_cfg(txt))
    assert not ok and "fiducial" in reason


def test_predicate_rejects_multi_plane():
    txt = _BASE + """
'sie1': (1, 'sie', 0.216, {150, 350}, 0.0, 0.0, 0.2, 30.0)
'sie2': (2, 'sie', 0.30, 250.0, 0.1, 0.1, 0.2, 30.0)
"""
    ok, reason = can_batch_gpu(_cfg(txt))
    assert not ok and "single-plane" in reason


# ---------------------------------------------------------------------------
# objective behaviour (needs torch + rhongomyniad; CPU torch is enough)
# ---------------------------------------------------------------------------

def _objective(text):
    from core.optimize.batched import BatchedGPUObjective
    from core.optimize.loss import LossConfig
    from core.optimize.problem import OptProblem
    from core.optimize.scene import build_obs
    cfg = _cfg(text)
    problem = OptProblem(cfg)
    obj = BatchedGPUObjective(problem, build_obs(cfg), LossConfig.from_cfg(cfg))
    return cfg, problem, obj


def test_mode_selection():
    if not _torch_ready():
        print("  (skipped: torch/rhongomyniad not importable)")
        return
    _, _, obj = _objective(LOCKED_PLUS_POINT)
    obj._build_cache()
    assert obj._cache["legacy"] and obj._cache["points"]

    _, _, obj = _objective(FREE_SIE)
    obj._build_cache()
    assert not obj._cache["legacy"]
    assert [n for n, _ in obj._cache["opt_lenses"]] == ["sie"]


def test_general_loss_matches_per_candidate_gpu():
    """The generalized batched solve scores a candidate like the per-candidate
    Rhongomyniad engine (different image finders -> small tolerance)."""
    if not _torch_ready():
        print("  (skipped: torch/rhongomyniad not importable)")
        return
    from core.optimize.backends import make_backend
    from core.optimize.objective import INVALID_LOSS, Objective

    cfg, problem, obj = _objective(FREE_SIE)
    per_cand = Objective(problem, obj.obs, obj.loss_cfg, make_backend("gpu"))

    rng = np.random.default_rng(7)
    lo = np.array([b[0] for b in problem.bounds])
    hi = np.array([b[1] for b in problem.bounds])
    cands = lo + (hi - lo) * rng.uniform(0.25, 0.75, size=(6, len(lo)))

    batched = obj(cands.T)
    checked = 0
    for k in range(len(cands)):
        ref = per_cand.evaluate_one(cands[k])
        if ref >= INVALID_LOSS or batched[k] >= INVALID_LOSS:
            assert (ref >= INVALID_LOSS) == (batched[k] >= INVALID_LOSS), \
                (k, ref, batched[k])
            continue
        assert abs(batched[k] - ref) <= 1e-6 * max(1.0, abs(ref)), \
            (k, ref, batched[k])
        checked += 1
    assert checked >= 3, f"only {checked} candidates were comparable"


def test_general_optimizable_source_position():
    """An optimizable source position is injected per candidate: moving the
    source dimension must change the images/loss."""
    if not _torch_ready():
        print("  (skipped: torch/rhongomyniad not importable)")
        return
    txt = FREE_SIE.replace("source_x = 0.03", "source_x = {-0.05, 0.08}")
    cfg, problem, obj = _objective(txt)
    labels = [d.label for d in problem.dims]
    assert "source_x" in labels
    mid = np.array([0.5 * (d.lo + d.hi) for d in problem.dims])
    a = mid.copy()
    b = mid.copy()
    i = labels.index("source_x")
    a[i], b[i] = 0.01, 0.05
    img_a = obj.images_for(a)
    img_b = obj.images_for(b)
    assert img_a and img_b and img_a != img_b


def test_general_chunking_invariance():
    if not _torch_ready():
        print("  (skipped: torch/rhongomyniad not importable)")
        return
    cfg, problem, obj = _objective(FREE_SIE)
    rng = np.random.default_rng(11)
    lo = np.array([b[0] for b in problem.bounds])
    hi = np.array([b[1] for b in problem.bounds])
    cands = (lo + (hi - lo) * rng.uniform(0.3, 0.7, size=(5, len(lo)))).T

    prev = os.environ.pop("GLADE_GPU_CHUNK", None)
    try:
        full = obj(cands)                # default chunk: one batch of 5
        os.environ["GLADE_GPU_CHUNK"] = "2"
        from core.optimize.batched import BatchedGPUObjective
        obj2 = BatchedGPUObjective(problem, obj.obs, obj.loss_cfg)
        chunked = obj2(cands)
    finally:
        if prev is None:
            os.environ.pop("GLADE_GPU_CHUNK", None)
        else:
            os.environ["GLADE_GPU_CHUNK"] = prev
    assert np.array_equal(full, chunked), (full, chunked)


def test_shared_var_batched_matches_per_candidate():
    """A shared {lo,hi} user variable maps two component slots onto ONE dim:
    the generalized batched solve must score candidates like the per-candidate
    engine (which goes through make_scene)."""
    if not _torch_ready():
        print("  (skipped: torch/rhongomyniad not importable)")
        return
    from core.optimize.backends import make_backend
    from core.optimize.objective import INVALID_LOSS, Objective

    txt = _BASE + """
missing_img_penalty = 50000
gal_x = {-0.05, 0.05}
'sie1': (1, 'sie', lens_z, {80, 160}, gal_x, {-0.05, 0.05}, {0.01, 0.3}, {0, 180})
'point1': (2, 'point', lens_z, {1e5, 1e7}, gal_x, {-0.05, 0.05})
"""
    cfg, problem, obj = _objective(txt)
    assert not obj._legacy_eligible(cfg)            # shared vars -> generalized
    # sie1: sigma/y/e/pa (4) + point1: mass/y (2) + the shared gal_x (1) = 7
    labels = [d.label for d in problem.dims]
    assert labels.count("gal_x") == 1 and problem.ndim == 7, labels

    backend = make_backend("gpu")
    per_cand = Objective(problem, obj.obs, obj.loss_cfg, backend)
    rng = np.random.default_rng(5)
    lo = np.array([b[0] for b in problem.bounds])
    hi = np.array([b[1] for b in problem.bounds])
    cands = lo + (hi - lo) * rng.uniform(0.25, 0.75, size=(6, len(lo)))

    batched = obj(cands.T)
    checked = 0
    for k in range(len(cands)):
        # only compare candidates where both finders agree on the image count:
        # the per-candidate adaptive quadtree can miss a genuine image that the
        # batched uniform fine grid catches (a pre-existing finder-robustness
        # difference, not a parameter-mapping issue — which is what this test
        # is for).
        imgs_b = obj.images_for(cands[k])
        imgs_p = backend.compute_images(problem.make_scene(cands[k])) or []
        if len(imgs_b) != len(imgs_p):
            continue
        ref = per_cand.evaluate_one(cands[k])
        if ref >= INVALID_LOSS or batched[k] >= INVALID_LOSS:
            assert (ref >= INVALID_LOSS) == (batched[k] >= INVALID_LOSS), \
                (k, ref, batched[k])
            continue
        assert abs(batched[k] - ref) <= 1e-6 * max(1.0, abs(ref)), \
            (k, ref, batched[k])
        checked += 1
    assert checked >= 3, f"only {checked} candidates were comparable"


def test_gpu_precision_validation():
    # bad values are a config error; 32/48/64 lint clean
    from core.format import lint_text
    _cfg(FREE_SIE + "gpu_precision = 48\n")
    _cfg(FREE_SIE + "gpu_precision = 32\n")
    _, issues = lint_text(FREE_SIE + "gpu_precision = 50\n",
                          backend="gpu", with_defaults=True)
    assert any(i.code == "bad_gpu_precision" and i.is_error for i in issues)


def test_gpu_precision_mixed_matches_fp64():
    """48 (fp32 fields + fp64 Newton) must reproduce fp64 losses: the Newton
    refine converges the fp32-seeded roots in fp64."""
    if not _torch_ready():
        print("  (skipped: torch/rhongomyniad not importable)")
        return
    _, problem, obj64 = _objective(FREE_SIE)
    _, _, obj48 = _objective(FREE_SIE + "gpu_precision = 48\n")
    rng = np.random.default_rng(7)
    lo = np.array([b[0] for b in problem.bounds])
    hi = np.array([b[1] for b in problem.bounds])
    cands = (lo + (hi - lo) * rng.uniform(0.25, 0.75, size=(8, len(lo)))).T
    a, b = obj64(cands), obj48(cands)
    assert obj48._cache["precision"] == 48
    # fp32 fields halve the bytes per candidate: the default chunk doubles
    assert obj48._cache["chunk"] == 2 * obj64._cache["chunk"]
    valid = a < 1e14
    assert (valid == (b < 1e14)).all()
    assert valid.sum() >= 4, "too few valid candidates to compare"
    # The fp32 field/triangle phase only perturbs the loss for NEAR-CRITICAL
    # candidates (an image with |mag| >~ 400, where 8 fp64 Newton iterations
    # cannot wash out the ~uas fp32 seed shift). These fixed candidates
    # measure max|mag| <= ~17; guard against a reseed moving one onto a
    # caustic by comparing only safely sub-critical candidates.
    maxmag = np.array([max((abs(m) for _x, _y, m in obj64.images_for(c)),
                           default=0.0) for c in cands.T])
    stable = valid & (maxmag < 100.0)
    assert stable.sum() >= 4, maxmag
    np.testing.assert_allclose(b[stable], a[stable], rtol=1e-9, atol=0.0)


def test_gpu_precision_fp32_close_to_fp64():
    if not _torch_ready():
        print("  (skipped: torch/rhongomyniad not importable)")
        return
    _, problem, obj64 = _objective(FREE_SIE)
    _, _, obj32 = _objective(FREE_SIE + "gpu_precision = 32\n")
    rng = np.random.default_rng(7)
    lo = np.array([b[0] for b in problem.bounds])
    hi = np.array([b[1] for b in problem.bounds])
    cands = (lo + (hi - lo) * rng.uniform(0.25, 0.75, size=(8, len(lo)))).T
    a, b = obj64(cands), obj32(cands)
    valid = a < 1e14
    assert (valid == (b < 1e14)).all()
    assert valid.sum() >= 4, "too few valid candidates to compare"
    np.testing.assert_allclose(b[valid], a[valid], rtol=1e-3, atol=0.0)


def test_gpu_precision_legacy_point_path():
    """The legacy analytic point pipeline honors gpu_precision too; at the
    default 64 the added casts are no-ops (bitwise-identical losses)."""
    if not _torch_ready():
        print("  (skipped: torch/rhongomyniad not importable)")
        return
    txt = LOCKED_PLUS_POINT.replace("250.0", "160.0") + "missing_img_penalty = 50000\n"
    _, problem, obj_def = _objective(txt)
    _, _, obj64 = _objective(txt + "gpu_precision = 64\n")
    _, _, obj48 = _objective(txt + "gpu_precision = 48\n")
    rng = np.random.default_rng(3)
    lo = np.array([b[0] for b in problem.bounds])
    hi = np.array([b[1] for b in problem.bounds])
    cands = (lo + (hi - lo) * rng.uniform(0.1, 0.9, size=(12, len(lo)))).T
    a = obj_def(cands)
    assert obj_def._cache["legacy"]
    assert np.array_equal(a, obj64(cands))          # explicit 64 == default, bitwise
    b = obj48(cands)
    assert obj48._cache["legacy"] and obj48._cache["precision"] == 48
    valid = a < 1e14
    assert (valid == (b < 1e14)).all()
    assert valid.sum() >= 6, "too few valid candidates to compare"
    np.testing.assert_allclose(b[valid], a[valid], rtol=1e-9, atol=0.0)


def test_gpu_precision_fp32_dedup_no_phantom_images():
    """fp32 Newton roots from duplicate seeds land ~ulp apart — far above the
    fp64-scaled mag dedup tolerance. The ulp-floor merge must keep fp32 image
    counts equal to fp64's (phantom duplicates corrupted ~6.5% of losses on
    this config before the fix). Uses the real pm_gpu_test.dat (gitignored);
    skips when absent."""
    if not _torch_ready():
        print("  (skipped: torch/rhongomyniad not importable)")
        return
    pm = os.path.join(_ROOT, "InputFiles", "pm_gpu_test.dat")
    if not os.path.exists(pm):
        print("  (skipped: InputFiles/pm_gpu_test.dat not present)")
        return
    text = open(pm, encoding="utf-8").read()
    _, problem, obj64 = _objective(text)
    _, _, obj32 = _objective(text + "\ngpu_precision = 32\n")
    rng = np.random.default_rng(1)
    lo = np.array([b[0] for b in problem.bounds])
    hi = np.array([b[1] for b in problem.bounds])
    cands = lo + (hi - lo) * rng.uniform(0.0, 1.0, size=(300, len(lo)))

    obj64(cands.T)
    obj32(cands.T)
    sx, sy, lm = obj64._batch_tensors(cands.T)
    n64 = [len(im) for im in obj64._solve(
        sx, sy, lm, float(obj64._cache["source_x"]),
        float(obj64._cache["source_y"]))]
    sx, sy, lm = obj32._batch_tensors(cands.T)
    n32 = [len(im) for im in obj32._solve(
        sx, sy, lm, float(obj32._cache["source_x"]),
        float(obj32._cache["source_y"]))]
    inflated = sum(1 for a_n, b_n in zip(n64, n32) if b_n > a_n)
    assert inflated == 0, f"{inflated}/300 candidates grew phantom images at fp32"


def test_legacy_general_cross_parity():
    """Regression armor for the bit-parity split: the legacy analytic point
    pipeline and the generalized kernel path must score the same point-mass
    config identically. Any future edit inside _solve/_build_cache that
    drifts either path shows up here."""
    if not _torch_ready():
        print("  (skipped: torch/rhongomyniad not importable)")
        return
    # sigma=160 keeps the Einstein ring inside the grid box (250 -> ~0.8",
    # outside +-0.5", which would make every loss INVALID and the test vacuous);
    # the graded penalty keeps 2-image candidates finite (and comparable).
    txt = LOCKED_PLUS_POINT.replace("250.0", "160.0") + "missing_img_penalty = 50000\n"
    _, problem, obj_legacy = _objective(txt)
    _, _, obj_general = _objective(txt)
    obj_general._legacy_eligible = lambda cfg: False  # force generalized path

    rng = np.random.default_rng(3)
    lo = np.array([b[0] for b in problem.bounds])
    hi = np.array([b[1] for b in problem.bounds])
    cands = (lo + (hi - lo) * rng.uniform(0.1, 0.9, size=(16, len(lo)))).T

    a = obj_legacy(cands)
    b = obj_general(cands)
    assert obj_legacy._cache["legacy"] and not obj_general._cache["legacy"]
    valid = a < 1e14
    assert valid.sum() >= 8, "too few valid candidates to compare"
    np.testing.assert_allclose(b, a, rtol=1e-9, atol=0.0)


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
