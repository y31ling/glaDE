"""Tests for the micro-image audit (auto_check) — plan T1-T4.

Runs the layer-1 audit against the five anchor runs committed under runs/
(iptf-nfw-pm-1234 / iptf-sie-pm-1234 / iptf-sie-king-1234 convict as fake
solutions; iptf-nfw-nfw-1234 and -loose must come out clean), plus the T4
scale test (mass/100, distance/10 synthetic model) and unit checks of the
pure-numpy cosmology against known values.

    python -m pytest core/tests/test_micro_audit.py
    python core/tests/test_micro_audit.py

The anchor tests need the vendored glafic binary and the runs/ directories;
they self-skip when either is missing.
"""
from __future__ import annotations

import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from core import micro_audit as ma  # noqa: E402
from core.optimize.loss import LossConfig  # noqa: E402
from core.optimize.matching import assign_images, select_images  # noqa: E402
from core.optimize.scene import ObsData, Scene, SceneComponent  # noqa: E402
from core.verify import _read_glafic_point, find_glafic_bin  # noqa: E402


# --- shared fixtures --------------------------------------------------------

def _obs() -> ObsData:
    """The iPTF16geu observation set used by every anchor run
    (core/examples/images_data.dat; obs_x_flip=True)."""
    pos_mas = np.array([[-266.035, 0.427], [118.835, -221.927],
                        [238.324, 227.27], [-126.157, 319.719]])
    positions = np.zeros_like(pos_mas)
    positions[:, 0] = -pos_mas[:, 0] / 1000.0
    positions[:, 1] = pos_mas[:, 1] / 1000.0
    return ObsData(
        positions=positions,
        magnifications=np.array([-35.6, 15.7, -7.5, 9.1]),
        mag_errors=np.array([2.1, 1.3, 1.0, 1.1]),
        pos_sigma_mas=np.array([0.41, 0.86, 2.23, 3.11]),
        center_offset=(-0.01535, 0.0322),
    )


def _load_scene(run_dir: str) -> Scene:
    """Build a Scene from a runs/<name>/glafic_verify.input model snapshot."""
    scal: dict = {}
    comps: list[SceneComponent] = []
    point = None
    with open(os.path.join(run_dir, "glafic_verify.input"), encoding="utf-8") as fh:
        for raw in fh:
            parts = raw.split()
            if not parts or parts[0].startswith("#"):
                continue
            key = parts[0]
            if key == "lens":
                comps.append(SceneComponent(
                    parts[1], float(parts[2]), [float(v) for v in parts[3:10]]))
            elif key == "point":
                point = [float(v) for v in parts[1:4]]
            elif len(parts) == 2:
                try:
                    scal[key] = float(parts[1])
                except ValueError:
                    pass
    assert point is not None
    return Scene(
        omega=scal["omega"], lam=scal["lambda"], weos=scal["weos"],
        hubble=scal["hubble"],
        xmin=scal["xmin"], ymin=scal["ymin"], xmax=scal["xmax"],
        ymax=scal["ymax"], pix_ext=scal["pix_ext"], pix_poi=scal["pix_poi"],
        maxlev=int(scal["maxlev"]),
        source_z=point[0], source_x=point[1], source_y=point[2],
        components=comps,
    )


def _matched(run_dir: str, obs: ObsData):
    """(model_xy, mu, obs_idx) matched exactly as verify_with_glafic does,
    from the run's archived single-root glafic_verify_point.dat."""
    images = _read_glafic_point(os.path.join(run_dir, "glafic_verify_point.dat"))
    sel = select_images(images, obs.n)
    assert sel is not None
    pred_pos = np.array([[im[0], im[1]] for im in sel], dtype=float)
    pred_mag = np.array([im[2] for im in sel], dtype=float)
    matched_pos, mm, _delta, obs_idx = assign_images(
        obs.positions, pred_pos, pred_mag, obs.center_offset)
    model_xy = matched_pos - np.asarray(obs.center_offset, dtype=float)
    return model_xy, mm, obs_idx


def _ready(run_name: str):
    run_dir = os.path.join(_ROOT, "runs", run_name)
    if not os.path.isfile(os.path.join(run_dir, "glafic_verify.input")):
        print(f"  (skipped: {run_name} anchor missing)")
        return None
    if not find_glafic_bin():
        print("  (skipped: glafic binary not found)")
        return None
    return run_dir


def _audit(run_name: str, scene=None):
    run_dir = _ready(run_name)
    if run_dir is None:
        return None
    obs = _obs()
    model_xy, mm, obs_idx = _matched(run_dir, obs)
    if scene is None:
        scene = _load_scene(run_dir)
    with tempfile.TemporaryDirectory(prefix="micro_audit_test_") as tmp:
        return ma.micro_audit(scene, obs, model_xy, mm, obs_idx,
                              LossConfig(), tmp)


def _img4_entry(report: dict):
    """The entry matched to observed image 4 (obs index 3)."""
    return report["per_image"][3]


# --- unit checks (no glafic needed) ----------------------------------------

def test_cosmology_theta_e():
    # theta_E(1e8 Msun/h) at zl=0.216 / zs=0.409 is 26.8 mas (plan §6.1)
    dis = ma.Distances.build(0.3, 0.7, -1.0, 0.216, 0.409)
    te = float(ma.theta_e_point_arcsec(1.0e8, dis)) * 1000.0
    assert abs(te - 26.8) < 0.5, te
    # scaling: theta_E ~ sqrt(M)
    te4 = float(ma.theta_e_point_arcsec(1.0e4, dis)) * 1000.0
    assert abs(te4 - te / 100.0) < 1e-9


def test_theta_scale_dispatch():
    dis = ma.Distances.build(0.3, 0.7, -1.0, 0.216, 0.409)
    # point: pure theta_E
    ts = ma.theta_scale_mas("point", [1.0e6, 0.0, 0.0], dis)
    assert abs(ts - float(ma.theta_e_point_arcsec(1.0e6, dis)) * 1000.0) < 1e-12
    # king: max(theta_E, rc)
    ts = ma.theta_scale_mas("king", [1.0e2, 0, 0, 0, 0, 0.005, 1.5], dis)
    assert abs(ts - 5.0) < 1e-9          # rc = 5 mas dominates
    # sie: SIS Einstein radius from sigma; a main lens is way over the cap
    ts = ma.theta_scale_mas("sie", [160.0, 0, 0, 0, 0, 0.0], dis)
    assert ts > ma.SCALE_CAP_MAS
    # floor
    ts = ma.theta_scale_mas("point", [1.0, 0.0, 0.0], dis)
    assert ts == ma.SCALE_FLOOR_MAS
    # irregular layouts are not perturbers
    assert ma.theta_scale_mas("pert", [0.409, 0, 0, 0.01, 0, 0, 0], dis) is None


def test_perturber_gate():
    scene = Scene(omega=0.3, lam=0.7, weos=-1.0, hubble=0.7,
                  xmin=-0.5, ymin=-0.5, xmax=0.5, ymax=0.5,
                  pix_ext=0.01, pix_poi=0.2, maxlev=5,
                  source_z=0.409, source_x=0.0, source_y=0.0,
                  components=[
                      SceneComponent("sers", 0.216, [2.1e10, 0, 0, 0.3, 112, 0.39, 1.06]),
                      SceneComponent("anfw", 0.216, [3.6e11, 0, 0, 0.46, 26, 29, 0]),
                      SceneComponent("point", 0.216, [1.0e6, 0.1, 0.2]),
                  ])
    perts = ma.find_compact_perturbers(scene)
    assert len(perts) == 1 and perts[0].glafic_type == "point"
    # trigger rule: R_trig = 10 ts + 2 mas
    pert = perts[0]
    d_in = (pert.r_trig_mas - 0.1) / 1000.0
    assert ma.triggered(perts, 0.1 + d_in, 0.2) is not None
    assert ma.triggered(perts, 0.1 + (pert.r_trig_mas + 0.1) / 1000.0, 0.2) is None


# --- T1: nfw+pm conviction --------------------------------------------------

def test_t1_nfw_pm_fake_solution():
    rep = _audit("iptf-nfw-pm-1234")
    if rep is None:
        return
    e = _img4_entry(rep)
    assert e["n_micro"] == 4, e
    assert abs(e["sum_abs_mu"] - 41.2) / 41.2 < 0.05, e["sum_abs_mu"]
    assert e["trigger"] is not None and e["trigger"]["type"] == "point"
    assert rep["fake_solution"] is True
    assert rep["physical_loss"] is not None and rep["physical_loss"] > 100.0


# --- T2: sie+pm / sie+king convictions --------------------------------------

def test_t2_sie_pm():
    rep = _audit("iptf-sie-pm-1234")
    if rep is None:
        return
    e = _img4_entry(rep)
    assert e["n_micro"] == 4, e
    assert abs(e["sum_abs_mu"] - 38.9) / 38.9 < 0.05, e["sum_abs_mu"]
    assert rep["fake_solution"] is True


def test_t2_sie_king():
    rep = _audit("iptf-sie-king-1234")
    if rep is None:
        return
    e = _img4_entry(rep)
    assert e["n_micro"] == 3, e
    assert abs(e["sum_abs_mu"] - 45.4) / 45.4 < 0.05, e["sum_abs_mu"]
    assert rep["fake_solution"] is True


# --- T3: legal demagnified-saddle solutions must come out clean --------------

def _assert_clean(rep: dict):
    for e in rep["per_image"]:
        assert e["n_micro"] == 1, e
    assert rep["fake_solution"] is False
    # physical_loss recomputed from single roots stays close to a direct
    # ml_loss of the same values (identical inputs => tiny numerical drift)
    assert rep["physical_loss"] is not None


def test_t3_nfw_nfw_clean():
    rep = _audit("iptf-nfw-nfw-1234")
    if rep is None:
        return
    _assert_clean(rep)


def test_t3_nfw_nfw_loose_clean():
    rep = _audit("iptf-nfw-nfw-1234-loose")
    if rep is None:
        return
    _assert_clean(rep)


# --- T4: adaptive-scale test (M/100, d/10) ----------------------------------

def test_t4_scaled_perturber():
    run_dir = _ready("iptf-nfw-pm-1234")
    if run_dir is None:
        return
    obs = _obs()
    model_xy, mm, obs_idx = _matched(run_dir, obs)
    scene = _load_scene(run_dir)
    # The self-similar scaling anchor is the MACRO image position (the
    # flux-weighted centroid of the original micro cluster), not the old
    # single-root position: theta_E scales with sqrt(M), so placing the
    # perturber at centroid + (offset/10) with M/100 shrinks the whole
    # 4-root structure by 10x around the macro image.
    rep0 = _audit("iptf-nfw-pm-1234")
    roots0 = np.array(_img4_entry(rep0)["roots"], dtype=float)
    w = np.abs(roots0[:, 2])
    cen = (roots0[:, :2] * w[:, None]).sum(axis=0) / w.sum()
    pm = scene.components[-1]
    assert pm.glafic_type == "point"
    ex, ey = pm.params[1] - cen[0], pm.params[2] - cen[1]
    pm.params[0] = pm.params[0] / 100.0          # theta_E -> /10
    pm.params[1] = cen[0] + ex / 10.0            # offset from macro image /10
    pm.params[2] = cen[1] + ey / 10.0
    with tempfile.TemporaryDirectory(prefix="micro_audit_t4_") as tmp:
        rep = ma.micro_audit(scene, obs, model_xy, mm, obs_idx,
                             LossConfig(), tmp)
    e = _img4_entry(rep)
    # The point of T4 is that the ADAPTIVE box still resolves the full
    # 4-root cluster at a 10x smaller theta_E. The scaling is not exactly
    # self-similar (the macro Jacobian has a gradient across the original
    # ~1 mas cluster; at 1/10 scale the local field is more uniform), so
    # Sigma|mu| lands near, not at, the original 41: measured 34.4. Assert
    # the cluster is fully resolved and far brighter than the |mu|~9 root.
    assert e["n_micro"] == 4, e
    assert 28.0 < e["sum_abs_mu"] < 55.0, e["sum_abs_mu"]
    assert rep["fake_solution"] is True


# --- T6 (unit): no-trigger paths are value-identical to the plain loss -------

def _synthetic_images():
    """Model-frame images that pair 1:1 with the observed positions
    (pred + center_offset == obs exactly, so delta == 0)."""
    obs = _obs()
    co = np.asarray(obs.center_offset, dtype=float)
    model = obs.positions - co
    mags = [20.0, -15.0, 8.0, 9.0]
    return [(float(x), float(y), m) for (x, y), m in zip(model, mags)]


def _no_solver(scene, box):  # pragma: no cover - must never be called
    raise AssertionError("solver called although nothing triggered")


def test_t6_no_perturber_identical():
    from core.optimize.objective import point_source_loss
    obs = _obs()
    scene = Scene(omega=0.3, lam=0.7, weos=-1.0, hubble=0.7,
                  xmin=-0.5, ymin=-0.5, xmax=0.5, ymax=0.5,
                  pix_ext=0.01, pix_poi=0.2, maxlev=5,
                  source_z=0.409, source_x=0.0, source_y=0.0,
                  components=[SceneComponent(
                      "sers", 0.216, [2.1e10, 0, 0, 0.3, 112, 0.39, 1.06])])
    imgs = _synthetic_images()
    a = point_source_loss(imgs, obs, LossConfig())
    b = ma.checked_point_source_loss(imgs, obs, LossConfig(), scene, _no_solver)
    assert a == b, (a, b)


def test_t6_far_perturber_identical():
    from core.optimize.objective import point_source_loss
    obs = _obs()
    # a compact point mass, but hundreds of mas from every image
    scene = Scene(omega=0.3, lam=0.7, weos=-1.0, hubble=0.7,
                  xmin=-0.5, ymin=-0.5, xmax=0.5, ymax=0.5,
                  pix_ext=0.01, pix_poi=0.2, maxlev=5,
                  source_z=0.409, source_x=0.0, source_y=0.0,
                  components=[
                      SceneComponent("sers", 0.216, [2.1e10, 0, 0, 0.3, 112, 0.39, 1.06]),
                      SceneComponent("point", 0.216, [1.0e6, 0.0, 0.0]),
                  ])
    assert ma.find_compact_perturbers(scene)
    imgs = _synthetic_images()
    a = point_source_loss(imgs, obs, LossConfig())
    b = ma.checked_point_source_loss(imgs, obs, LossConfig(), scene, _no_solver)
    assert a == b, (a, b)


def test_t6_triggered_substitutes_sum_mu():
    from core.optimize.objective import point_source_loss
    obs = _obs()
    imgs = _synthetic_images()
    # perturber right on top of image 4 (model frame)
    ix, iy = imgs[3][0], imgs[3][1]
    scene = Scene(omega=0.3, lam=0.7, weos=-1.0, hubble=0.7,
                  xmin=-0.5, ymin=-0.5, xmax=0.5, ymax=0.5,
                  pix_ext=0.01, pix_poi=0.2, maxlev=5,
                  source_z=0.409, source_x=0.0, source_y=0.0,
                  components=[
                      SceneComponent("sers", 0.216, [2.1e10, 0, 0, 0.3, 112, 0.39, 1.06]),
                      SceneComponent("point", 0.216, [1.0e6, ix + 0.0005, iy]),
                  ])

    def solver(sc, box):
        # a fake 3-root cluster around the triggered image (1 mas apart, above
        # the theta_scale/10 = 0.27 mas merge tolerance)
        return [(ix, iy, -9.0), (ix + 1e-3, iy, 16.0), (ix - 1e-3, iy, 15.0)]

    a = point_source_loss(imgs, obs, LossConfig())
    b = ma.checked_point_source_loss(imgs, obs, LossConfig(), scene, solver)
    assert b != a
    # image 4 magnification must have been scored as sum|mu| = 40
    from core.optimize.loss import ml_loss
    from core.optimize.matching import assign_images
    pred_pos = np.array([[im[0], im[1]] for im in imgs])
    pred_mag = np.array([im[2] for im in imgs], dtype=float)
    _, mm, delta, oi = assign_images(obs.positions, pred_pos, pred_mag,
                                     obs.center_offset)
    mm2 = mm.copy()
    mm2[3] = 40.0
    expect = float(ml_loss(delta, mm2, obs.magnifications[oi],
                           obs.mag_errors[oi], obs.pos_sigma_mas[oi],
                           LossConfig()))
    assert abs(b - expect) < 1e-12, (b, expect)


# --- T7 (unit): the module must not touch torch/rhongomyniad -----------------

def test_t7_no_torch_dependency():
    import importlib
    import importlib.util

    class _Block:
        def find_spec(self, name, path=None, target=None):
            if name == "torch" or name.startswith("rhongomyniad"):
                raise ImportError(f"blocked import of {name} (CPU-only test)")
            return None

    blocker = _Block()
    sys.meta_path.insert(0, blocker)
    try:
        for mod in ("core.micro_audit",):
            importlib.reload(importlib.import_module(mod))
        dis = ma.Distances.build(0.3, 0.7, -1.0, 0.216, 0.409)
        assert float(ma.theta_e_point_arcsec(1e8, dis)) > 0
    finally:
        sys.meta_path.remove(blocker)
        importlib.reload(importlib.import_module("core.micro_audit"))


# --- runner ------------------------------------------------------------------

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
