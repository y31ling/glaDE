"""Tests for the backend-agnostic DE-core.

Uses a small analytic FakeBackend so the optimizer / problem-building / matching /
loss pipeline is exercised end-to-end without the glafic or torch engines.

    python -m pytest core/tests/test_optimize.py
    python core/tests/test_optimize.py
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
from core.format.values import Bounds  # noqa: E402
from core.optimize import OptProblem, build_obs, optimize  # noqa: E402
from core.optimize.objective import INVALID_LOSS  # noqa: E402


TEST_CFG = """
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
'point1': (1, 'point', lens_z, {1e5, 1e7}, {-0.30, -0.20}, {-0.05, 0.05})
"""

TRUTH = {"source_x": 0.03, "point_x": -0.25, "point_y": 0.0, "log_mass": 6.0}


class FakeBackend:
    """Separable synthetic model: each optimizable dim controls one residual,
    so the global optimum sits exactly at TRUTH with loss 0."""

    name = "fake"

    def __init__(self, obs_positions, obs_mags):
        self.obs = np.asarray(obs_positions, dtype=float)
        self.mags = np.asarray(obs_mags, dtype=float)

    def compute_images(self, scene):
        comp = scene.components[0]
        mass, px, py = comp.params[0], comp.params[1], comp.params[2]
        d_src = scene.source_x - TRUTH["source_x"]
        d_px = px - TRUTH["point_x"]
        d_py = py - TRUTH["point_y"]
        d_m = math.log10(max(mass, 1e-30)) - TRUTH["log_mass"]

        rx = [d_src, d_px, 0.0, 0.0]
        ry = [0.0, 0.0, d_py, 0.0]
        images = []
        for i, (ox, oy) in enumerate(self.obs):
            images.append((ox + rx[i], oy + ry[i], self.mags[i] + d_m))
        return images


def _cfg():
    cfg, issues = lint_text(TEST_CFG, backend="cpu", with_defaults=True)
    assert not any(i.is_error for i in issues), [str(i) for i in issues]
    return cfg


# --------------------------------------------------------------------------- #
# problem building
# --------------------------------------------------------------------------- #

def test_problem_dims_and_log_bounds():
    p = OptProblem(_cfg())
    labels = [d.label for d in p.dims]
    assert labels == ["source_x", "point1.mass", "point1.x", "point1.y"]
    # mass dim is log10
    mass_dim = p.dims[1]
    assert mass_dim.log is True
    assert math.isclose(mass_dim.lo, 5.0) and math.isclose(mass_dim.hi, 7.0)
    # linear dims keep raw bounds
    assert p.dims[0].log is False and p.dims[0].lo == -0.1 and p.dims[0].hi == 0.1


def test_make_scene_injects_candidate():
    p = OptProblem(_cfg())
    # candidate order matches dims: [source_x, log_mass, point_x, point_y]
    cand = [0.03, 6.0, -0.25, 0.0]
    scene = p.make_scene(cand)
    assert math.isclose(scene.source_x, 0.03)
    assert math.isclose(scene.source_y, 0.024)   # locked
    comp = scene.components[0]
    assert comp.glafic_type == "point"
    assert math.isclose(comp.params[0], 1e6)      # 10 ** 6
    assert math.isclose(comp.params[1], -0.25)
    assert len(scene.components) == 1


def test_decode_unlogs_mass():
    p = OptProblem(_cfg())
    fitted = p.decode([0.03, 6.0, -0.25, 0.0])
    assert math.isclose(fitted["point1.mass"], 1e6)


# --------------------------------------------------------------------------- #
# objective + optimization
# --------------------------------------------------------------------------- #

def test_objective_zero_at_truth_big_when_invalid():
    from core.optimize.objective import Objective
    from core.optimize.loss import LossConfig
    cfg = _cfg()
    p = OptProblem(cfg)
    obs = build_obs(cfg)
    obj = Objective(p, obs, LossConfig.from_cfg(cfg),
                    FakeBackend(obs.positions, obs.magnifications))
    loss_truth = obj([0.03, 6.0, -0.25, 0.0])
    assert loss_truth < 1e-9
    loss_off = obj([-0.1, 7.0, -0.20, 0.05])
    assert loss_off > loss_truth


def test_optimize_converges_to_truth():
    cfg = _cfg()
    obs = build_obs(cfg)
    backend = FakeBackend(obs.positions, obs.magnifications)
    result = optimize(
        cfg, backend=backend,
        de_overrides={"maxiter": 300, "popsize": 20, "seed": 0,
                      "polish": True, "early_stopping": True,
                      "early_stop_patience": 25},
        record_population=False,
    )
    assert result.loss < 1e-3, f"loss={result.loss}"
    assert abs(result.fitted["source_x"] - 0.03) < 5e-3
    assert abs(result.fitted["point1.x"] - (-0.25)) < 5e-3
    assert abs(result.fitted["point1.y"] - 0.0) < 5e-3
    assert abs(math.log10(result.fitted["point1.mass"]) - 6.0) < 0.1
    # scene reflects the fit
    assert abs(result.scene.source_x - 0.03) < 5e-3


def test_missing_img_penalty_config_plumbing():
    from core.optimize.loss import LossConfig
    # default is present and disabled
    cfg = _cfg()
    assert cfg.get("missing_img_penalty") == 0.0
    assert LossConfig.from_cfg(cfg).missing_img_penalty == 0.0
    # set in the .dat -> classified to the algorithm section and read by LossConfig
    cfg2, issues = lint_text(TEST_CFG + "\nmissing_img_penalty = 250.0\n",
                             backend="cpu", with_defaults=True)
    assert not any(i.is_error for i in issues), [str(i) for i in issues]
    assert cfg2.algorithm.get("missing_img_penalty") == 250.0
    assert LossConfig.from_cfg(cfg2).missing_img_penalty == 250.0
    # the UPPER-case alias resolves to the same canonical key
    cfg3, _ = lint_text(TEST_CFG + "\nMISSING_IMG_PENALTY = 7.0\n",
                        backend="cpu", with_defaults=True)
    assert LossConfig.from_cfg(cfg3).missing_img_penalty == 7.0


def test_point_source_loss_missing_penalty():
    from core.optimize.objective import INVALID_LOSS, point_source_loss
    from core.optimize.loss import LossConfig
    from core.optimize.scene import ObsData

    obs = ObsData(
        positions=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        magnifications=np.array([10.0, 10.0, 10.0, 10.0]),
        mag_errors=np.array([1.0, 1.0, 1.0, 1.0]),
        pos_sigma_mas=np.array([1.0, 1.0, 1.0, 1.0]),
        center_offset=(0.0, 0.0),
    )
    full = [(0.0, 0.0, 10.0), (1.0, 0.0, 10.0), (0.0, 1.0, 10.0), (1.0, 1.0, 10.0)]

    disabled = LossConfig(missing_img_penalty=0.0)
    enabled = LossConfig(missing_img_penalty=500.0)

    # exact image count -> ~0 loss whether or not the penalty is on
    assert point_source_loss(full, obs, disabled) < 1e-9
    assert point_source_loss(full, obs, enabled) < 1e-9

    # n_obs + 1 (a faint central image) still drops the lowest-|mag| one -> ~0
    plus1 = full + [(0.5, 0.5, 0.001)]
    assert point_source_loss(plus1, obs, enabled) < 1e-9
    # over-imaged by 2 is rejected even with the penalty (only "fewer" is graded)
    assert point_source_loss(full + [(0.5, 0.5, 0.001), (0.6, 0.6, 0.002)],
                             obs, enabled) >= INVALID_LOSS

    # fewer images: hard reject when disabled, graded when enabled
    three = full[:3]
    assert point_source_loss(three, obs, disabled) >= INVALID_LOSS
    loss3 = point_source_loss(three, obs, enabled)   # 1 missing, base ~0
    loss2 = point_source_loss(full[:2], obs, enabled)
    loss0 = point_source_loss([], obs, enabled)
    assert math.isclose(loss3, 500.0, rel_tol=1e-9)
    assert math.isclose(loss2, 1000.0, rel_tol=1e-9)
    assert math.isclose(loss0, 2000.0, rel_tol=1e-9)
    assert loss3 < loss2 < loss0                      # strictly graded by shortfall


def test_objective_routes_missing_penalty():
    from core.optimize.objective import INVALID_LOSS, Objective
    from core.optimize.loss import LossConfig
    cfg = _cfg()
    p = OptProblem(cfg)
    obs = build_obs(cfg)

    class ShortBackend:
        """Returns the first ``k`` observed images exactly (k < n_obs)."""
        name = "short"

        def __init__(self, k):
            self.k = k

        def compute_images(self, scene):
            return [(obs.positions[i, 0], obs.positions[i, 1],
                     obs.magnifications[i]) for i in range(self.k)]

    cand = [0.03, 6.0, -0.25, 0.0]
    # disabled -> the short candidate is rejected
    obj0 = Objective(p, obs, LossConfig(missing_img_penalty=0.0), ShortBackend(3))
    assert obj0(cand) >= INVALID_LOSS
    # enabled -> graded (1 missing image * penalty, residuals ~0)
    obj1 = Objective(p, obs, LossConfig(missing_img_penalty=1000.0), ShortBackend(3))
    assert math.isclose(obj1(cand), 1000.0, rel_tol=1e-6)


def test_objective_is_picklable_by_name():
    import pickle
    from core.optimize.objective import Objective
    from core.optimize.loss import LossConfig
    cfg = _cfg()
    p = OptProblem(cfg)
    obs = build_obs(cfg)
    obj = Objective(p, obs, LossConfig.from_cfg(cfg), "cpu")  # backend by name
    blob = pickle.dumps(obj)               # must not try to pickle an engine module
    obj2 = pickle.loads(blob)
    assert obj2.backend_name == "cpu"
    assert obj2._backend is None


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

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
