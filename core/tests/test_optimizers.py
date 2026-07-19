"""Tests for the V0.7 point-source optimizers: BIPOP-CMA-ES and jSO.

Convergence on standard benchmark functions, same-seed determinism, the
OPTIMIZER dispatch through core.optimize.optimize() (FakeBackend, no engines
needed), and the alias/validation surface.

    python -m pytest core/tests/test_optimizers.py
    python core/tests/test_optimizers.py
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
from core.optimize import optimize  # noqa: E402
from core.optimize.cmaes import CMAESConfig, run_cmaes  # noqa: E402
from core.optimize.jso import JSOConfig, run_jso  # noqa: E402
from core.optimize.runner import normalize_algorithm  # noqa: E402

from test_optimize import TEST_CFG, TRUTH, FakeBackend  # noqa: E402


# --------------------------------------------------------------------------- #
# benchmark functions
# --------------------------------------------------------------------------- #

def _sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


def _rosen(x):
    x = np.asarray(x)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2
                        + (1.0 - x[:-1]) ** 2))


def _rastrigin(x):
    x = np.asarray(x)
    return float(10.0 * len(x)
                 + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


_BOUNDS10 = [(-5.0, 5.0)] * 10


def test_cmaes_sphere_and_rosenbrock():
    r = run_cmaes(_sphere, _BOUNDS10, CMAESConfig(maxevals=20000, seed=7),
                  record_population=False)
    assert r.fun < 1e-8, r.fun
    r = run_cmaes(_rosen, _BOUNDS10, CMAESConfig(maxevals=60000, seed=7),
                  record_population=False)
    assert r.fun < 1e-6, r.fun


def test_cmaes_bipop_restarts_solve_rastrigin():
    # multimodal: only the restart strategy gets to ~0
    r = run_cmaes(_rastrigin, _BOUNDS10, CMAESConfig(maxevals=150000, seed=7),
                  record_population=False)
    assert r.fun < 1.0, r.fun
    assert r.restarts >= 2


def test_jso_benchmarks():
    r = run_jso(_sphere, _BOUNDS10, JSOConfig(maxevals=20000, seed=7),
                record_population=False)
    assert r.fun < 1e-8, r.fun
    r = run_jso(_rastrigin, _BOUNDS10, JSOConfig(maxevals=100000, seed=7),
                record_population=False)
    assert r.fun < 1.0, r.fun


def test_same_seed_determinism():
    a = run_cmaes(_rosen, _BOUNDS10, CMAESConfig(maxevals=5000, seed=3),
                  record_population=False)
    b = run_cmaes(_rosen, _BOUNDS10, CMAESConfig(maxevals=5000, seed=3),
                  record_population=False)
    assert np.array_equal(a.x, b.x) and a.fun == b.fun
    a = run_jso(_rosen, _BOUNDS10, JSOConfig(maxevals=5000, seed=3),
                record_population=False)
    b = run_jso(_rosen, _BOUNDS10, JSOConfig(maxevals=5000, seed=3),
                record_population=False)
    assert np.array_equal(a.x, b.x) and a.fun == b.fun


def test_bounds_respected():
    r = run_cmaes(_sphere, [(1.0, 2.0)] * 4, CMAESConfig(maxevals=3000, seed=1),
                  record_population=False)
    assert np.all(r.x >= 1.0 - 1e-12) and np.all(r.x <= 2.0 + 1e-12)
    assert abs(r.fun - 4.0) < 1e-6          # optimum at the lower corner
    r = run_jso(_sphere, [(1.0, 2.0)] * 4, JSOConfig(maxevals=3000, seed=1),
                record_population=False)
    assert np.all(r.x >= 1.0) and np.all(r.x <= 2.0)
    assert abs(r.fun - 4.0) < 1e-6


def test_jso_np_init_formula():
    # NP_init = round(25 ln(D) sqrt(D)) — natural log (D=10 -> 182)
    assert int(round(25.0 * math.log(10) * math.sqrt(10))) == 182


# --------------------------------------------------------------------------- #
# dispatch through optimize()
# --------------------------------------------------------------------------- #

def _cfg(extra: str = ""):
    cfg, issues = lint_text(TEST_CFG + extra, backend="cpu", with_defaults=True)
    assert not any(i.is_error for i in issues), [str(i) for i in issues]
    return cfg


def _backend(cfg):
    from core.optimize import build_obs
    obs = build_obs(cfg)
    return FakeBackend(obs.positions, obs.magnifications)


def test_optimize_dispatch_cmaes():
    cfg = _cfg("OPTIMIZER = 'BIPOP-CMA-ES'\nCMAES_MAXEVALS = 6000\n")
    res = optimize(cfg, backend=_backend(cfg), record_population=False)
    assert res.algorithm == "BIPOP-CMA-ES"
    assert res.loss < 1e-6, res.loss
    assert abs(res.fitted["source_x"] - TRUTH["source_x"]) < 1e-3
    assert abs(math.log10(res.fitted["point1.mass"]) - TRUTH["log_mass"]) < 1e-3


def test_optimize_dispatch_jso():
    cfg = _cfg("OPTIMIZER = 'jSO'\nJSO_MAXEVALS = 8000\n")
    res = optimize(cfg, backend=_backend(cfg), record_population=False)
    assert res.algorithm == "JSO"
    assert res.loss < 1e-6, res.loss
    assert abs(res.fitted["point1.x"] - TRUTH["point_x"]) < 1e-3


def test_optimize_default_stays_de():
    cfg = _cfg()
    res = optimize(cfg, backend=_backend(cfg), record_population=False,
                   de_overrides={"maxiter": 30, "polish": False})
    assert res.algorithm == "DE"


def test_optimize_history_shape():
    cfg = _cfg("OPTIMIZER = 'jSO'\nJSO_MAXEVALS = 2000\n")
    seen = []

    def cb(it, pop, best, energies):
        seen.append((it, pop.shape, best, energies.shape))
        assert pop.shape[0] == energies.shape[0]
        assert pop.shape[1] == 4

    res = optimize(cfg, backend=_backend(cfg), on_iteration=cb)
    assert seen and res.de.history
    assert seen[0][0] == 1
    assert "population" in res.de.history[0]


def test_algorithm_aliases_and_validation():
    assert normalize_algorithm("de") == "DE"
    assert normalize_algorithm("cmaes") == "BIPOP-CMA-ES"
    assert normalize_algorithm("CMA-ES") == "BIPOP-CMA-ES"
    assert normalize_algorithm("bipop-cma-es") == "BIPOP-CMA-ES"
    assert normalize_algorithm("jSO") == "JSO"
    try:
        normalize_algorithm("simplex")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # validate() flags a bad OPTIMIZER value
    cfg, issues = lint_text(TEST_CFG + "OPTIMIZER = 'simplex'\n",
                            backend="cpu", with_defaults=True)
    assert any(i.code == "bad_optimizer" for i in issues), \
        [str(i) for i in issues]


# --------------------------------------------------------------------------- #
# glade_output writer
# --------------------------------------------------------------------------- #

def test_write_glade_output_roundtrip():
    import tempfile

    from core.format.values import Bounds
    from core.report import write_glade_output

    cfg = _cfg("OPTIMIZER = 'jSO'\nJSO_MAXEVALS = 4000\n")
    res = optimize(cfg, backend=_backend(cfg), record_population=False)
    with tempfile.TemporaryDirectory(prefix="glade_out_") as tmp:
        run_dir = os.path.join(tmp, "my-test-run")
        path = write_glade_output(res, run_dir)
        assert os.path.basename(path) == "glade_output_my-test-run.dat"
        text = open(path, encoding="utf-8").read()
        assert "{" not in text.replace("{lo", "")   # no bounds survive
        cfg2, issues = lint_text(text, path, backend="cpu", with_defaults=True)
        assert cfg2 is not None
        assert not any(i.is_error for i in issues), [str(i) for i in issues]
        # everything fixed: nothing optimizable remains
        from core.optimize import OptProblem
        assert OptProblem(cfg2).ndim == 0
        # fitted values survived the round trip exactly
        sx = cfg2.source["source_x"]
        assert not isinstance(sx, Bounds)
        assert abs(float(sx) - res.fitted["source_x"]) < 1e-15
        comp = cfg2.components[0]
        assert abs(comp.params[0].value - res.fitted["point1.mass"]) < 1e-9
        assert abs(comp.params[1].value - res.fitted["point1.x"]) < 1e-15


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
