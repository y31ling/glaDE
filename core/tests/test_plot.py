"""Tests for the self-contained plotting-core.

    python -m pytest core/tests/test_plot.py
    python core/tests/test_plot.py
"""
from __future__ import annotations

import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from core.plot import (  # noqa: E402
    plot_iteration_corner,
    plot_triptych,
    plot_triptych_compare,
    read_critical_curves,
    subhalo_label,
)
from core.optimize import build_obs, optimize  # noqa: E402
from core.report import make_triptych  # noqa: E402
from core.tests.test_optimize import FakeBackend, _cfg  # noqa: E402


def _exists_nonempty(path) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def test_subhalo_label_point_and_king():
    assert subhalo_label(1, "point", [1.0e6, -0.25, 0.0]).startswith("S1: 1.0e+06")
    lbl = subhalo_label(2, "king", [1.0e8, 0.1, -0.2, 0.0, 0.0, 0.02, 1.5])
    assert lbl.startswith("S2: 1.0e+08")
    assert "rc=" in lbl and "c=" in lbl


def test_read_critical_curves_real_file():
    path = os.path.join(_ROOT, "temp_v_king_gpu_best_crit.dat")
    if not os.path.isfile(path):
        return  # artifact may be absent; skip silently
    crit, caus = read_critical_curves(path)
    assert isinstance(crit, list) and isinstance(caus, list)
    assert len(crit) > 0 and len(crit[0]) == 2 and len(crit[0][0]) == 2


def test_read_critical_curves_missing_file():
    crit, caus = read_critical_curves("/no/such/file_crit.dat")
    assert crit == [] and caus == []


def test_plot_triptych_writes_png():
    n = 4
    obs = np.array([[-0.266, 0.0], [0.118, -0.221], [0.238, 0.227], [-0.126, 0.319]])
    pred = obs + 0.002
    crit = [[[-0.1, -0.1], [0.1, 0.1]], [[0.1, -0.1], [-0.1, 0.1]]]
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "trip.png")
        res = plot_triptych(
            img_numbers=list(range(1, n + 1)),
            delta_pos_mas=[0.3, 0.6, 1.2, 0.8],
            sigma_pos_mas=[0.41, 0.86, 2.23, 3.11],
            mu_obs=[-35.6, 15.7, -7.5, 9.1],
            mu_obs_err=[2.1, 1.3, 1.0, 1.1],
            mu_pred=[-34.0, 16.0, -7.1, 9.4],
            mu_at_obs_pred=[-34.2, 15.9, -7.2, 9.3],
            obs_positions_arcsec=obs,
            pred_positions_arcsec=pred,
            crit_segments=crit,
            subhalos=[(-0.25, 0.0, "S1: 1.0e+06")],
            output_file=out,
            show_2sigma=True,
        )
        assert res == out and _exists_nonempty(out)


def test_plot_triptych_abs_mag_modes():
    """abs_mag=True (default) renders the |mu| panel (bars upward from 0);
    abs_mag=False keeps the signed bars — both must render."""
    n = 4
    obs = np.array([[-0.266, 0.0], [0.118, -0.221], [0.238, 0.227], [-0.126, 0.319]])
    pred = obs + 0.002
    crit = [[[-0.1, -0.1], [0.1, 0.1]]]
    common = dict(
        img_numbers=list(range(1, n + 1)),
        delta_pos_mas=[0.3, 0.6, 1.2, 0.8],
        sigma_pos_mas=[0.41, 0.86, 2.23, 3.11],
        mu_obs=[-35.6, 15.7, -7.5, 9.1],
        mu_obs_err=[2.1, 1.3, 1.0, 1.1],
        mu_pred=[34.0, 16.0, 7.1, -9.4],      # parity flips vs obs
        mu_at_obs_pred=[34.2, 15.9, 7.2, -9.3],
        obs_positions_arcsec=obs,
        pred_positions_arcsec=pred,
        crit_segments=crit,
    )
    with tempfile.TemporaryDirectory() as d:
        for flag in (True, False):
            out = os.path.join(d, f"trip_abs_{flag}.png")
            res = plot_triptych(output_file=out, abs_mag=flag, **common)
            assert res == out and _exists_nonempty(out)


def test_plot_triptych_compare_writes_png():
    n = 4
    obs = np.array([[-0.266, 0.0], [0.118, -0.221], [0.238, 0.227], [-0.126, 0.319]])
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "cmp.png")
        plot_triptych_compare(
            img_numbers=list(range(1, n + 1)),
            delta_baseline=[5.0, 6.0, 7.0, 4.0],
            delta_optimized=[0.3, 0.6, 1.2, 0.8],
            sigma_pos_mas=0.5,
            mu_obs=[-35.6, 15.7, -7.5, 9.1],
            mu_obs_err=[2.1, 1.3, 1.0, 1.1],
            mu_pred_baseline=[-20.0, 10.0, -4.0, 5.0],
            mu_pred_optimized=[-34.0, 16.0, -7.1, 9.4],
            obs_positions_arcsec=obs,
            pred_positions_arcsec=obs + 0.002,
            crit_segments=[],
            output_file=out,
        )
        assert _exists_nonempty(out)


def test_plot_iteration_corner_writes_png():
    rng = np.random.default_rng(0)
    labels = ["sie1.sigma", "sie1.x", "sie1.y", "ext1.norm"]
    bounds = [(10.0, 300.0), (-0.2, 0.2), (-0.2, 0.2), (0.5, 2.0)]
    pop = np.column_stack([rng.uniform(lo, hi, 50) for lo, hi in bounds])
    energies = rng.random(50) * 100
    energies[0] = np.inf      # robustness: invalid candidates
    obs = np.array([[-0.1, 0.05], [0.12, -0.08]])
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "iteration_0000.png")
        res = plot_iteration_corner(pop, energies, labels, bounds, 0, out,
                                    is_log=[True, False, False, False],
                                    obs_positions_arcsec=obs)
        assert res == out and _exists_nonempty(out)


def test_plot_iteration_corner_single_dim():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "corner1.png")
        plot_iteration_corner(np.linspace(0, 1, 30), np.linspace(5, 1, 30),
                              ["point1.mass"], [(0.0, 1.0)], 3, out)
        assert _exists_nonempty(out)


def test_subhalo_markers_respect_index_suffix():
    from core.format import lint_text
    from core.optimize import OptProblem
    from core.optimize.runner import OptResult
    from core.report import _subhalo_markers
    from core.tests.test_optimize import TEST_CFG

    text = TEST_CFG.replace(
        "'point1': (1, 'point', lens_z, {1e5, 1e7}, {-0.30, -0.20}, {-0.05, 0.05})",
        "'anfw1': (1l, 'anfw', lens_z, 3.6e11, 0.0, 0.0, 0.46, 26.6, 29.4)\n"
        "'sers1': (2s, 'sers', lens_z, 2.1e10, 0.0, 0.0, 0.3, 112.0, 0.39, 1.06)\n"
        "'nfw1': (3, 'nfw', lens_z, 1e9, 0.1, 0.1, 0.0, 0.0, 10.0)\n"
        "'point1': (4, 'point', lens_z, {1e5, 1e7}, {-0.30, -0.20}, {-0.05, 0.05})")
    cfg, issues = lint_text(text, backend="cpu", with_defaults=True)
    assert not any(i.is_error for i in issues), [str(i) for i in issues]
    problem = OptProblem(cfg)
    x = np.array([0.5 * (lo + hi) for lo, hi in problem.bounds])
    res = OptResult(x=x, loss=0.0, fitted=problem.decode(x),
                    scene=problem.make_scene(x), problem=problem, de=None,
                    backend="cpu")
    labels = [m[2] for m in _subhalo_markers(res, (0.0, 0.0))]
    # '1l' anfw: locked galaxy-scale halo forced to LENS -> no marker;
    # '2s' sers: locked but forced to SUB -> marker; nfw: schema default sub;
    # point: optimizable -> sub.
    assert len(labels) == 3
    assert labels[0].startswith("S2") and labels[1].startswith("S3")
    assert labels[2].startswith("S4")


def test_make_triptych_end_to_end_with_fake_backend():
    cfg = _cfg()
    obs = build_obs(cfg)
    backend = FakeBackend(obs.positions, obs.magnifications)
    result = optimize(cfg, backend=backend,
                      de_overrides={"maxiter": 60, "popsize": 12, "seed": 0,
                                    "early_stopping": True, "early_stop_patience": 15},
                      record_population=False)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "result.png")
        path = make_triptych(result, obs, output_file=out, backend=backend,
                             suptitle="GLADE test")
        assert path == out and _exists_nonempty(out)


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
