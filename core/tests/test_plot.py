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
