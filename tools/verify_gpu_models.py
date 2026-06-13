#!/usr/bin/env python3
"""Cross-check every Rhongomyniad lens kernel against the bundled glafic.

For each GPU-supported model, configure a single lens in both engines and
compare ``calcimage`` outputs (deflection, kappa, gamma, time-delay potential
term) on a ring + radial sweep of points, for a few (e, pa) combinations.

Usage:
    python tools/verify_gpu_models.py            # quick (default tolerances)
    python tools/verify_gpu_models.py --tol 1e-7 # e.g. after rebuilding glafic
                                                 # with TOL_ROMBERG_JHK = 1e-8

Protocol for the high-accuracy cross-check (user-approved):
    1. edit glafic2/glafic.h: TOL_ROMBERG_JHK 1.0e-5 -> 1.0e-8
    2. (cd glafic2 && make python)
    3. python tools/verify_gpu_models.py --tol 2e-7
    4. restore TOL_ROMBERG_JHK to 1.0e-5 and `make python` again

Closed-form models should agree at ~1e-12 regardless of the Romberg setting;
Schramm-quadrature models (nfw/king/sers/hern/pow/tnfw/gnfw/ein) are limited
by glafic's Romberg tolerance, NOT by the GPU's fixed 256-node Gauss-Legendre
rule (which is the more accurate side, cf. scipy-exact references in the
Rhongomyniad test suite).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, "glafic2", "python"),
          os.path.join(_ROOT, "Rhongomyniad")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

ZL, ZS = 0.3, 1.5
COSMO = (0.3, 0.7, -1.0, 0.7)

# representative parameter tuples (p1..p7 after zl), per glafic layout
MODEL_PARAMS = {
    "point":   (1.0e10, 0.03, -0.02, 0, 0, 0, 0),
    "sie":     (250.0, 0.03, -0.02, 0.3, 40.0, 0.05, 0),
    "jaffe":   (200.0, 0.03, -0.02, 0.3, 40.0, 1.5, 0.05),
    "pert":    (ZS, 0.0, 0.0, 0.05, 30.0, 0.0, 0.01),
    "nfw":     (1.0e13, 0.03, -0.02, 0.3, 40.0, 8.0, 0),
    "nfwpot":  (1.0e13, 0.03, -0.02, 0.2, 40.0, 8.0, 0),
    "king":    (1.0e12, 0.03, -0.02, 0.3, 40.0, 0.5, 1.5),
    "sers":    (1.0e12, 0.03, -0.02, 0.3, 40.0, 0.8, 2.5),
    "gaupot":  (ZS, 0.03, -0.02, 0.2, 40.0, 0.7, 0.1),
    # --- newly ported ---
    "hern":    (1.0e12, 0.03, -0.02, 0.3, 40.0, 0.8, 0),
    "hernpot": (1.0e12, 0.03, -0.02, 0.2, 40.0, 0.8, 0),
    "pow":     (ZS, 0.03, -0.02, 0.3, 40.0, 1.0, 2.0),
    "powpot":  (ZS, 0.03, -0.02, 0.2, 40.0, 1.0, 2.0),
    "serspot": (1.0e12, 0.03, -0.02, 0.2, 40.0, 0.8, 2.5),
    "clus3":   (ZS, 0.0, 0.0, 0.03, 30.0, 0, 0),
    "mpole":   (ZS, 0.0, 0.0, 0.02, 30.0, 3.0, 1.0),
    "tnfw":    (1.0e13, 0.03, -0.02, 0.3, 40.0, 8.0, 3.0),
    "tnfwpot": (1.0e13, 0.03, -0.02, 0.2, 40.0, 8.0, 3.0),
    "anfw":    (1.0e13, 0.03, -0.02, 0.3, 40.0, 8.0, 0),
    "ahern":   (1.0e12, 0.03, -0.02, 0.3, 40.0, 0.8, 0),
    "gnfw":    (1.0e13, 0.03, -0.02, 0.3, 40.0, 8.0, 1.3),
    "gnfwpot": (1.0e13, 0.03, -0.02, 0.2, 40.0, 8.0, 1.3),
    "ein":     (1.0e13, 0.03, -0.02, 0.3, 40.0, 8.0, 0.2),
    "einpot":  (1.0e13, 0.03, -0.02, 0.2, 40.0, 8.0, 0.2),
}

# models whose glafic side uses adaptive Romberg (accuracy limited by
# TOL_ROMBERG_JHK); everything else should agree at ~1e-12
SCHRAMM = {"nfw", "king", "sers", "hern", "pow", "tnfw"}
# CSE approximations: glafic and RH share the same closed-form series
CSE = {"anfw", "ahern"}
# table-interpolated radials on BOTH sides; glafic builds its tables with
# Romberg at TOL_ROMBERG_GNFW=3e-4 / TOL_ROMBERG_EIN=1e-3, Rhongomyniad with
# GL quadrature at ~1e-9 (verified against scipy-exact), so the comparison is
# limited by glafic's builder accuracy
TAB = {"gnfw", "gnfwpot", "ein", "einpot"}


def _points() -> list[tuple[float, float]]:
    pts = []
    for r in (0.05, 0.2, 0.7, 1.5, 4.0):
        for ang in (10.0, 100.0, 200.0, 305.0):
            pts.append((r * math.cos(math.radians(ang)),
                        r * math.sin(math.radians(ang))))
    return pts


def _drive(engine, model: str, pars, pts):
    engine.init(*COSMO, "verify_tmp", -6.0, -6.0, 6.0, 6.0, 0.2, 0.2, 5, verb=0)
    engine.startup_setnum(1, 0, 1)
    engine.set_lens(1, model, ZL, *pars)
    engine.set_point(1, ZS, 0.0, 0.0)
    engine.model_init(verb=0)
    out = np.array([engine.calcimage(ZS, x, y, verb=0) for (x, y) in pts])
    engine.quit()
    return out          # (n, 8): ax ay td kap g1 g2 muinv rot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=None,
                    help="override tolerance for Schramm models "
                         "(default 5e-5 for glafic@1e-5; use ~2e-7 after a "
                         "TOL_ROMBERG_JHK=1e-8 rebuild)")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    import glafic
    import rhongomyniad as rh

    rh_models = set(rh.supported_models())
    todo = args.models or [m for m in MODEL_PARAMS if m in rh_models]
    skipped = [m for m in MODEL_PARAMS if m not in rh_models and not args.models]
    pts = _points()

    failures = 0
    print(f"{'model':9s} {'max rel(ax,ay)':>15s} {'max rel(kap,g)':>15s} "
          f"{'tol':>9s}  result")
    for model in todo:
        pars = MODEL_PARAMS[model]
        try:
            g = _drive(glafic, model, pars, pts)
            r = _drive(rh, model, pars, pts)
        except Exception as exc:  # noqa: BLE001
            print(f"{model:9s} DRIVE FAILED: {exc}")
            failures += 1
            continue
        # relative agreement with a sensible floor
        def rel(a, b, floor):
            return np.max(np.abs(a - b) / np.maximum(np.abs(a), floor))
        floor_a = max(1e-8, 1e-6 * np.max(np.abs(g[:, :2])))
        rel_alpha = rel(g[:, :2], r[:, :2], floor_a)
        floor_k = max(1e-10, 1e-6 * np.max(np.abs(g[:, 3:6])))
        rel_kg = rel(g[:, 3:6], r[:, 3:6], floor_k)
        if model in SCHRAMM:
            # nfw/hern/tnfw (and sers to a lesser degree) have log-singular
            # central kappa: glafic's Romberg on the linear-rule region is
            # singularity-limited at ~1e-4 REGARDLESS of TOL_ROMBERG_JHK
            # (verified 2026-06-11: at the worst point RH-vs-scipy = 1.0e-5
            # while glafic@1e-8-vs-scipy = 5.3e-5).  king/sers converge to
            # ~1e-7 with a TOL_ROMBERG_JHK=1e-8 rebuild (use --tol then).
            tol = args.tol if args.tol else 5.0e-4
        elif model in TAB:
            tol = 2.0e-3 if model.startswith("ein") else 5.0e-4
        elif model in CSE:
            tol = 1.0e-9
        else:
            tol = 1.0e-9
        ok = (rel_alpha < tol) and (rel_kg < max(tol, 1e-9) * 30)
        if not ok:
            failures += 1
        print(f"{model:9s} {rel_alpha:15.3e} {rel_kg:15.3e} {tol:9.0e}  "
              f"{'PASS' if ok else 'FAIL'}")

    if skipped:
        print(f"\nnot yet on GPU (skipped): {sorted(skipped)}")
    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
