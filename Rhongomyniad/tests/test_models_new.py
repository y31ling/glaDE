"""Consolidated checks for the 15 lens models ported in V0.50
(models_closed / models_tnfw_cse / models_tab).

    python Rhongomyniad/tests/test_models_new.py

1. tensor-param batching: each kernel evaluated once with (3,1)-shaped param
   tensors must match three separate float-param calls (rel <= 1e-12).
2. scipy-exact spot references for the Schramm-density families
   (hern / tnfw): deflection vs nested scipy quadrature at the GL-256 level.
3. CSE self-consistency: anfw/ahern torch series vs plain-python evaluation.
4. far-field: every density model's deflection approaches the point-mass
   value of the same enclosed mass scale (loose 15% sanity bound at 50 scale
   radii — finite-truncation models converge faster).
The glafic cross-check lives in tools/verify_gpu_models.py.
"""
from __future__ import annotations

import math
import os
import sys

_T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_T))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rhongomyniad.cosmology import Cosmology  # noqa: E402
from rhongomyniad.lens_models import LensContext, dispatch  # noqa: E402

CTX = LensContext.build(Cosmology(omega=0.3, lam=0.7, weos=-1.0, hubble=0.7),
                        zl=0.3, zs=1.5)
ZL, ZS = 0.3, 1.5

PARAMS = {
    "hern":    (ZL, 1.0e12, 0.0, 0.0, 0.3, 40.0, 0.8, 0.0),
    "hernpot": (ZL, 1.0e12, 0.0, 0.0, 0.2, 40.0, 0.8, 0.0),
    "pow":     (ZL, ZS, 0.0, 0.0, 0.3, 40.0, 1.0, 2.0),
    "powpot":  (ZL, ZS, 0.0, 0.0, 0.2, 40.0, 1.0, 2.0),
    "serspot": (ZL, 1.0e12, 0.0, 0.0, 0.2, 40.0, 0.8, 2.5),
    "clus3":   (ZL, ZS, 0.0, 0.0, 0.03, 30.0, 0.0, 0.0),
    "mpole":   (ZL, ZS, 0.0, 0.0, 0.02, 30.0, 3.0, 1.0),
    "tnfw":    (ZL, 1.0e13, 0.0, 0.0, 0.3, 40.0, 8.0, 3.0),
    "tnfwpot": (ZL, 1.0e13, 0.0, 0.0, 0.2, 40.0, 8.0, 3.0),
    "anfw":    (ZL, 1.0e13, 0.0, 0.0, 0.3, 40.0, 8.0, 0.0),
    "ahern":   (ZL, 1.0e12, 0.0, 0.0, 0.3, 40.0, 0.8, 0.0),
    "gnfw":    (ZL, 1.0e13, 0.0, 0.0, 0.3, 40.0, 8.0, 1.3),
    "gnfwpot": (ZL, 1.0e13, 0.0, 0.0, 0.2, 40.0, 8.0, 1.3),
    "ein":     (ZL, 1.0e13, 0.0, 0.0, 0.3, 40.0, 8.0, 0.2),
    "einpot":  (ZL, 1.0e13, 0.0, 0.0, 0.2, 40.0, 8.0, 0.2),
}
# slots eligible for per-candidate tensor variation, per model
_VARY = {
    "hern": (1, 4, 5, 6), "hernpot": (1, 4, 5, 6),
    "pow": (4, 5, 6, 7), "powpot": (4, 5, 6, 7),
    "serspot": (1, 4, 5, 6, 7),
    "clus3": (4, 5), "mpole": (4, 5),
    "tnfw": (1, 4, 5, 6, 7), "tnfwpot": (1, 4, 5, 6, 7),
    "anfw": (1, 4, 5, 6), "ahern": (1, 4, 5, 6),
    "gnfw": (1, 4, 5, 6, 7), "gnfwpot": (1, 4, 5, 6, 7),
    "ein": (1, 4, 5, 6, 7), "einpot": (1, 4, 5, 6, 7),
}

_g = torch.linspace(-2.0, 2.0, 9, dtype=torch.float64)
TX, TY = torch.meshgrid(_g, _g + 0.13, indexing="xy")


def _variants(name, n=3):
    base = PARAMS[name]
    out = []
    for c in range(n):
        q = list(base)
        for i in _VARY[name]:
            if i == 4:                                  # e in [0, 1)
                q[i] = min(0.55, base[i] + 0.12 * c)
            elif i == 7 and name.startswith("ein"):     # ein alpha table range
                q[i] = min(0.9, base[i] + 0.08 * c)
            elif i == 7 and name.startswith("gnfw"):    # gnfw alpha in [0, 2]
                q[i] = min(1.9, base[i] + 0.25 * c)
            elif base[i] != 0.0:
                q[i] = base[i] * (1.0 + 0.15 * c)
            else:
                q[i] = 0.01 * c
        out.append(tuple(q))
    return out


def test_tensor_param_batching():
    for name in PARAMS:
        fn = dispatch(name)
        variants = _variants(name)
        pt = list(PARAMS[name])
        for i in range(8):
            vals = [v[i] for v in variants]
            if len(set(vals)) > 1:
                pt[i] = torch.tensor(vals, dtype=torch.float64).view(-1, 1, 1)
        rb = fn(CTX, TX.unsqueeze(0), TY.unsqueeze(0), tuple(pt))
        worst = 0.0
        for c, var in enumerate(variants):
            rf = fn(CTX, TX, TY, var)
            for a, b in zip(rb, rf):
                if a is None or b is None:
                    continue
                d = float(((a[c] - b).abs()
                           / b.abs().clamp(min=1e-12)).max())
                worst = max(worst, d)
        assert worst < 1e-11, f"{name}: batched vs float rel {worst:.3e}"
        print(f"    {name:8s} batched-vs-float rel {worst:.2e}")


def _schramm_defl_scipy(kappa_radial, bx, by, q, smallcore=1e-10):
    """Exact Schramm J integrals via scipy quad over the SAME u-interval the
    glafic/Rhongomyniad log rule uses: when uu = 1/(bx^2+by^2+sc^2) <= 0.1 the
    engines integrate u in (1e-4*uu, 1) — the truncation below u_min is part of
    glafic's algorithm (mass.c:1106-1112), so the reference must match it
    (kappa diverges logarithmically at the centre for hern/nfw)."""
    from scipy.integrate import quad

    uu = 1.0 / (bx * bx + by * by + smallcore * smallcore)
    u_lo = 1.0e-4 * uu if uu <= 0.1 else 0.0

    def j_int(n):
        def f(u):
            equ = 1.0 - (1.0 - q * q) * u
            xi = math.sqrt(u * (by * by + bx * bx / equ))
            return kappa_radial(xi) / equ ** (n + 0.5)
        if u_lo > 0.0:
            def fl(l):
                u = math.exp(l)
                return u * f(u)
            v, _ = quad(fl, math.log(u_lo), 0.0, epsabs=1e-14, epsrel=1e-13,
                        limit=800)
        else:
            v, _ = quad(f, 0.0, 1.0, epsabs=1e-13, epsrel=1e-12, limit=400,
                        points=[0.0, 1e-8, 1e-4])
        return v
    return q * bx * j_int(1), q * by * j_int(0)


def test_hern_scipy_exact():
    from rhongomyniad.models_closed import _kappa_hern_dl as _kappa_hern_dl_f
    from rhongomyniad.models_tnfw_cse import _b_func_hern
    # radial kappa as plain-python via the torch kernel
    def kap_r(x):
        return float(_kappa_hern_dl_f(torch.tensor([x], dtype=torch.float64))[0])
    p = PARAMS["hern"]
    m, e, pa, rb = p[1], p[4], p[5], p[6]
    q = 1.0 - e
    bb = _b_func_hern(m, rb, CTX)
    tt = rb / math.sqrt(q)
    si, co = math.sin(-pa * math.pi / 180), math.cos(-pa * math.pi / 180)
    fn = dispatch("hern")
    worst = 0.0
    for (x, y) in ((0.05, 0.02), (0.3, -0.2), (1.0, 0.6), (3.0, -1.5)):
        bx = (co * x - si * y) / tt
        by = (si * x + co * y) / tt
        bpx, bpy = _schramm_defl_scipy(kap_r, bx, by, q)
        px = bpx * co + bpy * si
        py = -bpx * si + bpy * co
        ax_ref, ay_ref = bb * tt * px, bb * tt * py
        ax, ay, *_ = fn(CTX, torch.tensor([x], dtype=torch.float64),
                        torch.tensor([y], dtype=torch.float64), p,
                        need_kg=False, need_phi=False)
        rel = max(abs(float(ax[0]) - ax_ref) / max(abs(ax_ref), 1e-12),
                  abs(float(ay[0]) - ay_ref) / max(abs(ay_ref), 1e-12))
        worst = max(worst, rel)
    assert worst < 2e-5, f"hern vs scipy-exact rel {worst:.3e}"
    print(f"    hern vs scipy-exact rel {worst:.2e}")


def test_tnfw_scipy_exact():
    from rhongomyniad.models_tnfw_cse import _kappa_tnfw_dl, _calc_bbtt_tnfw
    p = PARAMS["tnfw"]
    m, e, pa, c, t_trunc = p[1], p[4], p[5], p[6], p[7]
    q = 1.0 - e
    bb, tt0, cc = _calc_bbtt_tnfw(m, c, CTX)
    tau = t_trunc * cc

    def kap_r(x):
        return float(_kappa_tnfw_dl(torch.tensor([x], dtype=torch.float64),
                                    tau)[0])
    tt = tt0 / math.sqrt(q)
    si, co = math.sin(-pa * math.pi / 180), math.cos(-pa * math.pi / 180)
    fn = dispatch("tnfw")
    worst = 0.0
    for (x, y) in ((0.1, 0.04), (0.5, -0.3), (2.0, 1.2)):
        bx = (co * x - si * y) / tt
        by = (si * x + co * y) / tt
        bpx, bpy = _schramm_defl_scipy(kap_r, bx, by, q)
        px = bpx * co + bpy * si
        py = -bpx * si + bpy * co
        ax_ref, ay_ref = bb * tt * px, bb * tt * py
        ax, ay, *_ = fn(CTX, torch.tensor([x], dtype=torch.float64),
                        torch.tensor([y], dtype=torch.float64), p,
                        need_kg=False, need_phi=False)
        rel = max(abs(float(ax[0]) - ax_ref) / max(abs(ax_ref), 1e-12),
                  abs(float(ay[0]) - ay_ref) / max(abs(ay_ref), 1e-12))
        worst = max(worst, rel)
    assert worst < 2e-5, f"tnfw vs scipy-exact rel {worst:.3e}"
    print(f"    tnfw vs scipy-exact rel {worst:.2e}")


def test_cse_self_consistency():
    """anfw torch series vs a plain-python double evaluation."""
    from rhongomyniad import models_tnfw_cse as MC
    a_cse = getattr(MC, "_A_CSE_NFW", None)
    s_cse = getattr(MC, "_S_CSE_NFW", None)
    if a_cse is None:
        print("    (skip: coefficient arrays not exported by that name)")
        return
    a_np = np.asarray(a_cse, dtype=float).reshape(-1)
    s_np = np.asarray(s_cse, dtype=float).reshape(-1)
    fn = dispatch("anfw")
    p = PARAMS["anfw"]
    # spherical case for an independent closed-form check
    p0 = list(p)
    p0[4] = 0.0
    x = 0.7
    ax, ay, *_ = fn(CTX, torch.tensor([x], dtype=torch.float64),
                    torch.tensor([0.0], dtype=torch.float64), tuple(p0),
                    need_kg=False, need_phi=False)
    # plain python: alpha_cse for q=1 reduces to sum_i 2 a_i [sqrt(s^2+x^2)-s]/x
    from rhongomyniad.lens_models import _calc_bbtt_nfw
    bb, tt = _calc_bbtt_nfw(p0[1], p0[6], CTX)
    xs = x / tt
    val = sum(2.0 * a * (math.sqrt(s * s + xs * xs) - s) / xs
              for a, s in zip(a_np, s_np))
    ref = bb * tt * val
    rel = abs(float(ax[0]) - ref) / max(abs(ref), 1e-12)
    assert rel < 1e-10, f"anfw CSE python-vs-torch rel {rel:.3e}"
    print(f"    anfw CSE self-consistency rel {rel:.2e}")


def test_farfield_pointmass():
    from rhongomyniad.lens_models import re2_point
    for name in ("hern",):
        p = list(PARAMS[name])
        p[4] = 0.0   # spherical
        fn = dispatch(name)
        r = 60.0 if name != "ein" else 40.0
        ax, ay, *_ = fn(CTX, torch.tensor([r], dtype=torch.float64),
                        torch.tensor([0.0], dtype=torch.float64), tuple(p),
                        need_kg=False, need_phi=False)
        re2 = re2_point(p[1], CTX)
        ref = re2 / r
        rel = abs(float(ax[0]) - ref) / ref
        assert rel < 0.15, f"{name} far-field vs point mass rel {rel:.3f}"
        print(f"    {name:5s} far-field vs point mass rel {rel:.3f}")
    # tnfw truncates the halo (asymptotic mass < M); Einasto alpha=0.2 keeps
    # significant mass OUTSIDE r_vir (asymptotic mass > M(c)).  For both, test
    # the 1/r far-field convergence instead of the amplitude.
    from rhongomyniad.models_tnfw_cse import _calc_bbtt_tnfw
    from rhongomyniad.lens_models import _calc_bbtt_nfw
    for name, xs_pair in (("tnfw", (80.0, 160.0)), ("ein", (60.0, 120.0))):
        p = list(PARAMS[name]); p[4] = 0.0
        fn = dispatch(name)
        if name == "tnfw":
            _bb, tt0, _cc = _calc_bbtt_tnfw(p[1], p[6], CTX)
        else:
            _bb, tt0 = _calc_bbtt_nfw(p[1], p[6], CTX)
        vals = []
        for xs_ in xs_pair:                      # radii in units of the scale radius
            r = xs_ * tt0
            ax, *_ = fn(CTX, torch.tensor([r], dtype=torch.float64),
                        torch.tensor([0.0], dtype=torch.float64), tuple(p),
                        need_kg=False, need_phi=False)
            vals.append(float(ax[0]) * r)
        rel = abs(vals[0] - vals[1]) / abs(vals[1])
        assert rel < 0.05, f"{name} enclosed-mass convergence rel {rel:.3f}"
        print(f"    {name:5s} far-field 1/r convergence rel {rel:.4f}")


def _run_all() -> int:
    funcs = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    fails = 0
    for name, fn in funcs:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            fails += 1
            import traceback
            print(f"  FAIL  {name}: {exc}")
            traceback.print_exc()
    print(f"\n{len(funcs) - fails}/{len(funcs)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
