"""
Truncated-NFW and CSE-approximated lens-model kernels, batched via PyTorch.

Ported verbatim from glafic:
    tnfw     -- truncated NFW elliptical DENSITY   (mass.c:2513-2579, radial
                closed forms mass.c:2298-2507, Schramm integrals)
    tnfwpot  -- truncated NFW elliptical POTENTIAL (mass.c:2233-2276)
    anfw     -- CSE-approximated analytic NFW      (mass.c:1146-1198,
                app_ell.c:230-248 / 344-357, coefficients app_ell.c:12-102)
    ahern    -- CSE-approximated analytic Hernquist (mass.c:1677-1729,
                app_ell.c:290-308, coefficients app_ell.c:104-188)

Kernel signature and conventions follow rhongomyniad.lens_models exactly:
    kapgam_<name>(ctx, tx, ty, p, smallcore=K.DEF_SMALLCORE,
                  need_kg=True, need_phi=True) -> (ax, ay, kap, g1, g2, phi)
with need_kg=False -> (ax, ay, None, None, None, None) and need_phi gating phi.

Tensor-param support: every physical parameter p[i] may be a python float OR
a torch tensor broadcastable against tx/ty (e.g. (C,1,1) vs (C,ny,nx)).  When
parameters are floats the code path is bit-identical to a scalar port of the
C; when tensors, python-level value branches are replaced by torch.where with
both branches guarded so no NaN/Inf leaks from the unused branch.  Parameter
validation raises only for float params.

glafic's static caches (tau_tnfw_sav, b_func/rs memoization, pa trig cache)
become plain recomputation here.
"""

from __future__ import annotations

import math
import os

import torch

from . import constants as K
from . import cosmology as cos_mod
from .elliptical import (
    ell_integ_i, ell_integ_j, ell_integ_k,
    ell_pxpy, ell_pxxpyy, u_calc_tensor,
)
from .lens_models import (
    LensContext,
    _pf, _is_t, _t_sqrt, _t_log, _full_b, _q_tensor,
    pa_trig, pa_minus_90_trig,
    _func_hern_dl,
    _calc_bbtt_nfw, _b_func_nfw, _rs_nfw,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _calc_bbtt_tnfw(m, c, ctx: LensContext,
                    nfw_users: int = K.DEF_NFW_USERS):
    """calc_bbtt_tnfw (mass.c:2278-2291).  Returns (bb, tt, cc).

    Identical to _calc_bbtt_nfw but also returns the effective concentration
    `cc` that scales the truncation parameter: tau = t * cc.
    """
    if nfw_users == 0:
        cc = c
        bb = _b_func_nfw(m, c, ctx)
        tt = cos_mod.rtotheta_dis(_rs_nfw(m, c, ctx.delome), ctx.dis_ol)
    else:
        tt = c
        cc = _rs_nfw(m, 1.0, ctx.delome) / cos_mod.thetator_dis(c, ctx.dis_ol)
        bb = _b_func_nfw(m, cc, ctx)
    return bb, tt, cc


def _b_func_hern(m, rb, ctx: LensContext):
    """b_func_hern (mass.c:1536-1542)."""
    rr = cos_mod.thetator_dis(rb, ctx.dis_ol)
    return (m * ctx.inv_sigma_crit / (rr * rr)) / (2.0 * math.pi)


def _func_tnfw_dl(x: torch.Tensor, tau) -> torch.Tensor:
    """func_tnfw_dl (mass.c:2493-2500): log(x / (sqrt(tau^2 + x^2) + tau))."""
    x_s = torch.clamp(x, min=1.0e-300)
    xx = x_s / (torch.sqrt(tau * tau + x_s * x_s) + tau)
    return torch.log(xx)


# ---------------------------------------------------------------------------
# Truncated-NFW radial closed forms (mass.c:2298-2507).
#
# Each takes x as a tensor (any shape) and tau as a float OR a tensor
# broadcastable against x.  glafic's static tau_tnfw_sav becomes an explicit
# argument.  Branch thresholds are EXACTLY glafic's; per-point branches use
# torch.where with guarded inputs.
# ---------------------------------------------------------------------------
def _kappa_tnfw_func1(x: torch.Tensor, u: torch.Tensor,
                      tau, t2, t4) -> torch.Tensor:
    """kappa_tnfw_dl_func1 (mass.c:2333-2347)."""
    # x ~ 1 closed value (depends only on tau); threshold 1e-7.
    near = (x > (1.0 - 1.0e-7)) & (x < (1.0 + 1.0e-7))
    st1 = _t_sqrt(1.0 + t2)
    mid = (tau / (24.0 * (1.0 + t2) * (1.0 + t2) * (1.0 + t2) * (1.0 + t2))) \
        * (tau * (-6.0 + 62.0 * t4 + 4.0 * t4 * t2
                  + t2 * (52.0 - 30.0 * math.pi * st1))
           + 3.0 * st1 * (1.0 + 7.0 * t2 - 4.0 * t4)
           * _t_log(1.0 + 2.0 * tau * (tau + st1)))

    # Regular branch with guards (the near-1 points are replaced below).
    x_s = torch.clamp(x, min=1.0e-300)
    u_safe = torch.where(near, torch.full_like(u, 4.0), u)
    st2u = torch.sqrt(t2 + u)
    fx = _func_hern_dl(x)
    lx = _func_tnfw_dl(x_s, tau)
    f = (t4 / (4.0 * (t2 + 1.0) * (t2 + 1.0) * (t2 + 1.0))) \
        * (2.0 * (t2 + 1.0) * (1.0 - fx) / (u_safe - 1.0)
           + 8.0 * fx
           + (t4 - 1.0) / (t2 * (t2 + u))
           - math.pi * (4.0 * (t2 + u) + t2 + 1.0) / (st2u * (t2 + u))
           + lx * (t2 * (t4 - 1.0) + (t2 + u) * (3.0 * t4 - 6.0 * t2 - 1.0))
           / (tau * t2 * st2u * (t2 + u)))
    return torch.where(near, _full_b(x, mid), f)


def _kappa_tnfw_func2(x: torch.Tensor, u: torch.Tensor,
                      tau, t2, t4) -> torch.Tensor:
    """kappa_tnfw_dl_func2 (mass.c:2349-2357) -- large-x asymptotic series.

    Only selected for u > 1e4 (or u >= 1e5*t2); x is clamped so the unused
    small-x evaluations stay finite.
    """
    x_s = torch.clamp(x, min=1.0e-25)
    u_s = x_s * x_s
    u3 = u_s * u_s * u_s
    f = (4.0 * t4 / (15.0 * u3)
         - 5.0 * math.pi * t4 / (32.0 * u3 * x_s)
         - 8.0 * (t4 * (-3.0 + 2.0 * t2)) / (35.0 * u3 * u_s)
         + 35.0 * t4 * math.pi * (t2 - 1.0) / (128.0 * u3 * u_s * x_s))
    return f


def _dkappa_tnfw_func1(x: torch.Tensor, u: torch.Tensor,
                       tau, t2, t4) -> torch.Tensor:
    """dkappa_tnfw_dl_func1 (mass.c:2394-2411)."""
    # x ~ 1 closed value; threshold 1e-6.
    near = (x > (1.0 - 1.0e-6)) & (x < (1.0 + 1.0e-6))
    st1 = _t_sqrt(1.0 + t2)
    mid = ((-1.0) * tau / (120.0 * (1.0 + t2) * (1.0 + t2) * (1.0 + t2)
                           * (1.0 + t2) * (1.0 + t2))) \
        * (2.0 * tau * (-15.0 + 271.0 * t4 + 56.0 * t4 * t2 + 12.0 * t4 * t4
                        + t2 * (212.0 - 105.0 * math.pi * st1))
           - 15.0 * st1 * (-1.0 - 9.0 * t2 + 6.0 * t4)
           * _t_log(1.0 + 2.0 * tau * (tau + st1)))

    x_s = torch.clamp(x, min=1.0e-300)
    u_safe = torch.where(near, torch.full_like(u, 4.0), u)
    st2u = torch.sqrt(t2 + u)
    t2u2 = (t2 + u) * (t2 + u)
    t2u52 = t2u2 * st2u
    fx = _func_hern_dl(x)
    lx = _func_tnfw_dl(x_s, tau)
    f = (t4 / (4.0 * (t2 + 1.0) * (t2 + 1.0) * (t2 + 1.0))) \
        * (2.0 * (t2 + 1.0) * (3.0 * u_safe * fx - 2.0 * u_safe - 1.0)
           / (x_s * (u_safe - 1.0) * (u_safe - 1.0))
           + 8.0 * (1.0 - u_safe * fx) / (x_s * (u_safe - 1.0))
           - 2.0 * x * (t4 - 1.0) / (t2 * t2u2)
           + math.pi * x * (4.0 * (t2 + u) + 3.0 * (t2 + 1.0)) / t2u52
           + (t2 * (t4 - 1.0) + (t2 + u) * (3.0 * t4 - 6.0 * t2 - 1.0))
           / (x_s * t2 * t2u2)
           - lx * x * (3.0 * t2 * (t4 - 1.0)
                       + (t2 + u) * (3.0 * t4 - 6.0 * t2 - 1.0))
           / (tau * t2 * t2u52))
    return torch.where(near, _full_b(x, mid), f)


def _dkappa_tnfw_func2(x: torch.Tensor, u: torch.Tensor,
                       tau, t2, t4) -> torch.Tensor:
    """dkappa_tnfw_dl_func2 (mass.c:2413-2421)."""
    x_s = torch.clamp(x, min=1.0e-25)
    u_s = x_s * x_s
    u3 = u_s * u_s * u_s
    f = ((-8.0 * t4) / (5.0 * u3 * x_s)
         + 35.0 * math.pi * t4 / (32.0 * u3 * u_s)
         + 64.0 * (t4 * (-3.0 + 2.0 * t2)) / (35.0 * u3 * u_s * x_s)
         - 315.0 * t4 * math.pi * (t2 - 1.0) / (128.0 * u3 * u_s * u_s))
    return f


def _tnfw_interp(x: torch.Tensor, tau, t2, t4, func1, func2) -> torch.Tensor:
    """The log-interpolation patch for the small-tau intermediate regime
    (mass.c:2319-2326 / 2380-2387): interpolate log f between
    func2 at xa=1e2 and func1 at xb=1e3*tau."""
    xa = torch.tensor(1.0e2, dtype=x.dtype, device=x.device)
    fa = func2(xa, xa * xa, tau, t2, t4)
    if torch.is_tensor(tau):
        xb = torch.clamp(1.0e3 * tau, min=1.0e-300)
    else:
        xb = torch.tensor(1.0e3 * tau, dtype=x.dtype, device=x.device)
        xb = torch.clamp(xb, min=1.0e-300)
    fb = func1(xb, xb * xb, tau, t2, t4)
    log_fa = torch.log(torch.clamp(fa, min=1.0e-300))
    log_fb = torch.log(torch.clamp(fb, min=1.0e-300))
    log_xa = math.log(1.0e2)
    log_xb = torch.log(xb)
    den = log_xa - log_xb
    den = torch.where(torch.abs(den) < 1.0e-12, torch.ones_like(den), den)
    log_x = torch.log(torch.clamp(x, min=1.0e-300))
    return torch.exp((log_fa - log_fb) / den * (log_x - log_xb) + log_fb)


def _tnfw_branch_dl(x: torch.Tensor, tau, func1, func2) -> torch.Tensor:
    """Shared branch structure of kappa_tnfw_dl (mass.c:2298-2331) and
    dkappa_tnfw_dl (mass.c:2359-2392):

        t2 > 1e-4:  u < 1e5*t2  -> func1   else func2
        t2 <= 1e-4: u > 1e4     -> func2
                    u < 1e6*t2  -> func1
                    else        -> log-interpolation patch
    """
    t2 = tau * tau
    t4 = t2 * t2
    u = x * x

    if not torch.is_tensor(t2):
        if t2 > 1.0e-4:
            f1 = func1(x, u, tau, t2, t4)
            f2 = func2(x, u, tau, t2, t4)
            return torch.where(u < (1.0e5 * t2), f1, f2)
        f1 = func1(x, u, tau, t2, t4)
        f2 = func2(x, u, tau, t2, t4)
        fi = _tnfw_interp(x, tau, t2, t4, func1, func2)
        return torch.where(u > 1.0e4, f2,
                           torch.where(u < (1.0e6 * t2), f1, fi))

    # Tensor tau: compute everything, blend per-element.
    f1 = func1(x, u, tau, t2, t4)
    f2 = func2(x, u, tau, t2, t4)
    fi = _tnfw_interp(x, tau, t2, t4, func1, func2)
    sel_big = torch.where(u < (1.0e5 * t2), f1, f2)
    sel_sml = torch.where(u > 1.0e4, f2,
                          torch.where(u < (1.0e6 * t2), f1, fi))
    return torch.where(t2 > 1.0e-4, sel_big, sel_sml)


def _kappa_tnfw_dl(x: torch.Tensor, tau) -> torch.Tensor:
    """kappa_tnfw_dl (mass.c:2298-2331)."""
    return _tnfw_branch_dl(x, tau, _kappa_tnfw_func1, _kappa_tnfw_func2)


def _dkappa_tnfw_dl(x: torch.Tensor, tau) -> torch.Tensor:
    """dkappa_tnfw_dl (mass.c:2359-2392)."""
    return _tnfw_branch_dl(x, tau, _dkappa_tnfw_func1, _dkappa_tnfw_func2)


def _dphi_tnfw_dl(x: torch.Tensor, tau) -> torch.Tensor:
    """dphi_tnfw_dl (mass.c:2423-2450)."""
    t2 = tau * tau
    t4 = t2 * t2
    u = x * x
    lnt = _t_log(tau)
    ln2 = math.log(2.0)
    x_s = torch.clamp(x, min=1.0e-300)

    # Main branch.
    st2u = torch.sqrt(t2 + u)
    fx = _func_hern_dl(x)
    lx = _func_tnfw_dl(x_s, tau)
    f_main = (t4 / (2.0 * (t2 + 1.0) * (t2 + 1.0) * (t2 + 1.0) * x_s)) \
        * (2.0 * (t2 + 4.0 * u - 3.0) * fx
           + (math.pi * (3.0 * t2 - 1.0) + 2.0 * tau * (t2 - 3.0) * lnt) / tau
           + ((-1.0) * t2 * tau * math.pi * (4.0 * u + 3.0 * t2 - 1.0)
              + (2.0 * t4 * t2 - 6.0 * t4 + u * (3.0 * t4 - 6.0 * t2 - 1.0)) * lx)
           / (t2 * tau * st2u))

    # Small-x branches.
    log_x = torch.log(x_s)
    f_small_a = (1.0 / (4.0 * ((t2 + 1.0) * (t2 + 1.0) * (t2 + 1.0)))) \
        * (3.0 * t2 + (-1.0) * math.pi * tau - 5.0 * math.pi * tau * t2
           + t4 * t2 * (-1.0 + 2.0 * ln2) + t4 * (2.0 + 10.0 * ln2)
           + (2.0 + 6.0 * t2 - 4.0 * t4) * (ln2 + lnt)
           - 2.0 * (t2 + 1.0) * (t2 + 1.0) * (t2 + 1.0) * log_x) * x
    f_small_b = (x * (2.0 * ln2 + 2.0 * lnt - 2.0 * log_x
                      - (math.pi * tau)) / 4.0) + 3.0 * x * t2 / 4.0

    if not torch.is_tensor(tau):
        if tau > 3.0e-3:
            cond_main = (x > 3.0e-3) | (u > (t2 * 1.0e-3))
            return torch.where(cond_main, f_main, f_small_a)
        cond_main = u > (t2 * 1.0e-3)
        return torch.where(cond_main, f_main, f_small_b)

    cond_main = ((x > 3.0e-3) & (tau > 3.0e-3)) | (u > (t2 * 1.0e-3))
    f_small = torch.where(tau > 3.0e-3, f_small_a, f_small_b)
    return torch.where(cond_main, f_main, f_small)


def _phi_tnfw_dl(x: torch.Tensor, tau) -> torch.Tensor:
    """phi_tnfw_dl (mass.c:2452-2478)."""
    t2 = tau * tau
    t4 = t2 * t2
    u = x * x
    ln2 = math.log(2.0)
    lnt = _t_log(tau)
    x_s = torch.clamp(x, min=1.0e-300)
    u_s = torch.clamp(u, min=1.0e-300)

    # Main branch.
    st2u = torch.sqrt(t2 + u)
    fx = _func_hern_dl(x)
    lx = _func_tnfw_dl(x_s, tau)
    f_main = (1.0 / (4.0 * (t2 + 1.0) * (t2 + 1.0) * (t2 + 1.0))) \
        * (2.0 * t2 * tau * math.pi
           * (-4.0 * tau * st2u + (3.0 * t2 - 1.0) * torch.log(st2u + tau))
           + 2.0 * tau * (3.0 * t4 - 6.0 * t2 - 1.0) * st2u * lx
           + 2.0 * t4 * (t2 - 3.0) * lx * lx
           + 16.0 * t4 * (u - 1.0) * fx
           + 2.0 * t4 * (t2 - 3.0) * (u - 1.0) * fx * fx
           + t2 * (2.0 * t2 * (t2 - 3.0) * lnt - 3.0 * t4 - 2.0 * t2 + 1.0)
           * torch.log(u_s)
           + 2.0 * t2 * (t2 * (4.0 * tau * math.pi + (t2 - 3.0) * ln2 * ln2
                               + 8.0 * ln2)
                         - (ln2 + lnt) * (1.0 + 6.0 * t2 - 3.0 * t4
                                          + t2 * (t2 - 3.0) * (ln2 + lnt))
                         - tau * math.pi * (3.0 * t2 - 1.0) * (ln2 + lnt)))

    # Small-x branches.
    log_x = torch.log(x_s)
    f_small_a = (1.0 / (8.0 * ((t2 + 1.0) * (t2 + 1.0) * (t2 + 1.0)))) \
        * (1.0 + tau * ((-1.0) * math.pi
                        + tau * (6.0 + tau * ((-5.0) * math.pi
                                              + tau * (5.0 + (5.0 + t2) * 2.0 * ln2))))
           + (2.0 + 6.0 * t2 - 4.0 * t4) * (ln2 + lnt)
           - 2.0 * (t2 + 1.0) * (t2 + 1.0) * (t2 + 1.0) * log_x) * u
    f_small_b = u * (1.0 + 2.0 * ln2 + 2.0 * lnt - 2.0 * log_x
                     - (math.pi * tau)) / 8.0

    if not torch.is_tensor(tau):
        if tau > 3.0e-3:
            cond_main = (x > 3.0e-3) | (u > (t2 * 1.0e-3))
            return torch.where(cond_main, f_main, f_small_a)
        cond_main = u > (t2 * 1.0e-3)
        return torch.where(cond_main, f_main, f_small_b)

    cond_main = ((x > 3.0e-3) & (tau > 3.0e-3)) | (u > (t2 * 1.0e-3))
    f_small = torch.where(tau > 3.0e-3, f_small_a, f_small_b)
    return torch.where(cond_main, f_main, f_small)


def _ddphi_tnfw_dl(x: torch.Tensor, tau) -> torch.Tensor:
    """ddphi_tnfw_dl (mass.c:2293-2296) = 2*kappa - dphi/x."""
    x_s = torch.clamp(x, min=1.0e-300)
    return 2.0 * _kappa_tnfw_dl(x, tau) - _dphi_tnfw_dl(x, tau) / x_s


# ---------------------------------------------------------------------------
# 1) tnfw -- truncated NFW elliptical DENSITY (mass.c:2513-2579)
#    p[1]=M  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=c  p[7]=t
# ---------------------------------------------------------------------------
def kapgam_tnfw(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                p: tuple, smallcore: float = K.DEF_SMALLCORE,
                need_kg: bool = True, need_phi: bool = True,
                nfw_users: int = K.DEF_NFW_USERS):
    m = _pf(p[1]); x0 = _pf(p[2]); y0 = _pf(p[3])
    e = _pf(p[4]); pa = _pf(p[5]); c = _pf(p[6]); t = _pf(p[7])
    if not _is_t(m, e, c, t):
        if m <= 0.0: raise ValueError("tnfw: m must be positive")
        if not (0.0 <= e < 1.0): raise ValueError("tnfw: e in [0,1)")
        if c <= 0.0: raise ValueError("tnfw: c must be positive")
        if t <= 0.0: raise ValueError("tnfw: t must be positive")

    q = 1.0 - e
    bb, tt, cc = _calc_bbtt_tnfw(m, c, ctx, nfw_users=nfw_users)
    tt = tt / _t_sqrt(q)
    tau = t * cc                        # tnfw_set_tau(t * cc) (mass.c:2530)
    si, co = pa_trig(pa)

    bx = (co * (tx - x0) - si * (ty - y0)) / tt
    by = (si * (tx - x0) + co * (ty - y0)) / tt
    q_t = _q_tensor(q, tx)

    def kappa_fn(x): return _kappa_tnfw_dl(x, tau)
    def dkappa_fn(x): return _dkappa_tnfw_dl(x, tau)
    def dphi_fn(x): return _dphi_tnfw_dl(x, tau)

    j1 = ell_integ_j(kappa_fn, 1, bx, by, q_t, smallcore)
    j0 = ell_integ_j(kappa_fn, 0, bx, by, q_t, smallcore)
    bpx = q * bx * j1
    bpy = q * by * j0
    px, py = ell_pxpy(bpx, bpy, si, co)
    ax = bb * tt * px
    ay = bb * tt * py

    if not need_kg:
        return ax, ay, None, None, None, None

    k2 = ell_integ_k(dkappa_fn, 2, bx, by, q_t, smallcore)
    k0 = ell_integ_k(dkappa_fn, 0, bx, by, q_t, smallcore)
    k1 = ell_integ_k(dkappa_fn, 1, bx, by, q_t, smallcore)
    bpxx = 2.0 * q * bx * bx * k2 + q * j1
    bpyy = 2.0 * q * by * by * k0 + q * j0
    bpxy = 2.0 * q * bx * by * k1
    pxx, pyy, pxy = ell_pxxpyy(bpxx, bpyy, bpxy, si, co)

    kap = 0.5 * bb * (pxx + pyy)
    gam1 = 0.5 * bb * (pxx - pyy)
    gam2 = bb * pxy
    phi = None
    if need_phi:
        phi_int = ell_integ_i(dphi_fn, bx, by, q_t, smallcore)
        phi = 0.5 * q * bb * phi_int * tt * tt
    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# 2) tnfwpot -- truncated NFW elliptical POTENTIAL (mass.c:2233-2276)
#    p[1]=M  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=c  p[7]=t
# ---------------------------------------------------------------------------
def kapgam_tnfwpot(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                   p: tuple, smallcore: float = K.DEF_SMALLCORE,
                   need_kg: bool = True, need_phi: bool = True,
                   nfw_users: int = K.DEF_NFW_USERS):
    m = _pf(p[1]); x0 = _pf(p[2]); y0 = _pf(p[3])
    e = _pf(p[4]); pa = _pf(p[5]); c = _pf(p[6]); t = _pf(p[7])
    if not _is_t(m, e, c, t):
        if m <= 0.0: raise ValueError("tnfwpot: m must be positive")
        if not (0.0 <= e < 1.0): raise ValueError("tnfwpot: e in [0,1)")
        if c <= 0.0: raise ValueError("tnfwpot: c must be positive")
        if t <= 0.0: raise ValueError("tnfwpot: t must be positive")

    bb, tt, cc = _calc_bbtt_tnfw(m, c, ctx, nfw_users=nfw_users)
    tau = t * cc                        # tnfw_set_tau(t * cc) (mass.c:2247)
    si, co = pa_trig(pa)

    u0, u_x, u_y, u_xx, u_xy, u_yy = u_calc_tensor(
        (tx - x0) / tt, (ty - y0) / tt, e, si, co, smallcore)

    a = bb * _dphi_tnfw_dl(u0, tau)
    ax = a * u_x * tt
    ay = a * u_y * tt

    if not need_kg:
        return ax, ay, None, None, None, None
    b = bb * _ddphi_tnfw_dl(u0, tau)
    pxx = b * u_x * u_x + a * u_xx
    pxy = b * u_x * u_y + a * u_xy
    pyy = b * u_y * u_y + a * u_yy
    kap = 0.5 * (pxx + pyy)
    gam1 = 0.5 * (pxx - pyy)
    gam2 = pxy
    phi = None
    if need_phi:
        phi = bb * _phi_tnfw_dl(u0, tau) * tt * tt
    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# CSE (cored steep ellipsoid) kernels (app_ell.c:314-357) and the fixed
# coefficient tables for the NFW / Hernquist decompositions
# (app_ell.c:12-102 and 104-188, M. Oguri arXiv:2106.11464).
# Tables are (s_i, w_i) pairs, ported VERBATIM.
# ---------------------------------------------------------------------------
_CSE_NFW_S = (
    1.082411e-06, 8.786566e-06, 3.292868e-06, 1.860019e-05, 3.274231e-05,
    6.232485e-05, 9.256333e-05, 1.546762e-04, 2.097321e-04, 3.391140e-04,
    5.178790e-04, 8.636736e-04, 1.405152e-03, 2.193855e-03, 3.179572e-03,
    4.970987e-03, 7.631970e-03, 1.119413e-02, 1.827267e-02, 2.945251e-02,
    4.562723e-02, 6.782509e-02, 1.596987e-01, 1.127751e-01, 2.169469e-01,
    3.423835e-01, 5.194527e-01, 8.623185e-01, 1.382737e+00, 2.034929e+00,
    3.402979e+00, 5.594276e+00, 8.052345e+00, 1.349045e+01, 2.603825e+01,
    4.736823e+01, 6.559320e+01, 1.087932e+02, 1.477673e+02, 2.495341e+02,
    4.305999e+02, 7.760206e+02, 2.143057e+03, 1.935749e+03,
)
_CSE_NFW_W = (
    1.648988e-18, 6.274458e-16, 3.646620e-17, 3.459206e-15, 2.457389e-14,
    1.059319e-13, 4.211597e-13, 1.142832e-12, 4.391215e-12, 1.556500e-11,
    6.951271e-11, 3.147466e-10, 1.379109e-09, 3.829778e-09, 1.384858e-08,
    5.370951e-08, 1.804384e-07, 5.788608e-07, 3.205256e-06, 1.102422e-05,
    4.093971e-05, 1.282206e-04, 4.575541e-04, 7.995270e-04, 5.013701e-03,
    1.403508e-02, 5.230727e-02, 1.898907e-01, 3.643448e-01, 7.203734e-01,
    1.717667e+00, 2.217566e+00, 3.187447e+00, 8.194898e+00, 1.765210e+01,
    1.974319e+01, 2.783688e+01, 4.482311e+01, 5.598897e+01, 1.426485e+02,
    2.279833e+02, 5.401335e+02, 9.743682e+02, 1.775124e+03,
)
_CSE_HERN_S = (
    1.199110e-06, 3.751762e-06, 9.927207e-06, 2.206076e-05, 3.781528e-05,
    6.659808e-05, 1.154366e-04, 1.924150e-04, 3.040440e-04, 4.683051e-04,
    7.745084e-04, 1.175953e-03, 1.675459e-03, 2.801948e-03, 9.712807e-03,
    5.469589e-03, 1.104654e-02, 1.893893e-02, 2.792864e-02, 4.152834e-02,
    6.640398e-02, 1.107083e-01, 1.648028e-01, 2.839601e-01, 4.129439e-01,
    8.239115e-01, 6.031726e-01, 1.145604e+00, 1.401895e+00, 2.512223e+00,
    2.038025e+00, 4.644014e+00, 9.301590e+00, 2.039273e+01, 4.896534e+01,
    1.252311e+02, 3.576766e+02, 2.579464e+04, 2.944679e+04, 2.834717e+03,
    5.931328e+04,
)
_CSE_HERN_W = (
    9.200445e-18, 2.184724e-16, 3.548079e-15, 2.823716e-14, 1.091876e-13,
    6.998697e-13, 3.142264e-12, 1.457280e-11, 4.472783e-11, 2.042079e-10,
    8.708137e-10, 2.423649e-09, 7.353440e-09, 5.470738e-08, 2.445878e-07,
    4.541672e-07, 3.227611e-06, 1.110690e-05, 3.725101e-05, 1.056271e-04,
    6.531501e-04, 2.121330e-03, 8.285518e-03, 4.084190e-02, 5.760942e-02,
    1.788945e-01, 2.092774e-01, 3.697750e-01, 3.440555e-01, 5.792737e-01,
    2.325935e-01, 5.227961e-01, 3.079968e-01, 1.633456e-01, 7.410900e-02,
    3.123329e-02, 1.292488e-02, 2.156527e+00, 1.652553e-02, 2.314934e-02,
    3.992313e-01,
)

_cse_table_cache: dict = {}


def _cse_tables_on(model: str, device: torch.device, dtype: torch.dtype):
    """Return (s, w) coefficient tensors for 'nfw' or 'hern' on device/dtype."""
    key = (model, device, dtype)
    if key not in _cse_table_cache:
        if model == "nfw":
            s_tup, w_tup = _CSE_NFW_S, _CSE_NFW_W
        else:
            s_tup, w_tup = _CSE_HERN_S, _CSE_HERN_W
        _cse_table_cache[key] = (
            torch.tensor(s_tup, device=device, dtype=dtype),
            torch.tensor(w_tup, device=device, dtype=dtype),
        )
    return _cse_table_cache[key]


def _shape_tab(tab: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast-shape a CSE coefficient table against x.

    1-D tables (fixed anfw/ahern coefficients, or an interpolated acnfw row
    for a scalar b) become (N, 1, ..., 1); higher-dim tables (per-candidate
    acnfw rows, already shaped (N, C, 1, ...)) pass through unchanged."""
    return tab.view(-1, *([1] * x.ndim)) if tab.ndim == 1 else tab


def _alpha_cse_sum(x: torch.Tensor, y: torch.Tensor, q,
                   s_tab: torch.Tensor, w_tab: torch.Tensor):
    """Weighted sum of alpha_cse_dl over the table (app_ell.c:344-357 with
    the loops of alpha_anfw_dl/alpha_ahern_dl, app_ell.c:230-248/290-308)."""
    s = _shape_tab(s_tab, x)
    w = _shape_tab(w_tab, x)
    psi = torch.sqrt(q * q * (s * s + x * x) + y * y)
    f = (psi + s) * (psi + s) + (1.0 - q * q) * x * x
    fac = q / (s * psi * f)
    ax = (w * (fac * x * (psi + q * q * s))).sum(dim=0)
    ay = (w * (fac * y * (psi + s))).sum(dim=0)
    return ax, ay


def _ddphi_cse_sum(x: torch.Tensor, y: torch.Tensor, q,
                   s_tab: torch.Tensor, w_tab: torch.Tensor):
    """Weighted sum of ddphi_cse_dl over the table (app_ell.c:326-342)."""
    s = _shape_tab(s_tab, x)
    w = _shape_tab(w_tab, x)
    psi = torch.sqrt(q * q * (s * s + x * x) + y * y)
    f = (psi + s) * (psi + s) + (1.0 - q * q) * x * x
    pi = 1.0 / psi
    fi = 1.0 / f
    qs = q / s
    pxx = (w * (qs * fi * (1.0 + pi * pi * pi * q * q * s
                           * (q * q * s * s + y * y)
                           - 2.0 * pi * pi * fi * x * x
                           * (psi + q * q * s) * (psi + q * q * s)))).sum(dim=0)
    pyy = (w * (qs * fi * (1.0 + pi * pi * pi * q * q * s * (s * s + x * x)
                           - 2.0 * pi * pi * fi * y * y
                           * (psi + s) * (psi + s)))).sum(dim=0)
    pxy = (w * ((-1.0) * qs * x * y * fi
                * (pi * pi * pi * q * q * s
                   + 2.0 * pi * pi * fi * (psi + q * q * s) * (psi + s)))).sum(dim=0)
    return pxx, pxy, pyy


def _phi_cse_sum(x: torch.Tensor, y: torch.Tensor, q,
                 s_tab: torch.Tensor, w_tab: torch.Tensor):
    """Weighted sum of phi_cse_dl over the table (app_ell.c:314-324)."""
    s = _shape_tab(s_tab, x)
    w = _shape_tab(w_tab, x)
    psi = torch.sqrt(q * q * (s * s + x * x) + y * y)
    aa = 0.5 * torch.log((psi + s) * (psi + s) + (1.0 - q * q) * x * x) \
        - torch.log((1.0 + q) * s)
    return (w * ((q / s) * aa)).sum(dim=0)


def _kapgam_cse_body(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                     x0, y0, e, pa, bb, tt, tables,
                     need_kg: bool, need_phi: bool):
    """Shared body of kapgam_anfw (mass.c:1146-1198), kapgam_ahern
    (mass.c:1677-1729) and kapgam_acnfw (mass.c:2945-3000).  ``tables`` is the
    (s, w) coefficient pair — fixed for anfw/ahern, b-interpolated for acnfw.
    NOTE: the CSE models use the (pa - 90) trig convention (mass.c:1166-1167 /
    1697-1698), unlike the Schramm density models, because the CSE kernel
    takes q directly with the major axis along the rotated y direction."""
    q = 1.0 - e
    si, co = pa_minus_90_trig(pa)
    s_tab, w_tab = tables

    dx = (tx - x0) / tt
    dy = (ty - y0) / tt
    ddx = co * dx - si * dy
    ddy = si * dx + co * dy

    aax, aay = _alpha_cse_sum(ddx, ddy, q, s_tab, w_tab)
    ax = bb * (aax * co + aay * si) * tt
    ay = bb * (aax * (-1.0) * si + aay * co) * tt

    if not need_kg:
        return ax, ay, None, None, None, None

    pxx, pxy, pyy = _ddphi_cse_sum(ddx, ddy, q, s_tab, w_tab)
    rpxx = co * co * pxx + 2.0 * co * si * pxy + si * si * pyy
    rpyy = si * si * pxx - 2.0 * co * si * pxy + co * co * pyy
    rpxy = si * co * (pyy - pxx) + (co * co - si * si) * pxy

    kap = 0.5 * bb * (rpxx + rpyy)
    gam1 = 0.5 * bb * (rpxx - rpyy)
    gam2 = bb * rpxy
    phi = None
    if need_phi:
        phi = bb * _phi_cse_sum(ddx, ddy, q, s_tab, w_tab) * tt * tt
    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# 3) anfw -- CSE-approximated NFW density (mass.c:1146-1198)
#    p[1]=M  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=c
# ---------------------------------------------------------------------------
def kapgam_anfw(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                p: tuple, smallcore: float = K.DEF_SMALLCORE,
                need_kg: bool = True, need_phi: bool = True,
                nfw_users: int = K.DEF_NFW_USERS):
    m = _pf(p[1]); x0 = _pf(p[2]); y0 = _pf(p[3])
    e = _pf(p[4]); pa = _pf(p[5]); c = _pf(p[6])
    if not _is_t(m, e, c):
        if m <= 0.0: raise ValueError("anfw: m must be positive")
        if not (0.0 <= e < 1.0): raise ValueError("anfw: e in [0,1)")
        if c <= 0.0: raise ValueError("anfw: c must be positive")

    q = 1.0 - e
    bb, tt = _calc_bbtt_nfw(m, c, ctx, nfw_users=nfw_users)
    tt = tt / _t_sqrt(q)
    tables = _cse_tables_on("nfw", tx.device, tx.dtype)
    return _kapgam_cse_body(ctx, tx, ty, x0, y0, e, pa, bb, tt, tables,
                            need_kg, need_phi)


# ---------------------------------------------------------------------------
# 4) ahern -- CSE-approximated Hernquist density (mass.c:1677-1729)
#    p[1]=M  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=rb
# ---------------------------------------------------------------------------
def kapgam_ahern(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                 p: tuple, smallcore: float = K.DEF_SMALLCORE,
                 need_kg: bool = True, need_phi: bool = True):
    m = _pf(p[1]); x0 = _pf(p[2]); y0 = _pf(p[3])
    e = _pf(p[4]); pa = _pf(p[5]); rb = _pf(p[6])
    if not _is_t(m, e, rb):
        if m <= 0.0: raise ValueError("ahern: m must be positive")
        if not (0.0 <= e < 1.0): raise ValueError("ahern: e in [0,1)")
        if rb <= 0.0: raise ValueError("ahern: rb must be positive")

    q = 1.0 - e
    bb = _b_func_hern(m, rb, ctx)
    tt = rb / _t_sqrt(q)
    tables = _cse_tables_on("hern", tx.device, tx.dtype)
    return _kapgam_cse_body(ctx, tx, ty, x0, y0, e, pa, bb, tt, tables,
                            need_kg, need_phi)


# ---------------------------------------------------------------------------
# 5) acnfw -- CSE-approximated CORED NFW density (mass.c:2945-3000,
#    app_cnfw.c).  p[1]=M  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=c  p[7]=b
#    (b = core radius in units of rs).
#
#    glafic tabulates 44 (s, w) CSE pairs on a 201-point log10(b) grid
#    (app_cnfw.c:8-12, b in [1e-8, 1e2] step 0.05 dex) and linearly
#    interpolates the CSE *sums* between the two bracketing rows
#    (calc_jp_acnfw app_cnfw.c:305-327; alpha/ddphi/phi_acnfw_dl
#    app_cnfw.c:229-303).  Interpolating the sums of rows j and j+1 with
#    weights (1-p) and p is identical to one 88-term CSE evaluation over the
#    concatenated coefficients — which is how we feed the shared kernel.
#    The table ships in _tab_cache/acnfw_cse_v1.npz (extracted verbatim from
#    app_cnfw.c's par_cse_cnfw).
# ---------------------------------------------------------------------------
_ACNFW_LB_MIN = -8.0     # CNFW_LB_MIN (app_cnfw.c:11)
_ACNFW_DLB = 0.05        # CNFW_LB_DLB (app_cnfw.c:13)
_ACNFW_NB = 201          # NUM_NB_CNFW (app_cnfw.c:10)
_acnfw_raw = None        # (S, W) numpy arrays, each (201, 44)
_acnfw_cache: dict = {}  # (device, dtype) -> (S, W) tensors


def _acnfw_tables_on(device: torch.device, dtype: torch.dtype):
    """The full (201, 44) acnfw CSE coefficient tables on device/dtype."""
    global _acnfw_raw
    key = (device, dtype)
    if key not in _acnfw_cache:
        if _acnfw_raw is None:
            import numpy as np
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "_tab_cache", "acnfw_cse_v1.npz")
            with np.load(path) as z:
                _acnfw_raw = (z["s"].copy(), z["w"].copy())
        s_np, w_np = _acnfw_raw
        _acnfw_cache[key] = (torch.tensor(s_np, device=device, dtype=dtype),
                            torch.tensor(w_np, device=device, dtype=dtype))
    return _acnfw_cache[key]


def _calc_jp_acnfw(b: float):
    """Row index j and interpolation weight p for log10(b)
    (app_cnfw.c:305-327, scalar port)."""
    if b <= 10.0 ** _ACNFW_LB_MIN:
        return 0, 0.0
    lb = math.log10(b)
    j = int((lb - _ACNFW_LB_MIN) / _ACNFW_DLB)
    if j >= _ACNFW_NB - 1:
        return _ACNFW_NB - 2, 1.0
    return j, (lb - (_ACNFW_LB_MIN + j * _ACNFW_DLB)) / _ACNFW_DLB


def _calc_jp_acnfw_tensor(b: torch.Tensor):
    """Tensor version of calc_jp_acnfw: elementwise (j, p) with the same
    b<=1e-8 and j>=NB-1 edge handling."""
    lb = torch.log10(torch.clamp(b, min=10.0 ** _ACNFW_LB_MIN))
    jf = (lb - _ACNFW_LB_MIN) / _ACNFW_DLB
    j = jf.floor().long()
    p = jf - j.to(b.dtype)
    over = j >= (_ACNFW_NB - 1)
    j = torch.where(over, torch.full_like(j, _ACNFW_NB - 2), j)
    p = torch.where(over, torch.ones_like(p), p)
    return j, p


def kapgam_acnfw(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                 p: tuple, smallcore: float = K.DEF_SMALLCORE,
                 need_kg: bool = True, need_phi: bool = True,
                 nfw_users: int = K.DEF_NFW_USERS):
    m = _pf(p[1]); x0 = _pf(p[2]); y0 = _pf(p[3])
    e = _pf(p[4]); pa = _pf(p[5]); c = _pf(p[6]); b = _pf(p[7])
    if not _is_t(m, e, c, b):
        if m <= 0.0: raise ValueError("acnfw: m must be positive")
        if not (0.0 <= e < 1.0): raise ValueError("acnfw: e in [0,1)")
        if c <= 0.0: raise ValueError("acnfw: c must be positive")
        # maximum b corresponds to CNFW_LB_MAX; glafic's checkmodelpar_max is
        # strict (util.c:21-27), so b must be < 100 (mass.c:2955-2957)
        if b < 0.0 or b >= 100.0: raise ValueError("acnfw: b in [0, 100)")

    q = 1.0 - e
    bb, tt = _calc_bbtt_nfw(m, c, ctx, nfw_users=nfw_users)
    tt = tt / _t_sqrt(q)

    s_full, w_full = _acnfw_tables_on(tx.device, tx.dtype)
    if torch.is_tensor(b):
        # batched b, shaped to broadcast against tx (e.g. (C,1,1) vs (C,ny,nx)):
        # gather the two bracketing rows per candidate.
        j, pw = _calc_jp_acnfw_tensor(b)
        j_flat = j.reshape(-1)
        pw_flat = pw.reshape(-1, 1)
        s_eff = torch.cat([s_full[j_flat], s_full[j_flat + 1]], dim=1)       # (C, 88)
        w_eff = torch.cat([(1.0 - pw_flat) * w_full[j_flat],
                           pw_flat * w_full[j_flat + 1]], dim=1)             # (C, 88)
        n_c = s_eff.shape[0]
        tail = [1] * (tx.ndim - 1)
        tables = (s_eff.permute(1, 0).reshape(-1, n_c, *tail),
                  w_eff.permute(1, 0).reshape(-1, n_c, *tail))
    else:
        j, pw = _calc_jp_acnfw(float(b))
        tables = (torch.cat([s_full[j], s_full[j + 1]]),
                  torch.cat([(1.0 - pw) * w_full[j], pw * w_full[j + 1]]))

    return _kapgam_cse_body(ctx, tx, ty, x0, y0, e, pa, bb, tt, tables,
                            need_kg, need_phi)


# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------
KERNELS: dict = {
    "tnfw":    kapgam_tnfw,
    "tnfwpot": kapgam_tnfwpot,
    "anfw":    kapgam_anfw,
    "ahern":   kapgam_ahern,
    "acnfw":   kapgam_acnfw,
}
