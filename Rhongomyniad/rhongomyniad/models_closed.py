"""
Closed-form / potential-form lens-model deflection kernels (v2 additions).

Same conventions as `lens_models.py`:

    kapgam_<name>(ctx, tx, ty, p, smallcore=K.DEF_SMALLCORE,
                  need_kg=True, need_phi=True) -> (ax, ay, kap, g1, g2, phi)

`tx`, `ty` are torch tensors of any shape; `p` is glafic's `para_lens[i]` row
(p[0]=zl unused here, p[1..7] model parameters).  Formulas are ported verbatim
from glafic's mass.c (branch thresholds, b_func normalizations, pa sign
conventions included).  See per-model comments for mass.c line references.

Models in this module:
    hern      Hernquist elliptical density   (Schramm integrals, mass.c:1607-1671)
    hernpot   Hernquist elliptical potential (u_calc form,       mass.c:1492-1534)
    pow       power-law elliptical density   (Tessore&Metcalf15, mass.c:1810-1999)
    powpot    power-law elliptical potential (u_calc form,       mass.c:1735-1779)
    serspot   Sersic elliptical potential    (u_calc form,       mass.c:2005-2051)
    clus3     cluster quadrupole+octupole    (closed form,       mass.c:431-479)
    mpole     multipole perturbation         (closed form,       mass.c:481-513)

Tensor-param support
--------------------
Every physical parameter p[i] may be a python float OR a torch tensor
broadcastable against tx/ty (e.g. shape (C,1,1) against tx of shape
(C,ny,nx)).  Float parameters follow the exact scalar code path; tensor
parameters replace python-level value branches with torch.where (both
branches guarded so the unused branch cannot produce NaN/Inf), and skip
the raise-on-invalid validation (callers pre-filter with range penalties).

EXCEPTION (must remain a scalar python float because it feeds the scipy
distance integral in `fac_pert` / `Cosmology.angulard`):
    * zs_fid = p[1] of "pow", "powpot", "clus3" and "mpole".
All other slots (masses, centers, e, pa, scale radii, gamma, n, g, tg,
multipole order m) may be tensors.

Deviations from glafic (documented):
    * "pow" replicates glafic's DEFAULT path: flag_pow_tm15 = 1
      (glafic.h:198 DEF_FLAG_POW_TM15) with tol_pow_tm15 = 1.0e-8
      (glafic.h:200 DEF_TOL_POW_TM15), i.e. the Tessore & Metcalf (2015)
      angular series.  The non-default direct Schramm integration
      (flag_pow_tm15 = 0) is NOT ported.
    * serspot's phi: glafic integrates dphi_sers_dl over [1e-12, x] with a
      fixed 21-point Gauss-Kronrod rule (gsl_qgaus, TOL_QGAUS=1e100 accepts
      the first rule), which is only ~1e-8 accurate for typical arguments.
      We evaluate the same integral with a log-substituted 256-node
      Gauss-Legendre rule (accurate to ~1e-12), so phi/td can differ from
      glafic by glafic's own quadrature error (~1e-8 relative).
"""

from __future__ import annotations

import math
import torch

from . import constants as K
from . import cosmology as cos_mod
from .elliptical import (
    ell_integ_i, ell_integ_j, ell_integ_k,
    ell_pxpy, ell_pxxpyy, u_calc_tensor, gl_nodes_on,
)
from .lens_models import (
    LensContext,
    _pf, _is_t, _t_sqrt, _q_tensor,
    pa_trig, pa_minus_90_trig,
    fac_pert,
    _func_hern_dl,
    _bnn_sers, _b_func_sers, _make_sers_kernels,
)

# glafic defaults for the power-law density model (glafic.h:198/200,
# init.c:647-648).  flag_pow_tm15 = 1 selects the TM15 series.
DEF_FLAG_POW_TM15 = 1
DEF_TOL_POW_TM15 = 1.0e-8
_POW_TM15_MAX_ITER = 100000     # safety cap; glafic's loop is unbounded


# ---------------------------------------------------------------------------
# Hernquist radial profiles (mass.c:1544-1601).  All closed forms built on
# func_hern_dl (= lens_models._func_hern_dl).  Tensor in, tensor out, with the
# same branch thresholds as the C code.
# ---------------------------------------------------------------------------
def _kappa_hern_dl(x: torch.Tensor) -> torch.Tensor:
    """kappa for dimensionless Hernquist (mass.c:1549-1556)."""
    outer = (x > (1.0 + 1.0e-5)) | (x < (1.0 - 1.0e-5))
    d = x * x - 1.0
    d2_safe = torch.where(outer, d * d, torch.ones_like(x))
    val = ((2.0 + x * x) * _func_hern_dl(x) - 3.0) / d2_safe
    return torch.where(outer, val, torch.full_like(x, 4.0 / 15.0))


def _dkappa_hern_dl(x: torch.Tensor) -> torch.Tensor:
    """d kappa / d x for Hernquist (mass.c:1558-1565).

    Note glafic's near-1 value is the literal -0.4571429 (not -16/35).
    """
    outer = (x > (1.0 + 1.0e-4)) | (x < (1.0 - 1.0e-4))
    d = x * x - 1.0
    den = x * d * d * d
    den_safe = torch.where(outer, den, torch.ones_like(x))
    val = (2.0 + 13.0 * x * x
           - 3.0 * x * x * (x * x + 4.0) * _func_hern_dl(x)) / den_safe
    return torch.where(outer, val, torch.full_like(x, -0.4571429))


def _dphi_hern_dl(x: torch.Tensor) -> torch.Tensor:
    """dphi/dx for Hernquist (mass.c:1567-1574)."""
    outer = (x > (1.0 + 1.0e-7)) | (x < (1.0 - 1.0e-7))
    d = x * x - 1.0
    d_safe = torch.where(outer, d, torch.ones_like(x))
    val = 2.0 * x * (1.0 - _func_hern_dl(x)) / d_safe
    return torch.where(outer, val, torch.full_like(x, 2.0 / 3.0))


def _ddphi_hern_dl(x: torch.Tensor) -> torch.Tensor:
    """ddphi/(dx^2) = 2*kappa - dphi/x  (mass.c:1544-1547, 963-966)."""
    return 2.0 * _kappa_hern_dl(x) - _dphi_hern_dl(x) / torch.clamp(x, min=1e-300)


def _phi_hern_dl(x: torch.Tensor) -> torch.Tensor:
    """phi(x) for Hernquist (mass.c:1576-1583); switch at x = 1e-3."""
    big = x > 1.0e-3
    xs = torch.clamp(x, min=1e-300)
    val_big = torch.log(xs * xs / 4.0) + 2.0 * _func_hern_dl(xs)
    val_sml = x * x * (torch.log(2.0 / xs) - 0.5)
    return torch.where(big, val_big, val_sml)


def _b_func_hern(m, rb, ctx: LensContext):
    """Hernquist normalization bb (mass.c:1536-1542).  m, rb may be tensors."""
    rr = cos_mod.thetator_dis(rb, ctx.dis_ol)
    return (m * ctx.inv_sigma_crit / (rr * rr)) / (2.0 * math.pi)


# ---------------------------------------------------------------------------
# 1) "hern" — Hernquist elliptical DENSITY (mass.c:1607-1671, Schramm pattern)
#    p[1]=M  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=rb
# ---------------------------------------------------------------------------
def kapgam_hern(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                p: tuple, smallcore: float = K.DEF_SMALLCORE,
                need_kg: bool = True, need_phi: bool = True):
    m = _pf(p[1]); x0 = _pf(p[2]); y0 = _pf(p[3])
    e = _pf(p[4]); pa = _pf(p[5]); rb = _pf(p[6])
    if not _is_t(m, e, rb):
        if m < 0.0: raise ValueError("hern: m >= 0")
        if not (0.0 <= e < 1.0): raise ValueError("hern: e in [0,1)")
        if rb <= 0.0: raise ValueError("hern: rb > 0")

    q = 1.0 - e
    bb = _b_func_hern(m, rb, ctx)
    tt = rb / _t_sqrt(q)
    si, co = pa_trig(pa)

    bx = (co * (tx - x0) - si * (ty - y0)) / tt
    by = (si * (tx - x0) + co * (ty - y0)) / tt
    q_t = _q_tensor(q, tx)

    j1 = ell_integ_j(_kappa_hern_dl, 1, bx, by, q_t, smallcore)
    j0 = ell_integ_j(_kappa_hern_dl, 0, bx, by, q_t, smallcore)
    bpx = q * bx * j1
    bpy = q * by * j0
    px, py = ell_pxpy(bpx, bpy, si, co)
    ax = bb * tt * px
    ay = bb * tt * py

    if not need_kg:
        return ax, ay, None, None, None, None

    k2 = ell_integ_k(_dkappa_hern_dl, 2, bx, by, q_t, smallcore)
    k0 = ell_integ_k(_dkappa_hern_dl, 0, bx, by, q_t, smallcore)
    k1 = ell_integ_k(_dkappa_hern_dl, 1, bx, by, q_t, smallcore)
    bpxx = 2.0 * q * bx * bx * k2 + q * j1
    bpyy = 2.0 * q * by * by * k0 + q * j0
    bpxy = 2.0 * q * bx * by * k1
    pxx, pyy, pxy = ell_pxxpyy(bpxx, bpyy, bpxy, si, co)

    kap = 0.5 * bb * (pxx + pyy)
    gam1 = 0.5 * bb * (pxx - pyy)
    gam2 = bb * pxy
    phi = None
    if need_phi:
        phi_int = ell_integ_i(_dphi_hern_dl, bx, by, q_t, smallcore)
        phi = 0.5 * q * bb * phi_int * tt * tt
    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# 2) "hernpot" — Hernquist elliptical POTENTIAL (mass.c:1492-1534)
#    p[1]=M  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=rb
# ---------------------------------------------------------------------------
def kapgam_hernpot(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                   p: tuple, smallcore: float = K.DEF_SMALLCORE,
                   need_kg: bool = True, need_phi: bool = True):
    m = _pf(p[1]); x0 = _pf(p[2]); y0 = _pf(p[3])
    e = _pf(p[4]); pa = _pf(p[5]); rb = _pf(p[6])
    if not _is_t(m, e, rb):
        if m < 0.0: raise ValueError("hernpot: m >= 0")
        if not (0.0 <= e < 1.0): raise ValueError("hernpot: e in [0,1)")
        if rb <= 0.0: raise ValueError("hernpot: rb > 0")

    bb = _b_func_hern(m, rb, ctx)
    tt = rb
    si, co = pa_trig(pa)

    u0, u_x, u_y, u_xx, u_xy, u_yy = u_calc_tensor(
        (tx - x0) / tt, (ty - y0) / tt, e, si, co, smallcore)

    a = bb * _dphi_hern_dl(u0)
    ax = a * u_x * tt
    ay = a * u_y * tt

    if not need_kg:
        return ax, ay, None, None, None, None
    b = bb * _ddphi_hern_dl(u0)
    pxx = b * u_x * u_x + a * u_xx
    pxy = b * u_x * u_y + a * u_xy
    pyy = b * u_y * u_y + a * u_yy
    kap = 0.5 * (pxx + pyy)
    gam1 = 0.5 * (pxx - pyy)
    gam2 = pxy
    phi = None
    if need_phi:
        phi = bb * _phi_hern_dl(u0) * tt * tt
    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# 3) "pow" — power-law elliptical DENSITY via Tessore & Metcalf 2015
#    (mass.c:1810-1818 dispatch, 1900-1999 implementation;
#     defaults flag_pow_tm15=1, tol_pow_tm15=1e-8 — glafic.h:198,200)
#    p[1]=zs_fid  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=r_ein  p[7]=gamma
# ---------------------------------------------------------------------------
def _pow_tm15_omega(psi: torch.Tensor, q, gam,
                    tol: float = DEF_TOL_POW_TM15):
    """Angular series omega(psi) of TM15 (mass.c:1961-1999).

    Replicates glafic's per-point termination exactly: each point stops
    accumulating at the first k where both |a0| and |a1| <= tol (terms are
    masked out once a point has converged, so batched results are identical
    to the scalar code).  The geometric ratio is f = (1-q)/(1+q) < 1, so the
    loop terminates; a safety cap of 100000 iterations guards q -> 0.
    """
    t = 3.0 - gam
    f = (1.0 - q) / (1.0 + q)

    c1 = torch.cos(psi)
    s1 = torch.sin(psi)
    c2 = c1 * c1 - s1 * s1
    s2 = 2.0 * c1 * s1

    # Broadcast the running term to the full output shape so the per-point
    # convergence mask is well-defined even for tensor q/gam.
    shape = torch.broadcast_shapes(
        psi.shape,
        f.shape if torch.is_tensor(f) else (),
        t.shape if torch.is_tensor(t) else ())
    a0 = torch.broadcast_to(c1, shape).clone()
    a1 = torch.broadcast_to(s1, shape).clone()
    ome0 = a0.clone()
    ome1 = a1.clone()
    active = torch.ones(shape, dtype=torch.bool, device=psi.device)

    k = 0
    while bool(active.any()):
        k += 1
        if k > _POW_TM15_MAX_ITER:
            break
        fac = (-1.0) * f * (2.0 * k - t) / (2.0 * k + t)
        b0 = fac * (c2 * a0 - s2 * a1)
        b1 = fac * (s2 * a0 + c2 * a1)
        a0, a1 = b0, b1
        ome0 = torch.where(active, ome0 + a0, ome0)
        ome1 = torch.where(active, ome1 + a1, ome1)
        active = active & ((a0.abs() > tol) | (a1.abs() > tol))
    return ome0, ome1


def kapgam_pow(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
               p: tuple, smallcore: float = K.DEF_SMALLCORE,
               need_kg: bool = True, need_phi: bool = True,
               tol_pow_tm15: float = DEF_TOL_POW_TM15):
    # zs_fid (p[1]) must remain a scalar float (distance integral).
    zs_fid = float(p[1]); x0 = _pf(p[2]); y0 = _pf(p[3])
    e = _pf(p[4]); pa = _pf(p[5])
    re = _pf(p[6]); gam = _pf(p[7])
    if not _is_t(e, re, gam):
        if not (0.0 <= e < 1.0): raise ValueError("pow: e in [0,1)")
        if re <= 0.0: raise ValueError("pow: re > 0")
        if not (1.0 < gam < 3.0): raise ValueError("pow: gamma in (1,3)")

    q = 1.0 - e
    fac = fac_pert(ctx, zs_fid)
    tt = re * _t_sqrt(q)                       # NOTE: re * sqrt(q) for TM15
    dx = tx - x0
    dy = ty - y0

    si, co = pa_minus_90_trig(pa)              # TM15 uses -(pa-90) trig

    ddx = co * dx - si * dy
    ddy = si * dx + co * dy

    r = torch.sqrt(q * q * ddx * ddx + ddy * ddy) + smallcore
    psi = torch.atan2(ddy, q * ddx)
    aa = 2.0 * tt * torch.pow(tt / r, gam - 2.0) / (1.0 + q)
    ome0, ome1 = _pow_tm15_omega(psi, q, gam, tol=tol_pow_tm15)

    aax = aa * ome0
    aay = aa * ome1

    ax = fac * (aax * co + aay * si)
    ay = fac * (aax * (-1.0) * si + aay * co)

    if not need_kg:
        return ax, ay, None, None, None, None

    kap = fac * 0.5 * (3.0 - gam) * torch.pow(r / tt, 1.0 - gam)
    r2 = torch.sqrt(ddx * ddx + ddy * ddy) + smallcore
    c1 = ddx / r2
    s1 = ddy / r2
    c2 = c1 * c1 - s1 * s1
    s2 = 2.0 * c1 * s1
    g1 = (-1.0) * c2 * kap + fac * (2.0 - gam) * (c1 * aax - s1 * aay) / r2
    g2 = (-1.0) * s2 * kap + fac * (2.0 - gam) * (c1 * aay + s1 * aax) / r2
    c2r = co * co - si * si
    s2r = 2.0 * co * si
    gam1 = c2r * g1 + s2r * g2
    gam2 = (-1.0) * s2r * g1 + c2r * g2
    phi = None
    if need_phi:
        phi = fac * (ddx * aax + ddy * aay) / (3.0 - gam)
    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# 4) "powpot" — power-law elliptical POTENTIAL (mass.c:1735-1779)
#    p[1]=zs_fid  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=r_ein  p[7]=gamma
#    Radial closed forms: mass.c:1781-1804.
# ---------------------------------------------------------------------------
def kapgam_powpot(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                  p: tuple, smallcore: float = K.DEF_SMALLCORE,
                  need_kg: bool = True, need_phi: bool = True):
    # zs_fid (p[1]) must remain a scalar float (distance integral).
    zs_fid = float(p[1]); x0 = _pf(p[2]); y0 = _pf(p[3])
    e = _pf(p[4]); pa = _pf(p[5])
    re = _pf(p[6]); gam = _pf(p[7])
    if not _is_t(e, re, gam):
        if not (0.0 <= e < 1.0): raise ValueError("powpot: e in [0,1)")
        if re <= 0.0: raise ValueError("powpot: re > 0")
        if not (1.0 < gam < 3.0): raise ValueError("powpot: gamma in (1,3)")

    fac = fac_pert(ctx, zs_fid)
    si, co = pa_trig(pa)

    u0, u_x, u_y, u_xx, u_xy, u_yy = u_calc_tensor(
        (tx - x0) / re, (ty - y0) / re, e, si, co, smallcore)

    a = fac * torch.pow(u0, 2.0 - gam)                 # dphi_pow_dl
    ax = a * u_x * re
    ay = a * u_y * re

    if not need_kg:
        return ax, ay, None, None, None, None
    b = fac * (2.0 - gam) * torch.pow(u0, 1.0 - gam)   # ddphi_pow_dl
    pxx = b * u_x * u_x + a * u_xx
    pxy = b * u_x * u_y + a * u_xy
    pyy = b * u_y * u_y + a * u_yy
    kap = 0.5 * (pxx + pyy)
    gam1 = 0.5 * (pxx - pyy)
    gam2 = pxy
    phi = None
    if need_phi:
        phi = fac * torch.pow(u0, 3.0 - gam) / (3.0 - gam) * re * re
    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# 5) "serspot" — Sersic elliptical POTENTIAL (mass.c:2005-2051)
#    p[1]=M_total  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=r_e  p[7]=n
#    Radial forms: mass.c:2113-2153 — dphi uses the regularized lower
#    incomplete gamma (torch.special.gammainc == gsl_sf_gamma_inc_P);
#    phi = int_{1e-12}^{x} dphi(t) dt (see module docstring for the
#    quadrature deviation vs glafic's 21-point gsl_qgaus).
# ---------------------------------------------------------------------------
def _phi_sers_dl(dphi_fn, x: torch.Tensor) -> torch.Tensor:
    """phi_sers_dl(x) = int_{1e-12}^{x} dphi(t) dt  (mass.c:2145-2153).

    Evaluated with the shared 256-node Gauss-Legendre rule on log t
    (t = exp(l), dt = t dl), which resolves the 1/t tail of dphi for
    arbitrarily large x to ~1e-12 relative accuracy.
    """
    nodes01, weights01 = gl_nodes_on(x.device, x.dtype)
    lmin = math.log(1.0e-12)
    lmax = torch.log(torch.clamp(x, min=1.0e-12))
    span = lmax - lmin                                       # shape (...)
    l_nodes = lmin + span.unsqueeze(0) * nodes01.view(-1, *([1] * x.ndim))
    t = torch.exp(l_nodes)
    f = dphi_fn(t) * t                                       # dt = t dl
    return (weights01.view(-1, *([1] * x.ndim)) * f).sum(dim=0) * span


def kapgam_serspot(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                   p: tuple, smallcore: float = K.DEF_SMALLCORE,
                   need_kg: bool = True, need_phi: bool = True):
    m = _pf(p[1]); x0 = _pf(p[2]); y0 = _pf(p[3])
    e = _pf(p[4]); pa = _pf(p[5])
    re = _pf(p[6]); n = _pf(p[7])
    if not _is_t(m, e, re, n):
        if m < 0.0: raise ValueError("serspot: m >= 0")
        if not (0.0 <= e < 1.0): raise ValueError("serspot: e in [0,1)")
        if re <= 0.0: raise ValueError("serspot: re > 0")
        if not (0.06 <= n <= 20.0): raise ValueError(
            f"serspot: n in [0.06, 20.0], got {n}")

    tt = re * _bnn_sers(n)
    bb = _b_func_sers(m, tt, n, ctx)
    si, co = pa_trig(pa)

    u0, u_x, u_y, u_xx, u_xy, u_yy = u_calc_tensor(
        (tx - x0) / tt, (ty - y0) / tt, e, si, co, smallcore)

    kappa_fn, _dkappa_fn, dphi_fn = _make_sers_kernels(n)

    a = bb * dphi_fn(u0)
    ax = a * u_x * tt
    ay = a * u_y * tt

    if not need_kg:
        return ax, ay, None, None, None, None
    # ddphi_sers_dl = 2*kappa - dphi/x  (mass.c:2113-2116, 963-966)
    b = bb * (2.0 * kappa_fn(u0) - dphi_fn(u0) / torch.clamp(u0, min=1e-300))
    pxx = b * u_x * u_x + a * u_xx
    pxy = b * u_x * u_y + a * u_xy
    pyy = b * u_y * u_y + a * u_yy
    kap = 0.5 * (pxx + pyy)
    gam1 = 0.5 * (pxx - pyy)
    gam2 = pxy
    phi = None
    if need_phi:
        phi = bb * _phi_sers_dl(dphi_fn, u0) * tt * tt
    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# 6) "clus3" — cluster quadrupole + octupole perturbation (mass.c:431-479)
#    p[1]=zs_fid  p[2..3]=center  p[4]=g  p[5]=theta_g
# ---------------------------------------------------------------------------
def kapgam_clus3(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                 p: tuple, smallcore: float = K.DEF_SMALLCORE,
                 need_kg: bool = True, need_phi: bool = True):
    # zs_fid (p[1]) must remain a scalar float (distance integral).
    zs_fid = float(p[1]); x0 = _pf(p[2]); y0 = _pf(p[3])
    g = _pf(p[4]); tg = _pf(p[5])

    fac = fac_pert(ctx, zs_fid)

    dx = tx - x0
    dy = ty - y0
    r = torch.sqrt(dx * dx + dy * dy) + smallcore  # avoid error at the center
    cox = dx / r
    six = dy / r
    cox3 = cox * (cox * cox - 3.0 * six * six)
    six3 = (3.0 * cox * cox - six * six) * six

    arg = tg * math.pi / 180.0
    if torch.is_tensor(arg):
        cog = torch.cos(arg); sig = torch.sin(arg)
    else:
        cog = math.cos(arg); sig = math.sin(arg)
    cog3 = cog * (cog * cog - 3.0 * sig * sig)
    sig3 = (3.0 * cog * cog - sig * sig) * sig

    co = cox * cog + six * sig
    si = six * cog - cox * sig
    co3 = cox3 * cog3 + six3 * sig3
    si3 = six3 * cog3 - cox3 * sig3

    ax = fac * (g / 4.0) * (3.0 * r * dx * (si + si3) - r * dy * (co + 3.0 * co3))
    ay = fac * (g / 4.0) * (3.0 * r * dy * (si + si3) + r * dx * (co + 3.0 * co3))

    if not need_kg:
        return ax, ay, None, None, None, None

    pxx = fac * (g / 4.0) * (3.0 * (r + dx * dx / r) * (si + si3)
                             - 4.0 * (dx * dy / r) * (co + 3.0 * co3)
                             - (dy * dy / r) * (si + 9.0 * si3))
    pyy = fac * (g / 4.0) * (3.0 * (r + dy * dy / r) * (si + si3)
                             + 4.0 * (dx * dy / r) * (co + 3.0 * co3)
                             - (dx * dx / r) * (si + 9.0 * si3))
    pxy = fac * (g / 4.0) * (3.0 * (dx * dy / r) * (si + si3)
                             + 2.0 * ((dx * dx - dy * dy) / r) * (co + 3.0 * co3)
                             + (dx * dy / r) * (si + 9.0 * si3))

    kap = 0.5 * (pxx + pyy)
    gam1 = 0.5 * (pxx - pyy)
    gam2 = pxy
    phi = None
    if need_phi:
        phi = fac * (g / 4.0) * r * r * r * (si + si3)
    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# 7) "mpole" — multipole perturbation (mass.c:481-513)
#    p[1]=zs_fid  p[2..3]=center  p[4]=g  p[5]=theta_g  p[6]=m (order)
#    p[7]=n (radial exponent)
# ---------------------------------------------------------------------------
def kapgam_mpole(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                 p: tuple, smallcore: float = K.DEF_SMALLCORE,
                 need_kg: bool = True, need_phi: bool = True):
    # zs_fid (p[1]) must remain a scalar float (distance integral).
    zs_fid = float(p[1]); x0 = _pf(p[2]); y0 = _pf(p[3])
    g = _pf(p[4]); tg = _pf(p[5])
    mm = _pf(p[6]); n = _pf(p[7])
    if not _is_t(mm):
        if mm <= 0.0: raise ValueError("mpole: m > 0")

    fac = fac_pert(ctx, zs_fid)

    dx = tx - x0
    dy = ty - y0
    r = torch.sqrt(dx * dx + dy * dy) + smallcore  # avoid error at the center
    # theta in DEGREES, with glafic's smallcore offset inside atan2.
    t = torch.atan2(dy, dx + smallcore) * 180.0 / math.pi

    co = torch.cos(mm * (t - tg - 90.0) * math.pi / 180.0)
    si = torch.sin(mm * (t - tg - 90.0) * math.pi / 180.0)

    f2 = (-1.0) * g * torch.pow(r, n - 2.0) / mm

    ax = fac * f2 * (n * dx * co + mm * dy * si)
    ay = fac * f2 * (n * dy * co - mm * dx * si)

    if not need_kg:
        return ax, ay, None, None, None, None

    pxx = fac * f2 * ((n - 2.0) * dx * (n * dx * co + mm * dy * si)
                      + n * r * r * co + n * mm * dx * dy * si
                      - mm * mm * dy * dy * co) / (r * r)
    pyy = fac * f2 * ((n - 2.0) * dy * (n * dy * co - mm * dx * si)
                      + n * r * r * co - n * mm * dx * dy * si
                      - mm * mm * dx * dx * co) / (r * r)
    pxy = fac * f2 * ((n * (n - 2.0) + mm * mm) * dx * dy * co
                      + mm * (n - 1.0) * (dy * dy - dx * dx) * si) / (r * r)

    kap = 0.5 * (pxx + pyy)
    gam1 = 0.5 * (pxx - pyy)
    gam2 = pxy
    phi = None
    if need_phi:
        phi = fac * f2 * r * r * co
    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------
KERNELS: dict[str, callable] = {
    "hern":    kapgam_hern,
    "hernpot": kapgam_hernpot,
    "pow":     kapgam_pow,
    "powpot":  kapgam_powpot,
    "serspot": kapgam_serspot,
    "clus3":   kapgam_clus3,
    "mpole":   kapgam_mpole,
}
