"""
Lens-model deflection kernels, batched on GPU via PyTorch.

Each model exposes `kapgam_<name>(ctx, tx, ty, params) -> (ax, ay, kap, g1, g2, phi)`
where `tx`, `ty` are torch tensors of any shape (the result tensors have the
same shape) and `ctx` is a `LensContext` holding pre-computed cosmological
distances for the (zl, zs) pair.

Formulas taken verbatim from glafic's mass.c.  Parameter layouts match
`para_lens[i][0..7]` in glafic.  See comments for each model.

Supported in this module (v1):
    point    sie     pert    nfwpot    nfw    king    jaffe    gaupot
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch

from . import constants as K
from . import cosmology as cos_mod
from .cosmology import Cosmology
from .elliptical import (
    ell_integ_i, ell_integ_j, ell_integ_k,
    ell_pxpy, ell_pxxpyy, u_calc_tensor,
)


# ---------------------------------------------------------------------------
# Context holding pre-computed distances (single lens plane, for now).
# ---------------------------------------------------------------------------
@dataclass
class LensContext:
    cosmo: Cosmology
    zl: float                    # lens redshift
    zs: float                    # source redshift
    dis_ol: float = 0.0
    dis_os: float = 0.0
    dis_ls: float = 0.0
    tdelay_fac: float = 0.0      # days per arcsec^2
    delome: float = 0.0          # overdensity * Omega_m(0)*(1+zl)^3
    sigma_crit: float = 0.0      # h M_sun Mpc^-2
    inv_sigma_crit: float = 0.0

    @classmethod
    def build(cls, cosmo: Cosmology, zl: float, zs: float,
              flag_hodensity: int = K.DEF_FLAG_HODENSITY,
              hodensity: float = K.DEF_HODENSITY) -> "LensContext":
        dis_ol = cosmo.angulard(0.0, zl)
        dis_os = cosmo.angulard(0.0, zs)
        dis_ls = cosmo.angulard(zl, zs)
        tf = cos_mod.tdelay_fac(zl, dis_os, dis_ol, dis_ls, cosmo.hubble)
        delome = cos_mod.deltaomega(cosmo, zl, flag_hodensity, hodensity)
        s_crit = cos_mod.sigma_crit_dis(dis_os, dis_ol, dis_ls)
        inv_s = cos_mod.inv_sigma_crit_dis(dis_os, dis_ol, dis_ls)
        return cls(cosmo=cosmo, zl=zl, zs=zs, dis_ol=dis_ol, dis_os=dis_os,
                   dis_ls=dis_ls, tdelay_fac=tf, delome=delome,
                   sigma_crit=s_crit, inv_sigma_crit=inv_s)


# ---------------------------------------------------------------------------
# Utility: rotation trig consistent with glafic.
#
# Two different sign conventions are used in glafic depending on the model.
# `pa_minus_90_trig` is used for *elliptical potential* models (sie/jaffe/
# gaupot/...) where the glafic code sets
#       si = sin(-(pa - 90) * pi/180)
#       co = cos(-(pa - 90) * pi/180)
# and `pa_trig` is used for *elliptical density* models (nfw/king/...) where
#       si = sin(-pa * pi/180)
#       co = cos(-pa * pi/180)
# We preserve both so numerical behaviour matches.
# ---------------------------------------------------------------------------
def pa_minus_90_trig(pa_deg: float) -> tuple[float, float]:
    arg = -(pa_deg - 90.0) * math.pi / 180.0
    return math.sin(arg), math.cos(arg)


def pa_trig(pa_deg: float) -> tuple[float, float]:
    arg = -pa_deg * math.pi / 180.0
    return math.sin(arg), math.cos(arg)


# ---------------------------------------------------------------------------
# 1) Point mass  (mass.c:634-673)  — parameter layout p[1]=M
# ---------------------------------------------------------------------------
def re2_point(m: float, ctx: LensContext) -> float:
    """Einstein radius squared [arcsec^2] for a point mass M [M_sun]."""
    d = ctx.dis_ls / (K.COVERH_MPCH * ctx.dis_ol * ctx.dis_os)
    return (2.0 * (K.R_SCHWARZ * m / K.MPC2METER) * d) / (K.ARCSEC2RADIAN ** 2)


def kapgam_point(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                 p: tuple, smallcore: float = K.DEF_SMALLCORE,
                 need_kg: bool = True, need_phi: bool = True):
    """
    p = (z, M, x0, y0, ...) — p[1]=M, p[2..3]=center
    """
    m = float(p[1]); x0 = float(p[2]); y0 = float(p[3])
    if m < 0.0:
        raise ValueError("point mass must be non-negative")

    re2 = re2_point(m, ctx)
    dx = tx - x0
    dy = ty - y0
    r2 = dx * dx + dy * dy

    rr = re2 / (r2 + smallcore * smallcore)
    ax = rr * dx
    ay = rr * dy

    if not need_kg:
        return ax, ay, None, None, None, None

    kap = torch.zeros_like(tx)
    # At r=0 glafic returns a special-case gamma: re2/(2*smallcore^2).
    sc2 = smallcore * smallcore
    near_center = r2 < sc2
    gam1_reg = (re2 / (r2 * r2)) * (dy * dy - dx * dx)
    gam2_reg = (re2 / (r2 * r2)) * (-2.0 * dx * dy)
    gam1 = torch.where(near_center, torch.full_like(tx, re2 / (2.0 * sc2)), gam1_reg)
    gam2 = torch.where(near_center, torch.full_like(tx, re2 / (2.0 * sc2)), gam2_reg)

    phi = None
    if need_phi:
        # glafic uses 0.5 * re2 * log(r2).  For numerical stability use the
        # raw r2 (no smallcore floor) unless r2 == 0.
        r2_floor = torch.clamp(r2, min=sc2)
        phi = 0.5 * re2 * torch.log(r2_floor)

    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# 2) External shear + convergence (pert)  (mass.c:398-424)
#    p[1]=zs_fid  p[2..3]=center  p[4]=g  p[5]=tg  p[7]=k
# ---------------------------------------------------------------------------
def fac_pert(ctx: LensContext, zs_fid: float) -> float:
    """
    Distance-ratio renormalisation so the (k,g) are referenced to zs_fid.
    Matches mass.c:510-523.
    """
    if ctx.zl >= zs_fid:
        raise ValueError("pert: zs_fid must exceed lens redshift")
    d_fid = ctx.cosmo.angulard(ctx.zl, zs_fid) / ctx.cosmo.angulard(0.0, zs_fid)
    return (ctx.dis_ls / ctx.dis_os) / d_fid


def kapgam_pert(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                p: tuple, smallcore: float = K.DEF_SMALLCORE,
                need_kg: bool = True, need_phi: bool = True):
    zs_fid = float(p[1]); x0 = float(p[2]); y0 = float(p[3])
    g = float(p[4]); tg = float(p[5]); k = float(p[7])
    fac = fac_pert(ctx, zs_fid)

    co = math.cos(2.0 * (tg - 90.0) * math.pi / 180.0)
    si = math.sin(2.0 * (tg - 90.0) * math.pi / 180.0)

    dx = tx - x0
    dy = ty - y0
    ax = fac * (dx * k - dx * g * co - dy * g * si)
    ay = fac * (dy * k + dy * g * co - dx * g * si)

    if not need_kg:
        return ax, ay, None, None, None, None

    kap = torch.full_like(tx, k * fac)
    gam1 = torch.full_like(tx, -g * fac * co)
    gam2 = torch.full_like(tx, -g * fac * si)
    phi = None
    if need_phi:
        phi = 0.5 * (dx * dx + dy * dy) * kap \
              + 0.5 * (dx * dx - dy * dy) * gam1 \
              + dx * dy * gam2
    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# 3) SIE — Singular Isothermal Ellipsoid  (mass.c:740-888)
#    p[1]=sigma_v[km/s]  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=s_core
# ---------------------------------------------------------------------------
def facq_sie(q: float) -> float:
    """glafic's 1/sqrt(q) normalization (mass.c:872-888)."""
    return 1.0 / math.sqrt(q)


def b_sie(ctx: LensContext, sig_kms: float, q: float) -> float:
    ss = sig_kms / K.C_LIGHT_KMS
    return facq_sie(q) * (4.0 * math.pi * ss * ss * ctx.dis_ls / ctx.dis_os) / K.ARCSEC2RADIAN


def _alpha_sie_dl(x: torch.Tensor, y: torch.Tensor, s: float, q: float):
    """Dimensionless SIE deflection (rotated/body frame)."""
    if (1.0 - q) > 1.0e-5:
        sq = math.sqrt(1.0 - q * q)
        psi = torch.sqrt(q * q * (s * s + x * x) + y * y)
        ax = (q / sq) * torch.atan(sq * x / (psi + s))
        ay = (q / sq) * torch.atanh(sq * y / (psi + q * q * s))
    else:
        psi = torch.sqrt(s * s + x * x + y * y)
        ax = x / (psi + s)
        ay = y / (psi + s)
    return ax, ay


def _ddphi_sie_dl(x: torch.Tensor, y: torch.Tensor, s: float, q: float):
    """Hessian in the rotated frame (mass.c:830-842)."""
    psi = torch.sqrt(q * q * (s * s + x * x) + y * y)
    f = (1.0 + q * q) * s * s + 2.0 * psi * s + x * x + y * y
    pxx = (q / psi) * (q * q * s * s + y * y + s * psi) / f
    pyy = (q / psi) * (s * s + x * x + s * psi) / f
    pxy = (q / psi) * (-x * y) / f
    return pxx, pxy, pyy


def _phi_sie_dl(x: torch.Tensor, y: torch.Tensor, s: float, q: float,
                ax_body: torch.Tensor, ay_body: torch.Tensor):
    """SIE potential (mass.c:817-828)."""
    psi = torch.sqrt(q * q * (s * s + x * x) + y * y)
    aa = math.log((1.0 + q) * s) - torch.log(
        torch.sqrt((psi + s) * (psi + s) + (1.0 - q * q) * x * x))
    return x * ax_body + y * ay_body + q * s * aa


def kapgam_sie(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
               p: tuple, smallcore: float = K.DEF_SMALLCORE,
               need_kg: bool = True, need_phi: bool = True):
    sig = float(p[1]); x0 = float(p[2]); y0 = float(p[3])
    e = float(p[4]); pa = float(p[5]); s = float(p[6])

    if sig < 0.0: raise ValueError("sie sigma must be non-negative")
    if not (0.0 <= e < 1.0): raise ValueError("sie e must be in [0, 1)")

    q = 1.0 - e
    bb = b_sie(ctx, sig, q)
    if s < smallcore: s = smallcore

    si, co = pa_minus_90_trig(pa)

    dx = tx - x0; dy = ty - y0
    # Rotate into body frame.
    ddx = co * dx - si * dy
    ddy = si * dx + co * dy
    ss = s * facq_sie(q)

    aax, aay = _alpha_sie_dl(ddx, ddy, ss, q)
    # Back-rotate the deflection (glafic mass.c:811-812 uses +aay*si / -aax*si).
    ax = bb * (aax * co + aay * si)
    ay = bb * (-aax * si + aay * co)

    if not need_kg:
        return ax, ay, None, None, None, None

    pxx_b, pxy_b, pyy_b = _ddphi_sie_dl(ddx, ddy, ss, q)
    # Rotate the Hessian back (mass.c:788-790).
    rpxx = co * co * pxx_b + 2.0 * co * si * pxy_b + si * si * pyy_b
    rpyy = si * si * pxx_b - 2.0 * co * si * pxy_b + co * co * pyy_b
    rpxy = si * co * (pyy_b - pxx_b) + (co * co - si * si) * pxy_b

    kap = 0.5 * bb * (rpxx + rpyy)
    gam1 = 0.5 * bb * (rpxx - rpyy)
    gam2 = bb * rpxy

    phi = None
    if need_phi:
        phi_body = _phi_sie_dl(ddx, ddy, ss, q, aax, aay)
        phi = bb * phi_body
    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# 4) NFW "nfwpot" — elliptical POTENTIAL (mass.c:894-935)
#    p[1]=M  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=c
# ---------------------------------------------------------------------------
def _hnfw(c: float) -> float:
    if c < 1.0e-6:
        return 0.5 * c * c
    return math.log(1.0 + c) - c / (1.0 + c)


def _rs_nfw(m: float, c: float, delome: float) -> float:
    """Scale radius in Mpc/h.  (mass.c:1023-1037)"""
    return K.NFW_RS_NORM * (m / delome) ** (1.0 / 3.0) / c


def _b_func_nfw(m: float, c: float, ctx: LensContext) -> float:
    """NFW Einstein-radius normalization (mass.c:1039-1056)."""
    return (K.NFW_B_NORM * ctx.dis_ol * ctx.dis_ls
            * (ctx.delome * ctx.delome * m) ** (1.0 / 3.0)
            * (c * c / _hnfw(c)) / ctx.dis_os)


def _calc_bbtt_nfw(m: float, c: float, ctx: LensContext,
                   nfw_users: int = K.DEF_NFW_USERS) -> tuple[float, float]:
    if nfw_users == 0:
        bb = _b_func_nfw(m, c, ctx)
        tt = cos_mod.rtotheta_dis(_rs_nfw(m, c, ctx.delome), ctx.dis_ol)
    else:
        tt = c
        cc = _rs_nfw(m, 1.0, ctx.delome) / cos_mod.thetator_dis(c, ctx.dis_ol)
        bb = _b_func_nfw(m, cc, ctx)
    return bb, tt


def _kappa_nfw_dl(x: torch.Tensor) -> torch.Tensor:
    """kappa for dimensionless NFW (mass.c:963-972).  Handles all three branches."""
    one = torch.tensor(1.0, dtype=x.dtype, device=x.device)
    gt = x > (1.0 + 1.0e-6)
    lt = x < (1.0 - 1.0e-6)
    # Branch: x > 1
    d_gt = torch.clamp(x * x - 1.0, min=1e-300)
    t_gt = torch.sqrt(torch.clamp((x - 1.0) / (x + 1.0), min=0.0))
    val_gt = 0.5 * (1.0 - 2.0 * torch.atan(t_gt) / torch.sqrt(d_gt)) / d_gt
    # Branch: x < 1
    d_lt = torch.clamp(1.0 - x * x, min=1e-300)
    t_lt = torch.sqrt(torch.clamp((1.0 - x) / (x + 1.0), min=0.0))
    val_lt = 0.5 * (2.0 * torch.atanh(t_lt) / torch.sqrt(d_lt) - 1.0) / d_lt
    # Branch: x ~ 1
    val_mid = torch.full_like(x, 0.5 / 3.0)
    return torch.where(gt, val_gt, torch.where(lt, val_lt, val_mid))


def _func_hern_dl(x: torch.Tensor) -> torch.Tensor:
    """
    Helper shared by Hernquist/NFW derivatives (mass.c:1580-1596).
        x > 1:     atan(sqrt(x^2-1)) / sqrt(x^2-1)
        x < 1:     atanh(sqrt(1-x^2)) / sqrt(1-x^2)
        x ≈ 1:     1.0
        x < 1e-5:  log(2/x)
    """
    big = x > (1.0 + 1.0e-9)
    sml = x < (1.0 - 1.0e-9)
    tiny = x <= 1.0e-5
    xx_big = torch.sqrt(torch.clamp(x * x - 1.0, min=1e-300))
    val_big = torch.atan(xx_big) / xx_big
    xx_sml = torch.sqrt(torch.clamp(1.0 - x * x, min=1e-300))
    val_sml = torch.atanh(torch.clamp(xx_sml, max=1.0 - 1.0e-16)) / xx_sml
    val_tiny = torch.log(2.0 / torch.clamp(x, min=1e-300))
    val_mid = torch.ones_like(x)
    result = torch.where(big, val_big,
              torch.where(sml, torch.where(tiny, val_tiny, val_sml), val_mid))
    return result


def _dkappa_nfw_dl(x: torch.Tensor) -> torch.Tensor:
    """d kappa / d x for NFW (mass.c:974-981)."""
    near_one = (x < (1.0 + 1.0e-5)) & (x > (1.0 - 1.0e-5))
    f_val = _func_hern_dl(x)
    denom = x * (x * x - 1.0) * (x * x - 1.0)
    # Guard denom near x=1 (the near_one branch is replaced below).
    denom_safe = torch.where(near_one, torch.full_like(x, 1.0), denom)
    val = 0.5 * (3.0 * x * x * f_val - 2.0 * x * x - 1.0) / denom_safe
    return torch.where(near_one, torch.full_like(x, -0.2), val)


def _dphi_nfw_dl(x: torch.Tensor) -> torch.Tensor:
    """dphi/dx / x (mass.c:983-996)."""
    big = x > (1.0 + 1.0e-9)
    sml = x < (1.0 - 1.0e-9)
    tiny = x <= 1.0e-5
    # big branch
    d_big = torch.sqrt(torch.clamp(x * x - 1.0, min=1e-300))
    t_big = torch.sqrt(torch.clamp((x - 1.0) / (x + 1.0), min=0.0))
    val_big = 2.0 * torch.atan(t_big) / (x * d_big) + torch.log(0.5 * torch.clamp(x, min=1e-300)) / x
    # small branch (but x > 1e-5)
    d_sml = torch.sqrt(torch.clamp(1.0 - x * x, min=1e-300))
    t_sml = torch.sqrt(torch.clamp((1.0 - x) / (x + 1.0), min=0.0))
    val_sml = 2.0 * torch.atanh(t_sml) / (x * d_sml) + torch.log(0.5 * torch.clamp(x, min=1e-300)) / x
    # tiny branch
    val_tiny = 0.5 * x * torch.log(2.0 / torch.clamp(x, min=1e-300))
    # boundary (x ~ 1)
    val_mid = torch.full_like(x, 1.0 + math.log(0.5))
    result = torch.where(big, val_big,
              torch.where(sml, torch.where(tiny, val_tiny, val_sml), val_mid))
    return result


def _phi_nfw_dl(x: torch.Tensor) -> torch.Tensor:
    """NFW dimensionless potential phi(x) / M_norm (mass.c:998-1019)."""
    small_x = x <= 1.0e-6
    in_lt = (x > 1.0e-6) & (x < (1.0 - 1.0e-9))
    in_gt = x > (1.0 + 1.0e-9)
    in_mid = (x >= (1.0 - 1.0e-9)) & (x <= (1.0 + 1.0e-9))
    a = torch.log(0.5 * torch.clamp(x, min=1e-300))
    # arccosh(1/x) for x < 1
    y_lt = 1.0 / torch.clamp(x, min=1e-300)
    b_lt = torch.log(y_lt + torch.sqrt(torch.clamp(y_lt * y_lt - 1.0, min=0.0)))
    bb_lt = -b_lt * b_lt
    # arccos(1/x) for x > 1
    b_gt = torch.acos(torch.clamp(1.0 / torch.clamp(x, min=1e-300), max=1.0 - 1e-16))
    bb_gt = b_gt * b_gt
    main = 0.5 * (a * a + torch.where(in_gt, bb_gt, torch.where(in_lt, bb_lt, torch.zeros_like(x))))
    small = 0.25 * x * x * torch.log(2.0 / torch.clamp(x, min=1e-300))
    return torch.where(small_x, small, main)


def _ddphi_nfw_dl(x: torch.Tensor) -> torch.Tensor:
    """ddphi/(dx^2) (mass.c:953-960)."""
    return 2.0 * _kappa_nfw_dl(x) - _dphi_nfw_dl(x) / x


def kapgam_nfwpot(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                  p: tuple, smallcore: float = K.DEF_SMALLCORE,
                  need_kg: bool = True, need_phi: bool = True,
                  nfw_users: int = K.DEF_NFW_USERS):
    m = float(p[1]); x0 = float(p[2]); y0 = float(p[3])
    e = float(p[4]); pa = float(p[5]); c = float(p[6])
    if m <= 0.0: raise ValueError("nfwpot: m must be positive")
    if not (0.0 <= e < 1.0): raise ValueError("nfwpot: e in [0,1)")
    if c <= 0.0: raise ValueError("nfwpot: c must be positive")

    bb, tt = _calc_bbtt_nfw(m, c, ctx, nfw_users=nfw_users)
    si, co = pa_trig(pa)

    u0, u_x, u_y, u_xx, u_xy, u_yy = u_calc_tensor(
        (tx - x0) / tt, (ty - y0) / tt, e, si, co, smallcore)

    a = bb * _dphi_nfw_dl(u0)
    ax = a * u_x * tt
    ay = a * u_y * tt

    if not need_kg:
        return ax, ay, None, None, None, None
    b = bb * _ddphi_nfw_dl(u0)
    pxx = b * u_x * u_x + a * u_xx
    pxy = b * u_x * u_y + a * u_xy
    pyy = b * u_y * u_y + a * u_yy
    kap = 0.5 * (pxx + pyy)
    gam1 = 0.5 * (pxx - pyy)
    gam2 = pxy
    phi = None
    if need_phi:
        phi = bb * _phi_nfw_dl(u0) * tt * tt
    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# 5) NFW — elliptical DENSITY (mass.c:1071-1135)  — Schramm integrals.
#    p[1]=M  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=c
# ---------------------------------------------------------------------------
def kapgam_nfw(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
               p: tuple, smallcore: float = K.DEF_SMALLCORE,
               need_kg: bool = True, need_phi: bool = True,
               nfw_users: int = K.DEF_NFW_USERS):
    m = float(p[1]); x0 = float(p[2]); y0 = float(p[3])
    e = float(p[4]); pa = float(p[5]); c = float(p[6])
    if m <= 0.0: raise ValueError("nfw: m must be positive")
    if not (0.0 <= e < 1.0): raise ValueError("nfw: e in [0,1)")
    if c <= 0.0: raise ValueError("nfw: c must be positive")

    q = 1.0 - e
    bb, tt = _calc_bbtt_nfw(m, c, ctx, nfw_users=nfw_users)
    tt = tt / math.sqrt(q)
    si, co = pa_trig(pa)

    # Rotate into body frame (mass.c:1095-1096).
    bx = (co * (tx - x0) - si * (ty - y0)) / tt
    by = (si * (tx - x0) + co * (ty - y0)) / tt
    q_t = torch.tensor(q, dtype=tx.dtype, device=tx.device)

    j1 = ell_integ_j(_kappa_nfw_dl, 1, bx, by, q_t, smallcore)
    j0 = ell_integ_j(_kappa_nfw_dl, 0, bx, by, q_t, smallcore)
    bpx = q * bx * j1
    bpy = q * by * j0
    px, py = ell_pxpy(bpx, bpy, si, co)
    ax = bb * tt * px
    ay = bb * tt * py

    if not need_kg:
        return ax, ay, None, None, None, None

    k2 = ell_integ_k(_dkappa_nfw_dl, 2, bx, by, q_t, smallcore)
    k0 = ell_integ_k(_dkappa_nfw_dl, 0, bx, by, q_t, smallcore)
    k1 = ell_integ_k(_dkappa_nfw_dl, 1, bx, by, q_t, smallcore)
    bpxx = 2.0 * q * bx * bx * k2 + q * j1
    bpyy = 2.0 * q * by * by * k0 + q * j0
    bpxy = 2.0 * q * bx * by * k1
    pxx, pyy, pxy = ell_pxxpyy(bpxx, bpyy, bpxy, si, co)

    kap = 0.5 * bb * (pxx + pyy)
    gam1 = 0.5 * bb * (pxx - pyy)
    gam2 = bb * pxy
    phi = None
    if need_phi:
        phi_int = ell_integ_i(_dphi_nfw_dl, bx, by, q_t, smallcore)
        phi = 0.5 * q * bb * phi_int * tt * tt
    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# 6) King profile (mass.c:3105-3266)
#    p[1]=M  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=rc  p[7]=c  (c = log10(rt/rc))
# ---------------------------------------------------------------------------
def _king_helpers(c_param: float) -> tuple[float, float, float]:
    xt = 10.0 ** c_param
    st = math.sqrt(1.0 + xt * xt)
    f0 = 1.0 / st
    norm = math.log(st) - 1.5 + 2.0 * f0 - 0.5 * f0 * f0
    if norm < K.OFFSET_LOG:
        norm = K.OFFSET_LOG
    return xt, f0, norm


def _b_func_king(m: float, rc: float, ctx: LensContext) -> float:
    rr = cos_mod.thetator_dis(rc, ctx.dis_ol)
    return (m * ctx.inv_sigma_crit / (rr * rr)) / (2.0 * math.pi)


def _make_king_kernels(xt: float, f0: float, norm: float,
                       smallcore: float):
    def kappa(x: torch.Tensor) -> torch.Tensor:
        f = 1.0 / torch.sqrt(1.0 + x * x) - f0
        f_pos = torch.clamp(f, min=0.0)
        v = f_pos * f_pos / norm
        return torch.where(x >= xt, torch.zeros_like(x), v)

    def dkappa(x: torch.Tensor) -> torch.Tensor:
        s2 = 1.0 + x * x
        s3 = s2 * torch.sqrt(s2)
        f = 1.0 / torch.sqrt(s2) - f0
        f_pos = torch.clamp(f, min=0.0)
        v = (-2.0) * x * f_pos / (s3 * norm)
        return torch.where(x >= xt, torch.zeros_like(x), v)

    def dphi(x: torch.Tensor) -> torch.Tensor:
        # Tiny-x limit (Taylor).
        df0 = 1.0 - f0
        small_val = x * df0 * df0 / norm
        # Outside tidal radius -> point-mass limit 2/x.
        big_val = 2.0 / torch.clamp(x, min=1e-300)
        # Interior branch.
        sx = torch.sqrt(1.0 + x * x)
        I_king = (0.5 * torch.log(1.0 + x * x)
                  - 2.0 * f0 * sx
                  + 2.0 * f0
                  + 0.5 * f0 * f0 * x * x)
        mid_val = 2.0 * I_king / (norm * torch.clamp(x, min=1e-300))
        small = x <= smallcore
        big = x >= xt
        return torch.where(small, small_val,
                torch.where(big, big_val, mid_val))

    return kappa, dkappa, dphi


def kapgam_king(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                p: tuple, smallcore: float = K.DEF_SMALLCORE,
                need_kg: bool = True, need_phi: bool = True):
    m = float(p[1]); x0 = float(p[2]); y0 = float(p[3])
    e = float(p[4]); pa = float(p[5])
    rc = float(p[6]); c_param = float(p[7])
    if m < 0.0: raise ValueError("king: m >= 0")
    if not (0.0 <= e < 1.0): raise ValueError("king: e in [0,1)")
    if rc <= 0.0: raise ValueError("king: rc > 0")
    if c_param < 0.0: raise ValueError("king: c >= 0")

    q = 1.0 - e
    xt, f0, norm = _king_helpers(c_param)
    bb = _b_func_king(m, rc, ctx)
    tt = rc / math.sqrt(q)

    si, co = pa_trig(pa)
    bx = (co * (tx - x0) - si * (ty - y0)) / tt
    by = (si * (tx - x0) + co * (ty - y0)) / tt
    q_t = torch.tensor(q, dtype=tx.dtype, device=tx.device)

    kappa_fn, dkappa_fn, dphi_fn = _make_king_kernels(xt, f0, norm, smallcore)

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
# 7) Pseudo-Jaffe (mass.c:679-735).  Implemented as the difference of two SIEs.
#    p[1]=sigma  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=a_outer  p[7]=rco_inner
# ---------------------------------------------------------------------------
def _sie_body(dx, dy, bb, s, q, si, co, need_kg, need_phi):
    ss = s * facq_sie(q)
    ddx = co * dx - si * dy
    ddy = si * dx + co * dy
    aax, aay = _alpha_sie_dl(ddx, ddy, ss, q)
    ax = bb * (aax * co + aay * si)
    ay = bb * (-aax * si + aay * co)

    kap = gam1 = gam2 = phi = None
    if need_kg:
        pxx_b, pxy_b, pyy_b = _ddphi_sie_dl(ddx, ddy, ss, q)
        rpxx = co * co * pxx_b + 2.0 * co * si * pxy_b + si * si * pyy_b
        rpyy = si * si * pxx_b - 2.0 * co * si * pxy_b + co * co * pyy_b
        rpxy = si * co * (pyy_b - pxx_b) + (co * co - si * si) * pxy_b
        kap = 0.5 * bb * (rpxx + rpyy)
        gam1 = 0.5 * bb * (rpxx - rpyy)
        gam2 = bb * rpxy
        if need_phi:
            phi_body = _phi_sie_dl(ddx, ddy, ss, q, aax, aay)
            phi = bb * phi_body
    return ax, ay, kap, gam1, gam2, phi


def kapgam_jaffe(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                 p: tuple, smallcore: float = K.DEF_SMALLCORE,
                 need_kg: bool = True, need_phi: bool = True):
    sig = float(p[1]); x0 = float(p[2]); y0 = float(p[3])
    e = float(p[4]); pa = float(p[5])
    a_out = float(p[6]); rco = float(p[7])
    if sig < 0.0: raise ValueError("jaffe sigma>=0")
    if rco < 0.0: raise ValueError("jaffe rco>=0")
    if a_out <= 0.0: raise ValueError("jaffe a>0")
    if rco < smallcore: rco = smallcore

    if a_out <= rco:
        # glafic returns all zeros in this regime.
        z = torch.zeros_like(tx)
        if not need_kg: return z, z, None, None, None, None
        return z, z, z, z, z, (z if need_phi else None)

    if not (0.0 <= e < 1.0): raise ValueError("jaffe e in [0,1)")
    q = 1.0 - e
    bb = b_sie(ctx, sig, q)
    si, co = pa_minus_90_trig(pa)
    dx = tx - x0; dy = ty - y0

    ax1, ay1, k1, g11, g21, ph1 = _sie_body(dx, dy, bb, rco, q, si, co, need_kg, need_phi)
    ax2, ay2, k2, g12, g22, ph2 = _sie_body(dx, dy, bb, a_out, q, si, co, need_kg, need_phi)
    ax = ax1 - ax2
    ay = ay1 - ay2

    if not need_kg:
        return ax, ay, None, None, None, None
    kap = k1 - k2
    gam1 = g11 - g12
    gam2 = g21 - g22
    phi = None
    if need_phi:
        phi = ph1 - ph2
    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# 7.5) Sersic density (mass.c:2154-2222)
#    p[1]=M_total  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=r_e  p[7]=n
# ---------------------------------------------------------------------------
def _bn_sers(n: float) -> float:
    """Sersic bn coefficient (mass.c:2060-2075).

    Asymptotic expansion (Ciotti & Bertin 1999) for n > 0.36; polynomial fit
    (MacArthur et al. 2003) below.  Agreement with glafic to 1e-15.
    """
    n2 = n * n; n3 = n2 * n; n4 = n3 * n
    if n > 0.36:
        return (2.0 * n - (1.0 / 3.0)
                + (4.0 / 405.0) / n
                + (46.0 / 25515.0) / n2
                + (131.0 / 1148175.0) / n3
                - (2194697.0 / 30690717750.0) / n4)
    return 0.01945 - 0.8902 * n + 10.95 * n2 - 19.67 * n3 + 13.43 * n4


def _bnn_sers(n: float) -> float:
    """bn^(-n), used as the dimensionless-scale factor bnn (mass.c:2077-2092)."""
    return _bn_sers(n) ** (-n)


def _gam2n1_sers(n: float) -> float:
    """Γ(2n + 1)."""
    import scipy.special as sp
    return float(sp.gamma(2.0 * n + 1.0))


def _b_func_sers(m: float, tt_dimless: float, n: float, ctx: LensContext) -> float:
    """Sersic normalisation bb (mass.c:2048-2058).

    m: total mass [M_sun]
    tt_dimless: dimensionless scale radius (re * bnn, in arcsec)
    """
    rr = cos_mod.thetator_dis(tt_dimless, ctx.dis_ol)    # physical Mpc/h
    gam = _gam2n1_sers(n)
    return (m * ctx.inv_sigma_crit / (math.pi * rr * rr)) / gam


def _make_sers_kernels(n: float):
    """Build (kappa, dkappa, dphi) kernels for Sersic of index n."""
    inv_n = 1.0 / n
    gam_fac = _gam2n1_sers(n)

    def kappa(x: torch.Tensor) -> torch.Tensor:
        # κ(x) = exp(-x^(1/n))  (mass.c:2113-2120)
        xs = torch.clamp(x, min=1.0e-300)
        return torch.exp(-xs.pow(inv_n))

    def dkappa(x: torch.Tensor) -> torch.Tensor:
        # dκ/dx = -x^(1/n) * exp(-x^(1/n)) / (x * n)  (mass.c:2122-2129)
        xs = torch.clamp(x, min=1.0e-300)
        xx = xs.pow(inv_n)
        return -xx * torch.exp(-xx) / (xs * n)

    def dphi(x: torch.Tensor) -> torch.Tensor:
        # dφ/dx = Γ(2n+1) · P(2n, x^(1/n)) / x   (mass.c:2131-2138)
        xs = torch.clamp(x, min=1.0e-300)
        xx = xs.pow(inv_n)
        # torch.special.gammainc is the regularised lower incomplete γ,
        # matching GSL's gsl_sf_gamma_inc_P exactly.
        return gam_fac * torch.special.gammainc(
            torch.tensor(2.0 * n, dtype=x.dtype, device=x.device), xx) / xs

    return kappa, dkappa, dphi


def kapgam_sers(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                p: tuple, smallcore: float = K.DEF_SMALLCORE,
                need_kg: bool = True, need_phi: bool = True):
    m = float(p[1]); x0 = float(p[2]); y0 = float(p[3])
    e = float(p[4]); pa = float(p[5])
    re = float(p[6]); n = float(p[7])
    if m < 0.0: raise ValueError("sers m>=0")
    if not (0.0 <= e < 1.0): raise ValueError("sers e in [0,1)")
    if re <= 0.0: raise ValueError("sers re>0")
    if not (0.06 <= n <= 20.0): raise ValueError(f"sers n in [0.06, 20.0], got {n}")

    q = 1.0 - e
    tt_dimless = re * _bnn_sers(n)               # scale factor, arcsec
    bb = _b_func_sers(m, tt_dimless, n, ctx)
    tt = tt_dimless / math.sqrt(q)

    si, co = pa_trig(pa)
    bx = (co * (tx - x0) - si * (ty - y0)) / tt
    by = (si * (tx - x0) + co * (ty - y0)) / tt
    q_t = torch.tensor(q, dtype=tx.dtype, device=tx.device)

    kappa_fn, dkappa_fn, dphi_fn = _make_sers_kernels(n)

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
# 8) Gaussian potential (mass.c:2865-2934)
#    p[1]=zs_fid  p[2..3]=center  p[4]=e  p[5]=pa  p[6]=sigma  p[7]=kap0
# ---------------------------------------------------------------------------
def _dphi_gau_dl(x: torch.Tensor) -> torch.Tensor:
    small = x < 1.0e-4
    val_big = 2.0 * (1.0 - torch.exp(-0.5 * x * x)) / torch.clamp(x, min=1e-300)
    val_small = x - (x * x * x) / 4.0
    return torch.where(small, val_small, val_big)


def _ddphi_gau_dl(x: torch.Tensor) -> torch.Tensor:
    small = x < 1.0e-4
    dphi = _dphi_gau_dl(x)
    val_big = 2.0 * torch.exp(-0.5 * x * x) - dphi / torch.clamp(x, min=1e-300)
    val_small = 1.0 - 3.0 * x * x / 4.0
    return torch.where(small, val_small, val_big)


def _phi_gau_dl(x: torch.Tensor) -> torch.Tensor:
    # glafic uses gsl_sf_expint_Ei.  We use scipy-like expression: Ei(-0.5 x^2).
    # Torch doesn't ship Ei, so we compute on CPU in double precision per-element
    # and copy back.  This is only called by Newton refinement / small output
    # queries, so the overhead is negligible.
    import scipy.special as sp
    import numpy as np
    y = x.detach().cpu().double().numpy()
    out = np.empty_like(y)
    tiny = y < 9.0e-5
    big = y >= 1.0e1
    mid = ~tiny & ~big
    out[tiny] = math.log(2.0) - 0.57721566490153286 + 0.5 * y[tiny] ** 2
    out[mid] = 2.0 * np.log(y[mid]) - sp.expi(-0.5 * y[mid] ** 2)
    out[big] = 2.0 * np.log(y[big])
    return torch.tensor(out, dtype=x.dtype, device=x.device)


def kapgam_gaupot(ctx: LensContext, tx: torch.Tensor, ty: torch.Tensor,
                  p: tuple, smallcore: float = K.DEF_SMALLCORE,
                  need_kg: bool = True, need_phi: bool = True):
    zs_fid = float(p[1]); x0 = float(p[2]); y0 = float(p[3])
    e = float(p[4]); pa = float(p[5])
    sig = float(p[6]); kap0 = float(p[7])
    if sig <= 0.0: raise ValueError("gaupot sigma>0")
    if not (0.0 <= e < 1.0): raise ValueError("gaupot e in [0,1)")

    fac = fac_pert(ctx, zs_fid) * kap0
    si, co = pa_trig(pa)

    u0, u_x, u_y, u_xx, u_xy, u_yy = u_calc_tensor(
        (tx - x0) / sig, (ty - y0) / sig, e, si, co, smallcore)
    a = fac * _dphi_gau_dl(u0)
    ax = a * u_x * sig
    ay = a * u_y * sig
    if not need_kg:
        return ax, ay, None, None, None, None
    b = fac * _ddphi_gau_dl(u0)
    pxx = b * u_x * u_x + a * u_xx
    pxy = b * u_x * u_y + a * u_xy
    pyy = b * u_y * u_y + a * u_yy
    kap = 0.5 * (pxx + pyy)
    gam1 = 0.5 * (pxx - pyy)
    gam2 = pxy
    phi = None
    if need_phi:
        phi = fac * _phi_gau_dl(u0) * sig * sig
    return ax, ay, kap, gam1, gam2, phi


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
_MODEL_DISPATCH = {
    "point":   kapgam_point,
    "sie":     kapgam_sie,
    "pert":    kapgam_pert,
    "nfwpot":  kapgam_nfwpot,
    "nfw":     kapgam_nfw,
    "king":    kapgam_king,
    "jaffe":   kapgam_jaffe,
    "sers":    kapgam_sers,
    "gaupot":  kapgam_gaupot,
}


def supported_models() -> tuple[str, ...]:
    return tuple(_MODEL_DISPATCH.keys())


def dispatch(model_name: str):
    fn = _MODEL_DISPATCH.get(model_name)
    if fn is None:
        raise NotImplementedError(
            f"lens model '{model_name}' is not implemented in Rhongomyniad v1. "
            f"supported: {sorted(_MODEL_DISPATCH.keys())}")
    return fn
