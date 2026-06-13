"""gnfw / gnfwpot / ein / einpot — table-interpolated radial profiles.

glafic evaluates these models' radial kappa/dkappa/dphi from lazily built
log-log lookup tables (gnfw_tab.c / ein_tab.c, defaults gnfw_usetab=1,
ein_usetab=1) and interpolates bilinearly in (alpha, ln x).  This module
reproduces the SAME grids and the SAME interpolation, but builds the tables
with vectorised fixed-order Gauss-Legendre quadrature at ~1e-9 accuracy —
better than glafic's builder (Romberg at TOL_ROMBERG_GNFW=3e-4 /
TOL_ROMBERG_EIN=1e-3) — and caches them on disk next to the package.

Parameter layout (glafic): p[1]=M  p[2..3]=centre  p[4]=e  p[5]=pa  p[6]=c
p[7]=alpha (gnfw inner slope in [0,2] / Einasto index in [0.02, 1.0]).

The elliptical-density forms follow glafic's special Schramm switch
(mass.c:1452-1461, 2825-2838): standard linear/log rule for alpha <= 1,
always-log with u_min = smallcore^2 * uu for alpha > 1.  Tensor params are
supported throughout (alpha per candidate interpolates per candidate).
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from . import constants as K
from . import cosmology as cos_mod
from .elliptical import ell_integ_i, ell_integ_j, ell_integ_k, ell_pxpy, \
    ell_pxxpyy, gl_nodes_on, u_calc_tensor
from .lens_models import LensContext, _is_t, _pf, _q_tensor, _rs_nfw, \
    _t_sqrt, pa_trig

# table grids (gnfw_tab.c:8-13 / ein_tab.c:8-13)
_NUM_LNX = 801
_LNX_MIN = -23.0
_DLNX = 0.0575
_GNFW_NUM_ALP = 101
_GNFW_ALP_MIN = 0.0
_GNFW_DALP = 0.02
_EIN_NUM_ALP = 99
_EIN_ALP_MIN = 0.02
_EIN_DALP = 0.01
_OFFSET_LOG = 1.0e-300

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tab_cache")


# ---------------------------------------------------------------------------
# exact radial functions (numpy, vectorised over x) — used to BUILD the tables
# ---------------------------------------------------------------------------
def _gl(n: int, a: float, b: float):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def _gnfw_kappa_exact(x: np.ndarray, alpha: float) -> np.ndarray:
    """0.5 * int_{-25}^{25} e^t / (aa^3 (a/aa)^alpha) dt  (mass.c:1284-1294)."""
    t, w = _gl(2048, -25.0, 25.0)
    xx = np.exp(t)[None, :]
    a = np.sqrt(x[:, None] ** 2 + xx * xx)
    aa = 1.0 + a
    pp = (a / aa) ** alpha
    return 0.5 * (w[None, :] * xx / (aa ** 3 * pp)).sum(axis=1)


def _gnfw_dkappa_exact(x: np.ndarray, alpha: float) -> np.ndarray:
    """mass.c:1306-1315."""
    t, w = _gl(2048, -25.0, 25.0)
    xx = np.exp(t)[None, :]
    xc = x[:, None]
    a = np.sqrt(xc * xc + xx * xx)
    aa = 1.0 + a
    pp = (a / aa) ** alpha
    f = xc * xx * (-3.0 * a - alpha) / (a * a * pp * aa ** 4)
    return 0.5 * (w[None, :] * f).sum(axis=1)


def _hgnfw_exact(x: np.ndarray, alpha: float) -> np.ndarray:
    """hgnfw(x) = int_0^x (t/(1+t))^{2-alpha}/(1+t) dt  (mass.c:1367-1383),
    via the substitution t = x*s."""
    s, w = _gl(1024, 0.0, 1.0)
    t = x[:, None] * s[None, :]
    f = (t / (1.0 + t)) ** (2.0 - alpha) / (1.0 + t)
    return x * (w[None, :] * f).sum(axis=1)


def _gnfw_dphi_exact(x: np.ndarray, alpha: float) -> np.ndarray:
    """dphi(x) = hgnfw(x)/x + x^{2-alpha} * F(x),
    F(x) = int (t+x)^{alpha-3} (1-sqrt(1-t^2))/t dt over t in (0,1)
    (mass.c:1325-1345), evaluated with the log substitution glafic uses for
    small x (valid for all x)."""
    l, w = _gl(4096, -30.0, 0.0)
    t = np.exp(l)[None, :]
    xc = x[:, None]
    f = (t + xc) ** (alpha - 3.0) * (1.0 - np.sqrt(np.maximum(1.0 - t * t, 0.0))) / t
    F = ((w[None, :] * t) * f).sum(axis=1)        # extra t from dt = t dl
    return _hgnfw_exact(x, alpha) / x + x ** (2.0 - alpha) * F


def _hein_exact(y: np.ndarray, alpha: float) -> np.ndarray:
    """hein (mass.c:2756-2764): (alpha/2)^{3/alpha} Gamma(3/a) P(3/a, 2 y^a / a) / a."""
    from scipy.special import gammainc, gammaln
    gam = gammainc(3.0 / alpha, 2.0 * np.power(y, alpha) / alpha) \
        * np.exp(gammaln(3.0 / alpha))
    return (0.5 * alpha) ** (3.0 / alpha) * gam / alpha


def _ein_kappa_exact(x: np.ndarray, alpha: float) -> np.ndarray:
    """mass.c:2665-2678."""
    t, w = _gl(2048, -25.0, 25.0)
    xx = np.exp(t)[None, :]
    yy = x[:, None] ** 2 + xx * xx
    ff = -2.0 * yy ** (0.5 * alpha) / alpha
    val = np.where(ff > -300.0, np.exp(np.maximum(ff, -300.0)) * xx, 0.0)
    return 0.5 * (w[None, :] * val).sum(axis=1)


def _ein_dkappa_exact(x: np.ndarray, alpha: float) -> np.ndarray:
    """mass.c:2690-2704."""
    t, w = _gl(2048, -25.0, 25.0)
    xx = np.exp(t)[None, :]
    xc = x[:, None]
    yy = xc * xc + xx * xx
    pp = yy ** (0.5 * alpha)
    ff = -2.0 * pp / alpha
    val = np.where(ff > -300.0,
                   np.exp(np.maximum(ff, -300.0)) * xx * (-2.0) * (pp / yy) * xc,
                   0.0)
    return 0.5 * (w[None, :] * val).sum(axis=1)


def _ein_dphi_exact(x: np.ndarray, alpha: float) -> np.ndarray:
    """mass.c:2706-2739: dphi(x) = (1/x) int e^l g(e^l) dl over (-30, 0),
    g(t) = (hein(x s) + t hein(x s / t)) / ((1+t^2) s),  s = sqrt(1+t^2)."""
    l, w = _gl(2048, -30.0, 0.0)
    t = np.exp(l)[None, :]
    xc = x[:, None]
    s = np.sqrt(1.0 + t * t)
    g = (_hein_exact(xc * s, alpha) + t * _hein_exact(xc * s / t, alpha)) \
        / ((1.0 + t * t) * s)
    return ((w[None, :] * t) * g).sum(axis=1) / x


# ---------------------------------------------------------------------------
# table build + cache
# ---------------------------------------------------------------------------
def _build_tables(family: str) -> dict:
    lnx = _LNX_MIN + _DLNX * np.arange(_NUM_LNX)
    x = np.exp(lnx)
    if family == "gnfw":
        alps = _GNFW_ALP_MIN + _GNFW_DALP * np.arange(_GNFW_NUM_ALP)
        # alpha=0 exactly: the (a/aa)^0 = 1 kernels are fine; dphi x^{2-0} fine
        fns = (_gnfw_kappa_exact, _gnfw_dkappa_exact, _gnfw_dphi_exact)
    else:
        alps = _EIN_ALP_MIN + _EIN_DALP * np.arange(_EIN_NUM_ALP)
        fns = (_ein_kappa_exact, _ein_dkappa_exact, _ein_dphi_exact)
    kap = np.empty((len(alps), _NUM_LNX))
    dkap = np.empty_like(kap)
    dphi = np.empty_like(kap)
    for i, alp in enumerate(alps):
        a_eff = max(alp, 1e-10) if family == "gnfw" else alp
        kap[i] = np.log(fns[0](x, a_eff) + _OFFSET_LOG)
        dkap[i] = np.log(-fns[1](x, a_eff) + _OFFSET_LOG)
        dphi[i] = np.log(fns[2](x, a_eff) + _OFFSET_LOG)
    return {"lnx": lnx, "alp": alps, "lnkappa": kap, "lndkappa": dkap,
            "lndphi": dphi}


_tables_np: dict = {}
_tables_torch: dict = {}


def _get_tables_np(family: str) -> dict:
    if family in _tables_np:
        return _tables_np[family]
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{family}_tab_v1.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            tab = {k: z[k] for k in z.files}
    else:
        tab = _build_tables(family)
        np.savez_compressed(path, **tab)
    _tables_np[family] = tab
    return tab


def _get_tables(family: str, device, dtype) -> dict:
    key = (family, device, dtype)
    if key not in _tables_torch:
        tab = _get_tables_np(family)
        _tables_torch[key] = {k: torch.tensor(v, device=device, dtype=dtype)
                              for k, v in tab.items()}
    return _tables_torch[key]


def _interp(tab2d: torch.Tensor, alp_grid: torch.Tensor, lnx_grid: torch.Tensor,
            alp, lnx: torch.Tensor) -> torch.Tensor:
    """Bilinear interpolation matching intpol_gnfw_lin (gnfw_tab.c:104-124):
    out-of-range queries snap to the edge.  ``alp`` may be a float or a tensor
    broadcastable against ``lnx``."""
    na, nx = tab2d.shape
    a_min = float(alp_grid[0])
    da = float(alp_grid[1] - alp_grid[0])
    x_min = float(lnx_grid[0])
    dx = float(lnx_grid[1] - lnx_grid[0])

    if not torch.is_tensor(alp):
        alp = torch.tensor(float(alp), device=lnx.device, dtype=lnx.dtype)
    alp_b = (alp + torch.zeros_like(lnx))
    i = torch.clamp(((alp_b - a_min) / da).floor().long(), 0, na - 2)
    j = torch.clamp(((lnx - x_min) / dx).floor().long(), 0, nx - 2)
    a_lo = a_min + i.to(lnx.dtype) * da
    x_lo = x_min + j.to(lnx.dtype) * dx
    t = torch.clamp((alp_b - a_lo) / da, 0.0, 1.0)
    u = torch.clamp((lnx - x_lo) / dx, 0.0, 1.0)

    flat = tab2d.reshape(-1)
    idx = i * nx + j
    v00 = flat[idx]
    v10 = flat[idx + nx]
    v01 = flat[idx + 1]
    v11 = flat[idx + nx + 1]
    return ((1.0 - t) * (1.0 - u) * v00 + t * (1.0 - u) * v10
            + t * u * v11 + (1.0 - t) * u * v01)


def _make_tab_kernels(family: str, alpha, like: torch.Tensor):
    """(kappa, dkappa, dphi) closures evaluating the lookup tables."""
    tabs = _get_tables(family, like.device, like.dtype)

    def kappa(x: torch.Tensor) -> torch.Tensor:
        lnx = torch.log(torch.clamp(x, min=1e-300))
        return torch.exp(_interp(tabs["lnkappa"], tabs["alp"], tabs["lnx"],
                                 alpha, lnx))

    def dkappa(x: torch.Tensor) -> torch.Tensor:
        lnx = torch.log(torch.clamp(x, min=1e-300))
        return -torch.exp(_interp(tabs["lndkappa"], tabs["alp"], tabs["lnx"],
                                  alpha, lnx))

    def dphi(x: torch.Tensor) -> torch.Tensor:
        lnx = torch.log(torch.clamp(x, min=1e-300))
        return torch.exp(_interp(tabs["lndphi"], tabs["alp"], tabs["lnx"],
                                 alpha, lnx))

    return kappa, dkappa, dphi


# ---------------------------------------------------------------------------
# normalisations (b_func_gnfw / b_func_ein + calc_bbtt, mass.c:1385-1405 etc.)
# ---------------------------------------------------------------------------
def _hgnfw_t(c, alpha, like: torch.Tensor):
    """hgnfw(c) for float-or-tensor c/alpha (vectorised GL over (0, c))."""
    nodes, weights = gl_nodes_on(like.device, like.dtype)   # on (0,1)
    if not torch.is_tensor(c):
        c = torch.tensor(float(c), device=like.device, dtype=like.dtype)
    if not torch.is_tensor(alpha):
        alpha = torch.tensor(float(alpha), device=like.device, dtype=like.dtype)
    sh = (-1,) + (1,) * c.dim()
    s = nodes.view(sh)
    w = weights.view(sh)
    t = c * s
    f = torch.pow(t / (1.0 + t), 2.0 - alpha) / (1.0 + t)
    return c * (w * f).sum(dim=0)


def _hein_t(y, alpha, like: torch.Tensor):
    """hein(y) (mass.c:2756-2764) for float-or-tensor y/alpha."""
    if not torch.is_tensor(y):
        y = torch.tensor(float(y), device=like.device, dtype=like.dtype)
    if not torch.is_tensor(alpha):
        alpha = torch.tensor(float(alpha), device=like.device, dtype=like.dtype)
    gam = torch.special.gammainc(3.0 / alpha, 2.0 * torch.pow(y, alpha) / alpha) \
        * torch.exp(torch.lgamma(3.0 / alpha))
    return torch.pow(0.5 * alpha, 3.0 / alpha) * gam / alpha


def _calc_bbtt_tab(family: str, m, c, alpha, ctx: LensContext,
                   like: torch.Tensor, nfw_users: int = K.DEF_NFW_USERS):
    """calc_bbtt_gnfw (mass.c:1252-1267) / calc_bbtt_ein (mass.c:2634-2648)."""
    if family == "gnfw":
        h_of_c = lambda cc_: _hgnfw_t(cc_, alpha, like)     # noqa: E731
        cvir = (2.0 - alpha) * c if family == "gnfw" else c  # c_-2 -> c_vir
    else:
        h_of_c = lambda cc_: _hein_t(cc_, alpha, like)      # noqa: E731
        cvir = c

    def b_func(m_, c_):
        return (K.NFW_B_NORM * ctx.dis_ol * ctx.dis_ls
                * (ctx.delome * ctx.delome * m_) ** (1.0 / 3.0)
                * (c_ * c_ / h_of_c(c_)) / ctx.dis_os)

    if nfw_users == 0:
        cc = cvir
        bb = b_func(m, cc)
        tt = cos_mod.rtotheta_dis(_rs_nfw(m, cc, ctx.delome), ctx.dis_ol)
    else:
        tt = c
        cc = _rs_nfw(m, 1.0, ctx.delome) / cos_mod.thetator_dis(c, ctx.dis_ol)
        bb = b_func(m, cc)
    return bb, tt


# ---------------------------------------------------------------------------
# the steep-slope Schramm switch (mass.c:1452-1461 / 2825-2838)
# ---------------------------------------------------------------------------
def _steep_switch(alpha, bx, by, smallcore):
    uu = 1.0 / (bx * bx + by * by + smallcore * smallcore)
    if torch.is_tensor(alpha):
        shallow = alpha <= 1.0
        use_linear = shallow + torch.zeros_like(bx, dtype=torch.bool)
        lnmin = torch.where(use_linear, 1.0e-4 * uu,
                            smallcore * smallcore * uu)
        return use_linear, lnmin
    if alpha <= 1.0:
        return None, None              # standard rule
    return False, smallcore * smallcore * uu


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
def _kapgam_tab_density(family: str, ctx, tx, ty, p, smallcore,
                        need_kg, need_phi, nfw_users=K.DEF_NFW_USERS):
    m = _pf(p[1]); x0 = _pf(p[2]); y0 = _pf(p[3])
    e = _pf(p[4]); pa = _pf(p[5]); c = _pf(p[6]); alpha = _pf(p[7])
    if not _is_t(m, e, c, alpha):
        if m < 0.0: raise ValueError(f"{family}: m >= 0")
        if not (0.0 <= e < 1.0): raise ValueError(f"{family}: e in [0,1)")
        if c <= 0.0: raise ValueError(f"{family}: c > 0")
        if family == "gnfw" and not (0.0 <= alpha <= 2.0):
            raise ValueError("gnfw: alpha in [0,2]")
        if family == "ein" and not (0.02 <= alpha <= 1.0):
            raise ValueError("ein: alpha in [0.02,1.0] (table range)")

    q = 1.0 - e
    bb, tt = _calc_bbtt_tab(family, m, c, alpha, ctx, tx, nfw_users)
    tt = tt / _t_sqrt(q)
    si, co = pa_trig(pa)

    bx = (co * (tx - x0) - si * (ty - y0)) / tt
    by = (si * (tx - x0) + co * (ty - y0)) / tt
    q_t = _q_tensor(q, tx)
    kappa_fn, dkappa_fn, dphi_fn = _make_tab_kernels(family, alpha, tx)
    use_linear, lnmin = _steep_switch(alpha, bx, by, smallcore)

    j1 = ell_integ_j(kappa_fn, 1, bx, by, q_t, smallcore, use_linear, lnmin)
    j0 = ell_integ_j(kappa_fn, 0, bx, by, q_t, smallcore, use_linear, lnmin)
    bpx = q * bx * j1
    bpy = q * by * j0
    px, py = ell_pxpy(bpx, bpy, si, co)
    ax = bb * tt * px
    ay = bb * tt * py

    if not need_kg:
        return ax, ay, None, None, None, None

    k2 = ell_integ_k(dkappa_fn, 2, bx, by, q_t, smallcore, use_linear, lnmin)
    k0 = ell_integ_k(dkappa_fn, 0, bx, by, q_t, smallcore, use_linear, lnmin)
    k1 = ell_integ_k(dkappa_fn, 1, bx, by, q_t, smallcore, use_linear, lnmin)
    bpxx = 2.0 * q * bx * bx * k2 + q * j1
    bpyy = 2.0 * q * by * by * k0 + q * j0
    bpxy = 2.0 * q * bx * by * k1
    pxx, pyy, pxy = ell_pxxpyy(bpxx, bpyy, bpxy, si, co)

    kap = 0.5 * bb * (pxx + pyy)
    gam1 = 0.5 * bb * (pxx - pyy)
    gam2 = bb * pxy
    phi = None
    if need_phi:
        phi_int = ell_integ_i(dphi_fn, bx, by, q_t, smallcore, use_linear, lnmin)
        phi = 0.5 * q * bb * phi_int * tt * tt
    return ax, ay, kap, gam1, gam2, phi


def _phi_tab(family: str, alpha, u0: torch.Tensor) -> torch.Tensor:
    """phi(x) = int_{-25}^{ln x} e^l dphi(e^l) dl (mass.c:1347-1357 /
    2741-2754), via GL nodes mapped onto the variable-range interval and the
    table-interpolated dphi."""
    tabs_like = u0
    nodes, weights = gl_nodes_on(u0.device, u0.dtype)
    lo = -25.0
    hi = torch.log(torch.clamp(u0, min=1e-300))
    span = hi - lo
    sh = (-1,) + (1,) * u0.dim()
    l = lo + span.unsqueeze(0) * nodes.view(sh)
    w = span.unsqueeze(0) * weights.view(sh)
    x = torch.exp(l)
    kappa_fn, dkappa_fn, dphi_fn = _make_tab_kernels(family, alpha, tabs_like)
    return (w * x * dphi_fn(x)).sum(dim=0)


def _kapgam_tab_pot(family: str, ctx, tx, ty, p, smallcore,
                    need_kg, need_phi, nfw_users=K.DEF_NFW_USERS):
    m = _pf(p[1]); x0 = _pf(p[2]); y0 = _pf(p[3])
    e = _pf(p[4]); pa = _pf(p[5]); c = _pf(p[6]); alpha = _pf(p[7])
    if not _is_t(m, e, c, alpha):
        if m < 0.0: raise ValueError(f"{family}pot: m >= 0")
        if not (0.0 <= e < 1.0): raise ValueError(f"{family}pot: e in [0,1)")
        if c <= 0.0: raise ValueError(f"{family}pot: c > 0")

    bb, tt = _calc_bbtt_tab(family, m, c, alpha, ctx, tx, nfw_users)
    si, co = pa_trig(pa)
    u0, u_x, u_y, u_xx, u_xy, u_yy = u_calc_tensor(
        (tx - x0) / tt, (ty - y0) / tt, e, si, co, smallcore)

    kappa_fn, _dk, dphi_fn = _make_tab_kernels(family, alpha, tx)
    dphi = dphi_fn(u0)
    a = bb * dphi
    ax = a * u_x * tt
    ay = a * u_y * tt
    if not need_kg:
        return ax, ay, None, None, None, None
    # ddphi = 2*kappa - dphi/x (mass.c:963-966)
    b = bb * (2.0 * kappa_fn(u0) - dphi / u0)
    pxx = b * u_x * u_x + a * u_xx
    pxy = b * u_x * u_y + a * u_xy
    pyy = b * u_y * u_y + a * u_yy
    kap = 0.5 * (pxx + pyy)
    gam1 = 0.5 * (pxx - pyy)
    gam2 = pxy
    phi = None
    if need_phi:
        phi = bb * _phi_tab(family, alpha, u0) * tt * tt
    return ax, ay, kap, gam1, gam2, phi


def kapgam_gnfw(ctx, tx, ty, p, smallcore=K.DEF_SMALLCORE,
                need_kg=True, need_phi=True, nfw_users=K.DEF_NFW_USERS):
    return _kapgam_tab_density("gnfw", ctx, tx, ty, p, smallcore,
                               need_kg, need_phi, nfw_users)


def kapgam_gnfwpot(ctx, tx, ty, p, smallcore=K.DEF_SMALLCORE,
                   need_kg=True, need_phi=True, nfw_users=K.DEF_NFW_USERS):
    return _kapgam_tab_pot("gnfw", ctx, tx, ty, p, smallcore,
                           need_kg, need_phi, nfw_users)


def kapgam_ein(ctx, tx, ty, p, smallcore=K.DEF_SMALLCORE,
               need_kg=True, need_phi=True, nfw_users=K.DEF_NFW_USERS):
    return _kapgam_tab_density("ein", ctx, tx, ty, p, smallcore,
                               need_kg, need_phi, nfw_users)


def kapgam_einpot(ctx, tx, ty, p, smallcore=K.DEF_SMALLCORE,
                  need_kg=True, need_phi=True, nfw_users=K.DEF_NFW_USERS):
    return _kapgam_tab_pot("ein", ctx, tx, ty, p, smallcore,
                           need_kg, need_phi, nfw_users)


KERNELS = {
    "gnfw": kapgam_gnfw,
    "gnfwpot": kapgam_gnfwpot,
    "ein": kapgam_ein,
    "einpot": kapgam_einpot,
}
