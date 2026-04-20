"""
Schramm (1990) elliptical-density integrals, evaluated on GPU via
fixed-node Gauss-Legendre quadrature.

These reproduce the behaviour of glafic's
    ell_integ_j(kappa, n)    -- mass.c:3006-3016
    ell_integ_k(dkappa, n)   -- mass.c:2974-2984
    ell_integ_i(dphi)        -- mass.c:3038-3047
but replace adaptive Romberg with a precomputed Gauss-Legendre rule,
so we can batch over thousands of (x, y) queries on the GPU.

glafic toggles between linear (u in [0,1]) and logarithmic
(log u in [log(1e-4*uu), 0], with uu = 1/xi^2(1,1)) integration at
uu = 0.1.  We reproduce that same switch exactly, using torch.where
to apply the correct rule per sample.

The kernel functions (kappa, dkappa, dphi) must accept a single tensor
of elliptical radii and return a tensor of the same shape.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from . import constants as K


# ---- Gauss-Legendre rules (generate once, move to GPU lazily) -----------
# 256 nodes: ~16 digits on smooth kernels; for log-singular NFW/Sersic the
# log-u substitution gives >6 digits.  128 was ~4 digits for Sersic, which
# let downstream DE exploit the numerical gap vs glafic (see v_pointmass_gpu
# verify_on_glafic.py history).
_N_NODES = 256

_gl_nodes_np, _gl_weights_np = np.polynomial.legendre.leggauss(_N_NODES)
# Shift the GL nodes from [-1, 1] to [0, 1].
_GL_U01_NODES_NP = 0.5 * (_gl_nodes_np + 1.0)
_GL_U01_WEIGHTS_NP = 0.5 * _gl_weights_np


_cached_nodes: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def gl_nodes_on(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (nodes_on_[0,1], weights_on_[0,1]) on the requested device/dtype."""
    key = (device, dtype)
    if key not in _cached_nodes:
        nodes = torch.tensor(_GL_U01_NODES_NP, device=device, dtype=dtype)
        weights = torch.tensor(_GL_U01_WEIGHTS_NP, device=device, dtype=dtype)
        _cached_nodes[key] = (nodes, weights)
    return _cached_nodes[key]


# ---- Schramm kernel utilities ------------------------------------------
def _equ(q: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """equ(q, u) = 1 - (1 - q^2) u  (mass.c:3074-3077)."""
    return 1.0 - (1.0 - q * q) * u


def _xi2(u: torch.Tensor, equ: torch.Tensor,
         x: torch.Tensor, y: torch.Tensor, smallcore: float) -> torch.Tensor:
    """
    xi^2(u) = u * (y^2 + x^2/equ + smallcore^2)  (mass.c:3069-3072).
    `x`, `y` are the (rotated) image-plane coordinates divided by scale.
    """
    return u * (y * y + x * x / equ + smallcore * smallcore)


def _nhalf(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    N_{n+1/2}(x) = x^{n+1/2}.  Matches mass.c:3079-3088 (ell_nhalf).
    """
    out = torch.sqrt(x)
    for _ in range(n):
        out = out * x
    return out


# ---- integral evaluators -----------------------------------------------
#
# Each evaluator takes
#   kernel : callable  (xi -> value)   -- broadcasts over any-shape tensor
#   x, y   : tensors of shape (...)    -- scaled/rotated image-plane coords
#   q      : scalar or same-shape tensor
#   n      : integer exponent           (0, 1 or 2 depending on integral)
#   smallcore : small regularization (glafic's `smallcore`)
#
# and returns a tensor of shape (...) with the integral value.
#
# Following glafic, we integrate linearly on [0,1] when uu > 0.1 and
# logarithmically on [log(1e-4*uu), 0] otherwise.  `uu` here is
# 1 / xi^2(u=1, equ(q,1)) = 1 / (x^2/q^2 + y^2 + smallcore^2).
# We compute both quadratures for every sample and blend per-point.


def _compute_both(
    kernel: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    q: torch.Tensor,
    integrand_builder: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    smallcore: float,
):
    """
    Returns (result_linear, result_log, uu).

    `integrand_builder(u, equ, xi, kappa_at_xi)` returns the integrand
    value f(u).  This lets us reuse the same machinery for j, k, i kernels.
    """
    device = x.device
    dtype = x.dtype
    nodes01, weights01 = gl_nodes_on(device, dtype)           # (N,)

    # Broadcast: u has shape (N, 1, 1, ...) over the query shape.
    sample_shape = x.shape
    u_linear = nodes01.view(-1, *([1] * x.ndim))              # (N, 1, 1, ...)
    w_linear = weights01.view(-1, *([1] * x.ndim))

    # Linear-u rule: a = 0, b = 1.  Integral = sum w_i f(u_i), already weighted by (b-a)/2 in w01.
    equ_lin = _equ(q, u_linear)
    xi2_lin = _xi2(u_linear, equ_lin, x, y, smallcore)
    xi_lin = torch.sqrt(xi2_lin)
    kap_lin = kernel(xi_lin)
    f_lin = integrand_builder(u_linear, equ_lin, xi_lin, kap_lin)
    result_linear = (w_linear * f_lin).sum(dim=0)

    # uu = 1 / ell_xi2(1.0, 1.0) in glafic  (mass.c passes literal 1.0 for equ),
    # so uu = 1 / (x^2 + y^2 + smallcore^2).  Not divided by q^2.
    uu = 1.0 / (x * x + y * y + smallcore * smallcore)

    # Log rule: substitute u = exp(l) with l in [ln(u_min), 0].
    # du = e^l dl  -> integrand *= u.
    # u_min = 1e-4 * uu (same factor as glafic mass.c:1108).
    # When uu > 0.1 we still compute the log branch but it's overwritten below.
    u_min = 1.0e-4 * uu
    # Protect against degenerate points where uu -> inf (would push u_min > 1).
    u_min = torch.clamp(u_min, max=1.0 - 1.0e-12)
    lmin = torch.log(torch.clamp(u_min, min=K.OFFSET_LOG))    # shape (...)
    lmax = torch.zeros_like(lmin)                             # log(1) = 0

    # Map GL nodes from [0,1] to [lmin, lmax]: l_i = lmin + (lmax-lmin) * u01_i
    # weights scale by (lmax - lmin).
    span = lmax - lmin                                        # shape (...)
    u01_shape = nodes01.view(-1, *([1] * x.ndim))             # (N, 1, 1, ...)
    w01_shape = weights01.view(-1, *([1] * x.ndim))
    l_nodes = lmin.unsqueeze(0) + span.unsqueeze(0) * u01_shape  # (N, ...)
    w_log = w01_shape * span.unsqueeze(0)                        # (N, ...)
    u_log = torch.exp(l_nodes)

    equ_log = _equ(q, u_log)
    xi2_log = _xi2(u_log, equ_log, x, y, smallcore)
    xi_log = torch.sqrt(xi2_log)
    kap_log = kernel(xi_log)
    f_log = integrand_builder(u_log, equ_log, xi_log, kap_log) * u_log  # * u due to dl = du/u
    result_log = (w_log * f_log).sum(dim=0)

    return result_linear, result_log, uu


def ell_integ_j(
    kernel: Callable[[torch.Tensor], torch.Tensor],
    n: int,
    x: torch.Tensor, y: torch.Tensor, q: torch.Tensor,
    smallcore: float,
) -> torch.Tensor:
    """
    J_n(x, y) = int_0^1 kappa(xi(u)) / equ^{n+1/2} du   (mass.c:3006-3027).

    n is 0 or 1 in practice.
    """
    def integrand(u, equ, xi, kap):                          # f(u) = kappa/equ^(n+1/2)
        return kap / _nhalf(equ, n)
    lin, logr, uu = _compute_both(kernel, x, y, q, integrand, smallcore)
    return torch.where(uu > 0.1, lin, logr)


def ell_integ_k(
    dkernel: Callable[[torch.Tensor], torch.Tensor],
    n: int,
    x: torch.Tensor, y: torch.Tensor, q: torch.Tensor,
    smallcore: float,
) -> torch.Tensor:
    """
    K_n(x, y) = int_0^1 u * dkappa(xi(u)) / (2*xi*equ^{n+1/2}) du  (mass.c:2974-3004).
    """
    def integrand(u, equ, xi, dkap):
        # guard against xi==0 from the smallcore; xi has smallcore^2 in xi^2.
        denom = 2.0 * xi * _nhalf(equ, n)
        return u * dkap / denom
    lin, logr, uu = _compute_both(dkernel, x, y, q, integrand, smallcore)
    return torch.where(uu > 0.1, lin, logr)


def ell_integ_i(
    dphi_kernel: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor, y: torch.Tensor, q: torch.Tensor,
    smallcore: float,
) -> torch.Tensor:
    """
    I(x, y) = int_0^1 xi * dphi(xi(u)) / (u * equ^{1/2}) du  (mass.c:3038-3066).
    """
    def integrand(u, equ, xi, dphi):
        # mass.c:3049-3057 defines f = u * ell_nhalf(equ, 0) = u*sqrt(equ).
        # Integrand returned: xi * dphi / f = xi * dphi / (u * sqrt(equ)).
        denom = u * _nhalf(equ, 0)
        return xi * dphi / denom
    lin, logr, uu = _compute_both(dphi_kernel, x, y, q, integrand, smallcore)
    return torch.where(uu > 0.1, lin, logr)


# ---- coordinate rotation helpers (mass.c:3090-3103) --------------------
def ell_pxpy(bpx: torch.Tensor, bpy: torch.Tensor,
             si: float, co: float) -> tuple[torch.Tensor, torch.Tensor]:
    px = bpx * co + bpy * si
    py = -bpx * si + bpy * co
    return px, py


def ell_pxxpyy(bpxx: torch.Tensor, bpyy: torch.Tensor, bpxy: torch.Tensor,
               si: float, co: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pxx = bpxx * co * co + bpxy * 2.0 * co * si + bpyy * si * si
    pxy = -bpxx * si * co + bpxy * (co * co - si * si) + bpyy * co * si
    pyy = bpxx * si * si - bpxy * 2.0 * co * si + bpyy * co * co
    return pxx, pyy, pxy


def u_calc_tensor(dx: torch.Tensor, dy: torch.Tensor, e: float,
                  si: float, co: float, smallcore: float):
    """
    Elliptical potential coordinate + Jacobian helpers (mass.c:2940-2968).

    Returns a 6-tuple (u, u_x, u_y, u_xx, u_xy, u_yy) of tensors with the
    shape of dx/dy.
    """
    ep = e
    si2 = 2.0 * si * co
    co2 = co * co - si * si
    ddx = co * dx - si * dy
    ddy = si * dx + co * dy

    u0 = torch.sqrt((1.0 + ep) * ddx * ddx + (1.0 - ep) * ddy * ddy) + smallcore
    u_x = (dx + ep * (dx * co2 - dy * si2)) / u0
    u_y = (dy - ep * (dy * co2 + dx * si2)) / u0
    u_xx = (1.0 + ep * co2 - u_x * u_x) / u0
    u_xy = (-ep * si2 - u_x * u_y) / u0
    u_yy = (1.0 - ep * co2 - u_y * u_y) / u0
    return u0, u_x, u_y, u_xx, u_xy, u_yy
