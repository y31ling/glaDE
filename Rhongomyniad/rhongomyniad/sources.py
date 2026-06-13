"""Extended-source surface-brightness profiles (port of glafic source.c).

Implements the five analytic profiles glafic's ``set_extend`` accepts
(gauss / sersic / tophat / moffat / jaffe) plus the ``source_all`` wrapper
with glafic's conditional sub-pixel integration (source.c:236-282): pixels
close to a source centre — or strongly magnified — average the profile over
an ``np x np`` midpoint rule, mapping each image-plane sub-offset through the
local linearised lens mapping.

Everything is tensor-native: positions are torch tensors of any shape, and
the profile parameters may be python floats OR broadcastable tensors
(per-candidate batching).  All math is float64 on the caller's device.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from . import constants as K

# glafic source-model ids (source.c:9): 1=gauss 2=sersic 3=tophat 4=moffat 5=jaffe
SOURCE_IDS = {"gauss": 1, "sersic": 2, "tophat": 3, "moffat": 4, "jaffe": 5}

# glafic defaults (glafic.h:170-176)
DEF_SOURCE_CALCR0 = 20.0
DEF_FLAG_EXTREF = 1
DEF_SOURCE_REFR0 = 3.0
DEF_NUM_PIXINT = 5


def _bn_sers(n):
    """Sersic bn (mass.c:2060-2075); n may be float or tensor."""
    if torch.is_tensor(n):
        n2 = n * n
        n3 = n2 * n
        n4 = n3 * n
        big = (2.0 * n - (1.0 / 3.0) + (4.0 / 405.0) / n + (46.0 / 25515.0) / n2
               + (131.0 / 1148175.0) / n3 - (2194697.0 / 30690717750.0) / n4)
        sml = 0.01945 - 0.8902 * n + 10.95 * n2 - 19.67 * n3 + 13.43 * n4
        return torch.where(n > 0.36, big, sml)
    n2 = n * n
    n3 = n2 * n
    n4 = n3 * n
    if n > 0.36:
        return (2.0 * n - (1.0 / 3.0) + (4.0 / 405.0) / n + (46.0 / 25515.0) / n2
                + (131.0 / 1148175.0) / n3 - (2194697.0 / 30690717750.0) / n4)
    return 0.01945 - 0.8902 * n + 10.95 * n2 - 19.67 * n3 + 13.43 * n4


def _gam2n1(n):
    """Gamma(2n+1); n float or tensor."""
    if torch.is_tensor(n):
        return torch.exp(torch.lgamma(2.0 * n + 1.0))
    return math.exp(math.lgamma(2.0 * n + 1.0))


def _trig_pa(pa, like: torch.Tensor):
    """sin/cos of -pa deg (ucalc convention, source.c:596-601)."""
    if torch.is_tensor(pa):
        arg = -pa * math.pi / 180.0
        return torch.sin(arg), torch.cos(arg)
    arg = -float(pa) * math.pi / 180.0
    return math.sin(arg), math.cos(arg)


def ucalc(dx: torch.Tensor, dy: torch.Tensor, e, pa) -> torch.Tensor:
    """Rotated elliptical radius^2 (source.c:590-607):
    u = ddx^2/(1-e) + (1-e)*ddy^2."""
    si, co = _trig_pa(pa, dx)
    ddx = co * dx - si * dy
    ddy = si * dx + co * dy
    return ddx * ddx / (1.0 - e) + (1.0 - e) * ddy * ddy


def _checkdis_zero(dx: torch.Tensor, dy: torch.Tensor, r0,
                   source_calcr0: float = DEF_SOURCE_CALCR0) -> torch.Tensor:
    """glafic source_checkdis (source.c:573-588): the profile is hard-zeroed
    outside a |dx|,|dy| <= source_calcr0*r0 box. Returns a bool keep-mask."""
    rr = source_calcr0 * r0
    return (torch.abs(dx) <= rr) & (torch.abs(dy) <= rr)


def profile(model_id: int, x: torch.Tensor, y: torch.Tensor,
            x0, y0, e, pa, r0, n, smallcore: float = K.DEF_SMALLCORE) -> torch.Tensor:
    """Raw (unnormalised, unit-amplitude) SB at source-plane points.

    Parameter slots follow glafic para_ext[i][2..7]: centre (x0, y0),
    ellipticity e, position angle pa [deg], size r0, index n
    (n is the Sersic index / Moffat beta / Jaffe rco; unused for gauss/tophat).
    """
    dx = x - x0
    dy = y - y0
    keep = _checkdis_zero(dx, dy, r0)
    u = ucalc(dx, dy, e, pa)

    if model_id == 1:        # gauss (source.c:422-435)
        f = torch.exp(-0.5 * u / (r0 * r0))
    elif model_id == 2:      # sersic (source.c:441-464)
        bn = _bn_sers(n)
        if torch.is_tensor(n):
            expo = 0.5 / n
        else:
            expo = 0.5 / float(n)
        f = torch.exp(-bn * torch.pow(torch.clamp(u / (r0 * r0), min=0.0), expo))
    elif model_id == 3:      # tophat (source.c:470-488)
        f = (u <= r0 * r0).to(x.dtype)
    elif model_id == 4:      # moffat (source.c:511-526), r0=a, n=b
        f = torch.pow(1.0 + u / (r0 * r0), -n if torch.is_tensor(n) else -float(n))
    elif model_id == 5:      # jaffe (source.c:544-567), r0=a, n=rco
        rco = n
        if torch.is_tensor(rco):
            rco = torch.clamp(rco, min=smallcore)
        else:
            rco = max(float(rco), smallcore)
        f1 = 1.0 / torch.sqrt(rco * rco + u)
        f2 = 1.0 / torch.sqrt(r0 * r0 + u)
        f = (f1 - f2) / ((1.0 / rco) - (1.0 / r0))
        # glafic returns 0 unless a > rco
        if torch.is_tensor(r0) or torch.is_tensor(rco):
            f = torch.where(torch.as_tensor(r0 > rco, device=x.device), f,
                            torch.zeros_like(f))
        elif not (r0 > rco):
            f = torch.zeros_like(u)
    else:
        raise ValueError(f"unknown extended-source model id {model_id}")

    return torch.where(keep, f, torch.zeros_like(f))


def source_all_norm(model_id: int, r0, n, pix_ext: float,
                    smallcore: float = K.DEF_SMALLCORE):
    """Total-flux normalisation factor (source.c:284-314), used when
    flag_extnorm=1 so the amplitude means total flux instead of peak SB."""
    if model_id == 1:
        return 2.0 * math.pi * r0 * r0 / (pix_ext * pix_ext)
    if model_id == 2:
        # calc_norm_sersic (source.c:316-330): norm = bnn_sers(n)^2 * Gamma(2n+1)
        bn = _bn_sers(n)
        if torch.is_tensor(n):
            bnn = torch.pow(bn, -n)
        else:
            bnn = bn ** (-float(n))
        return (math.pi * r0 * r0 / (pix_ext * pix_ext)) * (bnn * bnn * _gam2n1(n))
    if model_id == 3:
        return math.pi * r0 * r0 / (pix_ext * pix_ext)
    if model_id == 4:
        return math.pi * r0 * r0 / ((n - 1.0) * (pix_ext * pix_ext))
    if model_id == 5:
        rco = n
        if torch.is_tensor(rco):
            rco = torch.clamp(rco, min=smallcore)
        else:
            rco = max(float(rco), smallcore)
        return 2.0 * math.pi * r0 * rco / (pix_ext * pix_ext)
    raise ValueError(f"unknown extended-source model id {model_id}")


def _expand(v, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a float-or-tensor parameter to ``like``'s full shape."""
    if torch.is_tensor(v):
        return v.expand_as(like) if v.shape != like.shape else v
    return torch.full_like(like, float(v))


def source_all(model_id: int,
               sx: torch.Tensor, sy: torch.Tensor,
               x0, y0, e, pa, r0, n,
               pxx: torch.Tensor, pxy: torch.Tensor,
               pyx: torch.Tensor, pyy: torch.Tensor,
               pix: float, pix_ext: float,
               imag_ceil: float = K.DEF_IMAG_CEIL,
               flag_extref: int = DEF_FLAG_EXTREF,
               source_refr0: float = DEF_SOURCE_REFR0,
               num_pixint: int = DEF_NUM_PIXINT,
               flag_extnorm: int = 0,
               smallcore: float = K.DEF_SMALLCORE) -> torch.Tensor:
    """Port of glafic source_all (source.c:236-282), vectorised.

    ``sx, sy`` are source-plane positions of the pixel centres (any shape);
    ``pxx..pyy`` the local lens-mapping Jacobian entries scaled by the
    distance ratio (glafic's array_ext_mag * dis_fac).  ``pix`` is the
    image-plane pixel size used for the sub-pixel rule (glafic's pix_psf,
    = pix_ext when no PSF).  Returns unit-amplitude SB (the caller multiplies
    by para_ext[1]).
    """
    f = profile(model_id, sx, sy, x0, y0, e, pa, r0, n, smallcore)

    if model_id == 3 or not flag_extref or pix <= 0.0:
        if flag_extnorm:
            f = f / source_all_norm(model_id, r0, n, pix_ext, smallcore)
        return f

    muinv = torch.abs((1.0 - pxx) * (1.0 - pyy) - pxy * pyx + imag_ceil)
    dx0 = sx - x0
    dy0 = sy - y0
    dr2s = dx0 * dx0 + dy0 * dy0
    dr2 = dr2s / muinv

    refr0_2 = source_refr0 * source_refr0 * r0 * r0
    sub = (dr2s < refr0_2) | (dr2 < 4.0 * pix_ext * pix_ext)
    fine = sub & (dr2 < pix_ext * pix_ext)          # np = 4*num_pixint
    coarse = sub & ~fine                            # np = num_pixint

    # Both sub-pixel rules run on GATHERED masked pixels (fp64 transcendentals
    # are the cost wall on consumer GPUs — only evaluate where glafic would).
    x0_f, y0_f = _expand(x0, sx), _expand(y0, sx)
    e_f, pa_f = _expand(e, sx), _expand(pa, sx)
    r0_f, n_f = _expand(r0, sx), _expand(n, sx)

    flat = lambda t: t.reshape(-1)   # noqa: E731
    out = f.reshape(-1).clone()

    for mask, np_ in ((coarse, num_pixint), (fine, 4 * num_pixint)):
        idx = torch.nonzero(flat(mask), as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        g = lambda t: flat(t)[idx]   # noqa: E731
        gsx, gsy = g(sx), g(sy)
        gpxx, gpxy, gpyx, gpyy = g(pxx), g(pxy), g(pyx), g(pyy)
        gx0, gy0, ge, gpa, gr0, gn = (g(x0_f), g(y0_f), g(e_f),
                                      g(pa_f), g(r0_f), g(n_f))
        h = pix / float(np_)
        hh2 = 1.0 / float(np_ * np_)
        offs = (-0.5) * pix + (torch.arange(np_, device=sx.device,
                                            dtype=sx.dtype) + 0.5) * h
        dxs = offs.view(-1, 1, 1)            # (np, 1, 1)
        dys = offs.view(1, -1, 1)            # (1, np, 1)
        dsx = (1.0 - gpxx) * dxs - gpxy * dys     # (np, np, M)
        dsy = (1.0 - gpyy) * dys - gpyx * dxs
        sub_f = profile(model_id, gsx + dsx, gsy + dsy,
                        gx0, gy0, ge, gpa, gr0, gn, smallcore)
        out[idx] = sub_f.sum(dim=(0, 1)) * hh2

    out = out.reshape(f.shape)
    if flag_extnorm:
        out = out / source_all_norm(model_id, r0, n, pix_ext, smallcore)
    return out
