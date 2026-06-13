"""Extended-source (FITS) chi^2 engine — GPU port of glafic's extend pipeline.

Implements, with the same algorithms and branch thresholds as the C source:

* ``calc_obsnoise``      — iterative sigma-clip sky + per-pixel noise (fits.c:778-856)
* deflection table + finite-difference magnification tensor (extend.c:42-147)
* ray-shooting + surface-brightness evaluation (extend.c:169-222, via sources.py)
* pixel chi^2 + sky (opt_extend.c:319-361)
* parameter range checks & priors (opt_extend.c:367-484, opt_lens.c:546-657, init.c:651-737)
* point-source chi^2, source plane & image plane, including the inner
  source-position simplex solve and flux/time-delay zero points
  (opt_point.c:24-547, amoeba_opt.c)
* ``c2calc_each`` 8-component breakdown (opt_lens.c:337-374, glade-local)

No PSF convolution: GLADE's CPU pipeline never configures a PSF
(``flag_seeing`` stays 0), so parity does not require it.  Multi-plane is out
of scope (single lens plane, like the rest of Rhongomyniad).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from . import constants as K
from .sources import SOURCE_IDS, source_all

# ---------------------------------------------------------------------------
# glafic defaults (glafic.h)
# ---------------------------------------------------------------------------
CHI2PEN_RANGE = 1.0e30      # DEF_CHI2PEN_RANGE
CHI2PEN_NIMG = 1.0e30       # DEF_CHI2PEN_NIMG
CHI2PEN_PARITY = 1.0e30     # DEF_CHI2PEN_PARITY
CHI2_MIN_SET = 1.0e60
DIS2_MIN_SET = 1.0e60
NPIX_SMALL_OFFSET = 1.0e-10
TOL_AMOEBA = 1.0e-5         # DEF_TOL_AMOEBA
NMAX_AMOEBA_POINT = 100     # DEF_NMAX_AMOEBA_POINT

# parameter-slot range defaults (init.c:651-737 / glafic.h INIT_*)
_INIT_RANGES = (
    (1.0e-6, 1.0e4),        # j=0  z
    (0.0, 1.0e30),          # j=1  mass / norm / sigma
    (-1.0e30, 1.0e30),      # j=2  x
    (-1.0e30, 1.0e30),      # j=3  y
    (0.0, 1.0),             # j=4  e
    (-360.0, 360.0),        # j=5  pa
    (0.0, 1.0e30),          # j=6  r0
    (-1.0e30, 1.0e30),      # j=7  n
)
# point sources use the para_poi layout [zs, xs, ys] (init.c:718-723)
_INIT_RANGES_POI = (
    (1.0e-6, 1.0e4),        # j=0  zs
    (-1.0e30, 1.0e30),      # j=1  xs
    (-1.0e30, 1.0e30),      # j=2  ys
)
_COSMO_RANGES = {           # glafic.h:346-353
    "omega": (-1.0, 3.0), "lambda": (-1.0, 3.0),
    "weos": (-5.0, 1.0), "hubble": (0.0, 3.0),
}

# secondary-parameter defaults relevant to the chi2 (glafic.h)
SECONDARY_DEFAULTS = {
    "chi2_splane": 0, "chi2_checknimg": 1, "chi2_usemag": 0, "chi2_restart": 0,
    "obs_gain": 3.0, "obs_ncomb": 1, "obs_readnoise": 10.0,
    "noise_clip": 3.0, "skyfix": 0, "skyfix_value": 1.0e10,
    "flag_extnorm": 0, "flag_extref": 1, "source_refr0": 3.0,
    "num_pixint": 5, "source_calcr0": 20.0, "ran_seed": 0,
}


# ---------------------------------------------------------------------------
# Nelder-Mead simplex (verbatim port of amoeba_opt.c, Hutt nmsimplex)
# ---------------------------------------------------------------------------
def nm_simplex(v: np.ndarray, f: np.ndarray, ftol: float,
               func: Callable[[np.ndarray], float], nmax: int) -> tuple[np.ndarray, float]:
    """``v``: (n+1, n) start vertices, ``f``: (n+1,) start values.
    Returns (best_vertex, best_value).  Mirrors glafic's convergence rule:
    stop when 2|max-min|/(|max|+|min|) < ftol or |max| < 0.1*ftol."""
    ALPHA, BETA, GAMMA = 1.0, 0.5, 2.0
    n = v.shape[1]
    for _ in range(nmax):
        vg = int(np.argmax(f))                       # worst
        vs = int(np.argmin(f))                       # best
        f_masked = f.copy()
        f_masked[vg] = -np.inf
        vh = int(np.argmax(f_masked))                # second worst
        vm = (v.sum(axis=0) - v[vg]) / n             # centroid w/o worst

        vr = vm + ALPHA * (vm - v[vg])
        fr = func(vr)
        if f[vs] <= fr < f[vh]:
            v[vg], f[vg] = vr, fr
        if fr < f[vs]:
            ve = vm + GAMMA * (vr - vm)
            fe = func(ve)
            if fe < fr:
                v[vg], f[vg] = ve, fe
            else:
                v[vg], f[vg] = vr, fr
        if fr >= f[vh]:
            if fr < f[vg]:
                vc = vm + BETA * (vr - vm)           # outside contraction
            else:
                vc = vm - BETA * (vm - v[vg])        # inside contraction
            fc = func(vc)
            if fc < f[vg]:
                v[vg], f[vg] = vc, fc
            else:                                    # shrink toward best
                for row in range(n + 1):
                    if row != vs:
                        v[row] = v[vs] + (v[row] - v[vs]) / 2.0
                for row in range(n + 1):
                    f[row] = func(v[row])
        fmax = float(np.max(f))
        fmin = float(np.min(f))
        rtol = 2.0 * abs(fmax - fmin) / (abs(fmax) + abs(fmin) + 1.0e-300)
        if rtol < ftol or abs(fmax) < 0.1 * ftol:
            break
    vs = int(np.argmin(f))
    return v[vs].copy(), float(f[vs])


# ---------------------------------------------------------------------------
# prior / range store (parprior)
# ---------------------------------------------------------------------------
@dataclass
class PriorStore:
    """Ranges and Gaussian/log priors for lens / extend / point parameters.

    Keys are ``(i, j)`` 0-based (component, slot).  Defaults reproduce
    init.c:651-737; ``parprior`` lines override.
    """
    range_lens: dict = field(default_factory=dict)
    range_ext: dict = field(default_factory=dict)
    range_poi: dict = field(default_factory=dict)
    prior_lens: dict = field(default_factory=dict)   # (i, j) -> (med, sig)
    prior_ext: dict = field(default_factory=dict)
    prior_poi: dict = field(default_factory=dict)

    @staticmethod
    def _default_range(kind: str, j: int) -> tuple[float, float]:
        table = _INIT_RANGES_POI if kind == "poi" else _INIT_RANGES
        return table[j] if 0 <= j < len(table) else (-1.0e30, 1.0e30)

    def rng(self, kind: str, i: int, j: int) -> tuple[float, float]:
        d = getattr(self, f"range_{kind}")
        return d.get((i, j), self._default_range(kind, j))

    def pri(self, kind: str, i: int, j: int) -> Optional[tuple[float, float]]:
        return getattr(self, f"prior_{kind}").get((i, j))

    def parse_parprior(self, path: str) -> int:
        """glafic parprior file (init.c:934-1030): lines
        ``range lens|extend|point i j min max`` and
        ``prior lens|extend|point i j med sig`` (sig<0 => log10 prior).
        PSF/map lines are rejected (unsupported)."""
        n = 0
        kindmap = {"lens": "lens", "extend": "ext", "point": "poi"}
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                parts = raw.split()
                if not parts or parts[0].startswith("#"):
                    continue
                ptype, keyword = parts[0], (parts[1] if len(parts) > 1 else "")
                if keyword == "psf" or ptype == "map":
                    raise NotImplementedError(
                        "rhongomyniad parprior: psf/map priors are not supported")
                if ptype == "range" and keyword in kindmap:
                    i, j, lo, hi = (int(parts[2]), int(parts[3]),
                                    float(parts[4]), float(parts[5]))
                    getattr(self, f"range_{kindmap[keyword]}")[(i - 1, j - 1)] = (lo, hi)
                    n += 1
                elif ptype == "prior" and keyword in kindmap:
                    i, j, med, sig = (int(parts[2]), int(parts[3]),
                                      float(parts[4]), float(parts[5]))
                    getattr(self, f"prior_{kindmap[keyword]}")[(i - 1, j - 1)] = (med, sig)
                    n += 1
                elif ptype in ("range", "prior"):
                    raise NotImplementedError(
                        f"rhongomyniad parprior: unsupported target '{keyword}'")
        return n


def _gauss_or_log_chi2(value: float, med: float, sig: float) -> float:
    """One prior term (opt_extend.c:429-436 / opt_lens.c chi2prior_lens)."""
    if sig > 0.0:
        return (value - med) ** 2 / (sig * sig)
    if sig < 0.0:
        if value < 0.0 or med < 0.0:
            return CHI2PEN_RANGE
        lp, lm = math.log10(value), math.log10(med)
        return (lp - lm) ** 2 / (sig * sig)
    return 0.0


# ---------------------------------------------------------------------------
# observed data state
# ---------------------------------------------------------------------------
@dataclass
class EGrid:
    """Full grid description (the extend pipeline needs pix_ext, which the
    point-finder GridSpec does not carry)."""
    xmin: float = K.DEF_XMIN
    ymin: float = K.DEF_YMIN
    xmax: float = K.DEF_XMAX
    ymax: float = K.DEF_YMAX
    pix_ext: float = K.DEF_PIX_EXT
    pix_poi: float = K.DEF_PIX_POI
    maxlev: int = K.DEF_MAXLEV


@dataclass
class PointObs:
    """One readobs_point source: header ``id nimg zs [zserr]`` + nimg rows
    ``x y flux pos_err flux_err td td_err parity`` (missing columns = 0)."""
    zs: float
    rows: np.ndarray                      # (nimg, 8)

    @property
    def nimg(self) -> int:
        return int(self.rows.shape[0])


@dataclass
class ExtendData:
    """Everything readobs/readnoise/parprior establish (candidate-independent)."""
    obs: Optional[torch.Tensor] = None            # (ny, nx) float64
    mask: Optional[torch.Tensor] = None           # (ny, nx) bool, True = masked out
    noise_file: Optional[torch.Tensor] = None     # (ny, nx) user noise (flag_obssig)
    noise_computed: Optional[torch.Tensor] = None # (ny, nx) calc_obsnoise result
    skymed: float = 0.0
    skysigma: float = 0.0
    point_obs: list = field(default_factory=list)     # list[PointObs]
    priors: PriorStore = field(default_factory=PriorStore)
    # solved per-source values (para_poi[i][1..4]): set by chi2_opt_point
    solved_src: list = field(default_factory=list)    # [(xs, ys, flux0, td0)]

    def noise(self) -> torch.Tensor:
        return self.noise_file if self.noise_file is not None else self.noise_computed


def grid_nxy(xmin: float, xmax: float, ymin: float, ymax: float,
             pix_ext: float) -> tuple[int, int]:
    """nx_ext / ny_ext (init.c:62-63)."""
    nx = int((xmax - xmin + NPIX_SMALL_OFFSET) / pix_ext)
    ny = int((ymax - ymin + NPIX_SMALL_OFFSET) / pix_ext)
    return nx, ny


def pixel_centers(xmin: float, ymin: float, pix_ext: float, nx: int, ny: int,
                  device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """ktoxy_ext (extend.c:817-827): x = xmin + pix_ext*(ix+0.5)."""
    xs = xmin + pix_ext * (torch.arange(nx, device=device, dtype=dtype) + 0.5)
    ys = ymin + pix_ext * (torch.arange(ny, device=device, dtype=dtype) + 0.5)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")    # (ny, nx)
    return gx, gy


def calc_obsnoise(obs: torch.Tensor, mask: Optional[torch.Tensor],
                  noise_clip: float, obs_gain: float, obs_ncomb: int,
                  skyfix_value: float) -> tuple[torch.Tensor, float, float]:
    """Port of fits.c:778-856.  Returns (noise_map, skymed, skysigma).
    Iterative sigma-clip of unmasked pixels for (med, sig), then
    sigma_k = sig*sqrt(1 + |obs-med| / (sig^2*gain*ncomb)); masked pixels 0."""
    vals = obs[~mask] if mask is not None else obs.reshape(-1)
    a = vals.double()
    med = float(a.mean())
    sig = float(torch.sqrt(torch.clamp((a * a).mean() - med * med, min=0.0)))
    k = a.numel()
    while True:
        ko = k
        sel = (a > (med - noise_clip * sig)) & (a < (med + noise_clip * sig))
        sub = a[sel]
        k = int(sub.numel())
        med = float(sub.mean())
        sig = float(torch.sqrt(torch.clamp((sub * sub).mean() - med * med, min=0.0)))
        if k >= ko:
            break
    if skyfix_value > 0.1 * 1.0e10:      # DEF_SKYFIX_VALUE guard (fits.c:832)
        skyfix_value = med
    skymed = skyfix_value
    x = torch.abs(obs - med) / (sig * sig * (obs_gain * float(obs_ncomb)))
    noise = sig * torch.sqrt(1.0 + x)
    if mask is not None:
        noise = torch.where(mask, torch.zeros_like(noise), noise)
    return noise, skymed, sig


def read_point_obs_file(path: str) -> list[PointObs]:
    """glafic readobs_point format (same file core/optimize/extend.py writes)."""
    out: list[PointObs] = []
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines()]
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        i += 1
        if not raw or raw.startswith("#"):
            continue
        head = raw.split()
        nimg = int(head[1])
        zs = float(head[2]) if len(head) > 2 else 1.0
        rows = np.zeros((nimg, 8), dtype=float)
        for k in range(nimg):
            cols = [float(c) for c in lines[i].split()]
            i += 1
            rows[k, :len(cols[:8])] = cols[:8]
        out.append(PointObs(zs=zs, rows=rows))
    return out


# ---------------------------------------------------------------------------
# extended-image computation (ext_set_table + ext_set_image)
# ---------------------------------------------------------------------------
def deflection_table(sum_lensmodel, ctx, lenses, gx: torch.Tensor, gy: torch.Tensor,
                     mask: Optional[torch.Tensor], smallcore: float):
    """Deflection at every pixel centre (masked pixels forced to 0, like
    extend.c:77-87) and the finite-difference Jacobian (extend.c:90-145).

    Returns (ax, ay, pxx, pxy, pyx, pyy) each (..., ny, nx).  Leading batch
    dims are supported (tensor lens params)."""
    ax, ay, *_ = sum_lensmodel(ctx, lenses, gx, gy, need_kg=False,
                               need_phi=False, smallcore=smallcore)
    if mask is not None:
        zero = torch.zeros((), dtype=ax.dtype, device=ax.device)
        ax = torch.where(mask, zero, ax)
        ay = torch.where(mask, zero, ay)

    pix = float(gx[0, 1] - gx[0, 0]) if gx.shape[-1] > 1 else 1.0

    def _fd(a: torch.Tensor, dim: int) -> torch.Tensor:
        # central difference with one-sided replication at the borders and
        # the matching ddx accumulation (pix per available side)
        if dim == -1:                                  # d/dx
            am = torch.cat([a[..., :, :1], a[..., :, :-1]], dim=-1)
            ap = torch.cat([a[..., :, 1:], a[..., :, -1:]], dim=-1)
            dd = torch.full_like(a, 2.0 * pix)
            dd[..., :, :1] = pix
            dd[..., :, -1:] = pix
        else:                                          # d/dy
            am = torch.cat([a[..., :1, :], a[..., :-1, :]], dim=-2)
            ap = torch.cat([a[..., 1:, :], a[..., -1:, :]], dim=-2)
            dd = torch.full_like(a, 2.0 * pix)
            dd[..., :1, :] = pix
            dd[..., -1:, :] = pix
        return (ap - am) / dd

    # glafic stores phi_xx = dax/dx, phi_xy = day/dx, phi_yx = dax/dy,
    # phi_yy = day/dy (extend.c:135-142; note the ddy/ddx pairing in C is
    # (axxp-axxm)/ddx etc. with the roles as coded there).
    pxx = _fd(ax, -1)
    pxy = _fd(ay, -1)
    pyx = _fd(ax, -2)
    pyy = _fd(ay, -2)
    return ax, ay, pxx, pxy, pyx, pyy


def disratio(cosmo, zl: float, zs_fid: float, zs: float) -> float:
    """distance.c:310-322."""
    if zs_fid <= zl:
        return 0.0
    d1 = cosmo.angulard(zl, zs_fid) / cosmo.angulard(0.0, zs_fid)
    d2 = cosmo.angulard(zl, zs) / cosmo.angulard(0.0, zs)
    return d2 / d1


def ext_model_images(state, sum_lensmodel, LensContext,
                     flag_source: int = 0,
                     include_masked: bool = False) -> torch.Tensor:
    """Port of ext_set_image (extend.c:169-222) for all extend components.

    ``state`` is the api._State (cosmo/grid/lenses/extends/device/dtype/ext).
    Returns the per-source model image stack (num_ext, ny, nx); the chi2 /
    writeimage callers sum over sources and add sky."""
    ext = state.ext
    dev, dt = state.device, state.dtype
    nx, ny = grid_nxy(state.egrid.xmin, state.egrid.xmax,
                      state.egrid.ymin, state.egrid.ymax, state.egrid.pix_ext)
    gx, gy = pixel_centers(state.egrid.xmin, state.egrid.ymin,
                           state.egrid.pix_ext, nx, ny, dev, dt)

    extends = state.extends[:state.num_ext]
    if not extends:
        return torch.zeros((0, ny, nx), dtype=dt, device=dev)

    # fiducial source = highest zs (set_distance_facext, distance.c:284-308)
    zs_list = [float(p[0]) for (_m, p) in extends]
    i_fid = int(np.argmax(zs_list))
    zs_fid = zs_list[i_fid]
    zl = float(state.lenses[0][1][0])
    ctx = LensContext.build(state.cosmo, zl=zl, zs=zs_fid,
                            flag_hodensity=state.flag_hodensity,
                            hodensity=state.hodensity)

    mask = None
    if (not include_masked) and ext.mask is not None:
        mask = ext.mask
    lenses = state.lenses[:state.num_len]
    ax, ay, pxx, pxy, pyx, pyy = deflection_table(
        sum_lensmodel, ctx, lenses, gx, gy, mask, state.smallcore)

    sec = state.secondary
    pix_psf = state.egrid.pix_ext        # calc_pixpsf with flag_seeing=0
    imgs = []
    for (model_name, p) in extends:
        sid = SOURCE_IDS[model_name]
        fac = disratio(state.cosmo, zl, zs_fid, float(p[0]))
        if flag_source == 0:
            sx = gx - ax * fac
            sy = gy - ay * fac
            jxx, jxy, jyx, jyy = pxx * fac, pxy * fac, pyx * fac, pyy * fac
        else:                            # unlensed (writeimage ori=1)
            sx, sy = gx, gy
            z = torch.zeros_like(gx)
            jxx = jxy = jyx = jyy = z
        f = source_all(sid, sx, sy,
                       p[2], p[3], p[4], p[5], p[6], p[7],
                       jxx, jxy, jyx, jyy,
                       pix=pix_psf, pix_ext=state.egrid.pix_ext,
                       imag_ceil=state.imag_ceil,
                       flag_extref=int(sec.get("flag_extref", 1)),
                       source_refr0=float(sec.get("source_refr0", 3.0)),
                       num_pixint=int(sec.get("num_pixint", 5)),
                       flag_extnorm=int(sec.get("flag_extnorm", 0)),
                       smallcore=state.smallcore)
        f = p[1] * f
        if mask is not None:
            f = torch.where(mask, torch.zeros_like(f), f)
        imgs.append(f)
    return torch.stack(imgs, dim=0)


# ---------------------------------------------------------------------------
# range checks & priors
# ---------------------------------------------------------------------------
def check_para_lens_all(state) -> bool:
    """opt_lens.c:577-595 (True = violation -> chi2pen_range).  Covers every
    lens slot, extend zs, point zs and the cosmology box."""
    pri = state.ext.priors
    for i, (_m, p) in enumerate(state.lenses[:state.num_len]):
        for j in range(8):
            lo, hi = pri.rng("lens", i, j)
            if p[j] < lo or p[j] > hi:
                return True
    for i, (_m, p) in enumerate(state.extends[:state.num_ext]):
        lo, hi = pri.rng("ext", i, 0)
        if p[0] < lo or p[0] > hi:
            return True
    for i, po in enumerate(state.ext.point_obs):
        lo, hi = pri.rng("poi", i, 0)
        if po.zs < lo or po.zs > hi:
            return True
    cos = state.cosmo
    for key, val in (("omega", cos.omega), ("lambda", cos.lam),
                     ("weos", cos.weos), ("hubble", cos.hubble)):
        lo, hi = _COSMO_RANGES[key]
        if val < lo or val > hi:
            return True
    return False


def check_para_ext_all(state) -> bool:
    """opt_extend.c:401-419 (slots j=1..7 of every extend component)."""
    pri = state.ext.priors
    for i, (_m, p) in enumerate(state.extends[:state.num_ext]):
        for j in range(1, 8):
            lo, hi = pri.rng("ext", i, j)
            if p[j] < lo or p[j] > hi:
                return True
    return False


def chi2prior_ext(state) -> float:
    """opt_extend.c:421-484 (no PSF terms; flag_seeing=0)."""
    pri = state.ext.priors
    c2 = 0.0
    for i, (_m, p) in enumerate(state.extends[:state.num_ext]):
        for j in range(1, 8):
            pr = pri.pri("ext", i, j)
            if pr is not None:
                term = _gauss_or_log_chi2(float(p[j]), pr[0], pr[1])
                if term >= CHI2PEN_RANGE:
                    return CHI2PEN_RANGE
                c2 += term
    return c2


def chi2prior_lens(state) -> float:
    """opt_lens.c:597-657 Gaussian/log terms (parprior 'prior lens')."""
    pri = state.ext.priors
    c2 = 0.0
    for i, (_m, p) in enumerate(state.lenses[:state.num_len]):
        for j in range(8):
            pr = pri.pri("lens", i, j)
            if pr is not None:
                term = _gauss_or_log_chi2(float(p[j]), pr[0], pr[1])
                if term >= CHI2PEN_RANGE:
                    return CHI2PEN_RANGE
                c2 += term
    return c2


def chi2prior_point(state, i: int, xs: float, ys: float) -> float:
    """opt_point.c:696-709."""
    pri = state.ext.priors
    c2 = 0.0
    for j, val in ((1, xs), (2, ys)):
        pr = pri.pri("poi", i, j)
        if pr is not None and pr[1] > 0.0:
            c2 += (pr[0] - val) ** 2 / (pr[1] * pr[1])
    return c2


# ---------------------------------------------------------------------------
# extended-source pixel chi2 (chi2calc_extend)
# ---------------------------------------------------------------------------
def chi2_extend(state, sum_lensmodel, LensContext) -> tuple[float, float, float]:
    """Returns (total, pixel_chi2, prior_chi2); mirrors opt_extend.c:319-361."""
    ext = state.ext
    if ext.obs is None:
        raise RuntimeError("readobs_extend must be called before c2calc")
    if check_para_ext_all(state):
        return CHI2PEN_RANGE, CHI2PEN_RANGE, 0.0

    imgs = ext_model_images(state, sum_lensmodel, LensContext)
    model = imgs.sum(dim=0)

    sec = state.secondary
    skymed = ext.skymed
    if int(sec.get("skyfix", 0)) == 1:
        skymed = float(sec.get("skyfix_value", ext.skymed))
    model = model + skymed

    noise = ext.noise()
    keep = ~ext.mask if ext.mask is not None else torch.ones_like(
        ext.obs, dtype=torch.bool)
    diff = (model - ext.obs)[keep]
    sig = noise[keep]
    pixel = float(((diff * diff) / (sig * sig)).sum())

    prior = chi2prior_ext(state)
    if prior >= CHI2PEN_RANGE:
        return CHI2PEN_RANGE, pixel, CHI2PEN_RANGE
    return pixel + prior, pixel, prior


# ---------------------------------------------------------------------------
# point-source chi2 (chi2calc_opt_point port)
# ---------------------------------------------------------------------------
def _lensmodel_at(state, sum_lensmodel, ctx, xs: np.ndarray, ys: np.ndarray,
                  need_phi: bool):
    """calcimage at a few points; returns numpy arrays
    (ax, ay, td, kap, g1, g2, muinv)."""
    dev, dt = state.device, state.dtype
    tx = torch.tensor(xs, dtype=dt, device=dev)
    ty = torch.tensor(ys, dtype=dt, device=dev)
    lenses = state.lenses[:state.num_len]
    ax, ay, kap, g1, g2, phi, muinv = sum_lensmodel(
        ctx, lenses, tx, ty, need_kg=True, need_phi=need_phi,
        smallcore=state.smallcore)
    ax = ax.cpu().numpy()
    ay = ay.cpu().numpy()
    kap = kap.cpu().numpy()
    g1 = g1.cpu().numpy()
    g2 = g2.cpu().numpy()
    muinv = muinv.cpu().numpy()
    if need_phi and phi is not None:
        td = ctx.tdelay_fac * (0.5 * (ax * ax + ay * ay) - phi.cpu().numpy())
    else:
        td = np.zeros_like(ax)
    return ax, ay, td, kap, g1, g2, muinv


def _set_matrix(kap, g1, g2, muinv, imag_ceil):
    """opt_point.c:549-565 (rot = 0, single plane).  Vectorised over images."""
    norm = 1.0 / (muinv + imag_ceil)
    n2 = norm * norm
    a00 = n2 * ((1.0 - kap + g1) ** 2 + g2 * g2)
    a01 = 2.0 * n2 * (g2 * (1.0 - kap))
    a10 = a01.copy() if isinstance(a01, np.ndarray) else a01
    a11 = n2 * ((1.0 - kap - g1) ** 2 + g2 * g2)
    mu00 = norm * (1.0 - kap + g1)
    mu01 = norm * g2
    mu10 = norm * g2
    mu11 = norm * (1.0 - kap - g1)
    return (a00, a01, a10, a11), (mu00, mu01, mu10, mu11)


def _dp_lev(pix_poi: float, lev: int) -> float:
    return pix_poi * (0.5 ** lev)


class _SplaneCache:
    """Per-(lens-state, source) cached quantities for the source-plane chi2
    (the flag_scalc block, opt_point.c:435-486)."""

    def __init__(self, state, sum_lensmodel, LensContext, po: PointObs, i: int):
        zl = float(state.lenses[0][1][0])
        ctx = LensContext.build(state.cosmo, zl=zl, zs=po.zs,
                                flag_hodensity=state.flag_hodensity,
                                hodensity=state.hodensity)
        self.ctx = ctx
        obs_x = po.rows[:, 0].copy()
        obs_y = po.rows[:, 1].copy()
        hh = _dp_lev(state.egrid.pix_poi, state.egrid.maxlev - 1) * 0.1

        ax, ay, td, kap, g1, g2, muinv = _lensmodel_at(
            state, sum_lensmodel, ctx, obs_x, obs_y, need_phi=True)
        self.a_mat, self.mu_mat = _set_matrix(kap, g1, g2, muinv, state.imag_ceil)
        self.uobs_x = obs_x - ax
        self.uobs_y = obs_y - ay
        self.mag = 1.0 / (muinv + state.imag_ceil)      # rr[k][2]
        self.td = td                                    # rr[k][3]
        # dmu/dx, dmu/dy by central FD of muinv (opt_point.c:447-457)
        _, _, _, _, _, _, m1 = _lensmodel_at(state, sum_lensmodel, ctx,
                                             obs_x + 0.5 * hh, obs_y, need_phi=False)
        _, _, _, _, _, _, m2 = _lensmodel_at(state, sum_lensmodel, ctx,
                                             obs_x - 0.5 * hh, obs_y, need_phi=False)
        self.dmudx = -self.mag * self.mag * (m1 - m2) / hh
        _, _, _, _, _, _, m1 = _lensmodel_at(state, sum_lensmodel, ctx,
                                             obs_x, obs_y + 0.5 * hh, need_phi=False)
        _, _, _, _, _, _, m2 = _lensmodel_at(state, sum_lensmodel, ctx,
                                             obs_x, obs_y - 0.5 * hh, need_phi=False)
        self.dmudy = -self.mag * self.mag * (m1 - m2) / hh
        self.tdelay_fac = ctx.tdelay_fac
        self.obs_x, self.obs_y = obs_x, obs_y

        # closed-form best source position (opt_point.c:460-486)
        a00, a01, a10, a11 = self.a_mat
        perr = po.rows[:, 3]
        w = np.where(perr > 0.0, 1.0 / np.where(perr > 0.0, perr, 1.0) ** 2, 0.0)
        AA = np.zeros((2, 2))
        BB = np.zeros(2)
        AA[0, 0] = float((a00 * w).sum())
        AA[0, 1] = float((a01 * w).sum())
        AA[1, 0] = float((a10 * w).sum())
        AA[1, 1] = float((a11 * w).sum())
        BB[0] = float(((a00 * self.uobs_x + a01 * self.uobs_y) * w).sum())
        BB[1] = float(((a10 * self.uobs_x + a11 * self.uobs_y) * w).sum())
        pri = state.ext.priors
        for j, idx in ((1, 0), (2, 1)):
            pr = pri.pri("poi", i, j)
            if pr is not None and pr[1] > 0.0:
                AA[idx, idx] += 1.0 / (pr[1] * pr[1])
                BB[idx] += pr[0] / (pr[1] * pr[1])
        det = AA[0, 0] * AA[1, 1] - AA[0, 1] * AA[1, 0]
        self.umod = np.array([
            (AA[1, 1] * BB[0] - AA[0, 1] * BB[1]) / det,
            (AA[0, 0] * BB[1] - AA[1, 0] * BB[0]) / det])


def _chi2_point_splane_inner(state, cache: _SplaneCache, po: PointObs, i: int,
                             xs: float, ys: float) -> np.ndarray:
    """opt_point.c:407-547 given the cache.  Returns c2[5] =
    [tot, pos, flux, td, prior] and stores flux/td zero-points on the cache."""
    sec = state.secondary
    usemag = int(sec.get("chi2_usemag", 0))
    c2 = np.zeros(5)
    lo1, hi1 = state.ext.priors.rng("poi", i, 1)
    lo2, hi2 = state.ext.priors.rng("poi", i, 2)
    if xs < lo1 or xs > hi1 or ys < lo2 or ys > hi2:
        c2[0] = c2[4] = CHI2PEN_RANGE
        return c2

    a00, a01, a10, a11 = cache.a_mat
    mu00, mu01, mu10, mu11 = cache.mu_mat
    ux = cache.uobs_x - xs
    uy = cache.uobs_y - ys
    perr = po.rows[:, 3]
    use_pos = perr > 0.0
    cc = (a00 * ux * ux + (a01 + a10) * ux * uy + a11 * uy * uy)
    cc = np.where(use_pos, cc / np.where(use_pos, perr, 1.0) ** 2, 0.0)
    cc = np.where(cc < 0.0, CHI2PEN_NIMG, cc)
    c2[1] = float(cc.sum())

    dx = mu00 * ux + mu01 * uy
    dy = mu10 * ux + mu11 * uy
    mumod = cache.mag - (cache.dmudx * dx + cache.dmudy * dy)
    # single plane: def_lpl[0] = 0 (mass.c:109-110), so the td linearisation is
    # td - tdelay_fac * ((uobs - obs) . (uobs - s))  (opt_point.c:502-504)
    tdmod = cache.td - cache.tdelay_fac * (
        (cache.uobs_x - cache.obs_x) * ux + (cache.uobs_y - cache.obs_y) * uy)

    flux = po.rows[:, 2]
    ferr = po.rows[:, 4]
    tdo = po.rows[:, 5]
    tderr = po.rows[:, 6]
    parity = po.rows[:, 7]

    f1 = f2 = 0.0
    t1 = t2 = 0.0
    for k in range(po.nimg):
        if ferr[k] > 0.0:
            if usemag == 0:
                f1 += abs(flux[k] * mumod[k]) / (ferr[k] * ferr[k])
                f2 += (mumod[k] * mumod[k]) / (ferr[k] * ferr[k])
            else:
                f1 += (flux[k] + 2.5 * math.log10(abs(mumod[k]))) / (ferr[k] * ferr[k])
                f2 += 1.0 / (ferr[k] * ferr[k])
        if tderr[k] > 0.0:
            t1 += (tdo[k] - tdmod[k]) / (tderr[k] * tderr[k])
            t2 += 1.0 / (tderr[k] * tderr[k])
    flux0 = f1 / f2 if f2 > 0.0 else 1.0
    td0 = t1 / t2 if t2 > 0.0 else 0.0
    cache.flux0, cache.td0 = flux0, td0

    fp = False
    for k in range(po.nimg):
        if parity[k] != 0.0 and parity[k] * mumod[k] < 0.0:
            fp = True
        if ferr[k] > 0.0:
            if usemag == 0:
                c2[2] += (abs(flux[k]) - abs(mumod[k]) * flux0) ** 2 / (ferr[k] * ferr[k])
            elif usemag == -1:
                c2[2] += (abs(flux[k]) - abs(mumod[k])) ** 2 / (ferr[k] * ferr[k])
            else:
                c2[2] += (flux[k] + 2.5 * math.log10(abs(mumod[k])) - flux0) ** 2 \
                    / (ferr[k] * ferr[k])
        if tderr[k] > 0.0:
            c2[3] += (tdo[k] - tdmod[k] - td0) ** 2 / (tderr[k] * tderr[k])
    if fp:
        c2[2] = CHI2PEN_PARITY
    c2[4] = chi2prior_point(state, i, xs, ys)
    c2[0] = c2[1] + c2[2] + c2[3] + c2[4]
    return c2


def _chi2_point_iplane_inner(state, point_solve, po: PointObs, i: int,
                             xs: float, ys: float) -> np.ndarray:
    """opt_point.c:199-292: findimg-based image-plane chi2 (one evaluation)."""
    sec = state.secondary
    usemag = int(sec.get("chi2_usemag", 0))
    checknimg = int(sec.get("chi2_checknimg", 1))
    c2 = np.zeros(5)
    lo1, hi1 = state.ext.priors.rng("poi", i, 1)
    lo2, hi2 = state.ext.priors.rng("poi", i, 2)
    if xs < lo1 or xs > hi1 or ys < lo2 or ys > hi2:
        c2[0] = c2[4] = CHI2PEN_RANGE
        return c2

    imgs = point_solve(po.zs, xs, ys, verb=0)
    ni = len(imgs)
    nobs = po.nimg
    if (ni != nobs and checknimg == 1) or ni < nobs:
        c2[0] = c2[1] = CHI2PEN_NIMG
        return c2

    rr = np.asarray(imgs, dtype=float)         # (ni, 4): x y mag td
    used = np.zeros(ni, dtype=bool)
    kk = np.zeros(nobs, dtype=int)
    flux = po.rows[:, 2]
    perr = po.rows[:, 3]
    ferr = po.rows[:, 4]
    tdo = po.rows[:, 5]
    tderr = po.rows[:, 6]
    parity = po.rows[:, 7]

    f1 = f2 = 0.0
    t1 = t2 = 0.0
    for j in range(nobs):
        dm = DIS2_MIN_SET
        fp = False
        for k in range(ni):
            if used[k]:
                continue
            if parity[j] == 0.0 or parity[j] * rr[k, 2] > 0.0:
                dis2 = (rr[k, 0] - po.rows[j, 0]) ** 2 + (rr[k, 1] - po.rows[j, 1]) ** 2
                if dis2 < dm:
                    dm, kk[j], fp = dis2, k, True
        if not fp:
            c2[0] = c2[1] = CHI2PEN_NIMG
            return c2
        used[kk[j]] = True
        if perr[j] > 0.0:
            c2[1] += dm / (perr[j] * perr[j])
        if ferr[j] > 0.0:
            if usemag == 0:
                f1 += abs(flux[j] * rr[kk[j], 2]) / (ferr[j] * ferr[j])
                f2 += (rr[kk[j], 2] ** 2) / (ferr[j] * ferr[j])
            else:
                f1 += (flux[j] + 2.5 * math.log10(abs(rr[kk[j], 2]))) / (ferr[j] * ferr[j])
                f2 += 1.0 / (ferr[j] * ferr[j])
        if tderr[j] > 0.0:
            t1 += (tdo[j] - rr[kk[j], 3]) / (tderr[j] * tderr[j])
            t2 += 1.0 / (tderr[j] * tderr[j])

    flux0 = f1 / f2 if f2 > 0.0 else 1.0
    td0 = t1 / t2 if t2 > 0.0 else 0.0

    fp = False
    for j in range(nobs):
        if parity[j] != 0.0 and parity[j] * rr[kk[j], 2] < 0.0:
            fp = True
        if ferr[j] > 0.0:
            if usemag == 0:
                c2[2] += (abs(flux[j]) - abs(rr[kk[j], 2]) * flux0) ** 2 / (ferr[j] * ferr[j])
            elif usemag == -1:
                c2[2] += (abs(flux[j]) - abs(rr[kk[j], 2])) ** 2 / (ferr[j] * ferr[j])
            else:
                c2[2] += (flux[j] + 2.5 * math.log10(abs(rr[kk[j], 2])) - flux0) ** 2 \
                    / (ferr[j] * ferr[j])
        if tderr[j] > 0.0:
            c2[3] += (tdo[j] - rr[kk[j], 3] - td0) ** 2 / (tderr[j] * tderr[j])
    if fp:
        c2[2] = CHI2PEN_PARITY
    c2[4] = chi2prior_point(state, i, xs, ys)
    c2[0] = c2[1] + c2[2] + c2[3] + c2[4]
    return c2


def chi2_opt_point(state, sum_lensmodel, LensContext, point_solve) -> np.ndarray:
    """Port of chi2calc_opt_point (opt_point.c:24-56 + both drivers).

    Solves each source's free (x, y) with the glafic simplex, writes the
    solved positions back into ``state.points`` (so findimg_i sees them, as
    glafic's para_poi does) and returns the summed components
    [tot, pos, flux, td, prior]."""
    out = np.zeros(5)
    state.ext.solved_src = []
    for i, po in enumerate(state.ext.point_obs):
        flag = state.opt_point.get(i + 1, (0, 0, 0))
        free_x, free_y = int(flag[1]), int(flag[2])
        nd = free_x + free_y
        init_x, init_y = state.points[i][1], state.points[i][2]
        splane = int(state.secondary.get("chi2_splane", 0)) != 0

        cache = (_SplaneCache(state, sum_lensmodel, LensContext, po, i)
                 if splane else None)
        if splane:
            def feval(xs, ys):
                return _chi2_point_splane_inner(state, cache, po, i, xs, ys)
        else:
            def feval(xs, ys):
                return _chi2_point_iplane_inner(state, point_solve, po, i, xs, ys)

        xs_best, ys_best = init_x, init_y
        if nd > 0 and po.nimg >= 2:
            if splane:
                # start at the closed-form solution (opt_point.c:341-377)
                hh = _dp_lev(state.egrid.pix_poi, state.egrid.maxlev - 1)
                sx0, sy0 = float(cache.umod[0]), float(cache.umod[1])
                starts = []
                if nd == 2:
                    starts.append(np.array([[sx0, sy0],
                                            [sx0 + hh, sy0],
                                            [sx0, sy0 + hh]]))
                elif free_x:
                    starts.append(np.array([[sx0], [sx0 + hh]]))
                else:
                    starts.append(np.array([[sy0], [sy0 + hh]]))
            else:
                # back-traced obs + midpoints as seed groups (opt_point.c:127-160)
                zl = float(state.lenses[0][1][0])
                ctx = LensContext.build(state.cosmo, zl=zl, zs=po.zs,
                                        flag_hodensity=state.flag_hodensity,
                                        hodensity=state.hodensity)
                ax, ay, *_ = _lensmodel_at(state, sum_lensmodel, ctx,
                                           po.rows[:, 0], po.rows[:, 1],
                                           need_phi=False)
                bx = po.rows[:, 0] - ax
                by = po.rows[:, 1] - ay
                xs_seeds = list(bx) + [0.5 * (bx[k - 1] + bx[k]) for k in range(1, po.nimg)]
                ys_seeds = list(by) + [0.5 * (by[k - 1] + by[k]) for k in range(1, po.nimg)]
                starts = []
                for k in range((po.nimg - 1) // (nd + 1) + 1):
                    base = k * (nd + 1)
                    if base + nd > len(xs_seeds) - 1:
                        break
                    if nd == 2:
                        sl = slice(base, base + 3)
                        starts.append(np.column_stack([xs_seeds[sl], ys_seeds[sl]]))
                    elif free_x:
                        starts.append(np.array([[xs_seeds[base]], [xs_seeds[base + 1]]]))
                    else:
                        starts.append(np.array([[ys_seeds[base]], [ys_seeds[base + 1]]]))

            def packed(par):
                if nd == 2:
                    return feval(float(par[0]), float(par[1]))[0]
                if free_x:
                    return feval(float(par[0]), init_y)[0]
                return feval(init_x, float(par[0]))[0]

            best_v, best_f = None, CHI2_MIN_SET
            for v0 in starts:
                fvals = np.array([packed(v0[r]) for r in range(v0.shape[0])])
                v_fin, f_fin = nm_simplex(v0.copy(), fvals, TOL_AMOEBA,
                                          packed, NMAX_AMOEBA_POINT)
                if f_fin < best_f:
                    best_f, best_v = f_fin, v_fin
            if best_v is not None:
                if nd == 2:
                    xs_best, ys_best = float(best_v[0]), float(best_v[1])
                elif free_x:
                    xs_best = float(best_v[0])
                else:
                    ys_best = float(best_v[0])
        elif po.nimg == 1:
            # single image: back-trace it (opt_point.c:183-187 / 391-395)
            zl = float(state.lenses[0][1][0])
            ctx = LensContext.build(state.cosmo, zl=zl, zs=po.zs,
                                    flag_hodensity=state.flag_hodensity,
                                    hodensity=state.hodensity)
            ax, ay, *_ = _lensmodel_at(state, sum_lensmodel, ctx,
                                       po.rows[:1, 0], po.rows[:1, 1],
                                       need_phi=False)
            if free_x:
                xs_best = float(po.rows[0, 0] - ax[0])
            if free_y:
                ys_best = float(po.rows[0, 1] - ay[0])

        c2 = feval(xs_best, ys_best)
        state.points[i] = (po.zs, xs_best, ys_best)
        flux0 = getattr(cache, "flux0", 1.0) if cache is not None else 1.0
        td0 = getattr(cache, "td0", 0.0) if cache is not None else 0.0
        state.ext.solved_src.append((xs_best, ys_best, flux0, td0))
        out += c2
    return out


# ---------------------------------------------------------------------------
# c2calc_each (chi2tot_each port, opt_lens.c:337-374)
# ---------------------------------------------------------------------------
def c2calc_each_impl(state, sum_lensmodel, LensContext, point_solve) -> tuple:
    out = np.zeros(8)
    if check_para_lens_all(state):
        out[7] = CHI2PEN_RANGE
        return tuple(out)

    do_ext = state.num_ext > 0 and state.ext.obs is not None
    do_poi = len(state.ext.point_obs) > 0

    if do_ext:
        if check_para_ext_all(state):
            out[7] = CHI2PEN_RANGE
            return tuple(out)
        _tot, pixel, prior_e = chi2_extend(state, sum_lensmodel, LensContext)
        out[4] = pixel
        out[5] = prior_e

    if do_poi:
        c2p = chi2_opt_point(state, sum_lensmodel, LensContext, point_solve)
        out[0], out[1], out[2], out[3] = c2p[1], c2p[2], c2p[3], c2p[4]

    out[6] = chi2prior_lens(state)
    return tuple(out)
