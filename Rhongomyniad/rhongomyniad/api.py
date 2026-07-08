"""
Public Rhongomyniad API.

Function names and signatures mirror the glafic Python module
(`glade/glafic2/python/glafic/__init__.py`) so that code written against
glafic can run on Rhongomyniad with minimal changes.

Implemented (v1):
    init, set_cosmo, quit, startup_setnum, set_lens, set_point,
    model_init, calcimage, point_solve, findimg_i, findimg,
    get_device, set_device, supported_models

Extended-source support (v2, glafic-API drop-in for GLADE's extend path):
    set_extend, set_secondary, setopt_point, readobs_extend,
    readnoise_extend, readobs_point, parprior, c2calc, c2calc_each,
    get_nxy_ext, writeimage, getpar_extend, getpar_sky
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import math

import numpy as np
import torch

from . import constants as K
from .cosmology import Cosmology
from .extend import (EGrid, ExtendData, SECONDARY_DEFAULTS, c2calc_each_impl,
                     calc_obsnoise as _calc_obsnoise, ext_model_images,
                     grid_nxy, read_point_obs_file)
from .image_finder import GridSpec, findimg as _findimg_uniform, sum_lensmodel
from .image_finder_adaptive import findimg_adaptive as _findimg_adaptive
from .lens_models import (LensContext, supported_models as _supported_models,
                          ensure_gals_loaded as _ensure_gals_loaded,
                          set_galfile, set_gals, readgals, num_gals, clear_gals)


# ---------------------------------------------------------------------------
# Global state (mirrors glafic's globals in glafic.h + python.c).
# ---------------------------------------------------------------------------
@dataclass
class _State:
    cosmo: Cosmology = field(default_factory=Cosmology)
    file_prefix: str = "out"
    grid: GridSpec = field(default_factory=lambda: GridSpec(
        K.DEF_XMIN, K.DEF_YMIN, K.DEF_XMAX, K.DEF_YMAX,
        K.DEF_PIX_POI, K.DEF_MAXLEV))

    num_len: int = 0
    num_poi: int = 0
    num_ext: int = 0
    # lenses[i] = (model_name, params_tuple_of_8)  for i in 0..num_len-1
    lenses: list[tuple[str, tuple[float, float, float, float,
                                  float, float, float, float]]] = field(default_factory=list)
    # points[i] = (zs, xs, ys)
    points: list[tuple[float, float, float]] = field(default_factory=list)
    # extends[i] = (source_model_name, params_tuple_of_8: zs, norm, x, y, e, pa, r0, n)
    extends: list[tuple[str, tuple]] = field(default_factory=list)
    # setopt_point flags: 1-based id -> (flag_z, flag_x, flag_y)
    opt_point: dict = field(default_factory=dict)
    # set_secondary knobs (chi2_splane, obs_gain, ...)
    secondary: dict = field(default_factory=lambda: dict(SECONDARY_DEFAULTS))
    # observed extended/point data + priors (readobs_*, parprior)
    ext: ExtendData = field(default_factory=ExtendData)
    # full grid incl. pix_ext (the extend pipeline needs it)
    egrid: EGrid = field(default_factory=EGrid)

    initialised_model: bool = False
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))
    dtype: torch.dtype = torch.float64

    # Image-finder runtime parameters.
    max_poi_tol: float = K.DEF_MAX_POI_TOL
    nmax_poi_ite: int = K.DEF_NMAX_POI_ITE
    imag_ceil: float = K.DEF_IMAG_CEIL
    smallcore: float = K.DEF_SMALLCORE
    nfw_users: int = K.DEF_NFW_USERS
    flag_hodensity: int = K.DEF_FLAG_HODENSITY
    hodensity: float = K.DEF_HODENSITY
    poi_imag_max: float = K.DEF_POI_IMAG_MAX
    poi_imag_min: float = K.DEF_POI_IMAG_MIN
    # Image-finder backend: "adaptive" (default, matches glafic's quad-tree)
    # or "uniform" (dense fine grid — keeps the v1 fallback for reference).
    finder: str = "adaptive"


_STATE = _State()


def _ensure_lens_slots(n: int) -> None:
    while len(_STATE.lenses) < n:
        _STATE.lenses.append(("point", (0.0,) * 8))


def _ensure_point_slots(n: int) -> None:
    while len(_STATE.points) < n:
        _STATE.points.append((0.0, 0.0, 0.0))


def _ensure_extend_slots(n: int) -> None:
    while len(_STATE.extends) < n:
        _STATE.extends.append(("gauss", (0.0,) * 8))


# ---------------------------------------------------------------------------
# Device / dtype management
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    return _STATE.device


def set_device(device: torch.device | str) -> None:
    _STATE.device = torch.device(device)


def set_dtype(dtype: torch.dtype) -> None:
    _STATE.dtype = dtype


def supported_models() -> tuple[str, ...]:
    return _supported_models()


# ---------------------------------------------------------------------------
# glafic.init / glafic.set_cosmo
# ---------------------------------------------------------------------------
def init(omega: float, lam: float, weos: float, hubble: float,
         prefix: str = "out",
         xmin: float = K.DEF_XMIN, ymin: float = K.DEF_YMIN,
         xmax: float = K.DEF_XMAX, ymax: float = K.DEF_YMAX,
         pix_ext: float = K.DEF_PIX_EXT, pix_poi: float = K.DEF_PIX_POI,
         maxlev: int = K.DEF_MAXLEV,
         ran_seed: int = 0, verb: int = 0) -> None:
    """Reset state and set primary parameters (glafic_init)."""
    _STATE.cosmo = Cosmology(omega=omega, lam=lam, weos=weos, hubble=hubble)
    _STATE.file_prefix = prefix
    _STATE.grid = GridSpec(xmin, ymin, xmax, ymax, pix_poi, maxlev)
    _STATE.egrid = EGrid(xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev)
    _STATE.num_len = 0
    _STATE.num_poi = 0
    _STATE.num_ext = 0
    _STATE.lenses = []
    _STATE.points = []
    _STATE.extends = []
    _STATE.opt_point = {}
    _STATE.secondary = dict(SECONDARY_DEFAULTS)
    _STATE.ext = ExtendData()
    _STATE.initialised_model = False
    if verb:
        print(f"[rhongomyniad] init: device={_STATE.device}, grid={(xmin, ymin, xmax, ymax)}, "
              f"pix_poi={pix_poi}, maxlev={maxlev}")


def set_cosmo(omega: float, lam: float, weos: float, hubble: float) -> None:
    _STATE.cosmo = Cosmology(omega=omega, lam=lam, weos=weos, hubble=hubble)
    _STATE.initialised_model = False


def set_primary(omega: float, lam: float, weos: float, hubble: float,
                prefix: str, xmin: float, ymin: float, xmax: float, ymax: float,
                pix_ext: float, pix_poi: float, maxlev: int,
                verb: int = 0) -> None:
    """glafic_set_primary: like init but does not clear existing lenses."""
    _STATE.cosmo = Cosmology(omega=omega, lam=lam, weos=weos, hubble=hubble)
    _STATE.file_prefix = prefix
    _STATE.grid = GridSpec(xmin, ymin, xmax, ymax, pix_poi, maxlev)
    _STATE.egrid = EGrid(xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev)
    _STATE.initialised_model = False


def quit() -> None:
    """Release state (no-op on the GPU side; matches glafic_quit signature)."""
    _STATE.num_len = 0
    _STATE.num_poi = 0
    _STATE.num_ext = 0
    _STATE.lenses = []
    _STATE.points = []
    _STATE.extends = []
    _STATE.opt_point = {}
    _STATE.secondary = dict(SECONDARY_DEFAULTS)
    _STATE.ext = ExtendData()
    _STATE.initialised_model = False


# ---------------------------------------------------------------------------
# Lens / point setup (glafic.startup_setnum, set_lens, set_point)
# ---------------------------------------------------------------------------
def startup_setnum(num_len: int, num_ext: int, num_poi: int) -> None:
    _STATE.num_len = num_len
    _STATE.num_poi = num_poi
    _STATE.num_ext = num_ext
    _STATE.lenses = []
    _STATE.points = []
    _STATE.extends = []
    _ensure_lens_slots(num_len)
    _ensure_point_slots(num_poi)
    _ensure_extend_slots(num_ext)


def set_lens(i: int, model: str,
             p1: float, p2: float, p3: float, p4: float,
             p5: float, p6: float, p7: float, p8: float) -> None:
    """
    Register a lens by 1-based index `i` (matches glafic's `set_lens`).
    `model` is one of the names returned by `supported_models()`.
    p1..p8 are the usual 8 parameters p[0]..p[7] in glafic's ordering.
    """
    if model not in _supported_models():
        raise NotImplementedError(
            f"lens model '{model}' is not supported in this Rhongomyniad build; "
            f"available: {sorted(_supported_models())}")
    if i < 1 or i > K.NMAX_LEN:
        raise ValueError(f"lens id {i} out of range")
    if model == "gals":
        _ensure_gals_loaded()  # lazy readgals on first use (glafic call.c:137)
    _ensure_lens_slots(i)
    _STATE.lenses[i - 1] = (model, (p1, p2, p3, p4, p5, p6, p7, p8))
    if i > _STATE.num_len:
        _STATE.num_len = i
    _STATE.initialised_model = False


def set_point(i: int, zs: float, xs: float, ys: float) -> None:
    if i < 1 or i > K.NMAX_POI:
        raise ValueError(f"point id {i} out of range")
    _ensure_point_slots(i)
    _STATE.points[i - 1] = (zs, xs, ys)
    if i > _STATE.num_poi:
        _STATE.num_poi = i


def model_init(verb: int = 0) -> None:
    """
    Finalise the lens setup so image queries are ready.  For single lens
    plane mode this validates that all lenses share (roughly) one redshift.
    """
    if _STATE.num_len == 0:
        raise RuntimeError("model_init: no lenses registered")
    zls = [lens[1][0] for lens in _STATE.lenses[:_STATE.num_len]]
    # Single-plane check: all zl differ by < TOL_ZS.
    zl_min = min(zls)
    zl_max = max(zls)
    if zl_max - zl_min > K.TOL_ZS:
        raise NotImplementedError(
            "Rhongomyniad v1 only supports a single lens plane. "
            f"found lens redshifts in [{zl_min}, {zl_max}]")
    _STATE.initialised_model = True
    if verb:
        print(f"[rhongomyniad] model_init: {_STATE.num_len} lenses at zl≈{zl_min}")


def _build_context(zs: float) -> LensContext:
    if not _STATE.initialised_model:
        raise RuntimeError("call model_init() before computing images")
    zl = _STATE.lenses[0][1][0]
    return LensContext.build(_STATE.cosmo, zl=zl, zs=zs,
                             flag_hodensity=_STATE.flag_hodensity,
                             hodensity=_STATE.hodensity)


# ---------------------------------------------------------------------------
# Lens properties at a single point (glafic.calcimage)
# ---------------------------------------------------------------------------
def calcimage(zs: float, x: float, y: float,
              alponly: int = -1, verb: int = 0) -> tuple[float, ...]:
    """
    Returns the 8-tuple (ax, ay, td, kap, g1, g2, muinv, rot) matching
    glafic's python_calcimage output order.  `rot` is always 0 for
    single-plane mode.
    """
    ctx = _build_context(zs)
    tx = torch.tensor([x], dtype=_STATE.dtype, device=_STATE.device)
    ty = torch.tensor([y], dtype=_STATE.dtype, device=_STATE.device)
    lenses = _STATE.lenses[:_STATE.num_len]
    need_kg = alponly != 1
    need_phi = alponly < 0
    ax, ay, kap, g1, g2, phi, muinv = sum_lensmodel(
        ctx, lenses, tx, ty,
        need_kg=need_kg, need_phi=need_phi, smallcore=_STATE.smallcore)

    ax_v = float(ax.item())
    ay_v = float(ay.item())
    if need_kg:
        kap_v = float(kap.item())
        g1_v = float(g1.item())
        g2_v = float(g2.item())
        muinv_v = float(muinv.item())
    else:
        kap_v = g1_v = g2_v = muinv_v = 0.0
    td_v = 0.0
    if need_phi and phi is not None:
        td_v = ctx.tdelay_fac * (0.5 * (ax_v * ax_v + ay_v * ay_v) - float(phi.item()))
    rot_v = 0.0  # single-plane
    return (ax_v, ay_v, td_v, kap_v, g1_v, g2_v, muinv_v, rot_v)


# ---------------------------------------------------------------------------
# Lens equation solve for a point source (glafic.point_solve / findimg_i)
# ---------------------------------------------------------------------------
def point_solve(zs: float, x: float, y: float,
                verb: int = 0) -> list[tuple[float, float, float, float]]:
    ctx = _build_context(zs)
    lenses = _STATE.lenses[:_STATE.num_len]
    if _STATE.finder == "adaptive":
        images = _findimg_adaptive(
            ctx, lenses, float(x), float(y), _STATE.grid,
            device=_STATE.device, dtype=_STATE.dtype,
            max_iter=_STATE.nmax_poi_ite, tol=_STATE.max_poi_tol,
            imag_ceil=_STATE.imag_ceil, smallcore=_STATE.smallcore,
            poi_imag_max=_STATE.poi_imag_max, poi_imag_min=_STATE.poi_imag_min)
    else:
        images = _findimg_uniform(
            ctx, lenses, float(x), float(y), _STATE.grid,
            device=_STATE.device, dtype=_STATE.dtype,
            max_iter=_STATE.nmax_poi_ite, tol=_STATE.max_poi_tol,
            imag_ceil=_STATE.imag_ceil, smallcore=_STATE.smallcore)
    if verb:
        print(f"[rhongomyniad] point_solve: found {len(images)} images at "
              f"(zs={zs}, xs={x}, ys={y})")
        for xi, yi, mi, td in images:
            print(f"  x={xi:.6f} y={yi:.6f} mag={mi:+.4f} td={td:.4f}d")
    return images


def findimg_i(i: int, verb: int = 0) -> list[tuple[float, float, float, float]]:
    """Solve the lens equation for the i-th registered source (1-based)."""
    if i < 1 or i > _STATE.num_poi:
        raise ValueError(f"findimg_i: point id {i} out of range")
    zs, xs, ys = _STATE.points[i - 1]
    return point_solve(zs, xs, ys, verb=verb)


def findimg(verb: int = 0) -> list[list[tuple[float, float, float, float]]]:
    """Solve for every registered point source."""
    return [findimg_i(i + 1, verb=verb) for i in range(_STATE.num_poi)]


# ---------------------------------------------------------------------------
# Access helpers
# ---------------------------------------------------------------------------
def getpar_lens(i: int) -> tuple[str, tuple[float, ...]]:
    if i < 1 or i > _STATE.num_len:
        raise ValueError(f"getpar_lens: id {i} out of range")
    return _STATE.lenses[i - 1]


def getpar_point(i: int) -> tuple[float, float, float]:
    if i < 1 or i > _STATE.num_poi:
        raise ValueError(f"getpar_point: id {i} out of range")
    return _STATE.points[i - 1]


def getpar_omega() -> float:
    return _STATE.cosmo.omega


def getpar_lambda() -> float:
    return _STATE.cosmo.lam


def getpar_hubble() -> float:
    return _STATE.cosmo.hubble


def getpar_weos() -> float:
    return _STATE.cosmo.weos


def set_max_poi_tol(v: float) -> None:
    _STATE.max_poi_tol = v


def set_nmax_poi_ite(v: int) -> None:
    _STATE.nmax_poi_ite = v


def set_smallcore(v: float) -> None:
    _STATE.smallcore = v


def set_nfw_users(v: int) -> None:
    _STATE.nfw_users = v


def set_finder(name: str) -> None:
    """Select image-finder backend: 'adaptive' (default) or 'uniform'."""
    if name not in ("adaptive", "uniform"):
        raise ValueError("finder must be 'adaptive' or 'uniform'")
    _STATE.finder = name


def get_finder() -> str:
    return _STATE.finder


# ---------------------------------------------------------------------------
# Extended-source / chi2 API (glafic drop-in for GLADE's extend path)
# ---------------------------------------------------------------------------
def set_extend(i: int, model: str,
               p1: float, p2: float, p3: float, p4: float,
               p5: float, p6: float, p7: float, p8: float) -> None:
    """Register extended source ``i`` (1-based).  ``model`` is one of
    gauss/sersic/tophat/moffat/jaffe; p1=zs, p2=norm, p3..p8 = x, y, e, pa,
    r0, n (glafic para_ext layout)."""
    from .sources import SOURCE_IDS  # noqa: PLC0415
    if model not in SOURCE_IDS:
        raise NotImplementedError(
            f"extended-source model '{model}' is not supported; "
            f"available: {sorted(SOURCE_IDS)}")
    if i < 1:
        raise ValueError(f"extend id {i} out of range")
    _ensure_extend_slots(i)
    _STATE.extends[i - 1] = (model, (p1, p2, p3, p4, p5, p6, p7, p8))
    if i > _STATE.num_ext:
        _STATE.num_ext = i


# set_secondary keys mapped onto existing engine knobs rather than the dict.
_SECONDARY_STATE_KEYS = {
    "max_poi_tol": ("max_poi_tol", float),
    "nmax_poi_ite": ("nmax_poi_ite", int),
    "imag_ceil": ("imag_ceil", float),
    "smallcore": ("smallcore", float),
    "nfw_users": ("nfw_users", int),
    "flag_hodensity": ("flag_hodensity", int),
    "hodensity": ("hodensity", float),
}


def set_secondary(buffer: str, verb: int = 0) -> None:
    """Parse a glafic ``set_secondary``-style line, e.g. ``"chi2_splane 1"``."""
    parts = buffer.split()
    if len(parts) != 2:
        raise ValueError(f"set_secondary: cannot parse '{buffer}'")
    key, val = parts
    if key in _SECONDARY_STATE_KEYS:
        attr, cast = _SECONDARY_STATE_KEYS[key]
        setattr(_STATE, attr, cast(float(val)))
        return
    if key in SECONDARY_DEFAULTS:
        default = SECONDARY_DEFAULTS[key]
        _STATE.secondary[key] = type(default)(float(val)) \
            if isinstance(default, (int, float)) else val
        return
    print(f"[rhongomyniad] warning: set_secondary key '{key}' is not supported "
          f"by the GPU engine and was ignored")


def setopt_point(i: int, flag_z: int, flag_x: int, flag_y: int) -> None:
    """Mark which of source i's parameters glafic may adjust internally
    (the inner source-position solve).  flag_z must be 0 (zs is fixed)."""
    if flag_z:
        raise NotImplementedError("rhongomyniad: optimizable point zs is not supported")
    _STATE.opt_point[i] = (int(flag_z), int(flag_x), int(flag_y))


def readobs_extend(fname: str, mask: str = "N", verb: int = 0) -> None:
    """Read the observed extended FITS (+ optional int mask FITS), then derive
    the per-pixel noise + sky exactly like glafic (readobs_extend + readmask +
    calc_obsnoise; call.c:493-504)."""
    from astropy.io import fits  # noqa: PLC0415
    nx, ny = grid_nxy(_STATE.egrid.xmin, _STATE.egrid.xmax,
                      _STATE.egrid.ymin, _STATE.egrid.ymax, _STATE.egrid.pix_ext)
    data = np.asarray(fits.getdata(fname), dtype=np.float64)
    if data.ndim != 2 or data.shape != (ny, nx):
        raise ValueError(
            f"invalid obs fits size: got {data.shape[::-1]}, grid wants {(nx, ny)}")
    dev, dt = _STATE.device, _STATE.dtype
    _STATE.ext.obs = torch.tensor(data, dtype=dt, device=dev)
    if mask is not None and mask != "N":
        mdata = np.asarray(fits.getdata(mask))
        if mdata.shape != (ny, nx):
            raise ValueError("invalid mask fits size")
        _STATE.ext.mask = torch.tensor(mdata > 0, device=dev)
    noise, skymed, skysig = _calc_obsnoise(
        _STATE.ext.obs, _STATE.ext.mask,
        float(_STATE.secondary.get("noise_clip", 3.0)),
        float(_STATE.secondary.get("obs_gain", 3.0)),
        int(_STATE.secondary.get("obs_ncomb", 1)),
        float(_STATE.secondary.get("skyfix_value", 1.0e10)))
    _STATE.ext.noise_computed = noise
    _STATE.ext.skymed = skymed
    _STATE.ext.skysigma = skysig
    if verb:
        print(f"[rhongomyniad] readobs_extend: {fname} ({nx}x{ny}), "
              f"sky={skymed:.6g} sigma={skysig:.6g}")


def readnoise_extend(fname: str, verb: int = 0) -> None:
    """Read a user per-pixel noise FITS (glafic flag_obssig path)."""
    from astropy.io import fits  # noqa: PLC0415
    nx, ny = grid_nxy(_STATE.egrid.xmin, _STATE.egrid.xmax,
                      _STATE.egrid.ymin, _STATE.egrid.ymax, _STATE.egrid.pix_ext)
    data = np.asarray(fits.getdata(fname), dtype=np.float64)
    if data.shape != (ny, nx):
        raise ValueError("invalid noise fits size")
    _STATE.ext.noise_file = torch.tensor(data, dtype=_STATE.dtype,
                                         device=_STATE.device)


def readobs_point(fname: str, verb: int = 0) -> None:
    """Read a glafic point-constraint file.  ``set_point`` must already have
    declared each source with a matching redshift (same rule as glafic)."""
    obs = read_point_obs_file(fname)
    if len(obs) > _STATE.num_poi:
        raise ValueError(
            f"readobs_point: file has {len(obs)} sources but only "
            f"{_STATE.num_poi} declared via startup_setnum/set_point")
    for k, po in enumerate(obs):
        zs_declared = _STATE.points[k][0]
        if abs(po.zs - zs_declared) > 1e-6:
            raise ValueError(
                f"readobs_point: source {k+1} zs={po.zs} does not match "
                f"set_point zs={zs_declared}")
    _STATE.ext.point_obs = obs


def parprior(fname: str, verb: int = 0) -> None:
    """Read a glafic prior file (ranges + gaussian/log priors)."""
    n = _STATE.ext.priors.parse_parprior(fname)
    if verb:
        print(f"[rhongomyniad] parprior: {n} entries from {fname}")


def c2calc_each() -> tuple:
    """The 8-component chi2 breakdown (glade-local glafic c2calc_each):
    (pos, flux, td, prior_point, pixel, prior_ext, prior_lens, penalty)."""
    if not _STATE.initialised_model:
        raise RuntimeError("call model_init() before c2calc_each()")
    return c2calc_each_impl(_STATE, sum_lensmodel, LensContext, point_solve)


def c2calc() -> float:
    """Total chi2 (sum of the c2calc_each components)."""
    return float(sum(c2calc_each()))


def get_nxy_ext() -> tuple[int, int]:
    return grid_nxy(_STATE.egrid.xmin, _STATE.egrid.xmax,
                    _STATE.egrid.ymin, _STATE.egrid.ymax, _STATE.egrid.pix_ext)


def writeimage(id: int = 0, sky: float = 0.0, noise: float = 0.0,
               ori: int = 0):
    """Model extended image (glafic python_writeimage: returns the image
    array rather than writing a FITS).  ``noise > 0`` (mock noise) is not
    supported."""
    if noise != 0.0:
        raise NotImplementedError("rhongomyniad writeimage: noise mocks unsupported")
    imgs = ext_model_images(_STATE, sum_lensmodel, LensContext,
                            flag_source=int(ori), include_masked=True)
    if id == 0:
        img = imgs.sum(dim=0)
    else:
        img = imgs[id - 1]
    return (img + sky).cpu().numpy()


def getpar_extend(i: int) -> tuple[str, tuple]:
    if i < 1 or i > _STATE.num_ext:
        raise ValueError(f"getpar_extend: id {i} out of range")
    return _STATE.extends[i - 1]


def getpar_sky() -> float:
    return float(_STATE.ext.skymed)


def set_psf(*args, **kwargs) -> None:
    raise NotImplementedError(
        "rhongomyniad: PSF convolution is not implemented (GLADE's CPU "
        "pipeline never configures a PSF either)")


# State-introspection (useful for debugging + tests)
def _get_state() -> _State:
    return _STATE
