"""Extended-source (FITS) fitting on the CPU/glafic backend.

This is a *separate* path from the point-only DE optimiser. Where the point-only
path forward-solves images and computes glade's own ``ml_loss``, the extend path
drives glafic per DE candidate and reads back the **per-component chi2 breakdown**
(``glafic.c2calc_each``), then applies user weights:

    loss = W_POS*pos + W_FLUX*flux + W_TD*td + W_EXT*pixel + W_PRIOR*priors (+penalty)

With every weight at 1.0 the loss is exactly glafic's ``c2calc`` (so glafic does
all the hard pixel-chi2 work, including PSF/noise modelling and the inner
source-position solve). The SN point-source position is glafic's fast inner
parameter and is *not* an outer DE dimension.

Point-source constraints may be supplied either as a glafic-native
``constraint_file`` (full fidelity: position + flux + time-delay + parity) or as
glade observation arrays, which are converted to a temporary glafic constraint
file at build time. The extended image is always a FITS read via
``readobs_extend``.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..format.config import GladeConfig
from ..format.values import Bounds, Fixed
from .loss import ExtendLossConfig
from .problem import OptProblem
from .scene import Scene

# Loss returned for an engine error / unfit (matches objective.INVALID_LOSS).
INVALID_LOSS = 1e15
# Any single c2calc component at/above this is a glafic hard penalty (range /
# image-count / parity = 1e30); such a candidate is rejected outright.
PENALTY_FLOOR = 1e29


# ---------------------------------------------------------------------------
# path resolution
# ---------------------------------------------------------------------------

def resolve_path(path: Optional[str], base_dirs) -> Optional[str]:
    """Resolve a (possibly relative) file path against a list of base dirs.

    Returns the first existing candidate, else the path joined to the first base
    dir (so the engine raises a clear 'file not found' rather than us guessing).
    """
    if not path:
        return None
    if os.path.isabs(path) and os.path.exists(path):
        return path
    for base in base_dirs:
        if not base:
            continue
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return os.path.abspath(cand)
    if os.path.exists(path):
        return os.path.abspath(path)
    # not found anywhere; return a best-effort absolute path for a clear error
    base = next((b for b in base_dirs if b), os.getcwd())
    return os.path.abspath(path if os.path.isabs(path) else os.path.join(base, path))


# ---------------------------------------------------------------------------
# glafic point-constraint file: parse headers / write from glade arrays
# ---------------------------------------------------------------------------

@dataclass
class PointSource:
    zs: float
    nimg: int = 0           # observed image count (for missing_img_penalty)
    init_x: float = 0.0
    init_y: float = 0.0
    free_x: int = 1
    free_y: int = 1


def parse_point_file_sources(path: str) -> list[PointSource]:
    """Read the source headers of a glafic readobs_point file.

    Each source is a header line ``id nimg zs [zserr]`` followed by ``nimg``
    image rows. We only need the per-source redshift so set_point can declare it
    before readobs_point (which requires the redshift to match).
    """
    sources: list[PointSource] = []
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines()]
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        i += 1
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        try:
            _id = int(parts[0])
            nimg = int(parts[1])
            zs = float(parts[2]) if len(parts) > 2 else -1.0
        except (ValueError, IndexError):
            continue
        sources.append(PointSource(zs=zs, nimg=nimg))
        i += nimg  # skip the image rows
    return sources


def read_point_file_positions(path: str) -> np.ndarray:
    """All observed image positions ``(x, y)`` [arcsec] from a glafic
    readobs_point file (the ``nimg`` image rows under each ``id nimg zs`` header).

    These are in glafic's engine frame -- the same frame as the model component
    centres -- so they overlay directly on a DE-population iteration figure.
    Returns an empty ``(0, 2)`` array if the file has no parseable rows.
    """
    rows: list[tuple[float, float]] = []
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        i += 1
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        try:
            _id = int(parts[0])
            nimg = int(parts[1])
        except (ValueError, IndexError):
            continue
        for _ in range(nimg):
            if i >= len(lines):
                break
            cols = lines[i].split()
            i += 1
            if len(cols) >= 2:
                rows.append((float(cols[0]), float(cols[1])))
    return np.asarray(rows, dtype=float) if rows else np.zeros((0, 2), dtype=float)


def write_point_file_from_arrays(cfg: GladeConfig, out_path: str) -> int:
    """Convert glade point-observation arrays into a glafic readobs_point file.

    Columns written per image: ``x y flux pos_sigma flux_err td td_err parity``.
    Returns the number of point sources written (always 1 here: one source whose
    images are the obs arrays). Positions are converted to the engine frame
    (mas->arcsec, obs_x_flip applied, center_offset subtracted so glafic's
    engine-frame model lines up exactly as the point-only path's
    prediction+offset does).
    """
    obs = cfg.obs
    pos = np.asarray(obs["obs_positions_mas_list"], dtype=float)
    mags = np.asarray(obs["obs_magnifications_list"], dtype=float)
    magerr = np.asarray(obs["obs_mag_errors_list"], dtype=float)
    possig = np.asarray(obs["obs_pos_sigma_mas_list"], dtype=float)
    n = len(pos)
    td = np.asarray(obs.get("obs_td_list", [0.0] * n), dtype=float)
    tderr = np.asarray(obs.get("obs_td_err_list", [0.0] * n), dtype=float)
    parity = obs.get("obs_parity_list", [0] * n)

    x_sign = -1.0 if obs.get("obs_x_flip", False) else 1.0
    coff_x = x_sign * float(obs.get("center_offset_x", 0.0) or 0.0)
    coff_y = float(obs.get("center_offset_y", 0.0) or 0.0)

    zs = float(cfg.get("source_z", 1.0))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"1 {n} {zs} 0.0\n")
        for i in range(n):
            x = x_sign * pos[i, 0] / 1000.0 - coff_x
            y = pos[i, 1] / 1000.0 - coff_y
            fh.write(
                f"{x:.10g} {y:.10g} {mags[i]:.10g} {possig[i] / 1000.0:.10g} "
                f"{magerr[i]:.10g} {td[i]:.10g} {tderr[i]:.10g} {int(parity[i])}\n")
    return 1


# ---------------------------------------------------------------------------
# the static (candidate-independent) run spec
# ---------------------------------------------------------------------------

@dataclass
class ExtendSpec:
    """Everything an extend run needs that does NOT vary per DE candidate."""

    extended_file: str
    mask_file: Optional[str] = None
    noise_file: Optional[str] = None
    point_file: Optional[str] = None          # resolved glafic constraint file
    prior_file: Optional[str] = None
    secondary: dict = field(default_factory=dict)   # set_secondary key -> value
    point_sources: list[PointSource] = field(default_factory=list)
    _own_point_file: bool = False             # True if we created a temp point file

    @property
    def n_point(self) -> int:
        return len(self.point_sources)


def _source_free_flags(cfg: GladeConfig) -> tuple[int, int]:
    sx, sy = cfg.source.get("source_x"), cfg.source.get("source_y")
    return (1 if isinstance(sx, Bounds) else 0, 1 if isinstance(sy, Bounds) else 0)


def _source_init(cfg: GladeConfig) -> tuple[float, float]:
    def mid(v, default):
        if isinstance(v, Bounds):
            return 0.5 * (v.lo + v.hi)
        if isinstance(v, Fixed):
            return v.value
        if isinstance(v, (int, float)):
            return float(v)
        return default
    return (mid(cfg.source.get("source_x"), 0.0), mid(cfg.source.get("source_y"), 0.0))


# glafic set_secondary keys we forward, in a stable order.
_SECONDARY_FORWARD = (
    "chi2_splane", "chi2_checknimg", "chi2_restart", "chi2_usemag",
    "ran_seed", "obs_gain", "obs_ncomb", "obs_readnoise", "flag_extnorm",
)


def build_extend_spec(cfg: GladeConfig, base_dir: Optional[str] = None) -> ExtendSpec:
    """Build the static run spec from a merged config.

    ``base_dir`` is where relative FITS / constraint / prior paths are resolved
    (typically the directory of the .dat, falling back to the CWD).
    """
    bases = [base_dir, os.getcwd()]
    obs = cfg.obs

    extended = resolve_path(obs.get("extended_file"), bases)
    if extended is None:
        raise ValueError("extend run requires 'extended_file' (observed FITS)")

    spec = ExtendSpec(
        extended_file=extended,
        mask_file=resolve_path(obs.get("extend_mask_file"), bases),
        noise_file=resolve_path(obs.get("noise_file"), bases),
        prior_file=resolve_path(obs.get("prior_file"), bases),
    )

    # secondary settings (numbers -> "key value")
    sec = {}
    for k in _SECONDARY_FORWARD:
        v = cfg.get(k)
        if v is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
            sec[k] = v
    spec.secondary = sec

    # point-source constraints: constraint_file takes precedence over arrays
    free_x, free_y = _source_free_flags(cfg)
    init_x, init_y = _source_init(cfg)
    cfile = resolve_path(obs.get("constraint_file"), bases)
    has_arrays = "obs_positions_mas_list" in obs
    if cfile is not None:
        spec.point_file = cfile
        srcs = parse_point_file_sources(cfile)
        for k, s in enumerate(srcs):
            s.free_x, s.free_y = free_x, free_y
            if k == 0:
                s.init_x, s.init_y = init_x, init_y
            if s.zs < 0:
                s.zs = float(cfg.get("source_z", 1.0))
        spec.point_sources = srcs
    elif has_arrays:
        fd, tmp = tempfile.mkstemp(prefix="glade_point_", suffix=".dat")
        os.close(fd)
        n_img = len(np.asarray(obs["obs_positions_mas_list"], dtype=float))
        write_point_file_from_arrays(cfg, tmp)
        spec.point_file = tmp
        spec._own_point_file = True
        spec.point_sources = [PointSource(
            zs=float(cfg.get("source_z", 1.0)), nimg=n_img,
            init_x=init_x, init_y=init_y, free_x=free_x, free_y=free_y)]
    # else: no point constraints (pure extended fit), n_point == 0

    return spec


# ---------------------------------------------------------------------------
# per-candidate evaluation against glafic
# ---------------------------------------------------------------------------

def _pad7(params) -> list[float]:
    return ([float(p) for p in params] + [0.0] * 7)[:7]


def _count_missing_images(engine, spec: ExtendSpec) -> int:
    """Predicted-vs-observed image shortfall for the point sources, summed.

    Called *after* ``c2calc_each`` (so glafic has solved each source's inner
    position) and before ``quit``. ``findimg_i`` re-solves the images at that
    solved source position; we count how many fewer than observed there are.
    Returns 0 if the engine lacks ``findimg_i`` or a solve raises (no penalty).
    """
    if not hasattr(engine, "findimg_i"):
        return 0
    missing = 0
    for i, ps in enumerate(spec.point_sources, start=1):
        if ps.nimg <= 0:
            continue
        try:
            n_pred = len(engine.findimg_i(i, verb=0))
        except Exception:
            continue
        missing += max(0, ps.nimg - n_pred)
    return missing


def eval_components(engine, scene: Scene, spec: ExtendSpec, prefix: str,
                    count_missing: bool = False) -> tuple[Optional[tuple], int]:
    """Drive glafic for one scene and return ``(c2calc_each tuple, n_missing)``.

    The 8-tuple is ``(pos, flux, td, prior_pt, pixel, prior_ext, prior_lens,
    penalty)``; ``n_missing`` is the point-image shortfall (0 unless
    ``count_missing``). Returns ``(None, 0)`` on engine failure.
    """
    m = engine
    m.init(scene.omega, scene.lam, scene.weos, scene.hubble, prefix,
           scene.xmin, scene.ymin, scene.xmax, scene.ymax,
           scene.pix_ext, scene.pix_poi, scene.maxlev, verb=0)
    try:
        for k, v in spec.secondary.items():
            m.set_secondary(f"{k} {v}", verb=0)
        m.startup_setnum(len(scene.components), len(scene.extends), spec.n_point)
        for i, comp in enumerate(scene.components, start=1):
            m.set_lens(i, comp.glafic_type, comp.z, *_pad7(comp.params))
        for i, comp in enumerate(scene.extends, start=1):
            m.set_extend(i, comp.glafic_type, comp.z, *_pad7(comp.params))
        for i, ps in enumerate(spec.point_sources, start=1):
            m.set_point(i, ps.zs, ps.init_x, ps.init_y)
            # let glafic solve the source position (its fast inner parameter)
            m.setopt_point(i, 0, ps.free_x, ps.free_y)
        m.model_init(verb=0)
        m.readobs_extend(spec.extended_file, spec.mask_file or "N", verb=0) \
            if spec.mask_file else m.readobs_extend(spec.extended_file, verb=0)
        if spec.noise_file:
            m.readnoise_extend(spec.noise_file, verb=0)
        if spec.point_file and spec.n_point > 0:
            m.readobs_point(spec.point_file, verb=0)
        if spec.prior_file:
            m.parprior(spec.prior_file, verb=0)
        comp = m.c2calc_each()
        n_missing = (_count_missing_images(m, spec)
                     if count_missing and spec.n_point > 0 else 0)
    except Exception:
        return None, 0
    finally:
        m.quit()
    return tuple(float(c) for c in comp), n_missing


def render_images(engine, scene: Scene, spec: ExtendSpec, prefix: str):
    """Drive glafic for *scene* and return ``(model, obs, nx, ny)``.

    ``model`` is the best-fit lensed extended image from glafic ``writeimage`` and
    ``obs`` is the observed FITS, both as (ny, nx) float arrays on the same grid.
    Returns ``None`` on failure (e.g. glafic/astropy unavailable).
    """
    m = engine
    m.init(scene.omega, scene.lam, scene.weos, scene.hubble, prefix,
           scene.xmin, scene.ymin, scene.xmax, scene.ymax,
           scene.pix_ext, scene.pix_poi, scene.maxlev, verb=0)
    try:
        for k, v in spec.secondary.items():
            m.set_secondary(f"{k} {v}", verb=0)
        m.startup_setnum(len(scene.components), len(scene.extends), spec.n_point)
        for i, comp in enumerate(scene.components, start=1):
            m.set_lens(i, comp.glafic_type, comp.z, *_pad7(comp.params))
        for i, comp in enumerate(scene.extends, start=1):
            m.set_extend(i, comp.glafic_type, comp.z, *_pad7(comp.params))
        for i, ps in enumerate(spec.point_sources, start=1):
            m.set_point(i, ps.zs, ps.init_x, ps.init_y)
            m.setopt_point(i, 0, ps.free_x, ps.free_y)
        m.model_init(verb=0)
        if spec.mask_file:
            m.readobs_extend(spec.extended_file, spec.mask_file, verb=0)
        else:
            m.readobs_extend(spec.extended_file, verb=0)
        if spec.noise_file:
            m.readnoise_extend(spec.noise_file, verb=0)
        if spec.point_file and spec.n_point > 0:
            m.readobs_point(spec.point_file, verb=0)
        if spec.prior_file:
            m.parprior(spec.prior_file, verb=0)
        # one chi2 pass so glafic solves the inner source position before render
        m.c2calc_each()
        nx, ny = m.get_nxy_ext()
        model = np.array(m.writeimage(0, 0.0, 0.0, 0), dtype=float)
    except Exception:
        return None
    finally:
        m.quit()

    try:
        from astropy.io import fits  # noqa: PLC0415
        obs = np.array(fits.getdata(spec.extended_file), dtype=float)
    except Exception:
        obs = None
    return model, obs, nx, ny


class ExtendObjective:
    """Picklable DE objective for the extended-source CPU path.

    Each call: candidate -> scene -> drive glafic -> c2calc_each -> weighted loss.
    The engine module is rebuilt lazily per worker (so scipy's process pool can
    fan it out, exactly like the point-only :class:`Objective`).
    """

    def __init__(self, problem: OptProblem, spec: ExtendSpec,
                 loss_cfg: ExtendLossConfig):
        self.problem = problem
        self.spec = spec
        self.loss_cfg = loss_cfg
        self._engine = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_engine"] = None          # rebuilt per worker
        return state

    def _engine_module(self):
        if self._engine is None:
            from .backends import _import_glafic
            self._engine = _import_glafic()
        return self._engine

    def components_for(self, candidate) -> Optional[tuple]:
        """The raw c2calc_each component tuple for a candidate (or None)."""
        scene = self.problem.make_scene(np.asarray(candidate, dtype=float))
        prefix = f"temp_glade_ext_{os.getpid()}"
        comp, _ = eval_components(self._engine_module(), scene, self.spec, prefix)
        return comp

    def evaluate_one(self, candidate) -> float:
        mp = self.loss_cfg.missing_img_penalty
        scene = self.problem.make_scene(np.asarray(candidate, dtype=float))
        prefix = f"temp_glade_ext_{os.getpid()}"
        comp, n_missing = eval_components(
            self._engine_module(), scene, self.spec, prefix,
            count_missing=mp > 0.0)
        if comp is None:
            return INVALID_LOSS
        if mp > 0.0 and n_missing > 0:
            # Under-imaged SN: glafic poisons the point chi2 with chi2pen_nimg
            # (pos = flux = 1e30). Drop those, keep the still-valid extended-pixel
            # + prior chi2, and replace glafic's flat reject with a graded
            # (n_obs - n_pred) * missing_img_penalty. Any component we actually
            # *use* (pixel, ext prior, lens prior) hitting the penalty floor -- a
            # physical-range violation that may live in out[5]/out[6]/out[7], not
            # just out[7] -- is still a hard reject; only the nimg-poisoned point
            # components (0,1) and the now-meaningless point td/prior (2,3) are
            # excluded by design.
            if any(comp[i] >= PENALTY_FLOOR for i in (4, 5, 6, 7)):
                return INVALID_LOSS
            _pos, _flux, _td, _pp, pixel, prior_ext, prior_lens, _pen = comp
            base = (self.loss_cfg.w_ext * pixel
                    + self.loss_cfg.w_prior * (prior_ext + prior_lens))
            return float(base + n_missing * mp)
        if any(c >= PENALTY_FLOOR for c in comp):
            return INVALID_LOSS
        return float(self.loss_cfg.combine(comp))

    def __call__(self, candidate) -> float:
        return self.evaluate_one(candidate)
