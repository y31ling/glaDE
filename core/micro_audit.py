"""Micro-image audit helpers (the ``auto_check`` fix).

A compact perturber (point mass / small-core king / extreme-parameter NFW
sub-halo) sitting within a few mas of a matched image physically splits that
image into several micro-images whose separation (~ the perturber's theta_E,
sub-mas to ~11 mas in the archived examples) is below the run's image-finder
resolution (finest glafic findimg cell ``pix_poi / 2**(maxlev - 1)`` = 12.5 mas
at the standard grid). The finder then
converges to ONE root, and the loss scores that single-root mu against an
observation that physically measures the total PSF flux Sigma|mu| of the whole
cluster — letting the optimizer converge to fake "parity-flipped demagnified"
solutions (see microimage_auto_check_plan.md).

This module holds the pieces shared by both protection layers:

* pure-numpy/scipy flat-wCDM angular-diameter distances (NO torch, NO
  rhongomyniad — the CPU path must work on torch-less installs);
* the compact scale ``theta_scale`` of a component (§6.1 of the plan);
* the trigger rule ``d < 10 * theta_scale + 2 mas`` (§6.2);
* the two-stage zoomed ``findimg`` (glafic binary for the verify layer,
  python-binding cycle for the in-loop CPU layer);
* the verify-layer audit report (:func:`micro_audit`);
* the in-loop CPU checked loss (:func:`checked_point_source_loss`).

Cluster semantics (§6.3, iron rules): Sigma|mu| of a cluster replaces ONLY the
matched image's magnification in ``ml_loss``; the global image count seen by
``select_images`` / the ``n_obs + 1`` drop / ``missing_img_penalty`` always
comes from the main finder; with no trigger (or ``auto_check = False``) every
code path is bit-identical to the historical behaviour.
"""
from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import numpy as np

from .format import schema
from .optimize.loss import LossConfig, ml_loss
from .optimize.matching import assign_images, select_images
from .optimize.scene import ObsData, Scene

# --- physical constants, verbatim from glafic.h (see rhongomyniad.constants;
# copied here because this module must not import torch/rhongomyniad) --------
ARCSEC2RADIAN = 0.00000484813681109536
COVERH_MPCH = 2997.92458          # c/H0 in Mpc/h
MPC2METER = 3.085677581e22
R_SCHWARZ = 2953.339382           # Schwarzschild radius of M_sun [m]
C_LIGHT_KMS = 2.99792458e5
GSL_EPSREL_DISTANCE = 1.0e-6
TOL_CURVATURE = 1.0e-6

# --- audit tuning (plan §4 / §6) --------------------------------------------
SCALE_FLOOR_MAS = 0.02      # theta_scale lower guard (source size / numerics)
# compactness gate: anything larger is resolved by the run grid itself and
# cannot hide micro-images below the finder resolution
SCALE_CAP_MAS = 100.0
TRIG_BASE_MAS = 2.0         # R_trig = 10 * theta_scale + TRIG_BASE
COARSE_HALF_MAS = 15.0      # coarse zoom box half-width around each image
COARSE_PIX_POI = 5.0e-4     # coarse box pix_poi (finest 0.03125 mas at maxlev 5)
FINE_SCALE_MAS = 0.2        # theta_scale below which the fine box is added
FAKE_REL_TOL = 0.05         # |sum|mu|-|mu_single|| / max(|mu_single|,1) gate


# --------------------------------------------------------------------------- #
# cosmology: flat(-friendly) wCDM angular-diameter distances (pure scipy)
# --------------------------------------------------------------------------- #

def _hubble_ez2(z: float, omega: float, lam: float, weos: float) -> float:
    """E^2(z) exactly as glafic distance.c:177-188."""
    return ((1.0 + omega * z - lam) * (1.0 + z) ** 2
            + lam * (1.0 + z) ** (3.0 * (1.0 + weos)))


def _chi(z_a: float, z_b: float, omega: float, lam: float, weos: float) -> float:
    from scipy.integrate import quad

    if z_a >= z_b:
        return 0.0

    def integrand(a: float) -> float:
        z = 1.0 / a - 1.0
        return 1.0 / (a * a * math.sqrt(_hubble_ez2(z, omega, lam, weos)))

    val, _err = quad(integrand, 1.0 / (1.0 + z_b), 1.0 / (1.0 + z_a),
                     epsabs=0.0, epsrel=GSL_EPSREL_DISTANCE, limit=200)
    return val


@lru_cache(maxsize=256)
def angulard(omega: float, lam: float, weos: float,
             z_a: float, z_b: float) -> float:
    """Dimensionless angular-diameter distance (units of c/H0), mirroring
    glafic distance.c (curvature handled like comoving())."""
    if z_a >= z_b:
        return 0.0
    chi = _chi(z_a, z_b, omega, lam, weos)
    k = omega + lam - 1.0
    if abs(k) >= TOL_CURVATURE:
        if k > 0.0:
            chi = math.sin(chi * math.sqrt(k)) / math.sqrt(k)
        else:
            chi = math.sinh(chi * math.sqrt(-k)) / math.sqrt(-k)
    return chi / (1.0 + z_b)


@dataclass(frozen=True)
class Distances:
    """The three angular-diameter distances of a lens/source pair."""

    dol: float
    dos: float
    dls: float

    @classmethod
    def build(cls, omega: float, lam: float, weos: float,
              zl: float, zs: float) -> "Distances":
        return cls(dol=angulard(omega, lam, weos, 0.0, zl),
                   dos=angulard(omega, lam, weos, 0.0, zs),
                   dls=angulard(omega, lam, weos, zl, zs))


def theta_e_point_arcsec(mass, dis: Distances):
    """Einstein radius [arcsec] of a point mass [h^-1 Msun]; the exact
    expression Rhongomyniad's ``_re2_point`` evaluates (h-independent because
    glafic distances are in Mpc/h and M in h^-1 Msun). Vectorizes over
    ``mass`` (numpy arrays supported)."""
    if dis.dls <= 0.0 or dis.dol <= 0.0 or dis.dos <= 0.0:
        return np.zeros_like(np.asarray(mass, dtype=float)) + 0.0
    d = dis.dls / (COVERH_MPCH * dis.dol * dis.dos)
    re2 = 2.0 * (R_SCHWARZ * np.maximum(np.asarray(mass, dtype=float), 0.0)
                 / MPC2METER) * d / (ARCSEC2RADIAN ** 2)
    return np.sqrt(np.maximum(re2, 0.0))


def theta_e_sis_arcsec(sigma_kms, dis: Distances):
    """Einstein radius [arcsec] of an SIS with velocity dispersion sigma
    [km/s]: 4 pi (sigma/c)^2 D_ls / D_s. Vectorizes over ``sigma_kms``."""
    if dis.dos <= 0.0 or dis.dls <= 0.0:
        return np.zeros_like(np.asarray(sigma_kms, dtype=float)) + 0.0
    s = np.asarray(sigma_kms, dtype=float) / C_LIGHT_KMS
    return 4.0 * math.pi * s * s * (dis.dls / dis.dos) / ARCSEC2RADIAN


# --------------------------------------------------------------------------- #
# theta_scale + compact-perturber detection (plan §6.1 / §6.2)
# --------------------------------------------------------------------------- #

_CORE_PARAM_NAMES = ("rc", "rcore", "rco")


def _spec_positions(spec) -> Optional[tuple[int, int]]:
    """(ix, iy) indices of the centre parameters, or None."""
    ix = iy = None
    for j, p in enumerate(spec.params):
        if p.name == "x":
            ix = j
        elif p.name == "y":
            iy = j
    if ix is None or iy is None:
        return None
    return ix, iy


def theta_scale_mas(model_key: str, params, dis: Distances) -> Optional[float]:
    """Compact scale [mas] of one component (plan §6.1), or ``None`` when the
    model cannot act as a compact perturber (irregular layouts: pert / gaupot /
    mpole / crline / clus3 / gals, and extended sources).

    * point           : theta_E(M)
    * king            : max(theta_E(M), rc)
    * sigma models    : max(theta_E_SIS(sigma), core radius)   (sie / jaffe)
    * Einstein-radius models : the ``re`` parameter itself     (pow family)
    * other mass-led profiles: max(theta_E(M), floor) — a same-mass point
      lens upper-bounds the compact scale of any diffuse profile (§6.1)
    The result is clamped to ``>= SCALE_FLOOR_MAS``.
    """
    spec = schema.model(model_key)
    if spec is None or schema.is_extend_model(model_key):
        return None
    if spec.uncertain or _spec_positions(spec) is None:
        return None
    p = list(params) + [0.0] * 7

    # locate the log-searched (is_mass) leading parameter and read its unit
    # from the schema description — sie/jaffe carry a velocity dispersion,
    # pow an Einstein radius, everything else a mass.
    mi = next((j for j, ps in enumerate(spec.params) if ps.is_mass), None)
    if mi is None or mi >= len(p):
        return None
    mp = spec.params[mi]
    v = float(p[mi])
    if "km/s" in mp.desc:
        base_mas = float(theta_e_sis_arcsec(v, dis)) * 1000.0
    elif "arcsec" in mp.desc:
        base_mas = abs(v) * 1000.0
    elif "Msun" in mp.desc:
        base_mas = float(theta_e_point_arcsec(v, dis)) * 1000.0
    else:
        return None

    # broaden by an explicit core/softening radius when the model has one
    # (king rc, jaffe rco, sie rcore ...): a large core smears the critical
    # structure to that scale.
    for j, ps in enumerate(spec.params):
        if ps.name in _CORE_PARAM_NAMES and j < len(p):
            base_mas = max(base_mas, abs(float(p[j])) * 1000.0)
    return max(base_mas, SCALE_FLOOR_MAS)


@dataclass(frozen=True)
class Perturber:
    """One compact perturber of a concrete scene."""

    comp_index: int          # index into scene.components
    glafic_type: str
    x: float                 # model-frame centre [arcsec]
    y: float
    theta_scale: float       # [mas]

    @property
    def r_trig_mas(self) -> float:
        return 10.0 * self.theta_scale + TRIG_BASE_MAS


def scene_distances(scene: Scene, zl: Optional[float] = None) -> Distances:
    if zl is None:
        zl = scene.components[0].z if scene.components else 0.216
    return Distances.build(scene.omega, scene.lam, scene.weos,
                           float(zl), float(scene.source_z))


def find_compact_perturbers(scene: Scene) -> list[Perturber]:
    """Every component whose compact scale passes the ``SCALE_CAP_MAS`` gate.

    The gate replaces any category bookkeeping: a main-lens-scale component
    (theta_E ~ hundreds of mas .. arcsec) produces image structure the normal
    run grid already resolves, so it can never hide micro-images — while a
    LOCKED compact sub-halo must still be checked (historical run1). This is
    a pure function of the concrete scene, shared by both layers and by the
    per-candidate GPU trigger."""
    out: list[Perturber] = []
    for i, comp in enumerate(scene.components):
        spec = schema.model(comp.glafic_type)
        if spec is None:
            continue
        pos = _spec_positions(spec)
        if pos is None:
            continue
        dis = Distances.build(scene.omega, scene.lam, scene.weos,
                              float(comp.z), float(scene.source_z))
        ts = theta_scale_mas(comp.glafic_type, comp.params, dis)
        if ts is None or ts > SCALE_CAP_MAS:
            continue
        p = list(comp.params) + [0.0] * 7
        out.append(Perturber(i, comp.glafic_type,
                             float(p[pos[0]]), float(p[pos[1]]), ts))
    return out


def nearest_perturber(perturbers: list[Perturber],
                      ix: float, iy: float) -> Optional[tuple[Perturber, float]]:
    """Closest perturber to a model-frame image position, as
    ``(perturber, distance_mas)``; None when the list is empty."""
    best: Optional[tuple[Perturber, float]] = None
    for pert in perturbers:
        d = math.hypot(pert.x - ix, pert.y - iy) * 1000.0
        if best is None or d < best[1]:
            best = (pert, d)
    return best


def triggered(perturbers: list[Perturber], ix: float, iy: float
              ) -> Optional[tuple[Perturber, float]]:
    """The trigger rule (§6.2): the nearest perturber with
    ``d < 10 * theta_scale + 2 mas``, or None."""
    hit: Optional[tuple[Perturber, float]] = None
    for pert in perturbers:
        d = math.hypot(pert.x - ix, pert.y - iy) * 1000.0
        if d < pert.r_trig_mas and (hit is None or d < hit[1]):
            hit = (pert, d)
    return hit


# --------------------------------------------------------------------------- #
# zoom boxes (§4 step 2 — shared geometry for both layers)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ZoomBox:
    cx: float           # arcsec (model frame)
    cy: float
    half: float         # arcsec
    pix_poi: float      # arcsec

    @property
    def pix_ext(self) -> float:
        return max(2.0 * self.half / 200.0, 1.0e-12)


def zoom_boxes(ix: float, iy: float,
               trig: Optional[tuple[Perturber, float]]) -> list[ZoomBox]:
    """The two-stage box set for one image: always the coarse box around the
    image; plus the fine perturber box when the trigger's compact scale is
    below what the coarse grid resolves (§4 step 2)."""
    boxes = [ZoomBox(ix, iy, COARSE_HALF_MAS / 1000.0, COARSE_PIX_POI)]
    if trig is not None:
        pert, d = trig
        if pert.theta_scale < FINE_SCALE_MAS:
            half_mas = max(20.0 * pert.theta_scale, 2.0 * d)
            boxes.append(ZoomBox(pert.x, pert.y, half_mas / 1000.0,
                                 max(pert.theta_scale / 1000.0, 1.0e-9)))
    return boxes


def _merge_roots(groups: list[list[tuple[float, float, float]]],
                 tol_mas: float) -> list[tuple[float, float, float]]:
    """Union of root lists with pairwise dedup below ``tol_mas``."""
    tol2 = (tol_mas / 1000.0) ** 2
    out: list[tuple[float, float, float]] = []
    for grp in groups:
        for (x, y, mu) in grp:
            if any((x - xo) ** 2 + (y - yo) ** 2 < tol2 for xo, yo, _ in out):
                continue
            out.append((x, y, mu))
    return out


def solve_cluster(scene: Scene, ix: float, iy: float,
                  trig: Optional[tuple[Perturber, float]],
                  solver: Callable[[Scene, ZoomBox], Optional[list]],
                  other_images: Optional[list[tuple[float, float]]] = None,
                  ) -> Optional[list[tuple[float, float, float]]]:
    """Run the two-stage zoom for one image and return its micro-image
    cluster, or None when the zoom produced nothing usable.

    ``solver(scene, box)`` returns ``[(x, y, mu), ...]`` or None — the glafic
    BINARY for the verify layer, the python-binding cycle for the in-loop CPU
    layer. Roots closer to another matched image than to this one are
    discarded (§4 step 3: the boxes are far smaller than image separations,
    but assert it anyway)."""
    groups = []
    for box in zoom_boxes(ix, iy, trig):
        roots = solver(scene, box)
        if roots:
            groups.append(roots)
    if not groups:
        return None
    tol = (trig[0].theta_scale / 10.0) if trig is not None else 1.0e-3
    roots = _merge_roots(groups, tol)
    if other_images:
        kept = []
        for (x, y, mu) in roots:
            d_own = (x - ix) ** 2 + (y - iy) ** 2
            if all(d_own <= (x - ox) ** 2 + (y - oy) ** 2
                   for ox, oy in other_images):
                kept.append((x, y, mu))
        roots = kept
    return roots or None


# --------------------------------------------------------------------------- #
# solvers: glafic binary (verify layer) and python-binding cycle (in-loop CPU)
# --------------------------------------------------------------------------- #

def _zoom_input_text(scene: Scene, box: ZoomBox, prefix: str) -> str:
    def pad7(params):
        return ([float(v) for v in params] + [0.0] * 7)[:7]

    lines = ["# GLADE micro-image audit zoom input", "",
             f"omega      {scene.omega}", f"lambda     {scene.lam}",
             f"weos       {scene.weos}", f"hubble     {scene.hubble}", "",
             f"prefix     {prefix}", "",
             f"xmin       {box.cx - box.half:.12e}",
             f"ymin       {box.cy - box.half:.12e}",
             f"xmax       {box.cx + box.half:.12e}",
             f"ymax       {box.cy + box.half:.12e}",
             f"pix_ext    {box.pix_ext:.12e}",
             f"pix_poi    {box.pix_poi:.12e}",
             "maxlev     5", "outformat_exp 1", ""]
    lines.append(f"startup    {len(scene.components)} 0 1")
    for comp in scene.components:
        nums = "    ".join(f"{v:.10e}" for v in [comp.z, *pad7(comp.params)])
        lines.append(f"lens       {comp.glafic_type:<8} {nums}")
    lines.append(f"point      {scene.source_z}    {scene.source_x:.10e}    "
                 f"{scene.source_y:.10e}")
    lines += ["", "end_startup", "", "start_command", "", "findimg", "",
              "quit", ""]
    return "\n".join(lines)


def make_binary_solver(workdir: str, timeout: int = 60,
                       prefix: str = "micro_zoom"
                       ) -> Optional[Callable[[Scene, ZoomBox], Optional[list]]]:
    """A ``solver(scene, box)`` that runs the vendored glafic binary in
    ``workdir``; None when no binary is available."""
    from .verify import _read_glafic_point, find_glafic_bin

    bin_path = find_glafic_bin()
    if not bin_path:
        return None
    os.makedirs(workdir, exist_ok=True)
    counter = {"n": 0}

    def solver(scene: Scene, box: ZoomBox) -> Optional[list]:
        counter["n"] += 1
        pfx = f"{prefix}_{counter['n']}"
        input_path = os.path.join(workdir, f"{pfx}.input")
        with open(input_path, "w", encoding="utf-8") as fh:
            fh.write(_zoom_input_text(scene, box, pfx))
        try:
            proc = subprocess.run([bin_path, os.path.basename(input_path)],
                                  cwd=workdir, capture_output=True, text=True,
                                  timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            return None
        if proc.returncode != 0:
            return None
        return _read_glafic_point(os.path.join(workdir, f"{pfx}_point.dat"))

    return solver


def make_binding_solver(module) -> Callable[[Scene, ZoomBox], Optional[list]]:
    """A ``solver(scene, box)`` driving the glafic PYTHON BINDING through a
    zoomed init/point_solve cycle (in-loop CPU layer; §5b). The caller must
    invoke it only when no other init cycle is open (after the main
    ``compute_images`` cycle has quit)."""

    def pad7(params):
        return ([float(v) for v in params] + [0.0] * 7)[:7]

    def solver(scene: Scene, box: ZoomBox) -> Optional[list]:
        prefix = f"temp_glade_zoom_{os.getpid()}"
        try:
            module.init(scene.omega, scene.lam, scene.weos, scene.hubble,
                        prefix,
                        box.cx - box.half, box.cy - box.half,
                        box.cx + box.half, box.cy + box.half,
                        box.pix_ext, box.pix_poi, 5, verb=0)
            try:
                module.startup_setnum(len(scene.components), 0, 1)
                for k, comp in enumerate(scene.components, start=1):
                    module.set_lens(k, comp.glafic_type, comp.z,
                                    *pad7(comp.params))
                module.set_point(1, scene.source_z, scene.source_x,
                                 scene.source_y)
                module.model_init(verb=0)
                result = module.point_solve(scene.source_z, scene.source_x,
                                            scene.source_y, verb=0)
            finally:
                module.quit()
        except Exception:  # noqa: BLE001 — a failed zoom must never kill DE
            return None
        if not result:
            return None
        return [(float(im[0]), float(im[1]), float(im[2])) for im in result]

    return solver


# --------------------------------------------------------------------------- #
# layer 1: the verify-time audit (plan §4)
# --------------------------------------------------------------------------- #

def micro_audit(scene: Scene, obs: ObsData,
                matched_xy: np.ndarray, matched_mu: np.ndarray,
                obs_idx: np.ndarray,
                loss_cfg: LossConfig, workdir: str,
                solver: Optional[Callable] = None) -> dict:
    """Audit every matched image of a final solution for unresolved
    micro-image clusters. Never raises.

    ``matched_xy`` is (n, 2) MODEL-frame positions of the matched images,
    ``matched_mu`` their single-root magnifications, ``obs_idx`` the observed
    indices they were matched to (ascending, as from ``assign_images``).
    Returns the ``micro_audit`` report dict of the plan (§4 step 5)."""
    report: dict = {"per_image": [], "physical_loss": None,
                    "fake_solution": False, "warnings": []}
    try:
        if solver is None:
            solver = make_binary_solver(workdir)
        if solver is None:
            report["warnings"].append(
                "micro-audit skipped: glafic binary not found")
            return report

        perturbers = find_compact_perturbers(scene)
        n = len(matched_mu)
        model_xy = np.asarray(matched_xy, dtype=float).reshape(n, 2)
        sum_mu = np.abs(np.asarray(matched_mu, dtype=float)).copy()
        delta_mas = np.zeros(n)
        co = np.asarray(obs.center_offset, dtype=float)
        obs_pos = np.asarray(obs.positions, dtype=float)[np.asarray(obs_idx)]

        for i in range(n):
            ix, iy = float(model_xy[i, 0]), float(model_xy[i, 1])
            mu_single = float(matched_mu[i])
            near = nearest_perturber(perturbers, ix, iy)
            trig = None
            if near is not None and near[1] < COARSE_HALF_MAS:
                trig = near
            others = [(float(model_xy[j, 0]), float(model_xy[j, 1]))
                      for j in range(n) if j != i]
            roots = solve_cluster(scene, ix, iy, trig, solver, others)
            entry: dict = {"n_micro": 1, "mu_single": mu_single,
                           "sum_abs_mu": abs(mu_single),
                           "centroid_shift_mas": None,
                           "trigger": None, "roots": []}
            if trig is not None:
                pert, d = trig
                entry["trigger"] = {"comp_index": pert.comp_index,
                                    "type": pert.glafic_type,
                                    "d_mas": float(d),
                                    "theta_scale_mas": float(pert.theta_scale)}
            if roots:
                mus = np.array([r[2] for r in roots], dtype=float)
                xy = np.array([[r[0], r[1]] for r in roots], dtype=float)
                w = np.abs(mus)
                s = float(w.sum())
                entry["n_micro"] = len(roots)
                entry["sum_abs_mu"] = s
                entry["roots"] = [[float(a), float(b), float(m)]
                                  for (a, b, m) in roots]
                if s > 0.0:
                    cen = (xy * w[:, None]).sum(axis=0) / s
                    obs_frame = cen + co
                    shift = float(np.hypot(*(obs_frame - obs_pos[i])) * 1000.0)
                    entry["centroid_shift_mas"] = shift
                    sum_mu[i] = s
                    delta_mas[i] = shift
                else:
                    delta_mas[i] = float(
                        np.hypot(*((model_xy[i] + co) - obs_pos[i])) * 1000.0)
            else:
                report["warnings"].append(
                    f"micro-audit: zoomed findimg produced no roots for image "
                    f"{i + 1}; keeping the single-root value")
                delta_mas[i] = float(
                    np.hypot(*((model_xy[i] + co) - obs_pos[i])) * 1000.0)
            report["per_image"].append(entry)

        oi = np.asarray(obs_idx)
        base = ml_loss(delta_mas, sum_mu,
                       obs.magnifications[oi], obs.mag_errors[oi],
                       obs.pos_sigma_mas[oi], loss_cfg)
        n_missing = obs.n - n
        report["physical_loss"] = float(
            base + n_missing * loss_cfg.missing_img_penalty)

        fake = any(
            abs(e["sum_abs_mu"] - abs(e["mu_single"]))
            / max(abs(e["mu_single"]), 1.0) > FAKE_REL_TOL
            for e in report["per_image"])
        report["fake_solution"] = bool(fake)
        if fake:
            report["warnings"].append(
                "FAKE-SOLUTION SUSPECT: at least one matched image is an "
                "unresolved micro-image cluster (sum|mu| differs from the "
                "single-root |mu| by >5%). The nominal loss scored a single "
                "micro-root against the PSF total flux and is NOT physical; "
                "trust micro_audit.physical_loss instead "
                "(see microimage_auto_check_plan.md).")
    except Exception as exc:  # noqa: BLE001 — verify must never raise
        report["warnings"].append(f"micro-audit failed: {exc}")
    return report


# --------------------------------------------------------------------------- #
# layer 2 (CPU): the in-loop checked loss (plan §5b)
# --------------------------------------------------------------------------- #

def checked_point_source_loss(images, obs: ObsData, loss_cfg: LossConfig,
                              scene: Scene,
                              solver: Optional[Callable[[Scene, ZoomBox], Optional[list]]] = None,
                              cluster_fn: Optional[Callable] = None,
                              ) -> float:
    """``point_source_loss`` with the triggered Sigma|mu| substitution.

    Selection / Hungarian matching / the ``n_obs + 1`` drop / the missing-image
    penalty all run on the MAIN finder's images exactly as before (§6.3 rule
    2); only a triggered image's magnification is replaced by its cluster
    Sigma|mu| before ``ml_loss``. With no compact perturber near any matched
    image the returned float is bit-identical to ``point_source_loss``.

    The local multi-root solve is pluggable: pass either ``solver(scene, box)``
    (zoomed glafic cycle — the CPU layer) or ``cluster_fn(scene, ix, iy, trig,
    others) -> roots | None`` (the batched-GPU layer supplies its own seeded
    Newton solve)."""
    from .optimize.objective import INVALID_LOSS

    if images is None:
        return INVALID_LOSS
    allow_partial = loss_cfg.missing_img_penalty > 0.0
    sel = select_images(images, obs.n, allow_partial=allow_partial)
    if sel is None:
        return INVALID_LOSS

    n_pred = len(sel)
    if n_pred == 0:
        base = 0.0
    else:
        pred_pos = np.array([[im[0], im[1]] for im in sel], dtype=float)
        pred_mag = np.array([im[2] for im in sel], dtype=float)
        matched_pos, matched_mag, delta, obs_idx = assign_images(
            obs.positions, pred_pos, pred_mag, obs.center_offset)

        perturbers = find_compact_perturbers(scene)
        if perturbers:
            if cluster_fn is None:
                def cluster_fn(sc, ix, iy, trig, others):
                    return solve_cluster(sc, ix, iy, trig, solver, others)
            # matched_pos has the center offset applied; the solver works in
            # the model frame, so recover it from the assignment order.
            co = obs.center_offset
            model_xy = matched_pos - np.asarray(co, dtype=float)
            mm = matched_mag.copy()
            for i in range(len(mm)):
                ix, iy = float(model_xy[i, 0]), float(model_xy[i, 1])
                trig = triggered(perturbers, ix, iy)
                if trig is None:
                    continue
                others = [(float(model_xy[j, 0]), float(model_xy[j, 1]))
                          for j in range(len(mm)) if j != i]
                roots = cluster_fn(scene, ix, iy, trig, others)
                if roots:
                    mm[i] = float(np.sum(np.abs([r[2] for r in roots])))
            matched_mag = mm

        base = ml_loss(delta, matched_mag,
                       obs.magnifications[obs_idx], obs.mag_errors[obs_idx],
                       obs.pos_sigma_mas[obs_idx], loss_cfg)
    n_missing = obs.n - n_pred
    return float(base + n_missing * loss_cfg.missing_img_penalty)
