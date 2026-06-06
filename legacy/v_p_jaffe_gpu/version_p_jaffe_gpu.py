#!/usr/bin/env python3
"""
Version P-Jaffe GPU: Flexible Pseudo-Jaffe Sub-halos (Rhongomyniad, batched).

Batched GPU counterpart of v_p_jaffe_2_0.  The fixed lenses (2 Sersic +
main lens) are evaluated ONCE on a uniform grid; each DE generation
adds every candidate's spherical pseudo-Jaffe sub-halos as a
(C, Kk, ny, nx) tensor in a single GPU pass.  Pseudo-Jaffe =
SIE(rco) − SIE(a_out); spherical SIE has a closed-form deflection and
Hessian so no quadrature is needed.

Runnable standalone:
    python version_p_jaffe_gpu.py
"""

from __future__ import annotations

import os
import sys
import math
import random
import shutil
import subprocess
import time
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Locate Rhongomyniad
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE.parent.parent / "Rhongomyniad",
              _HERE.parent.parent.parent / "Rhongomyniad"):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break

# Make the shared post-processing module importable
sys.path.insert(0, str(_HERE.parent))
import gpu_postprocess as gp  # noqa: E402

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

import rhongomyniad as rh
from rhongomyniad import constants as K
from rhongomyniad.lens_models import LensContext
from rhongomyniad.image_finder import sum_lensmodel
from rhongomyniad.cosmology import Cosmology


# ==========================================================================
# Baseline loader (matches v_p_jaffe_2_0)
# ==========================================================================
def load_baseline_lens_params(directory):
    bestfit_path = os.path.join(directory, "bestfit.dat")
    if not os.path.isfile(bestfit_path):
        raise FileNotFoundError(f"未找到基准参数文件: {bestfit_path}")
    lens_lines, point_params = [], None
    with open(bestfit_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "lens":
                lens_lines.append(parts)
            elif parts[0] == "point":
                point_params = parts
    if not lens_lines:
        raise ValueError(f"bestfit.dat 至少需要 1 行 lens: {bestfit_path}")
    if point_params is None:
        raise ValueError(f"bestfit.dat 缺少 point 行: {bestfit_path}")
    params_dict, sers_count, type_counts, main_key = {}, 0, {}, None
    for parts in lens_lines:
        ltype = parts[1]
        z = float(parts[2])
        raw = [float(v) for v in parts[3:]]
        vals = (raw + [0.0] * 7)[:7]
        idx = len(params_dict) + 1
        if ltype == "sers":
            sers_count += 1
            key = f"sers{sers_count}"
        else:
            type_counts[ltype] = type_counts.get(ltype, 0) + 1
            n = type_counts[ltype]
            key = ltype if n == 1 else f"{ltype}{n}"
            main_key = key
        params_dict[key] = (idx, ltype, z, *vals)
    if main_key is None:
        main_key = list(params_dict.keys())[-1]
    return params_dict, float(point_params[2]), float(point_params[3]), main_key


# ==========================================================================
# Config (override-friendly)
# ==========================================================================
BASELINE_LENS_DIR = ""

active_subhalos = [1, 3, 4]
fine_tuning = False

SEARCH_RADIUS = 0.2
SIG_GUESS = 10.0
SIG_MIN = 0.01
SIG_MAX = 30.0
A_GUESS = 0.050
A_MIN = 0.0001
A_MAX = 0.300
RCO_GUESS = 0.005
RCO_MIN = 0.0
RCO_MAX = 0.05

fine_tuning_configs = {
    1: {"search_radius": 0.1,  "sig_min": 0.001, "sig_max": 10.0,
        "a_min": 0.010, "a_max": 0.200, "rco_min": 0.0001, "rco_max": 0.050},
    2: {"search_radius": 0.4,  "sig_min": 0.01,  "sig_max": 50.0,
        "a_min": 0.005, "a_max": 0.100, "rco_min": 0.0001, "rco_max": 0.020},
    3: {"search_radius": 0.1,  "sig_min": 1.0,   "sig_max": 25.0,
        "a_min": 0.0050, "a_max": 0.100, "rco_min": 0.001, "rco_max": 0.012},
    4: {"search_radius": 0.12, "sig_min": 0.1,   "sig_max": 20.0,
        "a_min": 0.0001, "a_max": 0.100, "rco_min": 5e-5, "rco_max": 0.005},
}

LOSS_COEF_A = 4.0
LOSS_COEF_B = 1.0
LOSS_PENALTY_PL = 1000.0

DE_MAXITER = 600
DE_POPSIZE = 15
DE_ATOL = 1e-5
DE_TOL = 1e-5
DE_SEED = random.randint(1, 100000)
DE_POLISH = True

EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 30

CONSTRAINT_SIGMA = 1.0

# MCMC posterior sampling
MCMC_ENABLED = False
MCMC_NWALKERS = 32
MCMC_NSTEPS = 2000
MCMC_BURNIN = 300
MCMC_THIN = 2
MCMC_PERTURBATION = 0.01
MCMC_PROGRESS = True

# Iteration plot control
Draw_Graph = 1
draw_interval = 5
SHOW_2SIGMA = False
COMPARE_GRAPH = True

OUTPUT_PREFIX = "v_p_jaffe_gpu"

# Observations
obs_positions_mas_list = [[-266.035, 0.427], [118.835, -221.927],
                          [238.324, 227.270], [-126.157, 319.719]]
obs_magnifications_list = [-35.6, 15.7, -7.5, 9.1]
obs_mag_errors_list = [2.1, 1.3, 1.0, 1.1]
obs_pos_sigma_mas_list = [0.41, 0.86, 2.23, 3.11]
center_offset_x = +0.01535
center_offset_y = +0.03220
obs_x_flip = True

# Cosmology + grid
omega = 0.3
lambda_cosmo = 0.7
weos = -1.0
hubble = 0.7
xmin, ymin = -0.5, -0.5
xmax, ymax = 0.5, 0.5
pix_ext = 0.01
pix_poi = 0.2
maxlev = 5

source_z = 0.4090
lens_z = 0.2160
source_x = 2.685497e-03
source_y = 2.443616e-02

lens_params = {
    "sers1": (1, "sers", 0.2160, 9.896617e+09, 2.656977e-03, 2.758473e-02,
              2.986760e-01, 1.124730e+02, 3.939718e-01, 1.057760e+00),
    "sers2": (2, "sers", 0.2160, 2.555580e+10, 2.656977e-03, 2.758473e-02,
              4.242340e-01, 5.396370e+01, 1.538855e+00, 1.000000e+00),
    "sie":   (3, "sie",  0.2160, 1.183382e+02, 2.656977e-03, 2.758473e-02,
              1.571203e-01, 2.920348e+01, 0.0, 0.0),
}
MAIN_LENS_KEY = "sie"


# ==========================================================================
# Setup
# ==========================================================================
print("=" * 70)
print("Version P-Jaffe GPU (Rhongomyniad, batched)")
print("=" * 70)

if BASELINE_LENS_DIR:
    _loaded, _sx, _sy, _mlk = load_baseline_lens_params(BASELINE_LENS_DIR)
    lens_params = _loaded
    source_x = _sx
    source_y = _sy
    MAIN_LENS_KEY = _mlk
    print(f"[baseline] loaded from {BASELINE_LENS_DIR} (main={MAIN_LENS_KEY})")
else:
    print("[baseline] using built-in default (2 Sersic + SIE)")

_x_sign = -1 if obs_x_flip else 1
obs_positions_mas = np.array(obs_positions_mas_list)
obs_positions = np.zeros_like(obs_positions_mas)
obs_positions[:, 0] = _x_sign * obs_positions_mas[:, 0] / 1000.0
obs_positions[:, 1] = obs_positions_mas[:, 1] / 1000.0
center_offset_x = _x_sign * center_offset_x
obs_magnifications = np.array(obs_magnifications_list)
obs_mag_errors = np.array(obs_mag_errors_list)
obs_pos_sigma_mas = np.array(obs_pos_sigma_mas_list)

active_subhalos = sorted(set(int(i) for i in active_subhalos))
for img_idx in active_subhalos:
    if img_idx not in (1, 2, 3, 4):
        raise ValueError(f"active_subhalos 无效索引: {img_idx}")
n_active_subhalos = len(active_subhalos)
n_params = n_active_subhalos * 5

subhalo_configs = {}
for img_idx in active_subhalos:
    if fine_tuning and img_idx in fine_tuning_configs:
        cfg = fine_tuning_configs[img_idx]
    else:
        cfg = {"search_radius": SEARCH_RADIUS,
               "sig_min": SIG_MIN, "sig_max": SIG_MAX,
               "a_min": A_MIN, "a_max": A_MAX,
               "rco_min": RCO_MIN, "rco_max": RCO_MAX}
    subhalo_configs[img_idx] = cfg

timestamp = datetime.now().strftime("%y%m%d_%H%M")
output_dir = timestamp
os.makedirs(output_dir, exist_ok=True)
print(f"output dir: {output_dir}")

device = rh.get_device()
dtype = torch.float64
print(f"device: {device}   finder: {rh.get_finder()}")


# ==========================================================================
# Fixed-lens grid cache (computed ONCE)
# ==========================================================================
def _fixed_lens_tuples():
    tuples = []
    for key, pv in lens_params.items():
        if key.startswith("sers") or key == MAIN_LENS_KEY:
            _, model, z, *p7 = pv
            tuples.append((model, (z, *p7)))
    return tuples


def _build_fixed_cache():
    rh.init(omega, lambda_cosmo, weos, hubble, OUTPUT_PREFIX,
            xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev, verb=0)
    rh.startup_setnum(len(lens_params), 0, 1)
    for key, pv in lens_params.items():
        rh.set_lens(*pv)
    rh.set_point(1, source_z, source_x, source_y)
    rh.model_init(verb=0)

    cosmo = Cosmology(omega=omega, lam=lambda_cosmo, weos=weos, hubble=hubble)
    ctx = LensContext.build(cosmo, zl=lens_z, zs=source_z)

    dp = pix_poi / (2 ** (maxlev - 1))
    nx = int(math.ceil((xmax - xmin) / dp)) + 1
    ny = int(math.ceil((ymax - ymin) / dp)) + 1
    xs_ax = torch.linspace(xmin, xmin + (nx - 1) * dp, nx, device=device, dtype=dtype)
    ys_ax = torch.linspace(ymin, ymin + (ny - 1) * dp, ny, device=device, dtype=dtype)
    gx, gy = torch.meshgrid(xs_ax, ys_ax, indexing="xy")

    fixed_lenses = _fixed_lens_tuples()
    t0 = time.perf_counter()
    ax_f, ay_f, kap_f, g1_f, g2_f, phi_f, _ = sum_lensmodel(
        ctx, fixed_lenses, gx, gy, need_kg=True, need_phi=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"fixed-lens grid {nx}x{ny} built in {(time.perf_counter()-t0)*1000:.1f} ms")
    return dict(ctx=ctx, gx=gx, gy=gy, dp=dp, nx=nx, ny=ny,
                ax=ax_f.contiguous(), ay=ay_f.contiguous(),
                kap=kap_f.contiguous(), g1=g1_f.contiguous(),
                g2=g2_f.contiguous(),
                fixed_lenses=fixed_lenses)


CACHE = _build_fixed_cache()


# ==========================================================================
# Batched spherical pseudo-Jaffe fields (closed form, SIE-sphere difference)
# ==========================================================================
_SIE_BB_FAC = 4.0 * math.pi / (K.ARCSEC2RADIAN * K.C_LIGHT_KMS * K.C_LIGHT_KMS)


def _jaffe_bb(sig, ctx):
    """bb_SIE for q=1 (sigma in km/s).  Same as rhongomyniad.b_sie(…, q=1)."""
    return _SIE_BB_FAC * sig * sig * ctx.dis_ls / ctx.dis_os


def _sie_sphere_terms(dx, dy, r2, s):
    """Spherical SIE with softening s.  Returns (inv_psi_ps, psi, kap_term, shear_term).
       ax_k = bb * dx * inv_psi_ps,  κ_k = 0.5·bb·kap_term,
       γ_term is such that γ1_k = 0.5·bb·(dy²-dx²)·shear_term and γ2_k = -bb·dx·dy·shear_term.
    """
    psi = torch.sqrt(s * s + r2)
    ps = psi + s
    inv_psi_ps = 1.0 / ps
    kap_term = 1.0 / psi
    shear_term = 1.0 / (psi * ps * ps)
    return inv_psi_ps, kap_term, shear_term


def _batched_jaffe_fields(sx, sy, sig, a_out, rco, ctx, gx, gy,
                          smallcore=K.DEF_SMALLCORE):
    """(C, Kk) params -> (ax, ay, kap, g1, g2) each (C, ny, nx)."""
    C, Kk = sx.shape
    ny, nx = gx.shape
    bb = _jaffe_bb(sig, ctx)

    sx_b = sx.view(C, Kk, 1, 1)
    sy_b = sy.view(C, Kk, 1, 1)
    bb_b = bb.view(C, Kk, 1, 1)
    # Enforce softening floors per glafic: if rco < smallcore treat as smallcore;
    # if a <= rco the lens returns zero fields.
    rco_f = torch.clamp(rco, min=smallcore)
    a_out_f = a_out
    valid = (a_out_f > rco_f).view(C, Kk, 1, 1)
    rco_b = rco_f.view(C, Kk, 1, 1)
    a_b = a_out_f.view(C, Kk, 1, 1)

    dx = gx.view(1, 1, ny, nx) - sx_b
    dy = gy.view(1, 1, ny, nx) - sy_b
    r2 = dx * dx + dy * dy

    ips_c, kt_c, st_c = _sie_sphere_terms(dx, dy, r2, rco_b)
    ips_a, kt_a, st_a = _sie_sphere_terms(dx, dy, r2, a_b)

    d_ips = torch.where(valid, ips_c - ips_a, torch.zeros_like(ips_c))
    d_kt = torch.where(valid, kt_c - kt_a, torch.zeros_like(kt_c))
    d_st = torch.where(valid, st_c - st_a, torch.zeros_like(st_c))

    ax_k = bb_b * dx * d_ips
    ay_k = bb_b * dy * d_ips
    kap_k = 0.5 * bb_b * d_kt
    g1_k = 0.5 * bb_b * (dy * dy - dx * dx) * d_st
    g2_k = -bb_b * dx * dy * d_st

    return (ax_k.sum(dim=1), ay_k.sum(dim=1),
            kap_k.sum(dim=1), g1_k.sum(dim=1), g2_k.sum(dim=1))


def _jaffe_pointwise(xi, yi, cand_cfg, sx_t, sy_t, bb_all, a_all, rco_all,
                     smallcore=K.DEF_SMALLCORE):
    """Per-candidate pseudo-Jaffe fields at (xi, yi).  Each (Ncand,)."""
    sub_sx = sx_t[cand_cfg]
    sub_sy = sy_t[cand_cfg]
    sub_bb = bb_all[cand_cfg]
    sub_a = a_all[cand_cfg]
    sub_rco = torch.clamp(rco_all[cand_cfg], min=smallcore)
    valid = sub_a > sub_rco

    dx = xi.unsqueeze(1) - sub_sx
    dy = yi.unsqueeze(1) - sub_sy
    r2 = dx * dx + dy * dy

    ips_c, kt_c, st_c = _sie_sphere_terms(dx, dy, r2, sub_rco)
    ips_a, kt_a, st_a = _sie_sphere_terms(dx, dy, r2, sub_a)

    d_ips = torch.where(valid, ips_c - ips_a, torch.zeros_like(ips_c))
    d_kt = torch.where(valid, kt_c - kt_a, torch.zeros_like(kt_c))
    d_st = torch.where(valid, st_c - st_a, torch.zeros_like(st_c))

    ax = (sub_bb * dx * d_ips).sum(dim=1)
    ay = (sub_bb * dy * d_ips).sum(dim=1)
    kap = 0.5 * (sub_bb * d_kt).sum(dim=1)
    g1 = 0.5 * (sub_bb * (dy * dy - dx * dx) * d_st).sum(dim=1)
    g2 = -(sub_bb * dx * dy * d_st).sum(dim=1)
    return ax, ay, kap, g1, g2


# ==========================================================================
# Triangle test + batched image solve
# ==========================================================================
def _tri_contains(xs, ys, ax, ay, bx, by, cx, cy):
    d1x = xs - ax; d1y = ys - ay
    d2x = xs - bx; d2y = ys - by
    d3x = xs - cx; d3y = ys - cy
    d12 = d1x * d2y - d1y * d2x
    d23 = d2x * d3y - d2y * d3x
    d31 = d3x * d1y - d3y * d1x
    return (((d12 >= 0) & (d23 >= 0) & (d31 >= 0))
            | ((d12 <= 0) & (d23 <= 0) & (d31 <= 0)))


def batched_point_solve(sx_t, sy_t, sig_t, a_t, rco_t,
                        xs_src, ys_src, cache, max_iter=8):
    C, Kk = sx_t.shape
    gx, gy = cache["gx"], cache["gy"]
    ny, nx = gx.shape
    dp = cache["dp"]
    ctx = cache["ctx"]

    bb_all = _jaffe_bb(sig_t, ctx)

    ax_k, ay_k, kap_k, g1_k, g2_k = _batched_jaffe_fields(
        sx_t, sy_t, sig_t, a_t, rco_t, ctx, gx, gy)
    ax = cache["ax"].unsqueeze(0) + ax_k
    ay = cache["ay"].unsqueeze(0) + ay_k

    sx_grid = gx.unsqueeze(0) - ax
    sy_grid = gy.unsqueeze(0) - ay
    bl_x = sx_grid[:, :-1, :-1]; bl_y = sy_grid[:, :-1, :-1]
    br_x = sx_grid[:, :-1,  1:]; br_y = sy_grid[:, :-1,  1:]
    tl_x = sx_grid[:,  1:, :-1]; tl_y = sy_grid[:,  1:, :-1]
    tr_x = sx_grid[:,  1:,  1:]; tr_y = sy_grid[:,  1:,  1:]
    in_A = _tri_contains(xs_src, ys_src, bl_x, bl_y, tr_x, tr_y, br_x, br_y)
    in_B = _tri_contains(xs_src, ys_src, bl_x, bl_y, tr_x, tr_y, tl_x, tl_y)
    ox = gx[:-1, :-1].unsqueeze(0).expand_as(in_A)
    oy = gy[:-1, :-1].unsqueeze(0).expand_as(in_A)
    idx_A = torch.nonzero(in_A, as_tuple=False)
    idx_B = torch.nonzero(in_B, as_tuple=False)
    if idx_A.numel() + idx_B.numel() == 0:
        return [[] for _ in range(C)]

    def _seeds(idx, off_x, off_y):
        cfg, j, i = idx[:, 0], idx[:, 1], idx[:, 2]
        return cfg, ox[cfg, j, i] + off_x * dp, oy[cfg, j, i] + off_y * dp

    cA, xA, yA = _seeds(idx_A, 0.667, 0.333)
    cB, xB, yB = _seeds(idx_B, 0.333, 0.667)
    cand_cfg = torch.cat([cA, cB])
    cand_x0 = torch.cat([xA, xB])
    cand_y0 = torch.cat([yA, yB])

    xi = cand_x0.clone()
    yi = cand_y0.clone()
    fixed_lenses = cache["fixed_lenses"]

    for _ in range(max_iter):
        ax_fix, ay_fix, kap_fix, g1_fix, g2_fix, _, _ = sum_lensmodel(
            ctx, fixed_lenses, xi, yi, need_kg=True, need_phi=False)
        ax_c, ay_c, kap_c, g1_c, g2_c = _jaffe_pointwise(
            xi, yi, cand_cfg, sx_t, sy_t, bb_all, a_t, rco_t)
        ax_t = ax_fix + ax_c
        ay_t = ay_fix + ay_c
        kap_t = kap_fix + kap_c
        g1_t = g1_fix + g1_c
        g2_t = g2_fix + g2_c
        pxx = kap_t + g1_t
        pyy = kap_t - g1_t
        pxy = g2_t
        ff = xs_src - xi + ax_t
        gg = ys_src - yi + ay_t
        mm = (1.0 - pxx) * (1.0 - pyy) - pxy * pxy
        xi = xi + ((1.0 - pyy) * ff + pxy * gg) / mm
        yi = yi + ((1.0 - pxx) * gg + pxy * ff) / mm

    muinv = (1.0 - kap_t) ** 2 - (g1_t * g1_t + g2_t * g2_t)
    mag = 1.0 / (muinv + K.DEF_IMAG_CEIL)

    dist2 = (xi - cand_x0) ** 2 + (yi - cand_y0) ** 2
    keep = dist2 <= (2.0 * dp * dp)

    cfg_cpu = cand_cfg.cpu().numpy()
    xi_cpu = xi.cpu().numpy()
    yi_cpu = yi.cpu().numpy()
    mag_cpu = mag.cpu().numpy()
    keep_cpu = keep.cpu().numpy()

    out = [[] for _ in range(C)]
    for i in range(len(cfg_cpu)):
        if not keep_cpu[i]:
            continue
        c = int(cfg_cpu[i])
        x, y, m = float(xi_cpu[i]), float(yi_cpu[i]), float(mag_cpu[i])
        dup = False
        for xj, yj, mj in out[c]:
            if ((x - xj) ** 2 + (y - yj) ** 2) / max(abs(m * mj), 1e-300) \
                    <= 10.0 * K.DEF_MAX_POI_TOL ** 2:
                dup = True
                break
        if not dup:
            out[c].append((x, y, m))

    result = []
    for imgs in out:
        if len(imgs) == 5:
            central = min(range(5), key=lambda k: abs(imgs[k][2]))
            imgs = [im for k, im in enumerate(imgs) if k != central]
        result.append(imgs)
    return result


# ==========================================================================
# DE objective (vectorised)
# ==========================================================================
def vectorised_chi2(params_arr):
    if params_arr.ndim == 1:
        params_arr = params_arr[:, None]
    popsize = params_arr.shape[1]
    sx_arr = np.stack([params_arr[i * 5]     for i in range(n_active_subhalos)], axis=1)
    sy_arr = np.stack([params_arr[i * 5 + 1] for i in range(n_active_subhalos)], axis=1)
    sg_arr = np.stack([params_arr[i * 5 + 2] for i in range(n_active_subhalos)], axis=1)
    a_arr  = np.stack([params_arr[i * 5 + 3] for i in range(n_active_subhalos)], axis=1)
    rc_arr = np.stack([params_arr[i * 5 + 4] for i in range(n_active_subhalos)], axis=1)
    sx_t = torch.tensor(sx_arr, device=device, dtype=dtype)
    sy_t = torch.tensor(sy_arr, device=device, dtype=dtype)
    sg_t = torch.tensor(sg_arr, device=device, dtype=dtype)
    a_t  = torch.tensor(a_arr,  device=device, dtype=dtype)
    rc_t = torch.tensor(rc_arr, device=device, dtype=dtype)

    all_images = batched_point_solve(sx_t, sy_t, sg_t, a_t, rc_t,
                                     float(source_x), float(source_y), CACHE)

    # Invalid candidates: any sub-halo with a_out <= rco gets a huge loss.
    invalid_mask = np.any(a_arr <= rc_arr, axis=1)

    loss = np.empty(popsize, dtype=np.float64)
    for c in range(popsize):
        if invalid_mask[c]:
            loss[c] = 1e15
            continue
        imgs = all_images[c]
        if len(imgs) != 4:
            loss[c] = 1e15
            continue
        pred_pos = np.array([[im[0], im[1]] for im in imgs]) + np.array([center_offset_x, center_offset_y])
        pred_mag = np.array([im[2] for im in imgs])
        distances = cdist(obs_positions, pred_pos)
        row_ind, col_ind = linear_sum_assignment(distances)
        pp = pred_pos[col_ind[np.argsort(row_ind)]]
        pm = pred_mag[col_ind[np.argsort(row_ind)]]
        delta_mas = np.sqrt(np.sum(((pp - obs_positions) * 1000) ** 2, axis=1))
        chi2_pos = np.sum((delta_mas / obs_pos_sigma_mas) ** 2)
        chi2_mag = np.sum(((pm - obs_magnifications) / obs_mag_errors) ** 2)
        penalty = 0.0
        for i in range(4):
            if delta_mas[i] > obs_pos_sigma_mas[i]:
                penalty += LOSS_PENALTY_PL * delta_mas[i]
        loss[c] = LOSS_COEF_A * chi2_pos + LOSS_COEF_B * chi2_mag + penalty
    return loss


# ==========================================================================
# Run DE
# ==========================================================================
def _build_bounds():
    bounds = []
    for img_idx in active_subhalos:
        cfg = subhalo_configs[img_idx]
        xc = obs_positions[img_idx - 1, 0]
        yc = obs_positions[img_idx - 1, 1]
        bounds.append((xc - cfg["search_radius"], xc + cfg["search_radius"]))
        bounds.append((yc - cfg["search_radius"], yc + cfg["search_radius"]))
        bounds.append((cfg["sig_min"], cfg["sig_max"]))
        bounds.append((cfg["a_min"], cfg["a_max"]))
        bounds.append((cfg["rco_min"], cfg["rco_max"]))
    return bounds


def _solve_for_subhalos(subhalos):
    """subhalos: list of (x, y, sig, a, rco)."""
    if subhalos:
        sx_t = torch.tensor([[s[0] for s in subhalos]], device=device, dtype=dtype)
        sy_t = torch.tensor([[s[1] for s in subhalos]], device=device, dtype=dtype)
        sg_t = torch.tensor([[s[2] for s in subhalos]], device=device, dtype=dtype)
        a_t  = torch.tensor([[s[3] for s in subhalos]], device=device, dtype=dtype)
        rc_t = torch.tensor([[s[4] for s in subhalos]], device=device, dtype=dtype)
    else:
        sx_t = torch.tensor([[1e3]], device=device, dtype=dtype)
        sy_t = torch.tensor([[1e3]], device=device, dtype=dtype)
        sg_t = torch.tensor([[0.001]], device=device, dtype=dtype)
        a_t  = torch.tensor([[1e-3]], device=device, dtype=dtype)
        rc_t = torch.tensor([[1e-4]], device=device, dtype=dtype)
    imgs = batched_point_solve(sx_t, sy_t, sg_t, a_t, rc_t,
                               float(source_x), float(source_y), CACHE)[0]
    return gp.match_images(imgs, obs_positions,
                           center_offset_x, center_offset_y, n_obs=4)


def main():
    bounds = _build_bounds()
    print("\n" + "=" * 70)
    print("Step 1: baseline (no sub-halos)")
    print("=" * 70)
    base_pos, base_mag, base_delta = _solve_for_subhalos([])
    if base_delta is None:
        base_delta = np.full(4, 1e3); base_mag = np.zeros(4); base_pos = np.zeros((4, 2))
    base_pos_chi2 = float(np.sum((base_delta / obs_pos_sigma_mas) ** 2))
    base_mag_chi2 = float(np.sum(((base_mag - obs_magnifications) / obs_mag_errors) ** 2))
    base_total_chi2 = base_pos_chi2 + base_mag_chi2
    print(f"  pos RMS    : {np.sqrt(np.mean(base_delta**2)):.3f} mas")
    print(f"  chi2_pos   : {base_pos_chi2:.2f}")
    print(f"  chi2_mag   : {base_mag_chi2:.2f}")
    print(f"  chi2_total : {base_total_chi2:.2f}")

    print("\n" + "=" * 70)
    print(f"Step 2: differential evolution (ndim={n_params}, popsize_mult={DE_POPSIZE},"
          f" pop={DE_POPSIZE * n_params}, seed={DE_SEED})")
    print("=" * 70)
    print(f"  device: {device}   finder: {rh.get_finder()}")

    rng = np.random.default_rng(0)
    pop = DE_POPSIZE * n_params
    test = np.empty((n_params, pop))
    for i in range(n_active_subhalos):
        img_idx = active_subhalos[i]
        cfg = subhalo_configs[img_idx]
        xc = obs_positions[img_idx - 1, 0]
        yc = obs_positions[img_idx - 1, 1]
        sr = cfg["search_radius"]
        test[i * 5]     = xc + rng.uniform(-sr, sr, pop)
        test[i * 5 + 1] = yc + rng.uniform(-sr, sr, pop)
        test[i * 5 + 2] = rng.uniform(cfg["sig_min"], cfg["sig_max"], pop)
        test[i * 5 + 3] = rng.uniform(cfg["a_min"], cfg["a_max"], pop)
        test[i * 5 + 4] = rng.uniform(cfg["rco_min"], cfg["rco_max"], pop)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = vectorised_chi2(test)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  warmup: {pop} members in {dt*1000:.1f} ms ({dt/pop*1000:.2f} ms/member)")

    np.random.seed(DE_SEED)
    de_kwargs = dict(
        maxiter=DE_MAXITER, popsize=DE_POPSIZE,
        atol=DE_ATOL, tol=DE_TOL, polish=DE_POLISH,
        disp=False, vectorized=True, updating="deferred")
    import inspect as _inspect
    if "rng" in _inspect.signature(DifferentialEvolutionSolver.__init__).parameters:
        de_kwargs["rng"] = np.random.default_rng(DE_SEED)
    else:
        de_kwargs["seed"] = DE_SEED
    solver = DifferentialEvolutionSolver(vectorised_chi2, bounds, **de_kwargs)

    # Use sigma (idx 2) as the "main" parameter for iteration plots
    sig_range = []
    for img_idx in active_subhalos:
        cfg = subhalo_configs[img_idx]
        sig_range.extend([cfg["sig_min"], cfg["sig_max"]])

    if Draw_Graph:
        gp.plot_iteration_subhalo_param(
            solver.population.copy(), 0, output_dir,
            active_subhalos, params_per_subhalo=5, main_param_idx=2,
            draw_interval=draw_interval,
            main_param_label=r"$\sigma$",
            main_param_range=sig_range, bounds=bounds)

    iteration = 1
    prev_best = float(np.min(solver.population_energies))
    converged_count = 0
    t_de = time.perf_counter()
    while True:
        try:
            solver.__next__()
        except StopIteration:
            print(f"  DE converged internally at iter {iteration}")
            break
        cur_best = float(np.min(solver.population_energies))
        abs_change = abs(cur_best - prev_best)
        rel_change = abs_change / abs(prev_best) if abs(prev_best) > 1e-10 else float("inf")
        if iteration % 5 == 0 or cur_best < prev_best:
            print(f"  iter {iteration:4d}: best={cur_best:.6f}  "
                  f"|Δ|={abs_change:.3e}  rel={rel_change:.3e}")

        converged_this = (abs_change < DE_ATOL) or (rel_change < DE_TOL)
        if EARLY_STOPPING:
            if converged_this:
                converged_count += 1
                if converged_count >= EARLY_STOP_PATIENCE:
                    print(f"  early-stop after {converged_count} stable iters")
                    if Draw_Graph:
                        gp.plot_iteration_subhalo_param(
                            solver.population.copy(), iteration, output_dir,
                            active_subhalos, 5, 2, draw_interval=draw_interval,
                            main_param_label=r"$\sigma$",
                            main_param_range=sig_range, bounds=bounds)
                    break
            else:
                converged_count = 0

        if Draw_Graph:
            gp.plot_iteration_subhalo_param(
                solver.population.copy(), iteration, output_dir,
                active_subhalos, 5, 2, draw_interval=draw_interval,
                main_param_label=r"$\sigma$",
                main_param_range=sig_range, bounds=bounds)

        prev_best = cur_best
        iteration += 1
        if iteration > DE_MAXITER:
            print(f"  reached DE_MAXITER={DE_MAXITER}")
            break
    de_dt = time.perf_counter() - t_de

    best_x = solver.x
    best_loss = float(np.min(solver.population_energies))
    print(f"\n  DE finished in {de_dt:.1f}s  iters={iteration}  loss={best_loss:.4f}")

    subhalos = [(float(best_x[i * 5]), float(best_x[i * 5 + 1]),
                 float(best_x[i * 5 + 2]),
                 float(best_x[i * 5 + 3]), float(best_x[i * 5 + 4]))
                for i in range(n_active_subhalos)]

    print("\n" + "=" * 70)
    print("Step 3: best-fit analysis")
    print("=" * 70)
    best_pos, best_mag, best_delta = _solve_for_subhalos(subhalos)
    if best_pos is None:
        print("  warn: best-fit solve did not yield 4 images")
        return
    best_pos_chi2 = float(np.sum((best_delta / obs_pos_sigma_mas) ** 2))
    best_mag_chi2 = float(np.sum(((best_mag - obs_magnifications) / obs_mag_errors) ** 2))
    best_total_chi2 = best_pos_chi2 + best_mag_chi2
    improvement = ((base_total_chi2 - best_total_chi2) / base_total_chi2 * 100
                   if base_total_chi2 > 0 else 0.0)
    print(f"  pos RMS    : {np.sqrt(np.mean(best_delta**2)):.3f} mas")
    print(f"  chi2_pos   : {best_pos_chi2:.2f}  (baseline {base_pos_chi2:.2f})")
    print(f"  chi2_mag   : {best_mag_chi2:.2f}  (baseline {base_mag_chi2:.2f})")
    print(f"  chi2_total : {best_total_chi2:.2f}  (baseline {base_total_chi2:.2f})")
    print(f"  improvement: {improvement:.1f}%  (on chi2_total)")
    for i, (x, y, sig, a, rco) in enumerate(subhalos):
        img_idx = active_subhalos[i]
        print(f"  sub-halo {img_idx}: ({x:+.6f}, {y:+.6f}) arcsec  "
              f"sig={sig:.3f}  a={a:.5f}  rco={rco:.5f}")
    constraint_ok = bool(np.all(best_delta <= obs_pos_sigma_mas * CONSTRAINT_SIGMA))
    print(f"  constraint satisfied: {constraint_ok}")

    # ---- best params file -------------------------------------------------
    params_path = os.path.join(output_dir, f"{OUTPUT_PREFIX}_best_params.txt")
    with open(params_path, "w") as f:
        f.write(f"# Version P-Jaffe GPU (Rhongomyniad batched)\n")
        f.write(f"# DE seed: {DE_SEED}\n")
        f.write(f"# active_subhalos = {active_subhalos}\n")
        f.write(f"# fine_tuning = {fine_tuning}\n")
        f.write(f"# finder = {rh.get_finder()}  device = {rh.get_device()}\n")
        f.write(f"# DE: iters={iteration} loss={best_loss:.6f}\n\n")
        for i, (x, y, sig, a, rco) in enumerate(subhalos):
            img_idx = active_subhalos[i]
            f.write(f"# Sub-halo at Image {img_idx}\n")
            f.write(f"x_sub{img_idx} = {x:.10e}  # arcsec\n")
            f.write(f"y_sub{img_idx} = {y:.10e}  # arcsec\n")
            f.write(f"sig_sub{img_idx} = {sig:.10e}\n")
            f.write(f"a_sub{img_idx} = {a:.10e}\n")
            f.write(f"rco_sub{img_idx} = {rco:.10e}\n\n")
        f.write("# Performance\n")
        f.write(f"chi2_pos_base = {base_pos_chi2:.4f}\n")
        f.write(f"chi2_mag_base = {base_mag_chi2:.4f}\n")
        f.write(f"chi2_total_base = {base_total_chi2:.4f}\n")
        f.write(f"chi2_pos_best = {best_pos_chi2:.4f}\n")
        f.write(f"chi2_mag_best = {best_mag_chi2:.4f}\n")
        f.write(f"chi2_total_best = {best_total_chi2:.4f}\n")
        f.write(f"improvement = {improvement:.2f}%\n")
        f.write(f"constraint_satisfied = {constraint_ok}\n")
    print(f"  saved best params: {params_path}")

    # ---- MCMC -------------------------------------------------------------
    # P-Jaffe parameterizes by sigma, not log-mass; skip the per-subhalo
    # mass posterior plot (mass_param_indices=None).
    param_names = []
    corner_labels = []
    for img_idx in active_subhalos:
        param_names += [f"x_{img_idx}", f"y_{img_idx}",
                        f"sig_{img_idx}", f"a_{img_idx}", f"rco_{img_idx}"]
        corner_labels += [f"$x_{img_idx}$", f"$y_{img_idx}$",
                          rf"$\sigma_{img_idx}$",
                          f"$a_{{{img_idx}}}$", f"$r_{{co,{img_idx}}}$"]

    if MCMC_ENABLED:
        bounds_arr = np.asarray(bounds)

        def log_prob(p_batch):
            single = (p_batch.ndim == 1)
            arr = p_batch.reshape(1, -1) if single else p_batch
            in_bounds = np.all((arr >= bounds_arr[:, 0]) &
                               (arr <= bounds_arr[:, 1]), axis=1)
            out = np.full(len(arr), -np.inf)
            if not np.any(in_bounds):
                return out[0] if single else out
            sub = arr[in_bounds].T
            losses = vectorised_chi2(sub)
            valid = losses < 1e10
            res_in = np.where(valid, -0.5 * losses, -np.inf)
            out[in_bounds] = res_in
            return out[0] if single else out

        gp.run_mcmc_pipeline(
            log_prob, log_prob_vectorized=True,
            bounds=bounds, best_x=best_x,
            output_dir=output_dir, output_prefix=OUTPUT_PREFIX,
            param_names=param_names, corner_labels=corner_labels,
            mass_param_indices=None,
            config=dict(NWALKERS=MCMC_NWALKERS, NSTEPS=MCMC_NSTEPS,
                        BURNIN=MCMC_BURNIN, THIN=MCMC_THIN,
                        PERTURBATION=MCMC_PERTURBATION,
                        PROGRESS=MCMC_PROGRESS, WORKERS=1),
            title_prefix="P-Jaffe")

    # ---- Critical curves + triptych --------------------------------------
    print("\n" + "=" * 70)
    print("Step 4: result plots")
    print("=" * 70)
    base_lens_lines = list(lens_params.values())
    extra_lens_lines = []
    for i, (x, y, sig, a, rco) in enumerate(subhalos):
        idx = len(base_lens_lines) + 1 + i
        # jaffe layout: (idx, model, z, sig, x, y, e, pa, a_outer, rco_inner)
        extra_lens_lines.append(
            (idx, "jaffe", lens_z, sig, x, y, 0.0, 0.0, a, rco))

    crit_segments, caus_segments = gp.compute_critical_curves(
        output_dir, OUTPUT_PREFIX,
        cosmo=(omega, lambda_cosmo, weos, hubble),
        grid=(xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev),
        base_lens_lines=base_lens_lines, extra_lens_lines=extra_lens_lines,
        source_z=source_z, source_x=float(source_x), source_y=float(source_y))

    triptych_path = os.path.join(output_dir, f"result_{OUTPUT_PREFIX}.png")
    # The shared triptych helper expects (x, y, mass) for label tags;
    # use sigma in the third slot for visual labelling.
    sub_labels = [(s[0], s[1], s[2]) for s in subhalos]
    gp.write_result_triptych(
        triptych_path,
        suptitle=f"iPTF16geu: {n_active_subhalos} P-Jaffe Sub-halos (GPU)",
        obs_positions=obs_positions, pred_positions=best_pos,
        delta_pos_mas=best_delta, sigma_pos_mas=obs_pos_sigma_mas,
        mu_obs=obs_magnifications, mu_obs_err=obs_mag_errors,
        mu_pred=best_mag,
        crit_segments=crit_segments, caus_segments=caus_segments,
        subhalo_positions=sub_labels, show_2sigma=SHOW_2SIGMA)

    if COMPARE_GRAPH and n_active_subhalos > 0:
        compare_path = os.path.join(output_dir,
                                    f"result_{OUTPUT_PREFIX}_compare.png")
        gp.write_compare_triptych(
            compare_path,
            suptitle=f"iPTF16geu: Baseline vs {n_active_subhalos} P-Jaffe Sub-halos (GPU)",
            obs_positions=obs_positions, pred_positions=best_pos,
            delta_pos_baseline=base_delta, delta_pos_optimized=best_delta,
            sigma_pos_mas=obs_pos_sigma_mas,
            mu_obs=obs_magnifications, mu_obs_err=obs_mag_errors,
            mu_pred_baseline=base_mag, mu_pred_optimized=best_mag,
            crit_segments=crit_segments, caus_segments=caus_segments,
            subhalo_positions=sub_labels, show_2sigma=SHOW_2SIGMA)

    # ---- Glafic CLI verification -----------------------------------------
    gp.run_glafic_and_compare(
        output_dir, OUTPUT_PREFIX,
        cosmo=(omega, lambda_cosmo, weos, hubble),
        grid=(xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev),
        base_lens_lines=base_lens_lines, extra_lens_lines=extra_lens_lines,
        source_z=source_z, source_x=float(source_x), source_y=float(source_y),
        obs_positions=obs_positions,
        center_offset_x=center_offset_x, center_offset_y=center_offset_y,
        best_pos_py=best_pos, best_mag_py=best_mag,
        header_comment=(f"{OUTPUT_PREFIX} verification\n"
                        f"active_subhalos = {active_subhalos}"))

    print("\n" + "=" * 70)
    print(f"Done. results in {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
