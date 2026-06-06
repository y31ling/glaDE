#!/usr/bin/env python3
"""
Version None GPU — No Subhalo: Source / Lens Parameter Optimization (Rhongomyniad, batched).

GPU counterpart of v_none_1_0.  Optimises source position and/or main-lens
parameters via differential evolution.  Two fast paths:

  * source_modify=True, lens_modify=False:
      The full lens stack is constant, so the grid (ax, ay, kap, g1, g2, phi)
      is cached ONCE on the GPU and all DE candidates share it.  Each
      candidate only differs in source position — the triangle finder and
      Newton refinement are run in one batched sweep over the population.

  * lens_modify=True:
      The grid must be recomputed per candidate (sers/sie analytic kernels
      still run on GPU, just once per candidate).  Triangle finding and
      Newton refinement remain batched across the (C, ny, nx) field stack.

Runnable standalone:
    python version_none_gpu.py
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
# Locate Rhongomyniad (sibling of the "legacy" folder inside the glade repo)
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
from rhongomyniad.lens_models import LensContext, dispatch
from rhongomyniad.image_finder import sum_lensmodel
from rhongomyniad.cosmology import Cosmology


# ==========================================================================
# Baseline loader (matches v_none_1_0)
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
    if len(lens_lines) < 1:
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
# Config (override-friendly top-level assignments)
# ==========================================================================
BASELINE_LENS_DIR = ""

CONSTRAINT_SIGMA = 1.0
PENALTY_COEFFICIENT = 1000.0

# Optimisation targets
source_modify = True
lens_modify = True

# fine_tuning=False: percentage / delta bounds.
# fine_tuning=True : explicit bounds via lens_optimize_bounds / source_*_bounds.
fine_tuning = False

# --- fine_tuning=False ---
modify_percentage = 0.1
source_x_delta = 0.01
source_y_delta = 0.01

# --- fine_tuning=True ---
source_x_bounds = [-0.3, 0.3]
source_y_bounds = [-0.3, 0.3]
lens_optimize_bounds = {}

# Loss function (same as v_none_1_0)
LOSS_COEF_A = 1.0
LOSS_COEF_B = 1.0
LOSS_PENALTY_PL = 1000.0

# DE settings
DE_MAXITER = 200
DE_POPSIZE = 15
DE_ATOL = 1e-4
DE_TOL = 1e-6
DE_SEED = random.randint(1, 100000)
DE_POLISH = True

EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 30

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

OUTPUT_PREFIX = "v_none_gpu"

# Observations (iPTF16geu)
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
# Setup (runs after override injection)
# ==========================================================================
print("=" * 70)
print("Version None GPU (Rhongomyniad, batched)")
print("=" * 70)

if BASELINE_LENS_DIR:
    _loaded, _sx, _sy, _mlk = load_baseline_lens_params(BASELINE_LENS_DIR)
    lens_params = _loaded
    source_x = _sx
    source_y = _sy
    MAIN_LENS_KEY = _mlk
    print(f"[baseline] loaded from {BASELINE_LENS_DIR} (main={MAIN_LENS_KEY})")
else:
    print("[baseline] built-in default (2 Sersic + SIE)")

_x_sign = -1 if obs_x_flip else 1
obs_positions_mas = np.array(obs_positions_mas_list)
obs_positions = np.zeros_like(obs_positions_mas)
obs_positions[:, 0] = _x_sign * obs_positions_mas[:, 0] / 1000.0
obs_positions[:, 1] = obs_positions_mas[:, 1] / 1000.0
# v_none_1_0 zeroes center_offset_x after building obs_positions.
center_offset_x = 0.0
obs_magnifications = np.array(obs_magnifications_list)
obs_mag_errors = np.array(obs_mag_errors_list)
obs_pos_sigma_mas = np.array(obs_pos_sigma_mas_list)
n_obs = len(obs_positions)

lens_params_ref = {k: list(v) for k, v in lens_params.items()}
source_x_ref = float(source_x)
source_y_ref = float(source_y)

timestamp = datetime.now().strftime("%y%m%d_%H%M")
output_dir = timestamp
os.makedirs(output_dir, exist_ok=True)
print(f"output dir: {output_dir}")

device = rh.get_device()
dtype = torch.float64
print(f"device: {device}   finder: {rh.get_finder()}")


# ==========================================================================
# Optimisation variable mapping (bounds + pmap, identical logic to v_none_1_0)
# ==========================================================================
def build_optimization():
    bounds, pmap = [], []
    if source_modify:
        sx, sy = source_x_ref, source_y_ref
        if fine_tuning:
            bounds += [(sx + source_x_bounds[0], sx + source_x_bounds[1]),
                       (sy + source_y_bounds[0], sy + source_y_bounds[1])]
        else:
            bounds += [(sx - source_x_delta, sx + source_x_delta),
                       (sy - source_y_delta, sy + source_y_delta)]
        pmap += [("src_x", None, None), ("src_y", None, None)]

    if lens_modify:
        for key, pv in lens_params_ref.items():
            ps = pv[3:]
            if fine_tuning:
                kbounds = lens_optimize_bounds.get(key, [None] * 7)
                for pi in range(7):
                    b = kbounds[pi] if pi < len(kbounds) else None
                    if b is not None:
                        bounds.append(tuple(b))
                        pmap.append(("lens", key, pi))
            else:
                for pi, val in enumerate(ps[:7]):
                    if abs(val) > 1e-30:
                        d = abs(val) * modify_percentage
                        bounds.append((val - d, val + d))
                        pmap.append(("lens", key, pi))

    return bounds, pmap


BOUNDS, PMAP = build_optimization()
NDIM = len(BOUNDS)
print(f"DE ndim={NDIM}  source_modify={source_modify}  lens_modify={lens_modify}  "
      f"fine_tuning={fine_tuning}")


# ==========================================================================
# Grid cache (fixed lens fast path when lens_modify=False)
# ==========================================================================
def _lens_tuples_from_dict(lp_dict):
    """Convert lp_dict[key] = (idx, model, z, p1..p7) into rhongomyniad input."""
    tuples = []
    for _, pv in lp_dict.items():
        model = pv[1]
        p7 = pv[3:]
        tuples.append((model, (float(pv[2]), *(float(x) for x in p7))))
    return tuples


def _build_grid():
    dp = pix_poi / (2 ** (maxlev - 1))
    nx = int(math.ceil((xmax - xmin) / dp)) + 1
    ny = int(math.ceil((ymax - ymin) / dp)) + 1
    xs_ax = torch.linspace(xmin, xmin + (nx - 1) * dp, nx, device=device, dtype=dtype)
    ys_ax = torch.linspace(ymin, ymin + (ny - 1) * dp, ny, device=device, dtype=dtype)
    gx, gy = torch.meshgrid(xs_ax, ys_ax, indexing="xy")
    return gx, gy, dp, nx, ny


GX, GY, DP, NX, NY = _build_grid()

# Build a single cosmology context; lens_z and source_z are both fixed.
_COSMO = Cosmology(omega=omega, lam=lambda_cosmo, weos=weos, hubble=hubble)
CTX = LensContext.build(_COSMO, zl=lens_z, zs=source_z)

# Fixed fields (only valid when lens_modify=False, reused every DE call)
FIXED_FIELDS = None
if not lens_modify:
    fixed_tuples = _lens_tuples_from_dict(lens_params_ref)
    t0 = time.perf_counter()
    ax_f, ay_f, kap_f, g1_f, g2_f, phi_f, _ = sum_lensmodel(
        CTX, fixed_tuples, GX, GY, need_kg=True, need_phi=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    FIXED_FIELDS = dict(ax=ax_f.contiguous(), ay=ay_f.contiguous(),
                        kap=kap_f.contiguous(), g1=g1_f.contiguous(),
                        g2=g2_f.contiguous(), phi=phi_f.contiguous(),
                        tuples=fixed_tuples)
    print(f"fixed-lens grid {NX}x{NY} built in {(time.perf_counter()-t0)*1000:.1f} ms")
else:
    print(f"grid {NX}x{NY}; lens fields recomputed per candidate "
          f"(lens_modify=True)")


# ==========================================================================
# Candidate expansion: DE parameter vector -> (sx, sy, lens_tuples)
# ==========================================================================
def _decode_candidate(x_vec):
    sx = source_x_ref
    sy = source_y_ref
    cur_lp = {k: list(v) for k, v in lens_params_ref.items()}
    for i, (ptype, key, pi) in enumerate(PMAP):
        if ptype == "src_x":
            sx = float(x_vec[i])
        elif ptype == "src_y":
            sy = float(x_vec[i])
        else:
            cur_lp[key][3 + pi] = float(x_vec[i])
    return float(sx), float(sy), cur_lp


# ==========================================================================
# Triangle containment helper (vectorised, works for any leading batch dims)
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


# ==========================================================================
# Batched image solve (source-only fast path — fixed lens cache)
# ==========================================================================
def batched_source_solve(sx_arr, sy_arr, max_iter=8):
    """
    sx_arr, sy_arr: numpy (C,) candidate source positions (shared fixed lens).
    Returns list of length C, each element = list of (x, y, mag, td) tuples.
    """
    C = len(sx_arr)
    ax = FIXED_FIELDS["ax"]
    ay = FIXED_FIELDS["ay"]
    kap = FIXED_FIELDS["kap"]
    g1 = FIXED_FIELDS["g1"]
    g2 = FIXED_FIELDS["g2"]
    phi = FIXED_FIELDS["phi"]
    tuples = FIXED_FIELDS["tuples"]

    # gx/gy: (ny, nx); fields: (ny, nx).  Map to source plane once.
    sx_grid = GX - ax
    sy_grid = GY - ay
    bl_x = sx_grid[:-1, :-1]; bl_y = sy_grid[:-1, :-1]
    br_x = sx_grid[:-1,  1:]; br_y = sy_grid[:-1,  1:]
    tl_x = sx_grid[ 1:, :-1]; tl_y = sy_grid[ 1:, :-1]
    tr_x = sx_grid[ 1:,  1:]; tr_y = sy_grid[ 1:,  1:]

    xs_src = torch.tensor(sx_arr, device=device, dtype=dtype).view(C, 1, 1)
    ys_src = torch.tensor(sy_arr, device=device, dtype=dtype).view(C, 1, 1)

    in_A = _tri_contains(xs_src, ys_src, bl_x, bl_y, tr_x, tr_y, br_x, br_y)
    in_B = _tri_contains(xs_src, ys_src, bl_x, bl_y, tr_x, tr_y, tl_x, tl_y)
    ox = GX[:-1, :-1].unsqueeze(0).expand_as(in_A)
    oy = GY[:-1, :-1].unsqueeze(0).expand_as(in_A)
    idx_A = torch.nonzero(in_A, as_tuple=False)
    idx_B = torch.nonzero(in_B, as_tuple=False)
    if idx_A.numel() + idx_B.numel() == 0:
        return [[] for _ in range(C)]

    def _seeds(idx, off_x, off_y):
        cfg, j, i = idx[:, 0], idx[:, 1], idx[:, 2]
        return cfg, ox[cfg, j, i] + off_x * DP, oy[cfg, j, i] + off_y * DP

    cA, xA, yA = _seeds(idx_A, 0.667, 0.333)
    cB, xB, yB = _seeds(idx_B, 0.333, 0.667)
    cand_cfg = torch.cat([cA, cB])
    cand_x0 = torch.cat([xA, xB])
    cand_y0 = torch.cat([yA, yB])

    xi = cand_x0.clone()
    yi = cand_y0.clone()
    sub_xs = torch.tensor(sx_arr, device=device, dtype=dtype)[cand_cfg]
    sub_ys = torch.tensor(sy_arr, device=device, dtype=dtype)[cand_cfg]

    for _ in range(max_iter):
        ax_pt, ay_pt, kap_pt, g1_pt, g2_pt, _, _ = sum_lensmodel(
            CTX, tuples, xi, yi, need_kg=True, need_phi=False)
        pxx = kap_pt + g1_pt
        pyy = kap_pt - g1_pt
        pxy = g2_pt
        ff = sub_xs - xi + ax_pt
        gg = sub_ys - yi + ay_pt
        mm = (1.0 - pxx) * (1.0 - pyy) - pxy * pxy
        xi = xi + ((1.0 - pyy) * ff + pxy * gg) / mm
        yi = yi + ((1.0 - pxx) * gg + pxy * ff) / mm

    muinv = (1.0 - kap_pt) ** 2 - (g1_pt * g1_pt + g2_pt * g2_pt)
    mag = 1.0 / (muinv + K.DEF_IMAG_CEIL)

    dist2 = (xi - cand_x0) ** 2 + (yi - cand_y0) ** 2
    keep = dist2 <= (2.0 * DP * DP)

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
    return out


# ==========================================================================
# Batched image solve (lens-varying path — grid recomputed per candidate)
# ==========================================================================
def batched_lens_solve(sx_arr, sy_arr, lens_tuples_list, max_iter=8):
    """
    sx_arr, sy_arr: numpy (C,) source positions.
    lens_tuples_list: list of length C; each element is the lens list for that
        candidate (rhongomyniad-style).
    Returns list of length C, each = list of (x, y, mag) tuples.
    """
    C = len(lens_tuples_list)

    # Build (C, ny, nx) field stack.  Sersic/SIE kernels accept scalar params,
    # so we evaluate per-candidate and stack.  Kernel work itself is batched
    # over all grid points on GPU, so each pass is cheap.
    ax_list, ay_list = [], []
    kap_list, g1_list, g2_list = [], [], []
    for c_idx in range(C):
        ax_c, ay_c, kap_c, g1_c, g2_c, _, _ = sum_lensmodel(
            CTX, lens_tuples_list[c_idx], GX, GY,
            need_kg=True, need_phi=False)
        ax_list.append(ax_c); ay_list.append(ay_c)
        kap_list.append(kap_c); g1_list.append(g1_c); g2_list.append(g2_c)
    ax_all = torch.stack(ax_list, dim=0)  # (C, ny, nx)
    ay_all = torch.stack(ay_list, dim=0)

    sx_grid = GX.unsqueeze(0) - ax_all
    sy_grid = GY.unsqueeze(0) - ay_all
    bl_x = sx_grid[:, :-1, :-1]; bl_y = sy_grid[:, :-1, :-1]
    br_x = sx_grid[:, :-1,  1:]; br_y = sy_grid[:, :-1,  1:]
    tl_x = sx_grid[:,  1:, :-1]; tl_y = sy_grid[:,  1:, :-1]
    tr_x = sx_grid[:,  1:,  1:]; tr_y = sy_grid[:,  1:,  1:]

    xs_src = torch.tensor(sx_arr, device=device, dtype=dtype).view(C, 1, 1)
    ys_src = torch.tensor(sy_arr, device=device, dtype=dtype).view(C, 1, 1)

    in_A = _tri_contains(xs_src, ys_src, bl_x, bl_y, tr_x, tr_y, br_x, br_y)
    in_B = _tri_contains(xs_src, ys_src, bl_x, bl_y, tr_x, tr_y, tl_x, tl_y)
    ox = GX[:-1, :-1].unsqueeze(0).expand_as(in_A)
    oy = GY[:-1, :-1].unsqueeze(0).expand_as(in_A)
    idx_A = torch.nonzero(in_A, as_tuple=False)
    idx_B = torch.nonzero(in_B, as_tuple=False)
    if idx_A.numel() + idx_B.numel() == 0:
        return [[] for _ in range(C)]

    def _seeds(idx, off_x, off_y):
        cfg, j, i = idx[:, 0], idx[:, 1], idx[:, 2]
        return cfg, ox[cfg, j, i] + off_x * DP, oy[cfg, j, i] + off_y * DP

    cA, xA, yA = _seeds(idx_A, 0.667, 0.333)
    cB, xB, yB = _seeds(idx_B, 0.333, 0.667)
    cand_cfg = torch.cat([cA, cB])
    cand_x0 = torch.cat([xA, xB])
    cand_y0 = torch.cat([yA, yB])

    # For Newton we need per-seed lens evaluation.  Group seeds by candidate
    # and run one sum_lensmodel per candidate (vectorised over its seeds).
    cfg_cpu = cand_cfg.cpu().numpy()
    xi_all = cand_x0.clone()
    yi_all = cand_y0.clone()
    sub_xs_src = torch.tensor(sx_arr, device=device, dtype=dtype)[cand_cfg]
    sub_ys_src = torch.tensor(sy_arr, device=device, dtype=dtype)[cand_cfg]

    group_indices = [np.where(cfg_cpu == c)[0] for c in range(C)]
    group_idx_tensors = [torch.as_tensor(g, device=device, dtype=torch.long)
                         for g in group_indices]

    kap_t = torch.zeros_like(xi_all)
    g1_t = torch.zeros_like(xi_all)
    g2_t = torch.zeros_like(xi_all)

    for _ in range(max_iter):
        ax_t = torch.zeros_like(xi_all)
        ay_t = torch.zeros_like(xi_all)
        kap_t = torch.zeros_like(xi_all)
        g1_t = torch.zeros_like(xi_all)
        g2_t = torch.zeros_like(xi_all)
        for c in range(C):
            g = group_idx_tensors[c]
            if g.numel() == 0:
                continue
            xi_c = xi_all[g]; yi_c = yi_all[g]
            ax_c, ay_c, kap_c, g1_c, g2_c, _, _ = sum_lensmodel(
                CTX, lens_tuples_list[c], xi_c, yi_c,
                need_kg=True, need_phi=False)
            ax_t[g] = ax_c
            ay_t[g] = ay_c
            kap_t[g] = kap_c
            g1_t[g] = g1_c
            g2_t[g] = g2_c

        pxx = kap_t + g1_t
        pyy = kap_t - g1_t
        pxy = g2_t
        ff = sub_xs_src - xi_all + ax_t
        gg = sub_ys_src - yi_all + ay_t
        mm = (1.0 - pxx) * (1.0 - pyy) - pxy * pxy
        xi_all = xi_all + ((1.0 - pyy) * ff + pxy * gg) / mm
        yi_all = yi_all + ((1.0 - pxx) * gg + pxy * ff) / mm

    muinv = (1.0 - kap_t) ** 2 - (g1_t * g1_t + g2_t * g2_t)
    mag = 1.0 / (muinv + K.DEF_IMAG_CEIL)

    dist2 = (xi_all - cand_x0) ** 2 + (yi_all - cand_y0) ** 2
    keep = dist2 <= (2.0 * DP * DP)

    xi_cpu = xi_all.cpu().numpy()
    yi_cpu = yi_all.cpu().numpy()
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
    return out


# ==========================================================================
# Image match + loss
# ==========================================================================
def _match_and_score(imgs, sx, sy):
    n_img = len(imgs)
    # Drop a central image if we get exactly n_obs + 1.
    if n_img == n_obs + 1:
        drop = int(np.argmin([abs(im[2]) for im in imgs]))
        imgs = [im for k, im in enumerate(imgs) if k != drop]
        n_img = len(imgs)

    n_missing = max(0, n_obs - n_img)
    extra_penalty = n_missing * 1e5

    matched_pos = np.zeros((n_obs, 2))
    matched_mag = np.zeros(n_obs)
    delta_mas = np.full(n_obs, 1000.0)

    if n_img > 0:
        n_use = min(n_img, n_obs)
        imgs_sorted = sorted(imgs, key=lambda im: abs(im[2]), reverse=True)
        pred_pos = np.array([[im[0], im[1]] for im in imgs_sorted[:n_use]])
        pred_mag = np.array([im[2] for im in imgs_sorted[:n_use]])
        pred_pos[:, 0] += center_offset_x
        pred_pos[:, 1] += center_offset_y
        row_ind, col_ind = linear_sum_assignment(cdist(obs_positions, pred_pos))
        for ri, ci in zip(row_ind, col_ind):
            matched_pos[ri] = pred_pos[ci]
            matched_mag[ri] = pred_mag[ci]
            delta_mas[ri] = np.sqrt(
                ((pred_pos[ci, 0] - obs_positions[ri, 0]) * 1000) ** 2 +
                ((pred_pos[ci, 1] - obs_positions[ri, 1]) * 1000) ** 2)

    total = 0.0
    for i in range(n_obs):
        chi2_pos = (delta_mas[i] / obs_pos_sigma_mas[i]) ** 2
        if obs_mag_errors[i] > 0:
            chi2_mag = ((matched_mag[i] - obs_magnifications[i])
                        / obs_mag_errors[i]) ** 2
        else:
            chi2_mag = 0.0
        penalty = (0.0 if delta_mas[i] <= obs_pos_sigma_mas[i]
                   else LOSS_PENALTY_PL * delta_mas[i])
        total += LOSS_COEF_A * chi2_pos + LOSS_COEF_B * chi2_mag + penalty
    return matched_pos, matched_mag, delta_mas, total + extra_penalty, extra_penalty


# ==========================================================================
# DE objective (vectorised over population)
# ==========================================================================
def vectorised_chi2(params_arr):
    if params_arr.ndim == 1:
        params_arr = params_arr[:, None]
    C = params_arr.shape[1]

    sx_list = np.empty(C)
    sy_list = np.empty(C)
    lp_dicts = [None] * C
    for c in range(C):
        sx, sy, lp = _decode_candidate(params_arr[:, c])
        sx_list[c] = sx
        sy_list[c] = sy
        lp_dicts[c] = lp

    if not lens_modify:
        all_images = batched_source_solve(sx_list, sy_list)
    else:
        tuples_list = [_lens_tuples_from_dict(lp) for lp in lp_dicts]
        all_images = batched_lens_solve(sx_list, sy_list, tuples_list)

    loss = np.empty(C, dtype=np.float64)
    for c in range(C):
        _, _, _, loss[c], _ = _match_and_score(all_images[c], sx_list[c], sy_list[c])
    return loss


# ==========================================================================
# Baseline evaluation
# ==========================================================================
def _baseline_eval():
    """Evaluate the baseline (no DE perturbation)."""
    if NDIM == 0:
        # Nothing to optimise; still evaluate the model at (source_ref, lens_ref).
        tuples = _lens_tuples_from_dict(lens_params_ref)
        imgs = batched_lens_solve(np.array([source_x_ref]),
                                  np.array([source_y_ref]), [tuples])[0]
        return _match_and_score(imgs, source_x_ref, source_y_ref)

    # Build a (NDIM,) starting vector matching the reference state and evaluate.
    x0 = np.empty(NDIM)
    for i, (ptype, key, pi) in enumerate(PMAP):
        if ptype == "src_x":
            x0[i] = source_x_ref
        elif ptype == "src_y":
            x0[i] = source_y_ref
        else:
            x0[i] = float(lens_params_ref[key][3 + pi])
    loss_vec = vectorised_chi2(x0[:, None])
    return None, None, None, float(loss_vec[0]), None


# ==========================================================================
# Run DE
# ==========================================================================
def _solve_for_state(sx, sy, lens_dict):
    """Run the GPU forward solver for a given (sx, sy, lens_dict).
    Returns (pred_pos, pred_mag, delta_pos_mas)."""
    if not lens_modify:
        imgs = batched_source_solve(np.array([sx]), np.array([sy]))[0]
    else:
        imgs = batched_lens_solve(np.array([sx]), np.array([sy]),
                                  [_lens_tuples_from_dict(lens_dict)])[0]
    return gp.match_images(imgs, obs_positions,
                           center_offset_x, center_offset_y, n_obs=n_obs)


def main():
    print("\n" + "=" * 70)
    print("Step 1: baseline")
    print("=" * 70)
    base_pos, base_mag, base_delta = _solve_for_state(
        source_x_ref, source_y_ref, lens_params_ref)
    if base_delta is None:
        base_delta = np.full(n_obs, 1e3)
        base_mag = np.zeros(n_obs)
        base_pos = np.zeros((n_obs, 2))
    base_pos_chi2 = float(np.sum((base_delta / obs_pos_sigma_mas) ** 2))
    base_mag_chi2 = float(np.sum(((base_mag - obs_magnifications) / obs_mag_errors) ** 2))
    base_total_chi2 = base_pos_chi2 + base_mag_chi2
    print(f"  pos RMS    : {np.sqrt(np.mean(base_delta**2)):.3f} mas")
    print(f"  chi2_pos   : {base_pos_chi2:.2f}")
    print(f"  chi2_mag   : {base_mag_chi2:.2f}")
    print(f"  chi2_total : {base_total_chi2:.2f}")

    if NDIM == 0:
        print("\nNothing to optimise (source_modify=False and lens_modify=False).")
        # Still emit best_params + triptych so the output set is consistent.
        _emit_outputs(
            base_pos, base_mag, base_delta,
            base_pos_chi2, base_mag_chi2,
            base_pos, base_mag, base_delta,
            base_pos_chi2, base_mag_chi2,
            source_x_ref, source_y_ref, lens_params_ref,
            best_x=np.array([]), best_loss=base_total_chi2,
            iteration=0, improvement=0.0, constraint_ok=True,
            param_names=[], corner_labels=[], bounds=[])
        return

    print("\n" + "=" * 70)
    print(f"Step 2: differential evolution (ndim={NDIM}, popsize_mult={DE_POPSIZE},"
          f" pop={DE_POPSIZE * NDIM}, seed={DE_SEED})")
    print("=" * 70)
    print(f"  device: {device}   finder: {rh.get_finder()}")

    # Warmup
    rng = np.random.default_rng(DE_SEED)
    popsize_total = DE_POPSIZE * NDIM
    warm = np.empty((NDIM, popsize_total))
    for i, (lo, hi) in enumerate(BOUNDS):
        warm[i] = rng.uniform(lo, hi, popsize_total)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = vectorised_chi2(warm)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  warmup: {popsize_total} members in {dt*1000:.1f} ms "
          f"({dt/popsize_total*1000:.2f} ms/member)")

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
    solver = DifferentialEvolutionSolver(vectorised_chi2, BOUNDS, **de_kwargs)

    param_labels = []
    for ptype, key, pi in PMAP:
        if ptype == "src_x":
            param_labels.append("src_x")
        elif ptype == "src_y":
            param_labels.append("src_y")
        else:
            param_labels.append(f"{key}_p{pi+1}")

    if Draw_Graph:
        gp.plot_iteration_general(
            solver.population.copy(), 0, output_dir,
            BOUNDS, param_labels, draw_interval=draw_interval)

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
                        gp.plot_iteration_general(
                            solver.population.copy(), iteration, output_dir,
                            BOUNDS, param_labels, draw_interval=draw_interval)
                    break
            else:
                converged_count = 0

        if Draw_Graph:
            gp.plot_iteration_general(
                solver.population.copy(), iteration, output_dir,
                BOUNDS, param_labels, draw_interval=draw_interval)

        prev_best = cur_best
        iteration += 1
        if iteration > DE_MAXITER:
            print(f"  reached DE_MAXITER={DE_MAXITER}")
            break
    de_dt = time.perf_counter() - t_de

    best_x = solver.x
    best_loss = float(np.min(solver.population_energies))
    print(f"\n  DE finished in {de_dt:.1f}s  iters={iteration}  loss={best_loss:.4f}")

    best_sx, best_sy, best_lp = _decode_candidate(best_x)

    print("\n" + "=" * 70)
    print("Step 3: best-fit analysis")
    print("=" * 70)
    opt_pos, opt_mag, opt_delta = _solve_for_state(best_sx, best_sy, best_lp)
    if opt_pos is None:
        print("  warn: best-fit solve did not yield 4 images; "
              "falling back to baseline state")
        opt_pos, opt_mag, opt_delta = base_pos, base_mag, base_delta
    opt_pos_chi2 = float(np.sum((opt_delta / obs_pos_sigma_mas) ** 2))
    opt_mag_chi2 = float(np.sum(((opt_mag - obs_magnifications) / obs_mag_errors) ** 2))
    opt_total_chi2 = opt_pos_chi2 + opt_mag_chi2
    improvement = ((base_total_chi2 - opt_total_chi2) / base_total_chi2 * 100
                   if base_total_chi2 > 0 else 0.0)
    print(f"  pos RMS    : {np.sqrt(np.mean(opt_delta**2)):.3f} mas")
    print(f"  chi2_pos   : {opt_pos_chi2:.2f}  (baseline {base_pos_chi2:.2f})")
    print(f"  chi2_mag   : {opt_mag_chi2:.2f}  (baseline {base_mag_chi2:.2f})")
    print(f"  chi2_total : {opt_total_chi2:.2f}  (baseline {base_total_chi2:.2f})")
    print(f"  improvement: {improvement:.1f}%  (on chi2_total)")
    if source_modify:
        print(f"  source_xy  : ({best_sx:+.6e}, {best_sy:+.6e}) arcsec")
    if lens_modify:
        opt_keys = {k for _, k, _ in PMAP if k is not None}
        for key, pv in best_lp.items():
            if key in opt_keys:
                print(f"  {key}        : "
                      + ", ".join(f"{v:.4g}" for v in pv[3:]))
    constraint_ok = bool(np.all(opt_delta <= obs_pos_sigma_mas))
    print(f"  constraint satisfied: {constraint_ok}")

    _emit_outputs(
        base_pos, base_mag, base_delta,
        base_pos_chi2, base_mag_chi2,
        opt_pos, opt_mag, opt_delta,
        opt_pos_chi2, opt_mag_chi2,
        best_sx, best_sy, best_lp,
        best_x=best_x, best_loss=best_loss,
        iteration=iteration, improvement=improvement,
        constraint_ok=constraint_ok,
        param_names=param_labels,
        corner_labels=param_labels, bounds=BOUNDS)


def _emit_outputs(
    base_pos, base_mag, base_delta,
    base_pos_chi2, base_mag_chi2,
    opt_pos, opt_mag, opt_delta,
    opt_pos_chi2, opt_mag_chi2,
    best_sx, best_sy, best_lp,
    best_x, best_loss, iteration, improvement, constraint_ok,
    param_names, corner_labels, bounds,
):
    base_total_chi2 = base_pos_chi2 + base_mag_chi2
    opt_total_chi2 = opt_pos_chi2 + opt_mag_chi2
    # ---- best params file -------------------------------------------------
    params_path = os.path.join(output_dir, f"{OUTPUT_PREFIX}_best_params.txt")
    with open(params_path, "w") as f:
        f.write(f"# Version None GPU (Rhongomyniad batched)\n")
        f.write(f"# DE seed: {DE_SEED}\n")
        f.write(f"# source_modify={source_modify}  lens_modify={lens_modify}  "
                f"fine_tuning={fine_tuning}\n")
        f.write(f"# finder = {rh.get_finder()}  device = {rh.get_device()}\n")
        f.write(f"# DE: iters={iteration} loss={best_loss:.6f}\n")
        f.write(f"# improvement = {improvement:.2f}% (on chi2_total)\n\n")
        f.write(f"source_x_optimized = {best_sx:.10e}\n")
        f.write(f"source_y_optimized = {best_sy:.10e}\n\n")
        f.write("# Optimized lens parameters\n")
        for key, pv in best_lp.items():
            ltype, z = pv[1], pv[2]
            ps = pv[3:]
            f.write(f"# {key}\n")
            f.write(f"  lens {ltype} {z:.4f}  "
                    + "  ".join(f"{v:.10e}" for v in ps) + "\n")
        f.write("\n# Performance\n")
        f.write(f"chi2_pos_base = {base_pos_chi2:.4f}\n")
        f.write(f"chi2_mag_base = {base_mag_chi2:.4f}\n")
        f.write(f"chi2_total_base = {base_total_chi2:.4f}\n")
        f.write(f"chi2_pos_best = {opt_pos_chi2:.4f}\n")
        f.write(f"chi2_mag_best = {opt_mag_chi2:.4f}\n")
        f.write(f"chi2_total_best = {opt_total_chi2:.4f}\n\n")
        f.write("# Per-image residuals\n")
        for i in range(n_obs):
            f.write(f"  image_{i+1}: delta_pos={opt_delta[i]:.4f} mas  "
                    f"mu_pred={opt_mag[i]:.3f}  mu_obs={obs_magnifications[i]:.3f}\n")
        f.write(f"\nconstraint_satisfied = {constraint_ok}\n")
    print(f"  saved best params: {params_path}")

    # ---- MCMC -------------------------------------------------------------
    if MCMC_ENABLED and len(bounds) > 0:
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
            title_prefix="None")

    # ---- Critical curves + triptych --------------------------------------
    print("\n" + "=" * 70)
    print("Step 4: result plots")
    print("=" * 70)
    base_lens_lines = list(best_lp.values())
    extra_lens_lines = []  # no sub-halos in v_none

    crit_segments, caus_segments = gp.compute_critical_curves(
        output_dir, OUTPUT_PREFIX,
        cosmo=(omega, lambda_cosmo, weos, hubble),
        grid=(xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev),
        base_lens_lines=base_lens_lines, extra_lens_lines=extra_lens_lines,
        source_z=source_z, source_x=float(best_sx), source_y=float(best_sy))

    triptych_path = os.path.join(output_dir, f"result_{OUTPUT_PREFIX}.png")
    gp.write_result_triptych(
        triptych_path,
        suptitle="iPTF16geu: Source/Lens Optimization (No Sub-halos, GPU)",
        obs_positions=obs_positions, pred_positions=opt_pos,
        delta_pos_mas=opt_delta, sigma_pos_mas=obs_pos_sigma_mas,
        mu_obs=obs_magnifications, mu_obs_err=obs_mag_errors,
        mu_pred=opt_mag,
        crit_segments=crit_segments, caus_segments=caus_segments,
        subhalo_positions=None, show_2sigma=SHOW_2SIGMA)

    if COMPARE_GRAPH:
        compare_path = os.path.join(output_dir,
                                    f"result_{OUTPUT_PREFIX}_compare.png")
        gp.write_compare_triptych(
            compare_path,
            suptitle="iPTF16geu: Baseline vs Optimized (No Sub-halos, GPU)",
            obs_positions=obs_positions, pred_positions=opt_pos,
            delta_pos_baseline=base_delta, delta_pos_optimized=opt_delta,
            sigma_pos_mas=obs_pos_sigma_mas,
            mu_obs=obs_magnifications, mu_obs_err=obs_mag_errors,
            mu_pred_baseline=base_mag, mu_pred_optimized=opt_mag,
            crit_segments=crit_segments, caus_segments=caus_segments,
            subhalo_positions=None, show_2sigma=SHOW_2SIGMA)

    # ---- Glafic CLI verification -----------------------------------------
    gp.run_glafic_and_compare(
        output_dir, OUTPUT_PREFIX,
        cosmo=(omega, lambda_cosmo, weos, hubble),
        grid=(xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev),
        base_lens_lines=base_lens_lines, extra_lens_lines=extra_lens_lines,
        source_z=source_z, source_x=float(best_sx), source_y=float(best_sy),
        obs_positions=obs_positions,
        center_offset_x=center_offset_x, center_offset_y=center_offset_y,
        best_pos_py=opt_pos, best_mag_py=opt_mag,
        header_comment=f"{OUTPUT_PREFIX} verification")

    print("\n" + "=" * 70)
    print(f"Done. results in {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
