#!/usr/bin/env python3
"""
Version None GPU: No Subhalo — Source/Lens Parameter Optimization (Rhongomyniad, batched).

Batched GPU counterpart of v_none_1_0.  No subhalos; source and/or main-
lens parameters are optimized via differential evolution.  Each DE
generation evaluates the full population in parallel: per-candidate grid
fields are computed via Rhongomyniad's sum_lensmodel (scalar lens API),
stacked into (C, ny, nx) tensors, and a custom batched Newton image
solver runs all candidates concurrently on GPU.  At the end the best
candidate is cross-checked against the glafic CLI.

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
# Locate Rhongomyniad
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE.parent.parent / "Rhongomyniad",
              _HERE.parent.parent.parent / "Rhongomyniad"):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, linear_sum_assignment

import rhongomyniad as rh
from rhongomyniad import constants as K
from rhongomyniad.lens_models import LensContext
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
# Config (override-friendly top-level assignments)
# ==========================================================================
BASELINE_LENS_DIR = ""

source_modify = False
lens_modify = True
fine_tuning = False

# fine_tuning=False
modify_percentage = 0.2
source_x_delta = 0.1
source_y_delta = 0.1

# fine_tuning=True
source_x_bounds = [-0.3, 0.3]
source_y_bounds = [-0.3, 0.3]
lens_optimize_bounds: dict = {}

LOSS_COEF_A = 1.0
LOSS_COEF_B = 1.0
LOSS_PENALTY_PL = 1000.0

DE_MAXITER = 200
DE_POPSIZE = 15
DE_ATOL = 1e-4
DE_TOL = 1e-6
DE_SEED = random.randint(1, 1000000)
DE_POLISH = True

OUTPUT_PREFIX = "v_none_gpu"

# Observations (cross-configuration defaults, match legacy)
obs_positions_mas_list = [[-330.461, 0], [330.461, 0],
                          [0, -262.771], [0, 262.771]]
obs_magnifications_list = [2.92052, 2.92052, -1.52908, -1.52908]
obs_mag_errors_list = [0.2, 0.2, 0.2, 0.2]
obs_pos_sigma_mas_list = [0.5, 0.5, 0.691, 0.691]
center_offset_x = 0.0
center_offset_y = 0.0
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
    print("[baseline] using built-in default (2 Sersic + SIE)")

lens_params_ref = {k: list(v) for k, v in lens_params.items()}
source_x_ref = source_x
source_y_ref = source_y

_x_sign = -1 if obs_x_flip else 1
obs_positions_mas = np.array(obs_positions_mas_list, dtype=float)
obs_positions = np.zeros_like(obs_positions_mas)
obs_positions[:, 0] = _x_sign * obs_positions_mas[:, 0] / 1000.0
obs_positions[:, 1] = obs_positions_mas[:, 1] / 1000.0
center_offset_x = _x_sign * center_offset_x
obs_magnifications = np.array(obs_magnifications_list, dtype=float)
obs_mag_errors = np.array(obs_mag_errors_list, dtype=float)
obs_pos_sigma_mas = np.array(obs_pos_sigma_mas_list, dtype=float)
n_obs = len(obs_positions)

timestamp = datetime.now().strftime("%y%m%d_%H%M")
output_dir = timestamp
os.makedirs(output_dir, exist_ok=True)
print(f"output dir: {output_dir}")

device = rh.get_device()
dtype = torch.float64
print(f"device: {device}   finder: {rh.get_finder()}")


# ==========================================================================
# Grid + LensContext (built ONCE; per-candidate fields rebuilt as needed)
# ==========================================================================
def _build_grid_ctx():
    cosmo = Cosmology(omega=omega, lam=lambda_cosmo, weos=weos, hubble=hubble)
    ctx = LensContext.build(cosmo, zl=lens_z, zs=source_z)

    dp = pix_poi / (2 ** (maxlev - 1))
    nx = int(math.ceil((xmax - xmin) / dp)) + 1
    ny = int(math.ceil((ymax - ymin) / dp)) + 1
    xs_ax = torch.linspace(xmin, xmin + (nx - 1) * dp, nx,
                           device=device, dtype=dtype)
    ys_ax = torch.linspace(ymin, ymin + (ny - 1) * dp, ny,
                           device=device, dtype=dtype)
    gx, gy = torch.meshgrid(xs_ax, ys_ax, indexing="xy")
    return ctx, gx, gy, dp, nx, ny


CTX, GX, GY, DP, NX, NY = _build_grid_ctx()


def _lenses_from_dict(lp_dict):
    """(lens_type, (z, p1..p7)) tuples for sum_lensmodel."""
    out = []
    for _key, pv in lp_dict.items():
        _, model, z, *p7 = pv
        out.append((model, (z, *p7)))
    return out


def _fields_for_candidate(lp_dict):
    """Evaluate full lens fields on the grid for one candidate.  Returns
    (ax, ay, kap, g1, g2) each (ny, nx)."""
    lenses = _lenses_from_dict(lp_dict)
    ax, ay, kap, g1, g2, _, _ = sum_lensmodel(
        CTX, lenses, GX, GY, need_kg=True, need_phi=False)
    return ax.contiguous(), ay.contiguous(), kap.contiguous(), \
           g1.contiguous(), g2.contiguous()


# ==========================================================================
# Triangle test + batched Newton solver
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


def _batched_point_solve(lp_per_cand, sx_per_cand, sy_per_cand, max_iter=8):
    """
    lp_per_cand: list of lens_params dicts, length C.
    sx_per_cand, sy_per_cand: numpy (C,) source positions.
    Returns list-of-lists: out[c] = [(x, y, mag), ...].
    """
    C = len(lp_per_cand)

    ax_list, ay_list, kap_list, g1_list, g2_list = [], [], [], [], []
    for lp in lp_per_cand:
        ax_c, ay_c, kap_c, g1_c, g2_c = _fields_for_candidate(lp)
        ax_list.append(ax_c)
        ay_list.append(ay_c)
        kap_list.append(kap_c)
        g1_list.append(g1_c)
        g2_list.append(g2_c)
    ax = torch.stack(ax_list, dim=0)        # (C, ny, nx)
    ay = torch.stack(ay_list, dim=0)
    kap_g = torch.stack(kap_list, dim=0)
    g1_g = torch.stack(g1_list, dim=0)
    g2_g = torch.stack(g2_list, dim=0)

    sx_t = torch.tensor(sx_per_cand, device=device, dtype=dtype).view(C, 1, 1)
    sy_t = torch.tensor(sy_per_cand, device=device, dtype=dtype).view(C, 1, 1)

    sx_grid = GX.unsqueeze(0) - ax
    sy_grid = GY.unsqueeze(0) - ay
    bl_x = sx_grid[:, :-1, :-1]; bl_y = sy_grid[:, :-1, :-1]
    br_x = sx_grid[:, :-1,  1:]; br_y = sy_grid[:, :-1,  1:]
    tl_x = sx_grid[:,  1:, :-1]; tl_y = sy_grid[:,  1:, :-1]
    tr_x = sx_grid[:,  1:,  1:]; tr_y = sy_grid[:,  1:,  1:]
    in_A = _tri_contains(sx_t, sy_t, bl_x, bl_y, tr_x, tr_y, br_x, br_y)
    in_B = _tri_contains(sx_t, sy_t, bl_x, bl_y, tr_x, tr_y, tl_x, tl_y)
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
    xs_per_pt = sx_t.view(C)[cand_cfg]
    ys_per_pt = sy_t.view(C)[cand_cfg]

    # Per-candidate lens lists for Newton iterations at the candidate points.
    cand_cfg_cpu = cand_cfg.cpu().numpy()

    for _ in range(max_iter):
        ax_t = torch.zeros_like(xi)
        ay_t = torch.zeros_like(xi)
        kap_t = torch.zeros_like(xi)
        g1_t = torch.zeros_like(xi)
        g2_t = torch.zeros_like(xi)
        # For each candidate, evaluate main lenses at its owned points.
        for c in range(C):
            mask = torch.from_numpy(cand_cfg_cpu == c).to(device)
            if not mask.any():
                continue
            xm = xi[mask]; ym = yi[mask]
            lenses = _lenses_from_dict(lp_per_cand[c])
            a_x, a_y, k_, g_1, g_2, _, _ = sum_lensmodel(
                CTX, lenses, xm, ym, need_kg=True, need_phi=False)
            ax_t[mask] = a_x
            ay_t[mask] = a_y
            kap_t[mask] = k_
            g1_t[mask] = g_1
            g2_t[mask] = g_2

        pxx = kap_t + g1_t
        pyy = kap_t - g1_t
        pxy = g2_t
        ff = xs_per_pt - xi + ax_t
        gg = ys_per_pt - yi + ay_t
        mm = (1.0 - pxx) * (1.0 - pyy) - pxy * pxy
        xi = xi + ((1.0 - pyy) * ff + pxy * gg) / mm
        yi = yi + ((1.0 - pxx) * gg + pxy * ff) / mm

    muinv = (1.0 - kap_t) ** 2 - (g1_t * g1_t + g2_t * g2_t)
    mag = 1.0 / (muinv + K.DEF_IMAG_CEIL)

    dist2 = (xi - cand_x0) ** 2 + (yi - cand_y0) ** 2
    keep = dist2 <= (2.0 * DP * DP)

    xi_cpu = xi.cpu().numpy()
    yi_cpu = yi.cpu().numpy()
    mag_cpu = mag.cpu().numpy()
    keep_cpu = keep.cpu().numpy()

    out = [[] for _ in range(C)]
    for k in range(len(cand_cfg_cpu)):
        if not keep_cpu[k]:
            continue
        c = int(cand_cfg_cpu[k])
        x, y, m = float(xi_cpu[k]), float(yi_cpu[k]), float(mag_cpu[k])
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
        if len(imgs) == n_obs + 1:
            central = min(range(len(imgs)), key=lambda k: abs(imgs[k][2]))
            imgs = [im for k, im in enumerate(imgs) if k != central]
        result.append(imgs)
    return result


# ==========================================================================
# Optimization variable mapping
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


bounds, pmap = build_optimization()
ndim = len(bounds)
print(f"\noptimize ndim = {ndim}  "
      f"(source_modify={source_modify}, lens_modify={lens_modify}, "
      f"fine_tuning={fine_tuning})")


def _decode(x):
    sx, sy = source_x_ref, source_y_ref
    lp = {k: list(v) for k, v in lens_params_ref.items()}
    for i, (ptype, key, pi) in enumerate(pmap):
        if ptype == "src_x":
            sx = float(x[i])
        elif ptype == "src_y":
            sy = float(x[i])
        else:
            lp[key][3 + pi] = float(x[i])
    return sx, sy, lp


# ==========================================================================
# Chi2
# ==========================================================================
def _match_and_score(imgs, sx, sy):
    if len(imgs) != n_obs:
        n_missing = max(0, n_obs - len(imgs))
        return 1e15 + 1.0e5 * n_missing
    pred_pos = np.array([[im[0] + center_offset_x,
                          im[1] + center_offset_y] for im in imgs])
    pred_mag = np.array([im[2] for im in imgs])
    distances = cdist(obs_positions, pred_pos)
    row_ind, col_ind = linear_sum_assignment(distances)
    pp = pred_pos[col_ind[np.argsort(row_ind)]]
    pm = pred_mag[col_ind[np.argsort(row_ind)]]
    delta_mas = np.sqrt(np.sum(((pp - obs_positions) * 1000) ** 2, axis=1))
    total = 0.0
    for i in range(n_obs):
        chi2_pos = (delta_mas[i] / obs_pos_sigma_mas[i]) ** 2
        if obs_mag_errors[i] > 0:
            chi2_mag = ((pm[i] - obs_magnifications[i]) / obs_mag_errors[i]) ** 2
        else:
            chi2_mag = 0.0
        penalty = (0.0 if delta_mas[i] <= obs_pos_sigma_mas[i]
                   else LOSS_PENALTY_PL * delta_mas[i])
        total += LOSS_COEF_A * chi2_pos + LOSS_COEF_B * chi2_mag + penalty
    return total


def vectorised_chi2(params_arr):
    if params_arr.ndim == 1:
        params_arr = params_arr[:, None]
    popsize = params_arr.shape[1]
    lp_list, sx_list, sy_list = [], [], []
    for c in range(popsize):
        sx, sy, lp = _decode(params_arr[:, c])
        lp_list.append(lp); sx_list.append(sx); sy_list.append(sy)
    sx_arr = np.asarray(sx_list)
    sy_arr = np.asarray(sy_list)
    try:
        all_imgs = _batched_point_solve(lp_list, sx_arr, sy_arr)
    except Exception:
        return np.full(popsize, 1e15)
    loss = np.empty(popsize, dtype=np.float64)
    for c in range(popsize):
        loss[c] = _match_and_score(all_imgs[c], sx_arr[c], sy_arr[c])
    return loss


# ==========================================================================
# Run DE
# ==========================================================================
def main():
    # Baseline.
    t0 = time.perf_counter()
    base_imgs = _batched_point_solve(
        [{k: list(v) for k, v in lens_params_ref.items()}],
        np.array([source_x_ref]), np.array([source_y_ref]))[0]
    base_loss = _match_and_score(base_imgs, source_x_ref, source_y_ref)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"baseline loss: {base_loss:.4f}  "
          f"eval: {(time.perf_counter()-t0)*1000:.1f} ms")

    if ndim == 0:
        print("\nNo parameters to optimize (source_modify=lens_modify=False).")
        return

    print(f"\nDE ndim={ndim}  popsize_mult={DE_POPSIZE}  "
          f"pop={DE_POPSIZE * ndim}  seed={DE_SEED}")

    # Warm-up timing.
    rng = np.random.default_rng(0)
    test_pop = DE_POPSIZE * ndim
    test = np.empty((ndim, test_pop))
    for i, (lo, hi) in enumerate(bounds):
        test[i] = rng.uniform(lo, hi, test_pop)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = vectorised_chi2(test)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"warmup: {test_pop} members in {dt*1000:.1f} ms "
          f"({dt/test_pop*1000:.2f} ms/member)")

    t0 = time.perf_counter()
    res = differential_evolution(
        vectorised_chi2, bounds,
        maxiter=DE_MAXITER, popsize=DE_POPSIZE,
        atol=DE_ATOL, tol=DE_TOL,
        seed=DE_SEED, polish=DE_POLISH, disp=True,
        vectorized=True,
    )
    dt = time.perf_counter() - t0
    print(f"\nDE finished in {dt:.1f}s  nit={res.nit}  nfev={res.nfev}  "
          f"loss={res.fun:.4f}")

    best_sx, best_sy, best_lp = _decode(res.x)
    print(f"\nBest source: ({best_sx:+.6e}, {best_sy:+.6e})")
    for key, pv in best_lp.items():
        print(f"  lens[{key}]: p1..p7 = "
              + ", ".join(f"{v:.4e}" for v in pv[3:]))

    out_path = os.path.join(output_dir, f"{OUTPUT_PREFIX}_best_params.txt")
    with open(out_path, "w") as f:
        f.write(f"# Version None GPU (Rhongomyniad batched)\n")
        f.write(f"# DE seed: {DE_SEED}\n")
        f.write(f"# finder = {rh.get_finder()}  device = {rh.get_device()}\n")
        f.write(f"# DE: nit={res.nit} nfev={res.nfev} loss={res.fun:.6f}\n\n")
        f.write(f"source_x = {best_sx:.10e}\n")
        f.write(f"source_y = {best_sy:.10e}\n\n")
        for key, pv in best_lp.items():
            _, ltype, z, *p7 = pv
            f.write(f"# lens {key} ({ltype}, z={z})\n")
            f.write(f"{key}_type = {ltype}\n")
            f.write(f"{key}_z = {z}\n")
            for i, v in enumerate(p7):
                f.write(f"{key}_p{i+1} = {v:.10e}\n")
            f.write("\n")
    print(f"  saved: {out_path}")

    _verify_with_glafic(best_sx, best_sy, best_lp)


# ==========================================================================
# Verification: Python/GPU solver vs glafic CLI
# ==========================================================================
def _find_glafic_bin():
    bin_path = shutil.which("glafic")
    if bin_path:
        return bin_path
    try:
        import glafic as _gl
        mod_dir = os.path.dirname(os.path.abspath(_gl.__file__))
        for rel in ("../glafic", "../../glafic", "./glafic", "../bin/glafic"):
            p = os.path.abspath(os.path.join(mod_dir, rel))
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    except Exception:
        pass
    return None


def _predict_best_images(best_sx, best_sy, best_lp):
    imgs = _batched_point_solve([best_lp],
                                np.array([best_sx]), np.array([best_sy]))[0]
    if len(imgs) != n_obs:
        return None, None
    pred_pos = np.array([[im[0] + center_offset_x,
                          im[1] + center_offset_y] for im in imgs])
    pred_mag = np.array([abs(im[2]) for im in imgs])
    d = cdist(obs_positions, pred_pos)
    ri, ci = linear_sum_assignment(d)
    order = ci[np.argsort(ri)]
    return pred_pos[order], pred_mag[order]


def _verify_with_glafic(best_sx, best_sy, best_lp):
    print("\n" + "=" * 70)
    print("Verification: Python/GPU solver vs glafic CLI")
    print("=" * 70)

    bin_path = _find_glafic_bin()
    if bin_path is None:
        print("  warn: glafic binary not found; skipping verification")
        return
    print(f"  glafic path: {bin_path}")

    best_pos_py, best_mag_py = _predict_best_images(best_sx, best_sy, best_lp)
    if best_pos_py is None:
        print("  warn: Python solver did not find all images; skipping comparison")
        return

    verify_input = os.path.join(output_dir, f"{OUTPUT_PREFIX}_verify_input.dat")
    verify_prefix = f"{OUTPUT_PREFIX}_verify"
    with open(verify_input, "w") as f:
        f.write(f"# {OUTPUT_PREFIX} verification\n")
        f.write(f"# generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"omega    {omega}\n")
        f.write(f"lambda   {lambda_cosmo}\n")
        f.write(f"weos     {weos}\n")
        f.write(f"hubble   {hubble}\n")
        f.write(f"prefix   {verify_prefix}\n")
        f.write(f"xmin     {xmin}\n")
        f.write(f"ymin     {ymin}\n")
        f.write(f"xmax     {xmax}\n")
        f.write(f"ymax     {ymax}\n")
        f.write(f"pix_ext  {pix_ext}\n")
        f.write(f"pix_poi  {pix_poi}\n")
        f.write(f"maxlev   {maxlev}\n\n")
        f.write(f"startup  {len(best_lp)} 0 1\n")
        for key, pv in best_lp.items():
            _, ltype, z, *p7 = pv
            f.write(f"lens   {ltype}  {z}  "
                    + "  ".join(f"{v:.10e}" for v in p7) + "\n")
        f.write(f"point  {source_z}  {float(best_sx):.10e}  {float(best_sy):.10e}\n")
        f.write("end_startup\n\nstart_command\nfindimg\nquit\n")
    print(f"  input: {verify_input}")

    try:
        proc = subprocess.run(
            [bin_path, os.path.basename(verify_input)],
            cwd=output_dir, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            print("  glafic run OK")
        else:
            print(f"  warn: glafic returned {proc.returncode}")
    except subprocess.TimeoutExpired:
        print("  warn: glafic timeout (>60s)")
        return
    except Exception as e:
        print(f"  warn: {e}")
        return

    verify_pt = os.path.join(output_dir, f"{verify_prefix}_point.dat")
    if not os.path.exists(verify_pt):
        print(f"  warn: output file missing: {verify_pt}")
        return
    try:
        data = np.loadtxt(verify_pt)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        n_imgs = int(data[0, 0])
        print(f"  glafic found {n_imgs} images")
        if n_imgs not in (n_obs, n_obs + 1):
            print(f"  warn: expected {n_obs} or {n_obs+1} images")
            return
        img_data = data[1:n_imgs + 1, :]
        if n_imgs == n_obs + 1:
            drop = int(np.argmin(np.abs(img_data[:, 2])))
            img_data = np.delete(img_data, drop, axis=0)
        gl_pos = img_data[:, 0:2].copy()
        gl_pos[:, 0] += center_offset_x
        gl_pos[:, 1] += center_offset_y
        gl_mag = np.abs(img_data[:, 2])
        d = cdist(obs_positions, gl_pos)
        ri, ci = linear_sum_assignment(d)
        order = ci[np.argsort(ri)]
        gl_pos_m = gl_pos[order]
        gl_mag_m = gl_mag[order]

        max_pos_diff = 0.0
        max_mag_pct = 0.0
        print(f"\n  {'Img':<5} {'Py x[mas]':>12} {'GL x[mas]':>12} {'|Δx|':>8}"
              f"  {'Py y[mas]':>12} {'GL y[mas]':>12} {'|Δy|':>8}")
        print("  " + "-" * 80)
        for k in range(n_obs):
            px = best_pos_py[k, 0] * 1000; py = best_pos_py[k, 1] * 1000
            gx = gl_pos_m[k, 0] * 1000;    gy = gl_pos_m[k, 1] * 1000
            dxv = abs(px - gx); dyv = abs(py - gy)
            max_pos_diff = max(max_pos_diff, dxv, dyv)
            print(f"  {k+1:<5} {px:>12.3f} {gx:>12.3f} {dxv:>8.3f}  "
                  f"{py:>12.3f} {gy:>12.3f} {dyv:>8.3f}")
        print(f"\n  {'Img':<5} {'Py |μ|':>12} {'GL |μ|':>12} {'Δ [%]':>10}")
        print("  " + "-" * 50)
        for k in range(n_obs):
            pm = best_mag_py[k]; gm = gl_mag_m[k]
            dmp = abs(pm - gm) / pm * 100 if pm else 0
            max_mag_pct = max(max_mag_pct, dmp)
            print(f"  {k+1:<5} {pm:>12.3f} {gm:>12.3f} {dmp:>9.3f}%")
        print(f"\n  max position diff: {max_pos_diff:.6f} mas")
        print(f"  max magnif. diff:  {max_mag_pct:.6f}%")
        if max_pos_diff < 0.01 and max_mag_pct < 0.1:
            print("  [PASS] consistency verified")
        elif max_pos_diff < 1.0 and max_mag_pct < 1.0:
            print("  [OK]   small differences")
        else:
            print("  [WARN] large discrepancy — check params")
    except Exception as e:
        print(f"  error reading verify output: {e}")


if __name__ == "__main__":
    main()
