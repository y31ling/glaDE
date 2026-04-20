"""
Real measurement: batched point_solve for N lens-config candidates (SIE+shear),
all evaluated against one shared image-plane grid in a single GPU pipeline.

Pipeline:
    1. Stack C=64 configs into parameter tensors.
    2. Evaluate (ax, ay) at every grid corner for every config in one kernel
       -> shape (C, Ny, Nx).
    3. Map corners to source plane -> (sx, sy) of shape (C, Ny, Nx).
    4. Run the 2-triangle containment test for every cell in every config
       in one kernel -> boolean masks (C, Ny-1, Nx-1) each.
    5. Extract candidate positions (small, use torch.nonzero once).
    6. Newton-refine: candidates carry their config index; lens params are
       gathered per candidate; do K Newton iterations as one batched kernel.
    7. Return (x, y, mag, td) lists grouped by config.

This is the minimum amount of code to answer the user's honest question:
"if we batch 64 configs, how fast is it actually?".

Compare against glafic's sequential 64 point_solve calls (only runnable in WSL).
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from rhongomyniad.cosmology import Cosmology
from rhongomyniad.lens_models import LensContext, fac_pert


# ---------------------------------------------------------------------------
# Scenario (same as bench_batch64.py)
# ---------------------------------------------------------------------------
GRID = dict(xmin=-30.0, ymin=-30.0, xmax=30.0, ymax=30.0,
            pix_poi=1.0, maxlev=5)
COSMO = dict(omega=0.3, lam=0.7, weos=-1.0, hubble=0.7)
ZL = 0.5
ZS = 2.0
SOURCE_XS = -0.15
SOURCE_YS = 0.05


def make_configs(C: int):
    # vary sigma + e slightly per config
    out = []
    for i in range(C):
        out.append(dict(
            sigma=280.0 + 0.5 * i,
            e=0.28 + 0.002 * (i % 20),
            pa=0.0,
            x0=0.0,
            y0=0.0,
            g_ext=0.05,
            tg=60.0,
        ))
    return out


# ---------------------------------------------------------------------------
# Batched SIE + shear deflection & Hessian
#
# All inputs are either scalars or tensors of shape (C, 1, 1) / (C,) etc.
# Outputs have shape (C, H, W) where H, W are grid size.
# ---------------------------------------------------------------------------
def _facq_sie(q):
    return 1.0 / torch.sqrt(q)


def _sie_deflection(
    sigma, q, pa, x0, y0,          # per-config (C,) tensors
    tx, ty,                         # (H, W) grid tensors
    ctx,                            # LensContext (scalars shared across configs)
    smallcore=1.0e-10,
):
    """SIE deflection only, vectorised over (C, H, W)."""
    C = sigma.shape[0]
    device, dtype = sigma.device, sigma.dtype
    # reshape config tensors for broadcasting
    s_ = sigma.view(C, 1, 1)
    q_ = q.view(C, 1, 1)
    x0_ = x0.view(C, 1, 1)
    y0_ = y0.view(C, 1, 1)
    pa_ = pa.view(C, 1, 1)

    facq = 1.0 / torch.sqrt(q_)
    bb = facq * (4.0 * math.pi * (s_ / 2.99792458e5) ** 2
                 * ctx.dis_ls / ctx.dis_os) / 4.84813681e-6

    # rotation trig: SIE uses -(pa-90) * pi/180
    ang = -(pa_ - 90.0) * math.pi / 180.0
    si = torch.sin(ang)
    co = torch.cos(ang)

    # Broadcast tx, ty from (H, W) to (1, H, W)
    tx_b = tx.unsqueeze(0); ty_b = ty.unsqueeze(0)
    dx = tx_b - x0_; dy = ty_b - y0_
    ddx = co * dx - si * dy
    ddy = si * dx + co * dy

    ss = smallcore * facq
    sq = torch.sqrt(1.0 - q_ * q_)
    psi = torch.sqrt(q_ * q_ * (ss * ss + ddx * ddx) + ddy * ddy)
    aax = (q_ / sq) * torch.atan(sq * ddx / (psi + ss))
    aay = (q_ / sq) * torch.atanh(sq * ddy / (psi + q_ * q_ * ss))

    ax = bb * (aax * co + aay * si)
    ay = bb * (-aax * si + aay * co)
    return ax, ay, bb, si, co, q_, ss


def _sie_hessian(
    bb, si, co, q_, ss,
    ddx, ddy,                      # in body frame (C, ...)
):
    """Return kap, g1, g2 (same shape as ddx) for the SIE."""
    psi = torch.sqrt(q_ * q_ * (ss * ss + ddx * ddx) + ddy * ddy)
    f = (1.0 + q_ * q_) * ss * ss + 2.0 * psi * ss + ddx * ddx + ddy * ddy
    pxx_b = (q_ / psi) * (q_ * q_ * ss * ss + ddy * ddy + ss * psi) / f
    pyy_b = (q_ / psi) * (ss * ss + ddx * ddx + ss * psi) / f
    pxy_b = (q_ / psi) * (-ddx * ddy) / f

    rpxx = co * co * pxx_b + 2.0 * co * si * pxy_b + si * si * pyy_b
    rpyy = si * si * pxx_b - 2.0 * co * si * pxy_b + co * co * pyy_b
    rpxy = si * co * (pyy_b - pxx_b) + (co * co - si * si) * pxy_b

    kap = 0.5 * bb * (rpxx + rpyy)
    g1 = 0.5 * bb * (rpxx - rpyy)
    g2 = bb * rpxy
    return kap, g1, g2


def _pert_all(
    g_ext, tg, x0, y0,              # per-config (C,)
    tx, ty,                         # grid
    fac_p,                          # scalar
):
    C = g_ext.shape[0]
    g_ = g_ext.view(C, 1, 1)
    tg_ = tg.view(C, 1, 1)
    x0_ = x0.view(C, 1, 1)
    y0_ = y0.view(C, 1, 1)
    tx_b = tx.unsqueeze(0); ty_b = ty.unsqueeze(0)
    dx = tx_b - x0_; dy = ty_b - y0_
    co2 = torch.cos(2.0 * (tg_ - 90.0) * math.pi / 180.0)
    si2 = torch.sin(2.0 * (tg_ - 90.0) * math.pi / 180.0)
    ax = fac_p * (-dx * g_ * co2 - dy * g_ * si2)   # pert k=0 here
    ay = fac_p * (dy * g_ * co2 - dx * g_ * si2)
    kap = torch.zeros_like(ax)
    gam1 = -g_ * fac_p * co2 * torch.ones_like(ax)
    gam2 = -g_ * fac_p * si2 * torch.ones_like(ax)
    return ax, ay, kap, gam1, gam2


# ---------------------------------------------------------------------------
# Batched grid build + triangle test for (C, Ny, Nx) configs
# ---------------------------------------------------------------------------
def batched_point_solve(
    configs, xs, ys, zs,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float64,
    max_iter=10,
    tol=1.0e-10,
    imag_ceil=1.0e-10,
    chunk_size: int | None = None,
):
    """
    Process `configs` in chunks of `chunk_size` to keep peak VRAM bounded.
    If `chunk_size` is None, process all configs in one GPU pass.
    """
    if chunk_size is not None and chunk_size < len(configs):
        # Split into sub-batches and concatenate results.
        out_all = []
        for i in range(0, len(configs), chunk_size):
            sub = configs[i:i + chunk_size]
            out_all.extend(batched_point_solve(
                sub, xs, ys, zs, device=device, dtype=dtype,
                max_iter=max_iter, tol=tol, imag_ceil=imag_ceil,
                chunk_size=None))
        return out_all
    """
    Solve the lens equation for C configs of (SIE + external shear) against
    one source (xs, ys) at redshift zs.  Returns a list of C lists of
    (x, y, mag, td) tuples.
    """
    C = len(configs)
    cosmo = Cosmology(**COSMO)
    ctx = LensContext.build(cosmo, zl=ZL, zs=zs)
    fac_p = fac_pert(ctx, zs)       # zs_fid = zs for this scenario

    # Stack params.
    sigma = torch.tensor([c["sigma"] for c in configs], device=device, dtype=dtype)
    e = torch.tensor([c["e"] for c in configs], device=device, dtype=dtype)
    q = 1.0 - e
    pa = torch.tensor([c["pa"] for c in configs], device=device, dtype=dtype)
    x0 = torch.tensor([c["x0"] for c in configs], device=device, dtype=dtype)
    y0 = torch.tensor([c["y0"] for c in configs], device=device, dtype=dtype)
    g_ext = torch.tensor([c["g_ext"] for c in configs], device=device, dtype=dtype)
    tg = torch.tensor([c["tg"] for c in configs], device=device, dtype=dtype)

    # Build shared grid.
    dp = GRID["pix_poi"] / (2 ** (GRID["maxlev"] - 1))
    nx = int(math.ceil((GRID["xmax"] - GRID["xmin"]) / dp)) + 1
    ny = int(math.ceil((GRID["ymax"] - GRID["ymin"]) / dp)) + 1
    gxs = torch.linspace(GRID["xmin"], GRID["xmin"] + (nx - 1) * dp, nx,
                         device=device, dtype=dtype)
    gys = torch.linspace(GRID["ymin"], GRID["ymin"] + (ny - 1) * dp, ny,
                         device=device, dtype=dtype)
    gx, gy = torch.meshgrid(gxs, gys, indexing="xy")          # (ny, nx)

    # === (1) Deflection everywhere ===
    ax_s, ay_s, bb, si, co, q_, ss = _sie_deflection(sigma, q, pa, x0, y0,
                                                      gx, gy, ctx)
    ax_p, ay_p, _, _, _ = _pert_all(g_ext, tg, x0, y0, gx, gy, fac_p)
    ax = ax_s + ax_p
    ay = ay_s + ay_p

    # Source-plane positions at each corner.
    sx = gx.unsqueeze(0) - ax                             # (C, ny, nx)
    sy = gy.unsqueeze(0) - ay

    # === (2) Triangle containment test ===
    bl_x = sx[:, :-1, :-1]; bl_y = sy[:, :-1, :-1]
    br_x = sx[:, :-1,  1:]; br_y = sy[:, :-1,  1:]
    tl_x = sx[:,  1:, :-1]; tl_y = sy[:,  1:, :-1]
    tr_x = sx[:,  1:,  1:]; tr_y = sy[:,  1:,  1:]

    def tri(ax_, ay_, bx_, by_, cx_, cy_):
        d1x = xs - ax_; d1y = ys - ay_
        d2x = xs - bx_; d2y = ys - by_
        d3x = xs - cx_; d3y = ys - cy_
        d12 = d1x * d2y - d1y * d2x
        d23 = d2x * d3y - d2y * d3x
        d31 = d3x * d1y - d3y * d1x
        return ((d12 >= 0) & (d23 >= 0) & (d31 >= 0)) | \
               ((d12 <= 0) & (d23 <= 0) & (d31 <= 0))

    in_A = tri(bl_x, bl_y, tr_x, tr_y, br_x, br_y)
    in_B = tri(bl_x, bl_y, tr_x, tr_y, tl_x, tl_y)

    # box-origin coordinates (broadcast from shared grid)
    ox = gx[:-1, :-1].unsqueeze(0).expand_as(in_A)
    oy = gy[:-1, :-1].unsqueeze(0).expand_as(in_A)

    # === (3) Extract candidates (one sync here) ===
    idx_A = torch.nonzero(in_A, as_tuple=False)       # (NA, 3) = (cfg, j, i)
    idx_B = torch.nonzero(in_B, as_tuple=False)

    if idx_A.numel() + idx_B.numel() == 0:
        return [[] for _ in range(C)]

    # Candidate coordinates
    def _gather(idx, off_x, off_y):
        cfg = idx[:, 0]; j = idx[:, 1]; i = idx[:, 2]
        cx = ox[cfg, j, i] + off_x * dp
        cy = oy[cfg, j, i] + off_y * dp
        return cfg, cx, cy

    cfg_A, xA, yA = _gather(idx_A, 0.667, 0.333)
    cfg_B, xB, yB = _gather(idx_B, 0.333, 0.667)

    cand_cfg = torch.cat([cfg_A, cfg_B], dim=0)
    cand_x = torch.cat([xA, xB], dim=0)
    cand_y = torch.cat([yA, yB], dim=0)
    dpi = torch.full_like(cand_x, dp)

    # === (4) Newton refinement ===
    # For each candidate, fetch its config's params.
    g_cand_sigma = sigma[cand_cfg]
    g_cand_q = q[cand_cfg]
    g_cand_pa = pa[cand_cfg]
    g_cand_x0 = x0[cand_cfg]
    g_cand_y0 = y0[cand_cfg]
    g_cand_g = g_ext[cand_cfg]
    g_cand_tg = tg[cand_cfg]

    # reshape to (N, 1, 1) so we can reuse _sie_deflection - actually easier
    # to write a candidate-level kernel that takes per-candidate params.
    xi = cand_x.clone()
    yi = cand_y.clone()
    xi0 = cand_x.clone(); yi0 = cand_y.clone()

    for _ in range(max_iter + 1):
        # Per-candidate SIE+shear full lensmodel.
        ang = -(g_cand_pa - 90.0) * math.pi / 180.0
        si_c = torch.sin(ang); co_c = torch.cos(ang)
        facq = 1.0 / torch.sqrt(g_cand_q)
        bb_c = facq * (4.0 * math.pi * (g_cand_sigma / 2.99792458e5) ** 2
                       * ctx.dis_ls / ctx.dis_os) / 4.84813681e-6
        dx = xi - g_cand_x0; dy = yi - g_cand_y0
        ddx = co_c * dx - si_c * dy; ddy = si_c * dx + co_c * dy
        ss_c = 1.0e-10 * facq
        sq = torch.sqrt(1.0 - g_cand_q * g_cand_q)
        psi = torch.sqrt(g_cand_q * g_cand_q * (ss_c * ss_c + ddx * ddx) + ddy * ddy)
        aax = (g_cand_q / sq) * torch.atan(sq * ddx / (psi + ss_c))
        aay = (g_cand_q / sq) * torch.atanh(sq * ddy / (psi + g_cand_q * g_cand_q * ss_c))
        ax_cs = bb_c * (aax * co_c + aay * si_c)
        ay_cs = bb_c * (-aax * si_c + aay * co_c)

        # SIE Hessian in image plane.
        f = (1.0 + g_cand_q * g_cand_q) * ss_c * ss_c + 2.0 * psi * ss_c + ddx * ddx + ddy * ddy
        pxx_b = (g_cand_q / psi) * (g_cand_q * g_cand_q * ss_c * ss_c + ddy * ddy + ss_c * psi) / f
        pyy_b = (g_cand_q / psi) * (ss_c * ss_c + ddx * ddx + ss_c * psi) / f
        pxy_b = (g_cand_q / psi) * (-ddx * ddy) / f
        rpxx = co_c * co_c * pxx_b + 2.0 * co_c * si_c * pxy_b + si_c * si_c * pyy_b
        rpyy = si_c * si_c * pxx_b - 2.0 * co_c * si_c * pxy_b + co_c * co_c * pyy_b
        rpxy = si_c * co_c * (pyy_b - pxx_b) + (co_c * co_c - si_c * si_c) * pxy_b
        kap_s = 0.5 * bb_c * (rpxx + rpyy)
        g1_s = 0.5 * bb_c * (rpxx - rpyy)
        g2_s = bb_c * rpxy

        # Shear.
        co2 = torch.cos(2.0 * (g_cand_tg - 90.0) * math.pi / 180.0)
        si2 = torch.sin(2.0 * (g_cand_tg - 90.0) * math.pi / 180.0)
        ax_cp = fac_p * (-dx * g_cand_g * co2 - dy * g_cand_g * si2)
        ay_cp = fac_p * (dy * g_cand_g * co2 - dx * g_cand_g * si2)
        g1_p = -g_cand_g * fac_p * co2
        g2_p = -g_cand_g * fac_p * si2

        ax_tot = ax_cs + ax_cp
        ay_tot = ay_cs + ay_cp
        kap_tot = kap_s
        g1_tot = g1_s + g1_p
        g2_tot = g2_s + g2_p

        pxx = kap_tot + g1_tot
        pyy = kap_tot - g1_tot
        pxy = g2_tot
        ff = xs - xi + ax_tot
        gg = ys - yi + ay_tot
        mm = (1.0 - pxx) * (1.0 - pyy) - pxy * pxy
        dx_n = ((1.0 - pyy) * ff + pxy * gg) / mm
        dy_n = ((1.0 - pxx) * gg + pxy * ff) / mm
        xi = xi + dx_n
        yi = yi + dy_n

    # Magnifications at the refined positions.
    muinv = (1.0 - kap_tot) ** 2 - (g1_tot * g1_tot + g2_tot * g2_tot)
    mag = 1.0 / (muinv + imag_ceil)
    td_raw = ctx.tdelay_fac * (0.5 * (ax_tot * ax_tot + ay_tot * ay_tot))  # phi=0 here -- no SIE phi

    # Reject runaways.
    dist2 = (xi - xi0) ** 2 + (yi - yi0) ** 2
    keep = dist2 <= (2.0 * dpi * dpi)

    # Move to CPU for dedup + grouping.
    cfg_cpu = cand_cfg.cpu().tolist()
    x_cpu = xi.cpu().tolist()
    y_cpu = yi.cpu().tolist()
    mag_cpu = mag.cpu().tolist()
    td_cpu = td_raw.cpu().tolist()
    keep_cpu = keep.cpu().tolist()

    out: list[list[tuple[float, float, float, float]]] = [[] for _ in range(C)]
    for i, c in enumerate(cfg_cpu):
        if not keep_cpu[i]:
            continue
        out[c].append((x_cpu[i], y_cpu[i], mag_cpu[i], td_cpu[i]))

    # Duplicate removal + td re-zero per config.
    result: list[list[tuple[float, float, float, float]]] = []
    for imgs in out:
        filtered = []
        for ii, (xi_, yi_, mi_, ti_) in enumerate(imgs):
            dup = False
            for xj_, yj_, mj_, _ in filtered:
                mm_ = abs(mi_ * mj_)
                dd = ((xi_ - xj_) ** 2 + (yi_ - yj_) ** 2) / max(mm_, 1e-300)
                if dd <= 10.0 * tol * tol:
                    dup = True
                    break
            if not dup:
                filtered.append((xi_, yi_, mi_, ti_))
        if filtered:
            tdmin = min(t for _, _, _, t in filtered)
            filtered = [(x_, y_, m_, t_ - tdmin) for x_, y_, m_, t_ in filtered]
        result.append(filtered)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    C = int(os.environ.get("C", "64"))
    N_REPS = int(os.environ.get("N_REPS", "10"))
    CHUNK = os.environ.get("CHUNK")
    chunk = int(CHUNK) if CHUNK else None
    DTYPE = os.environ.get("DTYPE", "f64")
    dtype = torch.float64 if DTYPE == "f64" else torch.float32
    cfgs = make_configs(C)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, C={C}, reps={N_REPS}, chunk={chunk}, dtype={dtype}")

    # warm-up
    _ = batched_point_solve(cfgs, SOURCE_XS, SOURCE_YS, ZS,
                            device=device, dtype=dtype, chunk_size=chunk)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_REPS):
        out = batched_point_solve(cfgs, SOURCE_XS, SOURCE_YS, ZS,
                                  device=device, dtype=dtype, chunk_size=chunk)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total = (t1 - t0) / N_REPS
    total_imgs = sum(len(o) for o in out)
    print(f"batched point_solve for {C} configs: {total * 1000:.2f} ms "
          f"({total / C * 1000:.3f} ms per config)")
    print(f"total images found across configs: {total_imgs}")
    print(f"sample: config 0 has {len(out[0])} images")


if __name__ == "__main__":
    main()
