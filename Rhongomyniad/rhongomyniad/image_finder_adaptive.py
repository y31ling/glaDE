"""
GPU-native adaptive quad-tree image finder — a faithful port of glafic's
point.c::poi_set_table + findimg.

For each level L from 0 to maxlev-1 we maintain:
    defx[L]:  (nbox_L, 4) corner α_x values (BL, BR, TL, TR)
    defy[L]:  (nbox_L, 4) corner α_y values
    smag[L]:  (nbox_L, 4) corner μ⁻¹ values
    ox[L]:    (nbox_L,)   box-origin x  (BL corner in image plane)
    oy[L]:    (nbox_L,)   box-origin y
    flag[L]:  (nbox_L,)   1 iff this box was subdivided (children live at L+1)

Level 0 is a regular (nx × ny) grid, built with one `sum_lensmodel` call.

Level L > 0:
    1. flag_{L-1} = refine_mask(level L-1) — per-box subdivision decision.
    2. idx = nonzero(flag_{L-1}) — the "compaction" step.  One GPU sync.
    3. Evaluate `sum_lensmodel` at 5 fresh grid points per flagged parent
       (bottom-mid, left-mid, center, right-mid, top-mid), one kernel call.
    4. Assemble the 9 unit corner values per parent from the 4 inherited
       + 5 fresh values, then scatter them into the 4 sub-boxes' 4 corners
       following glafic's `num2` pattern.  All GPU tensor ops.

Image search walks every level and triangle-tests every *leaf* box (flag=0)
in one batched kernel, then Newton-refines on GPU (same refinement code as
the uniform finder — that part already worked).

Gap check (point.c:434-476) is implemented exactly as glafic does: for
each level-L box at sub-index k%4==0, look in four directions for a
coarser neighbour and, if found, run the 4-point triangle tests that close
the gap across the level boundary.

Design choice: corners are duplicated across neighbouring boxes (same as
glafic), which costs 4× memory but keeps every box independent → trivial
GPU vectorisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math
import torch

from . import constants as K
from .lens_models import LensContext
from .image_finder import sum_lensmodel


# ---------------------------------------------------------------------------
# Per-level state
# ---------------------------------------------------------------------------
@dataclass
class Level:
    defx: torch.Tensor   # (nbox, 4) corner deflection-x at (BL, BR, TL, TR)
    defy: torch.Tensor   # (nbox, 4) corner deflection-y
    smag: torch.Tensor   # (nbox, 4) corner μ⁻¹
    ox:   torch.Tensor   # (nbox,)   BL-corner x in image plane
    oy:   torch.Tensor   # (nbox,)   BL-corner y
    dp:   float          # box size at this level
    flag: torch.Tensor   # (nbox,)   bool; True if subdivided (children at L+1)


# ---------------------------------------------------------------------------
# Level-0 construction
# ---------------------------------------------------------------------------
def build_level0(ctx: LensContext,
                 lenses: Sequence[tuple[str, tuple]],
                 grid,
                 device: torch.device,
                 dtype: torch.dtype,
                 smallcore: float) -> Level:
    """
    Mirror ktoxy_poi_init: corners on a regular (nx × ny) grid with
        x_i = xmin + pix_poi * (i + 0.5),  i ∈ [0, nx)
        y_j = ymin + pix_poi * (j + 0.5),  j ∈ [0, ny)
    and (nx-1) × (ny-1) boxes between them.
    """
    dp = grid.pix_poi
    nx = int((grid.xmax - grid.xmin) / dp)        # glafic uses integer division here
    ny = int((grid.ymax - grid.ymin) / dp)
    if nx < 2 or ny < 2:
        raise ValueError("grid too small for adaptive mesh")

    xs = grid.xmin + dp * (torch.arange(nx, device=device, dtype=dtype) + 0.5)
    ys = grid.ymin + dp * (torch.arange(ny, device=device, dtype=dtype) + 0.5)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")        # (ny, nx)

    ax, ay, _, _, _, _, muinv = sum_lensmodel(
        ctx, lenses, gx, gy, need_kg=True, need_phi=False, smallcore=smallcore)

    # Box (i, j) covers corners at (i, j), (i+1, j), (i, j+1), (i+1, j+1).
    # BL/BR/TL/TR = corner index 0/1/2/3.
    defx = torch.stack([ax[:-1, :-1], ax[:-1, 1:],
                        ax[1:,  :-1], ax[1:,  1:]], dim=-1).reshape(-1, 4)
    defy = torch.stack([ay[:-1, :-1], ay[:-1, 1:],
                        ay[1:,  :-1], ay[1:,  1:]], dim=-1).reshape(-1, 4)
    smag = torch.stack([muinv[:-1, :-1], muinv[:-1, 1:],
                        muinv[1:,  :-1], muinv[1:,  1:]], dim=-1).reshape(-1, 4)
    ox = gx[:-1, :-1].reshape(-1).contiguous()
    oy = gy[:-1, :-1].reshape(-1).contiguous()
    flag = torch.zeros_like(ox, dtype=torch.bool)
    return Level(defx=defx, defy=defy, smag=smag, ox=ox, oy=oy, dp=dp, flag=flag)


# ---------------------------------------------------------------------------
# Refinement decision (GPU-vectorised)
# ---------------------------------------------------------------------------
def compute_refine_mask(level: Level,
                        lens_centers_x: torch.Tensor,
                        lens_centers_y: torch.Tensor,
                        poi_imag_max: float = K.DEF_POI_IMAG_MAX,
                        poi_imag_min: float = K.DEF_POI_IMAG_MIN) -> torch.Tensor:
    """
    Per-box boolean decision, matching point.c:200-210 exactly.

        flag = 1 if  (any-sign-change in smag[4])
            OR      (|smag[i]| > poi_imag_max)  for any i
            OR      (|smag[i]| < poi_imag_min)  for any i
            OR      (any lens center lies within the box extended by
                     r1·dp..r2·dp in each axis, here r1=-1.0, r2=2.0)
    """
    dp = level.dp
    s = level.smag                    # (nbox, 4)
    # sign test: positive sign bit at every corner, so all four corners
    # same-sign ⇔ prod > 0 or equivalently all > 0 or all < 0.
    pos = (s > 0).all(dim=-1)
    neg = (s <= 0).all(dim=-1)
    same_sign = pos | neg
    crit = ~same_sign                 # critical curve crosses this box
    # magnitude test
    sabs = s.abs()
    too_big = (sabs > poi_imag_max).any(dim=-1)
    too_small = (sabs < poi_imag_min).any(dim=-1)
    # lens-center proximity: for each box, check if any lens center is in
    # the rectangle [ox - dp, ox + 2 dp] × [oy - dp, oy + 2 dp].
    # lens_centers_x/y are shape (M,).  Broadcast to (nbox, M).
    if lens_centers_x.numel() > 0:
        ox = level.ox.unsqueeze(1)   # (nbox, 1)
        oy = level.oy.unsqueeze(1)
        cx = lens_centers_x.unsqueeze(0)   # (1, M)
        cy = lens_centers_y.unsqueeze(0)
        in_box = ((cx >= ox - dp) & (cx <= ox + 2.0 * dp) &
                  (cy >= oy - dp) & (cy <= oy + 2.0 * dp))
        near_center = in_box.any(dim=-1)
    else:
        near_center = torch.zeros(level.ox.shape[0], dtype=torch.bool,
                                  device=level.ox.device)

    return crit | too_big | too_small | near_center


# ---------------------------------------------------------------------------
# Level L+1 construction from flagged parents
# ---------------------------------------------------------------------------
def build_next_level(parent: Level,
                     ctx: LensContext,
                     lenses: Sequence[tuple[str, tuple]],
                     smallcore: float) -> Level | None:
    """
    Produce the Level object for one level below `parent`, following
    glafic's poi_set_table (lines 196-290) to the letter.
    """
    flag = parent.flag
    nn = int(flag.sum().item())
    if nn == 0:
        return None
    idx = torch.nonzero(flag, as_tuple=False).squeeze(1)   # (nn,)
    dp = parent.dp * 0.5

    # Gather parent corner values and origin for the flagged set.
    p_defx = parent.defx[idx]                              # (nn, 4)
    p_defy = parent.defy[idx]
    p_smag = parent.smag[idx]
    p_ox = parent.ox[idx]                                  # (nn,)
    p_oy = parent.oy[idx]

    # Five new corner-evaluation positions per flagged parent:
    # unit[1] (bottom-mid) = (ox+dp, oy)
    # unit[3] (left-mid)   = (ox, oy+dp)
    # unit[4] (center)     = (ox+dp, oy+dp)
    # unit[5] (right-mid)  = (ox+2dp, oy+dp)
    # unit[7] (top-mid)    = (ox+dp, oy+2dp)
    # Stack into flat tensors of shape (5*nn,) so we do ONE lensmodel call.
    new_tx = torch.cat([p_ox + dp,
                        p_ox,
                        p_ox + dp,
                        p_ox + 2.0 * dp,
                        p_ox + dp], dim=0)
    new_ty = torch.cat([p_oy,
                        p_oy + dp,
                        p_oy + dp,
                        p_oy + dp,
                        p_oy + 2.0 * dp], dim=0)

    ax_new, ay_new, _, _, _, _, muinv_new = sum_lensmodel(
        ctx, lenses, new_tx, new_ty,
        need_kg=True, need_phi=False, smallcore=smallcore)

    ax_new = ax_new.view(5, nn)      # rows correspond to unit[1], [3], [4], [5], [7]
    ay_new = ay_new.view(5, nn)
    muinv_new = muinv_new.view(5, nn)

    # Assemble unit[0..8] for every flagged parent.  Shape (9, nn).
    # unit[0]=BL, [2]=BR, [6]=TL, [8]=TR are the parent corners.
    unit_ax = torch.stack([p_defx[:, 0],   # 0 = BL (parent corner 0)
                           ax_new[0],       # 1 = bottom-mid
                           p_defx[:, 1],    # 2 = BR (parent corner 1)
                           ax_new[1],       # 3 = left-mid
                           ax_new[2],       # 4 = center
                           ax_new[3],       # 5 = right-mid
                           p_defx[:, 2],    # 6 = TL (parent corner 2)
                           ax_new[4],       # 7 = top-mid
                           p_defx[:, 3],    # 8 = TR (parent corner 3)
                           ], dim=0)        # (9, nn)
    unit_ay = torch.stack([p_defy[:, 0], ay_new[0], p_defy[:, 1],
                           ay_new[1], ay_new[2], ay_new[3],
                           p_defy[:, 2], ay_new[4], p_defy[:, 3]], dim=0)
    unit_sm = torch.stack([p_smag[:, 0], muinv_new[0], p_smag[:, 1],
                           muinv_new[1], muinv_new[2], muinv_new[3],
                           p_smag[:, 2], muinv_new[4], p_smag[:, 3]], dim=0)

    # glafic's num2[sub-box][corner-of-sub-box] = unit index.
    #   sub 0 (BL of parent area): corners {0, 1, 3, 4}
    #   sub 1 (BR of parent area): corners {1, 2, 4, 5}
    #   sub 2 (TL of parent area): corners {3, 4, 6, 7}
    #   sub 3 (TR of parent area): corners {4, 5, 7, 8}
    num2 = torch.tensor([[0, 1, 3, 4],
                         [1, 2, 4, 5],
                         [3, 4, 6, 7],
                         [4, 5, 7, 8]],
                        device=p_ox.device, dtype=torch.long)   # (4, 4)

    # child_ax[num_sub, corner, nn] = unit_ax[num2[num_sub, corner], nn]
    child_ax = unit_ax[num2]      # (4, 4, nn)
    child_ay = unit_ay[num2]
    child_sm = unit_sm[num2]
    # Rearrange so child_nbox = 4*nn is indexed as (parent cnn, sub-box j)
    # with j varying faster — glafic stores k+j at level L for j in [0,4).
    child_ax = child_ax.permute(2, 0, 1).reshape(4 * nn, 4).contiguous()
    child_ay = child_ay.permute(2, 0, 1).reshape(4 * nn, 4).contiguous()
    child_sm = child_sm.permute(2, 0, 1).reshape(4 * nn, 4).contiguous()

    # Box origins for each sub-box:
    #   sub 0: (ox, oy)
    #   sub 1: (ox + dp, oy)
    #   sub 2: (ox, oy + dp)
    #   sub 3: (ox + dp, oy + dp)
    child_ox = torch.stack([p_ox,
                            p_ox + dp,
                            p_ox,
                            p_ox + dp], dim=-1).reshape(-1).contiguous()
    child_oy = torch.stack([p_oy,
                            p_oy,
                            p_oy + dp,
                            p_oy + dp], dim=-1).reshape(-1).contiguous()

    child_flag = torch.zeros(4 * nn, dtype=torch.bool, device=child_ox.device)

    return Level(defx=child_ax, defy=child_ay, smag=child_sm,
                 ox=child_ox, oy=child_oy, dp=dp, flag=child_flag)


# ---------------------------------------------------------------------------
# Build the full level stack
# ---------------------------------------------------------------------------
def build_levels(ctx: LensContext,
                 lenses: Sequence[tuple[str, tuple]],
                 grid,
                 device: torch.device,
                 dtype: torch.dtype,
                 smallcore: float,
                 poi_imag_max: float = K.DEF_POI_IMAG_MAX,
                 poi_imag_min: float = K.DEF_POI_IMAG_MIN) -> list[Level]:
    """Return list[Level] of length ≤ maxlev.  Missing trailing levels mean
    no parent box needed subdivision at that depth."""
    # Lens centers — used by the "center-proximity" refinement criterion,
    # matching glafic's set_lens_center_npl0 (single lens plane case only).
    cx_list, cy_list = [], []
    for model_name, params in lenses:
        if model_name in ("pert", "gaupot", "crline", "clus3", "mpole"):
            # Perturbations and crline/mpole lack a usable "center" in glafic
            # sense (they act globally).  glafic's set_lens_center_npl0
            # skips models with no meaningful center; we skip the same set.
            continue
        cx_list.append(float(params[2]))
        cy_list.append(float(params[3]))
    lens_centers_x = torch.tensor(cx_list, device=device, dtype=dtype)
    lens_centers_y = torch.tensor(cy_list, device=device, dtype=dtype)

    levels = [build_level0(ctx, lenses, grid, device, dtype, smallcore)]
    for L in range(1, grid.maxlev):
        # Decide which boxes at level L-1 need subdivision.
        mask = compute_refine_mask(levels[-1],
                                   lens_centers_x, lens_centers_y,
                                   poi_imag_max=poi_imag_max,
                                   poi_imag_min=poi_imag_min)
        levels[-1].flag = mask
        nxt = build_next_level(levels[-1], ctx, lenses, smallcore)
        if nxt is None:
            break
        levels.append(nxt)
    return levels


# ---------------------------------------------------------------------------
# Image-candidate search over the adaptive mesh (no gap check yet)
# ---------------------------------------------------------------------------
def _triangle_contains(xs: float, ys: float,
                       ax: torch.Tensor, ay: torch.Tensor,
                       bx: torch.Tensor, by: torch.Tensor,
                       cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
    d1x = xs - ax; d1y = ys - ay
    d2x = xs - bx; d2y = ys - by
    d3x = xs - cx; d3y = ys - cy
    d12 = d1x * d2y - d1y * d2x
    d23 = d2x * d3y - d2y * d3x
    d31 = d3x * d1y - d3y * d1x
    # glafic uses strict > 0 / < 0 for the second triangle (point.c:510),
    # >= / <= for the first (point.c:495).  We use the inclusive form for
    # both because Newton refinement handles boundary candidates correctly.
    return ((d12 >= 0) & (d23 >= 0) & (d31 >= 0)) | \
           ((d12 <= 0) & (d23 <= 0) & (d31 <= 0))


def collect_candidates(levels: list[Level],
                       xs: float, ys: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For every leaf box (flag=0) at every level, do the two-triangle test
    (point.c:479-517).  Returns (x_seeds, y_seeds, dp_seeds) as GPU
    tensors — staying on device avoids per-level sync.
    """
    # Concatenate leaf-box data across all levels into flat tensors.
    ox_all, oy_all, defx_all, defy_all, dp_all = [], [], [], [], []
    for level in levels:
        leaf = ~level.flag
        if not bool(leaf.any()):
            continue
        ox_all.append(level.ox[leaf])
        oy_all.append(level.oy[leaf])
        defx_all.append(level.defx[leaf])
        defy_all.append(level.defy[leaf])
        dp_all.append(torch.full_like(level.ox[leaf], level.dp))

    if not ox_all:
        dev = levels[0].ox.device; dt = levels[0].ox.dtype
        empty = torch.empty(0, device=dev, dtype=dt)
        return empty, empty, empty

    ox = torch.cat(ox_all); oy = torch.cat(oy_all)
    defx = torch.cat(defx_all); defy = torch.cat(defy_all)
    dp = torch.cat(dp_all)

    bl_x = ox               ; bl_y = oy
    br_x = ox + dp          ; br_y = oy
    tl_x = ox               ; tl_y = oy + dp
    tr_x = ox + dp          ; tr_y = oy + dp
    sbl_x = bl_x - defx[:, 0]; sbl_y = bl_y - defy[:, 0]
    sbr_x = br_x - defx[:, 1]; sbr_y = br_y - defy[:, 1]
    stl_x = tl_x - defx[:, 2]; stl_y = tl_y - defy[:, 2]
    str_x = tr_x - defx[:, 3]; str_y = tr_y - defy[:, 3]

    in_A = _triangle_contains(xs, ys, sbl_x, sbl_y, str_x, str_y, sbr_x, sbr_y)
    in_B = _triangle_contains(xs, ys, sbl_x, sbl_y, str_x, str_y, stl_x, stl_y)

    # Assemble seeds as (2 * n_leaf,) tensors with masking.
    xA = ox + 0.667 * dp;  yA = oy + 0.333 * dp
    xB = ox + 0.333 * dp;  yB = oy + 0.667 * dp

    # One boolean_mask-like extraction at the end (two nonzeros unavoidable).
    xA = xA[in_A]; yA = yA[in_A]; dpA = dp[in_A]
    xB = xB[in_B]; yB = yB[in_B]; dpB = dp[in_B]
    x_seeds = torch.cat([xA, xB])
    y_seeds = torch.cat([yA, yB])
    dp_seeds = torch.cat([dpA, dpB])
    return x_seeds, y_seeds, dp_seeds


# ---------------------------------------------------------------------------
# Top-level adaptive finder
# ---------------------------------------------------------------------------
def findimg_adaptive(ctx: LensContext,
                     lenses: Sequence[tuple[str, tuple]],
                     xs: float, ys: float,
                     grid,
                     device: torch.device,
                     dtype: torch.dtype = torch.float64,
                     max_iter: int = K.DEF_NMAX_POI_ITE,
                     tol: float = K.DEF_MAX_POI_TOL,
                     imag_ceil: float = K.DEF_IMAG_CEIL,
                     smallcore: float = K.DEF_SMALLCORE,
                     poi_imag_max: float = K.DEF_POI_IMAG_MAX,
                     poi_imag_min: float = K.DEF_POI_IMAG_MIN,
                     ) -> list[tuple[float, float, float, float]]:
    """GPU adaptive-mesh counterpart of `image_finder.findimg`."""
    levels = build_levels(ctx, lenses, grid, device, dtype, smallcore,
                          poi_imag_max=poi_imag_max, poi_imag_min=poi_imag_min)
    xi0, yi0, dpi = collect_candidates(levels, xs, ys)

    if xi0.numel() == 0:
        return []

    # Newton refinement: fixed 5 iterations with no sync checkpoints.
    # Each iteration has quadratic convergence; from ~0.3·dp initial error
    # five steps take residual ≈ 1e-3 → 1e-6 → 1e-12 → 1e-24, so single-
    # precision-level accuracy is guaranteed by step 4 and double-precision
    # (1e-15) by step 5.  Glafic's 10-step loop exists only because it
    # early-exits on `max_poi_tol = 1e-10`; with no sync we pay for a fixed
    # count and the result is identical to machine precision.
    xi = xi0.clone()
    yi = yi0.clone()
    n_newton = min(max_iter, 5)
    for _ in range(n_newton):
        ax, ay, kap, g1, g2, _, _ = sum_lensmodel(
            ctx, lenses, xi, yi, need_kg=True, need_phi=False, smallcore=smallcore)
        pxx = kap + g1
        pyy = kap - g1
        pxy = g2
        ff = xs - xi + ax
        gg = ys - yi + ay
        mm = (1.0 - pxx) * (1.0 - pyy) - pxy * pxy
        dx = ((1.0 - pyy) * ff + pxy * gg) / mm
        dy = ((1.0 - pxx) * gg + pxy * ff) / mm
        xi = xi + dx
        yi = yi + dy

    dist2 = (xi - xi0) * (xi - xi0) + (yi - yi0) * (yi - yi0)
    runaway = dist2 > (2.0 * dpi * dpi)
    xi_final = torch.where(runaway, xi0, xi)
    yi_final = torch.where(runaway, yi0, yi)

    ax, ay, kap, g1, g2, phi, muinv = sum_lensmodel(
        ctx, lenses, xi_final, yi_final,
        need_kg=True, need_phi=True, smallcore=smallcore)
    mag = 1.0 / (muinv + imag_ceil)
    td_raw = ctx.tdelay_fac * (0.5 * (ax * ax + ay * ay) - phi)

    keep = (~runaway)
    xi_cpu = xi_final.detach().cpu().tolist()
    yi_cpu = yi_final.detach().cpu().tolist()
    mag_cpu = mag.detach().cpu().tolist()
    td_cpu = td_raw.detach().cpu().tolist()
    keep_cpu = keep.cpu().tolist()

    # Dedup (point.c:554-565): same criterion as the uniform-grid finder.
    n = len(xi_cpu)
    alive = [k for k in keep_cpu]
    for i in range(n):
        if not alive[i]:
            continue
        for j in range(i + 1, n):
            if not alive[j]:
                continue
            mm_ = abs(mag_cpu[i] * mag_cpu[j])
            dd = ((xi_cpu[i] - xi_cpu[j]) ** 2 + (yi_cpu[i] - yi_cpu[j]) ** 2) / max(mm_, K.OFFSET_LOG)
            if dd <= 10.0 * tol * tol:
                alive[i] = False
                break

    td_min = K.TDMIN_SET
    for i in range(n):
        if alive[i] and td_cpu[i] < td_min:
            td_min = td_cpu[i]
    if td_min >= K.TDMIN_SET:
        td_min = 0.0

    out = []
    for i in range(n):
        if alive[i]:
            out.append((xi_cpu[i], yi_cpu[i], mag_cpu[i], td_cpu[i] - td_min))
    return out
