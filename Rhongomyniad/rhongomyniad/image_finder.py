"""
GPU image finder for a single lens plane.

Strategy (matches glafic point.c but replaces the adaptive-mesh/Romberg
combo with dense uniform-grid parallel work):

    1. Build a uniform fine grid at spacing dp = pix_poi / 2^{maxlev-1}
       covering [xmin, xmax] x [ymin, ymax].
    2. Evaluate the total deflection angle (ax, ay) at every corner in
       one GPU kernel (batched over all lenses and query points).
    3. For every box, test the two triangles (glafic's diagonal split:
       {bl, tr, br} and {bl, tr, tl}) to decide if the source-plane
       point (xs, ys) lies inside the mapped triangle.  A 2D cross-
       product test gives per-box pass/fail in parallel.
    4. Deduplicate the (tiny) list of candidates, then run Newton
       refinement (on GPU) to convergence.
    5. Evaluate magnification and time delay at the refined images,
       drop images whose Newton iteration ran away, subtract the
       minimum time delay, and return.

Output format matches glafic's `findimg`:
    list of (x_image, y_image, magnification, time_delay_days)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, List

import math
import torch

from . import constants as K
from .lens_models import LensContext, dispatch


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------
@dataclass
class GridSpec:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    pix_poi: float
    maxlev: int

    @property
    def dp(self) -> float:
        """Fine-grid spacing."""
        return self.pix_poi / (2 ** (self.maxlev - 1))

    def corner_counts(self) -> tuple[int, int]:
        """(nx, ny) = number of corner grid points in each axis."""
        dp = self.dp
        nx = int(math.ceil((self.xmax - self.xmin) / dp)) + 1
        ny = int(math.ceil((self.ymax - self.ymin) / dp)) + 1
        return nx, ny


# ---------------------------------------------------------------------------
# Total lensing calculation (sum over all lenses in a single plane)
# ---------------------------------------------------------------------------
def sum_lensmodel(ctx: LensContext,
                  lenses: Sequence[tuple[str, tuple]],
                  tx: torch.Tensor, ty: torch.Tensor,
                  need_kg: bool = True, need_phi: bool = True,
                  smallcore: float = K.DEF_SMALLCORE):
    """
    Compute (ax, ay, kap, g1, g2, phi, muinv) summed over every lens.

    `lenses` is a list of (model_name, params) tuples.
    """
    ax = torch.zeros_like(tx)
    ay = torch.zeros_like(tx)
    kap = torch.zeros_like(tx) if need_kg else None
    g1 = torch.zeros_like(tx) if need_kg else None
    g2 = torch.zeros_like(tx) if need_kg else None
    phi = torch.zeros_like(tx) if (need_kg and need_phi) else None

    for model_name, params in lenses:
        kernel = dispatch(model_name)
        rax, ray, rk, rg1, rg2, rph = kernel(
            ctx, tx, ty, params, smallcore=smallcore,
            need_kg=need_kg, need_phi=need_phi)
        ax = ax + rax
        ay = ay + ray
        if need_kg:
            kap = kap + rk
            g1 = g1 + rg1
            g2 = g2 + rg2
            if need_phi and rph is not None:
                phi = phi + rph

    muinv = None
    if need_kg:
        muinv = (1.0 - kap) ** 2 - (g1 * g1 + g2 * g2)
    return ax, ay, kap, g1, g2, phi, muinv


# ---------------------------------------------------------------------------
# Candidate image search via triangle mapping on the uniform fine grid
# ---------------------------------------------------------------------------
def _triangle_contains(xs: float, ys: float,
                       ax: torch.Tensor, ay: torch.Tensor,
                       bx: torch.Tensor, by: torch.Tensor,
                       cx: torch.Tensor, cy: torch.Tensor):
    """Vectorised 2D triangle-containment test using cross products."""
    d1x = xs - ax; d1y = ys - ay
    d2x = xs - bx; d2y = ys - by
    d3x = xs - cx; d3y = ys - cy
    d12 = d1x * d2y - d1y * d2x
    d23 = d2x * d3y - d2y * d3x
    d31 = d3x * d1y - d3y * d1x
    all_pos = (d12 >= 0) & (d23 >= 0) & (d31 >= 0)
    all_neg = (d12 <= 0) & (d23 <= 0) & (d31 <= 0)
    return all_pos | all_neg


def find_candidates(ctx: LensContext,
                    lenses: Sequence[tuple[str, tuple]],
                    xs: float, ys: float,
                    grid: GridSpec,
                    device: torch.device,
                    dtype: torch.dtype = torch.float64,
                    smallcore: float = K.DEF_SMALLCORE) -> list[tuple[float, float, float]]:
    """
    Return a list of (x_init, y_init, dp) tuples: coarse image positions
    inside candidate triangles.  These feed Newton refinement.
    """
    dp = grid.dp
    nx, ny = grid.corner_counts()

    # Corner coordinates (row-major: x varies fastest).
    xs_corner = torch.linspace(grid.xmin, grid.xmin + (nx - 1) * dp, nx,
                               device=device, dtype=dtype)
    ys_corner = torch.linspace(grid.ymin, grid.ymin + (ny - 1) * dp, ny,
                               device=device, dtype=dtype)
    gx, gy = torch.meshgrid(xs_corner, ys_corner, indexing="xy")
    # gx, gy have shape (ny, nx) because indexing="xy".

    # Evaluate deflection at every corner.
    ax, ay, _, _, _, _, _ = sum_lensmodel(
        ctx, lenses, gx, gy, need_kg=False, need_phi=False, smallcore=smallcore)

    # Source-plane positions at each corner.
    sx = gx - ax
    sy = gy - ay

    # Box corners (bottom-left, bottom-right, top-left, top-right).
    bl_x = sx[:-1, :-1]; bl_y = sy[:-1, :-1]       # i in [0, nx-1), j in [0, ny-1)
    br_x = sx[:-1,  1:]; br_y = sy[:-1,  1:]
    tl_x = sx[ 1:, :-1]; tl_y = sy[ 1:, :-1]
    tr_x = sx[ 1:,  1:]; tr_y = sy[ 1:,  1:]

    # Triangle A: (bl, tr, br); Triangle B: (bl, tr, tl)
    in_A = _triangle_contains(xs, ys, bl_x, bl_y, tr_x, tr_y, br_x, br_y)
    in_B = _triangle_contains(xs, ys, bl_x, bl_y, tr_x, tr_y, tl_x, tl_y)

    # box-origin x/y (= bottom-left image-plane coords)
    ox = gx[:-1, :-1]
    oy = gy[:-1, :-1]

    # Collect candidates: triangle A -> (xx + 2/3 dp, yy + 1/3 dp); B -> (1/3, 2/3).
    candidates: list[tuple[float, float, float]] = []
    if in_A.any():
        idx = torch.nonzero(in_A, as_tuple=False)
        for j, i in idx.tolist():
            candidates.append((float(ox[j, i]) + 0.667 * dp,
                               float(oy[j, i]) + 0.333 * dp, dp))
    if in_B.any():
        idx = torch.nonzero(in_B, as_tuple=False)
        for j, i in idx.tolist():
            candidates.append((float(ox[j, i]) + 0.333 * dp,
                               float(oy[j, i]) + 0.667 * dp, dp))
    return candidates


# ---------------------------------------------------------------------------
# Newton refinement (vectorised over all candidates)
# ---------------------------------------------------------------------------
def _newton_refine(ctx: LensContext,
                   lenses: Sequence[tuple[str, tuple]],
                   xs: float, ys: float,
                   xi0: torch.Tensor, yi0: torch.Tensor,
                   dpi: torch.Tensor,
                   max_iter: int = K.DEF_NMAX_POI_ITE,
                   tol: float = K.DEF_MAX_POI_TOL,
                   smallcore: float = K.DEF_SMALLCORE):
    """
    Refine a batch of (xi, yi) candidates via glafic's Newton step
    (point.c:523-552).  Returns refined (xi, yi) and a boolean flag
    tensor: True where the iteration ran away (> 2*dp^2) -- these
    candidates are rejected.
    """
    xi = xi0.clone()
    yi = yi0.clone()

    # We run up to max_iter+1 evaluations because glafic's do-while structure
    # evaluates residuals on the final iterate before exiting.
    for _ in range(max_iter + 1):
        ax, ay, kap, g1, g2, _, _ = sum_lensmodel(
            ctx, lenses, xi, yi, need_kg=True, need_phi=False,
            smallcore=smallcore)
        # glafic assumes rot=0 in findimg (point.c:532-533), i.e. single plane.
        pxx = kap + g1
        pyy = kap - g1
        pxy = g2
        pyx = g2
        ff = xs - xi + ax
        gg = ys - yi + ay
        mm = (1.0 - pxx) * (1.0 - pyy) - pxy * pyx
        dx = ((1.0 - pyy) * ff + pxy * gg) / mm
        dy = ((1.0 - pxx) * gg + pyx * ff) / mm
        xi = xi + dx
        yi = yi + dy
        # Terminate early if all converged (unused in strict glafic parity).
        if (ff.abs().max().item() <= tol) and (gg.abs().max().item() <= tol):
            break

    # Reject runaway iterations: distance from initial guess > sqrt(2)*dp.
    dist2 = (xi - xi0) * (xi - xi0) + (yi - yi0) * (yi - yi0)
    runaway = dist2 > (2.0 * dpi * dpi)
    # glafic keeps the original position in that case (so downstream muinv eval
    # doesn't blow up), then marks them fff=1 for removal.
    xi_final = torch.where(runaway, xi0, xi)
    yi_final = torch.where(runaway, yi0, yi)
    return xi_final, yi_final, runaway


# ---------------------------------------------------------------------------
# Top-level image finder
# ---------------------------------------------------------------------------
def findimg(ctx: LensContext,
            lenses: Sequence[tuple[str, tuple]],
            xs: float, ys: float,
            grid: GridSpec,
            device: torch.device,
            dtype: torch.dtype = torch.float64,
            max_iter: int = K.DEF_NMAX_POI_ITE,
            tol: float = K.DEF_MAX_POI_TOL,
            imag_ceil: float = K.DEF_IMAG_CEIL,
            smallcore: float = K.DEF_SMALLCORE,
            ) -> list[tuple[float, float, float, float]]:
    """
    Solve the lens equation for a source at (xs, ys, zs=ctx.zs).

    Returns a list of (x_img, y_img, magnification, time_delay_days) tuples,
    with time delay referenced to the minimum across the returned images.
    """
    candidates = find_candidates(ctx, lenses, xs, ys, grid, device,
                                 dtype=dtype, smallcore=smallcore)
    if len(candidates) == 0:
        return []

    xs_t = xs
    ys_t = ys
    xi0 = torch.tensor([c[0] for c in candidates], device=device, dtype=dtype)
    yi0 = torch.tensor([c[1] for c in candidates], device=device, dtype=dtype)
    dpi = torch.tensor([c[2] for c in candidates], device=device, dtype=dtype)

    xi, yi, runaway = _newton_refine(ctx, lenses, xs_t, ys_t, xi0, yi0, dpi,
                                     max_iter=max_iter, tol=tol,
                                     smallcore=smallcore)

    # Full evaluation at refined positions for magnification + time delay.
    ax, ay, kap, g1, g2, phi, muinv = sum_lensmodel(
        ctx, lenses, xi, yi, need_kg=True, need_phi=True, smallcore=smallcore)
    mag = 1.0 / (muinv + imag_ceil)
    td_raw = ctx.tdelay_fac * (0.5 * (ax * ax + ay * ay) - phi)

    # Duplicate removal (point.c:554-565): d^2 / |mag_i*mag_j| <= 10*tol^2.
    n = xi.shape[0]
    keep = torch.ones(n, dtype=torch.bool, device=device)
    keep = keep & (~runaway)
    xi_cpu = xi.detach().cpu().tolist()
    yi_cpu = yi.detach().cpu().tolist()
    mag_cpu = mag.detach().cpu().tolist()
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            mm = abs(mag_cpu[i] * mag_cpu[j])
            dd = ((xi_cpu[i] - xi_cpu[j]) ** 2 + (yi_cpu[i] - yi_cpu[j]) ** 2) / max(mm, K.OFFSET_LOG)
            if dd <= 10.0 * tol * tol:
                # glafic marks the first duplicate removed, keeps the later entry.
                keep[i] = False
                break

    mask = keep.cpu().tolist()
    # Time delay re-zero
    td_vals = td_raw.detach().cpu().tolist()
    td_min = K.TDMIN_SET
    for i, k in enumerate(mask):
        if k and td_vals[i] < td_min:
            td_min = td_vals[i]
    if td_min >= K.TDMIN_SET:  # edge case: no kept images
        td_min = 0.0

    images: list[tuple[float, float, float, float]] = []
    for i, k in enumerate(mask):
        if not k:
            continue
        images.append((xi_cpu[i], yi_cpu[i], mag_cpu[i], td_vals[i] - td_min))
    return images
