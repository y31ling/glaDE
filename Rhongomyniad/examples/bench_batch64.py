"""
"What if we compute 64 lens configs at once?" benchmark.

Bench A — sequential 64 point_solve calls with SIE+shear parameters varying
per call (mimics one DE generation).  Both glafic and Rhongomyniad go through
their respective Python bindings one call at a time.

Bench B — batched raw-deflection evaluation.  Stack 64 lens-parameter sets
into tensors, broadcast against a single grid, and evaluate α(x,y) for every
(config, corner) pair in ONE fused GPU call.  This skips the full image
finder but shows where Rhongomyniad's architecture actually beats glafic:
when the 64 configs share a kernel launch, Python/Torch dispatch overhead is
paid once instead of 64 times.

Run on Windows (Rhongomyniad only):
    python examples/bench_batch64.py

Run on WSL (glafic too):
    BENCH_GLAFIC=1 PYTHONPATH=/home/luukiaun/glafic251018/work/glade/glafic2/python \
      python3 examples/bench_batch64.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
N_CONFIGS = 64
# SIE + shear varying sigma/e around nominal.  Same grid every call.
GRID = dict(xmin=-30.0, ymin=-30.0, xmax=30.0, ymax=30.0,
            pix_ext=0.1, pix_poi=1.0, maxlev=5)
COSMO = dict(omega=0.3, lam=0.7, weos=-1.0, hubble=0.7)
SOURCE = dict(zs=2.0, xs=-0.15, ys=0.05)


def configs():
    """Produce 64 SIE+shear parameter sets (sigma_v varying, other stuff fixed)."""
    cfgs = []
    for i in range(N_CONFIGS):
        sigma = 280.0 + 0.5 * i       # 280 ... 311 km/s
        e = 0.28 + 0.002 * (i % 20)   # gently vary
        cfgs.append(dict(
            sie=(0.5, sigma, 0.0, 0.0, e, 0.0, 0.0, 0.0),
            pert=(0.5, 2.0, 0.0, 0.0, 0.05, 60.0, 0.0, 0.0),
        ))
    return cfgs


# ---------------------------------------------------------------------
# Bench A — sequential point_solve × 64
# ---------------------------------------------------------------------
def bench_A_glafic(cfgs):
    import glafic
    glafic.init(COSMO["omega"], COSMO["lam"], COSMO["weos"], COSMO["hubble"],
                "b", GRID["xmin"], GRID["ymin"], GRID["xmax"], GRID["ymax"],
                GRID["pix_ext"], GRID["pix_poi"], GRID["maxlev"], 0, 0)
    glafic.startup_setnum(2, 0, 1)
    for lens in cfgs[0].values(): pass
    glafic.set_lens(1, "sie", *cfgs[0]["sie"])
    glafic.set_lens(2, "pert", *cfgs[0]["pert"])
    glafic.set_point(1, SOURCE["zs"], SOURCE["xs"], SOURCE["ys"])
    glafic.model_init(verb=0)
    glafic.point_solve(SOURCE["zs"], SOURCE["xs"], SOURCE["ys"], 0)  # warm

    t0 = time.perf_counter()
    total_imgs = 0
    for c in cfgs:
        # Change lens params and re-solve.  glafic.set_lens forces model reinit.
        glafic.set_lens(1, "sie", *c["sie"])
        glafic.set_lens(2, "pert", *c["pert"])
        glafic.model_init(verb=0)
        imgs = glafic.point_solve(SOURCE["zs"], SOURCE["xs"], SOURCE["ys"], 0)
        total_imgs += len(imgs)
    t1 = time.perf_counter()
    return t1 - t0, total_imgs


def bench_A_rhongomyniad(cfgs):
    import torch
    import rhongomyniad as rh
    rh.init(COSMO["omega"], COSMO["lam"], COSMO["weos"], COSMO["hubble"],
            "b", GRID["xmin"], GRID["ymin"], GRID["xmax"], GRID["ymax"],
            GRID["pix_ext"], GRID["pix_poi"], GRID["maxlev"], verb=0)
    rh.startup_setnum(2, 0, 1)
    rh.set_lens(1, "sie", *cfgs[0]["sie"])
    rh.set_lens(2, "pert", *cfgs[0]["pert"])
    rh.set_point(1, SOURCE["zs"], SOURCE["xs"], SOURCE["ys"])
    rh.model_init(verb=0)
    rh.point_solve(SOURCE["zs"], SOURCE["xs"], SOURCE["ys"])  # warm
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    total_imgs = 0
    for c in cfgs:
        rh.set_lens(1, "sie", *c["sie"])
        rh.set_lens(2, "pert", *c["pert"])
        rh.model_init(verb=0)
        imgs = rh.point_solve(SOURCE["zs"], SOURCE["xs"], SOURCE["ys"])
        total_imgs += len(imgs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return t1 - t0, total_imgs


# ---------------------------------------------------------------------
# Bench B — batched raw deflection evaluation on GPU
#
# Rewrites SIE + pert as fully-vectorised kernels accepting per-config
# parameter tensors of shape (C,) broadcast against a shared grid of
# shape (H, W).  Output: (ax, ay) of shape (C, H, W).
# ---------------------------------------------------------------------
def bench_B_rhongomyniad(cfgs):
    import math
    import torch
    from rhongomyniad.cosmology import Cosmology, tdelay_fac
    from rhongomyniad.lens_models import LensContext, _alpha_sie_dl, facq_sie, b_sie, fac_pert
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    # Build shared grid (one copy).
    dp = GRID["pix_poi"] / (2 ** (GRID["maxlev"] - 1))
    nx = int(math.ceil((GRID["xmax"] - GRID["xmin"]) / dp)) + 1
    ny = int(math.ceil((GRID["ymax"] - GRID["ymin"]) / dp)) + 1
    xs = torch.linspace(GRID["xmin"], GRID["xmin"] + (nx - 1) * dp, nx,
                        device=device, dtype=dtype)
    ys = torch.linspace(GRID["ymin"], GRID["ymin"] + (ny - 1) * dp, ny,
                        device=device, dtype=dtype)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")       # (ny, nx)

    # Stack per-config scalar params into (C,) tensors.
    C = len(cfgs)
    sigma = torch.tensor([c["sie"][1] for c in cfgs], device=device, dtype=dtype)
    ee = torch.tensor([c["sie"][4] for c in cfgs], device=device, dtype=dtype)
    pa = torch.tensor([c["sie"][5] for c in cfgs], device=device, dtype=dtype)
    x0 = torch.tensor([c["sie"][2] for c in cfgs], device=device, dtype=dtype)
    y0 = torch.tensor([c["sie"][3] for c in cfgs], device=device, dtype=dtype)
    g  = torch.tensor([c["pert"][4] for c in cfgs], device=device, dtype=dtype)
    tg = torch.tensor([c["pert"][5] for c in cfgs], device=device, dtype=dtype)

    # Pre-compute SIE normalisation that depends on (sigma, q, distances).
    # All 64 configs share zl=0.5 and zs=2.0 so distances are fixed.
    cosmo = Cosmology(**COSMO)
    ctx = LensContext.build(cosmo, zl=0.5, zs=SOURCE["zs"])
    q = 1.0 - ee
    fac_q = 1.0 / torch.sqrt(q)       # facq_sie
    ss_const = sigma / 2.99792458e5
    bb = fac_q * (4.0 * math.pi * ss_const * ss_const * ctx.dis_ls / ctx.dis_os) / 4.84813681e-6
    # pert fac
    fac_p = fac_pert(ctx, 2.0)        # scalar

    # Reshape for broadcasting: params (C, 1, 1), grid (1, H, W) -> (C, H, W)
    bb_b = bb.view(C, 1, 1)
    q_b = q.view(C, 1, 1)
    pa_b = pa.view(C, 1, 1)
    x0_b = x0.view(C, 1, 1)
    y0_b = y0.view(C, 1, 1)
    gx_b = gx.unsqueeze(0)            # (1, H, W)
    gy_b = gy.unsqueeze(0)
    g_b = g.view(C, 1, 1)
    tg_b = tg.view(C, 1, 1)

    # Warm up generously to JIT-compile kernels and stabilise allocator.
    for _ in range(5):
        ax, ay = batched_deflection(bb_b, q_b, pa_b, x0_b, y0_b, gx_b, gy_b,
                                    g_b, tg_b, fac_p)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time 10 batched evaluations and report the average.
    N_REPS = 10
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        ax, ay = batched_deflection(bb_b, q_b, pa_b, x0_b, y0_b, gx_b, gy_b,
                                    g_b, tg_b, fac_p)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / N_REPS, int(ax.numel())


def batched_deflection(bb_b, q_b, pa_b, x0_b, y0_b, gx_b, gy_b, g_b, tg_b, fac_p):
    """SIE + external shear, fully batched over (C, H, W).  Returns (ax, ay)."""
    import math
    import torch
    # SIE: use pa-90 convention.
    pa_shift = (pa_b - 90.0) * (math.pi / 180.0)
    si = -torch.sin(pa_shift)
    co = torch.cos(pa_shift)            # (C,1,1)

    # Rotate body frame.
    dx = gx_b - x0_b
    dy = gy_b - y0_b
    ddx = co * dx - si * dy
    ddy = si * dx + co * dy
    s = 1.0e-10                          # smallcore
    ss = s * (1.0 / torch.sqrt(q_b))     # facq_sie

    # alpha_sie_dl vectorised.
    sq = torch.sqrt(1.0 - q_b * q_b)
    psi = torch.sqrt(q_b * q_b * (ss * ss + ddx * ddx) + ddy * ddy)
    aax = (q_b / sq) * torch.atan(sq * ddx / (psi + ss))
    aay = (q_b / sq) * torch.atanh(sq * ddy / (psi + q_b * q_b * ss))

    # Back-rotate.
    ax_sie = bb_b * (aax * co + aay * si)
    ay_sie = bb_b * (-aax * si + aay * co)

    # External shear.
    cos_2 = torch.cos(2.0 * (tg_b - 90.0) * math.pi / 180.0)
    sin_2 = torch.sin(2.0 * (tg_b - 90.0) * math.pi / 180.0)
    ax_p = fac_p * (-dx * g_b * cos_2 - dy * g_b * sin_2)
    ay_p = fac_p * ( dy * g_b * cos_2 - dx * g_b * sin_2)

    return ax_sie + ax_p, ay_sie + ay_p


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    cfgs = configs()
    print(f"64 configs, grid = {GRID['xmax']-GRID['xmin']:.0f}\"/side, "
          f"pix_poi = {GRID['pix_poi']}, maxlev = {GRID['maxlev']}")
    print()

    print("=== Bench A: sequential 64 point_solve calls (full image finder) ===")
    if os.environ.get("BENCH_GLAFIC") == "1":
        dt_g, n_g = bench_A_glafic(cfgs)
        print(f"  glafic:        {dt_g * 1000:.1f} ms  ({dt_g / N_CONFIGS * 1000:.2f} ms/call)  {n_g} images total")
    else:
        print(f"  glafic:        (skipped; set BENCH_GLAFIC=1 in WSL)")
    dt_r, n_r = bench_A_rhongomyniad(cfgs)
    print(f"  Rhongomyniad:  {dt_r * 1000:.1f} ms  ({dt_r / N_CONFIGS * 1000:.2f} ms/call)  {n_r} images total")

    print()
    print("=== Bench B: batched raw deflection, 64 configs × whole grid in one kernel ===")
    dt_b, nsamples = bench_B_rhongomyniad(cfgs)
    per_eval_ns = dt_b / nsamples * 1e9
    print(f"  Rhongomyniad (batched): {dt_b * 1000:.2f} ms for {nsamples} deflections total "
          f"({per_eval_ns:.2f} ns / deflection)")
    if os.environ.get("BENCH_GLAFIC") == "1":
        # Also time glafic's 64 × nsamples_per_config calcimage calls for reference.
        import math
        dp = GRID["pix_poi"] / (2 ** (GRID["maxlev"] - 1))
        per_cfg = nsamples // N_CONFIGS
        print(f"  (per config: {per_cfg} grid points; batched Rhongomyniad computes all "
              f"{N_CONFIGS}×{per_cfg} at once)")
        print(f"  No direct glafic equivalent: glafic always rebuilds its grid per call.")
        print(f"  64×point_solve time above is the apples-to-apples comparison.")


if __name__ == "__main__":
    main()
