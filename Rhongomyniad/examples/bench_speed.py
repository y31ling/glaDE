"""
Speed benchmark: Rhongomyniad (GPU) vs Rhongomyniad (CPU) on repeated
point_solve calls, which is the inner loop of the DE optimiser in glade.

To compare against glafic, run this same script in WSL Python with
`BENCH_BACKEND=glafic` (requires glafic.so on PYTHONPATH).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


BENCH_BACKEND = os.environ.get("BENCH_BACKEND", "rh")
N_ITERS = int(os.environ.get("N_ITERS", "30"))


def bench_rh(device: str) -> tuple[float, int]:
    import torch
    import rhongomyniad as rh
    rh.set_device(device)

    rh.init(0.3, 0.7, -1.0, 0.7, "bench",
            -30.0, -30.0, 30.0, 30.0, 0.1, 1.0, 5, verb=0)
    rh.startup_setnum(2, 0, 1)
    rh.set_lens(1, "sie",  0.5, 300.0, 0.0, 0.0, 0.35,  0.0, 0.0, 0.0)
    rh.set_lens(2, "pert", 0.5,   2.0, 0.0, 0.0, 0.05, 60.0, 0.0, 0.0)
    rh.set_point(1, 2.0, -0.15, 0.05)
    rh.model_init(verb=0)

    # warm-up
    rh.point_solve(2.0, -0.15, 0.05)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_imgs = 0
    for i in range(N_ITERS):
        # wiggle source position slightly to avoid caching
        imgs = rh.point_solve(2.0, -0.15 + 0.001 * i, 0.05)
        n_imgs += len(imgs)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0), n_imgs


def bench_glafic() -> tuple[float, int]:
    import glafic
    glafic.init(0.3, 0.7, -1.0, 0.7, "bench",
                -30.0, -30.0, 30.0, 30.0, 0.1, 1.0, 5, 0, 0)
    glafic.startup_setnum(2, 0, 1)
    glafic.set_lens(1, "sie",  0.5, 300.0, 0.0, 0.0, 0.35,  0.0, 0.0, 0.0)
    glafic.set_lens(2, "pert", 0.5,   2.0, 0.0, 0.0, 0.05, 60.0, 0.0, 0.0)
    glafic.set_point(1, 2.0, -0.15, 0.05)
    glafic.model_init(verb=0)
    glafic.point_solve(2.0, -0.15, 0.05, 0)
    t0 = time.perf_counter()
    n_imgs = 0
    for i in range(N_ITERS):
        imgs = glafic.point_solve(2.0, -0.15 + 0.001 * i, 0.05, 0)
        n_imgs += len(imgs)
    t1 = time.perf_counter()
    return (t1 - t0), n_imgs


def main() -> int:
    print(f"backend = {BENCH_BACKEND}, iters = {N_ITERS}")
    if BENCH_BACKEND == "glafic":
        dt, n = bench_glafic()
    elif BENCH_BACKEND == "rh-cpu":
        dt, n = bench_rh("cpu")
    else:
        dt, n = bench_rh("cuda")
    print(f"total: {dt:.3f}s   per call: {dt / N_ITERS * 1e3:.2f}ms   "
          f"total images found: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
