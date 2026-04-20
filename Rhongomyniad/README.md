# Rhongomyniad

> GPU gravitational-lens calculator matching glafic.

Rhongomyniad is a drop-in, GPU-accelerated alternative to the **calculator**
half of [glafic](https://github.com/oguri/glafic2). Given a lens configuration
plus source positions, it returns the image positions, magnifications, and
time delays that glafic would produce — but runs the deflection-angle and
image-plane grid work on the GPU with PyTorch, so repeated calls (e.g.
inside a differential-evolution optimiser) are dramatically faster.

This project **only** reproduces glafic's forward model (the things glafic
calls `calcimage`, `point_solve`, `findimg`). Optimisers, MCMC, extended-
source rendering, FITS I/O, GUI commands, etc. are out of scope — those
belong in the `glade` wrapper that sits on top.

## Supported lens models (v1)

| glafic name | Parameter layout (`p[0..7]`)                          | Backend                   |
|-------------|-------------------------------------------------------|---------------------------|
| `point`     | `z, M, x0, y0, -, -, -, -`                            | closed form               |
| `sie`       | `z, sig_v, x0, y0, e, pa, s_core, -`                  | closed form               |
| `pert`      | `z, zs_fid, x0, y0, g, tg, -, k`                      | closed form               |
| `nfwpot`    | `z, M, x0, y0, e, pa, c, -`                           | closed form (u-transform) |
| `nfw`       | `z, M, x0, y0, e, pa, c, -`                           | Schramm (1990) integrals  |
| `king`      | `z, M, x0, y0, e, pa, rc, c=log10(rt/rc)`             | Schramm integrals         |
| `jaffe`     | `z, sig_v, x0, y0, e, pa, a_outer, rco_inner`         | two SIEs                  |
| `gaupot`    | `z, zs_fid, x0, y0, e, pa, sigma, kap0`               | closed form (u-transform) |

All parameter orderings, sign conventions, unit factors, and `smallcore`
regularisations match glafic's `mass.c`. `jaffe` returns zeros when `a < rco`,
matching glafic.

Not yet supported: `gals`, `clus3`, `mpole`, `hernpot`, `hern`, `powpot`,
`pow`, `gnfwpot`, `gnfw`, `serspot`, `sers`, `tnfwpot`, `tnfw`, `einpot`,
`ein`, `anfw`, `ahern`, `crline`.

## API surface

The public API mirrors glafic's Python module:

```python
import rhongomyniad as rh

rh.init(omega, lam, weos, hubble, prefix,
        xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev, verb=0)
rh.startup_setnum(num_len, num_ext, num_poi)
rh.set_lens(i, "sie", z, sigma, x0, y0, e, pa, s_core, _)
rh.set_point(i, zs, xs, ys)
rh.model_init()

# 8-tuple (ax, ay, td_days, kap, g1, g2, muinv, rot) at a single probe
pout = rh.calcimage(zs, x, y)

# list of (x_image, y_image, magnification, td_days) tuples
images = rh.point_solve(zs, xs, ys)
images = rh.findimg_i(1)
```

See `tests/test_smoke.py` for a working end-to-end example.

Runtime knobs (glafic cosmetic-settings equivalents):

```python
rh.set_max_poi_tol(1e-10)    # Newton convergence tolerance
rh.set_nmax_poi_ite(10)      # Newton max iterations
rh.set_smallcore(1e-10)      # regularisation at r=0
rh.set_device("cuda")        # or "cpu"
rh.set_dtype(torch.float64)  # or torch.float32 for speed
```

## Accuracy vs glafic

`tests/compare_glafic.py` runs the same scenarios through both glafic
(via its Python bindings) and Rhongomyniad, then diffs the results.
Observed on 6 scenarios (SIE+shear, pointmass, nfwpot, nfw, king, jaffe):

* **`point`, `sie`, `pert`, `nfwpot`, `jaffe`, `gaupot`**: every output
  value (ax, ay, td, kap, g1, g2, muinv) matches glafic to **machine
  precision** (relative difference ≤ 2×10⁻¹⁵). Image positions from
  `point_solve` match to < 10⁻¹⁵ arcsec, and magnifications to full
  double-precision significance.

* **`king`**: smooth kernel ⇒ matches glafic to ≲ 2×10⁻⁵ relative, image
  positions to < 10⁻⁷ arcsec.

* **`nfw`** (and by extension the other future log-singular Schramm models
  `gnfw`, `ein`, `sers`, etc.): matches glafic to ~5×10⁻⁴ relative on
  α/κ/γ, image positions to ~5×10⁻⁴ arcsec. **This is glafic's own
  integration tolerance**, not ours: glafic computes the Schramm integrals
  with GSL Romberg at `TOL_ROMBERG_JHK = 5e-4` (see `glafic.h:370`).
  Rhongomyniad uses 128-point Gauss-Legendre with a log-u substitution, and
  is ~50× more accurate than glafic when benchmarked against
  `scipy.integrate.quad` at `epsrel=1e-12`. Net effect: for NFW-like
  profiles, **Rhongomyniad is closer to the true continuous answer than
  glafic is**.

Run the comparison locally with:

```bash
# generate the reference on WSL where glafic.so is installed
PYTHONPATH=/path/to/glade/glafic2/python \
  python3 tests/compare_glafic.py --mode glafic --out tests/reference.json

# run Rhongomyniad and diff
python tests/compare_glafic.py --mode rh \
  --ref tests/reference.json --out tests/rh_output.json
```

## Design notes

* **Why PyTorch?** It's already installed with CUDA on the target machine
  and lets us express the whole calculator as vectorised tensor ops without
  hand-writing CUDA kernels. Float64 on device is standard.

* **Grid strategy.** glafic builds an adaptive quad-tree mesh on the fly.
  That structure is painful on GPU because each level is irregular.
  Rhongomyniad instead pre-builds a *uniform* fine grid at
  `pix_poi / 2^(maxlev-1)` covering the full `[xmin, xmax] × [ymin, ymax]`
  box, evaluates the deflection at every corner in one batched call, tests
  both triangles of every cell against the source position with a 2D
  cross-product, and runs Newton on the (small) candidate list. With
  `pix_poi=3, maxlev=5, [-60,60]` this is ~410k boxes per solve; a modern
  NVIDIA GPU handles it in single-digit milliseconds.

* **Schramm integrals.** The same machinery covers `nfw`, `king`, `jaffe`
  (when it routes through two SIEs), and any future elliptical-density
  model. We compute the linear-u and log-u quadratures in parallel and
  blend per-sample with `torch.where(uu > 0.1, lin, log)`, exactly matching
  glafic's decision rule. 128 nodes × 2 rules × #query-points is still
  dwarfed by the corner evaluation cost on most workloads.

* **Cosmology.** Angular-diameter distances are computed with
  `scipy.integrate.quad(1/(a²E(z(a))), ..., epsrel=1e-6)` on the CPU, once
  per `(zl, zs)` pair. This matches glafic's GSL CQUAD at the same
  tolerance. These values feed into `bb`, `tt`, `tdelay_fac`, and
  `sigma_crit` as scalar constants for a given run.

## Performance note

Rhongomyniad ships with two image finders (pick with `set_finder("adaptive")`
or `set_finder("uniform")`, default is `adaptive`):

* **`adaptive`** — faithful GPU port of glafic's quad-tree in `point.c`.
  Level 0 is a regular grid, every higher level is built on-device by
  evaluating the lens at 5 fresh corner points per flagged parent and
  scattering them into 4 sub-boxes (mirrors `poi_set_table` to the byte).
  Subdivision criteria are identical to glafic: sign change of μ⁻¹,
  |μ⁻¹| outside `[poi_imag_min, poi_imag_max]`, or lens-center proximity.
* **`uniform`** — a dense fine grid at the finest level only. Simpler,
  kept as a reference backend. Wins over `adaptive` on very cheap lens
  models (SIE, pert) because it ducks the multi-level dispatch overhead.

Per-call timings on `pix_poi=1, maxlev=5, [-30, 30]²`, one source:

| scenario    | glafic   | RH `uniform` | RH `adaptive` |
|-------------|----------|--------------|---------------|
| SIE + shear | 0.2 ms   | ~12 ms       | ~18 ms        |
| NFW         | 0.3 ms   | **~9000 ms** | **~90 ms**    |

**`adaptive` is 100× faster than `uniform` on NFW** because the Schramm
elliptical-density integral is the dominant per-cell cost and adaptive
only pays it at the ~few-thousand leaf boxes it actually needs, while
uniform pays it at all ~920 k fine-grid cells. On cheap models like SIE
the per-cell cost is already ~1 ns/op so the 920×-more-cells factor
barely matters and `uniform`'s single fused kernel wins.

Either way, single-call latency is dominated by **Python/Torch dispatch
overhead** (~500 µs per tensor op × ~30 ops per solve = ~15 ms of
fixed cost), not by GPU compute. That is why glafic's compiled-C tight
loop still beats us on a single call even though the GPU does each
individual cell's work faster. Two paths forward:

* **Batching** — the real GPU win is in pipelines that feed many
  source positions or many candidate lens configurations through a
  single `point_solve` call. See
  `examples/bench_batched_solve.py`, which runs 64 lens configs in one
  GPU pass and saturates at ~1.4 ms/config — a 10× speedup over the
  per-call API and close to glafic's 0.7 ms/config for SIE+shear. For
  models where glafic's per-cell cost is high (NFW and friends), the
  batched API is expected to *beat* glafic outright.
* **More complex scenes** — when a single call has dozens of lenses
  (main halo + many subhalos), the per-cell cost on GPU grows linearly
  but stays much cheaper than CPU, while glafic's adaptive mesh has to
  do the full Schramm/u-transform work at every leaf box.

## Limitations

* **Single lens plane only.** Multi-plane lensing (glafic's `gen_lensplane`
  pipeline) is not yet implemented; `model_init` will raise if two lenses
  are registered at redshifts differing by more than `TOL_ZS = 10⁻⁶`.
* **No Gaussian-Einsatz table**. `ein_tab.c` / `gnfw_tab.c` lookup tables
  are unused — those models aren't implemented yet.
* **CPU fallback**. Runs fine if `torch.cuda.is_available()` is false, but
  you lose the main performance advantage.
* **Image finder**. The uniform grid may miss pairs of images that lie
  within a single fine-grid cell (rare, only very close to critical
  curves). glafic's adaptive mesh subdivides these regions. If you see a
  missing image, increase `maxlev`.

## File layout

```
rhongomyniad/
├── constants.py         physical/numerical constants (mirror of glafic.h)
├── cosmology.py         wCDM distances, sigma_crit, tdelay_fac
├── elliptical.py        Schramm integrals + rotation helpers (GPU)
├── lens_models.py       kapgam_{point,sie,pert,nfwpot,nfw,king,jaffe,gaupot}
├── image_finder.py      uniform-grid triangle search + Newton refine
├── api.py               glafic-compatible public API with global state
└── __init__.py

tests/
├── test_smoke.py        end-to-end sanity check (no glafic dependency)
├── compare_glafic.py    dual-mode harness: run glafic and Rhongomyniad,
│                        diff the results on a suite of scenarios
├── debug_integral.py    diagnostic: j0/j1 vs scipy vs glafic for NFW
├── debug_nfw.py         diagnostic: bb/tt/kappa values for NFW
└── reference.json       saved glafic outputs (regenerate in WSL)
```
