# GLADE `.dat` configuration format (V0.4)

This is the canonical configuration format authored in the Editor and consumed
directly by the optimizers (CPU / GPU) and the glafic backend. It deliberately
mirrors the inline Python config that the legacy `version_*.py` scripts already
use, with two small additions:

* `{lower, upper}` marks a parameter as **optimizable** (a search dimension).
* `$float` / `$int` / `$float{lower,upper}` are **fill-in placeholders** that a
  template inserts; the user must replace them. A placeholder left in a file
  that is actually run is a hard error.

The format is **not** executed as Python. It is read by a restricted parser
(`core.format.parser`) that understands only literals, `{lo,hi}` bounds, lists,
component tuples, and name references — never arbitrary expressions or calls.

---

## 1. Lines and comments

* `#` (outside a string) starts a comment: everything from `#` to the end of
  that *physical* line is dropped. This applies even inside a multi-line
  bracketed value, where bracket-matching simply continues on the next line — so
  a trailing comment inside a multi-line list is fine; just never comment out a
  bracket itself.
* Blank lines are ignored.
* A statement may span multiple physical lines as long as a bracket is open
  (`(`, `[`, `{`) — exactly like the legacy `lens_params = { ... }` blocks.

## 2. Scalar assignments

```
name = value
```

`value` is one of:

| form                         | meaning                                             |
|------------------------------|-----------------------------------------------------|
| `0.3`, `-1`, `2.65e-3`, `1E6`| a fixed number (locked)                             |
| `{lo, hi}`                   | an **optimizable** scalar with bounds `[lo, hi]`    |
| `[ ... ]` / `[[...],...]`    | a Python list literal (observation arrays)          |
| `True` / `False`            | a boolean flag                                       |
| `'text'` / `"text"`         | a string (e.g. an output prefix)                    |
| `name`                       | a reference to an earlier-defined numeric scalar    |
| `$float` / `$int`            | unfilled placeholder (error if run)                 |
| `$float{lower,upper}`        | unfilled placeholder for an optimizable value       |

Multiple assignments to the **same name within one file** is an error.

### Known scalar groups

* **Cosmology**: `omega`, `lambda_cosmo` (alias `lambda`), `weos`, `hubble`
* **Grid**: `xmin`, `ymin`, `xmax`, `ymax`, `pix_ext`, `pix_poi`, `maxlev`
* **Redshifts**: `source_z`, `lens_z` — also usable as references in tuples
* **Source**: `source_x`, `source_y` (each lockable or `{lo,hi}`)
* **Observation** (the four arrays are *hard-required* for a run):
  `obs_positions_mas_list`, `obs_magnifications_list`, `obs_mag_errors_list`,
  `obs_pos_sigma_mas_list`, plus `center_offset_x`, `center_offset_y`,
  `obs_x_flip`
* **Algorithm**: `DE_MAXITER`, `DE_POPSIZE`, `DE_ATOL`, `DE_TOL`, `DE_SEED`,
  `DE_POLISH`, `DE_WORKERS`, `EARLY_STOPPING`, `EARLY_STOP_PATIENCE`,
  `LOSS_COEF_A`, `LOSS_COEF_B`, `LOSS_PENALTY_PL`, `CONSTRAINT_SIGMA`,
  `Draw_Graph`, `draw_interval`, `COMPARE_GRAPH`, `SHOW_2SIGMA`,
  `OUTPUT_PREFIX`, `MCMC_*`, `gpu_precision`, `abs_mag`
* `abs_mag` (default `True`) compares magnifications by ABSOLUTE value in the
  point-source loss (parity-insensitive: obs `30` vs model `-29` differs by 1,
  not 59 — near critical curves the model parity flips easily and a signed
  comparison punishes an otherwise good match), and the result triptych's
  magnification panel shows `|μ|` (all bars upward from 0). `False` restores
  the signed, parity-sensitive comparison and the signed panel. The two are
  identical whenever the matched signs agree, so well-fit results are
  unaffected. Applies to the point-source path on every backend (DE, MCMC and
  the glafic verification recompute); the extended-source flux chi2 keeps
  glafic's own `chi2_usemag`/parity semantics.
* `gpu_precision` (default `64`) selects the compute precision of the batched
  GPU paths (DE and MCMC): `64` = fp64 everywhere; `48` = mixed — the
  deflection-field / triangle-test phase runs fp32 while the Newton refine and
  the magnifications keep fp64; `32` = fp32 everywhere. `48`/`32` mainly speed
  up Schramm-quadrature models (`sers`/`nfw`/`king`/...) on consumer GPUs
  (fp64 there runs at 1/64 rate). Other values are a validation error; the
  key has no effect on the CPU/glafic backends, the per-candidate GPU
  fallback, or the extended-source (FITS) batched path (fp64 for now).

### User-defined variables

Any assignment whose name is not a known scalar key defines a **variable**,
referencable from component tuples:

* a **fixed** variable (`my_re = 0.39`) simply substitutes its value at every
  reference (this also works for the known fixed scalars, e.g. `lens_z`);
* an **optimizable** variable (`lens_x = {-0.1, 0.1}`) becomes ONE shared
  search dimension: every parameter referencing it is fitted to the SAME
  value (the dimension appears once, under the variable's name, in the
  corner plots / fit output). Use it to tie parameters across components,
  e.g. a common centre:

  ```
  lens_x = {-0.1, 0.1}
  'sers1': (1, 'sers', lens_z, {1e9,1e12}, lens_x, {-0.1,0.1},  ...)
  'sers2': (2, 'sers', lens_z, {1e9,1e12}, lens_x, {-0.05,0.05}, ...)
  ```

  For independent searches just write `{lo, hi}` inline instead. A variable
  used in a mass-like slot is searched in log10 like any mass; using one
  variable in both mass-like and linear slots is a validation error
  (`var_mixed_usage`); a defined-but-never-referenced variable gets a
  `var_unused` warning. Optimizable *schema* scalars (e.g. `source_x =
  {lo, hi}`) cannot be referenced — they already own their own dimension.
  Variables work across multiple selected files (define in one, reference in
  another) and on every backend; the batched GPU paths support them directly,
  except a variable on a component *redshift* or a `zs_fid` slot, which falls
  back to the (correct, slower) per-candidate evaluation. The glade→glafic
  export resolves variables to concrete values (the tie is not representable
  in a glafic input file).

## 3. Component tuples (lens + sub-structure share one stack)

A lens or sub-structure component is written as a dict-entry line:

```
'name': (N, 'type', z, p1, p2, ... , pk)
```

* `name` — a quoted identifier, e.g. `'sers1'`, `'point1'`, `'king1'`.
* `N` — an integer index. **It is recomputed globally** (1-based, in selection
  order across all appended files); the literal value is only a hint. An
  optional one-letter suffix overrides how the component is **classified**
  (never how it lenses): `3l` = treat as a main *lens*, `3s` = treat as a
  *sub-structure* (drawn as a sub-halo marker in the result triptych). A plain
  number keeps the default classification: the model's schema category, or
  sub-structure when any parameter is optimizable.
* `'type'` — a model keyword (see `core/format/schema.py`).
* `z` — the component redshift: a literal float **or** a reference such as
  `lens_z`. References resolve to the nearest earlier numeric assignment.
* `p1..pk` — model parameters in glafic order. Each is a fixed number, a
  `{lo,hi}` optimizable bound, a name reference, or a placeholder. Trailing
  unused glafic slots may be omitted; they default to `0.0` when emitted.

Main-lens models (`sers`, `sie`, …) and sub-structure models (`point`, `nfw`,
`king`, `jaffe`, …) use the **same** tuple syntax and live in **one combined
stack**. They are mechanically identical (both become glafic `set_lens` calls);
the Editor's *Lens* vs *Sub-structure* menus are only authoring categories.
Sub-structures are freely **composable and mixed-type** (e.g. one `point` + one
`king` in the same run).

When a model's default category does not match its role in a particular fit,
recategorize it with the index suffix. For example an `anfw` halo (schema
category: sub-structure) used as the *galaxy-scale* main lens, locked at its
best-fit values, should not appear as a sub-halo marker:

```python
'anfw1': (3l, 'anfw', lens_z, 3.606e+11, -2.89e-03, 2.71e-02, 0.464, 26.56, 29.36)
```

The reverse (`3s`) forces any model to be displayed as a sub-structure.

## 4. Optimization semantics

* A parameter written as a bare number is **locked**.
* A parameter written as `{lo, hi}` is a **search dimension** with those bounds.
* **Mass-like** parameters (point `mass`; sie/jaffe `sigma`; nfw/king/… `mass`;
  pert `shear`/`kappa`; flagged `is_mass` in the schema) are searched in
  **log10** space: the bounds `{lo, hi}` are the actual physical values and are
  converted to `log10` internally. Their representative midpoint (used by the
  glade→glafic translator) is the **geometric mean**, so `{1E5, 1E7}` → `1E6`.
* All other parameters are linear; their midpoint is the arithmetic mean.

## 5. Multiple files (FindImage)

Selecting several `.dat` files merges them by section:

* Each scalar may be defined in **at most one** file across the selection; a
  conflicting redefinition is an error that names the variable and both files.
* Components from all files **concatenate** in selection order, and `N` is
  recomputed globally 1-based to match the engine's component ordering.
* Missing basics fall back to defaults (see `core/format/defaults.py`, taken
  verbatim from the legacy point-mass script). The four observation arrays and
  at least one component are **hard-required** — no defaults; the run is blocked
  with a clear message if they are absent.

## 6. Backends

* **CPU** (glafic) and **Glafic-direct** support all models.
* **GPU** (Rhongomyniad, V0.5.0) supports every deflector model except the
  file-based `gals` catalogue, all five extended-source models and the full
  extended-source (FITS) chi2 pipeline — still single lens plane. Selecting
  GPU with an unsupported model (`gals`, or multi-plane configs) is blocked,
  naming the offending component.

## 7. Example

```python
# ---- constants ----
omega = 0.3
lambda_cosmo = 0.7
weos = -1.0
hubble = 0.7
xmin, ymin = -0.5, -0.5          # (also accepted as two separate assignments)
xmax, ymax = 0.5, 0.5
pix_ext = 0.01
pix_poi = 0.2
maxlev = 5
source_z = 0.4090
lens_z   = 0.2160

# ---- source (point) ----
source_x = {-0.10, 0.10}         # optimize the source x position
source_y = 0.0244                # lock the source y position

# ---- observation data ----
obs_positions_mas_list = [[-266.035, 0.427], [118.835, -221.927],
                          [238.324, 227.27], [-126.157, 319.719]]
obs_magnifications_list = [-35.6, 15.7, -7.5, 9.1]
obs_mag_errors_list     = [2.1, 1.3, 1.0, 1.1]
obs_pos_sigma_mas_list  = [0.41, 0.86, 2.23, 3.11]
center_offset_x = 0.01535
center_offset_y = 0.0322
obs_x_flip = True

# ---- main lens (locked baseline) ----
'sers1': (1, 'sers', lens_z, 9.896617e+09, 2.656977e-03, 2.758473e-02,
          2.986760e-01, 1.124730e+02, 3.939718e-01, 1.057760e+00)
'sie1':  (2, 'sie',  lens_z, 1.183382e+02, 2.656977e-03, 2.758473e-02,
          1.571203e-01, 2.920348e+01)

# ---- sub-structure (mixed types, optimizable) ----
'point1': (3, 'point', lens_z, {1e5, 1e7}, {-0.30, -0.20}, {-0.05, 0.05})
'king1':  (4, 'king',  lens_z, {1e6, 1e9}, {0.10, 0.16}, {-0.24, -0.18},
           0.0, 0.0, {0.001, 0.05}, {0.8, 2.2})

# ---- algorithm ----
DE_MAXITER = 650
DE_POPSIZE = 64
EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 30
```
