# GLADE: Gravitational Lensing Analysis and Differential Evolution

[[License: MIT](https://opensource.org/licenses/MIT)]
[[Python 3.8+](https://www.python.org/downloads/)]
[[glafic2](https://github.com/oguri/glafic2)]

GLADE is a gravitational-lensing analysis workbench for strong-lens modelling,
substructure searches, Differential Evolution, and MCMC. It wraps a modified
`glafic2` CPU/reference engine, a unified backend-agnostic `core/` optimizer,
and the `Rhongomyniad/` GPU lens engine behind a browser UI and
command-line tools.

> **V0.7.0-Rave current release — micro-image blind-spot auto-check +
> BIPOP-CMA-ES/jSO optimizers + unit system**: `auto_check` (default on) adds a
> verify-layer micro-image audit (`micro_audit`: per-matched-image local
> fine-grid `findimg`, Σ|μ|, physical-loss and fake-solution flags) plus a
> trigger-based Σ|μ| check inside the optimization loop itself (CPU via glafic
> rescale cycles, GPU via batched local-seed Newton) — `auto_check=False`
> reproduces prior-version behaviour bit-for-bit. Two new point-source
> optimizers, **BIPOP-CMA-ES** and **jSO**, join Differential Evolution on both
> CPU (glafic multiprocess) and GPU (Rhongomyniad batched) paths via `.dat`
> `OPTIMIZER`/`CMAES_*`/`JSO_*` keys or `glade.optimize(algorithm=...)`. Every
> run now writes `glade_output_<run>.dat`, a ready-to-reuse GLADE input file
> with optimizable parameters fixed at the best fit. The WebUI FindImage panel
> gains a Backend settings icon (Backend × purpose × algorithm, memorised) in
> place of the five backend buttons, the Editor's Algorithm-parameters
> templates split into CPU-glafic and GPU-rhongomyniad groups, and a new
> unit-profile system (`InputFiles/<name>.units.json`, `.dat` key
> `UnitSetting`) adds real mass/position unit conversions (h⁻¹M⊙↔M⊙,
> mas↔arcsec) and fixes a doc bug (glafic mass is really `h^-1 Msun`, not
> `Msun`).
>
> See [Update.txt](Update.txt) for full changelogs.

Full changelogs: [Chinese](Update.txt) / [English](update_en.txt).

User manual (V0.6.0-GREY, comprehensive & fact-checked):
[English](manual/GLADE_Manual_en.md) / [中文](manual/GLADE_Manual_zh.md).

## Features

- **Unified optimizer core**: one Differential Evolution pipeline drives CPU
  glafic, direct glafic, or GPU Rhongomyniad backends.
- **Flexible `.dat` format**: inline, glafic-like configuration files use
  `{lower, upper}` bounds for optimizable dimensions and locked bare values for
  fixed parameters.
- **Composable lens stacks**: main lenses and substructures share one component
  list, so point mass, NFW, King, pseudo-Jaffe, Sersic, SIE, shear, and other
  glafic models can be mixed where the chosen backend supports them.
- **GPU acceleration**: Rhongomyniad provides a PyTorch lens calculator covering
  **all 27 glafic lens models** (V0.6) and a batched GPU path that evaluates the
  whole DE/MCMC population in single CUDA calls.
- **Library use**: `import glade` from any script — the facade package
  bootstraps `sys.path` and exposes `load_config` / `optimize` / `build_obs` /
  `make_triptych` / `run_mcmc` / `verify_with_glafic` with lazy heavy imports.
- **Clave lens calculator**: the interactive drag-and-compute lens visualizer is
  built in as the third WebUI tab (mounted at `/clave`, CPU/GPU backends).
- **MCMC workflows**: run MCMC only, or run DE first and seed emcee walkers from
  the DE best fit; corner, trace, summary, and result figures are written to the
  run directory and surfaced in the WebUI.
- **Extended-source fitting**: a `.dat` can reference an observed FITS image and
  external files (`extended_file`, `constraint_file`, `prior_file`) to fit
  extended (Sersic etc.) sources on the CPU via glafic's per-component `c2calc`,
  with weightable chi2 terms, an optimizable Hubble constant, and MCMC support.
- **Independent verification**: `glafic_verified = True` reruns the best model
  through the glafic binary, while the scipy-exact check reports Sersic
  deflection accuracy and source-plane consistency.
- **Browser workbench**: the current `webui/` app provides FindImage runs,
  real-time terminal streaming, result previews, an Explorer with right-click
  copy/paste, Monaco editor, template insertion for `InputFiles/`, a dark/light
  theme toggle (default dark) and an English/中文 language toggle (default
  English), both in the top-right corner.
- **Translation tools**: convert between glafic input/obs files and GLADE `.dat`
  files from the WebUI or `python -m core.translate.cli`.

## Quick Start

### System Requirements

- Linux (GCC + standard build tools), or macOS (Xcode Command Line Tools +
  Homebrew) — macOS support is **untested** (never compiled or run on a real
  Mac so far)
- Python 3.8 or higher
- CFITSIO, FFTW3, and GSL for building glafic2 (apt on Linux, Homebrew on macOS)
- Optional CUDA-capable PyTorch environment for GPU runs (NVIDIA only; Macs run
  the optimizer on CPU)

### Installation

```bash
git clone https://github.com/y31ling/glaDE.git
cd glaDE

# First-time setup: build dependencies, glafic2, Python bindings, and launchers.
bash bootstrap_linux.sh      # Linux

# macOS (Apple Silicon or Intel) — requires Homebrew + Xcode Command Line Tools.
# NOTE: untested on a real Mac; the script self-checks `import glafic`.
bash bootstrap_macos.sh      # macOS
```

The bootstrap script creates `env.sh`, `run_glade.sh`, and `run_webui.sh`, and
installs the Python dependencies listed in `requirements.txt`. The macOS
installer uses Homebrew for CFITSIO/FFTW/GSL (no source build) and selects the
`spawn` multiprocessing start method (Linux keeps `fork`).

### Launch the WebUI

```bash
./run_webui.sh

# Or choose another port; default is 6017.
GLADE_PORT=8080 ./run_webui.sh
```

Open <http://localhost:6017>. The main WebUI is `webui/`; the older `web/`
directory is kept for historical compatibility.

### Run from the Command Line

For legacy-script compatibility, `main.py` still dispatches through `runner.py`
to the archived `legacy/version_*.py` workflows:

```bash
./run_glade.sh
```

For the current v0.4 core, use `.dat` files directly:

```bash
source env.sh
python webui/runjob.py \
  --backend cpu \
  --mode findimage \
  --out runs/manual_cpu \
  --files core/examples/constants.dat \
          core/examples/images_data.dat \
          core/examples/lens_and_substructure.dat \
  --force
```

Backends are `cpu`, `gpu`, and `glafic`. Modes are `findimage`, `de+mcmc`, and
`mcmc`.

### Use as a Python Library (`import glade`)

Since V0.6 GLADE is importable like glafic itself. With the repo root on
`sys.path` (automatic when your working directory is the repo, or from anywhere
after `source env.sh`):

```python
import glade

cfg, issues = glade.load_config(
    ["constants.dat", "images_data.dat", "lens.dat"], backend="gpu")
assert not glade.has_errors(issues)

result = glade.optimize(cfg, backend="gpu")          # DE fit
glade.make_triptych(result, glade.build_obs(cfg), "triptych.png")
```

Importing `glade` bootstraps the paths for the whole tree, so `import core`,
`import glafic`, and `import rhongomyniad` all work afterwards. Heavy modules
(matplotlib, emcee, torch, the glafic C extension) load lazily on first use.
`glade.engine("cpu")` / `glade.engine("gpu")` return the low-level engine
modules with the glafic-style imperative API (`init` / `set_lens` /
`point_solve` / ...).

## `.dat` Configuration

The v0.4 `.dat` format is the canonical format used by the Editor and optimizer.
It is intentionally close to glafic/legacy inline configuration, with two key
additions:

```python
source_x = {-0.10, 0.10}   # optimize within bounds
source_y = 0.0244          # lock this value

'sers1':  (1, 'sers', lens_z, 9.896617e+09, 2.656977e-03, 2.758473e-02,
           2.986760e-01, 1.124730e+02, 3.939718e-01, 1.057760e+00)
'point1': (2, 'point', lens_z, {1e5, 1e7}, {-0.30, -0.20}, {-0.05, 0.05})
```

- `{lower, upper}` marks an optimizable parameter.
- Bare numbers are locked.
- Mass-like parameters are searched in `log10` space.
- Multiple selected files are merged by section, and component indices are
  recomputed globally.
- Missing basic constants fall back to defaults; observation arrays and at least
  one component are required.

See [core/SPEC.md](core/SPEC.md) for the full specification.

## WebUI Workflow

1. Put or create `.dat` files under `InputFiles/` from the Editor page
   (the file tree supports right-click Copy/Paste of files and whole folders).
2. Use the Template panel to insert constants, observation data, lenses,
   substructures, and MCMC settings.
3. Select one or more `.dat` files in FindImage.
4. Choose `CPU`, `GPU`, `Glafic`, or `MCMC`.
5. Run and watch the spawned terminal stream back into the browser.
6. Inspect `result.png`, `mcmc_corner.png`, `mcmc_trace.png`, `best_params.txt`,
   `mcmc_summary.txt`, `status.json`, and verification outputs in `runs/<job>/`.

The third tab, **Clave**, embeds the interactive lens calculator: drag lenses
and sources on a gridded canvas and watch the lensed images update in real time
(CPU via glafic, GPU via Rhongomyniad). The top-right buttons switch the UI
language (English/中文, default English) and the theme (dark/light, default
dark).

## Project Structure

```text
glade/
├── main.py                 # legacy-compatible CLI entry point
├── runner.py               # legacy model dispatcher
├── bootstrap_linux.sh      # one-click Linux setup
├── run_webui.sh            # generated WebUI launcher
├── InputFiles/             # editable .dat inputs for the WebUI
├── core/                   # v0.4 optimizer, parser, translator, plots, verify
│   ├── format/             # .dat parser, defaults, schema, validation
│   ├── optimize/           # backend-agnostic DE and CPU/GPU/glafic backends
│   ├── mcmc/               # emcee posterior sampling and plotting
│   ├── plot/               # triptych, critical curves, iteration frames
│   └── translate/          # glafic <-> glade conversion
├── glade/                  # importable facade package (import glade)
├── webui/                  # current Flask + Monaco browser workbench
├── clave/                  # Clave lens calculator (Flask blueprint at /clave)
├── glafic2/                # modified glafic v2 source and Python bindings
├── Rhongomyniad/           # GPU lens engine (PyTorch/CUDA, glafic-matching)
├── legacy/                 # archived pre-v0.4 model scripts
└── tools/                  # verification, plotting, and post-processing helpers
```

## Supported Models

All glafic lens models are registered in `core/format/schema.py`, and since
V0.6 the GPU backend (Rhongomyniad) implements **every one of glafic's 27 lens
models** — including the V0.6 additions `crline` (straight critical line),
`acnfw` (CSE-approximated cored NFW), and `gals` (external galaxy catalogue,
summed pseudo-Jaffe) — each verified against the glafic binary to machine
precision. Multi-plane lensing remains CPU/glafic-only.

`gals` reads its catalogue like glafic does: a `galfile.dat` (rows
`x y L [e pa]`) in the working directory, loaded lazily on first use; or inject
it explicitly with `rhongomyniad.set_galfile(path)` / `readgals()` /
`set_gals(rows)`.

## Translation

```bash
# glafic input -> GLADE .dat files
python -m core.translate.cli to-glade some_model.input -o InputFiles/imported

# GLADE .dat files -> glafic model/obs files
python -m core.translate.cli to-glafic \
  InputFiles/constants.dat InputFiles/lens.dat InputFiles/images_data.dat \
  -o exported/run
```

The translator converts glafic optimization-matrix flags into `{value, value}`
bounds for manual widening and uses representative midpoints when exporting
bounded GLADE values back to glafic.

## Outputs

Current v0.4 WebUI/core runs write into `runs/<job_id>/`:

- `job.log`: full terminal output streamed to the WebUI.
- `status.json`: machine-readable state, figures, fitted parameters, and
  verification reports.
- `best_params.txt`: DE best-fit physical parameters.
- `result.png`: triptych result figure.
- `mcmc_corner.png`, `mcmc_trace.png`, `mcmc_summary.txt`: MCMC outputs when
  MCMC is enabled or selected.
- `glafic_verify.input` and glafic point outputs: independent verification
  artifacts when `glafic_verified` is enabled.

Legacy `main.py` runs continue to write model-specific outputs under `results/`.

## Tests

```bash
source env.sh
for t in format optimize translate plot; do .venv/bin/python core/tests/test_$t.py; done
```

The core tests use lightweight fake backends where possible. Live CPU/GPU runs
also require built glafic bindings and, for GPU, a working PyTorch/CUDA setup.

## History

| version | comments |
|:---|:---|
| 0.7.0-Rave | **Micro-image blind-spot auto-check + BIPOP-CMA-ES/jSO optimizers + unit system**: a verify-layer `micro_audit` (local fine-grid `findimg`, Σ\|μ\|, physical-loss, fake-solution flags) plus an in-loop trigger-based Σ\|μ\| check catch missed/fake images by default (`auto_check=False` restores bit-identical prior behaviour); BIPOP-CMA-ES and jSO join Differential Evolution as selectable point-source optimizers on CPU and GPU; every run writes a ready-to-reuse `glade_output_<run>.dat`; WebUI FindImage gets a Backend settings icon and the Editor a UnitSetting dialog backed by a real mass/position unit-conversion system (also fixes a long-standing doc bug: glafic mass is `h^-1 Msun`, not `Msun`). |
| 0.6.0-GREY | **Bilingual user manuals**: two comprehensive manuals — [manual/GLADE_Manual_en.md](manual/GLADE_Manual_en.md) (English) and [manual/GLADE_Manual_zh.md](manual/GLADE_Manual_zh.md) (中文) — covering installation (incl. WSL2), the three WebUI tabs, the full `.dat` reference (27-model parameter tables, every key + default), DE/MCMC/extended-source fitting, run outputs, CLI & `import glade`, a dedicated `TOL_ROMBERG_JHK` accuracy chapter, troubleshooting and appendices; written for physics undergraduates and adversarially fact-checked against the code (20 corrections applied). README gains the GREY release image; `update_en.txt` back-filled with the missing V0.6.0 entry. |
| 0.6.0 | **Clave integration + full lens-model coverage + library packaging + WebUI theme/i18n**: Rhongomyniad gained the last three glafic lens models (`crline`, `acnfw`, `gals`) — all 27 now run on GPU, verified against the glafic binary to machine precision; the Clave interactive lens calculator was merged into the repo as the third WebUI tab (`/clave` blueprint); `import glade` works as a library facade with lazy heavy imports; the WebUI gained dark/light themes (default dark), an English/中文 toggle (default English), Explorer right-click Copy/Paste for files and folders, and dropped the FindImage rail note. |
| 0.5.1 | Repository cleanup: internal dev notes (glafic upstream-issue drafts, the GPU-MCMC exploration report), exploration-phase dev scripts and the deprecated pre-V0.4 `web/` UI were untracked and gitignored (files stay local); `source/` assets are kept for the future WebUI. |
| 0.5.0 | **Rhongomyniad major update** (consolidates the unreleased 0.4.6–0.4.8): full GPU/CPU behaviour parity (24 tensor-parameterised kernels, extended-source FITS pipeline on GPU), whole-population batched DE/MCMC for any optimizable model (~27× on free main-lens fits), `gpu_precision` 64/48/32 mixed precision, user-defined shared `.dat` variables, parity-insensitive `abs_mag` magnification convention, `MCMC-GPU` rail with walker auto-tuning, `Nl`/`Ns` classification suffixes, scipy-exact + glafic cross-verification tools, and extensive adversarial-review hardening. |
| 0.4.5 | macOS deployment support was added: a `bootstrap_macos.sh` one-click installer (Homebrew deps, macOS glafic build) and a platform-aware multiprocessing layer (`core/parallel.py`) that keeps Linux on `fork` and uses `spawn` on macOS so multi-process DE/MCMC is safe and result-identical. **Untested on a real Mac**; Linux behavior is unchanged (core tests 81/81). |
| 0.4.4 | Extended-source (FITS) CPU fitting was added: a `.dat` can reference a FITS image and external file addresses and is optimized via glafic's new per-component `c2calc` with weightable terms, an optimizable Hubble constant, MCMC support, and a graded missing-image penalty. |
| 0.4.3 | The bundled glafic was synced to upstream v2.1.14, preserving GLADE's King model (renumbered to #27 to avoid the new `acnfw`), the tolerance override, and the vendored build. |
| 0.4.2 | GPU result plots now reuse the optimizer's batched solver images and add informational glafic plus scipy-exact verification. |
| 0.4.1 | Batched GPU DE/MCMC and the new MCMC modes were added, with MCMC priors unified to the DE bounds. |
| 0.4.0 | The project was reunified around `core/`, the new `.dat` format, the rewritten `webui/`, and glafic <-> GLADE translation. |
| 0.3.0 | Rhongomyniad introduced experimental GPU acceleration, glafic verification tools, Clave visualization, and tighter glafic constants. |
| 0.2.2 | The changelog, glafic input export, improved observation editor, resizable sidebar, consistent None-model output, and WebUI concurrency fixes were added. |
| 0.2.0 | The first bilingual WebUI added browser-based parameter editing and model execution. |
| 0.1.0 | The prototype combined glafic with Differential Evolution. |

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE). glafic2
itself is GPL-licensed upstream software by Masamune Oguri; cite the upstream
papers listed in [glafic2/README.md](glafic2/README.md) when using glafic or
modified glafic for research.

## Acknowledgments

- **glafic2**: gravitational lensing engine by Masamune Oguri.
- **Rhongomyniad**: local GPU lens-calculation component used by GLADE.
- **SciPy, NumPy, Astropy, emcee, corner, matplotlib, Flask, Monaco**: the
  scientific and UI ecosystem supporting the current workflow.

---

![GREY](source/GREY.jpg)
