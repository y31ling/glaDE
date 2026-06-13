# GLADE: Gravitational Lensing Analysis and Differential Evolution

[[License: MIT](https://opensource.org/licenses/MIT)]
[[Python 3.8+](https://www.python.org/downloads/)]
[[glafic2](https://github.com/oguri/glafic2)]

GLADE is a gravitational-lensing analysis workbench for strong-lens modelling,
substructure searches, Differential Evolution, and MCMC. It wraps a modified
`glafic2` CPU/reference engine, a unified backend-agnostic `core/` optimizer,
and the `Rhongomyniad/` GPU lens engine behind a browser UI and
command-line tools.

> **V0.5.0 current release — Rhongomyniad major update**: the PyTorch/CUDA GPU
> engine leaves beta and takes its first real version name. The GPU backend now
> matches the CPU's full behaviour: every deflector model except the file-based
> `gals` (24 tensor-parameterised kernels) plus all 5 extended-source (FITS)
> profiles run on the GPU; **any** optimizable lens model is evaluated
> whole-DE-population batched (a free 20-dim main-lens fit drops from ~21 s to
> ~0.86 s per generation, ~27×); a new `gpu_precision = 64/48/32` key adds a
> mixed fp32/fp64 mode (48: fp32 speed at fp64 accuracy, recommended for
> Schramm-heavy models); MCMC gains a batched-CUDA `MCMC-GPU` rail with
> auto-tuned walkers. The `.dat` format adds user-defined shared variables
> (`lens_x = {-0.1, 0.1}` ties one fitted value across components), an
> `abs_mag` parity-insensitive magnification chi2/plot convention (default),
> and `Nl`/`Ns` component classification suffixes. Verified throughout against
> scipy-exact references and the glafic binary; pure point-mass configs keep
> bit-identical same-seed DE trajectories. See [Update.txt](Update.txt).
>
> **V0.4.5**: GLADE adds **macOS deployment support** (Apple
> Silicon and Intel) via a new `bootstrap_macos.sh` one-click installer and a
> platform-aware multiprocessing layer (`core/parallel.py`) that keeps Linux on
> `fork` while macOS uses `spawn`, so multi-process DE/MCMC runs safely on a Mac
> with results identical to Linux. GPU stays optional (CPU fallback on Macs).
>
> ⚠️ **The macOS support is UNTESTED**: it has not yet been compiled or run on a
> real Mac — it was only verified on Linux to not regress existing behavior
> (core tests 81/81, Linux multiprocessing still `fork`). On a Mac, trust the
> `import glafic` self-check at the end of `bootstrap_macos.sh`, and please
> report any build/link issues. See [Update.txt](Update.txt) for details.
>
> **V0.4.4**: GLADE fits extended sources (FITS images). A `.dat` can reference
> an observed FITS image plus external file addresses (`extended_file`,
> `constraint_file`, `prior_file`, …) and is optimized on the CPU via glafic's
> per-component `c2calc`, with weightable chi2 terms
> (`W_POS`/`W_FLUX`/`W_TD`/`W_EXT`/`W_PRIOR`), an optionally optimizable Hubble
> constant, MCMC support, and a graded `missing_img_penalty` for under-imaged
> candidates.

Full changelogs: [Chinese](Update.txt) / [English](update_en.txt).

## Features

- **Unified optimizer core**: one Differential Evolution pipeline drives CPU
  glafic, direct glafic, or GPU Rhongomyniad backends.
- **Flexible `.dat` format**: inline, glafic-like configuration files use
  `{lower, upper}` bounds for optimizable dimensions and locked bare values for
  fixed parameters.
- **Composable lens stacks**: main lenses and substructures share one component
  list, so point mass, NFW, King, pseudo-Jaffe, Sersic, SIE, shear, and other
  glafic models can be mixed where the chosen backend supports them.
- **GPU acceleration**: Rhongomyniad provides a PyTorch lens calculator and a
  batched GPU path for compatible point-mass optimization and MCMC workloads.
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
  real-time terminal streaming, result previews, an Explorer, Monaco editor, and
  template insertion for `InputFiles/`.
- **Translation tools**: convert between glafic input/obs files and GLADE `.dat`
  files from the WebUI or `python -m core.translate.cli`.

## Quick Start

### System Requirements

- Linux (GCC + standard build tools), or macOS (Xcode Command Line Tools +
  Homebrew) — see the macOS note below; macOS support is **untested** in v0.4.5
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
# NOTE: untested on a real Mac in v0.4.5; the script self-checks `import glafic`.
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

1. Put or create `.dat` files under `InputFiles/` from the Editor page.
2. Use the Template panel to insert constants, observation data, lenses,
   substructures, and MCMC settings.
3. Select one or more `.dat` files in FindImage.
4. Choose `CPU`, `GPU`, `Glafic`, or `MCMC`.
5. Run and watch the spawned terminal stream back into the browser.
6. Inspect `result.png`, `mcmc_corner.png`, `mcmc_trace.png`, `best_params.txt`,
   `mcmc_summary.txt`, `status.json`, and verification outputs in `runs/<job>/`.

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
├── webui/                  # current Flask + Monaco browser workbench
├── glafic2/                # modified glafic v2 source and Python bindings
├── Rhongomyniad/           # GPU lens engine (PyTorch/CUDA, glafic-matching)
├── legacy/                 # archived pre-v0.4 model scripts
└── tools/                  # verification, plotting, and post-processing helpers
```

## Supported Models

CPU/glafic support follows the known glafic models registered in
`core/format/schema.py`. GPU support currently covers:

| Model | Typical Role | GPU |
|:---|:---|:---:|
| `point` | point-mass substructure | yes |
| `nfw`, `nfwpot` | NFW substructure | yes |
| `king` | King-profile substructure | yes |
| `jaffe` | pseudo-Jaffe substructure | yes |
| `sers` | Sersic lens component | yes |
| `sie` | SIE lens component | yes |
| `pert` | external shear/convergence | yes |
| `gaupot` | Gaussian potential | yes |

Other registered glafic models can still be used on CPU/direct glafic when their
parameters are provided in the `.dat` component syntax.

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
