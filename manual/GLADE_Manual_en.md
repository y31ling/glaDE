# GLADE User Manual

**Version V0.6.0 · for the local Linux/WSL2 installation · 2026-07**

GLADE (*Gravitational Lensing Analysis and Differential Evolution*) is a workbench for
strong-gravitational-lens modelling. It wraps a modified build of Masamune Oguri's
**glafic 2** as its CPU reference engine, adds a GPU lens engine (**Rhongomyniad**,
PyTorch/CUDA), and drives both through one unified optimizer (**Differential Evolution
+ MCMC**) that you control either from a browser (**WebUI**) or from Python / the
command line.

This manual is written for a physics undergraduate who is comfortable with basic
programming but is **not** a lens-modelling specialist. It assumes you know what
gravitational lensing is; every more specialised term (MCMC, corner plot, critical
curve, χ², parity, …) is explained where it first matters and in the Glossary
(Appendix A).

> **Scope note.** This manual documents the **local V0.6.0 tree running under WSL2**
> (Windows Subsystem for Linux). A few in-repo documents (the Rhongomyniad README,
> `core/SPEC.md` §6) still describe V0.5.0 behaviour; where they disagree with this
> manual, this manual reflects the actual V0.6.0 code.

---

## Table of contents

1. [Introduction](#1-introduction)
2. [Supported systems & requirements](#2-supported-systems--requirements)
3. [Installation](#3-installation)
4. [Quick start: your first fit in 15 minutes](#4-quick-start-your-first-fit-in-15-minutes)
5. [The WebUI](#5-the-webui)
   — 5.2 [FindImage](#52-the-findimage-tab) · 5.3 [Editor](#53-the-editor-tab) · 5.4 [Clave](#54-the-clave-tab)
6. [The `.dat` configuration format](#6-the-dat-configuration-format)
7. [Fitting engines & algorithms](#7-fitting-engines--algorithms)
   — DE · loss function · backends & GPU batching · MCMC · extended sources · verification
8. [Understanding run outputs](#8-understanding-run-outputs)
9. [Command line & Python library](#9-command-line--python-library)
10. [Numerical accuracy: the `TOL_ROMBERG_JHK` note](#10-numerical-accuracy-the-tol_romberg_jhk-note)
11. [Troubleshooting & FAQ](#11-troubleshooting--faq)
12. [Appendices](#12-appendices)
    — A Glossary · B Version history · C glafic, license & citations · D Environment variables · E Auxiliary file formats

---

# 1 Introduction

## 1.1 What GLADE does

A strong-lens model is a parameterised mass distribution (one or more "lens
components") plus a source. Given observed image positions and magnifications, lens
modelling means finding the parameter values that reproduce the observations. GLADE
automates this:

1. You describe the model and the observations in one or more plain-text **`.dat`
   files** (Chapter 6). Any parameter written as `{lower, upper}` is *searched*; any
   parameter written as a bare number is *locked*.
2. GLADE fits the free parameters with **Differential Evolution (DE)** — a global,
   derivative-free optimizer (§7.1) — evaluating each candidate model with a lens
   engine that solves the lens equation and predicts image positions/magnifications.
3. Optionally, **MCMC** (Markov-chain Monte Carlo, §7.4) samples the region around the
   best fit so you get *uncertainties*, not just a single best value.
4. Every run ends with an optional **independent verification**: the best-fit model is
   re-solved by the untouched glafic binary and by a scipy-exact reference calculation
   (§7.8), so you can trust the numbers.

## 1.2 The pieces

| Directory | What it is |
|---|---|
| `glafic2/` | Bundled, locally modified **glafic 2.1.14** (C source + `glafic` binary + `import glafic` Python module). The CPU reference engine. GLADE's local additions: the King model (#27), a tightened Romberg tolerance (Chapter 10), and a per-component χ² binding `c2calc_each`. |
| `Rhongomyniad/` | The **GPU lens engine** (PyTorch/CUDA). Mirrors glafic's Python API; implements **all 27 glafic lens models** (V0.6) and evaluates whole DE/MCMC populations in single CUDA calls. Single lens plane only. |
| `core/` | The backend-agnostic optimizer: `.dat` parser/validator (`core/format`), DE (`core/optimize`), MCMC (`core/mcmc`), result figures (`core/plot`), glafic↔GLADE translation (`core/translate`), verification (`core/verify.py`). |
| `webui/` | The Flask + Monaco browser workbench: **FindImage** (run fits), **Editor** (edit `.dat` files), plus the **Clave** tab. |
| `clave/` | Clave, the interactive drag-and-compute lens calculator (third WebUI tab, also standalone). |
| `glade/` | The `import glade` library facade (V0.6): one import gives you `load_config`, `optimize`, `run_mcmc`, `make_triptych`, … from any script. |
| `InputFiles/` | Your working directory for `.dat` inputs — this is the only tree the WebUI's file manager can see. |
| `runs/` | One directory per WebUI/CLI run: logs, figures, `status.json`, best-fit parameters. |
| `tools/`, `legacy/`, `results/` | Pre-V0.4 workflow and its helper scripts. Kept for old results; **not** used by the current pipeline (§9.6). |

## 1.3 How to read this manual

- Chapters 3–4 are **tutorials** — do them in order once.
- Chapters 5–9 are **reference** — look things up as needed.
- Chapter 10 is a **must-read warning note** if you fit Sersic/NFW-like profiles or
  care about milliarcsecond-level accuracy.
- `Monospace` text is something you type or a literal file/key/button name. UI
  elements are quoted as they appear on screen, with the Chinese UI string added when
  the interface is bilingual (e.g. `▶ Run` / `▶ 运行`).
- Units: angular positions on the sky are in **arcseconds** (″) almost everywhere;
  the *observed image positions and their errors* are in **milliarcseconds (mas)**
  (1 mas = 0.001″). Masses are in solar masses (M☉). Watch for the unit callouts in
  Chapter 6.

---

# 2 Supported systems & requirements

## 2.1 Operating systems

| System | Status |
|---|---|
| **Linux (apt-based, e.g. Ubuntu)** | Primary target. Everything in this manual is tested here. |
| **WSL2 on Windows** | The development platform — fully supported. Jobs open in `gnome-terminal` windows via **WSLg** (the GUI layer of WSL2 on Windows 10 21H2+/11); the WebUI is reachable from the Windows browser at `http://localhost:6017` through WSL2's automatic port forwarding. See §3.5. |
| **macOS (Apple Silicon & Intel)** | An installer exists (`bootstrap_macos.sh`), but macOS support is **untested on real hardware** — it has never been compiled or run on an actual Mac. Use at your own risk and trust the installer's final `import glafic` self-check. GPU runs are not available on macOS (CUDA only). |

Other Linux distributions work if you install the equivalents of the apt packages in
§3.1 manually.

## 2.2 Software prerequisites

- **Python 3.8+** officially; the project is developed and CI-tested on
  **Python 3.12** — use 3.10–3.12 if you can.
- **C build tools** (`gcc`, `make`) — the installer builds glafic from source.
- **CFITSIO, FFTW3, GSL** — required to build glafic. On Linux the installer
  downloads and builds pinned versions (CFITSIO 4.6.2, FFTW 3.3.10, GSL 2.8) into
  `deps/install/`, so you do **not** need system packages for these. On macOS they
  come from Homebrew.
- Python packages (installed automatically from `requirements.txt`, unpinned):
  `numpy scipy matplotlib emcee corner tqdm astropy flask`.

## 2.3 GPU requirements (optional)

- An **NVIDIA GPU with CUDA** and a CUDA-enabled **PyTorch**. AMD/Intel GPUs and
  Apple Metal are not supported.
- **PyTorch is NOT installed by the bootstrap** (`torch` is deliberately absent from
  `requirements.txt` because the right wheel depends on your CUDA setup). Installing
  it is a manual step — see §3.3. Without torch, every CPU feature works; selecting a
  GPU rail fails fast with a clear error at job start.
- Under WSL2, CUDA works through the Windows NVIDIA driver (no driver install inside
  WSL is needed; just install a CUDA-build PyTorch wheel inside the venv).
- VRAM: the batched GPU optimizer's memory use scales with the population chunk size;
  the defaults fit comfortably in ~6 GB for typical configurations, and the chunk size
  is tunable via `GLADE_GPU_CHUNK` (§7.3).

## 2.4 Disk and time expectations

Measured on the reference WSL2 install: `deps/` ≈ 350 MB (dependency sources +
builds), `glafic2/` built ≈ 17 MB, `.venv/` ≈ 300 MB CPU-only — or ≈ 5.7 GB once a
CUDA PyTorch wheel is added. First-time installation is dominated by the three
`configure && make` dependency builds (minutes, hardware-dependent); re-runs skip
them.

---

# 3 Installation

## 3.1 One-command bootstrap (Linux / WSL2)

```bash
git clone https://github.com/y31ling/glaDE.git
cd glaDE
bash bootstrap_linux.sh
```

The script is interactive, with bilingual (English/中文) prompts. Two menus:

1. **Action** — `[1] Install / 安装` (default) or `[2] Uninstall / 卸载`.
2. **Install mode** — `[1] Virtual environment` (default; everything isolated in
   `.venv/`, recommended) or `[2] Global / System Python` (installs packages into the
   system Python; on Ubuntu 23.04+ this uses `--break-system-packages`).

What the install does, in order:

1. Checks the required apt packages (`build-essential pkg-config python3 python3-dev
   python3-venv python3-pip wget curl tar git libcurl4-openssl-dev zlib1g-dev`) and
   installs missing ones via `sudo apt-get install`. If `sudo` is unavailable, it
   stops and asks you to install them yourself.
2. Downloads and builds **CFITSIO 4.6.2 → FFTW 3.3.10 → GSL 2.8** into
   `deps/install/` (each is skipped on re-runs if the library already exists). Each
   tarball is fetched from a list of mirrors (the GSL list additionally includes
   Chinese mirrors — Tsinghua, USTC, Aliyun; CFITSIO and FFTW use only their
   upstream/macports/Fedora mirrors). If *all* mirrors fail, the script tells you exactly which tarball to
   download by hand and to place it in `deps/src/`, then re-run.
3. Builds **glafic2**: regenerates `glafic2/Makefile` (backing up any original to
   `Makefile.original` once) and runs `make clean && make -j all`, producing the
   `glafic` binary, `libglafic.a`, and the Python extension
   `glafic2/python/glafic/glafic.so`. glafic is always rebuilt from scratch on every
   install run.
4. Creates the Python environment and installs `requirements.txt`.
5. Writes a **`.pth` file** (`glafic_glade.pth`) into site-packages pointing at
   `glafic2/python`, so `import glafic` works from any Python session without extra
   setup.
6. Generates three launchers in the repo root: **`env.sh`**, **`run_glade.sh`**
   (legacy CLI), **`run_webui.sh`**.
7. Self-checks: imports `glafic` directly and again via `source env.sh`, printing
   `[OK] glafic import succeeded` lines. The install ends with "全部验证通过。"
   (all checks passed) and a "Next steps" summary.

### `env.sh`

`source env.sh` (note: *source*, don't execute) activates the venv and exports
`GLADE_ROOT`, `GLAFIC_HOME`, `PYTHONPATH` (repo root, `glafic2/python`,
`Rhongomyniad`, `tools`), `LD_LIBRARY_PATH` (`deps/install/lib`) and puts the
`glafic` binary on `PATH`.

> ⚠️ `env.sh` begins with `set -e`, which the *sourcing* shell inherits: after
> `source env.sh`, any failing command may close your interactive terminal. If that
> bites you, run `set +e` afterwards, or prefer the `run_*.sh` wrappers which source
> it in a subshell.

## 3.2 macOS installer (untested)

```bash
bash bootstrap_macos.sh
```

Differences from Linux: requires Xcode Command Line Tools and Homebrew; installs
`pkg-config gsl cfitsio fftw` via `brew` instead of building from source; builds
glafic with `cc` and macOS-style linking; `env.sh` exports
`DYLD_FALLBACK_LIBRARY_PATH`; multiprocessing uses `spawn` instead of `fork` (results
are identical). Again: **never run on a real Mac so far** — report problems if you
try it.

## 3.3 Installing PyTorch for the GPU (manual step)

The bootstrap does not install PyTorch. To enable the GPU rails:

```bash
source env.sh                      # activate the GLADE venv
pip install torch --index-url https://download.pytorch.org/whl/cu124   # example: CUDA 12.4 wheel
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Pick the wheel matching your CUDA setup at <https://pytorch.org/get-started/locally/>
(the reference machine runs `torch 2.6.0+cu124`). A CPU-only `pip install torch` also
works — Rhongomyniad silently falls back to CPU tensors — but then you lose the whole
point of the GPU rail. If `torch.cuda.is_available()` prints `False` under WSL2,
update the *Windows* NVIDIA driver.

## 3.4 Launching

```bash
./run_webui.sh                     # WebUI at http://localhost:6017 ; Ctrl+C stops it
GLADE_PORT=8080 ./run_webui.sh     # another port
```

> **Note.** The server binds `0.0.0.0`, so it is reachable from your local network,
> not just localhost. Don't run it on an untrusted shared network — the WebUI can
> read/write files under `InputFiles/` and launch processes.

For command-line runs and library use, see Chapter 9. `./run_glade.sh` starts the
pre-V0.4 *legacy* pipeline (`main.py`) — you only need it to reproduce old
`results/` runs (§9.6).

## 3.5 WSL2 specifics

- **Terminal windows.** Each WebUI run opens in its own `gnome-terminal` window,
  which WSLg displays like a normal Windows window. Install it once with
  `sudo apt install gnome-terminal`. If no terminal emulator is available at all,
  the job still runs (detached in the background) and its output still streams to the
  browser — you just don't get a window (§5.2.6).
- **Browser.** Open `http://localhost:6017` in your normal Windows browser; WSL2
  forwards localhost automatically.
- **Files from Windows.** The repo is inside the WSL filesystem. From Windows
  Explorer, reach it at `\\wsl$\<distro-name>\home\<user>\...` — the practical way
  to drop observed FITS images into `InputFiles/` (the WebUI has no upload button,
  §5.3.2).
- **Performance.** Keep the repo on the WSL (ext4) side, not under `/mnt/c/…`;
  Windows-drive I/O from WSL is an order of magnitude slower.

## 3.6 Updating after `git pull`

Re-run `bash bootstrap_linux.sh` → Install. The dependency builds are skipped, glafic
is rebuilt, Python requirements are re-installed, and the launchers/`.pth` are
regenerated. If only glafic C sources changed, `cd glafic2 && make all` is an
equivalent shortcut.

## 3.7 Uninstalling

`bash bootstrap_linux.sh` → `[2] Uninstall` lists the actions it will take — delete
`deps/`, delete `.venv/`, clean the glafic build (restores `Makefile.original`),
remove the generated launcher scripts, remove `.pth` files — and asks for one
overall confirmation; confirming runs them all. Only the follow-up shell-rc cleanup
asks per file before removing stray GLADE-related lines (each file is backed up as
`*.glade_uninstall_backup` first), and a final step merely reports leftover
environment-variable paths. Your source tree, `.dat` inputs and `runs/`
results are never deleted.

---

# 4 Quick start: your first fit in 15 minutes

This walkthrough runs the bundled example — a lens + substructure fit to a set of
four observed point-source images — on the CPU.

**1. Start the server and open the UI.**

```bash
./run_webui.sh
```

Open `http://localhost:6017`. You land on the **FindImage** tab, dark theme,
English UI (both are toggleable in the top-right corner).

**2. Get some input files.** The WebUI can only see the `InputFiles/` directory.
Copy the shipped example there:

```bash
cp core/examples/constants.dat core/examples/images_data.dat \
   core/examples/lens_and_substructure.dat InputFiles/
```

Click the `⟳` refresh button in the file panel — the three files appear.

**3. Select the three `.dat` files** with their checkboxes. A model may be split
across any number of files (constants / observations / lens components here); GLADE
merges them at run time (§6.8).

**4. Pick a backend.** Leave the left rail on **CPU** (the default; it evaluates
models with the glafic library — see §5.2.1 for what the five rail buttons mean).

**5. Click `▶ Run`.** If some basic settings were omitted from the files, a dialog
titled *"Missing basic values — use defaults?"* lists exactly which keys will fall
back to defaults and to what values. **Read this list** (Chapter 6 explains why:
some defaults are survey-specific, not neutral). Click `Confirm`.

**6. Watch the run.** A new terminal window opens (gnome-terminal under WSLg) and its
output simultaneously streams into the browser's *Terminal output* panel: the list of
optimizable dimensions, one `iter NNNN best_loss = …` line every few DE generations,
the DE result, and the verification report. The status chip next to the panel title
shows `running`, then `done · loss <x> · <N> iters`.

**7. Inspect results.** Below the terminal, the result figure (`result.png`, the
"triptych", §8.4) appears. Everything on disk lives in `runs/<job_id>/`:
`best_params.txt`, `status.json`, `job.log`, figures. The terminal window itself
stays open until you press Enter in it.

**8. Try MCMC.** Open the Editor tab, open one of your copies, insert the template
`MCMC → MCMC-GeneralConfig`, fill in the `$int`/`$float` placeholders (e.g.
walkers 32, steps 2000, burn-in 300), save, go back to FindImage and run the **CPU**
rail again — with `MCMC_ENABLED = True` in the file it automatically becomes a
"DE, then MCMC" run and adds a corner plot and trace plot to the results (§8.6).

That's the whole loop: *edit `.dat` → select → run → read figures*. The rest of this
manual is detail.

---

# 5 The WebUI

## 5.1 Overview

The WebUI is a single page with a 40-px top bar:

- Brand `GLADE`, then three tabs: **`FindImage`** / **`找像`**, **`Editor`** /
  **`编辑器`**, **`Clave`** (the name Clave is not translated).
- Top-right: the **language button** (shows `EN` or `中文`; default English) and the
  **theme button** (`🌙` dark / `☀️` light; default dark). Both persist in the
  browser's localStorage (`glade_lang`, `glade_theme`). The language toggle re-labels
  the whole UI; text already printed to the terminal panel is not re-translated.
- All server-side messages (run logs, validation errors) are English-only; only the
  UI chrome is bilingual.

Tab switching is instant; the Editor loads its code editor lazily on first visit, and
the Clave tab loads its page (an embedded iframe) on first visit.

## 5.2 The FindImage tab

Layout, left to right: **Backend rail** (the five run modes) → **file picker**
("Select .dat files" / "选择 .dat 文件") → **run column** (summary line, `▶ Run`
button, terminal panel, results area).

### 5.2.1 The five backend rails

| Button | Subtitle | What actually runs |
|---|---|---|
| **CPU** | `glafic` | GLADE's **Differential Evolution**, candidate models evaluated by the glafic *library* (the compiled C extension) in a multi-process pool. If the config sets `MCMC_ENABLED = True`, MCMC runs after DE. |
| **GPU** | `Rhongomyniad` | The same DE, but candidates evaluated by the GPU engine — whole population per CUDA call when the config is batchable (§7.3). Also honours `MCMC_ENABLED`. |
| **Glafic** | `amoeba` | **Not DE.** glafic's own downhill-simplex optimizer (`optimize`, "amoeba") run in the standalone glafic binary (§7.3.3). GLADE `.dat` selections are converted to glafic input files first; native glafic `.input` files run directly. |
| **MCMC** | `emcee only` / `仅 emcee` | MCMC **without** DE, on the CPU engine. The prior is the `{lo, hi}` box; walkers start uniformly inside it. Ignores `MCMC_ENABLED`. |
| **MCMC-GPU** | `emcee · batched CUDA` / `emcee · 批量 CUDA` | MCMC-only on the GPU, with the whole walker ensemble evaluated in batched CUDA calls where possible; walker count is auto-raised to 1024 in favourable cases (§7.5.3). |

> ⚠️ **The most confusable pair of buttons.** *CPU (subtitle "glafic")* means
> "GLADE's DE using the glafic library to score candidates". *Glafic (subtitle
> "amoeba")* means "glafic's own built-in simplex optimizer, no DE at all". They are
> different algorithms with different outputs.

The selection resets to **CPU** on every page reload (it is not persisted).

### 5.2.2 The file picker

- Shows the `InputFiles/` tree (same tree as the Editor's Explorer). Files have
  checkboxes — **multi-select across folders is allowed**; folders expand/collapse on
  click but cannot be selected as a whole.
- Selecting several files means "merge these into one configuration" (§6.8). The
  summary line shows `N file(s): <paths>`.
- The list refreshes when you click `⟳`, switch to this tab, or save/paste a file in
  the Editor. If a selected file disappears, its selection is dropped silently.
- There is no file-type filter — anything visible in `InputFiles/` is selectable, but
  only `.dat` (or, on the Glafic rail, glafic `.input`) selections make sense.

### 5.2.3 Run flow and dialogs

`▶ Run` is disabled with zero files selected. Clicking it can produce:

1. **Validation errors** — modal *"Cannot run — configuration errors"* /
   *"无法运行 — 配置错误"* listing every problem (one `✗` line per error; the
   messages name the offending key or component but carry no file/line numbers).
   The run is aborted; fix the `.dat` in the Editor. The complete error
   catalogue is in §6.10.
2. **Defaults confirmation** — modal *"Missing basic values — use defaults?"* /
   *"缺少基础变量 — 使用默认值?"* listing each omitted key and the default it will
   take, as `key = value` lines. `Confirm` proceeds; `Cancel` aborts. For a GPU MCMC
   run that qualifies for walker auto-raise, the dialog already shows
   `1024 (auto-raised for the batched GPU sampler)` so you see what will actually run.
3. **Success** — the terminal panel starts streaming.

Special cases on the **Glafic** rail: selecting only native glafic `.input` files
skips GLADE validation entirely (they run as-is; if several are selected only the
*first* runs); mixing `.dat` and `.input` in one selection is rejected with a clear
message.

### 5.2.4 The terminal panel

- Title: *"Terminal output — job `<job_id>` (`<terminal>`)"*, where `<terminal>` is
  the emulator the job runs in (`gnome-terminal`, `x-terminal-emulator`, `xterm`,
  `tmux`, or `detached`).
- The body is a live tail of the job's `job.log`, streamed over Server-Sent Events
  and auto-scrolled. Starting a new run clears the panel — **only one job stream is
  shown at a time and there is no scrollback of previous jobs** (the full log always
  remains in `runs/<job_id>/job.log`).
- The state chip shows `running` / `运行中`, then `done` / `完成` (green) with
  ` · loss <x> · <N> iters` and, if MCMC ran, ` · MCMC accept <a> (<n> samples)` —
  or the error state in red.
- If the browser loses the stream (`stream error`), **the job keeps running** in its
  terminal; the UI cannot re-attach. Check the terminal window or
  `runs/<job_id>/` on disk. The stream also times out by design after 30 minutes
  with no new output.

### 5.2.5 The results area

Appears under the terminal when the job finishes in state `done`, showing (in order,
only those that exist): **Result** / **结果** (`result.png`), **MCMC corner** /
**MCMC 角图**, **MCMC trace** / **MCMC 迹线**. Chapter 8 explains how to read each
figure. Other artifacts (`best_params.txt`, iteration frames, verification files) are
not displayed — fetch them from `runs/<job_id>/` on disk or via
`http://localhost:6017/api/run/<job_id>/result/<filename>`.

A job can legitimately finish `done` **without** a Result figure (e.g. the best-fit
model produced a different number of images than observed; the log then contains
`[warn] triptych failed: …`). The results area simply stays hidden.

### 5.2.6 Job execution model — terminals, stopping, concurrency

- Every run gets an id like `260708_153012_a3f1` (`yymmdd_HHMMSS_xxxx`) and its own
  directory `runs/<id>/`. Nothing stops you from launching several runs in parallel;
  they never share files.
- The job runs in its **own OS terminal window** titled `GLADE <job_id>`. Emulator
  priority: gnome-terminal → x-terminal-emulator → xterm → tmux (headless) →
  fully detached background process. In the detached case there is no window, but
  the browser stream still works.
- **There is no Stop button.** To abort a run, press `Ctrl+C` in (or close) its
  terminal window. The UI's status check notices the dead worker and reports state
  `interrupted` with an explanatory message.
- When the run finishes, the window prints
  `[GLADE job finished — press Enter to close]` and waits — press Enter to close it.
- **Restarting the WebUI server forgets all jobs**: their status becomes `unknown`
  and their figures can no longer be fetched through the UI, although everything is
  still on disk under `runs/`.

## 5.3 The Editor tab

A miniature VS Code for `InputFiles/`: a 50-px icon rail (panels **Explorer** /
**资源管理器** and **Template** / **模板**), a resizable side panel (drag the
divider; 120–600 px), a tab bar, the Monaco editor, and a footer with `🗑 Delete`,
the open file's path, and `💾 Save` / `💾 保存`.

### 5.3.1 Explorer

- Rooted at `InputFiles/` — the Editor cannot see or write anywhere else (any path
  escaping it is rejected server-side). Dotfiles are hidden.
- Header mini-buttons: `＋` New file (default name `untitled.dat`), `🗀` New folder,
  `⟳` Refresh. Typing a path like `sub/f.dat` into "New file" auto-creates the
  intermediate folder.
- **Right-click menu on a file**: `Open`, `Copy`, `Paste`, `Import glafic → glade…`,
  `Export glade → glafic…`, `Import to Clave`, `Rename…`, `Delete`.
  On a folder: `New file…`, `New folder…`, `Copy`, `Paste`,
  `Import glafic → glade…`, `Rename…`, `Delete folder`. (Chinese labels: `打开`,
  `复制`, `粘贴`, `导入 glafic → glade…`, `导出 glade → glafic…`, `导入到 Clave`,
  `重命名…`, `删除`, `删除文件夹`.)
- **Copy/Paste** works on files *and whole folders* (recursive). Paste lands inside a
  right-clicked folder, or next to a right-clicked file. Name collisions are
  auto-resolved as `name_copy.ext`, `name_copy2.ext`, …. Pasting a folder into itself
  is refused. There is no Cut.
- **Delete** always confirms (*"Delete `<name>`? This cannot be undone."*); folder
  deletion is recursive — one confirmation deletes everything inside.
- Known quirk: `Import glafic → glade…` also appears on *folder* menus but only works
  on files (on a folder it fails with an "Import failed" alert).

### 5.3.2 Getting files in and out

There is **no upload/download button**. To bring in external files — most commonly an
observed FITS image for extended-source fitting — copy them into
`<repo>/InputFiles/` on the filesystem (from Windows:
`\\wsl$\<distro>\home\<user>\...\InputFiles\`), then click `⟳`. Run outputs are
served back only through the FindImage results area / result API.

### 5.3.3 The editor itself

- **Monaco** (the VS Code editor component), vendored locally — multiple tabs, per
  tab undo history, a `●` dirty marker, syntax highlighting for the `.dat` format:
  `#` comments, `{lo, hi}` brace groups in teal, `$float`/`$int`/`$str` template
  placeholders in bold gold, strings, numbers.
- **No lint in the editor**: mistakes are only caught at run time by FindImage's
  validation dialog. There is also no autocomplete beyond Monaco's word suggestions.
- **Saving**: the `💾 Save` button only. There is **no Ctrl+S binding** (Ctrl+S opens
  the browser's own save dialog) and **no autosave**, and closing/reloading the
  browser tab silently discards unsaved changes in all tabs. Closing a dirty tab asks
  for confirmation — note the confirm button is labelled `Delete` / `删除` but means
  "discard changes and close".
- Saving a file refreshes both the Explorer and the FindImage picker, so a new `.dat`
  is immediately runnable.
- ⚠️ Every file in the tree opens in the editor, including binaries. A FITS file will
  display as garbage — **do not save it from the editor**, or the file on disk will
  be corrupted (the read/decode is lossy).

### 5.3.4 The Template panel

Templates insert ready-made, commented `.dat` snippets **at the cursor** of the
active tab (open a file first). Placeholders `$float` / `$int` / `$str` mark values
you must fill in; `$float{lower, upper}` marks an optimizable parameter — replace it
either with a real `{lo, hi}` range or a plain number to lock it. **A file with
unfilled placeholders fails validation at run time.** Group and template names are
English-only.

Component templates (Lens / Sub-structure / Extend Source) are auto-renumbered on
insert: the component key becomes `'nfw2'`, `'nfw3'`, … and the leading index becomes
one more than the highest already in the *current document* (mind this if you split
components across files).

The groups (V0.6.0):

| Group | Templates |
|---|---|
| `OBS DATA` | `Images Data` (source/lens redshifts, source position, the four observation arrays — positions and position errors in **mas**, magnifications and their errors dimensionless — `center_offset_*`, `obs_x_flip`), `Constants` (cosmology + grid), `Extend_images` (extended-source FITS keys + glafic χ²/noise settings) |
| `Source` | `point` (the others are placeholders, disabled) |
| `Lens` | `Sersic`, `SIE`, `power-law`, `Hernquist`, `Einasto`, `perturbation (shear)`, `Gaussian` (gaupot), `ahern`, `clus3`, `mpole`, `crline`, `gals` — all marked `# GPU-supported` |
| `Sub-structure` | `point-mass`, `NFW`, `gNFW`, `tNFW`, `analytic NFW`, `King`, `p-jaffe`, `acnfw` |
| `Extend Source` | `Sersic (extend)`, `Gaussian (extend)`, `top-hat (extend)`, `Moffat (extend)`, `Jaffe (extend)` — CPU-oriented extended-source profiles, require `extended_file` |
| `Algorithm parameters` | `DE-CPU`, `DE-GPU` (adds `gpu_precision`, drops `DE_WORKERS`), `DE-Extend (CPU)` (the `W_*` chi² weights) |
| `MCMC` | `MCMC-GeneralConfig` |

"Lens" vs "Sub-structure" is an *authoring convenience* — both insert the same kind
of component tuple and both end up in the same lens stack (§6.3). Each lens snippet's
comment lists the parameter meaning and order; models whose labels are best-effort
carry an extra "verify order vs glafic docs" comment.

### 5.3.5 Import / Export / Import to Clave

- **`Import glafic → glade…`** (right-click a glafic `.input` file): converts it into
  `<name>_model.dat` (+ `<name>_obs.dat` if it has constraints) in the same folder,
  overwriting silently. Parameters that glafic's optimize-matrix marked free arrive
  as **degenerate bounds `{v, v}`** — widen them by hand before fitting (a zero-width
  range searches nothing). Imported observations get `obs_x_flip = False` and zero
  `center_offset_*`; fix these if your data used the sky convention (§6.5).
- **`Export glade → glafic…`** (right-click a `.dat` file): prompts for an output
  name and writes into the `InputFiles/` **root**: always `<base>_model.input`; if
  the selection contains `{lo, hi}` parameters, the export is *optimize-ready* — the
  `.input` gains a `start_setopt` matrix and an `optimize` command, plus
  `<base>_obs.dat` (glafic `readobs_point` constraints) and `<base>_prior.dat`
  (glafic `parprior` ranges). (With no `{lo, hi}` but with observation data,
  `<base>_obs.dat` is still written — as a GLADE `start_obs` round-trip block that
  the glafic binary itself cannot read.) Bounds collapse to a representative starting value
  (geometric mean for masses, arithmetic mean otherwise). Shared variables (§6.6)
  export as glafic `match` ties.
- **`Import to Clave`** (right-click a `.dat` *or* `.input`): converts the model into
  a Clave scene and switches to the Clave tab. `{lo, hi}` values collapse to the same
  representative midpoints; only one point source is carried over; **the current
  Clave scene is replaced without confirmation**.

## 5.4 The Clave tab

Clave is an interactive **point-source lens calculator**: put lenses and sources on a
canvas, drag them around, and the lensed image positions, magnifications and time
delays are recomputed live by a real lens-equation solver (glafic on CPU,
Rhongomyniad on GPU). It is a scratchpad for building intuition and rough setups —
not a fitting tool (nothing is optimized; there are no extended sources and no
critical-curve display).

Ways to run it: as the third WebUI tab (`http://localhost:6017/clave/`), or
standalone with `python -m clave` (own server, default port 6019, override with
`CLAVE_PORT`).

Two Clave-specific quirks: it has its **own language toggle** (`中/EN` in its
toolbar) and **defaults to Chinese**, independent of the main WebUI language; and it
is **always light-themed**, even when the surrounding WebUI is dark.

### 5.4.1 Toolbar

`CLAVE` logo · subtitle · **CPU/GPU switch** with a status badge (`CUDA` = GPU ready,
with the GPU name in the tooltip; `CPU` = Rhongomyniad present but no CUDA device;
`N/A` = Rhongomyniad not importable, toggle disabled) · status dot/text (`就绪` /
`Ready`, `计算中...` / `Computing...`, `完成` / `Done`) · `中/EN` · mode button
**`实时模式` / `Realtime Mode`** ⇄ **`稳定模式` / `Stable Mode`**.

- **Realtime Mode** (default): recompute throttled to 10 Hz while you drag.
- **Stable Mode**: dragging hides the images; nothing recomputes until you press the
  `计算像位置` / `Calculate` button that appears in the sidebar footer. (Dialog edits
  and add/delete still compute immediately — only drags are deferred.)

### 5.4.2 Canvas

- **Pan**: drag empty space. **Zoom**: mouse wheel (about the cursor). The adaptive
  grid labels switch between ″ and mas automatically as you zoom.
- **Select**: click an object (blue highlight); **move**: drag a selected object, or
  Alt-drag any object. Locked objects (padlock in the sidebar row) don't move.
- **Rotate** (lenses): with a lens selected, drag the small handle that appears on
  its major axis to set the position angle; the sidebar shows `PA=…°` live.
- Glyphs: sources are blue six-pointed stars; images are amber five-pointed stars
  whose opacity scales with |μ| (brighter image = more opaque); lenses are translucent
  shapes sized from their parameters (e.g. SIE disc ≈ its Einstein radius; NFW-family
  blobs are drawn mas-scale for substructure work; King shows core + tidal-radius
  ring).

### 5.4.3 Sidebar, dialogs and the model picker

- Sections `光源 Sources` and `透镜 Lenses` with `ADD SOURCE` / `ADD LENS` blocks.
  Rows show live coordinates; hover reveals lock (`🔒`) and delete (`✕`); drag the
  `≡` handle to reorder (order = engine lens numbering). Double-click a row to edit.
- The **Add Lens** dialog has a searchable type dropdown with 15 curated models in
  three groups — Common: `sie`, `nfw`, `king`, `gnfw`, `point`, `jaffe`; Extended:
  `sers`, `hern`, `pow`, `tnfw`, `ein`, `anfw`, `ahern`; Perturbation: `pert`,
  `gaupot` — each with sensible defaults (e.g. SIE σ = 200 km/s, z = 0.5). Other
  glafic models can still enter Clave via *Import to Clave*.
- Every dialog has an **Editor Mode** (`切换文本编辑器`): type the whole object as
  one line, `type z p1 x y e pa r1 r2`. Pasting a whitespace-separated line into the
  first field also distributes the values across the inputs.
- The `zs_fid` family (`pow`, `pert`, `gaupot`) follows the same convention as
  §6.4: the first "z" field is the **lens redshift** (default 0.5) and the
  `参考源红移 zs` / `Fiducial zs` field is the fiducial source redshift `zs_fid`
  (default 1 — keep it above the lens redshift). Also: a redshift entered as `0`
  silently reverts to the default (lens 0.5, source 2), and non-numeric entries
  become 0.
- Cosmology is fixed at (Ω_m, Ω_Λ, w, h) = (0.3, 0.7, −1, 0.7) and is not editable;
  redshifts are per-object.

### 5.4.4 Images panel and export

Once the scene has at least one lens and one source, a read-only
**`像 Images (n)` / `Images (n)`** section lists every image: `★j  x=…″ y=…″ μ=…`
(labelled per source when there are several). Time delays are computed but shown only
in the export. `导出数据` / `Export Data` downloads `clave_export.txt` — a
glafic-flavoured text file with the lens/source lines and an image table
(x, y, μ, time delay). The export is one-way (Clave cannot re-import it); scenes are
not persisted across reloads.

### 5.4.5 GPU mode notes

GPU mode requires all lenses on a single redshift plane (within 10⁻³) and only
models in `rhongomyniad.supported_models()` (all 27 in V0.6). The first GPU
interaction may pause briefly while kernels warm up. CPU and GPU use slightly
different initial search grids, so a very faint extra image can occasionally differ
between them near critical curves. Images falling outside the automatically sized
search box (± max(0.4″, 2 × scene extent)) are not found — relevant for
wide-separation configurations. If the compiled glafic module is missing, CPU mode
serves **mock** images (status shows `Mock mode`) — placeholder geometry, meaningless
numbers.

---

# 6 The `.dat` configuration format

A GLADE model is written in one or more `.dat` files. The syntax looks like Python
but is **parsed by a restricted reader, never executed**: only literals, `{lo, hi}`
pairs, lists, tuples, name references, subscripts, and `+ - * / **` arithmetic are
allowed. Anything else (function calls, comparisons, dictionaries…) is a syntax
error. Validation reports *all* problems at once (with file and line numbers in
library/CLI use; the WebUI dialog shows the messages only).

A minimal complete model, split over the canonical three files
(`core/examples/*.dat` ships a working set):

```python
# --- constants.dat -------------------------------------------
omega        = 0.3          # Ω_m
lambda_cosmo = 0.7          # Ω_Λ
weos         = -1.0         # dark-energy w
hubble       = 0.7          # H0/100
xmin, ymin   = -0.5, -0.5   # search field [arcsec]
xmax, ymax   =  0.5,  0.5
pix_ext      = 0.01         # extended-source pixel [arcsec]
pix_poi      = 0.2          # point-source grid cell [arcsec]
maxlev       = 5            # adaptive-mesh refinement levels

# --- images_data.dat -----------------------------------------
source_z = 0.4090
lens_z   = 0.2160
source_x = {-0.10, 0.10}    # {lo, hi} = optimize; bare value = lock
source_y = 0.0244

obs_positions_mas_list  = [[229.0, -390.6], [-286.4, -401.7],
                           [-267.4, 393.8], [ 385.8, 274.0]]   # mas!
obs_magnifications_list = [ 8.0, 6.1, 4.9, 3.7]
obs_mag_errors_list     = [ 0.8, 0.6, 0.5, 0.4]
obs_pos_sigma_mas_list  = [ 2.0, 2.0, 2.0, 2.0]                # mas
center_offset_x = 0.0
center_offset_y = 0.0
obs_x_flip = True           # True = sky convention (x sign flipped)

# --- lens_and_substructure.dat -------------------------------
'sie1':   (1, 'sie',   lens_z, {150, 350}, {-0.05, 0.05}, {-0.05, 0.05},
           {0.0, 0.6}, {0, 180}, 0.0)
'point1': (2, 'point', lens_z, {1e5, 1e7}, {-0.30, -0.20}, {-0.05, 0.05})
```

## 6.1 Line-level grammar

- `#` starts a comment (outside quoted strings) to end of line. Comments are legal
  inside multi-line values, but **never comment out a line containing an opening or
  closing bracket** — bracket matching happens after comments are stripped.
- A statement can span multiple lines while a `(`, `[` or `{` is open.
- Tuple unpacking works: `xmin, ymin = -0.5, -0.5`.
- Each scalar may be assigned **once per file** (and once across all selected files —
  §6.8); re-assigning is an error, including via an alias (`lambda` ≡
  `lambda_cosmo`).
- Two statement kinds exist: scalar assignments (`name = value`) and component
  entries (`'name': (…)`, §6.3). Anything else is an error.

## 6.2 Value forms

| Form | Meaning |
|---|---|
| `0.3`, `-1`, `2.65e-3` | fixed (locked) number |
| `{lo, hi}` | **optimizable**: one search dimension with inclusive bounds |
| `[a, b, …]`, `[[…], …]` | list (observation arrays) |
| `True` / `False` | boolean |
| `'text'` / `"text"` | string (file paths, output prefix) |
| `name` | reference to a previously defined numeric scalar |
| `$float`, `$int`, `$str`, `$float{lower, upper}` | unfilled template placeholder — **blocks the run** until replaced |

Details worth knowing:

- `{hi, lo}` written in the wrong order is silently normalised (no warning).
  `{a, b, c}` with ≠ 2 elements is an error.
- Pure-number arithmetic is folded at parse time: `re = 0.3 + 0.09` is fine
  anywhere. Expressions that reference *observations* are allowed **only inside
  component tuples** (§6.7).
- **Mass-like parameters are searched in log₁₀ space** but you always *write physical
  values*: `{1e5, 1e7}` means "search log₁₀M uniformly between 5 and 7". Both bounds
  must be > 0 for mass-like slots.
- A `{lo, hi}` on a scalar creates a search dimension only where supported:
  `source_x`, `source_y`, `hubble`, component `z`/parameters, and user variables.
  Notably, **`lambda_cosmo = {lo, hi}` is accepted but silently ignored** (it falls
  back to the fixed default — a documented limitation; only `hubble` among the
  cosmology keys can be fitted).

## 6.3 Component tuples

```python
'name': (N, 'type', z, p1, p2, ..., pk)
```

- `'name'` — a quoted label of your choice (`'sie1'`, `'sub_A'`); it becomes the
  prefix of the parameter labels in all outputs (`sie1.sigma`, `sub_A.mass`).
- `N` — an integer index. It is **advisory only**: indices are renumbered globally,
  1-based, in file-selection order when files merge.
  - Optional **classification suffix**: `3l`/`3L` forces the component to be treated
    as a *main lens*, `3s`/`3S` as a *sub-structure* (this only changes the red
    "Sub-halo" markers/labels on the result figure — never the physics). Without a
    suffix, the model's schema category applies, except that **any component with an
    optimizable parameter is treated as sub-structure by default**. Any other letter
    is an error.
- `'type'` — a quoted model keyword from §6.4.
- `z` — the component redshift: a number, `{lo, hi}`, or a reference like `lens_z`.
  For *extended-source* profiles this slot is the **source** redshift.
- `p1…pk` — model parameters **in glafic `set_lens` order** (§6.4). Each slot may be
  a number, `{lo, hi}`, a reference, an expression (§6.7), or a placeholder. Trailing
  parameters beyond the model's required minimum may be omitted (they become 0) —
  most deflectors need at least the first three (mass-like, x, y); extended profiles
  need 6–7 (`extsersic`/`extmoffat` require all 7). At most 7 parameters after `z`.

Main lenses and sub-structures live in **one shared component stack** — the Editor's
"Lens" vs "Sub-structure" template groups are just authoring categories. You may mix
any models the chosen backend supports (e.g. an SIE main lens plus a King and a
point-mass sub-halo).

## 6.4 Model reference

All 27 glafic deflector models are recognised, and in V0.6 **all of them run on the
GPU backend too** (the one remaining GPU exception is *multi-plane* configurations —
components at different redshifts). Parameter order after `z`; **bold** = mass-like
(log₁₀ search).

**Deflectors** (glafic `set_lens`):

| type | parameters after z (in order) | notes |
|---|---|---|
| `point` | **mass**, x, y | point mass [M☉] |
| `sie` | **sigma**, x, y, e, pa, rcore | singular isothermal ellipsoid; σ [km/s] |
| `jaffe` | **sigma**, x, y, e, pa, a, rco | pseudo-Jaffe (outer truncation a, core rco) |
| `sers` / `serspot` | **mass**, x, y, e, pa, re, n | Sersic; n ∈ [0.06, 20] |
| `nfw` / `nfwpot` / `anfw` | **mass**, x, y, e, pa, c | NFW (anfw = analytic CSE approx.) |
| `gnfw` / `gnfwpot` | **mass**, x, y, e, pa, c, alpha | generalized NFW (inner slope α, table range [0, 2]) |
| `tnfw` / `tnfwpot` | **mass**, x, y, e, pa, c, t | truncated NFW |
| `acnfw` | **mass**, x, y, e, pa, c, b | cored NFW (CSE); **b = core/rs ∈ [0, 100)** strict |
| `king` | **mass**, x, y, e, pa, rc, c | King (1962); **GLADE-local model #27**; c = log₁₀(rt/rc) ≥ 0 |
| `hern` / `hernpot` / `ahern` | **mass**, x, y, e, pa, rb | Hernquist |
| `ein` / `einpot` | **mass**, x, y, e, pa, c, alpha | Einasto; α table range [0.02, 1.0] |
| `pow` / `powpot` | zs_fid, x, y, e, pa, **re**, gamma | power law — note the mass-like slot is **re** (p6), and p1 is the fiducial source redshift |
| `pert` | zs_fid, x, y, **gamma**, theta_gamma, –, **kappa** | external shear + convergence |
| `gaupot` | zs_fid, x, y, e, pa, **sigma**, **kappa0** | Gaussian potential |
| `clus3` | zs_fid, x, y, **gamma**, theta_gamma | cluster perturbation |
| `mpole` | zs_fid, x, y, **gamma**, theta_gamma, m, n | multipole |
| `crline` | zs_fid, x, y, –, pa, epsilon, kappa | straight critical line |
| `gals` | **sigma**, x, y, –, –, a, alpha | galaxy catalogue (pseudo-Jaffe members); needs a `galfile.dat` catalogue — see §9.4 |

Common parameter meanings: `x, y` centre [arcsec]; `e` ellipticity ∈ [0, 1); `pa`
position angle [deg]; sizes (`rcore, re, rb, rc, a, …`) in arcsec. The `…pot`
variants put the ellipticity in the potential instead of the density. For the
`zs_fid` family (`pow`, `pert`, `gaupot`, `clus3`, `mpole`, `crline`), p1 is a
*fiducial source redshift* that fixes the normalisation — a common conversion trap.
Models flagged "best-effort labels" in the schema (`pow`, `pert`, `gaupot`, `clus3`,
`mpole`, `crline`, `gals`) emit a warning when you optimize them: double-check the
parameter order against the glafic manual (`glafic2/manual/man_glafic.pdf`).

**Extended-source profiles** (glafic `set_extend`; the tuple's z slot = **source**
redshift; p1 = `norm`, whose meaning follows `flag_extnorm`: 0 = peak surface
brightness, 1 = total flux):

| type | parameters after z | |
|---|---|---|
| `extsersic` | norm, x, y, e, pa, re, n | Sersic profile |
| `extgauss` | norm, x, y, e, pa, sigma | Gaussian |
| `exttophat` | norm, x, y, e, pa, radius | top hat |
| `extmoffat` | norm, x, y, e, pa, rd, beta | Moffat |
| `extjaffe` | norm, x, y, e, pa, a, rco | Jaffe |

## 6.5 Observation data (point-source path)

Four arrays are **hard-required** (no defaults) whenever you fit point-source images:

```python
obs_positions_mas_list  = [[x1, y1], [x2, y2], ...]  # image positions [mas]
obs_magnifications_list = [mu1, mu2, ...]            # signed magnifications
obs_mag_errors_list     = [d1, d2, ...]              # 1σ magnification errors (> 0)
obs_pos_sigma_mas_list  = [s1, s2, ...]              # 1σ position errors [mas] (> 0)
```

All arrays must have the same length; error entries must be finite and positive
(zero/negative errors would make χ² blow up and are rejected).

The observed frame is mapped into the engine (lens) frame by:

```
x_engine = ±(x_mas/1000 − center_offset_x)     # − sign when obs_x_flip = True
y_engine =   y_mas/1000 − center_offset_y
```

- `obs_x_flip = True` means the **sky convention** (x increases to the east ⇒ sign
  flip relative to a mathematical x-axis). It is a *sign flip on x*, never an x/y
  swap.
- `center_offset_x/y` [arcsec] shift the observed frame's origin onto the lens
  frame's origin.

> ⚠️ **The single most dangerous defaults in GLADE.** The built-in defaults for these
> keys are inherited from the iPTF16geu analysis the project grew out of:
> `center_offset_x = 0.01535`, `center_offset_y = 0.0322`, `obs_x_flip = True`,
> and `source_x/y`/`source_z`/`lens_z` default to iPTF16geu best-fit values. For any
> new lens system **set them explicitly** (usually `center_offset_x = 0`,
> `center_offset_y = 0`, and your own `obs_x_flip`). The defaults-confirmation
> dialog (§5.2.3) exists precisely so you notice.

Signed magnification and *parity*: images formed at a minimum/maximum of the time
delay surface have positive parity, saddle-point images negative — hence signed μ in
the array. By default GLADE compares |μ| (the `abs_mag` key, §7.2), so a parity flip
near a critical curve doesn't dominate your χ².

## 6.6 User-defined shared variables

Any assignment whose name is *not* a recognised key defines a **variable**:

- A fixed variable (`my_re = 0.39`) simply substitutes its value wherever referenced.
- An optimizable variable referenced from component tuples becomes **one shared
  search dimension** — every referencing slot is fitted to the *same* value:

```python
lens_x = {-0.07, 0.07}
lens_y = {-0.07, 0.07}
'sers1': (1, 'sers', lens_z, {1e9, 1e12}, lens_x, lens_y, {0.01, 0.5}, {0, 180}, {0.08, 0.5}, {0.5, 1.5})
'sers2': (2, 'sers', lens_z, {1e9, 1e12}, lens_x, lens_y, {0.01, 0.5}, {0, 180}, {0.5, 2.0}, {0.5, 1.5})
```

Here both Sersic components are forced to share one centre, and `lens_x`, `lens_y`
each appear once in corner plots and fit outputs, under their own names.

Rules:

- One variable may not serve a mass-like slot *and* a linear slot simultaneously
  (error `var_mixed_usage`: define two variables). Used in a mass-like slot it is
  log₁₀-searched like any mass.
- A `{lo, hi}` variable can only be referenced *from component tuples* — not from
  other scalar assignments or lists.
- References work **across files** in a multi-file selection.
- A defined-but-never-referenced `{lo, hi}` variable is a warning (`var_unused`) —
  usually a typo.
- ⚠️ Corollary: **a misspelled configuration key is silently accepted as an inert
  user variable.** `DE_MAXITTER = 900` does nothing and warns nobody. When a setting
  seems to have no effect, check its spelling first (and see the `[defaults]` line in
  the job log, which lists every key that fell back to a default).
- On the GPU backend, a shared variable on a component *redshift* or a `zs_fid` slot
  silently disables the fast batched path (correct but slower per-candidate mode).

## 6.7 Arithmetic & observation expressions (V0.5.3)

Inside component tuples (values and bounds), you can reference the observed image
positions:

```python
'point1': (4, 'point', lens_z, {1e2, 1e8},
           {img1_x - 0.075, img1_x + 0.075},
           {img1_y - 0.075, img1_y + 0.075})
```

- `imgN_x` / `imgN_y` = the N-th observed image (1-based).
  `obs_positions_mas_list[i][j]` (0-based; j: 0 = x, 1 = y) refers to the same data.
- **Critical unit rule:** these references are *not* raw mas numbers — they are the
  positions already converted into the engine frame (mas→arcsec, x-flip,
  center-offset; the exact formula of §6.5). So the example above is a ±0.075″
  search box centred on where image 1 actually sits in the lens frame, and any
  arithmetic constants are **arcsec, engine frame**.
- Elements of the *other* observation arrays (`obs_magnifications_list[k]`, …) are
  usable too but come through **raw** (no transform).
- Allowed operators: `+ - * / **` and parentheses. Errors (unknown name, index out
  of range, expression outside a component tuple, division by zero…) are reported
  with the component and parameter slot.
- If the `.dat` omits `center_offset_*`/`obs_x_flip`, the expressions use the
  *defaults* (the iPTF16geu values — §6.5) so that they match what the optimizer
  will use.

Expressions are resolved at load time, after all files merge — every backend
(CPU/GPU/amoeba export/editor lint) sees identical numbers.

## 6.8 Multi-file merge

Selecting several files in FindImage (or passing several paths to
`glade.load_config`) merges them:

- Every scalar may be defined in **at most one** file of the selection
  (error `conflict` otherwise, aliases included).
- Components concatenate in selection order and are renumbered globally.
- Cross-file references are fine: a tuple in file B may use `lens_z` from file A.
- After the merge, missing basics are defaulted (with the confirmation dialog); the
  four observation arrays and at least one component can never be defaulted.

The conventional split is *constants* / *observations* / *lens model*, which makes it
trivial to swap lens hypotheses against the same data.

## 6.9 Complete key reference

Aliases: `lambda` → `lambda_cosmo`, `MISSING_IMG_PENALTY` → `missing_img_penalty`,
`GPU_PRECISION` → `gpu_precision`. "Default" = value applied (after confirmation)
when the key is absent.

**Cosmology & field**

| key | default | meaning |
|---|---|---|
| `omega` | 0.3 | Ω_m |
| `lambda_cosmo` | 0.7 | Ω_Λ (`{lo,hi}` accepted but ignored — not fittable) |
| `weos` | −1.0 | dark-energy equation of state w |
| `hubble` | 0.7 | H₀/100. May be `{lo, hi}` → an extra fit dimension, **only meaningful for extended-source runs with time delays** (§7.6); on the point path it is a dead dimension (refused by the batched GPU). |
| `xmin, ymin, xmax, ymax` | ±0.5 | image-plane search field [arcsec] — images outside are not found |
| `pix_ext` | 0.01 | extended-source pixel size [arcsec]; defines the FITS grid |
| `pix_poi` | 0.2 | coarse grid cell for point-image search [arcsec] |
| `maxlev` | 5 | adaptive-mesh refinement depth (raise it if an image near a critical curve is missed) |

**Redshifts / point source**

| key | default | meaning |
|---|---|---|
| `source_z` | 0.4090 | source redshift (iPTF16geu default!) |
| `lens_z` | 0.2160 | main-lens redshift (iPTF16geu default!); usable as a reference in tuples |
| `source_x`, `source_y` | iPTF16geu values | point-source position [arcsec]; lockable or `{lo, hi}`. In extend mode they are *not* DE dimensions (glafic solves the source position internally; `{lo,hi}` just marks them free). |

**Observations** — §6.5 (the four arrays, `center_offset_x/y`, `obs_x_flip`);
extend-mode extras: `obs_td_list`/`obs_td_err_list` (time delays + errors, default 0
per image), `obs_parity_list` (default 0).

**Differential Evolution**

| key | default | meaning |
|---|---|---|
| `DE_MAXITER` | 650 | maximum generations |
| `DE_POPSIZE` | 64 | population multiplier — actual population = `DE_POPSIZE × ndim` |
| `DE_ATOL` / `DE_TOL` | 1e-4 / 1e-4 | absolute/relative convergence thresholds |
| `DE_SEED` | 42 | RNG seed. Same seed + same config ⇒ **identical trajectory on any backend and worker count** (also reused as the MCMC init seed). |
| `DE_POLISH` | True | accepted but **has no effect** in the current pipeline (scipy's polish step is never reached) |
| `DE_WORKERS` | −1 | CPU worker processes; −1 = all cores. Forced to 1 on the GPU backend. |
| `EARLY_STOPPING` | True | stop when the best loss stagnates |
| `EARLY_STOP_PATIENCE` | 30 | consecutive within-tolerance generations required to stop |

**Loss (point path)** — `Y = LOSS_COEF_A·χ²_pos + LOSS_COEF_B·χ²_mag + penalty` (§7.2)

| key | default | meaning |
|---|---|---|
| `LOSS_COEF_A` | 4 | weight of the position χ² |
| `LOSS_COEF_B` | 1 | weight of the magnification χ² |
| `LOSS_PENALTY_PL` | 10000 | extra linear penalty per image whose position residual exceeds its 1σ |
| `missing_img_penalty` | 0.0 | per-missing-image penalty. 0 = a candidate producing fewer images than observed is hard-rejected; > 0 = it is scored on the images it has, plus `(n_obs − n_pred) × penalty`, giving DE a gradient toward full image multiplicity. |
| `abs_mag` | True | compare (and plot) magnifications by absolute value — parity-insensitive. `False` restores the signed pre-V0.5 behaviour bit-exactly. |

**GPU**

| key | default | meaning |
|---|---|---|
| `gpu_precision` | 64 | batched-GPU float precision: `64` = fp64 everywhere; `48` = mixed (fp32 deflection fields + fp64 Newton refine — **recommended**, fp32 speed at fp64 accuracy); `32` = fp32 everywhere (fast but noisy near critical curves). Ignored by CPU runs and by the extended-source batched path (always fp64). Any other value is a validation error. |

**Output / verification**

| key | default | meaning |
|---|---|---|
| `Draw_Graph` | 1 | write DE-population corner frames into `runs/<id>/iterations/` |
| `draw_interval` | 5 | generations between frames |
| `OUTPUT_PREFIX` | `'glade_run'` | output prefix (used as glafic `prefix` on export) |
| `glafic_verified` | True | after the run, re-check the best fit with the independent glafic binary + scipy reference (§7.8) |
| `COMPARE_GRAPH` / `SHOW_2SIGMA` / `CONSTRAINT_SIGMA` / `PENALTY_COEFFICIENT` / `PRINT_INTERVAL` | True / False / 1 / 1000 / none | accepted legacy keys, **not consumed** by the current pipeline — they still receive these defaults (and so appear in the `[defaults]` log line and confirmation dialog) |

**MCMC** (§7.4–7.5)

| key | default | meaning |
|---|---|---|
| `MCMC_ENABLED` | False | on the CPU/GPU rails: True = run MCMC after DE ("de+mcmc"). The dedicated MCMC/MCMC-GPU rails ignore this flag. |
| `MCMC_NWALKERS` | 32 | ensemble size; floored at runtime to `2·ndim + 2`; auto-raised to 1024 on qualifying MCMC-GPU runs when left unset |
| `MCMC_NSTEPS` | 2000 | steps per walker |
| `MCMC_BURNIN` | 300 | initial steps discarded |
| `MCMC_THIN` | 2 | keep every N-th post-burn-in step |
| `MCMC_PERTURBATION` | 0.01 | de+mcmc walker-init scatter around the DE best (fraction of each bound's width) |
| `MCMC_WORKERS` | 1 | CPU pool for the likelihood; −1 = all cores, **only safe in a foreground terminal** (§7.7); ignored on the batched GPU path |
| `MCMC_PROGRESS` | True | progress bar (library use only; WebUI runs print step lines instead) |
| — | — | there is **no `MCMC_SEED`**; `DE_SEED` seeds walker initialization, but the sampler itself is not seeded, so chains are not exactly reproducible |
| `MCMC_CUSTOM_RANGE`, `MCMC_SEARCH_RADIUS`, `MCMC_LOG_M_MIN`, `MCMC_LOG_M_MAX` | — | removed in V0.4.1; accepted with a deprecation warning, ignored — the MCMC prior is *always* the `{lo, hi}` bounds |

**Extended-source mode** (§7.6) — file paths are strings, resolved relative to the
directory of the *first* selected `.dat`, then the working directory:

| key | default | meaning |
|---|---|---|
| `extended_file` | — (required in extend mode) | observed FITS image → glafic `readobs_extend`. Setting it (or including any `ext*` component) switches the whole run to the extended-source path. |
| `extend_mask_file` | none | optional pixel-mask FITS (pixels > 0 ignored) |
| `noise_file` | none | optional per-pixel noise FITS (else noise is derived analytically from `obs_gain`/`obs_readnoise`/…) |
| `constraint_file` | none | glafic-native `readobs_point` file with point constraints (takes precedence over the four obs arrays if both are present) |
| `prior_file` | none | glafic-native `parprior` file (Gaussian/range priors on parameters) |
| `W_POS, W_FLUX, W_TD, W_EXT, W_PRIOR` | 1.0 each | weights on glafic's χ² components: `loss = W_POS·pos + W_FLUX·flux + W_TD·td + W_EXT·pixel + W_PRIOR·priors + penalty`. All 1.0 reproduces glafic's `c2calc` exactly. |
| `chi2_splane, chi2_checknimg, chi2_restart, chi2_usemag, ran_seed, obs_gain, obs_ncomb, obs_readnoise, flag_extnorm` | glafic's | forwarded verbatim to glafic `set_secondary` (see the glafic manual). `chi2_splane 1` (source-plane point χ²) is required for the *batched* GPU extend path when point constraints exist. |

## 6.10 Validation

Errors (block the run): syntax errors; unfilled placeholders; unresolved/illegal
references; duplicate scalar definitions across files; bad expressions; missing
observation arrays (or, in extend mode, giving some-but-not-all of the four);
missing `extended_file` / extend component; malformed obs arrays (type/length/shape/
non-positive errors); unknown model type; too few/too many component parameters;
non-positive mass bounds; a model not supported by the chosen backend; a shared
variable used in both mass and linear slots; bad `gpu_precision`; unknown backend;
no components at all.

Warnings (run proceeds): deprecated keys; unused `{lo,hi}` variables; both obs
arrays and a `constraint_file` given (the file wins); best-effort parameter labels;
classification suffix on an extend component (ignored); GPU backend with components
at multiple lens redshifts (single-plane engine may reject it).

In library/CLI use, issues print as `[error] file.dat:LINE: message`; the WebUI
error dialog shows the message text only (no file/line).

---

# 7 Fitting engines & algorithms

## 7.1 Differential Evolution (DE)

**What it is.** DE is a *global*, derivative-free optimizer. It keeps a whole
*population* of candidate parameter vectors, and in every *generation* builds a trial
for each member by adding a scaled difference of two random members to the current
best (`best1bin` strategy), mixing it with the parent (crossover), and keeping
whichever scores better. Populations explore broadly early on and contract onto the
minimum — well suited to lens modelling, where the loss surface is riddled with local
minima and no gradients are available.

**GLADE's DE specifics** (built on `scipy.optimize.differential_evolution`):

- Every `{lo, hi}` in your `.dat` is one dimension; mass-like dimensions are searched
  in log₁₀. Population size = `DE_POPSIZE × ndim` (default 64 × dims).
- Strategy `best1bin`, mutation dithered in (0.5, 1), crossover 0.7, Latin-hypercube
  initialization — scipy defaults, not user-configurable.
- **Reproducibility invariant**: the population update is forced to scipy's
  "deferred" mode on all paths, so *the same config + `DE_SEED` walks an identical
  trajectory* whether you run CPU, GPU, 1 worker or 32.
- **Early stopping** (GLADE's own): a generation counts as "converged" when the best
  loss improved by less than `DE_ATOL` (absolute) or `DE_TOL` (relative);
  `EARLY_STOP_PATIENCE` consecutive such generations end the run. scipy's own
  population-spread criterion and `DE_MAXITER` also terminate it.
- The reported best is always a population member (`DE_POLISH` has no effect).
- Progress lines `iter NNNN best_loss = …` appear every 5 generations.

**Choosing ranges.** DE cost grows with dimensionality and range width. Lock
everything you can; give generous-but-physical `{lo, hi}` for the rest; exploit the
observation expressions (§6.7) to centre position boxes on observed images.

## 7.2 The point-source loss function

For each candidate, the engine solves the lens equation and returns the predicted
images. GLADE then computes

```
Y = LOSS_COEF_A · χ²_pos + LOSS_COEF_B · χ²_mag + penalty
```

- `χ²_pos = Σ (Δᵢ/σᵢ)²` — Δᵢ is the distance [mas] between observed image *i* and its
  matched predicted image (matching is a Hungarian optimal assignment on distance);
  σᵢ is `obs_pos_sigma_mas_list[i]`.
- `χ²_mag = Σ ((μ_pred − μ_obs)/δμ)²`, with |μ| values when `abs_mag = True` (the
  default) — so an observed +30 vs a model −29 counts as a difference of 1, not 59;
  parity flips near critical curves are not punished.
- `penalty = Σ LOSS_PENALTY_PL · Δᵢ` over images whose Δᵢ exceeds σᵢ — linear in the
  *full* residual, so it switches on as a large jump the moment Δᵢ crosses σᵢ: a
  strong shove toward sub-σ positions.

**Image-count rules.** If the model predicts exactly one image more than observed,
the faintest is dropped (assumed to be the demagnified central image of a cuspy
profile). Any other excess ⇒ the candidate is rejected outright (loss 10¹⁵). Fewer
images than observed: rejected by default, **unless** `missing_img_penalty > 0`, in
which case the candidate is scored on the images it produced plus
`(n_obs − n_pred) × missing_img_penalty` — this "graded" mode gives DE a slope to
climb toward full multiplicity instead of a cliff, and is recommended when good
candidates keep getting rejected early. (Its useful magnitude differs between the
point and extended paths — retune when switching.)

## 7.3 Backends

### 7.3.1 CPU (and why "glafic" appears twice)

The `cpu` backend evaluates candidates with the **glafic C extension** in a
multi-process pool (`DE_WORKERS`, default all cores). The `glafic` *backend name* in
`runjob.py` is the same engine; the WebUI's **Glafic rail** however runs glafic's
*amoeba* (§7.3.3). CPU is the reference: every other path is validated against it.

### 7.3.2 GPU: batched evaluation, precision, chunking

The `gpu` backend evaluates candidates with Rhongomyniad. Its speed comes from
**batching**: when the configuration allows, the *entire DE population* is evaluated
in one CUDA pass per generation. Reference numbers (RTX 4080 SUPER, 20-dim NFW fit,
180 candidates/generation): ~21 s per generation evaluated one-by-one → ~2.3 s
batched fp64 → **~0.86 s** at `gpu_precision = 48` (~27×).

A config is batchable unless: some model has no GPU kernel (none in V0.6), a
component *redshift* or `zs_fid` is optimizable, components sit on different lens
planes, or (point path) `hubble` is optimizable. Otherwise GLADE prints a
`[warn] batched GPU objective unavailable (<reason>)` and falls back to per-candidate
GPU evaluation — correct but usually no faster than the CPU pool, so consider the CPU
rail then. Watch the log line `GPU-batched objective active …` to confirm you got the
fast path.

- **`gpu_precision`** (§6.9): 64 = fp64 (bit-parity with the legacy path); **48 =
  recommended** — fp32 for the expensive deflection-field phase, fp64 for the Newton
  refinement and magnifications; measured indistinguishable from fp64 (losses agree
  to ~3 × 10⁻¹⁵) at ~2.7× the speed; 32 = fp32 everywhere — fine for exploration, but
  near-critical candidates can show relative loss noise up to ~10⁻⁴–5 × 10⁻².
- **`GLADE_GPU_CHUNK`** (environment variable): candidates per CUDA pass in the
  generalized mode. Default 32 (Schramm-heavy models: nfw/king/sers/hern/pow/gnfw/
  tnfw/ein) or 128 (cheap models), doubled at fp32 field precision. Raise for ~10 %
  more speed if you have VRAM headroom; **lower it if you hit CUDA out-of-memory**.
- ⚠️ A batched-GPU failure (e.g. OOM) does **not** stop the run: the whole population
  scores 10¹⁵ and DE keeps "running" without optimizing. The only signal is a single
  `[warn] batched GPU objective failed …` line — if your loss sits at 1e15, read the
  log.
- GPU runs are always single-process (`DE_WORKERS` ignored).

### 7.3.3 The Glafic rail (amoeba)

The WebUI **Glafic** rail (CLI: `--mode amoeba`) runs the standalone glafic binary's
own **downhill-simplex ("amoeba") optimizer** — a local optimizer starting from your
representative values, not a global search. Use it to cross-check a DE result with a
completely independent code path, to run existing glafic `.input` files unchanged, or
when you want exactly glafic's χ² definition.

- GLADE `.dat` selections are converted in the job directory into
  `amoeba_model.input` + `amoeba_obs.dat` (constraints) + `amoeba_prior.dat`
  (parameter ranges); every `{lo, hi}` becomes a free flag + range prior, starting
  from the geometric (mass) / arithmetic mean. `DE_*`/`MCMC_*` keys don't apply.
- Native `.input` selections are *staged* (copied with every referenced data file)
  into the job dir and run verbatim; if the input has no `optimize` command it just
  runs whatever commands it contains.
- Requirements: point-source configs only (extend refused), at least one `{lo, hi}`,
  observation arrays present. Wall-clock limit: env `GLAFIC_AMOEBA_TIMEOUT`
  (default 3600 s; 0 = unlimited) — glafic's simplex can stall on pathological
  models.
- Outputs: `chi^2` and image list in `best_params.txt`; glafic's own
  `<prefix>_optresult.dat` holds the fitted parameters (§8.3).

## 7.4 MCMC: what it is and how GLADE uses it

DE gives you a single best fit. **MCMC** (Markov-chain Monte Carlo) estimates the
*posterior distribution* — which parameter values are consistent with the data and
with what probability — so you can quote uncertainties and see correlations.

GLADE uses **emcee**, an *ensemble* sampler: instead of one chain, a swarm of
**walkers** moves through parameter space; each step, every walker proposes a move
constructed from the positions of other walkers ("stretch move") and accepts or
rejects it based on the likelihood ratio. After enough steps the walker positions are
samples from the posterior.

Key mechanics in GLADE:

- **Prior** = the uniform box spanned by your `{lo, hi}` bounds. Always. There are no
  separate prior settings (the pre-V0.4.1 prior keys are ignored with a warning).
  Mass dimensions are sampled in log₁₀.
- **Likelihood** = `exp(−loss/2)` with exactly the DE loss (§7.2 / §7.6), so DE and
  MCMC see the same landscape.
- **Burn-in** (`MCMC_BURNIN`): the first steps, while walkers migrate from their
  starting points, are discarded. **Thinning** (`MCMC_THIN`): keep every N-th step
  afterwards to reduce autocorrelation. Final sample count ≈
  `nwalkers × (nsteps − burnin) / thin` (defaults: 32 × (2000−300)/2 = 27 200).
- Walker count is floored at `2·ndim + 2` regardless of your setting.

## 7.5 The three ways to run MCMC

### 7.5.1 `de+mcmc` (recommended)

Set `MCMC_ENABLED = True` and run the **CPU** or **GPU** rail: DE runs first, then
walkers start in a small Gaussian ball around the DE best
(`MCMC_PERTURBATION` × bound width per dimension) and explore outward. The corner
plot marks the DE best with red lines. This is the robust default — the sampler
starts where the likelihood is high.

### 7.5.2 MCMC-only (the `MCMC` rail)

Walkers start *uniformly* over the prior box. Honest but dangerous in high
dimensions: if hardly any random point yields a valid model (e.g. wrong image count
⇒ likelihood 0), initialization struggles. GLADE re-draws infeasible walkers up to a
budget and then clones feasible ones (with a warning); if *no* feasible walker is
found the run aborts with a clear message telling you to use `de+mcmc` or narrow the
ranges. After an MCMC-only run, `result.png` shows the *posterior-median* model.

### 7.5.3 MCMC-GPU and walker auto-tuning

The `MCMC-GPU` rail evaluates the ensemble likelihood in batched CUDA calls
(emcee updates half the ensemble per call). Since 32 walkers leave a GPU ~97 % idle,
GLADE **auto-raises the walker count to 1024** when *all* of: you left
`MCMC_NWALKERS` unset, the config is batchable, and it is either an extend run or a
pure point-mass ("legacy") point run. On the generalized (chunked) path the default
stays 32 with an explanatory note, because there extra walkers cost proportionally.
An explicitly set value is never overridden (you just get a `[hint]` if it underuses
the GPU). All decisions are printed in the log, and the defaults dialog shows the
auto-raised value beforehand.

### 7.5.4 Judging MCMC health

- **Acceptance fraction** (printed live and in the summary): healthy emcee runs sit
  around **0.2–0.5**. Near 0 ⇒ walkers are stuck (ranges too wide, model too
  fragile); near 1 ⇒ steps too timid to explore.
- **Trace plot** (§8.6): after the red burn-in line, each parameter's traces should
  look like flat "fuzzy caterpillars". Drift or split bands ⇒ burn-in too short or a
  multimodal posterior.
- GLADE computes **no autocorrelation time** and only the extend path warns
  automatically at acceptance < 0.01 — judging convergence is on you.
- Runtime ballpark: point-mass models sample in minutes on a pooled CPU;
  Romberg-heavy profiles (Sersic/NFW…) take tens of minutes; extended-source MCMC on
  CPU is *much* slower (one full glafic evaluation per walker per step) — keep
  `MCMC_NSTEPS` modest or use `de+mcmc` with the GPU.

## 7.6 Extended-source fitting

When the config sets `extended_file` (an observed FITS image) or contains any `ext*`
component, the run switches to the **extended-source path**: instead of matching
point-image lists, glafic ray-traces the model source through the lens, renders the
lensed image on the pixel grid, and χ² includes a per-pixel term.

- The loss is glafic's `c2calc` broken into components, each weighted by your `W_*`
  keys: `loss = W_POS·pos + W_FLUX·flux + W_TD·td + W_EXT·pixel + W_PRIOR·(priors)
  + penalty`. All weights 1.0 ⇒ exactly glafic's native χ². The best-fit component
  breakdown is printed and stored (`status.json → components`).
- The FITS image must match the grid implied by `xmin/xmax/ymin/ymax/pix_ext`
  exactly (e.g. a ±0.5″ field at `pix_ext = 0.01` needs a 100×100 image) — a
  mismatch is an immediate error. The image must also be a **32-bit float** FITS
  (BITPIX −32): other pixel types abort with glafic's
  `input obs fits must be float` — convert with `data.astype(np.float32)` before
  writing.
- Point-source constraints (e.g. a lensed supernova in front of the host ring) enter
  either through the four obs arrays (converted internally to a temporary glafic
  constraint file) or a glafic-native `constraint_file` (which wins if both exist).
  `source_x/y` are solved *internally* by glafic per candidate — they are not DE
  dimensions in this mode.
- `hubble = {lo, hi}` **is** meaningful here (time delays scale as 1/h) — this is the
  time-delay-cosmography path.
- Backends: CPU drives glafic per candidate (multi-process); GPU batches the whole
  population (fp64 always) when eligible — with point constraints this additionally
  requires `chi2_splane = 1` (source-plane point χ²). The Glafic/amoeba rail refuses
  extend configs.
- MCMC works on the extend path too (see the speed warning above).

## 7.7 Multiprocessing notes

- Linux/WSL2 uses `fork` worker pools; macOS uses `spawn` (identical results — the
  DE trajectory is seed-deterministic).
- `DE_WORKERS = -1` (all cores) is the default and safe for DE.
- `MCMC_WORKERS` defaults to **1** deliberately: an all-core fork pool whose parent
  dies in a *detached/background* terminal can leave orphaned workers spinning at
  full CPU. Set `MCMC_WORKERS = -1` only when running in a real foreground terminal
  (the WebUI's spawned terminal windows qualify — but if your jobs run in the
  detached fallback mode, keep it at 1).

## 7.8 Verification: `glafic_verified` and the scipy-exact reference

With `glafic_verified = True` (the default), every DE/MCMC-median result is
independently re-checked, and the log states clearly:
`(verification is informational — the result above is unchanged)`.

1. **glafic-binary cross-check**: the best-fit model is written to
   `glafic_verify.input` and solved by the *standalone binary* (a code path
   independent of the Python bindings and of the GPU). Reported: glafic's image
   count, its loss vs the observations, and the maximum image-position offset in mas.
   - A relative loss difference above 50 % triggers a warning that explains itself:
     for Sersic-like profiles the *binary's* deflection is Romberg-tolerance-limited
     (Chapter 10) — the difference is expected there and does **not** mean your fit
     is wrong. A `glafic found N image(s); the result assumes M` warning usually
     means a marginal extra image near a critical curve.
2. **scipy-exact reference (ground truth)**: exact deflections at the observed
   positions (Sersic via adaptive quadrature at 10⁻¹¹ tolerance; other models via
   fp64 Rhongomyniad kernels). Reported: the Rhongomyniad-kernel Sersic deflection
   error vs the exact integral (expect ~10⁻⁹ arcsec; this check always evaluates the
   fp64 GPU kernels, whichever backend ran the fit — the CPU engine's own Sersic
   deflection is Romberg-limited, see Chapter 10), the **source-plane
   self-consistency scatter** in mas (how
   well the observed images map back to a single source under the exact model — the
   figure of merit for the fit's physical consistency), and back-projected vs fitted
   source position. Requires torch; skipped with an `[info]` line otherwise.

Extended-source runs verify analogously (`glafic_extverify.input`, binary `c2calc`
total vs GLADE's, warn at > 5 % relative difference).

Standalone deep checks live in `tools/verify_gpu_models.py` (every GPU kernel vs
glafic) and `tools/verify_gpu_precision.py` (the 64/48/32 tiers vs scipy-exact).

---

# 8 Understanding run outputs

## 8.1 The run directory

Every run — WebUI or CLI — writes into one directory, `runs/<job_id>/`:

| File | When | Content |
|---|---|---|
| `job.log` | WebUI runs | the complete terminal transcript (the WebUI's terminal wrapper `tee`s it; headless CLI runs must redirect stdout themselves) |
| `status.json` | always | machine-readable state + results (§8.2) |
| `best_params.txt` | DE / extend / amoeba | best-fit parameters (§8.3) |
| `result.png` | on success | the triptych (point runs, §8.4), the observed/model/residual figure (extend runs, §8.5), or the amoeba triptych |
| `mcmc_corner.png`, `mcmc_trace.png`, `mcmc_summary.txt` | when MCMC ran | posterior figures + percentile table (§8.6) |
| `iterations/iteration_%04d.png` | `Draw_Graph = 1` (default) | DE population corner frames (§8.7) |
| `best_crit.dat` / `best_ext_crit.dat` | figure rendering | glafic critical-curve segments for the figure |
| `glafic_verify.input`, `glafic_verify_point.dat` (extend: `glafic_extverify*`) | `glafic_verified = True` | verification artifacts (§7.8) |
| `amoeba_model.input`, `amoeba_obs.dat`, `amoeba_prior.dat`, `<prefix>_optresult.dat`, `<prefix>_point.dat`, `<prefix>_crit.dat` | Glafic rail | the converted/staged glafic inputs and glafic's own outputs |

The last line of a successful `job.log` is `RUN_COMPLETE`.

## 8.2 `status.json`

Fields accumulate as the run progresses: `state`
(`starting → running → done | error`; the WebUI additionally synthesizes
`interrupted` when the worker process died, and `unknown` after a server restart),
`backend`, `mode`, `files`, `worker_pid`; after DE: `loss`, `iterations`,
`triptych`, `fitted` (a `label → physical value` map — masses here are **linear**);
after MCMC: `mcmc {acceptance, n_samples, corner, trace, summary}`; extend runs add
`c2calc_total` and the 8-component breakdown `components {pos, flux, td, prior_pt,
pixel, prior_ext, prior_lens, penalty}`; verification adds `glafic_verify` +
`scipy_reference` (or `extend_verify`).

## 8.3 `best_params.txt` (and where the numbers live)

Point-source DE:

```
# GLADE DE result  backend=gpu  loss=0.50547131
point1.mass = 1558.908622
point1.x = 0.2819156363
...
```

One `label = value` line per fitted dimension, in **physical units** (masses linear,
not log). Extend runs add the `c2calc` total and component breakdown as comments.
Amoeba runs are different: `best_params.txt` holds only `chi^2 = …` and the image
list — the fitted lens parameters are in glafic's own `<prefix>_optresult.dat`.
MCMC-only runs write no `best_params.txt` at all (see `mcmc_summary.txt`).

> ⚠️ **Log vs linear masses.** `best_params.txt` and `status.json → fitted` store
> masses in M☉ (linear). `mcmc_summary.txt` and the corner-plot axes are in the
> *search space*, i.e. **log₁₀(mass)**; the linear median is available as
> `p50_linear` inside `status.json`. Don't mix them up when comparing.

## 8.4 `result.png` — the triptych (point runs)

Three panels; the title carries the loss (`GLADE DE result loss=…`,
`MCMC posterior-median model`, or `glafic amoeba result chi²=…`).

**Left — "Position residuals".** One bar per observed image: the distance ΔPos [mas]
between the observed position and the model's matched image. The blue dashed line is
your 1σ position error (per-bar segments if σ differs per image). A good fit keeps
every bar below its 1σ line.

**Middle — "Magnification".** Per image, three series: sky-blue bar = observed |μ|
with its error bar; hatched green bar = |μ_pred|, the model magnification at the
model's predicted image position; red dot = |μ@obs|, the model magnification
evaluated *exactly at the observed position*. Near a critical curve μ is
hypersensitive to position, so |μ@obs| may differ wildly from |μ_pred| even for a
good model — compare the green bars to the blue ones first, and treat red dots as a
sensitivity diagnostic. With `abs_mag = False` everything is signed instead
(negative bars = saddle-parity images).

**Right — "Image plane".** Gold stars = observed images (numbered); red × = model
images; blue curves = **critical curves** (image-plane loci of formally infinite
magnification); green curves = **caustics** (their source-plane counterparts, drawn
in the same axes); red diamonds = **sub-halo markers** with parameter labels
(`S1: 1.0e+06 …` — the component's mass and shape parameters).

Which components get sub-halo markers: those classified as substructure — by the
`Nl`/`Ns` index suffix if you gave one (§6.3), else by schema category, and any
component with an optimizable parameter is marked by default.

If the best-fit model cannot reproduce the observed image count, the figure is
skipped with `[warn] triptych failed …` — the run still counts as `done`.

## 8.5 `result.png` — extended-source figure

Three panels: **Observed** (your FITS) and **Model (best fit)** (the lensed model
image) share one brightness scale; **Residual (model − obs)** uses its own symmetric
diverging scale (± max |residual|) — structure in the residual panel is what your
model fails to explain. Critical curves
are overlaid in cyan. The title's second line lists the χ² components. If the
observed FITS cannot be read, only the model panel is drawn.

## 8.6 MCMC figures and summary

**`mcmc_corner.png`** — the *corner plot*: an N×N matrix over all fitted dimensions;
the diagonal shows each parameter's 1-D posterior histogram (with dashed lines at the
16th/50th/84th percentiles — median ± 1σ); each off-diagonal panel is the joint 2-D
distribution of a parameter pair, where tilted/curved contours reveal parameter
degeneracies (e.g. mass–concentration). Mass axes are labelled `log10(...)`. In
`de+mcmc` runs, red lines mark the DE best — they should sit inside the bulk of the
posterior. (Plots are down-sampled to 40 000 points above that count.)

**`mcmc_trace.png`** — one row per parameter; every walker's value vs step number,
with the red dashed line at the end of burn-in. Healthy: flat, well-mixed "fuzzy
caterpillars" after the line. Trending or split traces mean the chain has not
converged — raise burn-in/steps or narrow the ranges. (Display is capped at 2000
steps / 256 walkers.)

**`mcmc_summary.txt`** — header with backend and acceptance fraction, then one line
per parameter: `name = p50 [p16, p84]` — the posterior median and the ±1σ interval,
**in search space (log₁₀ for masses)**.

A degenerate chain (acceptance ≈ 0) may fail to produce a corner plot; the run
continues with `[warn] corner plot skipped: …` — treat such a run's "posterior" as
invalid (§7.5.4).

## 8.7 Iteration frames

With `Draw_Graph = 1` (default), every `draw_interval`-th DE generation writes
`iterations/iteration_%04d.png`: a corner-style scatter of the *whole DE population*
over every pair of dimensions, coloured by loss (per the colorbar — lower is
better), the current best marked `+`. Panels pairing a component's x/y dimensions overlay the observed image
positions as gold stars — you can watch sub-halo candidates cluster onto the images
as generations pass. The WebUI does **not** display these; browse
`runs/<id>/iterations/` on disk. Set `Draw_Graph = 0` for a small speedup
(~2.7 s/frame at 19 dims).

---

# 9 Command line & Python library

Everything the WebUI does is scriptable. Three levels: the headless job runner
(§9.1), the `import glade` library (§9.2–9.3), and the raw engines (§9.4).

## 9.1 Headless runs: `webui/runjob.py`

The WebUI's worker is directly usable:

```bash
source env.sh
python webui/runjob.py \
  --backend cpu \                     # cpu | gpu | glafic
  --mode findimage \                  # findimage | de+mcmc | mcmc | amoeba
  --out runs/manual_cpu \
  --files core/examples/constants.dat \
          core/examples/images_data.dat \
          core/examples/lens_and_substructure.dat \
  --force
```

- `--mode findimage` = DE only; `de+mcmc` = DE then MCMC; `mcmc` = MCMC only;
  `amoeba` = glafic's simplex (conventionally paired with `--backend glafic` as in
  the WebUI mapping — the flag isn't enforced; amoeba mode always drives the glafic
  binary).
- The WebUI rails map onto these: CPU/GPU → `findimage`/`de+mcmc` (per
  `MCMC_ENABLED`), MCMC → `--backend cpu --mode mcmc`, MCMC-GPU →
  `--backend gpu --mode mcmc`, Glafic → `--backend glafic --mode amoeba`.
- Exit code 0 on success (`RUN_COMPLETE`), 2 on any blocked/error condition. Output
  files are as in Chapter 8 — except `job.log`, which comes from the WebUI's `tee`
  wrapper: pipe stdout yourself if you want it
  (`… | tee runs/manual_cpu/job.log`). (`--force` is accepted for API symmetry;
  defaults are applied without confirmation in CLI runs.)
- `source env.sh` first (or replicate its `PYTHONPATH`/`LD_LIBRARY_PATH`).

## 9.2 `import glade` — the library facade

With the repo root on `sys.path` (automatic when your working directory is the repo,
or from anywhere after `source env.sh`):

```python
import glade

cfg, issues = glade.load_config(
    ["InputFiles/constants.dat", "InputFiles/images_data.dat", "InputFiles/lens.dat"],
    backend="gpu")
assert not glade.has_errors(issues)

result = glade.optimize(cfg, backend="gpu")            # DE fit
print(result.loss, result.fitted)                      # physical values

obs = glade.build_obs(cfg)
glade.make_triptych(result, obs, "triptych.png")       # result figure

from core.optimize.loss import LossConfig     # importable once glade is imported
mres = glade.run_mcmc(result.problem, obs, LossConfig.from_cfg(cfg),
                      backend="gpu", best_x=result.x,
                      mcmc_cfg=glade.MCMCConfig.from_cfg(cfg))
glade.plot_mcmc(mres, "out_dir")

report = glade.verify_with_glafic(result.scene, obs, "out_dir", opt_loss=result.loss)
```

`import glade` bootstraps `sys.path` for the whole tree (afterwards `import core`,
`import glafic`, `import rhongomyniad` all work) and is *lazy*: heavy modules
(matplotlib, emcee, torch, the glafic C extension) load only on first use.

Function reference (the essentials):

| Function | Notes |
|---|---|
| `load_config(paths, backend=None, with_defaults=True) → (cfg, issues)` | parse + merge + default + validate; bad file *content* never raises (syntax/validation problems become issues — check `has_errors(issues)`), but a missing/unreadable path still raises `OSError`. `issues` print as `[error] file:line: message`. |
| `lint_text(text, …) → (cfg or None, issues)` | validate one in-memory document (what the Editor would run) |
| `optimize(cfg, backend="cpu", on_iteration=None, de_overrides=None, base_dir=None) → OptResult` | the DE fit; auto-routes to the extended-source path for extend configs (`base_dir` resolves relative FITS paths). `de_overrides` takes *DEConfig attribute names* (`{"maxiter": 300, "seed": 7}`) and wins over the `.dat`. Raises `ValueError` when nothing is optimizable. |
| `OptResult` | `.x` (best vector, **search space** — masses log₁₀), `.loss`, `.fitted` (label → physical), `.scene`, `.problem`, `.de` (history), `.mode` (`"point"`/`"extend"`), extend: `.extend_components` |
| `build_obs(cfg) → ObsData` | observations converted to engine units (arcsec, flip/offset applied) |
| `make_triptych(result, obs, output_file, …)` / `make_extend_figure(result, output_file, …)` | figures of Chapter 8; `make_triptych` raises if the image count can't be matched; both write small `best*` glafic helper files next to the output |
| `run_mcmc(problem, obs, loss_cfg, backend="cpu", best_x=None, mcmc_cfg=None) → MCMCResult` | `best_x=None` ⇒ MCMC-only (uniform init); `best_x=result.x` ⇒ de+mcmc. `MCMCResult`: `.samples`, `.chain`, `.acceptance_fraction`, `.param_names`, `.summary` (p16/p50/p84 (+`p50_linear`)). |
| `plot_mcmc(mres, out_dir) → {"corner": path, "trace": path}` | writes both figures (a key is missing if that plot failed) |
| `verify_with_glafic(scene, obs, out_dir, opt_loss=…) → dict` / `verify_extend(result, out_dir)` / `reference_check(scene, obs)` | §7.8; never raise — check `["ok"]` |
| `engine(name)` | §9.3 |

Caveat for library users: `load_config(..., with_defaults=True)` is what gives you
the documented defaults; the lower-level `DEConfig.from_cfg` fallback for a missing
`DE_WORKERS` is 1 (single process), whereas WebUI runs get −1 (all cores) from the
defaults table.

## 9.3 The imperative engine API (`glade.engine`)

`glade.engine("cpu")` returns the glafic C extension; `glade.engine("gpu")` returns
`rhongomyniad`. Both expose the same glafic-style API, so exploratory code ports
between them by changing one line:

```python
eng = glade.engine("cpu")                     # or "gpu"
eng.init(0.3, 0.7, -1.0, 0.7, "out", -5.0, -5.0, 5.0, 5.0, 0.2, 3.0, 5, verb=0)
eng.startup_setnum(1, 0, 1)                   # 1 lens, 0 extended, 1 point source
eng.set_lens(1, "sie", 0.5, 300.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0)
eng.set_point(1, 2.0, 0.1, 0.05)
eng.model_init(verb=0)
images = eng.point_solve(2.0, 0.1, 0.05, verb=0)   # [(x, y, mag, td_days), ...]
eng.quit()
```

Ground rules (both engines): indices are 1-based; **call `model_init()` again after
every parameter change**; the module is a global-state singleton (one model per
process); `init(...)` resets everything. `point_solve` returns signed magnifications
and time delays in days (relative to the earliest image). `calcimage(zs, x, y)`
returns the 8-tuple `(αx, αy, td, κ, γ1, γ2, μ⁻¹, rot)`.

glafic-only extras include the optimizers (`optimize`, `optpoint`, `optextend`),
`writecrit`/`writelens`/`writemesh`, `calcein_i`, `kappa_ave`/`kappa_cum`,
coordinate converters, and `c2calc`/`c2calc_each`. Two glafic binding quirks: pass
`init` arguments positionally (the first keyword is misspelled `omgea` upstream),
and index-out-of-range errors in some calls **terminate the whole Python process**
rather than raising.

## 9.4 Using Rhongomyniad directly

`import rhongomyniad as rh` gives the GPU engine standalone (see
`Rhongomyniad/examples/` and `tests/test_smoke.py`):

- `rh.supported_models()` — all 27 glafic models in V0.6. The single structural
  limit: **one lens plane** (lens redshifts differing by > 10⁻⁶ make `model_init()`
  raise).
- Devices/precision: `rh.set_device("cuda"/"cpu")` (silent CPU fallback — check
  `rh.get_device()` if it feels slow), `rh.set_dtype(torch.float64/float32)`.
- Image finder: `rh.set_finder("adaptive")` (default; GPU port of glafic's quad-tree)
  or `"uniform"` (dense grid at the finest level — a reference mode; catastrophically
  slow for NFW-like models). If an image very close to a critical curve is missed,
  raise `maxlev`.
- Tensor-batched parameters: any physical parameter may be a torch tensor
  broadcastable against the grid (that's what GLADE's batched objectives use).
  Redshifts and `zs_fid` must stay Python floats. Note that **batched parameters skip
  the validity checks** — the caller must pre-filter garbage.
- `gals` catalogues: glafic-compatible — a `galfile.dat` (rows `x y L [e pa]`) in the
  *current working directory*, loaded lazily at first use; or explicit
  `rh.set_galfile(path)` / `rh.set_gals(rows)` / `rh.readgals()`. Catalogue values
  are deliberately quantized to float32 for bit-parity with glafic.
- First use of `gnfw`/`ein` may rebuild a lookup table (one-time delay) if the
  shipped cache under `Rhongomyniad/rhongomyniad/_tab_cache/` was deleted; the
  `acnfw` CSE table cannot be rebuilt — don't delete it.
- Not implemented (raise): `set_psf`/PSF convolution, noise mocks in `writeimage`
  (which, unlike glafic, returns a numpy array instead of writing FITS), optimizable
  point-source redshift.

## 9.5 The translator: `python -m core.translate.cli`

```bash
# glafic → GLADE
python -m core.translate.cli to-glade some_model.input -o InputFiles/imported
# (also accepts a Python driver script that calls glafic.* functions)

# GLADE → glafic (multiple .dat merge exactly like a run; -o is a path PREFIX)
python -m core.translate.cli to-glafic \
  InputFiles/constants.dat InputFiles/lens.dat InputFiles/images_data.dat \
  -o exported/run
```

`to-glade` writes `<base>_model.dat` (+ `<base>_obs.dat`). Conversion rules to know:
glafic optimize-flags become **degenerate `{v, v}` bounds — widen them by hand**;
observed positions are converted arcsec → mas; **`obs_x_flip = False` and zero
`center_offset_*` are hard-coded** (fix them if your frame was sky-convention); an
extended-source input yields an extend-mode `.dat` with the `W_*` weights appended
and file paths as basenames (place the FITS files next to the `.dat` yourself).

`to-glafic` writes `<prefix>_model.input`, and when anything is optimizable also
`<prefix>_obs.dat` + `<prefix>_prior.dat` with an `optimize` command wired in — an
input you can run with the plain glafic binary from that directory. Bounds collapse
to representative values (geometric mean for masses); shared variables become glafic
`match` ties; `hubble` bounds become `hvary 1` + a range prior. Numbers are written
at `%.6e` precision.

## 9.6 Legacy entry points (`main.py`, `tools/`)

`./run_glade.sh` → `main.py` is the pre-V0.4 workflow: you edit `model_use`
(`point_mass`/`nfw`/`king`/`p-jaffe`/`none`) and override dicts inside `main.py`,
and it re-runs the archived scripts under `legacy/`, writing into `results/`. The
helper scripts in `tools/` (`run_glafic.py`, `drawgraph.py`, `mcmc_from_result.py`,
`MCMC_GPU.py`, `replot_mcmc.py`, `glafic_verify.py`) all expect that **legacy**
`results/` layout (`*_best_params.txt` with model-specific headers, iPTF16geu
hard-codings) — **they cannot read the modern `runs/<id>/` output**, and modern runs
don't need them (figures, MCMC and verification are built in). The two exceptions
that do target the modern pipeline: `tools/verify_gpu_models.py` and
`tools/verify_gpu_precision.py` (§7.8). `tools/inverse_cal.py` is a self-contained
forward calculator (edit its CONFIG block) that is occasionally handy for quick
what-if imaging.

---

# 10 Numerical accuracy: the `TOL_ROMBERG_JHK` note

This chapter exists because one compile-time constant in glafic has caused more
confusion than any other numerical setting in this project. Read it if you fit
elliptical profiles without closed-form deflections, render extended images, or make
claims at the milliarcsecond level.

## 10.1 What it is

For elliptical lens models without analytic deflections, glafic evaluates the
Schramm (1990) **I/J/K line-of-sight integrals** numerically with Romberg
integration. `TOL_ROMBERG_JHK` is that integration's relative-error tolerance — a
`#define` in `glafic2/glafic.h`:

```c
/* glade local override: tightened from upstream 5.0e-4 for accuracy */
#define TOL_ROMBERG_JHK 1.0e-5
```

- **Upstream glafic ships 5·10⁻⁴. GLADE's bundled build ships 1·10⁻⁵** (30–50×
  more accurate, still much faster than the 10⁻⁸ used briefly in V0.3).
- Affected models (everything that goes through the J/K integrals): **`nfw`,
  `gnfw`, `hern`, `pow`, `sers`, `tnfw`, `ein`, `king`**. *Not* affected: `sie`,
  `point`, `anfw`, `acnfw`, the `…pot` variants and other closed-form/table models.
- The **GPU engine is entirely unaffected** — Rhongomyniad uses fixed-node
  Gauss–Legendre quadrature and, for these profiles, is *more* accurate than the
  glafic binary (its Sersic deflection matches a scipy-exact integral to ~10⁻⁹ ″).

## 10.2 Why it matters: the stripe artifact

At the upstream tolerance, the Romberg error is not smooth — it *jumps* at internal
refinement boundaries, i.e. it is a position-dependent step error. Two GLADE-relevant
consequences (both reproduced and quantified in `exception/stripe_repro/`):

1. **Dark stripes in extended images.** In `writeimage`-style lensed images of an
   NFW+Sersic system, glafic's finite-difference magnification amplifies the
   deflection step error ~500–1000×, rendering **vertical dark lines cutting through
   the lensed ring** (an "|o|" pattern). Measured stripe depth: **11.5 %** of ring
   brightness at 5·10⁻⁴ (clearly visible), **0.4 %** at GLADE's 1·10⁻⁵ (invisible).
   An aggravating factor: glafic's Romberg routine has its convergence check disabled
   (16 refinement levels, silent on non-convergence), so nothing warns you.
2. **Point-source positions near critical curves.** The deflection error is
   multiplied by |μ|. Near-critical images computed at 5·10⁻⁴ can be off by up to
   ~5 mas (and noticeably in magnification); at 1·10⁻⁵ they are accurate to roughly
   the µas–mas transition; at 10⁻⁸ they match exact references essentially perfectly.

## 10.3 What this means for you in practice

- **Using the bundled build (default): no action needed.** The 1·10⁻⁵ compromise
  makes the stripes invisible and point positions accurate for normal work.
- **You will see the artifacts if** you run a *stock* upstream glafic side by side
  (its extended images genuinely differ from GLADE's — that's the stripe, not a
  GLADE bug), or if you loosen the tolerance for speed.
- **Verification warnings are expected for Sersic-heavy models.** With
  `glafic_verified = True`, the cross-check may report a large relative loss
  difference along with the explanation that glafic's Sersic deflection is
  "Romberg-tolerance-limited … this difference is expected and is NOT a result
  error". Trust the *scipy-exact* section printed right after it — that one is the
  ground truth (§7.8).
- **When even 1·10⁻⁵ is not enough** (µas-level positional claims very close to
  critical curves): prefer the **GPU backend** — its quadrature is
  tolerance-independent and scipy-verified — rather than rebuilding glafic at 10⁻⁸
  (which costs ~2–4× in speed on the affected integrals).

## 10.4 Changing it (rarely needed)

There is **no `.dat` key, environment variable, or API** for this constant. To
change it: edit `glafic2/glafic.h`, then **rebuild** — `cd glafic2 && make clean &&
make all` (an edited header without a rebuild silently changes nothing; a stale
`glafic.so` once shipped with the old value for exactly this reason). The
documented protocol used for GPU cross-verification: set `1.0e-8`, `make python`,
run `tools/verify_gpu_models.py --tol 2e-7`, then restore `1.0e-5` and rebuild.

Related but distinct tolerances you may spot in `glafic.h` (unchanged from
upstream): `TOL_ROMBERG_GNFW 3.0e-4`, `TOL_ROMBERG_EIN 1.0e-3` (radial profile
tables), `ULIM_JHK 1.0e-8` (integral lower cutoff).

> Documentation caution: `Rhongomyniad/README.md` and
> `Rhongomyniad/rhongomyniad/constants.py` still quote `TOL_ROMBERG_JHK = 5e-4` —
> both are stale mirrors with no effect; the built value is `1.0e-5` in
> `glafic2/glafic.h`.

---

# 11 Troubleshooting & FAQ

**Install & startup**

- *A dependency tarball fails to download* → the script names the file and the
  directory; download it manually into `deps/src/` and re-run the bootstrap.
- *`import glafic` fails after recreating the venv or upgrading Python* → the
  `.pth` registration is gone; re-run the bootstrap (it regenerates
  `glafic_glade.pth`).
- *My terminal dies after `source env.sh` when a command fails* → `env.sh` sets
  `set -e` in your shell (§3.1); run `set +e` after sourcing.
- *WebUI port already in use* → `GLADE_PORT=8080 ./run_webui.sh`.

**Running jobs**

- *`[blocked] the GPU backend needs PyTorch … importing torch failed`* → install a
  CUDA PyTorch wheel into the venv (§3.3), or use the CPU/MCMC rails.
- *`[blocked] no optimizable {lo,hi} parameters.`* → everything in your `.dat` is
  locked (common right after a glafic import, which produces `{v, v}` degenerate
  bounds — widen them).
- *No terminal window appears* → no supported emulator found; the job still runs
  detached and streams to the browser. `sudo apt install gnome-terminal` (WSLg) to
  get windows. In detached mode, keep `MCMC_WORKERS = 1` (§7.7).
- *How do I stop a run?* → `Ctrl+C` in (or close) its terminal window; the UI will
  show `interrupted`. There is no stop button.
- *`stream error` in the terminal panel* → the browser lost the event stream; the
  job is unaffected. Reload won't re-attach; check the terminal window or
  `runs/<id>/job.log`.
- *After restarting the WebUI, my job shows `unknown` and figures won't load* → the
  job registry is in-memory. The files are all still in `runs/<id>/`; open them from
  the filesystem.
- *The defaults dialog lists keys I never heard of* → those keys fell back to
  built-in defaults; check §6.9 and remember several defaults are
  iPTF16geu-specific (§6.5) — set them explicitly.

**Fits that misbehave**

- *Loss is pinned at 1e15 and never improves* → every candidate is being rejected:
  ranges exclude any model with the right image count, or (GPU) the batched
  objective failed once and the log has a `[warn] batched GPU objective failed …`
  (OOM → lower `GLADE_GPU_CHUNK`). Consider `missing_img_penalty > 0` to give DE a
  gradient (§7.2).
- *`MCMC cannot start: every initial walker has zero likelihood`* → uniform
  initialization can't find a valid model in a high-dimensional box: run `de+mcmc`
  instead (walkers start at the DE best), or narrow the `{lo, hi}` ranges.
- *MCMC acceptance ≈ 0* → not a valid posterior (§7.5.4): seed from DE, narrow
  ranges, fewer dimensions.
- *`[warn] triptych failed: best-fit model produced N image(s) (expected M)`* → the
  run finished, but the best model has the wrong image multiplicity, so the figure
  is skipped. Usually the fit is simply bad — widen/adjust ranges — or a marginal
  image sits near a critical curve (raise `maxlev`).
- *glafic verification warns `glafic found 5 image(s); the result assumes 4`* → an
  extra faint image near a critical curve; informational (§7.8). (The
  similar-sounding `(expected M); skipping the result figure` is the separate
  triptych-skip warning covered above.)
- *A setting I added has no effect* → check its spelling: unknown keys silently
  become user variables (§6.6); check the `[defaults]` line in `job.log`.
- *GPU run slower than expected* → look for the `[warn] batched GPU objective
  unavailable …` line (§7.3.2) — an optimizable component redshift/`zs_fid` or a
  multi-plane setup disables batching; also confirm `torch.cuda.is_available()`.

**Editor & files**

- *I opened a FITS in the Editor and it's garbage* → binary files aren't editable;
  **don't save that tab** or the file on disk is corrupted.
- *Where do I upload files?* → no upload exists; copy into `InputFiles/` via the
  filesystem (`\\wsl$` from Windows), then `⟳` (§5.3.2).
- *I lost edits after a reload* → the Editor has no autosave and no unsaved-changes
  warning on page close; save early (§5.3.3).

**Clave**

- *Status shows `Mock mode`* → the glafic module isn't importable; CPU results are
  placeholders. Rebuild glafic (bootstrap) and restart the server.
- *`GPU错误: GPU mode requires all lenses on the same redshift plane…`* → the GPU
  engine is single-plane; put all lenses at one z or use CPU mode.
- *Expected images don't appear* → they may fall outside the auto-sized search box
  (§5.4.5) or the redshift you typed reverted (z = 0 is replaced by defaults).

---

# 12 Appendices

## Appendix A — Glossary

- **arcsec (″) / mas**: 1″ = 1/3600 degree; 1 mas = 10⁻³ ″. GLADE writes observed
  image positions in mas, nearly everything else in arcsec.
- **burn-in**: the initial MCMC steps discarded because walkers are still migrating
  from their start positions toward the posterior.
- **caustic**: the source-plane curve mapped from a critical curve; a source crossing
  a caustic changes its number of images.
- **χ² (chi-squared)**: sum of squared, error-normalised residuals; the building
  block of all GLADE loss functions.
- **corner plot**: a matrix of all 1-D histograms (diagonal) and 2-D joint
  distributions (off-diagonal) of the fitted parameters; the standard way to display
  a multi-dimensional posterior.
- **critical curve**: image-plane locus where magnification formally diverges
  (det J = 0); images near it are extreme and sensitive.
- **DE (Differential Evolution)**: population-based global optimizer; §7.1.
- **ellipticity e / position angle pa**: shape parameters of an elliptical profile;
  e ∈ [0, 1), pa in degrees.
- **emcee / walkers / stretch move**: the affine-invariant ensemble MCMC sampler
  GLADE uses; a *walker* is one member of the sampling swarm.
- **Hungarian matching**: the optimal one-to-one assignment (here: observed ↔
  predicted images minimising total distance).
- **lens plane / multi-plane**: single-plane = all deflectors at one redshift;
  configurations with deflectors at several redshifts are multi-plane (CPU-only in
  GLADE).
- **log₁₀ search**: mass-like parameters are optimized/sampled as their base-10
  logarithm, since they span orders of magnitude.
- **loss**: the scalar the optimizer minimises; GLADE's is a weighted χ² plus
  penalties (§7.2, §7.6).
- **magnification μ / parity**: flux amplification of an image; its sign encodes
  parity (mirror orientation) — negative for saddle-point images.
- **MCMC**: Markov-chain Monte Carlo — random-walk sampling whose long-run
  distribution is the posterior; §7.4.
- **posterior**: the probability distribution of parameters given the data (and the
  prior); what MCMC estimates.
- **prior**: what you assume before the data; in GLADE always the uniform `{lo, hi}`
  box.
- **Romberg integration**: an iterated-refinement numerical integration scheme;
  glafic uses it for elliptical-profile integrals (Chapter 10).
- **Schramm (1990) integrals**: the I/J/K line integrals expressing an elliptical
  mass distribution's potential/deflection/Hessian.
- **time delay (td)**: arrival-time difference between images [days]; scales with
  1/h, hence time-delay cosmography (§7.6).
- **triptych**: GLADE's three-panel result figure (§8.4).

## Appendix B — Version history (condensed)

| Version | Highlights |
|---|---|
| 0.1.0 | prototype: glafic + DE |
| 0.2.x | first bilingual WebUI; parameter editing in the browser |
| 0.3.0 | Rhongomyniad GPU engine (beta), Clave first appears, glafic verification tools |
| 0.4.0 "ReUnit" | unified `core/`, the `{lo, hi}` `.dat` format, rewritten `webui/` (FindImage + Editor, Monaco, templates), glafic↔GLADE translation |
| 0.4.1 | batched GPU DE/MCMC; MCMC-only rail; MCMC priors unified to the DE bounds |
| 0.4.2 | `glafic_verified` independent verification + scipy-exact reference |
| 0.4.3 | bundled glafic synced to upstream 2.1.14 (King → model #27; local mods preserved) |
| 0.4.4 | extended-source (FITS) fitting via `c2calc_each`; `W_*` weights; optimizable `hubble`; `missing_img_penalty` |
| 0.4.5 | macOS installer + fork/spawn multiprocessing layer (untested on real Macs) |
| 0.5.0 | GPU/CPU full parity (24 tensor kernels, extend pipeline on GPU); generalized whole-population batching (~27×); `gpu_precision` 64/48/32; user-defined shared variables; `abs_mag`; `Nl`/`Ns` suffixes; MCMC-GPU rail with walker auto-tuning |
| 0.5.1 | repository cleanup |
| 0.5.2 | Glafic rail = native amoeba; optimize-ready glafic export (`setopt` + `optimize` + constraint/prior files; shared variables → `match` ties) |
| 0.5.3 | `.dat` arithmetic + observed-image-position expressions (`img1_x ± …`) |
| 0.6.0 | 27/27 lens models on GPU (`crline`, `acnfw`, `gals`); Clave merged as the third tab; `import glade` library facade; WebUI dark/light theme + EN/中文 toggle; Explorer copy/paste; Images panel in Clave |
| 0.6.0-GREY | **bilingual user manuals** — this document and its Chinese counterpart added under `manual/`, adversarially fact-checked against the code; README gains the GREY release image; `update_en.txt` back-filled with the missing V0.6.0 entry |

Full changelogs: `Update.txt` (Chinese) and `update_en.txt` (English; its V0.6.0
entry is a condensed back-fill).

## Appendix C — glafic, license & citations

GLADE bundles a locally modified **glafic 2.1.14** by Masamune Oguri (GPLv3;
upstream: <https://github.com/oguri/glafic2>). Local modifications: the King (1962)
profile as model #27 (`king`), the `TOL_ROMBERG_JHK` override (Chapter 10), the
`c2calc_each` per-component χ² binding, and the regenerated Makefile (the
`TOL_ROMBERG_JHK` and `c2calc_each` changes are marked `glade local` in source; the
King code and the Makefile carry no marker — `Update.txt` is the authoritative
list). The full official glafic manual — the authoritative
reference for model parameters, secondary settings and file formats — is bundled at
**`glafic2/manual/man_glafic.pdf`** (plain-text: `man_glafic.txt`).

**If you publish results obtained with GLADE**, cite glafic:

- M. Oguri, *PASJ*, **62**, 1017 (2010) — required for any use of (modified) glafic.
- M. Oguri, *PASP*, **133**, 074504 (2021) — additionally, if you use the `anfw` or
  `ahern` models.

GLADE itself is MIT-licensed; emcee, corner, scipy/numpy/astropy/matplotlib have
their own citation requests if used prominently.

## Appendix D — Environment variables

| Variable | Default | Effect |
|---|---|---|
| `GLADE_PORT` | 6017 | WebUI port (`GLADE_PORT=8080 ./run_webui.sh`) |
| `CLAVE_PORT` | 6019 | standalone Clave port (`python -m clave`) |
| `GLAFIC_AMOEBA_TIMEOUT` | 3600 | seconds before a Glafic-rail amoeba run is killed; `0` = no limit |
| `GLADE_GPU_CHUNK` | 32/128 (heuristic) | candidates per CUDA pass in batched GPU mode; lower on OOM, raise for speed (§7.3.2) |
| `GLADE_ROOT`, `GLAFIC_HOME`, `GLAFIC_PYTHON_PATH`, `GLAFIC_LIB_PATH` | set by `env.sh` | install locations used by the launchers and job runner |

## Appendix E — Auxiliary file formats

- **glafic point-constraint file** (`constraint_file`, also produced as
  `amoeba_obs.dat` / `*_obs.dat` on export): header `1 <n_img> <z_s> 0.0`, then one
  row per image `x y flux pos_sigma flux_err td td_err parity` (arcsec / lens frame;
  GLADE writes flux = |μ| with the parity sign in the last column).
- **glafic prior file** (`prior_file`, `*_prior.dat`): lines like
  `gauss lens 1 3 0.0327 0.00097` (Gaussian prior) or `range lens 2 7 0.5 2.5`
  (bounds; `param_no` 1 = z, 2…8 = p1…p7), `range hubble lo hi`, and
  `match lens i j ii jj 1.0 0.0` (hard parameter tie — GLADE's shared variables).
- **critical-curve file** (`best_crit.dat`, `<prefix>_crit.dat`): 8 columns per row —
  a critical-curve segment (x1 y1 → x2 y2, columns 0,1,4,5) with its caustic segment
  (columns 2,3,6,7).
- **`galfile.dat`** (for `gals`): rows `x y L [e pa]`, `#` comments allowed.
- **glafic image list** (`<prefix>_point.dat`): header `n zs src_x src_y`, then one
  row `x y mu tdelay` per image.

---

*GLADE manual · V0.6.0 · generated 2026-07 · English edition (中文版:
`GLADE_Manual_zh.md`)*
