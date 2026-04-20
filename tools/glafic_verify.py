#!/usr/bin/env python3
"""
glafic_verify.py: Verify best_params against glafic CLI.

Reads a ``*_best_params.txt`` file, recomputes the predicted image
positions/magnifications with the batched Rhongomyniad solver, runs
glafic on the same model, and prints a side-by-side comparison.

Usage:
    python glafic_verify.py <folder_or_params_file>
    python glafic_verify.py <folder> --output <dir> --prefix verify
    python glafic_verify.py <folder> --verbose

The tool auto-detects the model type (pointmass / nfw / p_jaffe / king)
from the file name or header, parses the uniform GPU-format subhalo
block (``x_sub{i}``, ``y_sub{i}`` ...), and supports GPU-written
outputs from v_*_gpu scripts.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Path setup — mirrors run_glafic.py
# ---------------------------------------------------------------------------
GLADE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GLADE_ROOT))
try:
    from runtime_env import setup_runtime_env
    setup_runtime_env(str(GLADE_ROOT))
except Exception:
    pass

_RHONG_DIR = GLADE_ROOT / "Rhongomyniad"
if _RHONG_DIR.exists():
    sys.path.insert(0, str(_RHONG_DIR))

GLAFIC_BASE = GLADE_ROOT.parent.parent


# ---------------------------------------------------------------------------
# Fixed model defaults (match legacy GPU scripts)
# ---------------------------------------------------------------------------
OMEGA         = 0.3
LAMBDA_COSMO  = 0.7
WEOS          = -1.0
HUBBLE        = 0.7

XMIN, XMAX    = -0.5, 0.5
YMIN, YMAX    = -0.5, 0.5
PIX_EXT       = 0.01
PIX_POI       = 0.2
MAXLEV        = 5

SOURCE_Z      = 0.4090
SOURCE_X      = 2.685497e-03
SOURCE_Y      = 2.443616e-02
LENS_Z        = 0.2160

LENS_PARAMS = {
    "sers1": (1, "sers", 0.2160, 9.896617e+09, 2.656977e-03, 2.758473e-02,
              2.986760e-01, 1.124730e+02, 3.939718e-01, 1.057760e+00),
    "sers2": (2, "sers", 0.2160, 2.555580e+10, 2.656977e-03, 2.758473e-02,
              4.242340e-01, 5.396370e+01, 1.538855e+00, 1.000000e+00),
    "sie":   (3, "sie",  0.2160, 1.183382e+02, 2.656977e-03, 2.758473e-02,
              1.571203e-01, 2.920348e+01, 0.0, 0.0),
}
MAIN_LENS_KEY = "sie"

OBS_POS_MAS = np.array([[-266.035, 0.427], [118.835, -221.927],
                        [238.324, 227.270], [-126.157, 319.719]])
OBS_X_FLIP       = True
CENTER_OFFSET_X  = +0.01535
CENTER_OFFSET_Y  = +0.03220


def _obs_positions():
    sign = -1 if OBS_X_FLIP else 1
    out = np.zeros_like(OBS_POS_MAS)
    out[:, 0] = sign * OBS_POS_MAS[:, 0] / 1000.0
    out[:, 1] = OBS_POS_MAS[:, 1] / 1000.0
    return out


OBS_POSITIONS    = _obs_positions()
CENTER_OFFSET_X  = (-1 if OBS_X_FLIP else 1) * CENTER_OFFSET_X


# ---------------------------------------------------------------------------
# glafic binary lookup
# ---------------------------------------------------------------------------
def find_glafic_bin():
    for p in (GLADE_ROOT / "glafic2" / "glafic",
              GLAFIC_BASE / "glafic2" / "glafic",
              Path("/home/luukiaun/glafic251018/glafic2/glafic")):
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    try:
        import glafic as _gl
        mod_dir = Path(_gl.__file__).resolve().parent
        for rel in ("../glafic", "../../glafic", "./glafic", "../bin/glafic"):
            cand = (mod_dir / rel).resolve()
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    except Exception:
        pass
    found = shutil.which("glafic")
    return found


# ---------------------------------------------------------------------------
# best_params parsing — uniform GPU format (x_sub{i}, y_sub{i}, ...)
# ---------------------------------------------------------------------------
def find_params_file(folder: Path):
    matches = sorted(folder.glob("*_best_params.txt"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def detect_model_type(params_file: Path) -> str | None:
    header = params_file.read_text()[:400].lower()
    name = params_file.name.lower()
    for token, kind in (("king", "king"), ("nfw", "nfw"),
                        ("jaffe", "p_jaffe"), ("pointmass", "pointmass"),
                        ("pmass", "pointmass"), ("point_mass", "pointmass"),
                        ("none", "none")):
        if token in name or token in header:
            return kind
    if "p-jaffe" in header or "p_jaffe" in header:
        return "p_jaffe"
    return None


def parse_none_best(params_file: Path):
    """Parse a v_none_gpu best_params file.  Returns (source_x, source_y,
    ordered_lens_list) where each lens is (key, ltype, z, p1..p7)."""
    content = params_file.read_text()
    sx_m = re.search(r"(?<![A-Za-z0-9_])source_x\s*=\s*([-\d.eE+]+)", content)
    sy_m = re.search(r"(?<![A-Za-z0-9_])source_y\s*=\s*([-\d.eE+]+)", content)
    if not sx_m or not sy_m:
        return None
    sx = float(sx_m.group(1)); sy = float(sy_m.group(1))
    lens_list = []
    seen = set()
    for m in re.finditer(r"(?m)^([A-Za-z0-9]+)_type\s*=\s*(\w+)", content):
        key = m.group(1); ltype = m.group(2)
        if key in seen:
            continue
        seen.add(key)
        zm = re.search(rf"(?m)^{re.escape(key)}_z\s*=\s*([-\d.eE+]+)", content)
        if not zm:
            continue
        z = float(zm.group(1))
        ps = []
        for i in range(1, 8):
            pm = re.search(rf"(?m)^{re.escape(key)}_p{i}\s*=\s*([-\d.eE+]+)",
                           content)
            ps.append(float(pm.group(1)) if pm else 0.0)
        lens_list.append((key, ltype, z, *ps))
    if not lens_list:
        return None
    return sx, sy, lens_list


def _get(content: str, key: str, i: int) -> float | None:
    m = re.search(rf"(?<![A-Za-z0-9_]){key}{i}\s*=\s*([-\d.eE+]+)", content)
    return float(m.group(1)) if m else None


def parse_subhalos(params_file: Path, model: str) -> list[tuple]:
    """Return list of (img_idx, ...params) sorted by img_idx."""
    content = params_file.read_text()
    img_idxs = sorted({int(m) for m in
                       re.findall(r"x_sub(\d+)\s*=", content)})
    subs = []
    for i in img_idxs:
        x = _get(content, "x_sub", i)
        y = _get(content, "y_sub", i)
        if x is None or y is None:
            continue
        if model == "pointmass":
            m = _get(content, "mass_sub", i)
            if m is None:
                continue
            subs.append((i, x, y, m))
        elif model == "nfw":
            m = _get(content, "mass_sub", i)
            c = _get(content, "conc_sub", i)
            if m is None or c is None:
                continue
            subs.append((i, x, y, m, c))
        elif model == "king":
            m  = _get(content, "mass_sub", i)
            rc = _get(content, "rc_sub", i)
            c  = _get(content, "c_sub", i)
            if None in (m, rc, c):
                continue
            subs.append((i, x, y, m, rc, c))
        elif model == "p_jaffe":
            sig = _get(content, "sig_sub", i)
            a   = _get(content, "a_sub", i)
            rco = _get(content, "rco_sub", i)
            if None in (sig, a, rco):
                continue
            subs.append((i, x, y, sig, a, rco))
    return subs


# ---------------------------------------------------------------------------
# glafic input generation + execution
# ---------------------------------------------------------------------------
def _fmt_lens(pv):
    return (f"lens  {pv[1]}  {pv[2]}  "
            f"{pv[3]:.6e}  {pv[4]:.6e}  {pv[5]:.6e}  "
            f"{pv[6]:.6e}  {pv[7]:.6e}  {pv[8]:.6e}  {pv[9]:.6e}\n")


def generate_glafic_input_none(source_x: float, source_y: float,
                               lens_list: list[tuple],
                               output_dir: Path, prefix: str) -> Path:
    input_file = output_dir / f"{prefix}_input.dat"
    with open(input_file, "w") as f:
        f.write(f"# glafic_verify — NONE (main-lens optimization)\n")
        f.write(f"# generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"omega    {OMEGA}\n")
        f.write(f"lambda   {LAMBDA_COSMO}\n")
        f.write(f"weos     {WEOS}\n")
        f.write(f"hubble   {HUBBLE}\n")
        f.write(f"prefix   {prefix}\n")
        f.write(f"xmin     {XMIN}\n")
        f.write(f"ymin     {YMIN}\n")
        f.write(f"xmax     {XMAX}\n")
        f.write(f"ymax     {YMAX}\n")
        f.write(f"pix_ext  {PIX_EXT}\n")
        f.write(f"pix_poi  {PIX_POI}\n")
        f.write(f"maxlev   {MAXLEV}\n\n")
        f.write(f"startup  {len(lens_list)} 0 1\n")
        for key, ltype, z, *ps in lens_list:
            f.write(f"lens  {ltype}  {z}  "
                    + "  ".join(f"{v:.10e}" for v in ps) + "\n")
        f.write(f"point  {SOURCE_Z}  {source_x:.10e}  {source_y:.10e}\n")
        f.write("end_startup\n\nstart_command\nfindimg\nquit\n")
    return input_file


def generate_glafic_input(model: str, subs: list[tuple],
                          output_dir: Path, prefix: str) -> Path:
    input_file = output_dir / f"{prefix}_input.dat"
    with open(input_file, "w") as f:
        f.write(f"# glafic_verify — {model.upper()}\n")
        f.write(f"# generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"omega    {OMEGA}\n")
        f.write(f"lambda   {LAMBDA_COSMO}\n")
        f.write(f"weos     {WEOS}\n")
        f.write(f"hubble   {HUBBLE}\n")
        f.write(f"prefix   {prefix}\n")
        f.write(f"xmin     {XMIN}\n")
        f.write(f"ymin     {YMIN}\n")
        f.write(f"xmax     {XMAX}\n")
        f.write(f"ymax     {YMAX}\n")
        f.write(f"pix_ext  {PIX_EXT}\n")
        f.write(f"pix_poi  {PIX_POI}\n")
        f.write(f"maxlev   {MAXLEV}\n\n")
        n_lens = len(LENS_PARAMS) + len(subs)
        f.write(f"startup  {n_lens} 0 1\n")
        for _, pv in LENS_PARAMS.items():
            f.write(_fmt_lens(pv))
        for sub in subs:
            if model == "pointmass":
                _, x, y, m = sub
                f.write(f"lens  point  {LENS_Z}  {m:.10e}  "
                        f"{x:.10e}  {y:.10e}  0.0  0.0  0.0  0.0\n")
            elif model == "nfw":
                _, x, y, m, c = sub
                f.write(f"lens  gnfw  {LENS_Z}  {m:.10e}  "
                        f"{x:.10e}  {y:.10e}  0.0  0.0  {c:.10e}  1.0\n")
            elif model == "king":
                _, x, y, m, rc, c = sub
                f.write(f"lens  king  {LENS_Z}  {m:.10e}  "
                        f"{x:.10e}  {y:.10e}  0.0  0.0  "
                        f"{rc:.10e}  {c:.10e}\n")
            elif model == "p_jaffe":
                _, x, y, sig, a, rco = sub
                f.write(f"lens  jaffe  {LENS_Z}  {sig:.10e}  "
                        f"{x:.10e}  {y:.10e}  0.0  0.0  "
                        f"{a:.10e}  {rco:.10e}\n")
        f.write(f"point  {SOURCE_Z}  {SOURCE_X:.10e}  {SOURCE_Y:.10e}\n")
        f.write("end_startup\n\nstart_command\nfindimg\nquit\n")
    return input_file


def run_glafic(input_file: Path, output_dir: Path,
               verbose: bool = False, timeout: int = 120):
    bin_path = find_glafic_bin()
    if not bin_path:
        print("  warn: glafic binary not found")
        return None
    print(f"  glafic: {bin_path}")
    try:
        proc = subprocess.run(
            [bin_path, input_file.name],
            cwd=str(output_dir), capture_output=True, text=True,
            timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  warn: glafic timeout (>{timeout}s)")
        return None
    except Exception as e:
        print(f"  warn: {e}")
        return None
    if verbose and proc.stdout:
        print(proc.stdout)
    if proc.returncode != 0:
        print(f"  warn: glafic exit {proc.returncode}")
        if proc.stderr:
            print(proc.stderr[:500])
        return None
    print("  glafic run OK")
    return bin_path


def read_glafic_point(output_dir: Path, prefix: str, allow_all: bool = False):
    pt = output_dir / f"{prefix}_point.dat"
    if not pt.exists():
        print(f"  warn: missing {pt}")
        return None, None
    data = np.loadtxt(pt)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n = int(data[0, 0])
    print(f"  glafic found {n} images")
    if not allow_all and n not in (4, 5):
        return None, None
    if n < 1:
        return None, None
    img = data[1:n + 1, :]
    if not allow_all and n == 5:
        drop = int(np.argmin(np.abs(img[:, 2])))
        img = np.delete(img, drop, axis=0)
    gl_pos = img[:, 0:2].copy()
    if not allow_all:
        gl_pos[:, 0] += CENTER_OFFSET_X
        gl_pos[:, 1] += CENTER_OFFSET_Y
    gl_mag = np.abs(img[:, 2])
    return gl_pos, gl_mag


# ---------------------------------------------------------------------------
# Python-side prediction via Rhongomyniad (same math as v_*_gpu scripts)
# ---------------------------------------------------------------------------
def _python_predict_none(source_x, source_y, lens_list):
    """Return unordered (pos, |mag|) from v_none_gpu's batched solver.
    Ordering is matched to glafic downstream since v_none_gpu's observations
    may differ from this tool's defaults."""
    legacy_dir = GLADE_ROOT / "legacy"
    sys.path.insert(0, str(legacy_dir))
    sys.path.insert(0, str(legacy_dir / "v_none_gpu"))
    mod_path = legacy_dir / "v_none_gpu" / "version_none_gpu.py"
    mod_code = mod_path.read_text()
    # Neutralize any user-set BASELINE_LENS_DIR so module-level setup doesn't
    # try to load an external bestfit.dat.
    mod_code = re.sub(r"(?m)^BASELINE_LENS_DIR\s*=.*",
                      'BASELINE_LENS_DIR = ""', mod_code, count=1)
    mod_globals = {"__name__": "__glafic_verify_none__",
                   "__file__": str(mod_path)}
    _prev_cwd = os.getcwd()
    os.chdir(str(mod_path.parent))
    try:
        exec(compile(mod_code, str(mod_path), "exec"), mod_globals)
    finally:
        os.chdir(_prev_cwd)
    batched_solve = mod_globals["_batched_point_solve"]
    lp_dict = {}
    for idx, (key, ltype, z, *ps) in enumerate(lens_list, 1):
        lp_dict[key] = (idx, ltype, z, *ps)
    imgs = batched_solve([lp_dict], np.array([source_x]),
                         np.array([source_y]))[0]
    if not imgs:
        print("  warn: Python solver found 0 images")
        return None, None
    pred_pos = np.array([[im[0], im[1]] for im in imgs])
    pred_mag = np.array([abs(im[2]) for im in imgs])
    return pred_pos, pred_mag


def _python_predict(model: str, subs: list[tuple]):
    """Return (pos (4,2), |mag| (4,)) matched to OBS_POSITIONS order."""
    import torch
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    import rhongomyniad as rh
    from rhongomyniad import constants as K
    from rhongomyniad.lens_models import LensContext
    from rhongomyniad.image_finder import sum_lensmodel
    from rhongomyniad.cosmology import Cosmology

    device = rh.get_device()
    dtype = torch.float64

    # Import the matching batched solver module (as a library).
    pkg_map = {
        "king":     ("legacy.v_king_gpu.version_king_gpu",        5),
        "nfw":      ("legacy.v_nfw_gpu.version_nfw_gpu",          4),
        "p_jaffe":  ("legacy.v_p_jaffe_gpu.version_p_jaffe_gpu",  5),
        "pointmass":("legacy.v_pointmass_gpu.version_pointmass_gpu", 3),
    }
    if model not in pkg_map:
        return None, None

    # Build shared fixed-lens cache using the same config as the GPU scripts.
    rh.init(OMEGA, LAMBDA_COSMO, WEOS, HUBBLE, "glafic_verify",
            XMIN, YMIN, XMAX, YMAX, PIX_EXT, PIX_POI, MAXLEV, verb=0)
    rh.startup_setnum(len(LENS_PARAMS), 0, 1)
    for pv in LENS_PARAMS.values():
        rh.set_lens(*pv)
    rh.set_point(1, SOURCE_Z, SOURCE_X, SOURCE_Y)
    rh.model_init(verb=0)

    cosmo = Cosmology(omega=OMEGA, lam=LAMBDA_COSMO, weos=WEOS, hubble=HUBBLE)
    ctx = LensContext.build(cosmo, zl=LENS_Z, zs=SOURCE_Z)

    dp = PIX_POI / (2 ** (MAXLEV - 1))
    nx = int(math.ceil((XMAX - XMIN) / dp)) + 1
    ny = int(math.ceil((YMAX - YMIN) / dp)) + 1
    xs_ax = [XMIN + i * dp for i in range(nx)]
    ys_ax = [YMIN + i * dp for i in range(ny)]
    import torch as _t
    xs_t = _t.tensor(xs_ax, device=device, dtype=dtype)
    ys_t = _t.tensor(ys_ax, device=device, dtype=dtype)
    gx, gy = _t.meshgrid(xs_t, ys_t, indexing="xy")

    fixed_lenses = [(pv[1], (pv[2], *pv[3:10])) for pv in LENS_PARAMS.values()]
    ax_f, ay_f, kap_f, g1_f, g2_f, _, _ = sum_lensmodel(
        ctx, fixed_lenses, gx, gy, need_kg=True, need_phi=False)
    cache = dict(ctx=ctx, gx=gx, gy=gy, dp=dp, nx=nx, ny=ny,
                 ax=ax_f.contiguous(), ay=ay_f.contiguous(),
                 kap=kap_f.contiguous(), g1=g1_f.contiguous(),
                 g2=g2_f.contiguous(), fixed_lenses=fixed_lenses)

    # Pull the batched solver out of the matching module.
    legacy_dir = GLADE_ROOT / "legacy"
    sys.path.insert(0, str(legacy_dir))
    mod_sub = {
        "king":    ("v_king_gpu", "version_king_gpu"),
        "nfw":     ("v_nfw_gpu", "version_nfw_gpu"),
        "p_jaffe": ("v_p_jaffe_gpu", "version_p_jaffe_gpu"),
    }
    if model not in mod_sub:
        return None, None
    sub_dir, mod_name = mod_sub[model]
    sys.path.insert(0, str(legacy_dir / sub_dir))
    mod = __import__(mod_name)

    # Pack candidate tensors (C=1, Kk=n_subs).
    K_sub = len(subs)
    sx_t = _t.tensor([[s[1] for s in subs]], device=device, dtype=dtype)
    sy_t = _t.tensor([[s[2] for s in subs]], device=device, dtype=dtype)

    if model == "king":
        lm_t = _t.tensor([[math.log10(s[3]) for s in subs]], device=device, dtype=dtype)
        rc_t = _t.tensor([[s[4] for s in subs]], device=device, dtype=dtype)
        ck_t = _t.tensor([[s[5] for s in subs]], device=device, dtype=dtype)
        imgs = mod.batched_point_solve(sx_t, sy_t, lm_t, rc_t, ck_t,
                                       SOURCE_X, SOURCE_Y, cache)[0]
    elif model == "nfw":
        lm_t = _t.tensor([[math.log10(s[3]) for s in subs]], device=device, dtype=dtype)
        ck_t = _t.tensor([[s[4] for s in subs]], device=device, dtype=dtype)
        imgs = mod.batched_point_solve(sx_t, sy_t, lm_t, ck_t,
                                       SOURCE_X, SOURCE_Y, cache)[0]
    elif model == "p_jaffe":
        sg_t = _t.tensor([[s[3] for s in subs]], device=device, dtype=dtype)
        a_t  = _t.tensor([[s[4] for s in subs]], device=device, dtype=dtype)
        rc_t = _t.tensor([[s[5] for s in subs]], device=device, dtype=dtype)
        imgs = mod.batched_point_solve(sx_t, sy_t, sg_t, a_t, rc_t,
                                       SOURCE_X, SOURCE_Y, cache)[0]
    else:
        return None, None

    if len(imgs) != 4:
        print(f"  warn: Python solver found {len(imgs)} images (expected 4)")
        return None, None
    pred_pos = np.array([[im[0] + CENTER_OFFSET_X,
                          im[1] + CENTER_OFFSET_Y] for im in imgs])
    pred_mag = np.array([abs(im[2]) for im in imgs])
    d = cdist(OBS_POSITIONS, pred_pos)
    ri, ci = linear_sum_assignment(d)
    order = ci[np.argsort(ri)]
    return pred_pos[order], pred_mag[order]


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def _match_glafic_to_obs(gl_pos, gl_mag):
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    d = cdist(OBS_POSITIONS, gl_pos)
    ri, ci = linear_sum_assignment(d)
    order = ci[np.argsort(ri)]
    return gl_pos[order], gl_mag[order]


def _print_diff_table(py_pos, py_mag, gl_pos, gl_mag):
    print(f"\n  {'Img':<4} {'Py x[mas]':>12} {'GL x[mas]':>12} {'|Δx|':>8}"
          f"  {'Py y[mas]':>12} {'GL y[mas]':>12} {'|Δy|':>8}")
    print("  " + "-" * 78)
    max_pos = 0.0
    for k in range(4):
        px = py_pos[k, 0] * 1000; py = py_pos[k, 1] * 1000
        gx = gl_pos[k, 0] * 1000; gy = gl_pos[k, 1] * 1000
        dxv = abs(px - gx); dyv = abs(py - gy)
        max_pos = max(max_pos, dxv, dyv)
        print(f"  {k+1:<4} {px:>12.3f} {gx:>12.3f} {dxv:>8.3f}  "
              f"{py:>12.3f} {gy:>12.3f} {dyv:>8.3f}")

    print(f"\n  {'Img':<4} {'Py |μ|':>12} {'GL |μ|':>12} {'Δ [%]':>10}")
    print("  " + "-" * 45)
    max_mag = 0.0
    for k in range(4):
        pm = py_mag[k]; gm = gl_mag[k]
        pct = abs(pm - gm) / pm * 100 if pm else 0.0
        max_mag = max(max_mag, pct)
        print(f"  {k+1:<4} {pm:>12.3f} {gm:>12.3f} {pct:>9.3f}%")

    print(f"\n  max position diff: {max_pos:.6f} mas")
    print(f"  max magnif. diff:  {max_mag:.6f}%")
    if max_pos < 0.01 and max_mag < 0.1:
        print("  [PASS] consistency verified")
    elif max_pos < 1.0 and max_mag < 1.0:
        print("  [OK]   small differences")
    else:
        print("  [WARN] large discrepancy — check params")
    return max_pos, max_mag


def _match_py_to_gl(py_pos, py_mag, gl_pos, gl_mag):
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    n = min(len(py_pos), len(gl_pos))
    if n == 0:
        return py_pos, py_mag, gl_pos, gl_mag
    d = cdist(py_pos[:n], gl_pos[:n])
    ri, ci = linear_sum_assignment(d)
    return py_pos[:n][ri], py_mag[:n][ri], gl_pos[:n][ci], gl_mag[:n][ci]


def _print_diff_table_n(py_pos, py_mag, gl_pos, gl_mag):
    n = min(len(py_pos), len(gl_pos))
    print(f"\n  {'Img':<4} {'Py x[mas]':>12} {'GL x[mas]':>12} {'|Δx|':>8}"
          f"  {'Py y[mas]':>12} {'GL y[mas]':>12} {'|Δy|':>8}")
    print("  " + "-" * 78)
    max_pos = 0.0
    for k in range(n):
        px = py_pos[k, 0] * 1000; py = py_pos[k, 1] * 1000
        gx = gl_pos[k, 0] * 1000; gy = gl_pos[k, 1] * 1000
        dxv = abs(px - gx); dyv = abs(py - gy)
        max_pos = max(max_pos, dxv, dyv)
        print(f"  {k+1:<4} {px:>12.3f} {gx:>12.3f} {dxv:>8.3f}  "
              f"{py:>12.3f} {gy:>12.3f} {dyv:>8.3f}")
    print(f"\n  {'Img':<4} {'Py |μ|':>12} {'GL |μ|':>12} {'Δ [%]':>10}")
    print("  " + "-" * 45)
    max_mag = 0.0
    for k in range(n):
        pm = py_mag[k]; gm = gl_mag[k]
        pct = abs(pm - gm) / pm * 100 if pm else 0.0
        max_mag = max(max_mag, pct)
        print(f"  {k+1:<4} {pm:>12.3f} {gm:>12.3f} {pct:>9.3f}%")
    print(f"\n  max position diff: {max_pos:.6f} mas")
    print(f"  max magnif. diff:  {max_mag:.6f}%")
    if max_pos < 0.01 and max_mag < 0.1:
        print("  [PASS] consistency verified")
    elif max_pos < 1.0 and max_mag < 1.0:
        print("  [OK]   small differences")
    else:
        print("  [WARN] large discrepancy — check params")
    return max_pos, max_mag


def _print_glafic_only(gl_pos, gl_mag):
    print(f"\n  {'Img':<4} {'GL x[mas]':>12} {'GL y[mas]':>12} {'GL |μ|':>10}")
    print("  " + "-" * 48)
    for k in range(4):
        print(f"  {k+1:<4} {gl_pos[k, 0]*1000:>12.3f} "
              f"{gl_pos[k, 1]*1000:>12.3f} {gl_mag[k]:>10.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _resolve_folder(arg: str) -> Path | None:
    p = Path(arg)
    if p.is_file():
        return p
    if p.is_dir():
        return p
    for root in (GLADE_ROOT, GLADE_ROOT / "results",
                 GLAFIC_BASE, GLAFIC_BASE / "results"):
        cand = root / arg
        if cand.is_dir():
            return cand
        if cand.is_file():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Verify best_params against glafic CLI")
    ap.add_argument("path", help="Folder containing best_params.txt, or the file itself")
    ap.add_argument("--output", help="glafic output dir (default: params' folder)")
    ap.add_argument("--prefix", default="glafic_verify", help="glafic output prefix")
    ap.add_argument("--model",
                    choices=("king", "nfw", "p_jaffe", "pointmass", "none"),
                    help="Override model type (otherwise auto-detected)")
    ap.add_argument("--no-python", action="store_true",
                    help="Skip Python/Rhongomyniad prediction; only run glafic")
    ap.add_argument("--verbose", action="store_true",
                    help="Print glafic stdout")
    ap.add_argument("--timeout", type=int, default=120,
                    help="glafic timeout in seconds")
    args = ap.parse_args()

    target = _resolve_folder(args.path)
    if target is None:
        print(f"error: cannot find '{args.path}'")
        sys.exit(1)

    if target.is_file():
        params_file = target
        folder = target.parent
    else:
        folder = target
        params_file = find_params_file(folder)
        if params_file is None:
            print(f"error: no *_best_params.txt under {folder}")
            sys.exit(1)

    print("=" * 70)
    print("glafic_verify")
    print("=" * 70)
    print(f"params: {params_file}")

    model = args.model or detect_model_type(params_file)
    if model is None:
        print("error: cannot detect model type; use --model")
        sys.exit(1)
    print(f"model:  {model}")

    output_dir = Path(args.output).resolve() if args.output else folder
    output_dir.mkdir(parents=True, exist_ok=True)

    if model == "none":
        parsed = parse_none_best(params_file)
        if parsed is None:
            print("error: cannot parse v_none best_params")
            sys.exit(1)
        src_x, src_y, lens_list = parsed
        print(f"source: ({src_x:+.6e}, {src_y:+.6e})")
        print(f"lenses: {len(lens_list)}")
        for key, ltype, z, *ps in lens_list:
            head = "  ".join(f"{v:.4g}" for v in ps[:3])
            print(f"  {key:<8} {ltype:<6} z={z}  p1..3={head}...")
        print(f"output: {output_dir}")
        input_file = generate_glafic_input_none(
            src_x, src_y, lens_list, output_dir, args.prefix)
        print(f"input:  {input_file}")
        subs = None  # marker for downstream branches
    else:
        subs = parse_subhalos(params_file, model)
        if not subs:
            print("error: no sub-halos parsed")
            sys.exit(1)
        print(f"subhalos: {len(subs)}")
        for s in subs:
            idx = s[0]
            tail = "  ".join(f"{v:.4g}" for v in s[3:])
            print(f"  img {idx}: ({s[1]:+.6f}, {s[2]:+.6f})  {tail}")
        print(f"output: {output_dir}")
        input_file = generate_glafic_input(model, subs, output_dir, args.prefix)
        print(f"input:  {input_file}")

    print("\n-- running glafic --")
    if run_glafic(input_file, output_dir, args.verbose, args.timeout) is None:
        sys.exit(1)

    gl_pos, gl_mag = read_glafic_point(output_dir, args.prefix,
                                       allow_all=(model == "none"))
    if gl_pos is None:
        sys.exit(1)
    if model != "none":
        gl_pos, gl_mag = _match_glafic_to_obs(gl_pos, gl_mag)

    print("\n-- verification --")
    if args.no_python:
        _print_glafic_only(gl_pos, gl_mag)
        result = {"max_pos_mas": None, "max_mag_pct": None}
    else:
        print("  running Rhongomyniad solver...")
        if model == "none":
            py_pos, py_mag = _python_predict_none(src_x, src_y, lens_list)
        else:
            py_pos, py_mag = _python_predict(model, subs)
        if py_pos is None:
            print("  [skipped] Python prediction unavailable")
            _print_glafic_only(gl_pos, gl_mag)
            result = {"max_pos_mas": None, "max_mag_pct": None}
        elif model == "none":
            py_pos, py_mag, gl_pos, gl_mag = _match_py_to_gl(
                py_pos, py_mag, gl_pos, gl_mag)
            max_pos, max_mag = _print_diff_table_n(py_pos, py_mag, gl_pos, gl_mag)
            result = {"max_pos_mas": max_pos, "max_mag_pct": max_mag}
        else:
            max_pos, max_mag = _print_diff_table(py_pos, py_mag, gl_pos, gl_mag)
            result = {"max_pos_mas": max_pos, "max_mag_pct": max_mag}

    report = output_dir / f"{args.prefix}_report.txt"
    with open(report, "w") as f:
        f.write(f"# glafic_verify report\n")
        f.write(f"# time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# params: {params_file}\n")
        f.write(f"# model:  {model}\n")
        if subs is not None:
            f.write(f"# subhalos: {len(subs)}\n")
        else:
            f.write(f"# lenses: {len(lens_list)}\n")
        if result["max_pos_mas"] is not None:
            f.write(f"max_pos_mas = {result['max_pos_mas']:.6f}\n")
            f.write(f"max_mag_pct = {result['max_mag_pct']:.6f}\n")
        for k in range(len(gl_pos)):
            f.write(f"img{k+1}_gl  x={gl_pos[k,0]:.10e}  y={gl_pos[k,1]:.10e}  "
                    f"|mu|={gl_mag[k]:.6f}\n")
    print(f"\nreport: {report}")
    return result


if __name__ == "__main__":
    main()
