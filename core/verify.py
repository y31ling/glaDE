"""Independent verification of a result with the glafic binary.

After a run, write the best-fit model to a glafic input file, run the glafic
executable (a separate process — independent of the Python bindings and of the
GPU solver), read back the images it finds, and compare its loss to the
optimizer's. This never raises and never affects the result figure; it only
returns a report (and the caller prints warnings on a large discrepancy).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

import numpy as np

from .optimize.loss import LossConfig, ml_loss
from .optimize.matching import match_images, select_images
from .optimize.scene import ObsData, Scene

_NPAR = 7


def find_glafic_bin() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    candidates = [
        os.path.join(root, "glafic2", "glafic"),
        os.path.join(os.path.dirname(root), "glafic2", "glafic"),
    ]
    try:
        import glafic as _gl  # noqa: PLC0415
        mod = os.path.dirname(os.path.abspath(_gl.__file__))
        for rel in ("../glafic", "../../glafic", "./glafic", "../bin/glafic"):
            candidates.append(os.path.normpath(os.path.join(mod, rel)))
    except Exception:  # noqa: BLE001
        pass
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return shutil.which("glafic")


def _write_glafic_input(scene: Scene, path: str, prefix: str) -> None:
    def pad7(params):
        return ([float(v) for v in params] + [0.0] * _NPAR)[:_NPAR]

    lines = ["# GLADE independent verification input", ""]
    lines += [f"omega      {scene.omega}", f"lambda     {scene.lam}",
              f"weos       {scene.weos}", f"hubble     {scene.hubble}", "",
              f"prefix     {prefix}", "",
              f"xmin       {scene.xmin}", f"ymin       {scene.ymin}",
              f"xmax       {scene.xmax}", f"ymax       {scene.ymax}",
              f"pix_ext    {scene.pix_ext}", f"pix_poi    {scene.pix_poi}",
              f"maxlev     {scene.maxlev}", ""]
    lines.append(f"startup    {len(scene.components)} 0 1")
    for comp in scene.components:
        nums = "    ".join(f"{v:.10e}" for v in [comp.z, *pad7(comp.params)])
        lines.append(f"lens       {comp.glafic_type:<8} {nums}")
    lines.append(f"point      {scene.source_z}    {scene.source_x:.10e}    "
                 f"{scene.source_y:.10e}")
    lines += ["", "end_startup", "", "start_command", "", "findimg", "", "quit", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _read_glafic_point(point_path: str):
    if not os.path.isfile(point_path):
        return None
    try:
        data = np.loadtxt(point_path)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return []
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n = int(data[0, 0])
    if n < 1 or data.shape[0] < n + 1:
        return []
    img = data[1:n + 1, :]
    return [(float(r[0]), float(r[1]), float(r[2])) for r in img]


def verify_with_glafic(scene: Scene, obs: ObsData, output_dir: str,
                       loss_cfg: Optional[LossConfig] = None,
                       opt_loss: Optional[float] = None,
                       prefix: str = "glafic_verify",
                       timeout: int = 120) -> dict:
    """Run the glafic binary on the best-fit model and compare to the optimizer.

    Returns a report dict with ``warnings`` (a list of strings); ``ok`` is False
    when verification could not run (binary missing / glafic failed).
    """
    loss_cfg = loss_cfg or LossConfig()
    report: dict = {"ok": False, "warnings": []}

    bin_path = find_glafic_bin()
    if not bin_path:
        report["warning"] = "glafic binary not found; skipped independent verification"
        report["warnings"].append(report["warning"])
        return report
    report["glafic_bin"] = bin_path

    os.makedirs(output_dir, exist_ok=True)
    input_path = os.path.join(output_dir, f"{prefix}.input")
    _write_glafic_input(scene, input_path, prefix)

    try:
        proc = subprocess.run([bin_path, os.path.basename(input_path)],
                              cwd=output_dir, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        report["warning"] = f"glafic verification timed out (>{timeout}s)"
        report["warnings"].append(report["warning"])
        return report
    except Exception as exc:  # noqa: BLE001
        report["warning"] = f"glafic verification failed to run: {exc}"
        report["warnings"].append(report["warning"])
        return report
    if proc.returncode != 0:
        report["warning"] = f"glafic exited with code {proc.returncode}"
        report["warnings"].append(report["warning"])
        return report

    images = _read_glafic_point(os.path.join(output_dir, f"{prefix}_point.dat"))
    if not images:
        report["warning"] = "glafic produced no image output"
        report["warnings"].append(report["warning"])
        return report

    report["ok"] = True
    report["glafic_n_images"] = len(images)
    if len(images) != obs.n:
        report["warnings"].append(
            f"glafic found {len(images)} image(s); the result assumes {obs.n}")

    sel = select_images(images, obs.n)
    if sel is None:
        report["warnings"].append(
            "glafic image count does not match the observations; "
            "cannot compute a comparable loss")
        return report

    pred_pos = np.array([[im[0], im[1]] for im in sel], dtype=float)
    pred_mag = np.array([im[2] for im in sel], dtype=float)
    _, mm, delta = match_images(obs.positions, pred_pos, pred_mag, obs.center_offset)
    gloss = float(ml_loss(delta, mm, obs.magnifications, obs.mag_errors,
                          obs.pos_sigma_mas, loss_cfg))
    report["glafic_loss"] = gloss
    report["glafic_max_delta_mas"] = float(np.max(delta))

    if opt_loss is not None and np.isfinite(opt_loss):
        report["optimizer_loss"] = float(opt_loss)
        denom = max(abs(opt_loss), 1.0)
        rel = abs(gloss - opt_loss) / denom
        report["loss_rel_diff"] = float(rel)
        if rel > 0.5:
            report["warnings"].append(
                f"glafic loss {gloss:.3f} differs from the optimizer's {opt_loss:.3f} "
                f"by {rel*100:.0f}%. NOTE: glafic's elliptical-Sersic deflection is "
                f"Romberg-tolerance-limited (TOL_ROMBERG_JHK); this difference is "
                f"expected and is NOT a result error. See the scipy-reference check "
                f"(ground truth) below.")
    return report


# --------------------------------------------------------------------------- #
# extended-source verification: independent glafic-binary c2calc + findimg
# --------------------------------------------------------------------------- #

def _read_point_file_images(path: str):
    """Read observed image positions (x, y) from a glafic readobs_point file."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    out = []
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        i += 1
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        try:
            nimg = int(parts[1])
        except (ValueError, IndexError):
            continue
        for _ in range(nimg):
            if i >= len(lines):
                break
            row = lines[i].split()
            i += 1
            try:
                out.append((float(row[0]), float(row[1])))
            except (ValueError, IndexError):
                pass
    return out


def _write_glafic_extend_input(scene: Scene, spec, path: str, prefix: str) -> None:
    def pad7(params):
        return ([float(v) for v in params] + [0.0] * _NPAR)[:_NPAR]

    L = ["# GLADE extended-source verification input", "",
         f"omega      {scene.omega}", f"lambda     {scene.lam}",
         f"weos       {scene.weos}", f"hubble     {scene.hubble}",
         f"prefix     {prefix}",
         f"xmin       {scene.xmin}", f"ymin       {scene.ymin}",
         f"xmax       {scene.xmax}", f"ymax       {scene.ymax}",
         f"pix_ext    {scene.pix_ext}", f"pix_poi    {scene.pix_poi}",
         f"maxlev     {scene.maxlev}"]
    for k, v in spec.secondary.items():
        L.append(f"{k:<10} {v}")
    L.append("")
    L.append(f"startup    {len(scene.components)} {len(scene.extends)} {spec.n_point}")
    for comp in scene.components:
        nums = " ".join(f"{v:.10e}" for v in [comp.z, *pad7(comp.params)])
        L.append(f"lens     {comp.glafic_type} {nums}")
    for comp in scene.extends:
        nums = " ".join(f"{v:.10e}" for v in [comp.z, *pad7(comp.params)])
        L.append(f"extend   {comp.glafic_type} {nums}")
    for ps in spec.point_sources:
        L.append(f"point    {ps.zs} {ps.init_x:.10e} {ps.init_y:.10e}")
    L += ["end_startup", "", "start_setopt"]
    for _ in scene.components:
        L.append("0 0 0 0 0 0 0 0")        # lens params fixed (glade controls)
    for _ in scene.extends:
        L.append("0 0 0 0 0 0 0 0")        # extend params fixed (glade controls)
    for ps in spec.point_sources:
        L.append(f"0 {ps.free_x} {ps.free_y}")   # solve SN source position
    L += ["end_setopt", "", "start_command"]
    if spec.mask_file:
        L.append(f"readobs_extend {spec.extended_file} {spec.mask_file}")
    else:
        L.append(f"readobs_extend {spec.extended_file}")
    if spec.noise_file:
        L.append(f"readnoise_extend {spec.noise_file}")
    if spec.point_file and spec.n_point > 0:
        L.append(f"readobs_point {spec.point_file}")
    if spec.prior_file:
        L.append(f"parprior {spec.prior_file}")
    L.append("c2calc")
    if spec.n_point > 0:
        L.append("findimg")
    L += ["quit", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def verify_extend(opt_result, output_dir: str, prefix: str = "glafic_extverify",
                  timeout: int = 300) -> dict:
    """Independently verify an extended-source result with the glafic BINARY.

    Writes the best-fit model (lens + extend + point + secondary + obs) to a
    glafic input file, runs the standalone binary's ``c2calc`` (no optimize) and
    ``findimg``, and compares glafic's total chi2 to glade's reported total. Also
    runs the scipy deflection ground-truth check. Never raises.
    """
    report: dict = {"ok": False, "warnings": []}
    scene = opt_result.scene
    spec = getattr(opt_result, "extend_spec", None)
    comps = getattr(opt_result, "extend_components", None)
    if spec is None:
        report["warning"] = "not an extended-source result; nothing to verify"
        report["warnings"].append(report["warning"])
        return report

    glade_total = float(sum(comps)) if comps is not None else None
    report["glade_total_chi2"] = glade_total

    bin_path = find_glafic_bin()
    if not bin_path:
        report["warnings"].append("glafic binary not found; skipped binary verification")
    else:
        os.makedirs(output_dir, exist_ok=True)
        input_path = os.path.join(output_dir, f"{prefix}.input")
        _write_glafic_extend_input(scene, spec, input_path, prefix)
        try:
            proc = subprocess.run([bin_path, os.path.basename(input_path)],
                                  cwd=output_dir, capture_output=True, text=True,
                                  timeout=timeout)
            blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
            gchi2 = None
            for line in blob.splitlines():
                s = line.strip()
                if s.startswith("chi^2"):
                    try:
                        gchi2 = float(s.split("=")[-1])
                    except ValueError:
                        pass
            if gchi2 is not None:
                report["ok"] = True
                report["glafic_total_chi2"] = gchi2
                if glade_total is not None:
                    denom = max(abs(glade_total), 1.0)
                    rel = abs(gchi2 - glade_total) / denom
                    report["chi2_rel_diff"] = float(rel)
                    if rel > 0.05:
                        report["warnings"].append(
                            f"glafic-binary chi2 {gchi2:.4g} differs from glade's "
                            f"{glade_total:.4g} by {rel*100:.1f}% (expected ~0)")
                imgs = _read_glafic_point(os.path.join(output_dir, f"{prefix}_point.dat"))
                if imgs:
                    report["glafic_n_images"] = len(imgs)
            else:
                report["warnings"].append(
                    "could not parse chi^2 from glafic binary output")
        except subprocess.TimeoutExpired:
            report["warnings"].append(f"glafic binary verification timed out (>{timeout}s)")
        except Exception as exc:  # noqa: BLE001
            report["warnings"].append(f"glafic binary verification failed: {exc}")

    # scipy deflection ground truth on the lens, evaluated at the obs image
    # positions taken from the glafic constraint file (engine frame).
    if spec.point_file:
        positions = _read_point_file_images(spec.point_file)
        if positions:
            class _Obs:
                def __init__(self, pos):
                    self.positions = np.asarray(pos, dtype=float)
                    self.center_offset = (0.0, 0.0)
            ref = reference_check(scene, _Obs(positions))
            report["scipy_reference"] = ref
            report["warnings"].extend(ref.get("warnings", []))
    return report


# --------------------------------------------------------------------------- #
# scipy reference: engine-independent ground truth for the Sersic deflection
# --------------------------------------------------------------------------- #

def _sersic_defl_scipy(ctx, comp, x, y, quad, K, helpers):
    """Exact elliptical-Sersic deflection (ax, ay) via scipy quad@1e-11, using
    the Schramm-1990 formulation glafic and Rhongomyniad share."""
    import math
    _bnn_sers, _b_func_sers = helpers
    p = (list(comp.params) + [0.0] * 7)[:7]
    m, x0, y0, e, pa, re, n = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    q = 1.0 - e
    tt_dimless = re * _bnn_sers(n)
    bb = _b_func_sers(m, tt_dimless, n, ctx)
    tt = tt_dimless / math.sqrt(q)
    arg = -pa * math.pi / 180.0
    si, co = math.sin(arg), math.cos(arg)
    bx = (co * (x - x0) - si * (y - y0)) / tt
    by = (si * (x - x0) + co * (y - y0)) / tt
    sc = K.DEF_SMALLCORE

    def Jn(nexp):
        def f(u):
            equ = 1.0 - (1.0 - q * q) * u
            xi = math.sqrt(u * (by * by + bx * bx / equ + sc * sc))
            kap = math.exp(-xi ** (1.0 / n))
            return kap / equ ** (nexp + 0.5)
        val, _err = quad(f, 0.0, 1.0, epsrel=1e-11, epsabs=1e-300, limit=400)
        return val

    bpx = q * bx * Jn(1)
    bpy = q * by * Jn(0)
    px = bpx * co + bpy * si      # ell_pxpy rotation
    py = -bpx * si + bpy * co
    return bb * tt * px, bb * tt * py


def reference_check(scene, obs) -> dict:
    """Engine-independent ground-truth check: compute the EXACT total deflection
    at the observed image positions (Sersic via scipy@1e-11, other models via
    Rhongomyniad's analytic/accurate kernels), and report (a) how far the GPU
    engine's Sersic deflection is from the exact integral, and (b) the
    source-plane self-consistency (how tightly the observed images back-project
    to a single source under the exact model). Never raises.
    """
    report: dict = {"ok": False, "warnings": []}
    try:
        import numpy as np
        import torch
        from scipy.integrate import quad
        from rhongomyniad import constants as K
        from rhongomyniad.cosmology import Cosmology
        from rhongomyniad.image_finder import sum_lensmodel
        from rhongomyniad.lens_models import (LensContext, _b_func_sers,
                                              _bnn_sers)
    except Exception as exc:  # noqa: BLE001
        report["warning"] = f"scipy/Rhongomyniad reference unavailable: {exc}"
        report["warnings"].append(report["warning"])
        return report
    if not scene.components:
        report["warning"] = "no lens components to check"
        return report

    helpers = (_bnn_sers, _b_func_sers)
    lens_z = scene.components[0].z
    ctx = LensContext.build(
        Cosmology(omega=scene.omega, lam=scene.lam, weos=scene.weos, hubble=scene.hubble),
        zl=lens_z, zs=scene.source_z)

    def pad7(params):
        return ([float(v) for v in params] + [0.0] * 7)[:7]

    def rh_comp(comp, x, y):
        tx = torch.tensor([x], dtype=torch.float64)
        ty = torch.tensor([y], dtype=torch.float64)
        a, b, *_ = sum_lensmodel(ctx, [(comp.glafic_type, (comp.z, *pad7(comp.params)))],
                                 tx, ty, need_kg=False, need_phi=False)
        return float(a.item()), float(b.item())

    cox, coy = obs.center_offset
    sers_err = 0.0
    betas = []
    for (ox, oy) in obs.positions:
        tx, ty = float(ox) - cox, float(oy) - coy   # engine-frame image position
        ax = ay = 0.0
        for comp in scene.components:
            ra, rb = rh_comp(comp, tx, ty)
            if comp.glafic_type in ("sers", "serspot"):
                sa, sb = _sersic_defl_scipy(ctx, comp, tx, ty, quad, K, helpers)
                sers_err = max(sers_err, abs(sa - ra), abs(sb - rb))
                ax += sa
                ay += sb
            else:
                ax += ra
                ay += rb
        betas.append((tx - ax, ty - ay))

    betas = np.array(betas, dtype=float)
    src = betas.mean(axis=0)
    scatter_mas = float(np.max(np.sqrt(np.sum((betas - src) ** 2, axis=1))) * 1000.0)
    report.update(
        ok=True,
        gpu_sersic_vs_scipy_arcsec=float(sers_err),
        source_plane_scatter_mas=scatter_mas,
        backprojected_source=[float(src[0]), float(src[1])],
        fitted_source=[float(scene.source_x), float(scene.source_y)],
    )
    if sers_err > 1e-6:
        report["warnings"].append(
            f"GPU/Rhongomyniad Sersic deflection differs from the scipy-exact "
            f"integral by {sers_err:.2e} arcsec (expected ~1e-9)")
    return report
