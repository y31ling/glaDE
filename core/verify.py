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
        nums = "    ".join(f"{v:.8e}" for v in [comp.z, *pad7(comp.params)])
        lines.append(f"lens       {comp.glafic_type:<8} {nums}")
    lines.append(f"point      {scene.source_z}    {scene.source_x:.8e}    "
                 f"{scene.source_y:.8e}")
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
                f"by {rel*100:.0f}% — the solution may be image-finder dependent")
    return report
