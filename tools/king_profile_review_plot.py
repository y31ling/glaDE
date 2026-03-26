#!/usr/bin/env python3
"""
Review current glafic King profile implementation.

Outputs:
  - king_profile_review.png   (kappa vs radius)
  - king_profile_review.csv   (sampled data)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GLADE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GLADE_ROOT)
from runtime_env import setup_runtime_env  # noqa: E402
setup_runtime_env(GLADE_ROOT)
sys.path.insert(0, f"{GLADE_ROOT}/glafic2/python")

import glafic


def kappa_king_dl(x: np.ndarray, c: float) -> np.ndarray:
    """Dimensionless shape from mass.c (kappa_king_dl)."""
    xt = 10.0**c
    st = np.sqrt(1.0 + xt * xt)
    f0 = 1.0 / st
    norm = np.log(st) - 1.5 + 2.0 * f0 - 0.5 * f0 * f0
    norm = max(norm, 1.0e-30)

    f = 1.0 / np.sqrt(1.0 + x * x) - f0
    out = (f * f) / norm
    out = np.where(x >= xt, 0.0, out)
    out = np.where(f <= 0.0, 0.0, out)
    return out


def sample_glafic_kappa(
    zl: float, zs: float, mass: float, rc: float, c: float, radii: np.ndarray
) -> np.ndarray:
    glafic.init(
        0.3,
        0.7,
        -1.0,
        0.7,
        "/tmp/king_profile_review",
        -30.0,
        -30.0,
        30.0,
        30.0,
        0.1,
        0.2,
        5,
        verb=0,
    )
    glafic.startup_setnum(1, 0, 1)
    glafic.set_lens(1, "king", zl, mass, 0.0, 0.0, 0.0, 0.0, rc, c)
    glafic.set_point(1, zs, 0.01, 0.0)
    glafic.model_init(verb=0)

    vals = []
    for r in radii:
        # calcimage returns: ax, ay, tdelay, kappa, gam1, gam2, ...
        vals.append(glafic.calcimage(zs, float(r), 0.0)[3])

    glafic.quit()
    return np.array(vals, dtype=float)


def denoise_kappa_profile(kappa: np.ndarray, radii: np.ndarray, rt: float) -> np.ndarray:
    """
    Remove numerical noise while keeping physical behavior:
    - kappa = 0 for r >= rt (truncated King profile)
    - tiny/negative values near machine noise are set to 0
    """
    out = np.array(kappa, dtype=float)

    # Adaptive floor from central amplitude, with a hard minimum.
    peak = max(float(np.max(np.abs(out))), 1.0e-30)
    floor = max(peak * 1.0e-9, 1.0e-12)

    out[np.abs(out) < floor] = 0.0
    out[out < 0.0] = 0.0
    out[radii >= rt] = 0.0

    return out


def positive_or_nan(v: np.ndarray) -> np.ndarray:
    """For log plotting: hide exact zeros instead of showing fake floors."""
    return np.where(v > 0.0, v, np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot King profile review curves from glafic and mass.c formula."
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable denoising and use raw glafic kappa for plotting/normalization.",
    )
    args = parser.parse_args()

    use_denoise = not args.no_denoise

    # Example parameter set (easy to see full core-to-tidal behavior)
    zl = 0.5
    zs = 3.0
    mass = 5.0e7
    rc = 1.0
    rt = 10
    #rt = rc * (10.0**c)
    c = np.log10(rt/rc)
    

    radii = np.logspace(-3, np.log10(rt * 3.0), 240)
    x = radii / rc

    kappa_glafic_raw = sample_glafic_kappa(zl, zs, mass, rc, c, radii)
    if use_denoise:
        kappa_glafic = denoise_kappa_profile(kappa_glafic_raw, radii, rt)
    else:
        kappa_glafic = np.array(kappa_glafic_raw, copy=True)
    kappa_shape = kappa_king_dl(x, c)

    # Normalized comparison to check shape consistency only.
    kg0 = max(kappa_glafic[0], 1.0e-30)
    ks0 = max(kappa_shape[0], 1.0e-30)
    kappa_glafic_n = kappa_glafic / kg0
    kappa_shape_n = kappa_shape / ks0

    out_dir = f"{ROOT}/work/tools"
    png_path = f"{out_dir}/king_profile_review.png"
    csv_path = f"{out_dir}/king_profile_review.csv"

    np.savetxt(
        csv_path,
        np.column_stack(
            [
                radii,
                x,
                kappa_glafic_raw,
                kappa_glafic,
                kappa_shape,
                kappa_glafic_n,
                kappa_shape_n,
            ]
        ),
        delimiter=",",
        header=(
            "r_arcsec,x_over_rc,kappa_glafic_raw,kappa_glafic_denoised,"
            "kappa_shape_dimless,"
            "kappa_glafic_norm,kappa_shape_norm"
        ),
        comments="",
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

    label_main = "glafic kappa(r), denoised" if use_denoise else "glafic kappa(r), raw"
    label_norm = (
        "glafic normalized (denoised)" if use_denoise else "glafic normalized (raw)"
    )

    ax1.loglog(radii, positive_or_nan(kappa_glafic), lw=2.0, label=label_main)
    ax1.axvline(rc, ls="--", lw=1.0, color="tab:gray", label=f"rc = {rc:.1f}")
    ax1.axvline(rt, ls=":", lw=1.2, color="tab:red", label=f"rt = {rt:.1f}")
    ax1.set_ylabel("kappa = Sigma / Sigma_crit")
    ax1.set_title("King profile from current glafic code")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend()

    ax2.loglog(radii, positive_or_nan(kappa_glafic_n), lw=2.0, label=label_norm)
    ax2.loglog(
        radii,
        positive_or_nan(kappa_shape_n),
        lw=1.6,
        ls="--",
        label="mass.c kappa_king_dl normalized",
    )
    ax2.axvline(rc, ls="--", lw=1.0, color="tab:gray")
    ax2.axvline(rt, ls=":", lw=1.2, color="tab:red")
    ax2.set_xlabel("r [arcsec]")
    ax2.set_ylabel("shape (normalized to center)")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)

    print(f"Saved: {png_path}")
    print(f"Saved: {csv_path}")
    print(f"Denoise: {'ON' if use_denoise else 'OFF'}")
    print(f"Parameters: zl={zl}, zs={zs}, M={mass:.3e}, rc={rc}, c={c}, rt={rt:.3f}")


if __name__ == "__main__":
    main()
