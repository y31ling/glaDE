"""Extended-source result figure: observed / model / residual panels.

For an extended (FITS) fit the meaningful result picture is the pixel comparison
between the observed image and the best-fit lensed model, not a point-image
triptych. This renders three panels sharing a brightness scale (observed and
model) plus a diverging residual, with an optional critical-curve overlay.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def plot_extend_result(
    obs: Optional[np.ndarray],
    model: np.ndarray,
    output_file: str = "extend_result.png",
    *,
    extent: Optional[tuple] = None,
    crit_segments: Optional[list] = None,
    image_positions: Optional[np.ndarray] = None,
    suptitle: str = "GLADE extended-source result",
    components: Optional[dict] = None,
) -> str:
    """Render the observed/model/residual figure; returns ``output_file``.

    ``obs`` may be ``None`` (e.g. astropy missing) — then only the model is
    drawn. ``extent`` is ``(xmin, xmax, ymin, ymax)`` in arcsec for axis labels.
    ``components`` is an optional dict of chi2 component values for the subtitle.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = np.asarray(model, dtype=float)
    have_obs = obs is not None and np.asarray(obs).shape == model.shape
    ncol = 3 if have_obs else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 5.0), squeeze=False)
    axes = axes[0]

    imkw = dict(origin="lower", extent=extent, cmap="inferno")

    def _overlay(ax):
        if crit_segments:
            for seg in crit_segments:
                seg = np.asarray(seg)
                if seg.ndim == 2 and seg.shape[1] >= 2:
                    ax.plot(seg[:, 0], seg[:, 1], color="cyan", lw=0.6, alpha=0.8)
        if image_positions is not None and len(image_positions):
            ip = np.asarray(image_positions)
            ax.scatter(ip[:, 0], ip[:, 1], s=60, facecolors="none",
                       edgecolors="lime", linewidths=1.2)

    if have_obs:
        obs = np.asarray(obs, dtype=float)
        vmax = float(np.nanmax([obs.max(), model.max()]))
        vmin = float(min(0.0, np.nanmin([obs.min(), model.min()])))
        im0 = axes[0].imshow(obs, vmin=vmin, vmax=vmax, **imkw)
        axes[0].set_title("Observed")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(model, vmin=vmin, vmax=vmax, **imkw)
        axes[1].set_title("Model (best fit)")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        resid = model - obs
        rlim = float(np.nanmax(np.abs(resid))) or 1.0
        im2 = axes[2].imshow(resid, vmin=-rlim, vmax=rlim, origin="lower",
                             extent=extent, cmap="RdBu_r")
        axes[2].set_title("Residual (model - obs)")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        for ax in axes:
            _overlay(ax)
    else:
        im = axes[0].imshow(model, **imkw)
        axes[0].set_title("Model (best fit)")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        _overlay(axes[0])

    for ax in axes:
        if extent is not None:
            ax.set_xlabel("x [arcsec]")
            ax.set_ylabel("y [arcsec]")

    sub = suptitle
    if components:
        parts = ", ".join(f"{k}={v:.3g}" for k, v in components.items())
        sub = f"{suptitle}\n{parts}"
    fig.suptitle(sub, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_file, dpi=130)
    plt.close(fig)
    return output_file
