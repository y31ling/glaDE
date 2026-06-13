"""The 3-panel result figure ("triptych") and its baseline-vs-optimized variant.

Self-contained extraction of the legacy ``plot_paper_style`` functions. Uses the
non-interactive Agg backend so it works headless.

Sub-halos are passed as ``(x, y, label)`` tuples so this module stays
model-agnostic; format the label with :func:`core.plot.subhalo_label`.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _sigma_array(sigma, n):
    if isinstance(sigma, (int, float)):
        return np.full(n, float(sigma))
    return np.asarray(sigma, dtype=float)


def _draw_sigma_lines(ax, x_positions, sigma, bar_w, show_2sigma):
    if np.allclose(sigma, sigma[0]):
        ax.axhline(sigma[0], linestyle="--", linewidth=1.5, color="blue",
                   label="1σ", alpha=0.7)
        if show_2sigma:
            ax.axhline(2.0 * sigma[0], linestyle=":", linewidth=1.5, color="red",
                       label="2σ", alpha=0.7)
    else:
        for x, s in zip(x_positions, sigma):
            ax.hlines(s, x - bar_w / 2, x + bar_w / 2, linestyles="--",
                      linewidth=2.0, colors="blue", alpha=0.7)
            if show_2sigma:
                ax.hlines(2.0 * s, x - bar_w / 2, x + bar_w / 2, linestyles=":",
                          linewidth=2.0, colors="red", alpha=0.7)
        ax.plot([], [], linestyle="--", linewidth=2.0, color="blue",
                label="1σ (per image)", alpha=0.7)
        if show_2sigma:
            ax.plot([], [], linestyle=":", linewidth=2.0, color="red",
                    label="2σ (per image)", alpha=0.7)


def _draw_image_plane(ax, obs_pos, pred_pos, crit_segments, caus_segments,
                      subhalos, subhalo_color="red"):
    for seg in crit_segments or []:
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                "b-", linewidth=1.2, alpha=0.6)
    if crit_segments:
        ax.plot([], [], "b-", linewidth=1.2, label="Critical curve")
    if caus_segments:
        for seg in caus_segments:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                    "g-", linewidth=0.8, alpha=0.5)
        ax.plot([], [], "g-", linewidth=0.8, label="Caustics")

    ax.scatter(obs_pos[:, 0], obs_pos[:, 1], marker="*", s=200, color="gold",
               edgecolors="black", linewidths=1.5, label="Observed", zorder=5)
    ax.scatter(pred_pos[:, 0], pred_pos[:, 1], marker="x", s=100, color="red",
               linewidths=2.5, label="Predicted", zorder=4)
    for i, (xo, yo) in enumerate(obs_pos, start=1):
        ax.text(xo + 0.01, yo + 0.01, f"{i}", va="bottom", ha="left",
                fontsize=11, fontweight="bold", color="darkblue")

    if subhalos:
        for (xs, ys, label) in subhalos:
            ax.scatter(xs, ys, marker="D", s=150, color=subhalo_color,
                       edgecolor="black", linewidth=2, zorder=10, alpha=0.9)
            ax.text(xs, ys - 0.025, label, va="top", ha="center", fontsize=8,
                    fontweight="bold", color=subhalo_color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=subhalo_color, alpha=0.9))
        ax.scatter([], [], marker="D", s=150, color=subhalo_color,
                   edgecolor="black", linewidth=2, label="Sub-halo")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Δx [arcsec]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Δy [arcsec]", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=True,
              fontsize=9, facecolor="white", edgecolor="gray", framealpha=0.9, ncol=2)


def plot_triptych(img_numbers, delta_pos_mas, sigma_pos_mas,
                  mu_obs, mu_obs_err, mu_pred, mu_at_obs_pred,
                  obs_positions_arcsec, pred_positions_arcsec,
                  crit_segments, caus_segments=None,
                  subhalos: Optional[Sequence] = None,
                  output_file: str = "triptych.png",
                  suptitle: str = "GLADE result",
                  show_2sigma: bool = False,
                  abs_mag: bool = True):
    """Render and save the 3-panel result figure. Returns ``output_file``.

    ``abs_mag`` mirrors the loss convention: the magnification panel shows
    |mu| (all bars upward from 0); ``False`` keeps the signed values.
    """
    img_numbers = np.asarray(img_numbers)
    delta_pos_mas = np.asarray(delta_pos_mas, dtype=float)
    obs_pos = np.asarray(obs_positions_arcsec, dtype=float)
    pred_pos = np.asarray(pred_positions_arcsec, dtype=float)
    sigma = _sigma_array(sigma_pos_mas, len(img_numbers))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    # left: position residuals
    ax = axes[0]
    bar_w = 0.6
    ax.bar(img_numbers, delta_pos_mas, width=bar_w, label="ΔPos",
           color="lightcoral", edgecolor="black", linewidth=1.5)
    _draw_sigma_lines(ax, img_numbers, sigma, bar_w, show_2sigma)
    ax.set_xlabel("Image Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("ΔPos [mas]", fontsize=12, fontweight="bold")
    ax.set_xticks(img_numbers)
    ax.set_title("Position residuals", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=True,
              fontsize=9, ncol=2)

    # middle: magnifications
    ax = axes[1]
    idx = np.arange(len(img_numbers))
    w = 0.22
    mo = np.asarray(mu_obs, dtype=float)
    mp = np.asarray(mu_pred, dtype=float)
    ma = np.asarray(mu_at_obs_pred, dtype=float)
    if abs_mag:
        mo, mp, ma = np.abs(mo), np.abs(mp), np.abs(ma)
    ax.bar(idx - w, mo, width=w,
           yerr=np.asarray(mu_obs_err, dtype=float), capsize=3,
           label="|μ_obs|" if abs_mag else "μ_obs",
           color="skyblue", edgecolor="black", linewidth=1.5)
    ax.bar(idx, mp, width=w, hatch="//",
           label="|μ_pred|" if abs_mag else "μ_pred",
           color="lightgreen", edgecolor="black", linewidth=1.5)
    ax.errorbar(idx + w, ma, fmt="o", markersize=8,
                label="|μ@obs|" if abs_mag else "μ@obs",
                color="red", linewidth=2)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(i)) for i in img_numbers])
    ax.set_xlabel("Image Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("|μ|" if abs_mag else "μ", fontsize=12, fontweight="bold")
    ax.set_title("Magnification", fontsize=12, fontweight="bold")
    if abs_mag:
        ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=True,
              fontsize=9, ncol=3)

    # right: image plane
    _draw_image_plane(axes[2], obs_pos, pred_pos, crit_segments, caus_segments,
                      subhalos)
    axes[2].set_title("Image plane", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_file


def plot_triptych_compare(img_numbers, delta_baseline, delta_optimized,
                          sigma_pos_mas, mu_obs, mu_obs_err,
                          mu_pred_baseline, mu_pred_optimized,
                          obs_positions_arcsec, pred_positions_arcsec,
                          crit_segments, caus_segments=None,
                          subhalos: Optional[Sequence] = None,
                          output_file: str = "triptych_compare.png",
                          suptitle: str = "GLADE: baseline vs optimized",
                          show_2sigma: bool = False,
                          abs_mag: bool = True):
    """Baseline-vs-optimized 3-panel comparison figure."""
    img_numbers = np.asarray(img_numbers)
    idx = np.arange(len(img_numbers))
    obs_pos = np.asarray(obs_positions_arcsec, dtype=float)
    pred_pos = np.asarray(pred_positions_arcsec, dtype=float)
    sigma = _sigma_array(sigma_pos_mas, len(img_numbers))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    # left: position offset comparison
    ax = axes[0]
    bar_w = 0.35
    ax.bar(idx - bar_w / 2, np.asarray(delta_baseline, dtype=float), width=bar_w,
           label="Baseline", color="lightgray", edgecolor="black", linewidth=1.5)
    ax.bar(idx + bar_w / 2, np.asarray(delta_optimized, dtype=float), width=bar_w,
           label="Optimized", color="lightcoral", edgecolor="black", linewidth=1.5)
    _draw_sigma_lines(ax, idx, sigma, 2 * bar_w, show_2sigma)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(i)) for i in img_numbers])
    ax.set_xlabel("Image Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("ΔPos [mas]", fontsize=12, fontweight="bold")
    ax.set_title("Position offset", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=8, ncol=2)

    # middle: magnification comparison
    ax = axes[1]
    w = 0.25
    mo = np.asarray(mu_obs, dtype=float)
    mb = np.asarray(mu_pred_baseline, dtype=float)
    mp = np.asarray(mu_pred_optimized, dtype=float)
    if abs_mag:
        mo, mb, mp = np.abs(mo), np.abs(mb), np.abs(mp)
    ax.bar(idx - w, mo, width=w,
           yerr=np.asarray(mu_obs_err, dtype=float), capsize=3,
           label="|μ_obs|" if abs_mag else "μ_obs",
           color="skyblue", edgecolor="black", linewidth=1.5)
    ax.bar(idx, mb, width=w,
           label="|μ_baseline|" if abs_mag else "μ_baseline",
           color="lightgray", edgecolor="black",
           linewidth=1.5, hatch="\\\\")
    ax.bar(idx + w, mp, width=w,
           label="|μ_optimized|" if abs_mag else "μ_optimized",
           color="lightgreen", edgecolor="black",
           linewidth=1.5, hatch="//")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(i)) for i in img_numbers])
    ax.set_xlabel("Image Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("|μ|" if abs_mag else "μ", fontsize=12, fontweight="bold")
    ax.set_title("Magnification", fontsize=12, fontweight="bold")
    if abs_mag:
        ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=8, ncol=3)

    # right: image plane
    _draw_image_plane(axes[2], obs_pos, pred_pos, crit_segments, caus_segments,
                      subhalos, subhalo_color="purple")
    axes[2].set_title("Image plane", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_file
