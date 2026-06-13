"""DE population figures for an iteration frame (the ``draw_interval`` animation).

Two renderers, both pure-array (no optimizer import):

* :func:`plot_iteration_corner` -- the legacy-style N x N corner over EVERY
  searched dimension: diagonal = per-parameter histogram, lower triangle = the
  2D population cross-section of each parameter pair coloured by objective.
  This is the frame both the point and extend paths draw.
* :func:`plot_iteration` -- the older single-axes (x, y) component scatter,
  kept for backwards compatibility.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


def _xy_pair_panels(labels: Sequence[str]) -> dict:
    """Map ``(row, col)`` -> True for lower-triangle panels whose axes are one
    component's ``<name>.x`` (x-axis) and ``<name>.y`` (y-axis) -- the panels
    where observed image positions can be overlaid directly."""
    panels = {}
    idx = {lab: k for k, lab in enumerate(labels)}
    for lab, k in idx.items():
        if not lab.endswith(".x"):
            continue
        ky = idx.get(lab[:-2] + ".y")
        if ky is None:
            continue
        row, col = (ky, k) if ky > k else (k, ky)
        # x-axis is dim ``col``: stars go (x, y) if col is the .x dim, else flipped
        panels[(row, col)] = (col == k)
    return panels


def plot_iteration_corner(population,
                          energies,
                          labels: Sequence[str],
                          bounds: Sequence,
                          iteration_num: int,
                          output_file: str,
                          is_log: Optional[Sequence[bool]] = None,
                          obs_positions_arcsec=None,
                          suptitle: Optional[str] = None,
                          xy_pairs: Optional[Sequence] = None):
    """Legacy-format corner of the whole DE population at one iteration.

    ``population`` is ``(npop, ndim)`` in real search space (mass-like dims are
    log10), ``energies`` aligns row-wise. Diagonal panels are histograms of each
    dimension; each lower-triangle panel scatters the population over that
    parameter pair, coloured by objective (lower = better). The best candidate
    is marked with a red cross. Panels whose two axes are one component's
    ``.x``/``.y`` pair additionally overlay the observed image positions as
    gold stars. Returns ``output_file``.
    """
    pop = np.asarray(population, dtype=float)
    if pop.ndim == 1:
        pop = pop[:, None]
    n = pop.shape[1]
    energies = np.asarray(energies, dtype=float)
    finite = energies[np.isfinite(energies)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.percentile(finite, 90)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    best = int(np.argmin(np.where(np.isfinite(energies), energies, np.inf))) \
        if energies.size else None

    is_log = list(is_log) if is_log is not None else [False] * n
    disp = [f"log10({lab})" if lg else lab for lab, lg in zip(labels, is_log)]
    obs = (np.asarray(obs_positions_arcsec, dtype=float)
           if obs_positions_arcsec is not None else np.zeros((0, 2)))
    if not obs.size:
        star_panels = {}
    elif xy_pairs is not None:
        # explicit (kx, ky) dim-index pairs from the caller — resolves shared
        # user-variable centres, where the dim label is the variable name and
        # the '<comp>.x'/'<comp>.y' suffix convention does not hold
        star_panels = {}
        for kx, ky in xy_pairs:
            row, col = (ky, kx) if ky > kx else (kx, ky)
            star_panels[(row, col)] = (col == kx)
    else:
        star_panels = _xy_pair_panels(list(labels))

    panel = 1.8 if n <= 8 else 1.35
    side = max(6.0, panel * n)
    dpi = 110 if n <= 10 else 90
    lab_fs = max(5, 9 - n // 4)
    tick_fs = max(4, lab_fs - 2)

    fig = plt.figure(figsize=(side, side))
    gs = fig.add_gridspec(n, n, left=0.06, right=0.97, bottom=0.06, top=0.94,
                          hspace=0.08, wspace=0.08)
    sc = None
    for i in range(n):
        for j in range(i + 1):
            ax = fig.add_subplot(gs[i, j])
            if i == j:
                if finite.size and np.isfinite(pop[:, i]).any():
                    ax.hist(pop[:, i], bins=20, range=bounds[i],
                            alpha=0.65, color="steelblue")
                ax.set_xlim(bounds[i])
            else:
                sc = ax.scatter(pop[:, j], pop[:, i], c=energies,
                                cmap="viridis_r", vmin=vmin, vmax=vmax,
                                s=7, alpha=0.6, edgecolors="none")
                if best is not None:
                    ax.plot(pop[best, j], pop[best, i], "+", color="red",
                            markersize=7, markeredgewidth=1.4, zorder=6)
                key = (i, j)
                if key in star_panels:
                    ox, oy = (obs[:, 0], obs[:, 1]) if star_panels[key] \
                        else (obs[:, 1], obs[:, 0])
                    ax.scatter(ox, oy, marker="*", s=60, color="gold",
                               edgecolors="black", linewidths=0.6, zorder=5)
                ax.set_xlim(bounds[j])
                ax.set_ylim(bounds[i])
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(labelsize=tick_fs, length=2)
            ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.3)
            if i == n - 1:
                ax.set_xlabel(disp[j], fontsize=lab_fs)
                ax.tick_params(axis="x", labelrotation=45)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i != 0:
                ax.set_ylabel(disp[i], fontsize=lab_fs)
            else:
                ax.tick_params(labelleft=False)

    if sc is not None:
        cax = fig.add_axes([0.70, 0.92, 0.26, 0.012])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_label("objective (lower = better)", fontsize=lab_fs)
        cb.ax.tick_params(labelsize=tick_fs)
    fig.suptitle(suptitle or f"DE population — iteration {iteration_num}",
                 fontsize=max(11, lab_fs + 4), fontweight="bold",
                 x=0.06, ha="left")

    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)
    return output_file


def plot_iteration(component_xy: Sequence,
                   energies,
                   obs_positions_arcsec,
                   iteration_num: int,
                   output_file: str,
                   suptitle: Optional[str] = None):
    """Scatter the population of each component's (x, y) candidates.

    ``component_xy`` is a list of ``(xs, ys)`` arrays (one per component being
    searched); points are coloured by ``energies`` (lower = better).
    Returns ``output_file``.
    """
    energies = np.asarray(energies, dtype=float)
    obs = np.asarray(obs_positions_arcsec, dtype=float)
    finite = energies[np.isfinite(energies)]
    vmax = float(np.percentile(finite, 90)) if finite.size else 1.0
    vmin = float(np.min(finite)) if finite.size else 0.0

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = None
    for ci, (xs, ys) in enumerate(component_xy):
        sc = ax.scatter(xs, ys, c=energies, cmap="viridis_r", s=18,
                        vmin=vmin, vmax=vmax, alpha=0.7, edgecolors="none")

    ax.scatter(obs[:, 0], obs[:, 1], marker="*", s=220, color="gold",
               edgecolors="black", linewidths=1.5, label="Observed", zorder=5)
    for i, (xo, yo) in enumerate(obs, start=1):
        ax.text(xo + 0.01, yo + 0.01, f"{i}", va="bottom", ha="left",
                fontsize=10, fontweight="bold", color="darkblue")

    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("objective (lower = better)")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [arcsec]", fontsize=11, fontweight="bold")
    ax.set_ylabel("y [arcsec]", fontsize=11, fontweight="bold")
    ax.set_title(suptitle or f"DE population — iteration {iteration_num}",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_file
