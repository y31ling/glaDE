"""DE population scatter for an iteration frame (the ``draw_interval`` animation).

Pure-array interface: the caller extracts each sub-structure's (x, y) candidate
columns from the DE population using the problem's dimension layout and passes
them here. Kept self-contained so the plotting module has no optimizer import.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


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
