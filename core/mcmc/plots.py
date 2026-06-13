"""MCMC corner + trace plots (self-contained; Agg backend)."""
from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# Cap the number of points/steps/walkers actually drawn so a huge run (many
# walkers × steps, e.g. the 1024-walker GPU-batched MCMC) does not make
# plotting OOM/hang. Down-sampling is cosmetic only.
_MAX_CORNER_POINTS = 40000
_MAX_TRACE_STEPS = 2000
_MAX_TRACE_WALKERS = 256


def _disp_labels(labels, is_log):
    out = []
    for lab, lg in zip(labels, is_log):
        out.append(f"log10({lab})" if lg else lab)
    return out


def plot_corner(samples, labels, is_log, output_file, truths=None,
                suptitle: Optional[str] = None):
    """Corner plot. ``truths`` (DE best, search space) draws red lines; pass None
    for MCMC-only so no truth line appears. Returns ``output_file``."""
    import corner
    disp = _disp_labels(labels, is_log)
    samples = np.asarray(samples)
    if samples.shape[0] > _MAX_CORNER_POINTS:   # cosmetic down-sample for speed
        idx = np.random.default_rng(0).choice(
            samples.shape[0], _MAX_CORNER_POINTS, replace=False)
        samples = samples[idx]
    fig = corner.corner(samples, labels=disp,
                        quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        title_kwargs={"fontsize": 9}, label_kwargs={"fontsize": 9},
                        truths=(list(truths) if truths is not None else None),
                        truth_color="red")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_file


def plot_trace(chain, labels, is_log, burnin, output_file):
    """Per-parameter walker traces with a burn-in marker. ``chain`` is
    (nsteps, nwalkers, ndim). Returns ``output_file``."""
    chain = np.asarray(chain)
    nsteps, nwalkers, ndim = chain.shape
    if nsteps > _MAX_TRACE_STEPS:               # thin the step axis for speed
        step = int(np.ceil(nsteps / _MAX_TRACE_STEPS))
        chain = chain[::step]
        burnin = max(0, burnin // step)
    if nwalkers > _MAX_TRACE_WALKERS:           # draw a walker subset for speed
        idx = np.random.default_rng(0).choice(nwalkers, _MAX_TRACE_WALKERS,
                                              replace=False)
        chain = chain[:, idx, :]
    disp = _disp_labels(labels, is_log)
    fig, axes = plt.subplots(ndim, 1, figsize=(9, 1.7 * ndim + 1), sharex=True,
                             squeeze=False)
    for i in range(ndim):
        ax = axes[i, 0]
        ax.plot(chain[:, :, i], color="k", alpha=0.25, linewidth=0.5)
        ax.axvline(burnin, color="red", linestyle="--", linewidth=1.2,
                   label="burn-in" if i == 0 else None)
        ax.set_ylabel(disp[i], fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.4)
    axes[-1, 0].set_xlabel("step", fontsize=10)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("MCMC walker traces", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_file, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output_file


def plot_mcmc(result, out_dir, suptitle: Optional[str] = None) -> dict:
    """Write corner + trace for an :class:`MCMCResult`. Returns the file paths.

    The DE-truth red line appears only when ``result.de_truth`` is set
    (i.e. DE+MCMC); MCMC-only has no truth line.
    """
    os.makedirs(out_dir, exist_ok=True)
    corner_path = os.path.join(out_dir, "mcmc_corner.png")
    trace_path = os.path.join(out_dir, "mcmc_trace.png")
    out: dict = {}
    # a degenerate chain (e.g. acceptance ~ 0) has ~no spread; corner can't draw
    # contours. Plot what we can but never let a plot failure abort the run.
    try:
        plot_corner(result.samples, result.param_names, result.is_log, corner_path,
                    truths=result.de_truth, suptitle=suptitle)
        out["corner"] = corner_path
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] corner plot skipped: {exc}", flush=True)
    try:
        plot_trace(result.chain, result.param_names, result.is_log, result.burnin,
                   trace_path)
        out["trace"] = trace_path
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] trace plot skipped: {exc}", flush=True)
    return out
