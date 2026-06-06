#!/usr/bin/env python3
"""
Shared post-processing utilities for the GPU legacy versions.

All v_*_gpu scripts share the same output expectations as v_pointmass_1_0:

  iteration_NNNN.png            DE population snapshot (mass / param scatter)
  <prefix>_best_params.txt      best DE parameters
  <prefix>_mcmc_chain.dat       flat MCMC chain
  <prefix>_posterior.txt        median ±1σ posterior summary
  <prefix>_corner.png           corner plot of subhalo params
  <prefix>_trace.png            walker trace plot
  <prefix>_mass_posterior_1d.png  per-subhalo mass posterior KDE
  result_<prefix>.png           paper triptych (Δpos / μ / image plane)
  result_<prefix>_compare.png   baseline-vs-optimized triptych
  <prefix>_verify_input.dat     glafic CLI input
  <prefix>_verify_report.txt    Python-vs-glafic comparison

This module exposes helpers that take (mostly) primitive arguments so the
caller stays decoupled from the lens-model specifics.  The plot_paper_style
implementation is reused from v_pointmass_1_0/plot_paper_style.py.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# plot_paper_style: reuse v_pointmass_1_0/plot_paper_style.py
# ---------------------------------------------------------------------------
_PLOT_DIR = Path(__file__).resolve().parent / "v_pointmass_1_0"
if str(_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOT_DIR))
from plot_paper_style import (  # noqa: E402
    plot_paper_style,
    plot_paper_style_compare,
    read_critical_curves,
)


# ===========================================================================
# DE iteration plot (mass / parameter scatter per generation)
# ===========================================================================
def plot_iteration_subhalo_param(
    population,
    iteration_num,
    output_dir,
    active_subhalos,
    params_per_subhalo,
    main_param_idx,
    draw_interval=1,
    main_param_label="log(M)",
    main_param_range=None,
    bounds=None,
    title_prefix="Iteration",
):
    """
    Reproduce v_pointmass_1_0's iteration mass-distribution scatter for any
    GPU model.

    population : (popsize, ndim) np.ndarray (or normalized [0,1]).
    main_param_idx : index *within each subhalo block* of the parameter to
                     plot (e.g. log_m -> 2 for pointmass, 2 for NFW, etc).
    """
    if iteration_num != 0 and iteration_num % draw_interval != 0:
        return None
    n_halos = len(active_subhalos)
    if n_halos == 0:
        return None

    if bounds is not None and population.size > 0 \
            and population.max() <= 1.0 and population.min() >= 0.0:
        denorm = np.zeros_like(population)
        for i in range(population.shape[1]):
            lo, hi = bounds[i]
            denorm[:, i] = population[:, i] * (hi - lo) + lo
        population = denorm

    main_cols = [population[:, i * params_per_subhalo + main_param_idx]
                 for i in range(n_halos)]

    fig = plt.figure(figsize=(4 * n_halos, 4 * n_halos))
    labels = [f"Halo {active_subhalos[i]}" for i in range(n_halos)]

    for i in range(n_halos):
        for j in range(n_halos):
            slot = i * n_halos + j + 1
            if i < j:
                ax = plt.subplot(n_halos, n_halos, slot)
                ax.scatter(main_cols[j], main_cols[i],
                           c=np.arange(len(main_cols[i])),
                           cmap="viridis", alpha=0.5, s=20)
                ax.set_xlabel(f"{labels[j]} {main_param_label}", fontsize=8)
                ax.set_ylabel(f"{labels[i]} {main_param_label}", fontsize=8)
                ax.grid(True, linestyle=":", alpha=0.3)
                if main_param_range is not None:
                    ax.set_xlim(main_param_range[2 * j],
                                main_param_range[2 * j + 1])
                    ax.set_ylim(main_param_range[2 * i],
                                main_param_range[2 * i + 1])
            elif i == j:
                ax = plt.subplot(n_halos, n_halos, slot)
                ax.hist(main_cols[i], bins=20, alpha=0.6, color="steelblue")
                ax.set_xlabel(f"{labels[i]} {main_param_label}", fontsize=8)
                ax.set_ylabel("Count", fontsize=8)
                ax.grid(True, linestyle=":", alpha=0.3)
                if main_param_range is not None:
                    ax.set_xlim(main_param_range[2 * i],
                                main_param_range[2 * i + 1])

    plt.suptitle(f"{title_prefix} {iteration_num}: "
                 f"{main_param_label} Distribution ({n_halos} Sub-halos)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_file = os.path.join(output_dir, f"iteration_{iteration_num:04d}.png")
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_iteration_general(
    population,
    iteration_num,
    output_dir,
    bounds,
    param_labels,
    draw_interval=1,
    title_prefix="Iteration",
):
    """v_none_1_0-style histogram of every parameter (used by v_none_gpu)."""
    if iteration_num != 0 and iteration_num % draw_interval != 0:
        return None
    n_params = population.shape[1]
    if n_params == 0:
        return None

    if population.max() <= 1.0 and population.min() >= 0.0:
        denorm = np.zeros_like(population)
        for i in range(n_params):
            lo, hi = bounds[i]
            denorm[:, i] = population[:, i] * (hi - lo) + lo
        population = denorm

    ncols = min(n_params, 4)
    nrows = (n_params + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)

    for idx in range(n_params):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.hist(population[:, idx], bins=20, alpha=0.6, color="steelblue")
        ax.set_xlabel(param_labels[idx], fontsize=9)
        ax.set_ylabel("Count", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.set_xlim(bounds[idx])
    for idx in range(n_params, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.suptitle(f"{title_prefix} {iteration_num}: Parameter Distribution",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_file = os.path.join(output_dir, f"iteration_{iteration_num:04d}.png")
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_file


# ===========================================================================
# Critical curves via glafic Python module
# ===========================================================================
def compute_critical_curves(
    output_dir,
    output_prefix,
    cosmo,             # (omega, lam, weos, hubble)
    grid,              # (xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev)
    base_lens_lines,   # iterable of full lens tuples for glafic.set_lens
    extra_lens_lines,  # iterable of full lens tuples for glafic.set_lens
    source_z, source_x, source_y,
):
    """
    Drive glafic.writecrit to dump the critical / caustic curves, then read
    them with plot_paper_style.read_critical_curves.

    Returns (crit_segments, caus_segments) or (None, None) on failure.
    """
    try:
        import glafic
    except ImportError:
        print("  [warn] glafic Python module unavailable; skipping crit curves")
        return None, None

    try:
        omega, lam, weos, hubble = cosmo
        xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev = grid
        prefix = f"temp_{output_prefix}_best"
        glafic.init(omega, lam, weos, hubble, prefix,
                    xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev, verb=0)
        all_lens = list(base_lens_lines) + list(extra_lens_lines)
        glafic.startup_setnum(len(all_lens), 0, 1)
        for ll in all_lens:
            glafic.set_lens(*ll)
        glafic.set_point(1, source_z, source_x, source_y)
        glafic.model_init(verb=0)
        glafic.writecrit(source_z)
        crit_path = f"{prefix}_crit.dat"
        crit_segments, caus_segments = read_critical_curves(crit_path)
        glafic.quit()
        return crit_segments, caus_segments
    except Exception as e:
        print(f"  [warn] critical-curve generation failed: {e}")
        try:
            glafic.quit()
        except Exception:
            pass
        return None, None


# ===========================================================================
# Triptych wrappers
# ===========================================================================
def write_result_triptych(
    output_path,
    suptitle,
    obs_positions,
    pred_positions,
    delta_pos_mas,
    sigma_pos_mas,
    mu_obs,
    mu_obs_err,
    mu_pred,
    crit_segments,
    caus_segments=None,
    subhalo_positions=None,
    show_2sigma=False,
):
    plot_paper_style(
        img_numbers=np.arange(1, len(obs_positions) + 1),
        delta_pos_mas=delta_pos_mas,
        sigma_pos_mas=sigma_pos_mas,
        mu_obs=mu_obs,
        mu_obs_err=mu_obs_err,
        mu_pred=mu_pred,
        mu_at_obs_pred=np.asarray(mu_pred).copy(),
        obs_positions_arcsec=obs_positions,
        pred_positions_arcsec=pred_positions,
        crit_segments=crit_segments or [],
        caus_segments=caus_segments,
        suptitle=suptitle,
        output_file=output_path,
        title_left="Position Offset",
        title_mid="Magnification",
        title_right="Image Positions & Critical Curves",
        subhalo_positions=subhalo_positions,
        show_2sigma=show_2sigma,
    )


def write_compare_triptych(
    output_path,
    suptitle,
    obs_positions,
    pred_positions,
    delta_pos_baseline,
    delta_pos_optimized,
    sigma_pos_mas,
    mu_obs,
    mu_obs_err,
    mu_pred_baseline,
    mu_pred_optimized,
    crit_segments,
    caus_segments=None,
    subhalo_positions=None,
    show_2sigma=False,
):
    plot_paper_style_compare(
        img_numbers=np.arange(1, len(obs_positions) + 1),
        delta_pos_mas_baseline=delta_pos_baseline,
        delta_pos_mas_optimized=delta_pos_optimized,
        sigma_pos_mas=sigma_pos_mas,
        mu_obs=mu_obs,
        mu_obs_err=mu_obs_err,
        mu_pred_baseline=mu_pred_baseline,
        mu_pred_optimized=mu_pred_optimized,
        obs_positions_arcsec=obs_positions,
        pred_positions_arcsec=pred_positions,
        crit_segments=crit_segments or [],
        caus_segments=caus_segments,
        suptitle=suptitle,
        output_file=output_path,
        title_left="Position Offset Comparison",
        title_mid="Magnification Comparison",
        title_right="Image Positions & Critical Curves",
        subhalo_positions=subhalo_positions,
        show_2sigma=show_2sigma,
    )


# ===========================================================================
# MCMC pipeline (chain + corner + trace + mass-1D + posterior summary)
# ===========================================================================
def run_mcmc_pipeline(
    log_prob_fn,                # callable, accepts either (params,) or (n,ndim)
    log_prob_vectorized: bool,  # whether log_prob_fn accepts batches
    bounds,
    best_x,
    output_dir,
    output_prefix,
    param_names,                # list[str] length ndim
    corner_labels=None,         # list[str] for corner plot tex labels
    mass_param_indices=None,    # list of (img_idx, ndim_index) for mass posteriors
    config=None,
    title_prefix="GPU",
):
    """
    Run emcee, save the standard outputs.

    config keys (with defaults):
      NWALKERS=32, NSTEPS=2000, BURNIN=300, THIN=2,
      PERTURBATION=0.01, PROGRESS=True, WORKERS=1
    """
    cfg = {
        "NWALKERS": 32, "NSTEPS": 2000, "BURNIN": 300, "THIN": 2,
        "PERTURBATION": 0.01, "PROGRESS": True, "WORKERS": 1,
    }
    if config:
        cfg.update(config)

    try:
        import emcee
        import corner
    except ImportError as e:
        print(f"  [MCMC] missing dependency: {e}.  pip install emcee corner")
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda it, **kw: it  # noqa: E731

    ndim = len(bounds)
    nwalkers = max(int(cfg["NWALKERS"]), 2 * ndim + 2)
    nsteps = int(cfg["NSTEPS"])
    burnin = int(cfg["BURNIN"])
    thin = max(int(cfg["THIN"]), 1)
    pert = float(cfg["PERTURBATION"])
    show_prog = bool(cfg["PROGRESS"])

    print("\n" + "=" * 70)
    print("MCMC posterior sampling")
    print("=" * 70)
    print(f"  ndim={ndim}  walkers={nwalkers}  steps={nsteps}  burnin={burnin}")

    # initial walker positions: best_x ± perturbation * (hi-lo)
    bounds_arr = np.asarray(bounds)
    lo = bounds_arr[:, 0]
    hi = bounds_arr[:, 1]
    width = hi - lo
    rng = np.random.default_rng()
    init = np.empty((nwalkers, ndim))
    for w in range(nwalkers):
        init[w] = np.clip(np.asarray(best_x)
                          + rng.normal(0.0, pert * width), lo, hi)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob_fn, vectorize=log_prob_vectorized)

    iterator = sampler.sample(init, iterations=nsteps)
    if show_prog:
        iterator = tqdm(iterator, total=nsteps, desc="MCMC")
    for _ in iterator:
        pass

    chain = sampler.get_chain()                           # (nsteps, nwalkers, ndim)
    flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    print(f"  effective samples: {len(flat)}")

    chain_path = os.path.join(output_dir, f"{output_prefix}_mcmc_chain.dat")
    np.savetxt(chain_path, flat, header=" ".join(param_names))
    print(f"  saved chain    : {chain_path}")

    # posterior summary
    posterior = {}
    for i, name in enumerate(param_names):
        vals = flat[:, i]
        median = np.median(vals)
        low = np.percentile(vals, 16)
        high = np.percentile(vals, 84)
        posterior[name] = dict(
            median=median, lower=low, upper=high,
            err_plus=high - median, err_minus=median - low)

    posterior_path = os.path.join(output_dir, f"{output_prefix}_posterior.txt")
    with open(posterior_path, "w") as f:
        f.write("# MCMC posterior summary\n")
        f.write(f"# walkers={nwalkers} steps={nsteps} burnin={burnin} thin={thin}\n")
        f.write(f"# effective samples = {len(flat)}\n\n")
        f.write("# parameter median 16% 84% +err -err\n")
        for name in param_names:
            st = posterior[name]
            f.write(f"{name}  {st['median']:.10e}  {st['lower']:.10e}  "
                    f"{st['upper']:.10e}  {st['err_plus']:.10e}  {st['err_minus']:.10e}\n")
        if mass_param_indices:
            f.write("\n# Mass posteriors (M_sun)\n")
            for img_idx, p_idx in mass_param_indices:
                ms = 10 ** flat[:, p_idx]
                med = np.median(ms)
                lo_ = np.percentile(ms, 16)
                hi_ = np.percentile(ms, 84)
                f.write(f"mass_{img_idx}  {med:.10e}  {lo_:.10e}  "
                        f"{hi_:.10e}  {hi_ - med:.10e}  {med - lo_:.10e}\n")
    print(f"  saved posterior: {posterior_path}")

    # corner plot
    if ndim > 0:
        clabels = corner_labels or param_names
        try:
            fig = corner.corner(
                flat, labels=clabels[:ndim],
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True, title_fmt=".3f",
                truths=np.asarray(best_x),
                truth_color="red",
                hist_kwargs={"alpha": 0.75})
            corner_path = os.path.join(output_dir, f"{output_prefix}_corner.png")
            fig.savefig(corner_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved corner   : {corner_path}")
        except Exception as e:
            print(f"  [warn] corner plot failed: {e}")

    # trace plot
    if ndim > 0:
        try:
            fig, axes = plt.subplots(ndim, figsize=(10, 2 * ndim), sharex=True)
            if ndim == 1:
                axes = [axes]
            for i in range(ndim):
                axes[i].plot(chain[:, :, i], alpha=0.3)
                axes[i].axvline(burnin, color="red", linestyle="--", label="Burn-in" if i == 0 else None)
                axes[i].set_ylabel((corner_labels or param_names)[i])
                axes[i].yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("Step")
            axes[0].legend(loc="upper right")
            trace_path = os.path.join(output_dir, f"{output_prefix}_trace.png")
            fig.savefig(trace_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved trace    : {trace_path}")
        except Exception as e:
            print(f"  [warn] trace plot failed: {e}")

    # 1D mass posterior plot
    if mass_param_indices:
        try:
            from scipy.stats import gaussian_kde
            n_mass = len(mass_param_indices)
            fig, axes = plt.subplots(1, n_mass, figsize=(5 * n_mass, 4))
            if n_mass == 1:
                axes = [axes]
            for ax_i, (img_idx, p_idx) in enumerate(mass_param_indices):
                samples_log_m = flat[:, p_idx]
                de_logm = float(np.asarray(best_x)[p_idx])
                kde = gaussian_kde(samples_log_m, bw_method="scott")
                lo_ = min(samples_log_m.min() - 0.3, de_logm - 0.3)
                hi_ = max(samples_log_m.max() + 0.3, de_logm + 0.3)
                xg = np.linspace(lo_, hi_, 500)
                yg = kde(xg)
                ax = axes[ax_i]
                ax.plot(xg, yg, color="steelblue", lw=2)
                ax.fill_between(xg, yg, alpha=0.25, color="steelblue")
                med = np.median(samples_log_m)
                p16 = np.percentile(samples_log_m, 16)
                p84 = np.percentile(samples_log_m, 84)
                ax.axvline(med, color="steelblue", lw=1.5, ls="--",
                           label=f"median = {med:.2f}")
                ax.axvspan(p16, p84, alpha=0.15, color="steelblue", label=r"1$\sigma$")
                ax.axvline(de_logm, color="tomato", lw=2,
                           label=f"DE best = {de_logm:.2f}")
                ax.set_xlabel(r"$\log_{10}(M / M_\odot)$", fontsize=13)
                ax.set_ylabel("Posterior density", fontsize=13)
                ax.set_title(f"{title_prefix} Sub-halo {img_idx} mass", fontsize=12)
                ax.legend(fontsize=9)
                ax.grid(True, linestyle=":", alpha=0.4)
                ax.set_xlim(lo_, hi_)
                ax.set_ylim(bottom=0)
            plt.tight_layout()
            mass1d_path = os.path.join(output_dir, f"{output_prefix}_mass_posterior_1d.png")
            fig.savefig(mass1d_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved mass 1D  : {mass1d_path}")
        except Exception as e:
            print(f"  [warn] mass 1D posterior failed: {e}")

    return dict(chain=chain, flat=flat, posterior=posterior, sampler=sampler)


# ===========================================================================
# glafic CLI verification
# ===========================================================================
def find_glafic_bin(default_path=""):
    if default_path and os.path.isfile(default_path) and os.access(default_path, os.X_OK):
        return default_path
    bin_path = shutil.which("glafic")
    if bin_path:
        return bin_path
    try:
        import glafic as _gl
        mod_dir = os.path.dirname(os.path.abspath(_gl.__file__))
        for rel in ("../glafic", "../../glafic", "./glafic", "../bin/glafic"):
            p = os.path.abspath(os.path.join(mod_dir, rel))
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    except Exception:
        pass
    return None


def write_glafic_input(
    path,
    cosmo,                # (omega, lam, weos, hubble)
    grid,                 # (xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev)
    prefix,
    base_lens_lines,      # iterable of (idx, model, z, p1..p7) tuples
    extra_lens_lines,     # iterable of (idx, model, z, p1..p7) tuples
    source_z, source_x, source_y,
    header_comment="",
):
    """Write a glafic CLI input file that runs `findimg` then quits."""
    omega, lam, weos, hubble = cosmo
    xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev = grid
    all_lines = list(base_lens_lines) + list(extra_lens_lines)
    n_lens = len(all_lines)
    with open(path, "w") as f:
        if header_comment:
            for line in header_comment.splitlines():
                f.write(f"# {line}\n")
        f.write(f"# generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"omega    {omega}\n")
        f.write(f"lambda   {lam}\n")
        f.write(f"weos     {weos}\n")
        f.write(f"hubble   {hubble}\n")
        f.write(f"prefix   {prefix}\n")
        f.write(f"xmin     {xmin}\n")
        f.write(f"ymin     {ymin}\n")
        f.write(f"xmax     {xmax}\n")
        f.write(f"ymax     {ymax}\n")
        f.write(f"pix_ext  {pix_ext}\n")
        f.write(f"pix_poi  {pix_poi}\n")
        f.write(f"maxlev   {maxlev}\n\n")
        f.write(f"startup  {n_lens} 0 1\n")
        for ll in all_lines:
            # ll = (idx, model, z, p1..p7)
            _, model, z, *p7 = ll
            f.write(f"lens   {model}  {z}  "
                    + "  ".join(f"{v:.10e}" for v in p7) + "\n")
        f.write(f"point  {source_z}  {float(source_x):.10e}  {float(source_y):.10e}\n")
        f.write("end_startup\n\nstart_command\nfindimg\nquit\n")


def run_glafic_and_compare(
    output_dir,
    output_prefix,
    cosmo,
    grid,
    base_lens_lines,
    extra_lens_lines,
    source_z, source_x, source_y,
    obs_positions,
    center_offset_x, center_offset_y,
    best_pos_py,    # (n_obs, 2) np array, already with center_offset applied
    best_mag_py,    # (n_obs,) np array (signed magnifications)
    glafic_bin=None,
    pos_tolerance=0.01,
    mag_tolerance_pct=0.1,
    header_comment="",
):
    """
    Generate glafic CLI input, run it, then compare image positions and
    magnifications against the supplied Python-side predictions.  Writes
    <prefix>_verify_input.dat and <prefix>_verify_report.txt.
    """
    bin_path = glafic_bin or find_glafic_bin()
    print("\n" + "=" * 70)
    print("Verification: Python/GPU solver vs glafic CLI")
    print("=" * 70)
    if bin_path is None:
        print("  warn: glafic CLI not found; skipping verification")
        return False
    print(f"  glafic path: {bin_path}")

    verify_input = os.path.join(output_dir, f"{output_prefix}_verify_input.dat")
    verify_prefix = f"{output_prefix}_verify"
    write_glafic_input(
        verify_input, cosmo, grid, verify_prefix,
        base_lens_lines, extra_lens_lines,
        source_z, source_x, source_y,
        header_comment=header_comment,
    )
    print(f"  input file : {verify_input}")

    try:
        proc = subprocess.run(
            [bin_path, os.path.basename(verify_input)],
            cwd=output_dir, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            print(f"  warn: glafic returned {proc.returncode}")
            print(f"  stderr: {proc.stderr.strip()[:500]}")
    except subprocess.TimeoutExpired:
        print("  warn: glafic timed out")
        return False
    except Exception as e:
        print(f"  warn: {e}")
        return False

    verify_pt = os.path.join(output_dir, f"{verify_prefix}_point.dat")
    if not os.path.exists(verify_pt):
        print(f"  warn: output missing: {verify_pt}")
        return False

    try:
        data = np.loadtxt(verify_pt)
    except Exception as e:
        print(f"  warn: failed to read {verify_pt}: {e}")
        return False

    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_imgs = int(data[0, 0])
    print(f"  glafic found {n_imgs} images")
    n_obs = len(obs_positions)
    if n_imgs not in (n_obs, n_obs + 1):
        print(f"  warn: expected {n_obs} or {n_obs+1} images")
        return False

    img_data = data[1:n_imgs + 1, :]
    if n_imgs == n_obs + 1:
        drop = int(np.argmin(np.abs(img_data[:, 2])))
        print(f"  dropped central image (idx {drop}, |μ|={abs(img_data[drop,2]):.4f})")
        img_data = np.delete(img_data, drop, axis=0)

    gl_pos = img_data[:, 0:2].copy()
    gl_pos[:, 0] += center_offset_x
    gl_pos[:, 1] += center_offset_y
    gl_mag = np.abs(img_data[:, 2])
    d = cdist(obs_positions, gl_pos)
    ri, ci = linear_sum_assignment(d)
    order = ci[np.argsort(ri)]
    gl_pos_m = gl_pos[order]
    gl_mag_m = gl_mag[order]

    py_mag_abs = np.abs(np.asarray(best_mag_py))

    max_pos_diff = 0.0
    max_mag_pct = 0.0
    print(f"\n  {'Img':<5} {'Py x[mas]':>12} {'GL x[mas]':>12} {'|Δx|':>8}"
          f"  {'Py y[mas]':>12} {'GL y[mas]':>12} {'|Δy|':>8}")
    print("  " + "-" * 80)
    for k in range(n_obs):
        px = best_pos_py[k, 0] * 1000
        py = best_pos_py[k, 1] * 1000
        gx = gl_pos_m[k, 0] * 1000
        gy = gl_pos_m[k, 1] * 1000
        dxv = abs(px - gx); dyv = abs(py - gy)
        max_pos_diff = max(max_pos_diff, dxv, dyv)
        print(f"  {k+1:<5} {px:>12.3f} {gx:>12.3f} {dxv:>8.3f}"
              f"  {py:>12.3f} {gy:>12.3f} {dyv:>8.3f}")
    print(f"\n  {'Img':<5} {'Py |μ|':>12} {'GL |μ|':>12} {'Δ [%]':>10}")
    print("  " + "-" * 50)
    for k in range(n_obs):
        pm = py_mag_abs[k]
        gm = gl_mag_m[k]
        dmp = abs(pm - gm) / pm * 100 if pm else 0.0
        max_mag_pct = max(max_mag_pct, dmp)
        print(f"  {k+1:<5} {pm:>12.3f} {gm:>12.3f} {dmp:>9.3f}%")
    print(f"\n  max position diff: {max_pos_diff:.6f} mas")
    print(f"  max magnif. diff:  {max_mag_pct:.6f}%")

    if max_pos_diff < pos_tolerance and max_mag_pct < mag_tolerance_pct:
        verdict = "[PASS] consistency verified"
    elif max_pos_diff < 1.0 and max_mag_pct < 1.0:
        verdict = "[OK]   small differences"
    else:
        verdict = "[WARN] large discrepancy — check params"
    print(f"  {verdict}")

    report_path = os.path.join(output_dir, f"{output_prefix}_verify_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Python/GPU solver vs glafic CLI\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Position [mas]:\n")
        f.write(f"{'Img':<5} {'Py x':>12} {'GL x':>12} {'|Δx|':>8}"
                f"  {'Py y':>12} {'GL y':>12} {'|Δy|':>8}\n")
        f.write("-" * 80 + "\n")
        for k in range(n_obs):
            px = best_pos_py[k, 0] * 1000
            py = best_pos_py[k, 1] * 1000
            gx = gl_pos_m[k, 0] * 1000
            gy = gl_pos_m[k, 1] * 1000
            dxv = abs(px - gx); dyv = abs(py - gy)
            f.write(f"{k+1:<5} {px:>12.3f} {gx:>12.3f} {dxv:>8.3f}"
                    f"  {py:>12.3f} {gy:>12.3f} {dyv:>8.3f}\n")
        f.write(f"\nmax position diff = {max_pos_diff:.6f} mas\n\n")
        f.write("Magnification |μ|:\n")
        f.write(f"{'Img':<5} {'Py':>12} {'GL':>12} {'Δ [%]':>10}\n")
        f.write("-" * 50 + "\n")
        for k in range(n_obs):
            pm = py_mag_abs[k]
            gm = gl_mag_m[k]
            dmp = abs(pm - gm) / pm * 100 if pm else 0.0
            f.write(f"{k+1:<5} {pm:>12.3f} {gm:>12.3f} {dmp:>9.3f}%\n")
        f.write(f"\nmax magnif. diff = {max_mag_pct:.6f}%\n")
        f.write(f"\nverdict: {verdict}\n")
    print(f"  saved report: {report_path}")
    return True


# ===========================================================================
# Image-matching helper (reused by all GPU drivers)
# ===========================================================================
def match_images(
    images,                # list of (x, y, mag, ...) from solver
    obs_positions,
    center_offset_x,
    center_offset_y,
    n_obs=None,
):
    """
    Hungarian-match a list of solver image tuples to observed positions.

    Returns (pred_pos, pred_mag, delta_pos_mas) on success or
    (None, None, None) if the image count cannot be matched to n_obs.

    A leading central image (5 → 4) is dropped automatically.
    """
    if n_obs is None:
        n_obs = len(obs_positions)
    if not images:
        return None, None, None
    if len(images) == n_obs + 1:
        drop = int(np.argmin([abs(im[2]) for im in images]))
        images = [im for k, im in enumerate(images) if k != drop]
    if len(images) != n_obs:
        return None, None, None

    pred_pos = np.array([[im[0], im[1]] for im in images]) \
        + np.array([center_offset_x, center_offset_y])
    pred_mag = np.array([im[2] for im in images])
    distances = cdist(obs_positions, pred_pos)
    ri, ci = linear_sum_assignment(distances)
    order = ci[np.argsort(ri)]
    pp = pred_pos[order]
    pm = pred_mag[order]
    delta_mas = np.sqrt(np.sum(((pp - obs_positions) * 1000) ** 2, axis=1))
    return pp, pm, delta_mas
