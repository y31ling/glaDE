#!/usr/bin/env python3
"""Worker that a FindImage run executes inside its own terminal.

Modes (``--mode``):
  findimage  Point-source optimization (the default). ``--optimizer`` picks the
             algorithm: DE (default) / BIPOP-CMA-ES / jSO. Also runs MCMC after
             it when MCMC_ENABLED is set (legacy ``de+mcmc``).
  de+mcmc    DE, then emcee MCMC seeded around the DE best (legacy-like).
  mcmc       emcee MCMC only (no DE); prior = the {lower,upper} bounds; walkers
             start uniformly across the box; no DE-truth overlay.
  calcimage  Find images at the representative parameter values WITHOUT optimizing
             ({lo,hi} collapse to their search-space midpoints — geometric mean for
             mass-like dims, arithmetic mean otherwise). Point-source only; CPU uses
             glafic, GPU uses Rhongomyniad. MCMC_ENABLED is ignored here.
  amoeba     glafic's own simplex optimizer (`optimize`) on a glafic input.
             glade ``.dat`` selections are first converted to temp glafic files;
             native glafic ``.input`` selections are run directly. NOT DE.

``--optimizer`` (DE / BIPOP-CMA-ES / jSO; default = the OPTIMIZER .dat key or DE)
selects the point-source optimization algorithm for the findimage / de+mcmc modes.

All output goes to stdout (tee'd to the job log the WebUI tails over SSE).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for p in (_ROOT, os.path.join(_ROOT, "glafic2", "python"),
          os.path.join(_ROOT, "Rhongomyniad")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _hr(title: str) -> None:
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64, flush=True)


def _make_frame_cb(args, cfg, problem, obs_xy, suptitle_fmt):
    """Per-iteration corner-frame renderer honouring Draw_Graph/draw_interval.

    Returns ``(frame_fn or None, interval)``; ``frame_fn(it, pop, energies)``
    writes ``iterations/iteration_%04d.png`` — the legacy-format corner over
    every DE dimension (see :func:`core.plot.plot_iteration_corner`).
    """
    draw = int(cfg.algorithm.get("Draw_Graph", 0) or 0)
    interval = max(1, int(cfg.algorithm.get("draw_interval", 5) or 5))
    if not draw or problem.ndim == 0:
        return None, interval
    from core.plot import plot_iteration_corner
    frames_dir = os.path.join(args.out, "iterations")
    os.makedirs(frames_dir, exist_ok=True)
    labels = [d.label for d in problem.dims]
    is_log = [d.log for d in problem.dims]
    bounds = problem.bounds
    print(f"  drawing a DE-population corner frame every {interval} iters -> "
          f"{frames_dir}/", flush=True)

    xy_pairs = problem.xy_dim_pairs()

    def frame(it, pop, energies):
        plot_iteration_corner(
            pop, energies, labels, bounds, it,
            os.path.join(frames_dir, f"iteration_{it:04d}.png"),
            is_log=is_log, obs_positions_arcsec=obs_xy,
            suptitle=suptitle_fmt.format(it=it), xy_pairs=xy_pairs)

    return frame, interval


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GLADE run worker")
    ap.add_argument("--backend", required=True, choices=["cpu", "gpu", "glafic"])
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="findimage",
                    choices=["findimage", "de+mcmc", "mcmc", "calcimage", "amoeba"])
    # point-source optimizer (findimage / de+mcmc): DE (default) | BIPOP-CMA-ES |
    # jSO. None -> the OPTIMIZER .dat key -> 'DE'. Aliases are accepted (case-
    # insensitive) via core.optimize.runner.normalize_algorithm.
    ap.add_argument("--optimizer", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    status_path = os.path.join(args.out, "status.json")
    # Stamp the worker's OWN pid (not the terminal emulator's, which is what
    # JobManager records) so the WebUI can tell a crashed/killed/OOM'd run apart
    # from a slow one: a worker that dies before writing a terminal state leaves
    # status at 'running', and JobManager.status() flips it to 'interrupted' once
    # this pid is gone (see webui.jobs._pid_alive).
    status = {"state": "starting", "backend": args.backend, "mode": args.mode,
              "optimizer": args.optimizer, "files": args.files,
              "worker_pid": os.getpid()}

    def write_status(**extra):
        status.update(extra)
        with open(status_path, "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=2)

    write_status()

    opt_disp = (f"  ·  optimizer={args.optimizer}"
                if args.optimizer and args.mode in ("findimage", "de+mcmc")
                else "")
    _hr(f"GLADE run  ·  mode={args.mode}  ·  backend={args.backend}{opt_disp}  ·  "
        f"{len(args.files)} file(s)")
    for f in args.files:
        print(f"  · {f}", flush=True)

    # glafic's native amoeba (`optimize`) — runs the glafic binary, not DE. Glade
    # files are converted to temp glafic files first; native glafic inputs run
    # as-is. Handled before load_config so a native .input never has to parse as
    # a glade .dat.
    if args.mode == "amoeba":
        return _run_amoeba(args, write_status)

    from core.format import load_config
    from core.optimize import build_obs, optimize
    from core.optimize.loss import LossConfig
    from core.optimize.problem import OptProblem

    cfg, issues = load_config(args.files, backend=args.backend, with_defaults=True)
    errors = [i for i in issues if i.is_error]
    if cfg.applied_defaults:
        print(f"\n[defaults] {', '.join(sorted(cfg.applied_defaults))}", flush=True)
    for w in (i for i in issues if not i.is_error):
        print(f"[warning] {w.message}", flush=True)
    if errors:
        print("\n[blocked] configuration errors:", flush=True)
        for e in errors:
            print(f"  ✗ {e.message}", flush=True)
        write_status(state="error", errors=[e.message for e in errors])
        return 2

    # fail fast with a clear message when the GPU stack is absent — otherwise
    # the batched likelihood only fails later, masked as "zero likelihood
    # everywhere" (a misleading prior/range diagnosis).
    if args.backend == "gpu":
        try:
            import torch  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            print(f"\n[blocked] the GPU backend needs PyTorch (Rhongomyniad) but "
                  f"importing torch failed: {exc}", flush=True)
            print("  install torch, or use the CPU/Glafic rail ('MCMC' instead "
                  "of 'MCMC-GPU' for sampling).", flush=True)
            write_status(state="error",
                         errors=["gpu backend unavailable (torch import failed)"])
            return 2

    from core.format.validate import is_extend_mode

    # ---- Calcimage: find images at representative values, no optimization ----
    # (handled before the extend dispatch so it can reject extend configs with a
    # clear message; ndim==0 / fully-locked configs are the primary use case.)
    if args.mode == "calcimage":
        return _run_calcimage(args, cfg, write_status)

    # ---- extended-source (FITS) path: DE/MCMC over the c2calc loss ----------
    if is_extend_mode(cfg):
        return _run_extend(args, cfg, write_status)

    obs = build_obs(cfg)
    problem = OptProblem(cfg)
    loss_cfg = LossConfig.from_cfg(cfg)
    print(f"\noptimizable dimensions ({problem.ndim}): "
          f"{[d.label for d in problem.dims]}", flush=True)
    if problem.ndim == 0:
        print("[blocked] no optimizable {lo,hi} parameters.", flush=True)
        write_status(state="error", errors=["no optimizable parameters"])
        return 2

    write_status(state="running")
    best_x = None
    de_result = None

    # ---- point-source optimization (findimage / de+mcmc) ----
    if args.mode in ("findimage", "de+mcmc"):
        from core.optimize.fine_tuning import resolve_fine_tuning
        ft_spec, _ft_errors, _ft_warns = resolve_fine_tuning(cfg)
        if ft_spec is not None:
            de_result, best_x = _run_fine_tuning(args, cfg, obs, problem,
                                                 loss_cfg, ft_spec,
                                                 write_status)
        else:
            de_result = _run_de(args, cfg, obs, problem, loss_cfg)
            best_x = de_result.x
        write_status(loss=de_result.loss, iterations=de_result.de.nit,
                     algorithm=de_result.algorithm,
                     fitted={k: float(v) for k, v in de_result.fitted.items()})
        # pin the glade_output deliverable (pure Python) before the glafic figure
        # / verification, which can abort the C binding on pathological models.
        _write_glade_output(args, de_result)
        _write_triptych(args, de_result, obs,
                        f"GLADE {de_result.algorithm} result  loss={de_result.loss:.1f}",
                        os.path.join(args.out, "result.png"))
        write_status(triptych="result.png")
        _verify(args, cfg, obs, loss_cfg, de_result.scene, de_result.loss, write_status)

    # ---- MCMC (de+mcmc / mcmc) ----
    if args.mode in ("de+mcmc", "mcmc"):
        _run_mcmc(args, cfg, obs, problem, loss_cfg, best_x, write_status)

    _hr("DONE")
    write_status(state="done")
    print("RUN_COMPLETE", flush=True)
    return 0


_EXT_LABELS = ["pos", "flux", "td", "prior_pt", "pixel", "prior_ext",
               "prior_lens", "penalty"]

# GPU-batched MCMC ensemble size when MCMC_NWALKERS was left at its default.
# emcee's StretchMove updates half the ensemble per likelihood call, so the
# default 32 walkers feed the batched CUDA kernel only 16 candidates at a time
# (GPU ~97% idle, ~2.6x a 24-core CPU pool); 1024 walkers measured ~100x on the
# batched point-source likelihood. See GPU_MCMC_exploration.md.
_GPU_MCMC_DEFAULT_NWALKERS = 1024


def _gpu_batchable(cfg, extend=False):
    """``(ok, reason)`` of the batched-GPU-likelihood predicate for ``cfg``."""
    if extend:
        from core.optimize.batched_extend import can_batch_extend_gpu
        return can_batch_extend_gpu(cfg)
    from core.optimize.batched import can_batch_gpu
    return can_batch_gpu(cfg)


def _point_batch_is_legacy(cfg) -> bool:
    """Whether a batchable point config takes the single-pass analytic
    point-mass pipeline (the configuration the 1024-walker benchmark was
    measured on) rather than the chunked tensor-kernel path."""
    from core.optimize.batched import _legacy_eligible
    return _legacy_eligible(cfg)


def gpu_mcmc_auto_walkers(cfg, extend=False):
    """The walker count a GPU MCMC run will auto-apply because MCMC_NWALKERS
    was left unset, or ``None`` when no auto-raise would happen. Used by the
    WebUI confirm dialog so the displayed default matches what actually runs."""
    if "MCMC_NWALKERS" not in cfg.applied_defaults:
        return None
    if not _gpu_batchable(cfg, extend)[0]:
        return None
    # The ~100x @ 1024-walkers benchmark holds for the single-pass analytic
    # point pipeline. The generalized tensor-kernel path is chunked (the GPU
    # saturates at the chunk size), so more walkers only add per-step cost —
    # leave the ensemble size to the user there.
    if not extend and not _point_batch_is_legacy(cfg):
        return None
    return _GPU_MCMC_DEFAULT_NWALKERS


def _tune_mcmc_for_gpu(args, cfg, mcfg, extend=False):
    """Adjust/annotate an MCMC config for the GPU backend.

    Returns ``(mcfg, gpu_batched)``. When the batched CUDA likelihood applies
    and MCMC_NWALKERS was not set explicitly, the ensemble is raised to
    ``_GPU_MCMC_DEFAULT_NWALKERS``; otherwise only guidance is printed.
    """
    if args.backend != "gpu":
        return mcfg, False
    ok, reason = _gpu_batchable(cfg, extend)
    if not ok:
        print(f"  [warn] batched GPU likelihood unavailable ({reason}); "
              f"falling back to per-walker GPU evaluation, which is usually "
              f"slower than the CPU pool. Consider the CPU 'MCMC' rail unless "
              f"the per-evaluation cost is high.", flush=True)
        return mcfg, False
    print("  GPU-batched likelihood active (emcee vectorize=True: half the "
          "ensemble per CUDA call).", flush=True)
    if "MCMC_NWALKERS" in cfg.applied_defaults:
        auto = gpu_mcmc_auto_walkers(cfg, extend)
        if auto is not None:
            from dataclasses import replace
            print(f"  MCMC_NWALKERS not set -> using the GPU default of "
                  f"{auto} walkers (the CPU default of 32 "
                  f"leaves the GPU idle); set MCMC_NWALKERS to override.",
                  flush=True)
            return replace(mcfg, nwalkers=auto), True
        print(f"  MCMC_NWALKERS not set -> keeping {mcfg.nwalkers}: this "
              f"config batches through the chunked tensor-kernel path (the "
              f"GPU saturates at the chunk size), so extra walkers add "
              f"posterior coverage at proportional per-step cost; set "
              f"MCMC_NWALKERS explicitly for a denser ensemble.", flush=True)
        return mcfg, True
    if mcfg.nwalkers < 256 and (extend or _point_batch_is_legacy(cfg)):
        print(f"  [hint] MCMC_NWALKERS={mcfg.nwalkers} underuses the GPU "
              f"(~2.6x a 24-core CPU pool at 32 walkers vs ~100x at 1024+); "
              f"consider raising it — more walkers also improve posterior "
              f"coverage.", flush=True)
    return mcfg, True


def _cleanup_spec(spec):
    """Remove a temp glafic constraint file written from glade obs arrays."""
    if getattr(spec, "_own_point_file", False) and spec.point_file \
            and os.path.exists(spec.point_file):
        try:
            os.unlink(spec.point_file)
        except OSError:
            pass


def _extend_figure_and_verify(args, cfg, result, write_status, suptitle):
    """Render the observed/model/residual figure and run the independent
    glafic-binary + scipy verification for an extended-source result."""
    out_png = os.path.join(args.out, "result.png")
    try:
        from core.report import make_extend_figure
        make_extend_figure(result, output_file=out_png, suptitle=suptitle)
        write_status(triptych="result.png")
        print(f"  wrote {out_png}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] result figure failed: {exc}", flush=True)

    if bool(cfg.algorithm.get("glafic_verified", True)):
        _hr("Independent verification (glafic binary c2calc + scipy)")
        try:
            from core.verify import verify_extend
            rep = verify_extend(result, args.out)
            if "glafic_total_chi2" in rep:
                print(f"  glafic-binary c2calc total : {rep['glafic_total_chi2']:.4f}",
                      flush=True)
                print(f"  glade total (sum of comps) : "
                      f"{rep.get('glade_total_chi2', float('nan')):.4f}", flush=True)
                if "chi2_rel_diff" in rep:
                    print(f"  relative difference        : "
                          f"{rep['chi2_rel_diff']*100:.3f}%", flush=True)
            if "glafic_n_images" in rep:
                print(f"  glafic findimg images      : {rep['glafic_n_images']}",
                      flush=True)
            sref = rep.get("scipy_reference", {})
            if sref.get("ok"):
                print(f"  scipy source-plane scatter : "
                      f"{sref['source_plane_scatter_mas']:.3f} mas", flush=True)
            for w in rep.get("warnings", []):
                print(f"  [warn] {w}", flush=True)
            try:
                write_status(extend_verify=rep)
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] verification failed: {exc}", flush=True)


def _run_extend(args, cfg, write_status):
    """Extended-source run: DE and/or MCMC over the weighted c2calc loss.

    backend=cpu/glafic drives glafic per candidate; backend=gpu drives
    Rhongomyniad (batched over the population when the config allows)."""
    from core.optimize.problem import OptProblem

    base_dir = os.path.dirname(os.path.abspath(args.files[0]))
    problem = OptProblem(cfg, extend_mode=True)
    print(f"\noptimizable dimensions ({problem.ndim}): "
          f"{[d.label for d in problem.dims]}", flush=True)
    if problem.ndim == 0:
        print("[blocked] no optimizable {lo,hi} parameters. Widen the {v,v} "
              "placeholders from an import into real search ranges.", flush=True)
        write_status(state="error", errors=["no optimizable parameters"])
        return 2

    # extended-source runs are DE-only; surface a non-DE OPTIMIZER selection
    # (CLI --optimizer or the .dat key) clearly instead of silently ignoring it.
    try:
        if _resolved_algorithm(args, cfg) != "DE":
            print("  note: BIPOP-CMA-ES / jSO are point-source only; the "
                  "extended-source path uses Differential Evolution — "
                  "continuing with DE.", flush=True)
    except ValueError:
        pass

    write_status(state="running")
    best_x = None

    # ---- Differential Evolution (findimage / de+mcmc) ----
    if args.mode in ("findimage", "de+mcmc"):
        result = _extend_de(args, cfg, base_dir, write_status, problem)
        best_x = result.x
        _extend_figure_and_verify(
            args, cfg, result, write_status,
            suptitle=f"GLADE extended-source result  loss={result.loss:.1f}")
        _write_glade_output(args, result)
        _cleanup_spec(result.extend_spec)

    # ---- MCMC (de+mcmc / mcmc) ----
    if args.mode in ("de+mcmc", "mcmc"):
        _extend_mcmc(args, cfg, base_dir, problem, best_x, write_status)

    _hr("DONE")
    write_status(state="done")
    print("RUN_COMPLETE", flush=True)
    return 0


def _extend_obs_positions(cfg, base_dir):
    """Observed SN image positions (x, y arcsec, engine frame) for the iteration
    overlay, or an empty (0, 2) array if none are available.

    Parsed from the resolved glafic constraint file (already engine-frame) when
    present, else from glade point arrays; either way the frame matches the
    optimized component centres so the gold-star overlay lines up.
    """
    import numpy as np
    from core.optimize.extend import read_point_file_positions, resolve_path
    cfile = resolve_path(cfg.obs.get("constraint_file"), [base_dir, os.getcwd()])
    if cfile and os.path.exists(cfile):
        try:
            return read_point_file_positions(cfile)
        except Exception:  # noqa: BLE001
            pass
    if "obs_positions_mas_list" in cfg.obs:
        try:
            from core.optimize.scene import build_obs
            return build_obs(cfg).positions
        except Exception:  # noqa: BLE001
            pass
    return np.zeros((0, 2), dtype=float)


def _extend_de(args, cfg, base_dir, write_status, problem):
    import time
    from core.optimize.runner import optimize
    _hr(f"Differential Evolution (extended source · backend={args.backend})")
    if args.backend == "gpu":
        ok, reason = _gpu_batchable(cfg, extend=True)
        print("  GPU-batched objective active (whole DE population per CUDA "
              "pass)." if ok else
              f"  [warn] batched GPU objective unavailable ({reason}); "
              f"evaluating per candidate on the GPU (single process).",
              flush=True)
    t0 = time.time()

    # iteration frames (Draw_Graph / draw_interval) — same knobs as the
    # point-source path: the legacy-format full-parameter corner (the source
    # position is glafic's inner parameter, not a DE dim, so it never appears).
    obs_xy = _extend_obs_positions(cfg, base_dir)
    frame, interval = _make_frame_cb(
        args, cfg, problem, obs_xy,
        "DE population (extended source) — iteration {it}")

    def on_iter(it, pop, best, energies):
        if it <= 2 or it % 5 == 0:
            print(f"  iter {it:4d}   best_loss = {best:.4f}   "
                  f"elapsed {time.time()-t0:.0f}s", flush=True)
        if frame is not None and it % interval == 0:
            try:
                frame(it, pop, energies)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] frame {it} failed: {exc}", flush=True)

    result = optimize(cfg, backend=args.backend, base_dir=base_dir,
                      on_iteration=on_iter, record_population=False)
    comp = result.extend_components or []
    _hr("DE result (extended source)")
    print(f"  best weighted loss : {result.loss:.4f}   iterations: {result.de.nit}",
          flush=True)
    if comp:
        print(f"  glafic c2calc total: {sum(comp):.4f}", flush=True)
        print("  components: " + "  ".join(
            f"{k}={v:.3g}" for k, v in zip(_EXT_LABELS, comp)), flush=True)
    for k, v in result.fitted.items():
        print(f"    {k:20s} = {float(v):.6g}", flush=True)

    with open(os.path.join(args.out, "best_params.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"# GLADE extended-source DE result  loss={result.loss:.8f}\n")
        if comp:
            fh.write(f"# glafic c2calc total = {sum(comp):.8f}\n")
            fh.write("# components: " + ", ".join(
                f"{k}={v:.6g}" for k, v in zip(_EXT_LABELS, comp)) + "\n")
        for k, v in result.fitted.items():
            fh.write(f"{k} = {float(v):.10g}\n")

    status_extra = {"loss": float(result.loss), "iterations": int(result.de.nit),
                    "fitted": {k: float(v) for k, v in result.fitted.items()}}
    if comp:
        status_extra["c2calc_total"] = float(sum(comp))
        status_extra["components"] = {k: float(v) for k, v in zip(_EXT_LABELS, comp)}
    write_status(**status_extra)
    return result


def _extend_mcmc(args, cfg, base_dir, problem, best_x, write_status):
    import numpy as np
    from core.mcmc import MCMCConfig, plot_mcmc, run_mcmc
    from core.optimize.extend import ExtendObjective, build_extend_spec
    from core.optimize.loss import ExtendLossConfig
    from core.optimize.runner import OptResult

    _hr(f"MCMC (emcee · extended source · backend={args.backend})")
    spec = build_extend_spec(cfg, base_dir=base_dir)
    loss_cfg = ExtendLossConfig.from_cfg(cfg)
    mcfg = MCMCConfig.from_cfg(cfg)
    mcfg, gpu_batched = _tune_mcmc_for_gpu(args, cfg, mcfg, extend=True)
    t0 = time.time()
    seeded = best_x is not None
    print(f"  walkers={max(mcfg.nwalkers, 2*problem.ndim+2)}  steps={mcfg.nsteps}  "
          f"burnin={mcfg.burnin}  "
          f"{'(seeded from DE best)' if seeded else '(uniform prior init)'}",
          flush=True)
    if not gpu_batched:
        eng = ("Rhongomyniad (single process)" if args.backend == "gpu"
               else "glafic")
        print(f"  note: each step evaluates {eng} per walker — extended-source "
              "MCMC is slow; use modest MCMC_NSTEPS.", flush=True)

    def on_step(k, sampler):
        acc = float(np.mean(sampler.acceptance_fraction))
        print(f"  mcmc step {k:4d}/{mcfg.nsteps}   accept={acc:.3f}   "
              f"elapsed {time.time()-t0:.0f}s", flush=True)

    try:
        mc_backend = args.backend if args.backend in ("cpu", "glafic", "gpu") else "cpu"
        res = run_mcmc(problem, None, loss_cfg, backend=mc_backend, best_x=best_x,
                       mcmc_cfg=mcfg, on_step=on_step, extend_spec=spec)
        print(f"\n  acceptance = {res.acceptance_fraction:.3f}   "
              f"samples = {res.samples.shape[0]}   ({time.time()-t0:.0f}s)", flush=True)
        if res.acceptance_fraction < 0.01:
            print("  [warn] acceptance ~ 0: the chain barely moved — these are NOT "
                  "a valid posterior. Use de+mcmc (seed from a DE best) and/or "
                  "fewer steps + narrower {lo,hi} ranges.", flush=True)
        print("  posterior (16/50/84 percentiles):", flush=True)
        for name, s in res.summary.items():
            extra = f"  (linear {s['p50_linear']:.3e})" if "p50_linear" in s else ""
            print(f"    {name:20s} = {s['p50']:.5g}  "
                  f"[{s['p16']:.5g}, {s['p84']:.5g}]{extra}", flush=True)

        suptitle = ("MCMC posterior (seeded from DE)" if seeded
                    else "MCMC posterior (MCMC-only)")
        plots = plot_mcmc(res, args.out, suptitle=suptitle)
        with open(os.path.join(args.out, "mcmc_summary.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"# GLADE extended-source MCMC  "
                     f"acceptance={res.acceptance_fraction:.4f}\n")
            for name, s in res.summary.items():
                fh.write(f"{name} = {s['p50']:.10g}  "
                         f"[{s['p16']:.10g}, {s['p84']:.10g}]\n")
        _corner, _trace = plots.get("corner"), plots.get("trace")
        write_status(mcmc={"acceptance": res.acceptance_fraction,
                           "n_samples": int(res.samples.shape[0]),
                           "corner": os.path.basename(_corner) if _corner else None,
                           "trace": os.path.basename(_trace) if _trace else None,
                           "summary": res.summary})

        # MCMC-only: render the posterior-median model + verify it
        if not seeded:
            median = np.array([res.summary[d.label]["p50"] for d in problem.dims])
            comp = ExtendObjective(problem, spec, loss_cfg).components_for(median)
            result = OptResult(
                x=median, loss=float(loss_cfg.combine(comp)) if comp else float("nan"),
                fitted=problem.decode(median), scene=problem.make_scene(median),
                problem=problem, de=None, backend="cpu", extend_spec=spec,
                extend_components=comp, mode="extend")
            _extend_figure_and_verify(args, cfg, result, write_status,
                                      suptitle="GLADE extended-source MCMC median")
            _write_glade_output(args, result)
    finally:
        _cleanup_spec(spec)


_ALGO_LABEL = {"DE": "Differential Evolution", "BIPOP-CMA-ES": "BIPOP-CMA-ES",
               "JSO": "jSO"}


def _resolved_algorithm(args, cfg) -> str:
    """The concrete point-source optimizer this run will use (CLI --optimizer
    wins, else the OPTIMIZER .dat key, else DE)."""
    from core.optimize.runner import normalize_algorithm
    return normalize_algorithm(
        args.optimizer if args.optimizer is not None
        else cfg.algorithm.get("OPTIMIZER", "DE"))


def _run_de(args, cfg, obs, problem, loss_cfg):
    from core.optimize import optimize
    algo = _resolved_algorithm(args, cfg)
    _hr(f"Optimize · {_ALGO_LABEL.get(algo, algo)} "
        f"(backend={args.backend})")
    if args.backend == "gpu":
        ok, reason = _gpu_batchable(cfg)
        print("  GPU-batched objective active (whole DE population per CUDA "
              "pass)." if ok else
              f"  [warn] batched GPU objective unavailable ({reason}); "
              f"evaluating per candidate on the GPU (single process) — "
              f"usually no faster than the CPU pool; consider the CPU rail.",
              flush=True)
    t0 = time.time()

    frame, interval = _make_frame_cb(args, cfg, problem, obs.positions,
                                     "DE population — iteration {it}")

    def on_iter(it, pop, best, energies):
        if it <= 2 or it % 5 == 0:
            print(f"  iter {it:4d}   best_loss = {best:.4f}   "
                  f"elapsed {time.time()-t0:.0f}s", flush=True)
        if frame is not None and it % interval == 0:
            try:
                frame(it, pop, energies)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] frame {it} failed: {exc}", flush=True)

    result = optimize(cfg, backend=args.backend, on_iteration=on_iter,
                      record_population=False, algorithm=args.optimizer)
    _hr(f"{result.algorithm} result")
    print(f"  best loss  : {result.loss:.4f}   iterations: {result.de.nit}", flush=True)
    for k, v in result.fitted.items():
        print(f"    {k:18s} = {float(v):.6g}", flush=True)
    with open(os.path.join(args.out, "best_params.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"# GLADE {result.algorithm} result  backend={args.backend}  "
                 f"loss={result.loss:.8f}\n")
        for k, v in result.fitted.items():
            fh.write(f"{k} = {float(v):.10g}\n")
    return result


def _run_fine_tuning(args, cfg, obs, problem, loss_cfg, spec, write_status):
    """The fine_tuning pipeline: macro -> substructure -> joint polish.

    Per-stage deliverables (best_params.txt + pinned glade_output .dat, both
    pure Python) go into ft_* subdirs as each stage completes -- BEFORE any
    glafic figure/verify can abort the C binding. The winner then flows through
    the normal post-run machinery (triptych / verify / MCMC) unchanged; its
    loss is re-scored under the .dat's OWN ``LOSS_COEF_A/B`` so the top-level
    outputs (status/best_params/triptych/verify) compare like a normal run
    (per-stage files keep each round's own convention).

    Returns ``(winner OptResult, best_x)`` where best_x re-expresses the winner
    in the FULL problem's dimensions, clipped into the user's ``{lo, hi}``
    bounds (round 3's box may lawfully poke past them and an out-of-prior MCMC
    seed collapses the walker init).
    """
    import math

    from core.optimize.fine_tuning import run_fine_tuning

    algos = " / ".join(r.algorithm for r in spec.rounds)
    _hr(f"Fine-tuning · 3 rounds ({algos}) · top_k={spec.top_k} · "
        f"backend={args.backend}")
    if args.optimizer:
        print(f"  [note] --optimizer {args.optimizer} is ignored: fine_tuning "
              f"picks each round's algorithm from the .dat tuple", flush=True)
    if int(cfg.algorithm.get("Draw_Graph", 0) or 0):
        print("  [note] Draw_Graph iteration frames are not rendered under "
              "fine_tuning (each round searches a different sub-space)",
              flush=True)
    t0 = time.time()
    ft_status = {"top_k": spec.top_k, "perturb": spec.perturb, "stages": []}

    def on_iter(it, pop, best, energies):
        if it <= 2 or it % 5 == 0:
            print(f"  iter {it:4d}   best_loss = {best:.4f}   "
                  f"elapsed {time.time()-t0:.0f}s", flush=True)

    def stage_hook(stage, chain, result):
        sub = stage if chain is None else f"{stage}_chain{chain}"
        stage_dir = os.path.join(args.out, f"ft_{sub}")
        os.makedirs(stage_dir, exist_ok=True)
        with open(os.path.join(stage_dir, "best_params.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(f"# GLADE fine_tuning {sub}  {result.algorithm}  "
                     f"loss={result.loss:.8f}\n")
            for k, v in result.fitted.items():
                fh.write(f"{k} = {float(v):.10g}\n")
        try:
            from core.report import write_glade_output
            write_glade_output(result, stage_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {sub} glade_output export failed: {exc}", flush=True)
        ft_status["stages"].append({"stage": stage, "chain": chain,
                                    "loss": float(result.loss),
                                    "algorithm": result.algorithm,
                                    "dir": os.path.basename(stage_dir)})
        write_status(fine_tuning=ft_status)

    ft = run_fine_tuning(cfg, backend=args.backend, spec=spec,
                         on_iteration=on_iter, stage_hook=stage_hook,
                         log=lambda m: print(m, flush=True))

    ft_status["winner_chain"] = ft.winner_chain
    ft_status["chains"] = [
        {"chain": ch.chain, "seed_loss": float(ch.seed_loss),
         "round2_loss": float(ch.round2.loss) if ch.round2 else None,
         "round3_loss": float(ch.round3.loss) if ch.round3 else None,
         "pruned": bool(ch.pruned),
         "final_loss": float(ch.final_loss)}
        for ch in ft.chains]
    write_status(fine_tuning=ft_status)

    result = ft.winner
    # Re-score the winner under the .dat's OWN loss coefficients so every
    # top-level consumer (status, best_params, triptych title, glafic verify)
    # compares in the same convention as a non-fine_tuning run. The per-round
    # (A3/B3) values remain in the ft_* stage files and status["fine_tuning"].
    try:
        from core.optimize.backends import make_backend
        from core.optimize.objective import point_source_loss
        images = make_backend(args.backend).compute_images(result.scene)
        round_loss = float(result.loss)
        result.loss = float(point_source_loss(images, obs, loss_cfg))
        if abs(round_loss - result.loss) > 1e-9:
            print(f"  winner loss {round_loss:.6g} (round-3 A/B) -> "
                  f"{result.loss:.6g} in the .dat's LOSS_COEF convention",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] could not re-score the winner under the .dat "
              f"coefficients: {exc}", flush=True)

    _hr(f"fine_tuning result · winner chain {ft.winner_chain}")
    print(f"  best loss  : {result.loss:.4f}", flush=True)
    for k, v in result.fitted.items():
        print(f"    {k:18s} = {float(v):.6g}", flush=True)
    with open(os.path.join(args.out, "best_params.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(f"# GLADE fine_tuning result (chain {ft.winner_chain})  "
                 f"backend={args.backend}  loss={result.loss:.8f}\n")
        for k, v in result.fitted.items():
            fh.write(f"{k} = {float(v):.10g}\n")

    # re-express the winner in the FULL problem's dimensions (search space,
    # log10 for mass dims), clipped into the user's bounds, so downstream MCMC
    # keeps the user's prior and its walker init cannot collapse on a bound
    best_x = []
    for d in problem.dims:
        v = ft.winner_values.get(d.target)
        if v is None or (d.log and v <= 0.0):
            x = 0.5 * (d.lo + d.hi)
        else:
            x = math.log10(v) if d.log else float(v)
        best_x.append(min(max(x, d.lo), d.hi))
    import numpy as np
    return result, np.asarray(best_x, dtype=float)


def _run_mcmc(args, cfg, obs, problem, loss_cfg, best_x, write_status):
    import numpy as np
    from core.mcmc import MCMCConfig, plot_mcmc, run_mcmc
    _hr(f"MCMC (emcee · backend={args.backend})")
    mcfg = MCMCConfig.from_cfg(cfg)
    mcfg, _gpu_batched = _tune_mcmc_for_gpu(args, cfg, mcfg)
    t0 = time.time()

    def on_step(k, sampler):
        acc = float(np.mean(sampler.acceptance_fraction))
        print(f"  mcmc step {k:4d}/{mcfg.nsteps}   accept={acc:.3f}   "
              f"elapsed {time.time()-t0:.0f}s", flush=True)

    seeded = best_x is not None
    print(f"  walkers={max(mcfg.nwalkers, 2*problem.ndim+2)}  steps={mcfg.nsteps}  "
          f"burnin={mcfg.burnin}  {'(seeded from DE best)' if seeded else '(uniform prior init)'}",
          flush=True)
    res = run_mcmc(problem, obs, loss_cfg, backend=args.backend, best_x=best_x,
                   mcmc_cfg=mcfg, on_step=on_step)
    print(f"\n  acceptance = {res.acceptance_fraction:.3f}   "
          f"samples = {res.samples.shape[0]}   ({time.time()-t0:.0f}s)", flush=True)
    print("  posterior (16/50/84 percentiles):", flush=True)
    for name, s in res.summary.items():
        extra = f"  (mass {s['p50_linear']:.3e})" if "p50_linear" in s else ""
        print(f"    {name:18s} = {s['p50']:.5g}  [{s['p16']:.5g}, {s['p84']:.5g}]{extra}",
              flush=True)

    suptitle = ("MCMC posterior (seeded from DE)" if seeded
                else "MCMC posterior (MCMC-only)")
    plots = plot_mcmc(res, args.out, suptitle=suptitle)

    # for MCMC-only, also render a triptych of the posterior-median model
    if not seeded:
        from core.optimize.runner import OptResult
        median = np.array([res.summary[d.label]["p50"] for d in problem.dims])
        med_result = OptResult(
            x=median, loss=float("nan"), fitted=problem.decode(median),
            scene=problem.make_scene(median), problem=problem, de=None,
            backend=args.backend, mode="point", algorithm="MCMC")
        # glade_output first (survives a glafic figure/verify C abort), then figure
        _write_glade_output(args, med_result)
        _write_triptych(args, med_result, obs, "MCMC posterior-median model",
                        os.path.join(args.out, "result.png"))
        write_status(triptych="result.png")
        _verify(args, cfg, obs, loss_cfg, problem.make_scene(median), None, write_status)

    with open(os.path.join(args.out, "mcmc_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"# GLADE MCMC  backend={args.backend}  acceptance={res.acceptance_fraction:.4f}\n")
        for name, s in res.summary.items():
            fh.write(f"{name} = {s['p50']:.10g}  [{s['p16']:.10g}, {s['p84']:.10g}]\n")
    _c, _t = plots.get("corner"), plots.get("trace")
    write_status(mcmc={"acceptance": res.acceptance_fraction,
                       "n_samples": int(res.samples.shape[0]),
                       "corner": os.path.basename(_c) if _c else None,
                       "trace": os.path.basename(_t) if _t else None,
                       "summary": res.summary})


def _verify(args, cfg, obs, loss_cfg, scene, opt_loss, write_status):
    if not bool(cfg.algorithm.get("glafic_verified", True)):
        return
    _hr("Independent verification (glafic binary)")
    from core.verify import verify_with_glafic
    auto_check = bool(cfg.algorithm.get("auto_check", True))
    rep = verify_with_glafic(scene, obs, args.out, loss_cfg=loss_cfg,
                             opt_loss=opt_loss, auto_check=auto_check)
    if rep.get("ok"):
        print(f"  glafic: {rep.get('glafic_bin', '?')}", flush=True)
        print(f"  glafic images: {rep.get('glafic_n_images')}", flush=True)
        if "glafic_loss" in rep:
            line = f"  glafic loss vs obs: {rep['glafic_loss']:.4f}"
            if "optimizer_loss" in rep:
                line += (f"   (optimizer {rep['optimizer_loss']:.4f}, "
                         f"diff {rep['loss_rel_diff']*100:.1f}%)")
            print(line, flush=True)
            print(f"  glafic max image offset: {rep['glafic_max_delta_mas']:.2f} mas",
                  flush=True)
    for w in rep.get("warnings", []):
        print(f"  [warn] {w}", flush=True)

    # micro-image audit summary (auto_check): per-image cluster counts, the
    # physical loss recomputed with the cluster Sigma|mu|, and a loud warning
    # when a single-root magnification hides an unresolved micro-image cluster.
    audit = rep.get("micro_audit")
    if audit:
        print("\n  -- micro-image audit --", flush=True)
        for i, e in enumerate(audit.get("per_image", []), start=1):
            print(f"  img {i}: n_micro={e.get('n_micro')} "
                  f"sum|mu|={float(e.get('sum_abs_mu', 0.0)):.4g} "
                  f"(single-root {float(e.get('mu_single', 0.0)):.4g})", flush=True)
        pl = audit.get("physical_loss")
        if pl is not None:
            print(f"  physical loss (cluster Sigma|mu|): {float(pl):.4f}", flush=True)
        if audit.get("fake_solution"):
            print("  [WARN] FAKE SOLUTION: a single-root magnification hides an "
                  "unresolved micro-image cluster — this fit is NOT physical "
                  "(see the micro-audit warnings above).", flush=True)

    # scipy-exact ground truth (engine-independent; glafic@1e-5 Sersic is only
    # Romberg-tolerance accurate, so this is the authoritative check)
    from core.verify import reference_check
    print("\n  -- scipy-exact reference (ground truth) --", flush=True)
    rref = reference_check(scene, obs)
    if rref.get("ok"):
        print(f"  run-engine Sersic deflection vs scipy-exact: "
              f"{rref['gpu_sersic_vs_scipy_arcsec']:.2e} arcsec", flush=True)
        print(f"  source-plane self-consistency: "
              f"{rref['source_plane_scatter_mas']:.3f} mas", flush=True)
        bs, fs = rref["backprojected_source"], rref["fitted_source"]
        print(f"  back-projected source ({bs[0]:+.5f}, {bs[1]:+.5f}) "
              f"vs fitted ({fs[0]:+.5f}, {fs[1]:+.5f})", flush=True)
    else:
        print(f"  [info] {rref.get('warning', 'scipy reference unavailable')}", flush=True)
    for w in rref.get("warnings", []):
        print(f"  [warn] {w}", flush=True)
    print("  (verification is informational — the result above is unchanged)", flush=True)
    try:
        write_status(glafic_verify=rep, scipy_reference=rref)
    except Exception:  # noqa: BLE001
        pass


def _write_triptych(args, result, obs, suptitle, out):
    try:
        from core.report import make_triptych
        make_triptych(result, obs, output_file=out, backend=args.backend,
                      suptitle=suptitle)
        print(f"  wrote {out}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] triptych failed: {exc}", flush=True)


def _write_glade_output(args, result) -> None:
    """Write ``glade_output_<runfolder>.dat`` — a COMPLETE glade input with every
    optimizable parameter pinned at its best-fit value, ready to drop back into
    GLADE. Never fatal: a failure only prints a ``[warn]``."""
    try:
        from core.report import write_glade_output
        path = write_glade_output(result, args.out)
        print(f"  wrote {os.path.basename(path)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] glade_output export failed: {exc}", flush=True)


def _run_calcimage(args, cfg, write_status) -> int:
    """Find images at the representative (midpoint) parameter values, no fit.

    Reuses the standard post-run machinery: triptych, glafic verification (incl.
    the micro-image audit) and the glade_output export. Robust for ndim==0
    (fully-locked) configs. MCMC_ENABLED is ignored here.
    """
    from core.format.validate import is_extend_mode
    from core.optimize import build_obs, make_backend
    from core.optimize.loss import LossConfig
    from core.optimize.runner import calcimage

    if is_extend_mode(cfg):
        print("[blocked] calcimage is point-source only; extended-source configs "
              "use Optimize (Differential Evolution) on the CPU/GPU rails.",
              flush=True)
        write_status(state="error",
                     errors=["calcimage does not support extended-source configs"])
        return 2
    if cfg.algorithm.get("MCMC_ENABLED", False):
        print("  note: MCMC_ENABLED is set but is IGNORED in calcimage mode "
              "(calcimage computes images once; it does not sample).", flush=True)

    _hr(f"Calcimage · images at representative values (backend={args.backend})")
    write_status(state="running")
    try:
        result = calcimage(cfg, backend=args.backend)
    except Exception as exc:  # noqa: BLE001
        print(f"[blocked] calcimage failed: {exc}", flush=True)
        write_status(state="error", errors=[f"calcimage failed: {exc}"])
        return 2

    obs = build_obs(cfg)
    loss_cfg = LossConfig.from_cfg(cfg)
    print(f"\n  {result.problem.ndim} optimizable dim(s) collapsed to their "
          f"representative midpoints:", flush=True)
    for k, v in result.fitted.items():
        print(f"    {k:18s} = {float(v):.6g}", flush=True)

    try:
        images = make_backend(args.backend).compute_images(result.scene) or []
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] could not re-list images for the table: {exc}", flush=True)
        images = []
    _hr("Calcimage result")
    print(f"  found {len(images)} image(s):", flush=True)
    if images:
        print(f"  {'#':>3}  {'x [arcsec]':>13}  {'y [arcsec]':>13}  {'mu':>13}",
              flush=True)
        for i, im in enumerate(images, start=1):
            print(f"  {i:>3}  {float(im[0]):>13.6f}  {float(im[1]):>13.6f}  "
                  f"{float(im[2]):>13.6f}", flush=True)
    print(f"  informative point-source loss: {result.loss:.6g}", flush=True)

    with open(os.path.join(args.out, "best_params.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"# GLADE calcimage  backend={args.backend}  "
                 f"loss={result.loss:.8f}\n")
        for k, v in result.fitted.items():
            fh.write(f"{k} = {float(v):.10g}\n")

    write_status(loss=float(result.loss), iterations=0,
                 fitted={k: float(v) for k, v in result.fitted.items()})
    # write the primary deliverable (glade_output, pure Python) BEFORE the
    # figure / glafic verification: a pathological model can abort the glafic
    # C binding (a buffer overflow it cannot be caught in Python), so pin the
    # result to disk first.
    _write_glade_output(args, result)
    _write_triptych(args, result, obs, f"GLADE calcimage  loss={result.loss:.1f}",
                    os.path.join(args.out, "result.png"))
    write_status(triptych="result.png")
    _verify(args, cfg, obs, loss_cfg, result.scene, result.loss, write_status)

    _hr("DONE")
    write_status(state="done")
    print("RUN_COMPLETE", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# glafic native amoeba (`optimize`) rail
# --------------------------------------------------------------------------- #

def _glafic_prefix(text: str) -> str:
    """The ``prefix`` glafic writes its output files with (default ``out``)."""
    import re
    m = re.search(r"(?m)^\s*prefix\s+(\S+)", text)
    return m.group(1) if m else "out"


def _stage_native_input(input_path: str, job_dir: str) -> tuple:
    """Copy a native glafic .input plus the files it loads into the job dir.

    Returns ``(staged_input_path, copied_basenames)``. Keeping everything in the
    job dir means glafic's outputs (``<prefix>_*.dat``) land there too — retrievable
    via the run-result API — instead of polluting the user's InputFiles/ tree.
    """
    import shutil
    src_dir = os.path.dirname(input_path)
    with open(input_path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    staged = os.path.join(job_dir, os.path.basename(input_path))
    shutil.copy2(input_path, staged)
    file_cmds = ("readobs_point", "parprior", "mapprior", "readobs_extend",
                 "readnoise_extend", "readpsf", "galfile", "srcfile")
    copied = []
    for line in text.splitlines():
        toks = line.split("#")[0].split()
        if len(toks) >= 2 and toks[0] in file_cmds:
            for tok in toks[1:]:
                cand = os.path.join(src_dir, tok)
                dst = os.path.join(job_dir, os.path.basename(tok))
                if os.path.isfile(cand) and os.path.abspath(cand) != os.path.abspath(dst):
                    shutil.copy2(cand, dst)
                    copied.append(os.path.basename(tok))
    return staged, copied


def _exec_glafic_stream(bin_path: str, input_path: str, run_dir: str) -> int:
    """Run glafic on *input_path* in *run_dir*, streaming output to stdout.

    A wall-clock timeout bounds the run: glafic's amoeba (downhill simplex) can
    stall indefinitely on a pathological model, which would otherwise block the
    worker forever with the job stuck at 'running' and no reaper. The output is
    pumped on a daemon thread so `proc.wait(timeout=...)` can fire even when glafic
    goes silent; on expiry the process is killed and a non-zero rc (124, the
    `timeout(1)` convention) is returned. The limit comes from
    GLAFIC_AMOEBA_TIMEOUT (seconds; default 3600; 0 = no limit)."""
    import subprocess
    import threading

    raw = os.environ.get("GLAFIC_AMOEBA_TIMEOUT", "3600")
    try:
        timeout = float(raw)
    except ValueError:
        print(f"  [warn] ignoring malformed GLAFIC_AMOEBA_TIMEOUT={raw!r}; "
              f"using 3600s", flush=True)
        timeout = 3600.0

    proc = subprocess.Popen([bin_path, os.path.basename(input_path)], cwd=run_dir,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)

    def _pump():
        for line in proc.stdout:                   # streamed -> tee'd to job.log
            print(line.rstrip("\n"), flush=True)

    pump = threading.Thread(target=_pump, daemon=True)
    pump.start()
    try:
        proc.wait(timeout=timeout if timeout > 0 else None)
    except subprocess.TimeoutExpired:
        print(f"\n[timeout] glafic amoeba exceeded {timeout:.0f}s without "
              f"finishing; terminating it. Raise GLAFIC_AMOEBA_TIMEOUT (0 = no "
              f"limit) if the model legitimately needs longer.", flush=True)
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        pump.join(timeout=5)
        return 124
    pump.join(timeout=5)
    return proc.returncode


def _read_glafic_chi2(path: str):
    """Final ``chi^2`` from a glafic ``_optresult.dat`` (last occurrence)."""
    import re
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            vals = re.findall(r"chi\^2\s*=\s*([\d.eE+-]+)", fh.read())
    except OSError:
        return None
    try:
        return float(vals[-1]) if vals else None
    except ValueError:
        return None


def _amoeba_figure(args, cfg, run_dir, prefix, images, chi2, write_status):
    """Render the standard triptych from glafic's amoeba outputs (point + crit)."""
    import numpy as np
    from core.optimize.matching import match_images, select_images
    from core.optimize.scene import build_obs
    from core.plot import plot_triptych, read_critical_curves

    obs = build_obs(cfg)
    sel = select_images(images, obs.n) if images else None
    if sel is None:
        print(f"  [warn] glafic found {0 if not images else len(images)} image(s) "
              f"(expected {obs.n}); skipping the result figure.", flush=True)
        return
    pred_pos = np.array([[im[0], im[1]] for im in sel], dtype=float)
    pred_mag = np.array([im[2] for im in sel], dtype=float)
    matched_pos, matched_mag, delta = match_images(
        obs.positions, pred_pos, pred_mag, obs.center_offset)

    crit, caus = [], []
    crit_path = os.path.join(run_dir, f"{prefix}_crit.dat")
    if os.path.isfile(crit_path) and os.path.getsize(crit_path) > 0:
        try:
            crit, caus = read_critical_curves(crit_path)
        except Exception:  # noqa: BLE001
            crit, caus = [], []

    abs_mag = bool(cfg.algorithm.get("abs_mag", True))
    title = "glafic amoeba result" + (f"  chi²={chi2:.1f}" if chi2 is not None else "")
    out_png = os.path.join(args.out, "result.png")
    plot_triptych(
        img_numbers=list(range(1, obs.n + 1)),
        delta_pos_mas=delta, sigma_pos_mas=obs.pos_sigma_mas,
        mu_obs=obs.magnifications, mu_obs_err=obs.mag_errors,
        mu_pred=matched_mag, mu_at_obs_pred=matched_mag,
        obs_positions_arcsec=obs.positions, pred_positions_arcsec=matched_pos,
        crit_segments=crit, caus_segments=caus or None,
        output_file=out_png, suptitle=title, abs_mag=abs_mag)
    print(f"  wrote {out_png}", flush=True)
    write_status(triptych="result.png")


def _extract_optresult_model_lines(opt_path: str) -> list:
    """The final amoeba model block (``lens`` / ``point`` / ``extend`` / ``psf``
    command lines) from a glafic ``<prefix>_optresult.dat``.

    ``dump_opt`` appends one block per amoeba restart, each preceded by an
    ``omega = ...`` header (see glafic2/opt_lens.c ``dump_opt`` / ``dump_model``);
    the final best fit is the block after the LAST such header.
    """
    if not os.path.isfile(opt_path):
        return []
    try:
        with open(opt_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    start = 0
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("omega ="):
            start = i + 1
    kinds = ("lens ", "point ", "extend ", "psf ")
    return [ln.strip() for ln in lines[start:] if ln.strip().startswith(kinds)]


def _splice_glafic_model(template: str, model_lines: list) -> str:
    """Replace the template's ``lens`` / ``point`` / ``extend`` / ``psf`` command
    lines with the amoeba-optimized ones (matched by command word, in order)."""
    by_kind: dict = {}
    for ln in model_lines:
        by_kind.setdefault(ln.split()[0], []).append(ln)
    used = {k: 0 for k in by_kind}
    out = []
    for line in template.splitlines():
        toks = line.split("#")[0].split()
        kind = toks[0] if toks else ""
        if kind in by_kind and used[kind] < len(by_kind[kind]):
            out.append(by_kind[kind][used[kind]])
            used[kind] += 1
        else:
            out.append(line)
    return "\n".join(out)


def _amoeba_glade_output(args, run_dir: str, prefix: str, template_path: str) -> None:
    """Best-effort ``glade_output_<runfolder>.dat`` from glafic's amoeba result.

    Parses ``<prefix>_optresult.dat`` for the optimized model, splices it into
    the run's staged glafic input, converts that to glade via core.translate,
    and writes the file. Never crashes the rail: any failure falls back to a
    best-effort file and/or a ``[warn]``.
    """
    try:
        from core.translate import glafic_to_glade
        run_name = os.path.basename(os.path.normpath(os.path.abspath(args.out)))
        out_name = f"glade_output_{run_name}.dat"
        template = ""
        if template_path and os.path.isfile(template_path):
            with open(template_path, encoding="utf-8", errors="replace") as fh:
                template = fh.read()
        model_lines = _extract_optresult_model_lines(
            os.path.join(run_dir, f"{prefix}_optresult.dat"))
        if template and model_lines:
            glafic_text = _splice_glafic_model(template, model_lines)
            note = ("# Built from glafic's amoeba best fit "
                    "(<prefix>_optresult.dat spliced into the run's glafic input).")
        elif template:
            glafic_text = template
            note = ("# WARNING: glafic's amoeba result block was not found; this "
                    "reflects the INPUT model (parameters may be pre-optimization).")
            print("  [warn] glade_output: no amoeba result block found; writing "
                  "the input model as a best-effort fallback.", flush=True)
        else:
            print("  [warn] glade_output: no glafic input template to convert; "
                  "skipped.", flush=True)
            return
        out = glafic_to_glade(glafic_text)
        model = (out.get("model") or "").strip()
        if not model:
            print("  [warn] glade_output: glafic->glade conversion produced no "
                  "model; skipped.", flush=True)
            return
        parts = [f"# GLADE result: {run_name}",
                 "# amoeba (glafic native optimize)", note, "", model]
        if out.get("obs"):
            parts += ["", out["obs"]]
        path = os.path.join(args.out, out_name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(parts) + "\n")
        print(f"  wrote {out_name}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] glade_output export failed: {exc}", flush=True)


def _run_amoeba(args, write_status) -> int:
    """Run glafic's native simplex optimizer (`optimize`), not GLADE's DE.

    A glade ``.dat`` selection is converted to temp glafic files (model +
    readobs_point + parprior) inside the job dir, then optimised; a native glafic
    ``.input`` selection is run verbatim. The figure is rendered for the glade
    path (where we know the observation frame).
    """
    from core.translate import glade_to_glafic, looks_like_glafic_input
    from core.verify import find_glafic_bin

    _hr("glafic amoeba (native `optimize`)")
    bin_path = find_glafic_bin()
    if not bin_path:
        print("[blocked] glafic binary not found; build it (bootstrap_linux.sh) "
              "or set GLAFIC_HOME.", flush=True)
        write_status(state="error", errors=["glafic binary not found"])
        return 2
    print(f"  glafic: {bin_path}", flush=True)

    texts = {}
    for f in args.files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                texts[f] = fh.read()
        except OSError as exc:
            print(f"[blocked] cannot read {f}: {exc}", flush=True)
            write_status(state="error", errors=[f"cannot read {f}"])
            return 2
    native = [f for f in args.files if looks_like_glafic_input(texts[f])]
    glade = [f for f in args.files if f not in native]
    if native and glade:
        print("[blocked] mixed selection: choose EITHER glade .dat file(s) OR a "
              "single glafic .input — not both.", flush=True)
        write_status(state="error", errors=["mixed glade/glafic selection"])
        return 2

    cfg = None
    if native:
        if len(native) > 1:
            print(f"  note: {len(native)} glafic inputs selected; running the first "
                  f"({os.path.basename(native[0])}).", flush=True)
        src_input = os.path.abspath(native[0])
        run_dir = args.out                             # keep glafic outputs in the job dir
        input_path, copied = _stage_native_input(src_input, run_dir)
        prefix = _glafic_prefix(texts[native[0]])
        print(f"  native glafic input: {src_input}", flush=True)
        print(f"  staged into the job dir ({len(copied)} referenced file(s) copied)",
              flush=True)
        if "optimize" not in texts[native[0]]:
            print("  [warn] this glafic input has no `optimize` command; it will run "
                  "verbatim (no amoeba fit).", flush=True)
    else:
        from core.format import load_config
        from core.format.validate import is_extend_mode
        cfg, issues = load_config(args.files, backend="glafic", with_defaults=True)
        if cfg.applied_defaults:
            print(f"\n[defaults] {', '.join(sorted(cfg.applied_defaults))}", flush=True)
        for w in (i for i in issues if not i.is_error):
            print(f"[warning] {w.message}", flush=True)
        errors = [i for i in issues if i.is_error]
        if errors:
            print("\n[blocked] configuration errors:", flush=True)
            for e in errors:
                print(f"  ✗ {e.message}", flush=True)
            write_status(state="error", errors=[e.message for e in errors])
            return 2
        if is_extend_mode(cfg):
            print("[blocked] the glafic amoeba rail fits point sources; "
                  "extended-source configs use the CPU/GPU rails.", flush=True)
            write_status(state="error", errors=["amoeba rail does not support extend mode"])
            return 2
        out = glade_to_glafic(cfg, base_name="amoeba")
        if not out["optimize"]:
            print("[blocked] no {lo,hi} optimizable parameters — glafic amoeba has "
                  "nothing to fit. Widen at least one parameter into {lo, hi}.",
                  flush=True)
            write_status(state="error", errors=["no optimizable parameters"])
            return 2
        if not out["constraint"]:
            print("[blocked] no point-source observations (obs_*_list) found; "
                  "glafic amoeba needs image constraints.", flush=True)
            write_status(state="error", errors=["no observation constraints"])
            return 2
        run_dir = args.out
        input_path = os.path.join(run_dir, "amoeba_model.input")
        with open(input_path, "w", encoding="utf-8") as fh:
            fh.write(out["model"])
        with open(os.path.join(run_dir, "amoeba_obs.dat"), "w", encoding="utf-8") as fh:
            fh.write(out["constraint"])
        with open(os.path.join(run_dir, "amoeba_prior.dat"), "w", encoding="utf-8") as fh:
            fh.write(out["prior"])
        prefix = _glafic_prefix(out["model"])
        nflag = sum(1 for ln in out["prior"].splitlines() if ln.startswith("range"))
        print("  converted glade -> temp glafic files (amoeba_model.input, "
              "amoeba_obs.dat, amoeba_prior.dat)", flush=True)
        print(f"  optimizing {nflag} parameter(s) via glafic's amoeba simplex",
              flush=True)

    write_status(state="running")
    _hr("glafic optimize (amoeba simplex)")
    rc = _exec_glafic_stream(bin_path, input_path, run_dir)
    if rc != 0:
        # glafic's terminator() exits non-zero on any hard error; surface that as a
        # failed job (consistent with the other rails) rather than a green "done".
        print(f"\n[blocked] glafic exited with code {rc} (see its output above).",
              flush=True)
        write_status(state="error", errors=[f"glafic exited with code {rc}"])
        return 2

    chi2 = _read_glafic_chi2(os.path.join(run_dir, f"{prefix}_optresult.dat"))
    images = None
    try:
        from core.verify import _read_glafic_point
        images = _read_glafic_point(os.path.join(run_dir, f"{prefix}_point.dat"))
    except Exception:  # noqa: BLE001
        images = None

    _hr("glafic amoeba result")
    if chi2 is not None:
        print(f"  best chi^2  : {chi2:.6g}", flush=True)
    print(f"  images found: {len(images) if images else 0}", flush=True)
    with open(os.path.join(args.out, "best_params.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"# glafic amoeba result  prefix={prefix}\n")
        if chi2 is not None:
            fh.write(f"chi^2 = {chi2:.10g}\n")
        for i, im in enumerate(images or [], start=1):
            fh.write(f"image{i} = x {im[0]:.8g}  y {im[1]:.8g}  mag {im[2]:.8g}\n")
    if chi2 is not None:
        write_status(loss=float(chi2))

    # glade_output first (pure Python): a pathological point-mass model can abort
    # the glafic C binding inside the figure's critical-curve pass, so pin the
    # exportable result to disk before rendering.
    _amoeba_glade_output(args, run_dir, prefix, input_path)

    if cfg is not None:
        try:
            _amoeba_figure(args, cfg, run_dir, prefix, images, chi2, write_status)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] result figure failed: {exc}", flush=True)

    _hr("DONE")
    write_status(state="done")
    print("RUN_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
