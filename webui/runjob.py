#!/usr/bin/env python3
"""Worker that a FindImage run executes inside its own terminal.

Modes (``--mode``):
  findimage  Differential Evolution only (the default).
  de+mcmc    DE, then emcee MCMC seeded around the DE best (legacy-like).
  mcmc       emcee MCMC only (no DE); prior = the {lower,upper} bounds; walkers
             start uniformly across the box; no DE-truth overlay.

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
                    choices=["findimage", "de+mcmc", "mcmc"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    status_path = os.path.join(args.out, "status.json")
    status = {"state": "starting", "backend": args.backend, "mode": args.mode,
              "files": args.files}

    def write_status(**extra):
        status.update(extra)
        with open(status_path, "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=2)

    write_status()

    from core.format import load_config
    from core.optimize import build_obs, optimize
    from core.optimize.loss import LossConfig
    from core.optimize.problem import OptProblem

    _hr(f"GLADE run  ·  mode={args.mode}  ·  backend={args.backend}  ·  "
        f"{len(args.files)} file(s)")
    for f in args.files:
        print(f"  · {f}", flush=True)

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

    # ---- extended-source (FITS) path: DE/MCMC over the c2calc loss ----------
    from core.format.validate import is_extend_mode
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

    # ---- Differential Evolution (findimage / de+mcmc) ----
    if args.mode in ("findimage", "de+mcmc"):
        de_result = _run_de(args, cfg, obs, problem, loss_cfg)
        best_x = de_result.x
        _write_triptych(args, de_result, obs, f"GLADE DE result  loss={de_result.loss:.1f}",
                        os.path.join(args.out, "result.png"))
        write_status(loss=de_result.loss, iterations=de_result.de.nit,
                     triptych="result.png",
                     fitted={k: float(v) for k, v in de_result.fitted.items()})
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

    write_status(state="running")
    best_x = None

    # ---- Differential Evolution (findimage / de+mcmc) ----
    if args.mode in ("findimage", "de+mcmc"):
        result = _extend_de(args, cfg, base_dir, write_status, problem)
        best_x = result.x
        _extend_figure_and_verify(
            args, cfg, result, write_status,
            suptitle=f"GLADE extended-source result  loss={result.loss:.1f}")
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
    finally:
        _cleanup_spec(spec)


def _run_de(args, cfg, obs, problem, loss_cfg):
    from core.optimize import optimize
    _hr(f"Differential Evolution (backend={args.backend})")
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
                      record_population=False)
    _hr("DE result")
    print(f"  best loss  : {result.loss:.4f}   iterations: {result.de.nit}", flush=True)
    for k, v in result.fitted.items():
        print(f"    {k:18s} = {float(v):.6g}", flush=True)
    with open(os.path.join(args.out, "best_params.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"# GLADE DE result  backend={args.backend}  loss={result.loss:.8f}\n")
        for k, v in result.fitted.items():
            fh.write(f"{k} = {float(v):.10g}\n")
    return result


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
        median = np.array([res.summary[d.label]["p50"] for d in problem.dims])
        _write_triptych_from_candidate(args, problem, obs, median,
                                       "MCMC posterior-median model",
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
    rep = verify_with_glafic(scene, obs, args.out, loss_cfg=loss_cfg, opt_loss=opt_loss)
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


def _write_triptych_from_candidate(args, problem, obs, candidate, suptitle, out):
    from core.optimize.runner import OptResult
    res = OptResult(x=candidate, loss=float("nan"),
                    fitted=problem.decode(candidate),
                    scene=problem.make_scene(candidate), problem=problem,
                    de=None, backend=args.backend)
    _write_triptych(args, res, obs, suptitle, out)


if __name__ == "__main__":
    raise SystemExit(main())
