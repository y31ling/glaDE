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

    # ---- MCMC (de+mcmc / mcmc) ----
    if args.mode in ("de+mcmc", "mcmc"):
        _run_mcmc(args, cfg, obs, problem, loss_cfg, best_x, write_status)

    _hr("DONE")
    write_status(state="done")
    print("RUN_COMPLETE", flush=True)
    return 0


def _run_de(args, cfg, obs, problem, loss_cfg):
    from core.optimize import optimize
    _hr("Differential Evolution")
    t0 = time.time()

    draw = int(cfg.algorithm.get("Draw_Graph", 0) or 0)
    interval = max(1, int(cfg.algorithm.get("draw_interval", 5) or 5))
    labels = [d.label for d in problem.dims]
    xy_cols = []
    for comp in cfg.components:
        xl, yl = f"{comp.name}.x", f"{comp.name}.y"
        if xl in labels and yl in labels:
            xy_cols.append((labels.index(xl), labels.index(yl)))
    frames_on = bool(draw and xy_cols and args.backend == "cpu")
    frames_dir = os.path.join(args.out, "iterations")
    if frames_on:
        os.makedirs(frames_dir, exist_ok=True)
        from core.plot import plot_iteration

    def on_iter(it, pop, best, energies):
        if it <= 2 or it % 5 == 0:
            print(f"  iter {it:4d}   best_loss = {best:.4f}   "
                  f"elapsed {time.time()-t0:.0f}s", flush=True)
        if frames_on and it % interval == 0:
            try:
                comp_xy = [(pop[:, xi], pop[:, yi]) for xi, yi in xy_cols]
                plot_iteration(comp_xy, energies, obs.positions, it,
                               os.path.join(frames_dir, f"iteration_{it:04d}.png"))
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] frame {it} failed: {exc}", flush=True)

    result = optimize(cfg, backend=args.backend, on_iteration=on_iter,
                      record_population=False)
    _hr("DE result")
    print(f"  best loss  : {result.loss:.4f}   iterations: {result.de.nit}", flush=True)
    for k, v in result.fitted.items():
        print(f"    {k:18s} = {float(v):.6g}", flush=True)
    with open(os.path.join(args.out, "best_params.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"# GLADE DE result  backend={args.backend}  loss={result.loss:.6f}\n")
        for k, v in result.fitted.items():
            fh.write(f"{k} = {float(v):.8g}\n")
    return result


def _run_mcmc(args, cfg, obs, problem, loss_cfg, best_x, write_status):
    import numpy as np
    from core.mcmc import MCMCConfig, plot_mcmc, run_mcmc
    _hr("MCMC (emcee)")
    mcfg = MCMCConfig.from_cfg(cfg)
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

    with open(os.path.join(args.out, "mcmc_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"# GLADE MCMC  backend={args.backend}  acceptance={res.acceptance_fraction:.4f}\n")
        for name, s in res.summary.items():
            fh.write(f"{name} = {s['p50']:.8g}  [{s['p16']:.8g}, {s['p84']:.8g}]\n")
    write_status(mcmc={"acceptance": res.acceptance_fraction,
                       "n_samples": int(res.samples.shape[0]),
                       "corner": os.path.basename(plots["corner"]),
                       "trace": os.path.basename(plots["trace"]),
                       "summary": res.summary})


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
