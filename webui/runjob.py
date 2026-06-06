#!/usr/bin/env python3
"""Worker that a FindImage run executes inside its own terminal.

This is the command auto-typed into the spawned terminal. It loads the selected
glade ``.dat`` files through ``core``, runs the Differential Evolution optimizer
on the chosen backend with live progress, then writes the best-fit parameters and
a result triptych into the job directory. All output goes to stdout (tee'd to the
job log that the WebUI tails over SSE).

Standalone use::

    python webui/runjob.py --backend cpu --out runs/<id> \
        --files InputFiles/constants.dat InputFiles/lens.dat
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# --- make core + the engines importable without relying on a sourced env -----
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for p in (_ROOT, os.path.join(_ROOT, "glafic2", "python"),
          os.path.join(_ROOT, "Rhongomyniad")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _hr(title: str) -> None:
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GLADE FindImage run worker")
    ap.add_argument("--backend", required=True, choices=["cpu", "gpu", "glafic"])
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="job output directory")
    ap.add_argument("--force", action="store_true",
                    help="proceed even if basic variables fall back to defaults")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    status_path = os.path.join(args.out, "status.json")

    def write_status(state, **extra):
        with open(status_path, "w", encoding="utf-8") as fh:
            json.dump({"state": state, "backend": args.backend,
                       "files": args.files, **extra}, fh, indent=2)

    write_status("starting")

    from core.format import load_config, has_errors
    from core.format.diagnostics import WARNING
    from core.optimize import optimize, build_obs

    _hr(f"GLADE run  ·  backend={args.backend}  ·  {len(args.files)} file(s)")
    for f in args.files:
        print(f"  · {f}", flush=True)

    cfg, issues = load_config(args.files, backend=args.backend, with_defaults=True)
    errors = [i for i in issues if i.is_error]
    warnings = [i for i in issues if not i.is_error]
    if cfg.applied_defaults:
        print(f"\n[defaults] using defaults for: "
              f"{', '.join(sorted(cfg.applied_defaults))}", flush=True)
    for w in warnings:
        print(f"[warning] {w.message}", flush=True)
    if errors:
        print("\n[blocked] configuration has errors:", flush=True)
        for e in errors:
            print(f"  ✗ {e.message}", flush=True)
        write_status("error", errors=[e.message for e in errors])
        return 2

    obs = build_obs(cfg)

    from core.optimize.problem import OptProblem
    problem = OptProblem(cfg)
    print(f"\noptimizable dimensions ({problem.ndim}): "
          f"{[d.label for d in problem.dims]}", flush=True)
    if problem.ndim == 0:
        print("[blocked] no optimizable {lo,hi} parameters to search.", flush=True)
        write_status("error", errors=["no optimizable parameters"])
        return 2

    _hr("Differential Evolution")
    write_status("running")
    t0 = time.time()

    # optional per-iteration population frames (CPU mode, if Draw_Graph is on)
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
        print(f"[frames] saving population frames every {interval} iters "
              f"-> {frames_dir}", flush=True)
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
    dt = time.time() - t0

    _hr("Result")
    print(f"  best loss   : {result.loss:.4f}", flush=True)
    print(f"  iterations  : {result.de.nit}   ({dt:.0f}s)", flush=True)
    print("  fitted parameters:", flush=True)
    for k, v in result.fitted.items():
        print(f"    {k:18s} = {float(v):.6g}", flush=True)

    params_path = os.path.join(args.out, "best_params.txt")
    with open(params_path, "w", encoding="utf-8") as fh:
        fh.write(f"# GLADE result  backend={args.backend}  loss={result.loss:.6f}\n")
        for k, v in result.fitted.items():
            fh.write(f"{k} = {float(v):.8g}\n")
    print(f"\n  wrote {params_path}", flush=True)

    triptych_path = os.path.join(args.out, "result.png")
    try:
        from core.report import make_triptych
        make_triptych(result, obs, output_file=triptych_path, backend=args.backend,
                      suptitle=f"GLADE result  loss={result.loss:.1f}")
        print(f"  wrote {triptych_path}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] triptych failed: {exc}", flush=True)

    write_status("done", loss=result.loss, iterations=result.de.nit,
                 seconds=round(dt, 1), best_params=params_path,
                 triptych=triptych_path,
                 fitted={k: float(v) for k, v in result.fitted.items()})
    _hr("DONE")
    print("RUN_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
