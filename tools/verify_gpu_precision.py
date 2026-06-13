#!/usr/bin/env python3
"""Verify the batched-GPU ``gpu_precision`` modes (64 / 48 / 32) against the
scipy-exact reference and the glafic binary (at its shipped
TOL_ROMBERG_JHK=1e-5 build — no rebuild needed).

For a representative multi-image candidate the script reports, per precision:
  * batched evaluation speed (ms/candidate over a full DE-sized population);
  * image-position / magnification drift vs the fp64 batched images;
  * the scipy-exact lens-equation check: the found images are back-projected
    through the scipy-exact (Sersic) + fp64-kernel model — the source-plane
    scatter and the offset from the candidate's true source measure how well
    that precision's images solve the lens equation (ground truth);
  * the glafic-binary cross-check: glafic findimg on the same scene, image
    positions matched against each precision's batched images (informational:
    glafic's Sersic is Romberg-limited to ~0.05 mas at the 1e-5 build).

    source env.sh && .venv/bin/python tools/verify_gpu_precision.py
    ... [--files InputFiles/nfw_only.dat] [--pop 180] [--seed 3]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "Rhongomyniad")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.format import load_config                      # noqa: E402
from core.optimize.batched import BatchedGPUObjective, can_batch_gpu  # noqa: E402
from core.optimize.loss import LossConfig                 # noqa: E402
from core.optimize.objective import INVALID_LOSS, point_source_loss  # noqa: E402
from core.optimize.problem import OptProblem              # noqa: E402
from core.optimize.scene import ObsData, build_obs        # noqa: E402
from core.verify import reference_check, verify_with_glafic  # noqa: E402

PRECISIONS = (64, 48, 32)


def _load(files, precision):
    """Config + problem + objective with gpu_precision forced to *precision*."""
    text = "\n".join(open(f, encoding="utf-8").read() for f in files)
    # drop any user-set gpu_precision (duplicate scalar assignment is a
    # parser syntax error) before forcing the precision under test
    text = re.sub(r"(?m)^\s*gpu_precision\s*=[^\n]*$", "", text)
    text += f"\ngpu_precision = {precision}\n"
    tmp = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False,
                                      encoding="utf-8")
    tmp.write(text)
    tmp.close()
    cfg, issues = load_config([tmp.name], backend="gpu", with_defaults=True)
    os.unlink(tmp.name)
    errs = [str(i) for i in issues if i.is_error]
    if errs:
        raise SystemExit(f"config errors: {errs}")
    problem = OptProblem(cfg)
    obj = BatchedGPUObjective(problem, build_obs(cfg), LossConfig.from_cfg(cfg))
    return cfg, problem, obj


def _match(ref_imgs, imgs):
    """Greedy nearest match; returns (max |dpos| arcsec, max rel |dmag|)."""
    if len(ref_imgs) != len(imgs):
        return None, None
    used = set()
    dpos = dmag = 0.0
    for (x, y, m) in ref_imgs:
        best_j, best_d = None, np.inf
        for j, (x2, y2, _m2) in enumerate(imgs):
            if j in used:
                continue
            d = (x - x2) ** 2 + (y - y2) ** 2
            if d < best_d:
                best_d, best_j = d, j
        used.add(best_j)
        x2, y2, m2 = imgs[best_j]
        dpos = max(dpos, float(np.sqrt(best_d)))
        dmag = max(dmag, abs(m2 - m) / max(abs(m), 1e-30))
    return dpos, dmag


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", default=["InputFiles/nfw_only.dat"])
    ap.add_argument("--pop", type=int, default=180)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args(argv)
    os.chdir(_ROOT)

    cfg64, problem, obj64 = _load(args.files, 64)
    ok, reason = can_batch_gpu(cfg64)
    print(f"can_batch_gpu: {ok} {reason!r}")
    if not ok:
        return 1
    obs = build_obs(cfg64)
    loss_cfg = LossConfig.from_cfg(cfg64)

    lo = np.array([b[0] for b in problem.bounds])
    hi = np.array([b[1] for b in problem.bounds])
    rng = np.random.default_rng(args.seed)
    cands = lo + (hi - lo) * rng.uniform(0.05, 0.95,
                                         size=(args.pop, problem.ndim))

    # representative candidate: a fully-imaged one as far from criticality as
    # possible (smallest max |mag|), so the glafic cross-check is not confused
    # by near-caustic multiplicity flips (glafic's Romberg-level Sersic can
    # genuinely form a different image count within ~3e-5 arcsec of a caustic).
    print(f"searching {args.pop} candidates for a stable multi-image "
          f"representative...")
    obj64._ensure_engine()
    if obj64._cache is None:
        obj64._build_cache()
    all_imgs = []
    if obj64._cache["legacy"]:
        # point-mass pipeline: the optimizable points live in _cache["points"],
        # not opt_lenses — dispatch like _evaluate does
        sx_t, sy_t, lm_t = obj64._batch_tensors(cands.T)
        all_imgs = obj64._solve(sx_t, sy_t, lm_t,
                                float(obj64._cache["source_x"]),
                                float(obj64._cache["source_y"]))
    else:
        chunk = obj64._cache["chunk"]
        for s in range(0, args.pop, chunk):
            sub = cands[s:s + chunk]
            arr_t = obj64._torch.tensor(sub.T, dtype=obj64.dtype,
                                        device=obj64.device)
            all_imgs.extend(obj64._solve_general(arr_t))
    best_k, best_key = 0, (np.inf, np.inf)
    for k, imgs in enumerate(all_imgs):
        # prefer exactly obs.n images, then the smallest max |mag|
        key = (abs(len(imgs) - obs.n),
               max((abs(m) for _x, _y, m in imgs), default=np.inf))
        if key < best_key:
            best_k, best_key = k, key
    rep = cands[best_k]
    print(f"  candidate #{best_k}: {len(all_imgs[best_k])} image(s) at fp64 "
          f"(obs has {obs.n}), max |mag| = {best_key[1]:.2f}")

    scene = problem.make_scene(rep)
    results = {}
    for prec in PRECISIONS:
        _, _, obj = _load(args.files, prec)
        obj(cands[:8].T)                      # warmup (cache + CUDA)
        t0 = time.time()
        losses = obj(cands.T)
        dt = time.time() - t0
        imgs = obj.images_for(rep)
        rep_loss = point_source_loss(imgs, obs, loss_cfg)
        results[prec] = dict(obj=obj, losses=losses, dt=dt, imgs=imgs,
                             loss=rep_loss)
        print(f"\n== gpu_precision = {prec} ==")
        print(f"  population eval : {dt:.3f}s ({1000 * dt / args.pop:.2f} ms/cand)")
        print(f"  representative  : {len(imgs)} image(s), loss = {rep_loss:.6f}")

        ref = results[64]
        dpos, dmag = _match(ref["imgs"], imgs)
        if dpos is not None:
            print(f"  vs fp64 batched : max |dpos| = {dpos * 1e6:.3f} uas, "
                  f"max |dmag|/|mag| = {dmag:.2e}")
        both = (ref["losses"] < INVALID_LOSS) & (losses < INVALID_LOSS)
        agree = np.mean((ref["losses"] >= INVALID_LOSS)
                        == (losses >= INVALID_LOSS))
        if both.any():
            rel = (np.abs(losses[both] - ref["losses"][both])
                   / np.maximum(1.0, np.abs(ref["losses"][both])))
            stats = (f"rel diff median {np.median(rel):.2e} / "
                     f"max {rel.max():.2e}")
        else:
            stats = "no candidate valid under both precisions"
        print(f"  population loss : valid-agreement {agree * 100:.1f}%, "
              f"{stats}")

        # scipy-exact: back-project THIS precision's images through the
        # scipy-exact (sers) + fp64-kernel model; tight clustering on the
        # candidate's true source <=> the images solve the lens equation.
        if imgs:
            fake_obs = ObsData(
                positions=np.array([[im[0], im[1]] for im in imgs]),
                magnifications=np.ones(len(imgs)),
                mag_errors=np.ones(len(imgs)),
                pos_sigma_mas=np.ones(len(imgs)),
                center_offset=(0.0, 0.0))
            ref_rep = reference_check(scene, fake_obs)
            if ref_rep.get("ok"):
                src = ref_rep["backprojected_source"]
                doff = np.hypot(src[0] - scene.source_x,
                                src[1] - scene.source_y) * 1000.0
                print(f"  scipy-exact     : engine-sers vs scipy "
                      f"{ref_rep['gpu_sersic_vs_scipy_arcsec']:.2e} arcsec; "
                      f"image back-projection scatter "
                      f"{ref_rep['source_plane_scatter_mas']:.4f} mas, "
                      f"offset from true source {doff:.4f} mas")
            else:
                print(f"  scipy-exact     : skipped "
                      f"({ref_rep.get('warning', 'unavailable')})")

    # glafic cross-check (TOL_ROMBERG_JHK=1e-5 build). The DECISIVE test is
    # calcimage: evaluate glafic's own deflection AT each precision's images —
    # the residual |theta - alpha(theta) - beta_true| says how well those
    # images solve the lens equation under glafic's physics, independent of
    # glafic findimg's coarse-to-fine search (which can miss genuine near-edge
    # / folded images that its own calcimage confirms).
    print("\n== glafic cross-check (TOL_ROMBERG_JHK=1e-5 build) ==")
    try:
        import glafic  # noqa: PLC0415
    except ImportError:
        print("  glafic python binding not importable; calcimage check skipped")
    else:
        glafic.init(scene.omega, scene.lam, scene.weos, scene.hubble,
                    "/tmp/glade_prec_verify", scene.xmin, scene.ymin,
                    scene.xmax, scene.ymax, scene.pix_ext, scene.pix_poi,
                    scene.maxlev, verb=0)
        try:
            glafic.startup_setnum(len(scene.components), 0, 1)
            for k, comp in enumerate(scene.components, start=1):
                p = (list(comp.params) + [0.0] * 7)[:7]
                glafic.set_lens(k, comp.glafic_type, comp.z, *p)
            glafic.set_point(1, scene.source_z, scene.source_x, scene.source_y)
            glafic.model_init(verb=0)
            for prec in PRECISIONS:
                resid = 0.0
                for (x, y, _m) in results[prec]["imgs"]:
                    gi = glafic.calcimage(scene.source_z, x, y)
                    resid = max(resid, np.hypot(x - gi[0] - scene.source_x,
                                                y - gi[1] - scene.source_y))
                print(f"  prec {prec}: max lens-equation residual under glafic"
                      f" = {resid * 1000:.6f} mas over "
                      f"{len(results[prec]['imgs'])} image(s)")
        finally:
            glafic.quit()

    # informational: glafic's own findimg via the standalone binary
    with tempfile.TemporaryDirectory() as td:
        rep64 = verify_with_glafic(scene, obs, td, loss_cfg,
                                   opt_loss=results[64]["loss"])
        if not rep64.get("ok"):
            print(f"  [findimg] skipped: {rep64.get('warning')}")
        else:
            n_us = len(results[64]["imgs"])
            n_gl = rep64["glafic_n_images"]
            note = ("" if n_gl == n_us else
                    " — count mismatches here usually mean glafic's "
                    "coarse-to-fine findimg missed a genuine image (its own "
                    "calcimage above is the decisive check)")
            print(f"  [findimg] glafic binary finds {n_gl} image(s) vs our "
                  f"{n_us}{note}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
