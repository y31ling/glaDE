"""Clave — interactive gravitational-lens calculator, mounted at ``/clave``.

Integrated into GLADE in V0.6 from the standalone Clave project. Serves a
self-contained single-page UI (``templates/clave.html``) and computes lensed
point-source images with glafic (CPU) or Rhongomyniad (GPU) through the same
``init/set_lens/set_point/point_solve`` API.

The heavy engine imports (glafic C extension, torch) are LAZY: nothing is
imported until the first ``/clave`` API request, so registering this blueprint
does not slow down webui startup.
"""
from __future__ import annotations

import os
import tempfile
import threading

from flask import Blueprint, jsonify, render_template, request

bp = Blueprint("clave", __name__, url_prefix="/clave",
               template_folder="templates")

_lock = threading.Lock()      # glafic (C global state)
_rh_lock = threading.Lock()   # rhongomyniad (Python global state)

# ── glafic (CPU backend), imported on first use ────────────────────────────
_glafic = None
_glafic_failed = False


def _get_glafic():
    global _glafic, _glafic_failed
    if _glafic is None and not _glafic_failed:
        try:
            import glafic as _g
            _glafic = _g
        except ImportError:
            _glafic_failed = True
            print("[Clave] Warning: glafic module not found; CPU computation disabled.")
    return _glafic


# ── Rhongomyniad (GPU backend), imported on first use ──────────────────────
_rh = None            # {"mod", "gpu_avail", "gpu_name", "gpu_device", "models"}
_rh_failed = None     # error string when the import failed
_rh_warmed = False
_rh_warm_lock = threading.Lock()


def _get_rh():
    global _rh, _rh_failed
    if _rh is None and _rh_failed is None:
        try:
            import rhongomyniad as rh_mod
            import torch
            gpu_avail = torch.cuda.is_available()
            _rh = {
                "mod": rh_mod,
                "gpu_avail": gpu_avail,
                "gpu_name": torch.cuda.get_device_name(0) if gpu_avail else "none",
                "gpu_device": str(rh_mod.get_device()),
                "models": set(rh_mod.supported_models()),
            }
            threading.Thread(target=_warmup_rh, daemon=True).start()
        except Exception as exc:
            _rh_failed = str(exc)
            print(f"[Clave] Warning: Rhongomyniad not available ({exc}); GPU disabled.")
    return _rh


def _warmup_rh():
    """Pre-compile GPU kernels with a tiny SIE solve so the first real call is fast."""
    global _rh_warmed
    if _rh is None or _rh_warmed:
        return
    with _rh_warm_lock:
        if _rh_warmed:
            return
        rh = _rh["mod"]
        try:
            with _rh_lock:
                rh.init(0.3, 0.7, -1.0, 0.7, "clave_warmup",
                        -2.0, -2.0, 2.0, 2.0, 0.01, 0.1, 5, 0, 0)
                rh.startup_setnum(1, 0, 1)
                rh.set_lens(1, "sie", 0.5, 200, 0, 0, 0.3, 0, 0, 0)
                rh.set_point(1, 2.0, 0.05, 0.05)
                rh.model_init(0)
                rh.point_solve(2.0, 0.05, 0.05, verb=0)
                rh.quit()
            _rh_warmed = True
            print("[Clave] Rhongomyniad GPU warmup done.")
        except Exception as e:
            print(f"[Clave] Rhongomyniad warmup failed: {e}")


def _safe_float(v, default=0.0):
    try:
        f = float(v)
        return f if (f == f) else default
    except (TypeError, ValueError):
        return default


def compute_images(lenses, sources):
    glafic = _get_glafic()
    if glafic is None:
        return _mock_images(sources)
    if not lenses or not sources:
        return []

    # Adaptive FOV + pix_poi (shared with GPU path)
    margin, pix_poi = _fov_params(lenses, sources)
    xmin, xmax, ymin, ymax = -margin, margin, -margin, margin

    with _lock:
        initialized = False
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                prefix = os.path.join(tmpdir, "clave")
                glafic.init(
                    0.3, 0.7, -1.0, 0.7,
                    prefix,
                    float(xmin), float(ymin), float(xmax), float(ymax),
                    0.01, pix_poi, 5,
                    0, 0
                )
                initialized = True

                glafic.startup_setnum(len(lenses), 0, len(sources))

                for i, lens in enumerate(lenses):
                    glafic.set_lens(
                        i + 1,
                        str(lens.get("type", "sie")),
                        _safe_float(lens.get("z"),  0.5),
                        _safe_float(lens.get("p1"), 0.0),
                        _safe_float(lens.get("x"),  0.0),
                        _safe_float(lens.get("y"),  0.0),
                        _safe_float(lens.get("e"),  0.0),
                        _safe_float(lens.get("pa"), 0.0),
                        _safe_float(lens.get("r1"), 0.0),
                        _safe_float(lens.get("r2"), 0.0),
                    )

                for i, src in enumerate(sources):
                    glafic.set_point(
                        i + 1,
                        _safe_float(src.get("z"),  2.0),
                        _safe_float(src.get("x"),  0.0),
                        _safe_float(src.get("y"),  0.0),
                    )

                glafic.model_init(0)

                results = []
                for i, src in enumerate(sources):
                    zs = _safe_float(src.get("z"), 2.0)
                    xs = _safe_float(src.get("x"), 0.0)
                    ys = _safe_float(src.get("y"), 0.0)
                    try:
                        imgs = glafic.point_solve(zs, xs, ys, verb=0)
                        images = [
                            {"x": float(im[0]), "y": float(im[1]),
                             "mu": float(im[2]), "td": float(im[3])}
                            for im in imgs
                        ]
                    except Exception:
                        images = []
                    results.append({"source_idx": i, "images": images})

                glafic.quit()
                return results

        except Exception:
            if initialized:
                try:
                    glafic.quit()
                except Exception:
                    pass
            raise


def _fov_params(lenses, sources):
    """Compute adaptive FOV (margin) and pix_poi from the lens/source layout."""
    all_x = [_safe_float(l.get("x")) for l in lenses] + \
            [_safe_float(s.get("x")) for s in sources] + [0.0]
    all_y = [_safe_float(l.get("y")) for l in lenses] + \
            [_safe_float(s.get("y")) for s in sources] + [0.0]

    for l in lenses:
        x  = _safe_float(l.get("x"))
        y  = _safe_float(l.get("y"))
        r1 = _safe_float(l.get("r1"))
        r2 = _safe_float(l.get("r2"))
        p1 = _safe_float(l.get("p1"))
        lt = str(l.get("type", ""))
        if lt in ("sie", "nsie", "softie"):
            re_est = max(p1 / 200.0 * 1.5, 0.5)
            all_x += [x + re_est, x - re_est]
            all_y += [y + re_est, y - re_est]
        elif lt == "nfw" and r1 > 0:
            re_est = max(p1 ** 0.5 * 0.5, 0.5)
            all_x += [x + re_est, x - re_est]
            all_y += [y + re_est, y - re_est]
        elif lt == "king" and r1 > 0 and r2 > 0:
            rt = r1 * (10 ** min(float(r2), 3.0))
            all_x += [x + rt, x - rt]
            all_y += [y + rt, y - rt]
        elif lt in ("pow", "jaffe", "sers", "hernquist", "gnfw", "tnfw"):
            re_est = max(r1 * 3, 0.5)
            all_x += [x + re_est, x - re_est]
            all_y += [y + re_est, y - re_est]
        elif r1 > 0:
            all_x += [x + r1 * 4, x - r1 * 4]
            all_y += [y + r1 * 4, y - r1 * 4]

    margin  = max(0.4, max(abs(v) for v in all_x + all_y) * 2.0)
    pix_poi = max(0.02, min(0.15, margin / 6.0))
    return margin, pix_poi


def compute_images_gpu(lenses, sources):
    """GPU path: uses Rhongomyniad (same API as glafic).
    Only supports single-lens-plane (all lenses same redshift).
    Falls back gracefully for unsupported model types.
    """
    st = _get_rh()
    if st is None:
        raise RuntimeError(f"Rhongomyniad not available ({_rh_failed})")
    rh = st["mod"]
    if not lenses or not sources:
        return []

    # Check all lens models are supported by Rhongomyniad
    for l in lenses:
        lt = str(l.get("type", "sie"))
        if lt not in st["models"]:
            raise NotImplementedError(
                f"Lens type '{lt}' not supported in GPU mode (Rhongomyniad). "
                f"Supported: {sorted(st['models'])}")

    # Rhongomyniad v1: single lens plane only — all lenses must share one zl
    zl_vals = [_safe_float(l.get("z"), 0.5) for l in lenses]
    if max(zl_vals) - min(zl_vals) > 1e-3:
        raise NotImplementedError(
            "GPU mode requires all lenses on the same redshift plane. "
            f"Found z in [{min(zl_vals):.3f}, {max(zl_vals):.3f}].")

    margin, pix_poi_cpu = _fov_params(lenses, sources)
    # GPU can afford slightly coarser initial grid (parallel computation);
    # the adaptive quad-tree still refines near critical curves.
    pix_poi = max(0.05, min(0.2, margin / 5.0))
    xmin, xmax, ymin, ymax = -margin, margin, -margin, margin

    with _rh_lock:
        try:
            rh.init(0.3, 0.7, -1.0, 0.7, "clave_gpu",
                    float(xmin), float(ymin), float(xmax), float(ymax),
                    0.01, pix_poi, 5, 0, 0)
            rh.startup_setnum(len(lenses), 0, len(sources))

            for i, lens in enumerate(lenses):
                rh.set_lens(
                    i + 1,
                    str(lens.get("type", "sie")),
                    _safe_float(lens.get("z"),  0.5),
                    _safe_float(lens.get("p1"), 0.0),
                    _safe_float(lens.get("x"),  0.0),
                    _safe_float(lens.get("y"),  0.0),
                    _safe_float(lens.get("e"),  0.0),
                    _safe_float(lens.get("pa"), 0.0),
                    _safe_float(lens.get("r1"), 0.0),
                    _safe_float(lens.get("r2"), 0.0),
                )

            for i, src in enumerate(sources):
                rh.set_point(
                    i + 1,
                    _safe_float(src.get("z"),  2.0),
                    _safe_float(src.get("x"),  0.0),
                    _safe_float(src.get("y"),  0.0),
                )

            rh.model_init(0)

            results = []
            for i, src in enumerate(sources):
                zs = _safe_float(src.get("z"), 2.0)
                xs = _safe_float(src.get("x"), 0.0)
                ys = _safe_float(src.get("y"), 0.0)
                try:
                    imgs = rh.point_solve(zs, xs, ys, verb=0)
                    images = [
                        {"x": float(im[0]), "y": float(im[1]),
                         "mu": float(im[2]), "td": float(im[3])}
                        for im in imgs
                    ]
                except Exception:
                    images = []
                results.append({"source_idx": i, "images": images})

            rh.quit()
            return results

        except Exception:
            try:
                rh.quit()
            except Exception:
                pass
            raise


def _mock_images(sources):
    import math
    results = []
    for i, src in enumerate(sources):
        xs, ys = _safe_float(src.get("x"), 0.05), _safe_float(src.get("y"), 0.05)
        th = math.atan2(ys, xs)
        rE = 0.5
        images = [
            {"x": rE * math.cos(th + a), "y": rE * math.sin(th + a),
             "mu": 3.0 * ((-1) ** k), "td": float(k)}
            for k, a in enumerate([0, math.pi/2, math.pi, 3*math.pi/2])
        ]
        results.append({"source_idx": i, "images": images})
    return results


@bp.route("/")
def index():
    return render_template("clave.html")


@bp.route("/api/findimg", methods=["POST"])
def api_findimg():
    data    = request.get_json(force=True) or {}
    lenses  = data.get("lenses",  [])
    sources = data.get("sources", [])
    backend = data.get("backend", "cpu")   # "cpu" or "gpu"
    if not lenses or not sources:
        return jsonify({"ok": True, "results": [], "backend": backend})
    try:
        if backend == "gpu":
            if _get_rh() is None:
                return jsonify({"ok": False,
                                "error": "GPU backend (Rhongomyniad) not available.",
                                "backend": "gpu"})
            results = compute_images_gpu(lenses, sources)
        else:
            results = compute_images(lenses, sources)
        return jsonify({"ok": True, "results": results, "backend": backend})
    except NotImplementedError as exc:
        return jsonify({"ok": False, "error": str(exc), "backend": backend})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "backend": backend})


@bp.route("/api/gpu_status")
def api_gpu_status():
    st = _get_rh()
    return jsonify({
        "ok":          True,
        "rhong_ok":    st is not None,
        "gpu_avail":   bool(st and st["gpu_avail"]),
        "gpu_name":    st["gpu_name"] if st else "none",
        "gpu_device":  st["gpu_device"] if st else "cpu",
        "models":      sorted(st["models"]) if st else [],
    })


@bp.route("/api/export", methods=["POST"])
def api_export():
    data    = request.get_json(force=True) or {}
    lenses  = data.get("lenses",  [])
    sources = data.get("sources", [])
    results = data.get("results", [])

    lines = ["## Clave Gravitational Lens Calculator Export\n", "\n"]
    for lens in lenses:
        t  = lens.get("type", "sie")
        z  = _safe_float(lens.get("z"),  0.5)
        p1 = _safe_float(lens.get("p1"), 0.0)
        x  = _safe_float(lens.get("x"),  0.0)
        y  = _safe_float(lens.get("y"),  0.0)
        e  = _safe_float(lens.get("e"),  0.0)
        pa = _safe_float(lens.get("pa"), 0.0)
        r1 = _safe_float(lens.get("r1"), 0.0)
        r2 = _safe_float(lens.get("r2"), 0.0)
        lines.append(
            f"lens {t}  {z:.4f}  {p1:.6e}  {x:.6e}  {y:.6e}"
            f"  {e:.4f}  {pa:.4f}  {r1:.6e}  {r2:.6e}\n"
        )
    for src in sources:
        z = _safe_float(src.get("z"), 2.0)
        x = _safe_float(src.get("x"), 0.0)
        y = _safe_float(src.get("y"), 0.0)
        lines.append(f"point  {z:.4f}  {x:.6e}  {y:.6e}\n")

    if results:
        lines.append("\n## Lensed image positions:\n")
        lines.append("## source  x(arcsec)  y(arcsec)  mu  time_delay\n")
        for res in results:
            si = res.get("source_idx", 0)
            for img in res.get("images", []):
                lines.append(
                    f"## img {si+1}  {img['x']:.6e}  {img['y']:.6e}"
                    f"  {img['mu']:.4f}  {img['td']:.4f}\n"
                )

    return jsonify({"ok": True, "content": "".join(lines)})


@bp.route("/api/status")
def api_status():
    return jsonify({"ok": True, "glafic": _get_glafic() is not None})
