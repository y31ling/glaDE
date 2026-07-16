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
        elif lt in ("nfw", "anfw", "gnfw", "tnfw", "ein"):
            # r1 is the CONCENTRATION (dimensionless), not a radius: estimate
            # the extent from the mass via a point-equivalent Einstein radius
            # (theta_E ~ sqrt(M / 2e11 Msun) arcsec at galaxy-scale redshifts).
            re_est = max((max(p1, 0.0) / 2e11) ** 0.5, 0.5)
            all_x += [x + re_est, x - re_est]
            all_y += [y + re_est, y - re_est]
        elif lt == "king" and r1 > 0 and r2 > 0:
            rt = r1 * (10 ** min(float(r2), 3.0))
            all_x += [x + rt, x - rt]
            all_y += [y + rt, y - rt]
        elif lt in ("pow", "jaffe", "sers", "hernquist", "hern", "ahern"):
            re_est = max(r1 * 3, 0.5)
            all_x += [x + re_est, x - re_est]
            all_y += [y + re_est, y - re_est]
        elif r1 > 0:
            all_x += [x + r1 * 4, x - r1 * 4]
            all_y += [y + r1 * 4, y - r1 * 4]

    # Hard cap: glafic's init() enforces nx_ext=(xmax-xmin)/pix_ext <= 20000
    # and calls terminator() (process exit!) beyond it. With pix_ext=0.01 that
    # means margin <= 100; cap at 60 so a huge-mass lens (e.g. the 1e13 Msun
    # NFW default, whose crude re_est blows up) can never kill the server.
    margin  = max(0.4, min(max(abs(v) for v in all_x + all_y) * 2.0, 60.0))
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


def _gnum(v) -> str:
    """Compact numeric literal for a glade ``.dat`` (1e+13, 0.3, 21.488...)."""
    return f"{float(v):.10g}"


def _render_glade_dat(lenses, sources, results):
    """Render the Clave scene as a legal glade ``.dat`` document.

    Returns ``(content, warnings)``. All parameters are locked to the scene
    values; observations (when Clave has computed images for the first source)
    are exported with placeholder error columns the user must edit.
    """
    from datetime import datetime

    from core.format.schema import MODELS

    warnings: list[str] = []
    if not lenses:
        raise ValueError("nothing to export: the scene has no lenses")

    margin, pix_poi = _fov_params(lenses, sources)
    src = sources[0] if sources else None
    src_z = _safe_float(src.get("z"), 2.0) if src else 2.0
    src_x = _safe_float(src.get("x"), 0.0) if src else 0.0
    src_y = _safe_float(src.get("y"), 0.0) if src else 0.0
    if not sources:
        warnings.append("scene has no source; exported source_z=2, source at (0,0)")

    L = [
        "# ============================================================",
        f"#  GLADE .dat exported from Clave  ({datetime.now():%Y-%m-%d %H:%M})",
        "#  All parameters are locked to the Clave scene values.",
        "#  To optimize a parameter, replace its value with {lo, hi}.",
        "#  pa convention (glafic): measured East of North, i.e. counter-",
        "#  clockwise from the +y axis; pa=0 puts the major axis along +y.",
        "#  位置角约定: 从 +y 轴(北)逆时针起算; pa=0 时长轴沿 +y。",
        "# ============================================================",
        "",
        "# --- grid (cosmology defaults: omega=0.3 lambda=0.7 hubble=0.7) ---",
        f"xmin, ymin = {_gnum(-margin)}, {_gnum(-margin)}",
        f"xmax, ymax = {_gnum(margin)}, {_gnum(margin)}",
        "pix_ext = 0.01",
        f"pix_poi = {_gnum(pix_poi)}",
        "maxlev = 5",
        "",
        "# --- source ---",
        f"source_z = {_gnum(src_z)}",
        f"source_x = {_gnum(src_x)}    # e.g. {{{_gnum(src_x - 0.05)}, {_gnum(src_x + 0.05)}}} to optimize",
        f"source_y = {_gnum(src_y)}",
    ]
    for k, extra in enumerate(sources[1:], start=2):
        warnings.append(f"GLADE point mode fits a single source; source {k} "
                        "was written as a comment")
        L.append(f"# source {k} (not exported): z={_gnum(_safe_float(extra.get('z'), 2.0))}, "
                 f"x={_gnum(_safe_float(extra.get('x')))}, y={_gnum(_safe_float(extra.get('y')))}")
    L.append("")

    L.append("# --- lenses (params in glafic set_lens order; values locked) ---")
    idx = 0
    clave_order = ("p1", "x", "y", "e", "pa", "r1", "r2")
    for lens in lenses:
        ltype = str(lens.get("type", "sie"))
        z = _safe_float(lens.get("z"), 0.5)
        vals = [_safe_float(lens.get(k), 0.0) for k in clave_order]
        spec = MODELS.get(ltype)
        if spec is None:
            warnings.append(f"lens type '{ltype}' is unknown to GLADE and was "
                            "written as a comment")
            L.append(f"# UNSUPPORTED: '{ltype}': ({', '.join(_gnum(v) for v in [z] + vals)})")
            continue
        idx += 1
        params = vals[:len(spec.params)]
        tup = ", ".join([str(idx), f"'{spec.key}'", _gnum(z)] + [_gnum(v) for v in params])
        L.append(f"'{spec.key}{idx}': ({tup})")
        descs = "; ".join(f"{p.name}: {p.desc}" for p in spec.params if p.desc)
        if descs:
            L.append(f"#   {descs}")
    if idx == 0:
        raise ValueError("nothing to export: no lens type is supported by GLADE")
    L.append("")

    imgs = []
    for res in results or []:
        if int(res.get("source_idx", 0)) == 0:
            imgs = list(res.get("images", []))
            break
    if not imgs:
        # a legal point-mode .dat needs the obs arrays -- solve the exported
        # source (the first scene source, or the default written above) now
        try:
            computed = compute_images(lenses, [{"z": src_z, "x": src_x, "y": src_y}])
            if computed:
                imgs = list(computed[0].get("images", []))
        except Exception as exc:
            warnings.append(f"could not compute images for the obs block: {exc}")
    if not imgs:
        warnings.append("no images -> no obs block; GLADE will refuse to load "
                        "this file until an obs block is added")
    if imgs:
        pos = ", ".join(f"[{_gnum(_safe_float(im.get('x')) * 1000.0)},"
                        f"{_gnum(_safe_float(im.get('y')) * 1000.0)}]" for im in imgs)
        mus = [_safe_float(im.get("mu")) for im in imgs]
        L += [
            "# --- observations (from the Clave computed images) ---",
            "# NOTE: the error columns are PLACEHOLDERS -- edit them to your",
            "#       measurement uncertainties before fitting.",
            f"obs_positions_mas_list = [{pos}]",
            f"obs_magnifications_list = [{', '.join(_gnum(m) for m in mus)}]",
            f"obs_mag_errors_list = [{', '.join(_gnum(max(abs(m) * 0.1, 0.05)) for m in mus)}]",
            f"obs_pos_sigma_mas_list = [{', '.join('1' for _ in mus)}]",
            "center_offset_x = 0",
            "center_offset_y = 0",
            "obs_x_flip = False      # Clave scene is already in the glafic math frame",
            "",
        ]
    else:
        L += ["# (no computed images -- add an obs_positions_mas_list block to fit)", ""]
    return "\n".join(L) + "\n", warnings


def _validate_glade_dat(content: str) -> list[str]:
    """Run GLADE's full ``load_config`` pipeline on ``content``; return errors."""
    try:
        from core.format.api import load_config
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(content)
            path = fh.name
        try:
            _cfg, issues = load_config([path])
        finally:
            os.unlink(path)
        return [i.message for i in issues if i.is_error]
    except Exception as exc:                       # pragma: no cover - safety net
        return [f"validation crashed: {exc}"]


@bp.route("/api/export", methods=["POST"])
def api_export():
    """Export the Clave scene as a legal glade ``.dat`` document."""
    data    = request.get_json(force=True) or {}
    lenses  = data.get("lenses",  [])
    sources = data.get("sources", [])
    results = data.get("results", [])
    try:
        content, warnings = _render_glade_dat(lenses, sources, results)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)})
    return jsonify({"ok": True, "content": content,
                    "warnings": warnings, "errors": _validate_glade_dat(content)})


@bp.route("/api/status")
def api_status():
    return jsonify({"ok": True, "glafic": _get_glafic() is not None})
