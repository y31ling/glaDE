#!/usr/bin/env python3
"""GLADE V0.4 WebUI (rewrite).

Two pages served by one Flask app:
  * FindImage (default): pick CPU/GPU/Glafic + .dat files -> spawn a real terminal
    that runs core.optimize, streamed back to the browser over SSE.
  * Editor: VSCode-like Explorer + Template + Monaco editor over InputFiles/.

Run:  ./run_webui.sh   (or  python webui/app.py),  then open http://localhost:6017
"""
from __future__ import annotations

import json
import os
import re
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for p in (_ROOT, os.path.join(_ROOT, "glafic2", "python"),
          os.path.join(_ROOT, "Rhongomyniad")):
    if p not in sys.path:
        sys.path.insert(0, p)

from flask import (Flask, Response, jsonify, request, send_file,  # noqa: E402
                   send_from_directory)

from webui.files import FileStore  # noqa: E402
from webui.jobs import JobManager  # noqa: E402
from webui.templates_lib import template_tree  # noqa: E402

from clave import bp as clave_bp  # noqa: E402

app = Flask(__name__, static_folder=os.path.join(_THIS, "static"), static_url_path="/static")
app.register_blueprint(clave_bp)  # Clave lens calculator at /clave (V0.6)

INPUT_DIR = os.path.join(_ROOT, "InputFiles")
store = FileStore(INPUT_DIR)
jobs = JobManager(_ROOT)


# --------------------------------------------------------------------------- #
# pages / static
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/assets/icon/<path:name>")
def icon(name):
    icon_dir = os.path.join(_ROOT, "source", "icon")
    return send_from_directory(icon_dir, name)


# --------------------------------------------------------------------------- #
# files API
# --------------------------------------------------------------------------- #
@app.route("/api/files/tree")
def files_tree():
    return jsonify(store.tree())


@app.route("/api/files/read")
def files_read():
    try:
        return jsonify({"path": request.args["path"],
                        "content": store.read(request.args["path"])})
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/files/write", methods=["POST"])
def files_write():
    data = request.get_json(force=True)
    try:
        store.write(data["path"], data.get("content", ""))
        return jsonify({"ok": True})
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/files/create", methods=["POST"])
def files_create():
    data = request.get_json(force=True)
    try:
        if data.get("type") == "folder":
            rel = store.create_folder(data["path"])
        else:
            rel = store.create_file(data["path"], data.get("content", ""))
        return jsonify({"ok": True, "path": rel})
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/files/rename", methods=["POST"])
def files_rename():
    data = request.get_json(force=True)
    try:
        return jsonify({"ok": True, "path": store.rename(data["path"], data["name"])})
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/files/copy", methods=["POST"])
def files_copy():
    data = request.get_json(force=True)
    try:
        return jsonify({"ok": True, "path": store.copy(data["path"], data["dest"])})
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/files/delete", methods=["POST"])
def files_delete():
    data = request.get_json(force=True)
    try:
        store.delete(data["path"])
        return jsonify({"ok": True})
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


# --------------------------------------------------------------------------- #
# translate (import glafic -> glade ; export glade -> glafic)
# --------------------------------------------------------------------------- #
@app.route("/api/files/import", methods=["POST"])
def files_import():
    """Translate a glafic input file into glade .dat file(s) in InputFiles/."""
    from core.translate import glafic_to_glade
    data = request.get_json(force=True)
    try:
        text = store.read(data["path"])
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    out = glafic_to_glade(text)
    base = os.path.splitext(os.path.basename(data["path"]))[0]
    folder = os.path.dirname(data["path"])
    written = []
    for kind in ("model", "obs"):
        if out.get(kind):
            rel = os.path.join(folder, f"{base}_{kind}.dat") if folder else f"{base}_{kind}.dat"
            store.write(rel, out[kind])
            written.append(rel.replace(os.sep, "/"))
    return jsonify({"ok": True, "written": written})


@app.route("/api/files/import_clave", methods=["POST"])
def files_import_clave():
    """Convert a glade ``.dat`` or a native glafic ``.input`` into a Clave
    scene (``{lenses, sources}`` in Clave's JSON layout).

    ``{lo, hi}`` parameters collapse to the same representative value the
    glade -> glafic export uses (geometric mean for mass-like parameters,
    arithmetic mean otherwise — ``core.translate.convert._midpoint``).
    """
    from core.format.config import apply_defaults, merge
    from core.format.parser import parse_text
    from core.translate import glafic_to_glade, looks_like_glafic_input
    from core.translate.convert import _glade_to_model
    data = request.get_json(force=True)
    try:
        text = store.read(data["path"])
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        if looks_like_glafic_input(text):
            out = glafic_to_glade(text)
            texts = [v for v in (out.get("model"), out.get("obs")) if v]
        else:
            texts = [text]
        cfg, issues = merge([parse_text(t, path=data["path"]) for t in texts])
        errors = [i.message for i in issues if i.is_error]
        if errors:
            return jsonify({"error": "; ".join(errors[:3])}), 400
        apply_defaults(cfg)
        model = _glade_to_model(cfg)
    except Exception as exc:  # parse/translate failures -> clean 400
        return jsonify({"error": str(exc)}), 400
    if not model.lenses:
        return jsonify({"error": "no lens components found in this file"}), 400
    lenses = [{"type": ln.type, "z": ln.z,
               "p1": ln.params[0], "x": ln.params[1], "y": ln.params[2],
               "e": ln.params[3], "pa": ln.params[4],
               "r1": ln.params[5], "r2": ln.params[6]}
              for ln in model.lenses]
    sources = [{"z": model.source_z if model.source_z is not None else 2.0,
                "x": model.source_x or 0.0, "y": model.source_y or 0.0}]
    return jsonify({"ok": True, "lenses": lenses, "sources": sources})


@app.route("/api/files/export", methods=["POST"])
def files_export():
    """Translate selected glade .dat file(s) into a runnable glafic input bundle.

    When the selection has any ``{lo, hi}`` parameters the exported ``.input``
    gains a ``start_setopt`` matrix + an ``optimize`` command (glafic's amoeba),
    and the matching ``readobs_point`` constraint (``_obs.dat``) and ``parprior``
    ranges (``_prior.dat``) are written too; otherwise just the model + a
    round-trip ``_obs.dat`` are written (findimg-only).
    """
    from core.format.config import apply_defaults, merge
    from core.format.parser import parse_file
    from core.translate import glade_to_glafic
    data = request.get_json(force=True)
    paths = [os.path.join(INPUT_DIR, p) for p in data["files"]]
    cfg, _ = merge([parse_file(p) for p in paths])
    apply_defaults(cfg)
    base = data.get("name", "glade_export")
    out = glade_to_glafic(cfg, base_name=base)
    written = []
    if out.get("model"):
        store.write(f"{base}_model.input", out["model"]); written.append(f"{base}_model.input")
    if out.get("optimize"):
        # optimize-ready: the model references these by name (readobs_point / parprior)
        if out.get("constraint"):
            store.write(f"{base}_obs.dat", out["constraint"]); written.append(f"{base}_obs.dat")
        if out.get("prior"):
            store.write(f"{base}_prior.dat", out["prior"]); written.append(f"{base}_prior.dat")
    elif out.get("obs"):
        store.write(f"{base}_obs.dat", out["obs"]); written.append(f"{base}_obs.dat")
    return jsonify({"ok": True, "written": written, "optimize": bool(out.get("optimize"))})


# --------------------------------------------------------------------------- #
# templates
# --------------------------------------------------------------------------- #
@app.route("/api/templates")
def templates():
    """Editor Template tree. ``?units=<profileName>`` renders unit-aware comments
    (and a UnitSetting line) for that profile; omitted / 'default' = engine units.
    """
    from core.format.units import resolve_profile
    prof = request.args.get("units")
    units = None
    if prof and prof.strip() not in ("", "default"):
        units, _issues = resolve_profile(prof, [INPUT_DIR])
        if units is None:                         # unknown/unreadable profile
            return jsonify(template_tree())       # fall back to engine defaults
    return jsonify(template_tree(units if units is not None else None))


# --------------------------------------------------------------------------- #
# unit profiles (the UnitSetting key)
# --------------------------------------------------------------------------- #
@app.route("/api/units")
def units_get():
    """Unit categories + options, the fixed-unit rows, and the saved profiles
    found in InputFiles/ (``*.units.json``)."""
    from core.format import units as U
    categories = {name: {"default": default, "options": list(opts)}
                  for name, (default, opts) in U.CATEGORIES.items()}
    profiles = []
    try:
        entries = sorted(os.listdir(INPUT_DIR))
    except OSError:
        entries = []
    for entry in entries:
        if not entry.endswith(U.PROFILE_SUFFIX):
            continue
        path = os.path.join(INPUT_DIR, entry)
        if not os.path.isfile(path):
            continue
        name = entry[:-len(U.PROFILE_SUFFIX)]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            raw = data.get("units", data) if isinstance(data, dict) else {}
            merged = dict(U.DEFAULT_UNITS)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if k in U.CATEGORIES:
                        merged[k] = v
            profiles.append({"name": name, "units": merged})
        except (OSError, ValueError) as exc:
            profiles.append({"name": name, "error": str(exc)})
    return jsonify({"categories": categories, "fixed": U.FIXED_UNITS,
                    "profiles": profiles})


@app.route("/api/units/save", methods=["POST"])
def units_save():
    """Validate + write a unit profile to InputFiles/<name>.units.json."""
    from core.format import units as U
    data = request.get_json(force=True)
    name = re.sub(r"[^A-Za-z0-9_-]", "", str(data.get("name", "")).strip())
    if not name:
        return jsonify({"error": "invalid profile name "
                        "(letters, digits, '_' and '-' only)"}), 400
    given = data.get("units", {})
    if not isinstance(given, dict):
        return jsonify({"error": "'units' must be an object"}), 400
    clean = {}
    for k, v in given.items():
        if k not in U.CATEGORIES:
            return jsonify({"error": f"unknown category {k!r} (expected one of "
                            f"{', '.join(U.CATEGORIES)})"}), 400
        if v not in U.CATEGORIES[k][1]:
            return jsonify({"error": f"{k} = {v!r}: expected one of "
                            f"{', '.join(U.CATEGORIES[k][1])}"}), 400
        clean[k] = v
    rel = f"{name}{U.PROFILE_SUFFIX}"
    body = json.dumps({"format": U.PROFILE_FORMAT, "units": clean}, indent=2)
    try:
        store.write(rel, body + "\n")
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, "path": rel, "name": name, "units": clean})


# --------------------------------------------------------------------------- #
# run (validate -> confirm defaults -> spawn terminal -> SSE)
# --------------------------------------------------------------------------- #
# MCMC-only rails and the engine each one drives. 'mcmc-gpu' samples with the
# batched CUDA likelihood (emcee vectorize=True) when the config allows it —
# see GPU_MCMC_exploration.md for when that pays off.
_MCMC_RAILS = {"mcmc": "cpu", "mcmc-gpu": "gpu"}


def _rail_engine(rail_backend: str) -> str:
    """The engine a FindImage rail choice validates/runs against."""
    return _MCMC_RAILS.get(rail_backend, rail_backend)


def _resolve_engine_mode(rail_backend: str, cfg) -> tuple[str, str]:
    """Map a FindImage rail choice to (engine, mode).

    'glafic' rail -> glafic's own amoeba (`optimize`), NOT DE. Glade selections
    are converted to temp glafic files first (see webui.runjob._run_amoeba).
    'mcmc' rail -> MCMC-only on the CPU/glafic engine.
    'mcmc-gpu' rail -> MCMC-only on the GPU engine (batched when possible).
    'cpu'/'gpu' -> DE; also MCMC afterwards iff MCMC_ENABLED is set.
    """
    if rail_backend == "glafic":
        return "glafic", "amoeba"
    if rail_backend in _MCMC_RAILS:
        return _MCMC_RAILS[rail_backend], "mcmc"
    mode = "de+mcmc" if bool(cfg.algorithm.get("MCMC_ENABLED", False)) else "findimage"
    return rail_backend, mode


def _glafic_selection_kind(rel_files) -> str:
    """Classify a Glafic-rail selection: 'native', 'glade', or 'mixed'.

    'native'  -> every file is a glafic .input (run directly via amoeba),
    'glade'   -> none are (convert then amoeba),
    'mixed'   -> a blend (rejected with a clear message, since the two paths
                 differ and load_config would otherwise emit a raw parse error).
    """
    from core.translate import looks_like_glafic_input
    if not rel_files:
        return "glade"
    flags = []
    for p in rel_files:
        try:
            with open(os.path.join(INPUT_DIR, p), encoding="utf-8",
                      errors="replace") as fh:
                flags.append(looks_like_glafic_input(fh.read()))
        except OSError:
            flags.append(False)
    if all(flags):
        return "native"
    if any(flags):
        return "mixed"
    return "glade"


_MIXED_MSG = ("mixed selection: choose EITHER glade .dat file(s) OR a glafic "
              ".input on the Glafic rail — not both.")


def _display_defaults(cfg, engine: str, mode: str) -> dict:
    """The applied-defaults dict for the confirm dialog. Shows the value a GPU
    MCMC run will actually use when the worker auto-raises an unset
    MCMC_NWALKERS (see webui.runjob._tune_mcmc_for_gpu)."""
    out = {name: cfg.all_scalars().get(name) for name in cfg.applied_defaults}
    if engine == "gpu" and mode in ("mcmc", "de+mcmc") and "MCMC_NWALKERS" in out:
        from core.format.validate import is_extend_mode
        from webui.runjob import gpu_mcmc_auto_walkers
        auto = gpu_mcmc_auto_walkers(cfg, extend=is_extend_mode(cfg))
        if auto:
            out["MCMC_NWALKERS"] = f"{auto} (auto-raised for the batched GPU sampler)"
    return out


# V0.7 pipeline contract: purpose + optimizer selectors -> the runjob command
# line. The optimizer strings map to core.optimize.runner's canonical names.
_OPT_CLI = {"de": "DE", "cmaes": "BIPOP-CMA-ES", "jso": "jSO"}


def _plan_from_payload(data) -> tuple[dict, object]:
    """Normalize an /api/run(/check) payload (old or new shape) into a plan.

    New V0.7 shape: ``{backend:'cpu'|'gpu', purpose:'calcimage'|'optimize'|
    'mcmc', optimizer:'amoeba'|'de'|'cmaes'|'jso'}``. Old shape: ``{backend}``
    with backend in cpu/gpu/glafic/mcmc/mcmc-gpu (unchanged).

    Returns ``(plan, err)``. ``plan`` keys: ``engine`` ('cpu'|'gpu'|'glafic',
    what load_config validates against + jobs.start's backend), ``glafic_rail``
    (run the native/mixed amoeba selection logic), ``optimizer`` (canonical name
    or None), ``purpose``, and ``rail`` (legacy only). ``err`` is a
    ``(body, status)`` tuple to return immediately, else None."""
    if "purpose" in data:
        backend = str(data.get("backend", "cpu")).lower()
        purpose = str(data["purpose"]).lower()
        if backend not in ("cpu", "gpu"):
            return {}, ({"ok": False, "errors": [
                f"backend must be 'cpu' or 'gpu' (got {backend!r})"]}, 400)
        if purpose in ("calcimage", "mcmc"):
            return {"engine": backend, "glafic_rail": False, "optimizer": None,
                    "purpose": purpose}, None
        if purpose == "optimize":
            optimizer = str(data.get("optimizer", "de")).lower()
            if optimizer == "amoeba":
                if backend == "gpu":
                    return {}, ({"ok": False, "errors": [
                        "amoeba is glafic-native and CPU-only; use the CPU "
                        "backend, or pick DE / BIPOP-CMA-ES / jSO for the GPU."]},
                        400)
                return {"engine": "glafic", "glafic_rail": True,
                        "optimizer": None, "purpose": "amoeba"}, None
            if optimizer not in _OPT_CLI:
                return {}, ({"ok": False, "errors": [
                    f"unknown optimizer {optimizer!r}; expected amoeba, de, "
                    f"cmaes or jso"]}, 400)
            return {"engine": backend, "glafic_rail": False,
                    "optimizer": _OPT_CLI[optimizer], "purpose": "optimize"}, None
        return {}, ({"ok": False, "errors": [
            f"unknown purpose {purpose!r}; expected calcimage, optimize or "
            f"mcmc"]}, 400)

    # ---- legacy contract: a single backend rail ----
    rail = str(data["backend"])
    return {"engine": _rail_engine(rail), "glafic_rail": rail == "glafic",
            "optimizer": None, "purpose": "legacy", "rail": rail}, None


def _resolve_mode(plan: dict, cfg) -> str:
    """The runjob ``--mode`` for a resolved plan + parsed cfg."""
    purpose = plan["purpose"]
    if purpose in ("calcimage", "mcmc", "amoeba"):
        return purpose
    if purpose == "optimize":
        return ("de+mcmc" if bool(cfg.algorithm.get("MCMC_ENABLED", False))
                else "findimage")
    return _resolve_engine_mode(plan["rail"], cfg)[1]   # legacy


@app.route("/api/run/check", methods=["POST"])
def run_check():
    """Validate a selection without launching. Returns blocking errors and the
    basic variables that would fall back to defaults (for the confirm dialog)."""
    from core.format import load_config
    from core.format.validate import is_extend_mode
    from core.optimize.problem import OptProblem
    data = request.get_json(force=True)
    plan, err = _plan_from_payload(data)
    if err is not None:
        body, _status = err                       # preview endpoint: surface at 200
        return jsonify({"ok": False, "mode": "?", "engine": "?",
                        "errors": body.get("errors", []), "warnings": [],
                        "defaulted": {}, "ndim": 0})
    engine = plan["engine"]
    # the glafic / amoeba rail runs glafic's amoeba; native .input selections skip
    # glade validation entirely, and a mixed selection is rejected clearly.
    if plan["glafic_rail"]:
        kind = _glafic_selection_kind(data["files"])
        if kind == "mixed":
            return jsonify({"ok": False, "mode": "amoeba", "engine": "glafic",
                            "errors": [_MIXED_MSG], "warnings": [],
                            "defaulted": {}, "ndim": 0})
        if kind == "native":
            return jsonify({
                "ok": True, "mode": "amoeba", "engine": "glafic", "errors": [],
                "warnings": ["native glafic input — runs directly via glafic's amoeba"],
                "defaulted": {}, "ndim": 0})
    files = [os.path.join(INPUT_DIR, p) for p in data["files"]]
    cfg, issues = load_config(files, backend=engine, with_defaults=True)
    errors = [i.message for i in issues if i.is_error]
    extend = is_extend_mode(cfg)
    mode = _resolve_mode(plan, cfg) if not errors else "?"
    defaulted = _display_defaults(cfg, engine, mode) if not errors else {}
    return jsonify({
        "ok": not errors,
        "mode": "extend" if extend else mode,
        "engine": engine,
        "optimizer": plan["optimizer"],
        "errors": errors,
        "warnings": [i.message for i in issues if not i.is_error],
        "defaulted": {k: _jsonable(v) for k, v in defaulted.items()},
        "ndim": OptProblem(cfg, extend_mode=extend).ndim if not errors else 0,
    })


@app.route("/api/run", methods=["POST"])
def run_start():
    from core.format import load_config
    data = request.get_json(force=True)
    rel_files = data["files"]
    force = bool(data.get("force"))

    plan, err = _plan_from_payload(data)
    if err is not None:
        body, status = err
        return jsonify(body), status
    engine = plan["engine"]

    # the glafic / amoeba rail runs glafic's amoeba. Native .input selections run
    # directly (no glade parsing / defaults confirmation); a mixed selection is
    # rejected.
    if plan["glafic_rail"]:
        kind = _glafic_selection_kind(rel_files)
        if kind == "mixed":
            return jsonify({"ok": False, "errors": [_MIXED_MSG]}), 200
        if kind == "native":
            job = jobs.start("glafic", [os.path.join("InputFiles", p) for p in rel_files],
                             mode="amoeba", force=True)
            return jsonify({"ok": True, "job_id": job.id, "terminal": job.terminal,
                            "mode": "amoeba"})

    files = [os.path.join(INPUT_DIR, p) for p in rel_files]
    cfg, issues = load_config(files, backend=engine, with_defaults=True)
    errors = [i.message for i in issues if i.is_error]
    if errors:
        # 200 (like needs_confirm) so the front-end renders the error list;
        # a 4xx would surface only as a generic "BAD REQUEST" message.
        return jsonify({"ok": False, "errors": errors}), 200
    mode = _resolve_mode(plan, cfg)
    if cfg.applied_defaults and not force:
        return jsonify({"ok": False, "needs_confirm": True,
                        "defaulted": {k: _jsonable(v) for k, v in
                                      _display_defaults(cfg, engine, mode).items()}}), 200

    job = jobs.start(engine, [os.path.join("InputFiles", p) for p in rel_files],
                     mode=mode, force=True, optimizer=plan["optimizer"])
    return jsonify({"ok": True, "job_id": job.id, "terminal": job.terminal,
                    "mode": mode, "optimizer": plan["optimizer"]})


@app.route("/api/run/<job_id>/stream")
def run_stream(job_id):
    def gen():
        yield "retry: 3000\n\n"
        for line in jobs.tail(job_id):
            yield f"data: {line}\n\n"
        yield "event: end\ndata: end\n\n"
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/run/<job_id>/status")
def run_status(job_id):
    return jsonify(jobs.status(job_id))


@app.route("/api/run/<job_id>/result/<path:fname>")
def run_result(job_id, fname):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    target = os.path.abspath(os.path.join(job.job_dir, fname))
    if not target.startswith(os.path.abspath(job.job_dir)) or not os.path.isfile(target):
        return jsonify({"error": "not found"}), 404
    return send_file(target)


def _jsonable(v):
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


if __name__ == "__main__":
    port = int(os.environ.get("GLADE_PORT", "6017"))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
