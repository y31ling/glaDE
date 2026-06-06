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

app = Flask(__name__, static_folder=os.path.join(_THIS, "static"), static_url_path="/static")

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


@app.route("/api/files/export", methods=["POST"])
def files_export():
    """Translate selected glade .dat file(s) into a glafic input + obs file."""
    from core.format.config import apply_defaults, merge
    from core.format.parser import parse_file
    from core.translate import glade_to_glafic
    data = request.get_json(force=True)
    paths = [os.path.join(INPUT_DIR, p) for p in data["files"]]
    cfg, _ = merge([parse_file(p) for p in paths])
    apply_defaults(cfg)
    out = glade_to_glafic(cfg)
    base = data.get("name", "glade_export")
    written = []
    if out.get("model"):
        store.write(f"{base}_model.input", out["model"]); written.append(f"{base}_model.input")
    if out.get("obs"):
        store.write(f"{base}_obs.dat", out["obs"]); written.append(f"{base}_obs.dat")
    return jsonify({"ok": True, "written": written})


# --------------------------------------------------------------------------- #
# templates
# --------------------------------------------------------------------------- #
@app.route("/api/templates")
def templates():
    return jsonify(template_tree())


# --------------------------------------------------------------------------- #
# run (validate -> confirm defaults -> spawn terminal -> SSE)
# --------------------------------------------------------------------------- #
def _resolve_engine_mode(rail_backend: str, cfg) -> tuple[str, str]:
    """Map a FindImage rail choice to (engine, mode).

    'mcmc' rail -> MCMC-only on the CPU/glafic engine.
    'cpu'/'gpu'/'glafic' -> DE; also MCMC afterwards iff MCMC_ENABLED is set.
    """
    if rail_backend == "mcmc":
        return "cpu", "mcmc"
    mode = "de+mcmc" if bool(cfg.algorithm.get("MCMC_ENABLED", False)) else "findimage"
    return rail_backend, mode


@app.route("/api/run/check", methods=["POST"])
def run_check():
    """Validate a selection without launching. Returns blocking errors and the
    basic variables that would fall back to defaults (for the confirm dialog)."""
    from core.format import load_config
    from core.optimize.problem import OptProblem
    data = request.get_json(force=True)
    rail = data["backend"]
    engine = "cpu" if rail == "mcmc" else rail
    files = [os.path.join(INPUT_DIR, p) for p in data["files"]]
    cfg, issues = load_config(files, backend=engine, with_defaults=True)
    errors = [i.message for i in issues if i.is_error]
    _engine, mode = _resolve_engine_mode(rail, cfg)
    defaulted = {name: cfg.all_scalars().get(name) for name in cfg.applied_defaults}
    return jsonify({
        "ok": not errors,
        "mode": mode,
        "engine": engine,
        "errors": errors,
        "warnings": [i.message for i in issues if not i.is_error],
        "defaulted": {k: _jsonable(v) for k, v in defaulted.items()},
        "ndim": OptProblem(cfg).ndim if not errors else 0,
    })


@app.route("/api/run", methods=["POST"])
def run_start():
    from core.format import load_config
    data = request.get_json(force=True)
    rail = data["backend"]
    rel_files = data["files"]
    force = bool(data.get("force"))
    engine = "cpu" if rail == "mcmc" else rail
    files = [os.path.join(INPUT_DIR, p) for p in rel_files]

    cfg, issues = load_config(files, backend=engine, with_defaults=True)
    errors = [i.message for i in issues if i.is_error]
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400
    if cfg.applied_defaults and not force:
        return jsonify({"ok": False, "needs_confirm": True,
                        "defaulted": {k: _jsonable(cfg.all_scalars().get(k))
                                      for k in cfg.applied_defaults}}), 200

    _engine, mode = _resolve_engine_mode(rail, cfg)
    job = jobs.start(engine, [os.path.join("InputFiles", p) for p in rel_files],
                     mode=mode, force=True)
    return jsonify({"ok": True, "job_id": job.id, "terminal": job.terminal, "mode": mode})


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
