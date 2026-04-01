#!/usr/bin/env python3
"""GLADE WebUI — Flask 后端"""
from __future__ import annotations

import json
import os
import queue
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

GLADE_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(GLADE_ROOT))

from injector import render_script_with_overrides  # noqa: E402

MODEL_TO_DIR = {
    "point_mass": "v_pointmass_1_0",
    "nfw": "v_nfw_2_0",
    "king": "v_king_1_0",
    "p-jaffe": "v_p_jaffe_2_0",
    "none": "v_none_1_0",
}
MODEL_TO_ENTRY = {
    "point_mass": "version_pointmass_1_0.py",
    "nfw": "version_nfw_2_0.py",
    "king": "version_king_1_0.py",
    "p-jaffe": "version_p_jaffe_2_0.py",
    "none": "version_none_1_0.py",
}

app = Flask(__name__, template_folder=str(WEB_ROOT / "templates"))
_jobs: dict[str, dict[str, Any]] = {}


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    glafic_python = GLADE_ROOT / "glafic2" / "python"
    glafic_bin_dir = GLADE_ROOT / "glafic2"
    local_lib_dir = GLADE_ROOT / "deps" / "install" / "lib"
    tools_dir = GLADE_ROOT / "tools"
    env["GLADE_ROOT"] = str(GLADE_ROOT)
    env["GLAFIC_HOME"] = str(glafic_bin_dir)
    env["GLAFIC_PYTHON_PATH"] = str(glafic_python)
    env["GLAFIC_LIB_PATH"] = str(local_lib_dir)
    env["MPLBACKEND"] = "Agg"          # 非交互式后端，避免子线程 GUI 警告
    env["PYTHONUNBUFFERED"] = "1"      # 禁用 Python 输出缓冲，确保实时流式输出
    env["PYTHONPATH"] = ":".join(
        filter(None, [str(glafic_python), str(tools_dir), env.get("PYTHONPATH", "")])
    )
    env["LD_LIBRARY_PATH"] = ":".join(
        filter(None, [str(local_lib_dir), env.get("LD_LIBRARY_PATH", "")])
    )
    env["PATH"] = ":".join(
        filter(None, [str(glafic_bin_dir), env.get("PATH", "")])
    )
    return env


@app.route("/")
def index():
    return render_template("index.html", glade_root=str(GLADE_ROOT))


@app.route("/api/save_defaults", methods=["POST"])
def api_save_defaults():
    """将当前参数直接覆写回原始模型脚本（设为默认值）。"""
    data = request.get_json(force=True)
    model = data.get("model")
    if model not in MODEL_TO_DIR:
        return jsonify({"error": f"未知模型: {model}"}), 400

    overrides: dict[str, Any] = data.get("overrides", {})
    if "fine_tuning_configs" in overrides:
        ftc = overrides.pop("fine_tuning_configs")
        overrides["fine_tuning_configs"] = {int(k): v for k, v in ftc.items()}

    legacy_dir = GLADE_ROOT / "legacy" / MODEL_TO_DIR[model]
    source_script = legacy_dir / MODEL_TO_ENTRY[model]

    # ── 处理手动输入的 bestfit.dat ──────────────────────────────
    bf_mode    = data.get("bf_mode", "path")
    bf_content = data.get("bf_content", "").strip()
    extra_msg  = ""
    if bf_mode == "manual" and bf_content:
        bf_dir = legacy_dir / "bestfit_default"
        bf_dir.mkdir(exist_ok=True)
        (bf_dir / "bestfit.dat").write_text(bf_content, encoding="utf-8")
        overrides["BASELINE_LENS_DIR"] = str(bf_dir.resolve())
        extra_msg = f"，bestfit.dat 已保存至 {bf_dir / 'bestfit.dat'}"

    try:
        render_script_with_overrides(source_script, overrides, source_script)
        return jsonify({"ok": True, "msg": f"已写入 {MODEL_TO_ENTRY[model]}{extra_msg}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/parse_bestfit", methods=["POST"])
def api_parse_bestfit():
    """解析 bestfit.dat 内容，返回透镜结构与源参数信息。"""
    content = (request.get_json(force=True) or {}).get("content", "")
    lens_lines, point_params = [], None
    for raw in content.splitlines():
        parts = raw.strip().split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "lens" and len(parts) >= 10:
            lens_lines.append(parts)
        elif parts[0] == "point" and len(parts) >= 4:
            point_params = parts
    n_sers = sum(1 for p in lens_lines if p[1] == "sers")
    main_types = [p[1] for p in lens_lines if p[1] != "sers"]
    valid = len(lens_lines) >= 1 and point_params is not None
    return jsonify({
        "valid": valid,
        "n_lens": len(lens_lines),
        "n_sers": n_sers,
        "lens_types": [p[1] for p in lens_lines],
        "main_lens_type": main_types[0] if main_types else (lens_lines[0][1] if lens_lines else None),
        "source_z": float(point_params[1]) if point_params else None,
        "source_x": float(point_params[2]) if point_params else None,
        "source_y": float(point_params[3]) if point_params else None,
        "error": None if valid else "至少需要1行 lens 和1行 point",
    })


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json(force=True)
    model = data.get("model")
    if model not in MODEL_TO_DIR:
        return jsonify({"error": f"未知模型: {model}"}), 400

    overrides: dict[str, Any] = data.get("overrides", {})

    # JSON 将 dict 的整数键序列化为字符串，需转回 int
    if "fine_tuning_configs" in overrides:
        ftc = overrides.pop("fine_tuning_configs")
        overrides["fine_tuning_configs"] = {int(k): v for k, v in ftc.items()}

    bf_mode    = data.get("bf_mode", "path")
    bf_content = data.get("bf_content", "").strip()
    source_dir = data.get("source_dir", "").strip()
    results_root = data.get("results_root", "results").strip() or "results"

    # 路径模式：解析目录路径
    if bf_mode == "path" and source_dir:
        sp = Path(source_dir)
        if not sp.is_absolute():
            sp = GLADE_ROOT.parent.parent / sp
        overrides["BASELINE_LENS_DIR"] = str(sp.resolve())

    job_id = str(uuid.uuid4())
    q: queue.Queue[dict] = queue.Queue()
    _jobs[job_id] = {"queue": q, "done": False, "returncode": None, "process": None}

    def _run() -> None:
        log_path: Path | None = None
        wrapper_path: Path | None = None
        rc_path: Path | None = None
        tmp_bf_dir: str | None = None
        try:
            legacy_dir = GLADE_ROOT / "legacy" / MODEL_TO_DIR[model]
            source_script = legacy_dir / MODEL_TO_ENTRY[model]
            generated_script = legacy_dir / "_glade_generated_run.py"

            # ── 手动输入模式：将 bestfit.dat 内容写入临时目录 ─────────────
            if bf_mode == "manual" and bf_content:
                tmp_bf_dir = tempfile.mkdtemp(prefix="glade_bf_")
                Path(tmp_bf_dir, "bestfit.dat").write_text(bf_content)
                overrides["BASELINE_LENS_DIR"] = tmp_bf_dir

            # ── Step 1: 注入参数（同时保存了当前配置）────────────────────
            render_script_with_overrides(source_script, overrides, generated_script)

            output_dir = GLADE_ROOT / results_root / model
            output_dir.mkdir(parents=True, exist_ok=True)

            venv_py = GLADE_ROOT / ".venv" / "bin" / "python"
            py = str(venv_py) if venv_py.exists() else sys.executable

            tag = job_id[:8]
            log_path     = output_dir / f"_glade_{tag}.log"
            rc_path      = output_dir / f"_glade_{tag}.rc"
            wrapper_path = output_dir / f"_glade_{tag}.sh"

            # ── Step 2: 写 bash 启动脚本 ────────────────────────────────
            # 脚本自带完整环境变量，由 bash 直接执行，与 Flask 零关联。
            env = _build_env()
            exports = "\n".join(
                f"export {k}={shlex.quote(v)}" for k, v in sorted(env.items())
            )
            wrapper_path.write_text(
                f"#!/bin/bash\n"
                f"{exports}\n"
                f"cd {shlex.quote(str(output_dir))}\n"
                f"{shlex.quote(py)} -u {shlex.quote(str(generated_script))}\n"
                f"echo $? > {shlex.quote(str(rc_path))}\n"
            )
            wrapper_path.chmod(
                wrapper_path.stat().st_mode
                | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )

            # ── Step 3: nohup 后台启动，与 Flask 进程树完全分离 ──────────
            # nohup + & 让脚本在新进程组中后台运行，bash 退出后由 init 接管，
            # 与 Flask 父进程再无任何关联，消除进程树上的所有调度干扰。
            launch = subprocess.run(
                ["bash", "-c",
                 f"nohup bash {shlex.quote(str(wrapper_path))} "
                 f">{shlex.quote(str(log_path))} 2>&1 & echo $!"],
                capture_output=True, text=True,
                close_fds=True,                   # Flask 的所有 fd 不传入
            )
            pid = int(launch.stdout.strip())
            _jobs[job_id]["pid"] = pid

            # ── Step 4: 轮询日志文件，去重后推送到 SSE 队列 ─────────────
            DEDUP_TTL     = 3.0
            POLL_INTERVAL = 0.05
            _seen: dict[str, float] = {}

            def _alive() -> bool:
                return os.path.exists(f"/proc/{pid}")

            def _send(line: str) -> None:
                stripped = line.strip()
                if not stripped:
                    q.put({"type": "output", "data": line})
                    return
                now = time.monotonic()
                for k in [k for k, t in _seen.items() if now - t > DEDUP_TTL]:
                    del _seen[k]
                if line in _seen:
                    return
                _seen[line] = now
                q.put({"type": "output", "data": line})

            # 等待日志文件出现（最多 10 s）
            for _ in range(200):
                if log_path.exists():
                    break
                time.sleep(0.05)

            with open(log_path, "r") as rf:
                while _alive():
                    chunk = rf.readline()
                    if chunk:
                        _send(chunk.rstrip("\n"))
                    else:
                        time.sleep(POLL_INTERVAL)
                for chunk in rf:          # 进程退出后读取剩余输出
                    _send(chunk.rstrip("\n"))

            # 读取退出码
            rc = 0
            if rc_path and rc_path.exists():
                try:
                    rc = int(rc_path.read_text().strip())
                except (ValueError, OSError):
                    rc = 0

            q.put({"type": "done", "returncode": rc})

        except Exception as exc:
            q.put({"type": "error", "data": str(exc)})
            q.put({"type": "done", "returncode": 1})
        finally:
            for p in [log_path, wrapper_path, rc_path]:
                if p and p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            if tmp_bf_dir:
                import shutil
                shutil.rmtree(tmp_bf_dir, ignore_errors=True)
            _jobs[job_id]["done"] = True

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def api_stream(job_id: str):
    if job_id not in _jobs:
        return jsonify({"error": "任务不存在"}), 404
    job = _jobs[job_id]

    def generate():
        while True:
            try:
                msg = job["queue"].get(timeout=1.0)
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                if msg.get("type") == "done":
                    break
            except queue.Empty:
                if job["done"]:
                    break
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/stop/<job_id>", methods=["POST"])
def api_stop(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "任务不存在"}), 404
    pid = job.get("pid")
    if pid:
        try:
            # 终止整个进程组（主进程 + bash wrapper + 所有 DE worker 子进程）
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            # 进程已退出或无权限，直接按 pid 尝试
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
    return jsonify({"status": "ok"})


@app.route("/api/status")
def api_status():
    return jsonify({"glade_root": str(GLADE_ROOT), "models": list(MODEL_TO_DIR.keys())})


if __name__ == "__main__":
    port = int(os.environ.get("GLADE_PORT", 6017))
    print(f"\n{'='*60}")
    print(f"  GLADE WebUI")
    print(f"  访问地址: http://localhost:{port}")
    print(f"  GLADE 根目录: {GLADE_ROOT}")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
