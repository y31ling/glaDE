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


# ── 观测数据解析辅助 ─────────────────────────────────────────────────────

def _parse_obs_point_content(content: str) -> dict:
    """
    解析 glafic readobs_point 格式文本，返回 GLADE 兼容的数组字典。

    格式：
      source_id  n_images  zs  zserr
      x(arcsec)  y(arcsec)  mu  pos_sigma(arcsec)  mu_err  td  td_err  parity
      ...（共 n_images 行）

    返回：
      {
        "n_images": int,
        "zs": float,
        "obs_positions_mas_list": [[x,y], ...],   # 以 mas 计，math 坐标
        "obs_magnifications_list": [mu, ...],
        "obs_mag_errors_list": [mu_err, ...],
        "obs_pos_sigma_mas_list": [sigma_mas, ...],
      }
    """
    # 去掉注释，提取有效行
    active_lines: list[str] = []
    for raw in content.splitlines():
        stripped = raw.split('#')[0].strip()
        if stripped:
            active_lines.append(stripped)

    if not active_lines:
        raise ValueError("内容为空")

    sources: dict[int, dict] = {}
    i = 0
    while i < len(active_lines):
        parts = active_lines[i].split()
        if len(parts) < 2:
            i += 1
            continue
        try:
            src_id = int(parts[0])
            n_img = int(parts[1])
            zs = float(parts[2]) if len(parts) > 2 else 0.0
            # zserr = float(parts[3]) if len(parts) > 3 else 0.0
        except (ValueError, IndexError):
            i += 1
            continue

        images = []
        i += 1
        for _ in range(n_img):
            if i >= len(active_lines):
                break
            ip = active_lines[i].split()
            if len(ip) < 7:
                break
            try:
                x = float(ip[0]);  y = float(ip[1])
                mu = float(ip[2]); pos_sig = float(ip[3]); mu_err = float(ip[4])
                parity = int(ip[7]) if len(ip) > 7 else 0
                images.append((x, y, mu, pos_sig, mu_err, parity))
            except (ValueError, IndexError):
                break
            i += 1

        if images:
            sources[src_id] = {"zs": zs, "images": images}

    if not sources:
        raise ValueError("未找到有效的观测数据（检查格式）")

    # 使用 source_id=1，或第一个可用 source
    src_key = 1 if 1 in sources else min(sources.keys())
    src = sources[src_key]

    obs_positions_mas, obs_mags, obs_mag_errs, obs_sigmas = [], [], [], []
    for x_arc, y_arc, mu, pos_sig_arc, mu_err, _parity in src["images"]:
        # glafic 格式为 arcsec math 坐标（右=正），转换为 mas
        # 存储为 obs_x_flip=False 格式（math 坐标不翻转）
        obs_positions_mas.append([x_arc * 1000.0, y_arc * 1000.0])
        obs_mags.append(mu)
        obs_mag_errs.append(mu_err)
        obs_sigmas.append(pos_sig_arc * 1000.0)

    return {
        "n_images": len(src["images"]),
        "zs": src["zs"],
        "obs_positions_mas_list": obs_positions_mas,
        "obs_magnifications_list": obs_mags,
        "obs_mag_errors_list": obs_mag_errs,
        "obs_pos_sigma_mas_list": obs_sigmas,
    }

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

    # ── 处理观测数据（file / manual 模式解析并注入到 overrides）────────
    obs_mode_sv = data.get("obs_mode", "table")
    obs_file_sv = data.get("obs_file", "").strip()
    obs_content_sv = data.get("obs_content", "").strip()

    if obs_mode_sv == "file" and obs_file_sv:
        try:
            p = Path(obs_file_sv)
            if not p.is_absolute():
                p = GLADE_ROOT.parent.parent / p
            obs_content_sv = p.read_text(encoding="utf-8")
            obs_mode_sv = "manual"
        except Exception as exc:
            return jsonify({"error": f"无法读取观测数据文件: {exc}"}), 400

    if obs_mode_sv == "manual" and obs_content_sv:
        try:
            parsed = _parse_obs_point_content(obs_content_sv)
            overrides["obs_positions_mas_list"]  = parsed["obs_positions_mas_list"]
            overrides["obs_magnifications_list"] = parsed["obs_magnifications_list"]
            overrides["obs_mag_errors_list"]     = parsed["obs_mag_errors_list"]
            overrides["obs_pos_sigma_mas_list"]  = parsed["obs_pos_sigma_mas_list"]
            overrides["obs_x_flip"] = False
        except Exception as exc:
            return jsonify({"error": f"观测数据解析失败: {exc}"}), 400

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


@app.route("/api/parse_obs_point", methods=["POST"])
def api_parse_obs_point():
    """解析 glafic readobs_point 格式，返回 GLADE 兼容数组。"""
    data = request.get_json(force=True) or {}
    content = data.get("content", "")
    filepath = data.get("filepath", "").strip()

    if filepath and not content:
        try:
            p = Path(filepath)
            if not p.is_absolute():
                p = GLADE_ROOT.parent.parent / p
            content = p.read_text(encoding="utf-8")
        except Exception as e:
            return jsonify({"error": f"无法读取文件: {e}"}), 400

    if not content.strip():
        return jsonify({"error": "内容为空"}), 400

    try:
        result = _parse_obs_point_content(content)
        result["ok"] = True
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/generate_glafic_input", methods=["POST"])
def api_generate_glafic_input():
    """根据当前参数生成 glafic .input 文件和 obs 观测文件。"""
    data = request.get_json(force=True) or {}
    model = data.get("model", "none")
    overrides: dict = data.get("overrides", {})
    bf_mode = data.get("bf_mode", "path")
    bf_content = data.get("bf_content", "")
    source_dir = data.get("source_dir", "").strip()
    obs_arrays: dict = data.get("obs_arrays", {})

    # ── 读取 bestfit.dat ────────────────────────────────────────────────
    if bf_mode == "path" and source_dir and not bf_content:
        try:
            sp = Path(source_dir)
            if not sp.is_absolute():
                sp = GLADE_ROOT.parent.parent / sp
            bf_path = sp / "bestfit.dat"
            if bf_path.exists():
                bf_content = bf_path.read_text(encoding="utf-8")
        except Exception:
            pass

    lens_lines: list = []
    point_params: list | None = None
    if bf_content:
        for raw in bf_content.splitlines():
            parts = raw.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "lens" and len(parts) >= 3:
                lens_lines.append(parts)
            elif parts[0] == "point" and len(parts) >= 4:
                point_params = parts

    source_z = float(point_params[1]) if point_params else 0.5
    source_x = float(point_params[2]) if point_params else 0.0
    source_y = float(point_params[3]) if point_params else 0.0

    # ── 优化标志 ─────────────────────────────────────────────────────────
    source_modify = bool(overrides.get("source_modify", False))
    lens_modify = bool(overrides.get("lens_modify", False))

    # ── Subhalo 参数 ───────────────────────────────────────────────────
    _MODEL_GLAFIC_LENS = {
        "point_mass": "point", "nfw": "gnfw", "king": "king", "p-jaffe": "jaffe",
    }
    active_subhalos: list[int] = overrides.get("active_subhalos", [])
    sub_lens_type = _MODEL_GLAFIC_LENS.get(model)
    has_subhalos = bool(sub_lens_type and active_subhalos)

    sub_lens_entries: list[tuple[str, float, list[float]]] = []
    if has_subhalos:
        obs_positions_mas: list = obs_arrays.get("obs_positions_mas_list", [])
        obs_x_flip_sub = bool(obs_arrays.get("obs_x_flip", overrides.get("obs_x_flip", False)))
        _xs = -1 if obs_x_flip_sub else 1
        offset_x = float(overrides.get("center_offset_x", 0.0))
        offset_y = float(overrides.get("center_offset_y", 0.0))
        fine_tuning = bool(overrides.get("fine_tuning", False))
        ftc: dict = overrides.get("fine_tuning_configs", {})
        lens_z = float(lens_lines[0][2]) if lens_lines else 0.5

        for img_idx in active_subhalos:
            k = img_idx - 1
            if k < 0 or k >= len(obs_positions_mas):
                continue
            xm, ym = obs_positions_mas[k]
            sub_x = _xs * xm / 1000.0 - _xs * offset_x
            sub_y = ym / 1000.0 - offset_y

            fc = ftc.get(img_idx, ftc.get(str(img_idx), {})) if fine_tuning else {}

            if model == "point_mass":
                mass = float(fc.get("mass_guess", overrides.get("MASS_GUESS", 1e6)))
                params = [mass, sub_x, sub_y, 0.0, 0.0, 0.0, 0.0]
            elif model == "nfw":
                mass = float(fc.get("mass_guess", overrides.get("MASS_GUESS", 1e6)))
                c_min = float(fc.get("conc_min", overrides.get("CONCENTRATION_MIN", 5.0)))
                c_max = float(fc.get("conc_max", overrides.get("CONCENTRATION_MAX", 50.0)))
                conc = (c_min + c_max) / 2.0
                params = [mass, sub_x, sub_y, 0.0, 0.0, conc, 1.0]
            elif model == "king":
                lm_min = float(fc.get("logM_min", overrides.get("LOGM_MIN", 3.5)))
                lm_max = float(fc.get("logM_max", overrides.get("LOGM_MAX", 8.0)))
                M = 10.0 ** ((lm_min + lm_max) / 2.0)
                rc_min = float(fc.get("rc_min", overrides.get("RC_MIN", 8e-5)))
                rc_max = float(fc.get("rc_max", overrides.get("RC_MAX", 8e-3)))
                rc = (rc_min + rc_max) / 2.0
                c_min = float(fc.get("c_min", overrides.get("C_MIN", 0.5)))
                c_max = float(fc.get("c_max", overrides.get("C_MAX", 2.5)))
                c_val = (c_min + c_max) / 2.0
                params = [M, sub_x, sub_y, 0.0, 0.0, rc, c_val]
            elif model == "p-jaffe":
                s_min = float(fc.get("sig_min", overrides.get("SIG_MIN", 0.01)))
                s_max = float(fc.get("sig_max", overrides.get("SIG_MAX", 30.0)))
                sig = (s_min + s_max) / 2.0
                a_min = float(fc.get("a_min", overrides.get("A_MIN", 0.0001)))
                a_max = float(fc.get("a_max", overrides.get("A_MAX", 0.3)))
                a_val = (a_min + a_max) / 2.0
                rco_min = float(fc.get("rco_min", overrides.get("RCO_MIN", 0.0)))
                rco_max = float(fc.get("rco_max", overrides.get("RCO_MAX", 0.05)))
                rco = (rco_min + rco_max) / 2.0
                params = [sig, sub_x, sub_y, 0.0, 0.0, a_val, rco]
            else:
                continue
            sub_lens_entries.append((sub_lens_type, lens_z, params))

    any_optimize = source_modify or lens_modify or has_subhalos

    # ── 宇宙学参数（使用脚本默认值）────────────────────────────────────
    omega = 0.3; lam = 0.7; weos = -1.0; hubble = 0.7
    xmin = -0.5; ymin = -0.5; xmax = 0.5; ymax = 0.5
    pix_ext = 0.01; pix_poi = 0.2; maxlev = 5

    # ── 主 input 文件 ────────────────────────────────────────────────────
    n_base_lens = len(lens_lines) if lens_lines else 1
    n_lens = n_base_lens + len(sub_lens_entries)
    IL: list[str] = []
    IL.append(f"## Generated by GLADE WebUI  model={model}")
    IL.append(f"## Run 'glafic <this_file>' to optimize with the amoeba algorithm.")
    IL.append("")
    IL.append(f"omega     {omega:.6f}")
    IL.append(f"lambda    {lam:.6f}")
    IL.append(f"weos      {weos:.6f}")
    IL.append(f"hubble    {hubble:.6f}")
    IL.append(f"prefix    out")
    IL.append(f"xmin      {xmin:.6f}")
    IL.append(f"ymin      {ymin:.6f}")
    IL.append(f"xmax      {xmax:.6f}")
    IL.append(f"ymax      {ymax:.6f}")
    IL.append(f"pix_ext   {pix_ext:.6f}")
    IL.append(f"pix_poi   {pix_poi:.6f}")
    IL.append(f"maxlev    {maxlev}")
    IL.append("")
    IL.append("chi2_usemag    0  ## 0=ignore mag; 1=use flux ratio; 2=use magnitude")
    IL.append("chi2_checknimg 1")
    IL.append("chi2_restart   -1")
    IL.append("")
    IL.append(f"startup {n_lens} 0 1")
    if lens_lines:
        for lp in lens_lines:
            ltype = lp[1]; z = float(lp[2])
            raw = [float(lp[i + 3]) if i + 3 < len(lp) else 0.0 for i in range(7)]
            ps = "  ".join(f"{p:.6e}" for p in raw)
            IL.append(f"lens {ltype}  {z:.4f}  {ps}")
    else:
        IL.append(f"## no lens data — add manually")
    for sl_type, sl_z, sl_params in sub_lens_entries:
        ps = "  ".join(f"{p:.6e}" for p in sl_params)
        IL.append(f"lens {sl_type}  {sl_z:.4f}  {ps}")
    IL.append(f"point {source_z:.4f}  {source_x:.6e}  {source_y:.6e}")
    IL.append("end_startup")
    IL.append("")

    # setopt
    if any_optimize and (lens_lines or sub_lens_entries):
        IL.append("start_setopt")
        for lp in lens_lines:
            ltype = lp[1]
            params = [float(lp[i + 3]) if i + 3 < len(lp) else 0.0 for i in range(7)]
            if lens_modify and ltype not in ("sers",):
                flags = [0] + [1 if abs(p) > 1e-30 else 0 for p in params]
            else:
                flags = [0] * 8
            IL.append("  " + " ".join(str(f) for f in flags) + f"  ## {ltype}")
        _SUB_SETOPT = {
            "point":  [0, 1, 1, 1, 0, 0, 0, 0],
            "gnfw":   [0, 1, 1, 1, 0, 0, 1, 0],
            "king":   [0, 1, 1, 1, 0, 0, 1, 1],
            "jaffe":  [0, 1, 1, 1, 0, 0, 1, 1],
        }
        for i, (sl_type, _, _) in enumerate(sub_lens_entries):
            sf = _SUB_SETOPT.get(sl_type, [0, 1, 1, 1, 0, 0, 0, 0])
            IL.append("  " + " ".join(str(f) for f in sf) + f"  ## subhalo {active_subhalos[i]} ({sl_type})")
        src_flags = "0 1 1" if source_modify else "0 0 0"
        IL.append(f"  {src_flags}  ## point source")
        IL.append("end_setopt")
        IL.append("")

    obs_filename = "obs_glade.dat"
    IL.append("start_command")
    IL.append("")
    IL.append(f"readobs_point {obs_filename}")
    IL.append("")
    if any_optimize:
        IL.append("optimize")
        IL.append("")
    IL.append("findimg")
    IL.append("")
    IL.append("quit")
    IL.append("end_command")
    input_content = "\n".join(IL)

    # ── obs 文件 ──────────────────────────────────────────────────────────
    obs_positions_mas: list = obs_arrays.get("obs_positions_mas_list", [])
    obs_magnifications: list = obs_arrays.get("obs_magnifications_list", [])
    obs_mag_errors: list = obs_arrays.get("obs_mag_errors_list", [])
    obs_pos_sigma_mas: list = obs_arrays.get("obs_pos_sigma_mas_list", [])
    obs_x_flip = bool(obs_arrays.get("obs_x_flip", False))

    OL: list[str] = []
    OL.append("## GLADE observation data — glafic readobs_point format")
    OL.append("## Positions in arcsec (math coordinates: right=positive, up=positive)")
    OL.append("## Columns: x  y  mu  pos_sigma  mu_err  time_delay  td_err  parity")

    if obs_positions_mas:
        n_img = len(obs_positions_mas)
        _xs = -1 if obs_x_flip else 1
        OL.append(f"1 {n_img} {source_z:.4f} 0.0")
        for k in range(n_img):
            xm, ym = obs_positions_mas[k]
            x_arc = _xs * xm / 1000.0
            y_arc = ym / 1000.0
            mu = obs_magnifications[k] if k < len(obs_magnifications) else 0.0
            sig_arc = (obs_pos_sigma_mas[k] / 1000.0) if k < len(obs_pos_sigma_mas) else 0.001
            mu_err = obs_mag_errors[k] if k < len(obs_mag_errors) else 0.2
            parity = -1 if mu < 0 else 0
            OL.append(f"  {x_arc:10.6f}  {y_arc:10.6f}  {mu:9.5f}  {sig_arc:10.6f}  {mu_err:8.5f}  0.000000  0.000000  {parity}")
    else:
        OL.append(f"## No obs data — fill in manually")
        OL.append(f"## 1 4 {source_z:.4f} 0.0")
        OL.append(f"##   x1  y1  mu1  pos_sigma1  mu_err1  td1  td_err1  parity1")
    obs_content = "\n".join(OL)

    return jsonify({
        "ok": True,
        "input_file": input_content,
        "obs_file": obs_content,
        "obs_filename": obs_filename,
    })


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

    # ── 观测数据处理 ──────────────────────────────────────────────────────
    obs_mode    = data.get("obs_mode", "table")    # "table" | "file" | "manual"
    obs_file    = data.get("obs_file", "").strip()
    obs_content_raw = data.get("obs_content", "").strip()

    if obs_mode == "file" and obs_file:
        try:
            p = Path(obs_file)
            if not p.is_absolute():
                p = GLADE_ROOT.parent.parent / p
            obs_content_raw = p.read_text(encoding="utf-8")
            obs_mode = "manual"   # 读取后统一走解析路径
        except Exception as exc:
            return jsonify({"error": f"无法读取观测数据文件: {exc}"}), 400

    if obs_mode == "manual" and obs_content_raw:
        try:
            parsed = _parse_obs_point_content(obs_content_raw)
            overrides["obs_positions_mas_list"]  = parsed["obs_positions_mas_list"]
            overrides["obs_magnifications_list"] = parsed["obs_magnifications_list"]
            overrides["obs_mag_errors_list"]     = parsed["obs_mag_errors_list"]
            overrides["obs_pos_sigma_mas_list"]  = parsed["obs_pos_sigma_mas_list"]
            overrides["obs_x_flip"] = False   # 已是 math 坐标，不再翻转
        except Exception as exc:
            return jsonify({"error": f"观测数据解析失败: {exc}"}), 400

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
            generated_script = legacy_dir / f"_glade_run_{job_id[:8]}.py"

            # ── 手动输入模式：将 bestfit.dat 内容写入临时目录 ─────────────
            if bf_mode == "manual" and bf_content:
                tmp_bf_dir = tempfile.mkdtemp(prefix="glade_bf_")
                Path(tmp_bf_dir, "bestfit.dat").write_text(bf_content)
                overrides["BASELINE_LENS_DIR"] = tmp_bf_dir
            # 注意：obs 数据已在上方提前解析并注入 overrides，此处无需额外处理

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
            for p in [log_path, wrapper_path, rc_path, generated_script]:
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
