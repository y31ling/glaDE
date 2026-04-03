from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from injector import render_script_with_overrides


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


def _build_overrides(
    model: str,
    source_dir: str,
    common_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    glade_root: Path,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    overrides.update(common_cfg)
    overrides.update(model_cfg.get(model, {}))

    if source_dir:
        source_path = Path(source_dir)
        if not source_path.is_absolute():
            project_root = glade_root.parent.parent
            source_path = project_root / source_path
        overrides["BASELINE_LENS_DIR"] = str(source_path.resolve())
    return overrides


def run_selected_model(
    model: str,
    glade_root: Path,
    source_dir: str,
    common_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    results_root: str,
) -> int:
    if model not in MODEL_TO_DIR:
        raise ValueError(f"不支持的 model_use: {model}")

    legacy_dir = glade_root / "legacy" / MODEL_TO_DIR[model]
    source_script = legacy_dir / MODEL_TO_ENTRY[model]
    generated_script = legacy_dir / "_glade_generated_run.py"

    if not source_script.exists():
        raise FileNotFoundError(f"模型入口不存在: {source_script}")

    overrides = _build_overrides(model, source_dir, common_cfg, model_cfg, glade_root)
    render_script_with_overrides(source_script, overrides, generated_script)

    output_dir = glade_root / results_root / model
    output_dir.mkdir(parents=True, exist_ok=True)

    glafic_python = glade_root / "glafic2" / "python"
    glafic_bin_dir = glade_root / "glafic2"
    local_lib_dir = glade_root / "deps" / "install" / "lib"
    tools_dir = glade_root / "tools"

    env = os.environ.copy()
    env["GLADE_ROOT"] = str(glade_root)
    env["GLAFIC_HOME"] = str(glafic_bin_dir)
    env["GLAFIC_PYTHON_PATH"] = str(glafic_python)
    env["GLAFIC_LIB_PATH"] = str(local_lib_dir)

    env["PYTHONPATH"] = ":".join(
        [str(glafic_python), str(tools_dir), env.get("PYTHONPATH", "")]
    ).strip(":")
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(local_lib_dir), env.get("LD_LIBRARY_PATH", "")]
    ).strip(":")
    env["PATH"] = ":".join(
        [str(glafic_bin_dir), env.get("PATH", "")]
    ).strip(":")

    venv_python = glade_root / ".venv" / "bin" / "python"
    py_exec = str(venv_python) if venv_python.exists() else sys.executable
    cmd = [py_exec, str(generated_script)]
    print(f"运行模型: {model}")
    print(f"入口脚本: {source_script}")
    print(f"输出目录: {output_dir}")
    print(f"参数覆盖键: {sorted(overrides.keys())}")

    proc = subprocess.run(cmd, cwd=str(output_dir), check=False, env=env)
    return proc.returncode
