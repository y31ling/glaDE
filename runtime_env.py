from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def glade_root_from_file(file_path: str) -> Path:
    p = Path(file_path).resolve()
    # tools/* 脚本在 glade/tools 下，主控在 glade 根目录
    if p.parent.name == "tools":
        return p.parent.parent
    return p.parent


def setup_runtime_env(glade_root: Union[Path, str]) -> None:
    glade_root = Path(glade_root).resolve()
    glafic_home = glade_root / "glafic2"
    glafic_python = glafic_home / "python"
    glafic_lib = glade_root / "deps" / "install" / "lib"
    tools_dir = glade_root / "tools"

    os.environ.setdefault("GLADE_ROOT", str(glade_root))
    os.environ.setdefault("GLAFIC_HOME", str(glafic_home))
    os.environ.setdefault("GLAFIC_PYTHON_PATH", str(glafic_python))
    os.environ.setdefault("GLAFIC_LIB_PATH", str(glafic_lib))

    py_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = ":".join(
        [str(glafic_python), str(tools_dir), py_path]
    ).strip(":")

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        [str(glafic_lib), ld_path]
    ).strip(":")

    path = os.environ.get("PATH", "")
    os.environ["PATH"] = ":".join([str(glafic_home), path]).strip(":")
