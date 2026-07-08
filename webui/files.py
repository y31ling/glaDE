"""Safe file operations confined to the ``InputFiles/`` directory."""
from __future__ import annotations

import os
import shutil
from typing import Optional


class FileStore:
    def __init__(self, root_dir: str):
        self.root = os.path.abspath(root_dir)
        os.makedirs(self.root, exist_ok=True)

    # -- path safety ---------------------------------------------------------
    def _abs(self, rel: str) -> str:
        rel = (rel or "").lstrip("/")
        p = os.path.abspath(os.path.join(self.root, rel))
        if p != self.root and not p.startswith(self.root + os.sep):
            raise ValueError("path escapes InputFiles")
        return p

    def _rel(self, abs_path: str) -> str:
        return os.path.relpath(abs_path, self.root).replace(os.sep, "/")

    # -- tree ----------------------------------------------------------------
    def tree(self) -> dict:
        def node(abs_path):
            name = os.path.basename(abs_path) or "InputFiles"
            rel = "" if abs_path == self.root else self._rel(abs_path)
            if os.path.isdir(abs_path):
                children = []
                for entry in sorted(os.listdir(abs_path),
                                    key=lambda e: (not os.path.isdir(os.path.join(abs_path, e)),
                                                   e.lower())):
                    if entry.startswith("."):
                        continue
                    children.append(node(os.path.join(abs_path, entry)))
                return {"name": name, "path": rel, "type": "dir", "children": children}
            return {"name": name, "path": rel, "type": "file"}
        return node(self.root)

    # -- read / write --------------------------------------------------------
    def read(self, rel: str) -> str:
        with open(self._abs(rel), "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def write(self, rel: str, content: str) -> None:
        p = self._abs(rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(content)

    # -- create / rename / delete -------------------------------------------
    def create_file(self, rel: str, content: str = "") -> str:
        p = self._abs(rel)
        if os.path.exists(p):
            raise FileExistsError(rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(content)
        return self._rel(p)

    def create_folder(self, rel: str) -> str:
        p = self._abs(rel)
        os.makedirs(p, exist_ok=False)
        return self._rel(p)

    def rename(self, rel: str, new_name: str) -> str:
        if "/" in new_name or new_name in ("", ".", ".."):
            raise ValueError("invalid name")
        src = self._abs(rel)
        dst = os.path.join(os.path.dirname(src), new_name)
        if os.path.exists(dst):
            raise FileExistsError(new_name)
        os.rename(src, dst)
        return self._rel(dst)

    def copy(self, src_rel: str, dst_rel: str) -> str:
        src = self._abs(src_rel)
        dst = self._abs(dst_rel)
        if not os.path.exists(src):
            raise ValueError(f"source not found: {src_rel}")
        if src == self.root:
            raise ValueError("cannot copy the root")
        if os.path.exists(dst):
            raise FileExistsError(f"already exists: {dst_rel}")
        if os.path.isdir(src):
            if dst == src or dst.startswith(src + os.sep):
                raise ValueError("cannot paste a folder into itself")
            shutil.copytree(src, dst)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        return self._rel(dst)

    def delete(self, rel: str) -> None:
        p = self._abs(rel)
        if p == self.root:
            raise ValueError("cannot delete the root")
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)

    def exists(self, rel: str) -> bool:
        try:
            return os.path.exists(self._abs(rel))
        except ValueError:
            return False
