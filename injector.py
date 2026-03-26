from __future__ import annotations

import ast
from pathlib import Path
from pprint import pformat
from typing import Any


def _literal(value: Any) -> str:
    return pformat(value, width=88, compact=False, sort_dicts=False)


def render_script_with_overrides(source_path: Path, overrides: dict[str, Any], output_path: Path) -> None:
    """将源码中同名顶层赋值替换为 overrides 值并写入 output_path。"""
    text = source_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)

    ranges: list[tuple[int, int, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        key = node.targets[0].id
        if key not in overrides:
            continue
        start = node.lineno
        end = node.end_lineno or node.lineno
        replacement = f"{key} = {_literal(overrides[key])}"
        ranges.append((start, end, replacement))

    if not ranges:
        output_path.write_text(text, encoding="utf-8")
        return

    # 从后往前替换，避免行号偏移
    rendered = lines[:]
    for start, end, replacement in sorted(ranges, key=lambda x: x[0], reverse=True):
        rendered[start - 1 : end] = [replacement]

    output_path.write_text("\n".join(rendered) + "\n", encoding="utf-8")
