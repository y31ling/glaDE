"""Convenience entry points tying parse + merge + defaults + validate together."""
from __future__ import annotations

import os
from typing import Optional, Sequence

from . import schema
from .config import GladeConfig, apply_defaults, merge
from .diagnostics import Issue
from .parser import parse_file, parse_text
from .values import ParsedFile


def _unit_search_dirs(paths: Sequence[str]) -> list[str]:
    """Where UnitSetting profiles are looked up: next to each selected .dat,
    one level up (a run subfolder's InputFiles root), and ./InputFiles."""
    dirs: list[str] = []
    for p in paths:
        d = os.path.dirname(os.path.abspath(p))
        for cand in (d, os.path.dirname(d)):
            if cand and cand not in dirs:
                dirs.append(cand)
    default = os.path.join(os.getcwd(), "InputFiles")
    if os.path.isdir(default) and default not in dirs:
        dirs.append(default)
    return dirs


def _resolve_units(parsed: Sequence[ParsedFile],
                   paths: Sequence[str]) -> tuple[Optional[dict], list[Issue]]:
    """Read the (pre-merge) ``UnitSetting`` key and load its profile."""
    from .units import resolve_profile

    name = None
    for pf in parsed:
        v = pf.scalar("UnitSetting")
        if isinstance(v, str) and v:
            name = v
            break
    if name is None:
        return None, []
    return resolve_profile(name, _unit_search_dirs(paths))


def load_config(
    paths: Sequence[str],
    backend: Optional[str] = None,
    with_defaults: bool = True,
) -> tuple[GladeConfig, list[Issue]]:
    """Parse, merge, default and validate a selection of ``.dat`` files.

    Returns ``(config, issues)``. Syntax errors are converted to error issues so
    a UI can surface every problem at once rather than aborting on the first.

    A ``UnitSetting = '<profile>'`` key selects a unit profile
    (``<profile>.units.json`` next to the .dat / under InputFiles/); authored
    values are converted to the engine convention during the merge.
    """
    from .diagnostics import GladeSyntaxError
    from .validate import validate

    parsed: list[ParsedFile] = []
    issues: list[Issue] = []
    for path in paths:
        try:
            parsed.append(parse_file(path))
        except GladeSyntaxError as exc:
            issues.append(exc.as_issue())
    units, unit_issues = _resolve_units(parsed, paths)
    issues.extend(unit_issues)
    cfg, merge_issues = merge(parsed, units=units)
    issues.extend(merge_issues)
    if with_defaults:
        apply_defaults(cfg)
    issues.extend(validate(cfg, backend=backend))
    return cfg, issues


def lint_text(
    text: str,
    path: str = "<text>",
    backend: Optional[str] = None,
    with_defaults: bool = False,
) -> tuple[Optional[GladeConfig], list[Issue]]:
    """Parse + validate a single in-memory document (for the editor)."""
    from .diagnostics import GladeSyntaxError
    from .validate import validate

    try:
        pf = parse_text(text, path=path)
    except GladeSyntaxError as exc:
        return None, [exc.as_issue()]
    real = [path] if path and os.path.isfile(path) else []
    units, issues = _resolve_units([pf], real)
    cfg, merge_issues = merge([pf], units=units)
    issues.extend(merge_issues)
    if with_defaults:
        apply_defaults(cfg)
    issues.extend(validate(cfg, backend=backend))
    return cfg, issues


def has_errors(issues: Sequence[Issue]) -> bool:
    return any(i.is_error for i in issues)
