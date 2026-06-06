"""Convenience entry points tying parse + merge + defaults + validate together."""
from __future__ import annotations

import os
from typing import Optional, Sequence

from . import schema
from .config import GladeConfig, apply_defaults, merge
from .diagnostics import Issue
from .parser import parse_file, parse_text
from .values import ParsedFile


def load_config(
    paths: Sequence[str],
    backend: Optional[str] = None,
    with_defaults: bool = True,
) -> tuple[GladeConfig, list[Issue]]:
    """Parse, merge, default and validate a selection of ``.dat`` files.

    Returns ``(config, issues)``. Syntax errors are converted to error issues so
    a UI can surface every problem at once rather than aborting on the first.
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
    cfg, merge_issues = merge(parsed)
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
    cfg, issues = merge([pf])
    if with_defaults:
        apply_defaults(cfg)
    issues.extend(validate(cfg, backend=backend))
    return cfg, issues


def has_errors(issues: Sequence[Issue]) -> bool:
    return any(i.is_error for i in issues)
