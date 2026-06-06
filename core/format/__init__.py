"""GLADE V0.4 ``.dat`` configuration format: parse, merge, default, validate.

See ``core/SPEC.md`` for the format specification.

Typical use::

    from core.format import load_config, has_errors
    cfg, issues = load_config(["constants.dat", "lens.dat"], backend="cpu")
    if has_errors(issues):
        for i in issues:
            print(i)
"""
from __future__ import annotations

from .api import has_errors, lint_text, load_config
from .config import GladeConfig, apply_defaults, merge
from .defaults import DEFAULTS
from .diagnostics import ERROR, WARNING, GladeSyntaxError, Issue
from .parser import parse_file, parse_text
from .validate import validate
from .values import (
    Assignment,
    Bounds,
    Component,
    Fixed,
    ParamValue,
    ParsedFile,
    Ref,
    Unfilled,
)
from . import schema

__all__ = [
    "load_config", "lint_text", "has_errors",
    "parse_text", "parse_file",
    "merge", "apply_defaults", "validate",
    "GladeConfig", "Issue", "ERROR", "WARNING", "GladeSyntaxError",
    "Fixed", "Bounds", "Unfilled", "Ref", "Component", "Assignment",
    "ParsedFile", "ParamValue", "DEFAULTS", "schema",
]
