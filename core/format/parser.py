"""Parser for the GLADE V0.4 ``.dat`` configuration format.

The format is Python-flavoured but **not** executed as Python. We:

1. strip ``#`` comments (respecting string literals),
2. group physical lines into logical statements by balancing brackets,
3. classify each statement as a component tuple (``'name': ( ... )``) or a
   scalar assignment (``name = value``),
4. evaluate the right-hand side with a *restricted* AST walker that permits
   only numbers, unary +/-, name references, list/tuple literals and
   2-element ``{lo, hi}`` set literals (bounds) -- never calls, arithmetic,
   attributes or subscripts.

``$float`` / ``$int`` / ``$type{lower,upper}`` placeholders are rewritten to a
sentinel name before parsing and surface as :class:`~core.format.values.Unfilled`.
Cross-file references (e.g. ``lens_z``) stay as :class:`~core.format.values.Ref`
until merge resolves them.
"""
from __future__ import annotations

import ast
import re
from typing import Any

from . import schema
from .diagnostics import GladeSyntaxError
from .values import (
    Assignment,
    Bounds,
    Component,
    Expr,
    Fixed,
    ParsedFile,
    Ref,
    Unfilled,
)

# names like ``img1_x`` / ``img12_y`` alias an observed image position; they are
# only resolvable once observations are merged, so they defer to an Expr.
_IMG_ALIAS_RE = re.compile(r"^img\d+_[xy]$")
# arithmetic operators permitted inside a .dat expression.
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)

# ---------------------------------------------------------------------------
# placeholder handling
# ---------------------------------------------------------------------------

_PH_PREFIX = "__GLADE_PH__"
# $float, $int, $str, optionally followed by a {...} group (e.g. {lower,upper}).
_PLACEHOLDER_RE = re.compile(r"\$\s*(float|int|str)\s*(\{[^{}]*\})?")


def _placeholder_sentinel(kind: str, optimizable: bool) -> str:
    return f"{_PH_PREFIX}{kind}__{'opt' if optimizable else 'fix'}"


def _decode_sentinel(name: str) -> Unfilled | None:
    if not name.startswith(_PH_PREFIX):
        return None
    rest = name[len(_PH_PREFIX):]
    kind, _, tail = rest.partition("__")
    return Unfilled(kind=kind, optimizable=(tail == "opt"))


def _rewrite_placeholders(text: str) -> str:
    def repl(m: re.Match) -> str:
        kind = m.group(1)
        optimizable = m.group(2) is not None
        return _placeholder_sentinel(kind, optimizable)

    return _PLACEHOLDER_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# comment stripping + statement grouping
# ---------------------------------------------------------------------------

_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {")", "]", "}"}


def _strip_comment(line: str) -> str:
    out: list[str] = []
    in_str: str | None = None
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if in_str is not None:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(line[i + 1])
                i += 2
                continue
            if c == in_str:
                in_str = None
            i += 1
            continue
        if c in "\"'":
            in_str = c
            out.append(c)
        elif c == "#":
            break
        else:
            out.append(c)
        i += 1
    return "".join(out).rstrip()


def _bracket_delta(s: str) -> int:
    """Net bracket depth change of *s*, ignoring brackets inside strings."""
    depth = 0
    in_str: str | None = None
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if in_str is not None:
            if c == "\\":
                i += 2
                continue
            if c == in_str:
                in_str = None
            i += 1
            continue
        if c in "\"'":
            in_str = c
        elif c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        i += 1
    return depth


def _split_statements(text: str) -> list[tuple[str, int]]:
    """Yield ``(statement_text, starting_lineno)`` for each logical statement."""
    statements: list[tuple[str, int]] = []
    buf: list[str] = []
    depth = 0
    start_line = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw)
        if not buf and not line.strip():
            continue
        if not buf:
            start_line = lineno
        buf.append(line)
        depth += _bracket_delta(line)
        if depth <= 0:
            stmt = "\n".join(buf).strip()
            depth = 0
            buf = []
            if stmt:
                statements.append((stmt, start_line))
    if buf:  # unterminated bracket
        stmt = "\n".join(buf).strip()
        if stmt:
            statements.append((stmt, start_line))
    return statements


# ---------------------------------------------------------------------------
# restricted expression evaluation
# ---------------------------------------------------------------------------


def _defer(node: ast.AST) -> Expr:
    """Capture *node* as a deferred expression (resolved at merge time)."""
    return Expr(ast.unparse(node), is_bounds=isinstance(node, ast.Set))


def _eval_node(node: ast.AST, symbols: dict[str, float], lineno: int,
               path: str | None) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return node.value
        if isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node.value, str):
            return node.value
        if node.value is None:
            return None
        raise GladeSyntaxError(f"unsupported constant {node.value!r}", lineno, path)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _eval_node(node.operand, symbols, lineno, path)
        if isinstance(val, (int, float)):
            return -val if isinstance(node.op, ast.USub) else +val
        if isinstance(val, (Ref, Expr)):
            return _defer(node)               # e.g. -img1_x
        raise GladeSyntaxError("unary +/- requires a number", lineno, path)

    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        lhs = _eval_node(node.left, symbols, lineno, path)
        rhs = _eval_node(node.right, symbols, lineno, path)
        if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
            return _fold_binop(node.op, lhs, rhs, lineno, path)   # fold locally
        if isinstance(lhs, (int, float, Ref, Expr)) and \
                isinstance(rhs, (int, float, Ref, Expr)):
            return _defer(node)               # e.g. img1_x - 0.075
        raise GladeSyntaxError(
            "arithmetic operands must be numbers or references", lineno, path)

    if isinstance(node, ast.Name):
        ph = _decode_sentinel(node.id)
        if ph is not None:
            return ph
        if node.id in symbols:
            return symbols[node.id]
        if _IMG_ALIAS_RE.match(node.id):
            return _defer(node)               # img1_x ... -> resolved with obs data
        # unresolved -> defer to merge-time Ref resolution (lens_z, user vars)
        return Ref(node.id)

    if isinstance(node, ast.Subscript):
        return _defer(node)                   # obs_positions_mas_list[0][0] ...

    if isinstance(node, ast.Set):
        elts = [_eval_node(e, symbols, lineno, path) for e in node.elts]
        if len(elts) != 2:
            raise GladeSyntaxError(
                f"bounds {{lo, hi}} must have exactly 2 values, got {len(elts)}",
                lineno, path,
            )
        lo, hi = elts
        for v in (lo, hi):
            if isinstance(v, Unfilled):
                # a placeholder inside bounds -> whole thing is unfilled+opt
                return Unfilled(kind="float", optimizable=True)
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return Bounds(float(lo), float(hi))
        if isinstance(lo, (int, float, Ref, Expr)) and \
                isinstance(hi, (int, float, Ref, Expr)):
            return _defer(node)               # {img1_x-0.075, img1_x+0.075} ...
        raise GladeSyntaxError(
            "bounds {lo, hi} must be numbers or expressions", lineno, path)

    if isinstance(node, ast.List):
        return [_eval_node(e, symbols, lineno, path) for e in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(e, symbols, lineno, path) for e in node.elts)

    raise GladeSyntaxError(
        f"unsupported expression ({type(node).__name__}); only numbers, "
        "arithmetic, {lo,hi} bounds, lists, tuples, subscripts and name "
        "references are allowed",
        lineno, path,
    )


def _fold_binop(op: ast.operator, lhs: float, rhs: float, lineno: int,
                path: str | None) -> float:
    if isinstance(op, ast.Add):
        return lhs + rhs
    if isinstance(op, ast.Sub):
        return lhs - rhs
    if isinstance(op, ast.Mult):
        return lhs * rhs
    if isinstance(op, ast.Div):
        if rhs == 0:
            raise GladeSyntaxError("division by zero", lineno, path)
        return lhs / rhs
    if isinstance(op, ast.Pow):
        try:
            return lhs ** rhs
        except (ArithmeticError, ValueError) as exc:
            raise GladeSyntaxError(f"invalid '**' ({exc})", lineno, path) from None
    raise GladeSyntaxError("unsupported operator", lineno, path)  # pragma: no cover


def _eval_expr(expr_text: str, symbols: dict[str, float], lineno: int,
               path: str | None) -> Any:
    rewritten = _rewrite_placeholders(expr_text)
    try:
        tree = ast.parse(rewritten.strip(), mode="eval")
    except SyntaxError as exc:  # pragma: no cover - message passthrough
        raise GladeSyntaxError(f"invalid syntax: {exc.msg}", lineno, path) from None
    return _eval_node(tree.body, symbols, lineno, path)


# ---------------------------------------------------------------------------
# statement classification
# ---------------------------------------------------------------------------

# 'name' : ( ... )
_COMPONENT_RE = re.compile(r"""^\s*(['"])(?P<name>.*?)\1\s*:\s*(?P<rest>\(.*)$""",
                           re.DOTALL)

# Optional category suffix on the component index: '3l' forces the component to
# be treated as a main LENS, '3s' as a SUB-STRUCTURE (e.g. in the result
# triptych); a plain number keeps the model's default classification. The
# suffix is stripped before AST parsing ('3l' is not valid Python).
_INDEX_SUFFIX_RE = re.compile(r"^(\(\s*)(\d+)\s*([A-Za-z])(?=\s*,)")
_CATEGORY_SUFFIXES = {"l": "lens", "s": "substructure"}


def _find_top_assign(stmt: str) -> int:
    """Index of the top-level ``=`` (not ``==``/``<=``/...), or -1."""
    in_str: str | None = None
    depth = 0
    i = 0
    n = len(stmt)
    while i < n:
        c = stmt[i]
        if in_str is not None:
            if c == "\\":
                i += 2
                continue
            if c == in_str:
                in_str = None
            i += 1
            continue
        if c in "\"'":
            in_str = c
        elif c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        elif c == "=" and depth == 0:
            prev = stmt[i - 1] if i else ""
            nxt = stmt[i + 1] if i + 1 < n else ""
            if prev not in "=!<>" and nxt != "=":
                return i
        i += 1
    return -1


def _contains_expr(v: Any) -> bool:
    """Whether *v* is (or nests, inside a list/tuple) a deferred Expr."""
    if isinstance(v, Expr):
        return True
    if isinstance(v, (list, tuple)):
        return any(_contains_expr(e) for e in v)
    return False


def _wrap_param(v: Any, lineno: int, path: str | None, what: str):
    from .values import Fixed as _Fixed
    if isinstance(v, bool):
        raise GladeSyntaxError(f"{what} cannot be a boolean", lineno, path)
    if isinstance(v, (int, float)):
        return _Fixed(float(v))
    if isinstance(v, (Bounds, Unfilled, Ref, Expr)):
        return v
    raise GladeSyntaxError(
        f"{what} must be a number, {{lo,hi}} bounds, an expression, a reference "
        f"or a placeholder, got {type(v).__name__}", lineno, path)


def _parse_component(name: str, rest: str, lineno: int, path: str | None,
                     symbols: dict[str, float]) -> Component:
    category_override = None
    m = _INDEX_SUFFIX_RE.match(rest)
    if m:
        suffix = m.group(3)
        if suffix.lower() not in _CATEGORY_SUFFIXES:
            raise GladeSyntaxError(
                f"unknown component index suffix '{m.group(2)}{suffix}'; use "
                f"'{m.group(2)}l' (treat as lens), '{m.group(2)}s' (treat as "
                f"sub-structure) or a plain number (default classification)",
                lineno, path)
        category_override = _CATEGORY_SUFFIXES[suffix.lower()]
        rest = m.group(1) + m.group(2) + rest[m.end():]

    value = _eval_expr(rest, symbols, lineno, path)
    if not isinstance(value, tuple):
        raise GladeSyntaxError("component value must be a (...) tuple", lineno, path)
    if len(value) < 3:
        raise GladeSyntaxError(
            "component tuple needs at least (N, 'type', z)", lineno, path)

    raw_index_val, type_val, z_val, *param_vals = value

    raw_index = None
    if isinstance(raw_index_val, (int, float)) and not isinstance(raw_index_val, bool):
        raw_index = int(raw_index_val)

    if not isinstance(type_val, str):
        raise GladeSyntaxError(
            f"component type must be a quoted string, got {type_val!r}", lineno, path)

    z_wrapped = _wrap_param(z_val, lineno, path, "component redshift z")
    params = [
        _wrap_param(p, lineno, path, f"parameter p{i + 1}")
        for i, p in enumerate(param_vals)
    ]

    return Component(
        name=name,
        type=type_val,
        z=z_wrapped,
        params=params,
        raw_index=raw_index,
        category_override=category_override,
        source_file=path,
        lineno=lineno,
    )


def _parse_assignment(stmt: str, eq: int, lineno: int, path: str | None,
                      symbols: dict[str, float]) -> list[Assignment]:
    lhs = stmt[:eq].strip()
    rhs = stmt[eq + 1:].strip()
    targets = [t.strip() for t in lhs.split(",")] if "," in lhs else [lhs]
    for t in targets:
        if not t.isidentifier():
            raise GladeSyntaxError(f"invalid assignment target {t!r}", lineno, path)

    value = _eval_expr(rhs, symbols, lineno, path)

    if len(targets) == 1:
        values = [value]
    else:
        if not isinstance(value, tuple) or len(value) != len(targets):
            raise GladeSyntaxError(
                f"cannot unpack {len(targets)} targets from right-hand side",
                lineno, path)
        values = list(value)

    out: list[Assignment] = []
    for name, val in zip(targets, values):
        if _contains_expr(val):
            # obs-position / arithmetic expressions are resolved per-component at
            # merge time; a scalar assignment (including a list element) has no
            # such context. Caught here so it never slips past validation into a
            # raw float() crash at fit time.
            raise GladeSyntaxError(
                f"'{name}': expressions that reference observations "
                f"(img1_x, obs_positions_mas_list[...], ...) are only allowed "
                f"inside component (...) tuples, not scalar assignments",
                lineno, path)
        if isinstance(val, Ref):
            # the parser is file-local: the name may simply not be a numeric
            # scalar defined ABOVE in this file (e.g. a {lo, hi} variable,
            # which can only be referenced from component tuples).
            raise GladeSyntaxError(
                f"'{name}' references '{val.name}', which is not a previously "
                f"defined numeric scalar in this file; a {{lo, hi}} variable "
                f"can only be referenced from component tuples", lineno, path)
        # feed numeric scalars into the symbol table for later references
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            symbols[name] = float(val)
        out.append(Assignment(name=name, value=val, source_file=path, lineno=lineno))
    return out


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def parse_text(text: str, path: str | None = None) -> ParsedFile:
    """Parse one ``.dat`` document into a :class:`ParsedFile`.

    Raises :class:`GladeSyntaxError` on malformed syntax.
    """
    parsed = ParsedFile(path=path)
    symbols: dict[str, float] = parsed.symbols
    seen_scalars: set[str] = set()

    for stmt, lineno in _split_statements(text):
        m = _COMPONENT_RE.match(stmt)
        if m:
            comp = _parse_component(
                m.group("name"), m.group("rest"), lineno, path, symbols)
            parsed.components.append(comp)
            continue

        eq = _find_top_assign(stmt)
        if eq == -1:
            raise GladeSyntaxError(
                f"statement is neither an assignment nor a component: {stmt!r}",
                lineno, path)

        for assign in _parse_assignment(stmt, eq, lineno, path, symbols):
            key = schema.SCALAR_ALIASES.get(assign.name, assign.name)
            if key in seen_scalars:
                raise GladeSyntaxError(
                    f"'{assign.name}' is assigned more than once in this file",
                    lineno, path)
            seen_scalars.add(key)
            parsed.assignments.append(assign)

    return parsed


def parse_file(path: str) -> ParsedFile:
    import os
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    return parse_text(text, path=os.path.basename(path))
