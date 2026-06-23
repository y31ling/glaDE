"""Evaluate deferred ``.dat`` expressions (arithmetic + observation references).

A component parameter may be written as an arithmetic expression that references
observed image positions, e.g.::

    'king1': (4, 'king', lens_z, {1e2, 1e8},
              {img1_x-0.075, img1_x+0.075},
              {obs_positions_mas_list[0][1]-0.075, obs_positions_mas_list[0][1]+0.075},
              ...)

These cannot be folded at file-local parse time (they need the observation
arrays + ``center_offset`` + ``obs_x_flip`` from the merged config), so the
parser keeps them as :class:`core.format.values.Expr` and merge resolves them
here.

Recognised references (all others are an error):

* ``imgN_x`` / ``imgN_y``                -- the N-th observed image (1-based)
* ``obs_positions_mas_list[i][j]``       -- the same image, 0-based; j=0 -> x, 1 -> y
* ``obs_*_list[...]``                    -- a raw element of any observation array
* any numeric scalar already defined (``lens_z`` ...)

``imgN_x`` and ``obs_positions_mas_list[i][0]`` are NOT the raw mas value: they
are converted to the engine (glafic lens) frame exactly as
``core.optimize.scene.build_obs`` does, so a lens placed there sits under the
observed image::

    x = x_sign * (mas_x / 1000 - center_offset_x)        # x_sign = -1 if obs_x_flip
    y =          (mas_y / 1000) - center_offset_y

i.e. milliarcsec -> arcsec, the ``obs_x_flip`` sign on x, and the center offset
applied. Arithmetic constants (``-0.075``) are therefore in arcsec, engine frame.
"""
from __future__ import annotations

import ast
import re
from typing import Any

from .defaults import DEFAULTS

_IMG_ALIAS_RE = re.compile(r"^img(\d+)_([xy])$")
_POS_KEY = "obs_positions_mas_list"


class ExprError(ValueError):
    """A deferred expression could not be resolved."""


def build_env(obs: dict, scalars: dict) -> dict:
    """Assemble the evaluation environment from a merged config's obs + scalars.

    ``obs`` is the merged ``cfg.obs`` section; ``scalars`` the numeric symbol
    table. Expressions are resolved during merge, BEFORE ``apply_defaults`` runs,
    so a ``.dat`` that omits ``center_offset`` / ``obs_x_flip`` must fall back to
    the SAME values :data:`core.format.defaults.DEFAULTS` will later fill in (and
    that the optimizer's :func:`core.optimize.scene.build_obs` then reads) -- not
    to 0 / False -- or the lens would be frozen in a different engine frame than
    the fit uses.
    """
    def _f(key):
        v = obs.get(key, DEFAULTS.get(key, 0.0))
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0

    flip = obs.get("obs_x_flip", DEFAULTS.get("obs_x_flip", False))
    return {
        "scalars": scalars,
        "positions": obs.get(_POS_KEY),
        "x_sign": -1.0 if flip else 1.0,
        "cox": _f("center_offset_x"),
        "coy": _f("center_offset_y"),
        "arrays": {k: v for k, v in obs.items() if isinstance(v, list)},
    }


def _num(v: Any, ctx: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ExprError(f"{ctx} is not a number")
    return float(v)


def _pos_coord(env: dict, i: int, j: int) -> float:
    pos = env["positions"]
    if not pos:
        raise ExprError(f"{_POS_KEY} is referenced but no observations are defined")
    if i < 0 or i >= len(pos):
        raise ExprError(f"{_POS_KEY}[{i}] out of range ({len(pos)} image(s))")
    row = pos[i]
    if not isinstance(row, (list, tuple)) or len(row) < 2:
        raise ExprError(f"{_POS_KEY}[{i}] is not an (x, y) pair")
    if j == 0:
        return env["x_sign"] * (_num(row[0], f"{_POS_KEY}[{i}][0]") / 1000.0 - env["cox"])
    if j == 1:
        return _num(row[1], f"{_POS_KEY}[{i}][1]") / 1000.0 - env["coy"]
    raise ExprError(f"{_POS_KEY}[{i}][{j}]: column must be 0 (x) or 1 (y)")


def _img_coord(env: dict, n: int, axis: str) -> float:
    pos = env["positions"]
    if not pos:
        raise ExprError(f"img{n}_{axis} used but no obs_positions_mas_list is defined")
    if n < 1 or n > len(pos):
        raise ExprError(f"img{n}_{axis}: only {len(pos)} observed image(s)")
    return _pos_coord(env, n - 1, 0 if axis == "x" else 1)


def _flatten_subscript(node: ast.Subscript):
    indices = []
    cur: ast.AST = node
    while isinstance(cur, ast.Subscript):
        indices.append(cur.slice)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        raise ExprError("only simple name[...] subscripts are allowed")
    indices.reverse()
    return cur.id, indices


def _eval(node: ast.AST, env: dict) -> float:
    if isinstance(node, ast.Constant):
        return _num(node.value, "constant")

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _num(_eval(node.operand, env), "operand")
        return -v if isinstance(node.op, ast.USub) else v

    if isinstance(node, ast.BinOp):
        lo = _num(_eval(node.left, env), "operand")
        hi = _num(_eval(node.right, env), "operand")
        op = node.op
        if isinstance(op, ast.Add):
            return lo + hi
        if isinstance(op, ast.Sub):
            return lo - hi
        if isinstance(op, ast.Mult):
            return lo * hi
        if isinstance(op, ast.Div):
            if hi == 0:
                raise ExprError("division by zero")
            return lo / hi
        if isinstance(op, ast.Pow):
            try:
                return lo ** hi
            except (ArithmeticError, ValueError) as exc:
                raise ExprError(f"invalid '**' ({exc})") from None
        raise ExprError(f"unsupported operator {type(op).__name__} "
                        "(only + - * / ** are allowed)")

    if isinstance(node, ast.Name):
        m = _IMG_ALIAS_RE.match(node.id)
        if m:
            return _img_coord(env, int(m.group(1)), m.group(2))
        if node.id in env["scalars"]:
            return float(env["scalars"][node.id])
        raise ExprError(f"unknown name '{node.id}' in expression")

    if isinstance(node, ast.Subscript):
        base, slices = _flatten_subscript(node)
        idx = [int(_num(_eval(s, env), "index")) for s in slices]
        if base == _POS_KEY and len(idx) == 2:
            return _pos_coord(env, idx[0], idx[1])
        if base == _POS_KEY:
            raise ExprError(f"{_POS_KEY} must be indexed as [image][0|1]")
        if base in env["arrays"]:
            val: Any = env["arrays"][base]
            try:
                for k in idx:
                    val = val[k]
            except (IndexError, TypeError, KeyError):
                raise ExprError(f"{base}{idx} index out of range") from None
            return _num(val, f"{base}{idx}")
        raise ExprError(f"unknown array '{base}' in expression")

    raise ExprError(f"unsupported expression element ({type(node).__name__})")


def evaluate(code: str, env: dict):
    """Evaluate *code* against *env*.

    Returns ``("scalar", value)`` for an arithmetic expression, or
    ``("bounds", lo, hi)`` for a ``{lo, hi}`` set. Raises :class:`ExprError`.
    """
    try:
        body = ast.parse(code, mode="eval").body
    except SyntaxError as exc:
        raise ExprError(f"invalid expression syntax: {exc.msg}") from None
    if isinstance(body, ast.Set):
        if len(body.elts) != 2:
            raise ExprError("bounds {lo, hi} must have exactly 2 values")
        lo = _num(_eval(body.elts[0], env), "lower bound")
        hi = _num(_eval(body.elts[1], env), "upper bound")
        return ("bounds", lo, hi)
    return ("scalar", _num(_eval(body, env), "expression"))
