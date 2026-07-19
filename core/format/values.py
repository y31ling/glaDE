"""Value model for parsed GLADE ``.dat`` configurations.

A parsed parameter is one of:

* :class:`Fixed`      -- a locked numeric value.
* :class:`Bounds`     -- an optimizable ``{lo, hi}`` search dimension.
* :class:`Unfilled`   -- a ``$float`` / ``$int`` placeholder still left in the file.

Scalars may additionally be plain Python objects (``list``, ``bool``, ``str``)
for things like the observation arrays and flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass(frozen=True)
class Fixed:
    """A locked numeric parameter."""

    value: float

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return repr(self.value)


@dataclass(frozen=True)
class Bounds:
    """An optimizable parameter with inclusive search bounds ``[lo, hi]``."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        # bounds may be written hi-first; normalise to lo <= hi.
        if self.hi < self.lo:
            lo, hi = self.hi, self.lo
            object.__setattr__(self, "lo", lo)
            object.__setattr__(self, "hi", hi)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return "{%r, %r}" % (self.lo, self.hi)


@dataclass(frozen=True)
class SharedBounds(Bounds):
    """A reference to a named user-defined ``{lo, hi}`` variable.

    ``my_var = {-0.1, 0.1}`` followed by component parameters written as
    ``my_var`` resolves each reference to one ``SharedBounds``: every
    referencing parameter SHARES a single search dimension (the fitted value
    is common to all of them). Subclasses :class:`Bounds` so every
    "is this optimizable" check keeps working unchanged; consumers that build
    per-parameter dimensions must check ``SharedBounds`` FIRST.
    """

    name: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name}{{%r, %r}}" % (self.lo, self.hi)


@dataclass(frozen=True)
class Unfilled:
    """An unresolved template placeholder (``$float`` / ``$int`` ...)."""

    kind: str = "float"          # 'float' | 'int' | 'str'
    optimizable: bool = False    # was it $type{lower,upper} ?

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return "${}{}".format(self.kind, "{lower,upper}" if self.optimizable else "")


@dataclass(frozen=True)
class Ref:
    """A reference to a named scalar (e.g. ``lens_z``) resolved after merge.

    References are kept symbolic until the full (possibly multi-file) symbol
    table is known, then replaced by a :class:`Fixed` value.
    """

    name: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


@dataclass(frozen=True)
class Expr:
    """A deferred arithmetic expression, resolved after merge.

    Captures component-parameter source such as ``img1_x - 0.075`` or
    ``{img1_x-0.075, img1_x+0.075}`` that references observation positions /
    arrays (and applies the engine-frame unit + center-offset transform) which
    are only known once every file is merged. ``core.format.config.merge``
    evaluates it (see :mod:`core.format.expr`) into a :class:`Fixed` (scalar
    ``code``) or :class:`Bounds` (``{lo, hi}`` ``code``, ``is_bounds=True``).
    """

    code: str                    # canonical source text (ast.unparse)
    is_bounds: bool = False      # True when code is a {lo, hi} set

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.code


# A component / source parameter (SharedBounds is a Bounds subclass).
ParamValue = Union[Fixed, Bounds, Unfilled, Ref, Expr]


@dataclass
class Component:
    """One lens or sub-structure component (both share a single stack)."""

    name: str                         # 'sers1'
    type: str                         # glade model keyword, e.g. 'sers'
    z: ParamValue                     # redshift (usually Fixed after resolution)
    params: list[ParamValue]          # p1..pk in glafic order
    category: str = "lens"            # 'lens' | 'substructure' (authoring hint only)
    raw_index: Optional[int] = None   # the literal N as written (advisory)
    # 'lens' | 'substructure' from an index suffix ('3l' / '3s'); None = use the
    # model's default classification (schema category / optimizability).
    category_override: Optional[str] = None
    index: Optional[int] = None       # globally recomputed 1-based index
    source_file: Optional[str] = None
    lineno: Optional[int] = None
    # Per-parameter unit factors for SHARED-variable references under a
    # non-default UnitSetting profile (None = all 1.0). A shared {lo, hi}
    # variable is dimensionless — the insertion slot decides the unit — so
    # the factor is applied at scene-injection time while literal values
    # were already converted at merge time. Aligned with ``params``.
    unit_scales: Optional[list] = None

    def is_optimizable(self) -> bool:
        return any(isinstance(p, Bounds) for p in self.params) or isinstance(self.z, Bounds)


@dataclass
class Assignment:
    """A single ``name = value`` scalar assignment, preserving order/origin."""

    name: str
    value: Any                        # Fixed | Bounds | Unfilled | list | bool | str
    source_file: Optional[str] = None
    lineno: Optional[int] = None


@dataclass
class ParsedFile:
    """The result of parsing one ``.dat`` file (order preserved)."""

    path: Optional[str] = None
    assignments: list[Assignment] = field(default_factory=list)
    components: list[Component] = field(default_factory=list)
    # name -> resolved numeric value, for reference substitution within the file
    symbols: dict[str, float] = field(default_factory=dict)

    def scalar(self, name: str) -> Any:
        for a in self.assignments:
            if a.name == name:
                return a.value
        return None
