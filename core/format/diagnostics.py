"""Diagnostics shared across the format package."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

ERROR = "error"
WARNING = "warning"


@dataclass
class Issue:
    severity: str            # ERROR | WARNING
    code: str                # short machine code, e.g. 'unfilled', 'gpu_unsupported'
    message: str             # human-readable
    source_file: Optional[str] = None
    lineno: Optional[int] = None

    @property
    def is_error(self) -> bool:
        return self.severity == ERROR

    def __str__(self) -> str:
        loc = ""
        if self.source_file:
            loc = self.source_file
            if self.lineno:
                loc += f":{self.lineno}"
            loc = f"{loc}: "
        return f"[{self.severity}] {loc}{self.message}"


class GladeSyntaxError(Exception):
    """Raised for unrecoverable syntax problems while parsing a ``.dat`` file."""

    def __init__(self, message: str, lineno: Optional[int] = None,
                 source_file: Optional[str] = None):
        self.message = message
        self.lineno = lineno
        self.source_file = source_file
        loc = ""
        if source_file:
            loc = source_file
            if lineno:
                loc += f":{lineno}"
            loc += ": "
        super().__init__(f"{loc}{message}")

    def as_issue(self) -> Issue:
        return Issue(ERROR, "syntax", self.message, self.source_file, self.lineno)
