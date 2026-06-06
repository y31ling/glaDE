"""Parse glafic critical-curve / caustic files (``*_crit.dat``).

glafic writes 8 columns per row; the critical-curve segment is
``(col0, col1) -> (col4, col5)`` and the matching caustic segment is
``(col2, col3) -> (col6, col7)``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def read_critical_curves(crit_file: str):
    """Return ``(crit_segments, caus_segments)``.

    Each is a list of ``[[x1, y1], [x2, y2]]`` segments. An empty/missing file
    yields two empty lists rather than raising.
    """
    try:
        data = np.loadtxt(crit_file)
    except (OSError, ValueError):
        return [], []
    if data.size == 0:
        return [], []
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 8:
        return [], []

    crit_segments = [[[row[0], row[1]], [row[4], row[5]]] for row in data]
    caus_segments = [[[row[2], row[3]], [row[6], row[7]]] for row in data]
    return crit_segments, caus_segments
