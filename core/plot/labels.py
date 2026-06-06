"""Per-model sub-halo marker label formatting.

The only true per-model variation in the legacy triptychs was the sub-halo label
(pointmass: mass; nfw: mass + c; king: mass + rc + c; p-jaffe: sigma + a + rco).
This maps a model type + its fitted params to a compact label string.
"""
from __future__ import annotations

from ..format import schema


def _sci(v: float) -> str:
    return f"{v:.1e}"


def subhalo_label(index: int, model_type: str, params) -> str:
    """Compact label for a sub-halo marker, e.g. ``S1: 1.0e+06`` (point mass) or
    ``S2: M=1.0e+09 rc=0.02 c=1.5`` (king)."""
    spec = schema.model(model_type)
    p = list(params)
    if spec is None or not p:
        return f"S{index}"

    # the mass-like parameter is the headline number
    head = ""
    if spec.mass_positions:
        mi = spec.mass_positions[0]
        if mi < len(p):
            head = _sci(p[mi])

    extras = []
    # show shape parameters beyond x, y (positions 1, 2) that aren't the mass
    for j, pspec in enumerate(spec.params):
        if j in spec.mass_positions:
            continue
        if pspec.name in ("x", "y", "e", "pa", "_unused"):
            continue
        if j < len(p):
            extras.append(f"{pspec.name}={p[j]:.3g}")

    label = f"S{index}: {head}".rstrip()
    if extras:
        label += " " + " ".join(extras)
    return label
