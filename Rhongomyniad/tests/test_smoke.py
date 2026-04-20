"""
Basic smoke test: can we import, initialise, and run findimg without errors?

Uses the example in glade/glafic2/samples/point.input (SIE + external shear).
This test is *self-contained* and does not call glafic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rhongomyniad as rh


def main() -> int:
    rh.init(omega=0.3, lam=0.7, weos=-1.0, hubble=0.7, prefix="smoke",
            xmin=-5.0, ymin=-5.0, xmax=5.0, ymax=5.0,
            pix_ext=0.02, pix_poi=0.5, maxlev=5, verb=1)

    rh.startup_setnum(2, 0, 1)

    # Lens 1: SIE at (0, 0), sigma=300 km/s, e=0.35, pa=0°
    rh.set_lens(1, "sie",   0.5, 300.0, 0.0, 0.0, 0.35,  0.0, 0.0, 0.0)
    # Lens 2: external shear at same plane, g=0.05, pa=60°, referenced to zs_fid=2
    rh.set_lens(2, "pert",  0.5,   2.0, 0.0, 0.0, 0.05, 60.0, 0.0, 0.0)
    rh.set_point(1, 2.0, -0.15, 0.05)

    rh.model_init(verb=1)

    # Sanity: cosmology-derived values at the central pixel.
    pout = rh.calcimage(2.0, 0.0, 0.0)
    print(f"calcimage(0,0) = {pout}")
    assert all(isinstance(v, float) for v in pout), pout

    # Image search.
    images = rh.point_solve(2.0, -0.15, 0.05, verb=1)
    print(f"found {len(images)} images")
    assert len(images) > 0, "expected at least 1 image for this config"
    return 0


if __name__ == "__main__":
    sys.exit(main())
