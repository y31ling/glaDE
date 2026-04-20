"""
Dual-mode comparison harness.

Run with `--mode glafic`  (inside WSL Ubuntu Python) to produce a reference
JSON file using the real glafic.so Python bindings.  Run with `--mode rh`
(on Windows or WSL) to compute the same quantities with Rhongomyniad and
diff them against the reference.

Test scenarios are defined once in `SCENARIOS`.  Each exercises a lens
model at varying query positions, and records:
  • `calcimage` output (ax, ay, td, kap, g1, g2, muinv, rot) at a list
    of image-plane probe points
  • `point_solve` output at a list of source positions

Usage
-----
Produce reference:
    wsl -d Ubuntu -- bash -c "cd /path/Rhongomyniad && \\
      python3 tests/compare_glafic.py --mode glafic --out reference.json"

Run comparison:
    python tests/compare_glafic.py --mode rh --ref reference.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# --- Test scenarios ---------------------------------------------------
# Each scenario describes:
#   lenses = list of (model_name, 8-tuple params in glafic ordering)
#   primary = (omega, lam, weos, hubble, xmin, ymin, xmax, ymax,
#              pix_ext, pix_poi, maxlev)
#   probes  = list of (zs, x, y) image-plane probe points (for calcimage)
#   sources = list of (zs, xs, ys) source positions (for point_solve)
#
# Feel free to add more.

SCENARIOS = [
    {
        "name": "sie_pert",
        "primary": (0.3, 0.7, -1.0, 0.7,
                    -5.0, -5.0, 5.0, 5.0,
                    0.02, 0.5, 5),
        "lenses": [
            ("sie",   (0.5, 300.0, 0.0, 0.0, 0.35,  0.0, 0.0, 0.0)),
            ("pert",  (0.5,   2.0, 0.0, 0.0, 0.05, 60.0, 0.0, 0.0)),
        ],
        "probes":  [(2.0,  0.5,  0.3),
                    (2.0, -0.8, -0.1),
                    (2.0,  1.5,  1.2),
                    (2.0, -2.0,  0.5)],
        "sources": [(2.0, -0.15, 0.05),
                    (2.0,  0.02,  0.07),
                    (2.0,  0.10, -0.05)],
    },
    {
        "name": "pointmass",
        "primary": (0.3, 0.7, -1.0, 0.7,
                    -3.0, -3.0, 3.0, 3.0,
                    0.02, 0.2, 5),
        "lenses": [
            ("point", (0.5, 1.0e12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ],
        "probes":  [(2.0,  0.5,  0.0),
                    (2.0,  0.0,  0.5),
                    (2.0,  0.3,  0.4),
                    (2.0, -0.7, -0.2)],
        "sources": [(2.0,  0.02,  0.0),
                    (2.0, -0.05,  0.03),
                    (2.0,  0.10,  0.02)],
    },
    {
        "name": "nfwpot",
        "primary": (0.3, 0.7, -1.0, 0.7,
                    -30.0, -30.0, 30.0, 30.0,
                    0.1, 1.0, 5),
        "lenses": [
            ("nfwpot", (0.3, 1.0e14, 0.0, 0.0, 0.2, 30.0, 5.0, 0.0)),
        ],
        "probes":  [(1.5,  5.0,  3.0),
                    (1.5, -4.0,  1.0),
                    (1.5,  0.0, -6.0),
                    (1.5, 10.0,  0.0)],
        "sources": [(1.5,  0.02,  0.03),
                    (1.5, -0.05,  0.00),
                    (1.5,  0.20, -0.10)],
    },
    {
        "name": "nfw",
        "primary": (0.3, 0.7, -1.0, 0.7,
                    -30.0, -30.0, 30.0, 30.0,
                    0.1, 1.0, 5),
        "lenses": [
            ("nfw", (0.3, 1.0e14, 0.0, 0.0, 0.2, 30.0, 5.0, 0.0)),
        ],
        "probes":  [(1.5,  5.0,  3.0),
                    (1.5, -4.0,  1.0),
                    (1.5,  0.0, -6.0),
                    (1.5, 10.0,  0.0)],
        "sources": [(1.5,  0.02,  0.03),
                    (1.5, -0.05,  0.00),
                    (1.5,  0.20, -0.10)],
    },
    {
        "name": "king",
        "primary": (0.3, 0.7, -1.0, 0.7,
                    -10.0, -10.0, 10.0, 10.0,
                    0.05, 0.4, 5),
        "lenses": [
            # king: p[1]=M, p[4]=e, p[5]=pa, p[6]=rc, p[7]=c (=log10(rt/rc))
            ("king", (0.4, 3.0e11, 0.0, 0.0, 0.15, 15.0, 1.5, 1.0)),
        ],
        "probes":  [(1.5,  2.0,  1.0),
                    (1.5, -1.5,  0.5),
                    (1.5,  0.0, -2.5),
                    (1.5,  3.0, -1.5)],
        "sources": [(1.5, 0.05, 0.10),
                    (1.5, -0.10, 0.02),
                    (1.5, 0.00, -0.15)],
    },
    {
        "name": "jaffe",
        "primary": (0.3, 0.7, -1.0, 0.7,
                    -5.0, -5.0, 5.0, 5.0,
                    0.02, 0.3, 5),
        "lenses": [
            # jaffe: p[1]=sigma_v, p[4]=e, p[5]=pa, p[6]=a (outer), p[7]=rco (inner)
            ("jaffe", (0.4, 250.0, 0.0, 0.0, 0.25, -10.0, 2.0, 0.1)),
        ],
        "probes":  [(1.5,  0.5,  0.3),
                    (1.5, -0.8,  0.2),
                    (1.5,  0.0, -1.2),
                    (1.5,  1.5, -0.5)],
        "sources": [(1.5,  0.01,  0.02),
                    (1.5, -0.04,  0.00),
                    (1.5,  0.08, -0.03)],
    },
]


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------
def run_glafic(scenario: dict) -> dict:
    """Produce calcimage + point_solve outputs using the real glafic.so."""
    import glafic
    omega, lam, weos, h, xmin, ymin, xmax, ymax, pe, pp, ml = scenario["primary"]
    glafic.init(omega, lam, weos, h, "out",
                xmin, ymin, xmax, ymax, pe, pp, ml, 0, 0)
    nlens = len(scenario["lenses"])
    nsrc = len(scenario["sources"])
    glafic.startup_setnum(nlens, 0, nsrc)
    for i, (model, p) in enumerate(scenario["lenses"], 1):
        glafic.set_lens(i, model, *p)
    for i, (zs, xs, ys) in enumerate(scenario["sources"], 1):
        glafic.set_point(i, zs, xs, ys)
    glafic.model_init(verb=0)

    probes = []
    for zs, x, y in scenario["probes"]:
        pout = glafic.calcimage(zs, x, y, -1, 0)
        probes.append({"zs": zs, "x": x, "y": y, "pout": list(pout)})

    solves = []
    for zs, xs, ys in scenario["sources"]:
        imgs = glafic.point_solve(zs, xs, ys, 0)
        solves.append({
            "zs": zs, "xs": xs, "ys": ys,
            "images": [list(im) for im in imgs],
        })
    glafic.quit()
    return {"calcimage": probes, "point_solve": solves}


def run_rhongomyniad(scenario: dict) -> dict:
    """Produce same outputs via Rhongomyniad."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import rhongomyniad as rh

    omega, lam, weos, h, xmin, ymin, xmax, ymax, pe, pp, ml = scenario["primary"]
    rh.init(omega, lam, weos, h, "out",
            xmin, ymin, xmax, ymax, pe, pp, ml, verb=0)
    nlens = len(scenario["lenses"])
    nsrc = len(scenario["sources"])
    rh.startup_setnum(nlens, 0, nsrc)
    for i, (model, p) in enumerate(scenario["lenses"], 1):
        rh.set_lens(i, model, *p)
    for i, (zs, xs, ys) in enumerate(scenario["sources"], 1):
        rh.set_point(i, zs, xs, ys)
    rh.model_init(verb=0)

    probes = []
    for zs, x, y in scenario["probes"]:
        pout = rh.calcimage(zs, x, y)
        probes.append({"zs": zs, "x": x, "y": y, "pout": list(pout)})

    solves = []
    for zs, xs, ys in scenario["sources"]:
        imgs = rh.point_solve(zs, xs, ys)
        solves.append({
            "zs": zs, "xs": xs, "ys": ys,
            "images": [list(im) for im in imgs],
        })
    return {"calcimage": probes, "point_solve": solves}


# ----------------------------------------------------------------------
# Comparison reporting
# ----------------------------------------------------------------------
def _pair_images(ref_images, test_images):
    """Greedy nearest-neighbour pairing by (x, y)."""
    pairs = []
    used = set()
    for r in ref_images:
        best = None
        best_d = float("inf")
        for j, t in enumerate(test_images):
            if j in used:
                continue
            d = (r[0] - t[0]) ** 2 + (r[1] - t[1]) ** 2
            if d < best_d:
                best_d = d; best = j
        if best is not None:
            used.add(best)
            pairs.append((r, test_images[best], math.sqrt(best_d)))
        else:
            pairs.append((r, None, float("inf")))
    extras = [test_images[j] for j in range(len(test_images)) if j not in used]
    return pairs, extras


SCHRAMM_LIKE = {"nfw", "king"}     # models that match glafic only to its Romberg tolerance


def compare(ref: dict, test: dict) -> int:
    """
    Return number of failing assertions.

    Models using Schramm-(1990) elliptical-density integrals (nfw, king, gnfw,
    ein, ...) inherit glafic's Romberg tolerance of 5e-4 (glafic.h:370), so
    their kap/gamma/ax/ay reproducibility vs glafic is limited to ~5e-4.
    Rhongomyniad uses 128-point Gauss-Legendre, which is *more* accurate than
    glafic (verified against scipy quad epsrel=1e-12), so exact parity is not
    achievable for these models.
    """
    fails = 0
    tol_scalar = 1.0e-5       # default relative tolerance for calcimage
    tol_abs = 1.0e-7          # absolute tolerance for small values
    for scen_name, ref_s in ref.items():
        test_s = test.get(scen_name)
        if test_s is None:
            print(f"[FAIL] scenario '{scen_name}' missing in test output")
            fails += 1
            continue
        is_schramm = scen_name in SCHRAMM_LIKE
        scen_tol = 3.0e-3 if is_schramm else tol_scalar   # 3x glafic Romberg tolerance
        pos_tol = 1.0e-3 if is_schramm else 1.0e-5
        mag_tol = 1.0e-2 if is_schramm else 1.0e-4
        # calcimage
        print(f"\n=== scenario: {scen_name}"
              f"{'  [Schramm-tolerance 1e-3]' if is_schramm else ''} ===")
        print(f"-- calcimage --")
        for r, t in zip(ref_s["calcimage"], test_s["calcimage"]):
            zs, x, y = r["zs"], r["x"], r["y"]
            print(f"  probe (zs={zs}, x={x}, y={y})")
            labels = ["ax", "ay", "td", "kap", "g1", "g2", "muinv", "rot"]
            for lab, rv, tv in zip(labels, r["pout"], t["pout"]):
                absdiff = abs(rv - tv)
                if abs(rv) > 1e-10:
                    reldiff = absdiff / abs(rv)
                else:
                    reldiff = absdiff
                mark = "  "
                if reldiff > scen_tol and absdiff > tol_abs:
                    mark = "!!"; fails += 1
                print(f"    {mark} {lab:5s}: ref={rv:+.8e}  test={tv:+.8e}  "
                      f"abs={absdiff:.2e}  rel={reldiff:.2e}")
        # point_solve
        print(f"-- point_solve --")
        for r, t in zip(ref_s["point_solve"], test_s["point_solve"]):
            print(f"  source (zs={r['zs']}, xs={r['xs']}, ys={r['ys']})")
            pairs, extras = _pair_images(r["images"], t["images"])
            if len(r["images"]) != len(t["images"]):
                print(f"    !! image count mismatch: ref={len(r['images'])}, "
                      f"test={len(t['images'])}")
                fails += 1
            for rr, tt, dist in pairs:
                if tt is None:
                    print(f"    !! ref image at ({rr[0]:.4f}, {rr[1]:.4f}) has no test match")
                    fails += 1
                    continue
                x_d = abs(rr[0] - tt[0])
                y_d = abs(rr[1] - tt[1])
                m_d = abs(rr[2] - tt[2])
                t_d = abs(rr[3] - tt[3])
                mark = "  "
                if x_d > pos_tol or y_d > pos_tol:
                    mark = "!!"; fails += 1
                if abs(rr[2]) > 1e-5 and m_d / abs(rr[2]) > mag_tol:
                    mark = "!!"; fails += 1
                print(f"    {mark} ref=({rr[0]:+.4f},{rr[1]:+.4f}) mag={rr[2]:+.4f} td={rr[3]:.3f}"
                      f"   test=({tt[0]:+.4f},{tt[1]:+.4f}) mag={tt[2]:+.4f} td={tt[3]:.3f}"
                      f"   Δxy={max(x_d, y_d):.2e}")
            for e in extras:
                print(f"    !! extra test image: ({e[0]:.4f}, {e[1]:.4f}) mag={e[2]:+.4f}")
                fails += 1
    return fails


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["glafic", "rh", "diff"], required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON path (for --mode glafic/rh)")
    ap.add_argument("--ref", type=Path, default=None,
                    help="reference JSON path (for --mode rh / diff)")
    ap.add_argument("--test", type=Path, default=None,
                    help="test JSON path (for --mode diff)")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="restrict to these scenario names")
    args = ap.parse_args()

    scenarios = SCENARIOS
    if args.scenarios:
        scenarios = [s for s in SCENARIOS if s["name"] in args.scenarios]

    if args.mode in ("glafic", "rh"):
        runner = run_glafic if args.mode == "glafic" else run_rhongomyniad
        data = {}
        for scen in scenarios:
            print(f"[{args.mode}] running {scen['name']}...")
            data[scen["name"]] = runner(scen)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(data, indent=2))
            print(f"wrote {args.out}")
        if args.mode == "rh" and args.ref:
            ref = json.loads(args.ref.read_text())
            fails = compare(ref, data)
            print(f"\n== {fails} diffs ==")
            return 0 if fails == 0 else 1
        return 0

    if args.mode == "diff":
        if not args.ref or not args.test:
            ap.error("--mode diff requires --ref and --test")
        ref = json.loads(args.ref.read_text())
        test = json.loads(args.test.read_text())
        fails = compare(ref, test)
        print(f"\n== {fails} diffs ==")
        return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
