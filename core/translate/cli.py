"""Command-line interface for glafic <-> glade translation.

    # glafic input -> glade .dat files (model + obs split into two files)
    python -m core.translate.cli to-glade SN_model.input -o out/

    # glade .dat files -> glafic input + obs file
    python -m core.translate.cli to-glafic constants.dat lens.dat obs.dat -o out/run
"""
from __future__ import annotations

import argparse
import os
import sys

from ..format.config import apply_defaults, merge
from ..format.parser import parse_file
from .convert import glade_to_glafic, glafic_to_glade


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  wrote {path}")


def _cmd_to_glade(args) -> int:
    with open(args.input, "r", encoding="utf-8") as fh:
        text = fh.read()
    result = glafic_to_glade(text)
    base = os.path.splitext(os.path.basename(args.input))[0]
    outdir = args.out or "."
    _write(os.path.join(outdir, f"{base}_model.dat"), result["model"])
    if result["obs"]:
        _write(os.path.join(outdir, f"{base}_obs.dat"), result["obs"])
    else:
        print("  (no observation block found in the glafic input)")
    return 0


def _cmd_to_glafic(args) -> int:
    parsed = [parse_file(p) for p in args.inputs]
    cfg, issues = merge(parsed)
    apply_defaults(cfg)
    for i in issues:
        if i.is_error:
            print(f"  warning (continuing): {i}", file=sys.stderr)
    out = args.out or "glade_export"
    # base_name ties the in-model readobs_point / parprior references to the
    # files written below (they are `<out>_obs.dat` / `<out>_prior.dat`).
    base = os.path.basename(out)
    result = glade_to_glafic(cfg, base_name=base)
    _write(f"{out}_model.input", result["model"])
    if result.get("optimize"):
        # optimize-ready: write the readobs_point constraint + parprior ranges
        # the model references (glafic cannot read the start_obs round-trip form).
        if result.get("constraint"):
            _write(f"{out}_obs.dat", result["constraint"])
        if result.get("prior"):
            _write(f"{out}_prior.dat", result["prior"])
        print("  (added a glafic `optimize` block — {lo,hi} parameters found)")
    elif result.get("obs"):
        _write(f"{out}_obs.dat", result["obs"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="core.translate.cli",
                                 description="Translate between glafic and glade")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g1 = sub.add_parser("to-glade", help="glafic input -> glade .dat")
    g1.add_argument("input")
    g1.add_argument("-o", "--out", default=".", help="output directory")
    g1.set_defaults(func=_cmd_to_glade)

    g2 = sub.add_parser("to-glafic", help="glade .dat(s) -> glafic input")
    g2.add_argument("inputs", nargs="+")
    g2.add_argument("-o", "--out", default="glade_export",
                    help="output path prefix")
    g2.set_defaults(func=_cmd_to_glafic)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
