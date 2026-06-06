"""Read and write glafic input / observation files.

Canonical glafic2 grammar (from ``glafic2/init.c``):

    omega <v> / lambda <v> / weos <v> / hubble <v> / prefix <s>
    xmin <v> ... maxlev <v>
    startup <nlens> <next> <npoint>
    lens  <type> <z> <p1> ... <p7>      # exactly 8 numbers after the type
    point <zs> <xs> <ys>
    start_command ... quit

We also tolerate the older ``lens <type> <id> <z> <p1..p7>`` variant (9 numbers
after the type) by dropping the leading id. Optimization flags are read from an
optional ``start_setopt ... end_setopt`` block (one row of 0/1 flags per lens,
then one for the point), and observations from ``start_obs ... end_obs`` /
``readobs_point`` files using glafic's column layout
``x y mag sigma_pos sigma_mag 0 0 flag`` (positions in arcsec).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

NPAR = 7  # p1..p7

PRIMARY_FLOAT_KEYS = (
    "omega", "lambda", "weos", "hubble",
    "xmin", "ymin", "xmax", "ymax", "pix_ext", "pix_poi",
)
PRIMARY_INT_KEYS = ("maxlev",)


@dataclass
class GlaficLens:
    type: str
    z: float
    params: list[float]              # length 7 (p1..p7)
    opt: Optional[list[int]] = None  # length 8 flags (z + p1..p7), 1 = optimize


@dataclass
class GlaficObs:
    zs: float
    # each image: (x_arcsec, y_arcsec, mag, sigma_pos_arcsec, sigma_mag, flag)
    images: list[tuple] = field(default_factory=list)


@dataclass
class GlaficModel:
    primary: dict = field(default_factory=dict)   # keys use 'lambda' (glafic spelling)
    prefix: str = "out"
    lenses: list[GlaficLens] = field(default_factory=list)
    source_z: Optional[float] = None
    source_x: Optional[float] = None
    source_y: Optional[float] = None
    point_opt: Optional[list[int]] = None          # 3 flags (zs, xs, ys)
    obs: Optional[GlaficObs] = None


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def _tokens(line: str) -> list[str]:
    return line.split()


def parse_glafic_input(text: str) -> GlaficModel:
    model = GlaficModel()
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        line = raw.split("#")[0].strip()
        i += 1
        if not line:
            continue
        tok = _tokens(line)
        key = tok[0]

        if key in PRIMARY_FLOAT_KEYS and len(tok) >= 2:
            model.primary[key] = float(tok[1])
        elif key in PRIMARY_INT_KEYS and len(tok) >= 2:
            model.primary[key] = int(float(tok[1]))
        elif key == "prefix" and len(tok) >= 2:
            model.prefix = tok[1]
        elif key == "startup":
            pass  # counts recomputed from the actual lines
        elif key == "lens":
            model.lenses.append(_parse_lens(tok))
        elif key == "point" and len(tok) >= 4:
            model.source_z = float(tok[1])
            model.source_x = float(tok[2])
            model.source_y = float(tok[3])
        elif key == "start_setopt":
            i = _parse_setopt(lines, i, model)
        elif key in ("start_obs", "startobs"):
            i = _parse_obs(lines, i, model)
        # any other keyword (commands, etc.) is ignored
    return model


def _parse_lens(tok: list[str]) -> GlaficLens:
    ltype = tok[1]
    nums = [float(t) for t in tok[2:]]
    # canonical: z + 7 params = 8 numbers. older variant: id + z + 7 = 9.
    if len(nums) == NPAR + 2:
        nums = nums[1:]  # drop leading id
    z = nums[0] if nums else 0.0
    params = (nums[1:] + [0.0] * NPAR)[:NPAR]
    return GlaficLens(type=ltype, z=z, params=params)


def _parse_setopt(lines: list[str], i: int, model: GlaficModel) -> int:
    """Read flag rows until end_setopt; assign to lenses in order, then point."""
    rows: list[list[int]] = []
    n = len(lines)
    while i < n:
        line = lines[i].split("#")[0].strip()
        i += 1
        if not line:
            continue
        if line.startswith("end_setopt"):
            break
        # rows may contain '[lower,upper]' placeholders (glade-authored) or ints
        flags = [0 if t.startswith("[") else int(float(t))
                 for t in line.replace(",", " ").split()
                 if t not in ("", "]")]
        rows.append(flags)
    for k, lens in enumerate(model.lenses):
        if k < len(rows):
            lens.opt = (rows[k] + [0] * (NPAR + 1))[: NPAR + 1]
    if len(rows) > len(model.lenses):
        model.point_opt = (rows[len(model.lenses)] + [0, 0, 0])[:3]
    return i


def _parse_obs(lines: list[str], i: int, model: GlaficModel) -> int:
    n = len(lines)
    header = None
    images: list[tuple] = []
    while i < n:
        line = lines[i].split("#")[0].strip()
        i += 1
        if not line:
            continue
        if line.startswith("end_obs"):
            break
        vals = [float(t) for t in line.split()]
        if header is None:
            # header: <point_id> <n_images> <z_source> <something>
            header = vals
            continue
        # image columns: x y mag [sigma_pos] [sigma_mag] [..] [..] [flag]
        x = vals[0] if len(vals) > 0 else 0.0
        y = vals[1] if len(vals) > 1 else 0.0
        mag = vals[2] if len(vals) > 2 else 0.0
        spos = vals[3] if len(vals) > 3 else 0.0
        smag = vals[4] if len(vals) > 4 else 0.0
        flag = vals[-1] if len(vals) > 5 else 0.0
        images.append((x, y, mag, spos, smag, flag))
    zs = header[2] if header and len(header) > 2 else (model.source_z or 0.0)
    model.obs = GlaficObs(zs=zs, images=images)
    return i


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def _fmt(v: float) -> str:
    return f"{v:.6e}"


def render_glafic_input(model: GlaficModel, command: bool = True) -> str:
    out: list[str] = ["## glafic input file generated by GLADE translate", ""]
    p = model.primary
    for k in ("omega", "lambda", "weos", "hubble"):
        if k in p:
            out.append(f"{k:<10}{p[k]}")
    out.append("")
    out.append(f"prefix     {model.prefix}")
    out.append("")
    for k in ("xmin", "ymin", "xmax", "ymax", "pix_ext", "pix_poi", "maxlev"):
        if k in p:
            out.append(f"{k:<10}{p[k]}")
    out.append("")

    npoint = 1 if model.source_z is not None else 0
    out.append(f"startup {len(model.lenses)} 0 {npoint}")
    for lens in model.lenses:
        nums = "  ".join(_fmt(v) for v in [lens.z, *lens.params])
        out.append(f"lens   {lens.type:<6} {nums}")
    if model.source_z is not None:
        out.append("")
        out.append(f"point  {_fmt(model.source_z)}  {_fmt(model.source_x)}  "
                   f"{_fmt(model.source_y)}")
    out.append("")

    if command:
        out += ["start_command", "", "findimg", "",
                f"writecrit  {_fmt(model.source_z) if model.source_z is not None else '0.0'}",
                "", "quit", ""]
    return "\n".join(out) + "\n"


def render_glafic_obs(obs: GlaficObs) -> str:
    out = ["## observation file generated by GLADE translate",
           "start_obs",
           f"1 {len(obs.images)} {obs.zs:.4f} 0.0"]
    for (x, y, mag, spos, smag, flag) in obs.images:
        out.append(f"  {x:12.6f} {y:12.6f} {mag:12.5f} {spos:10.6f} "
                   f"{smag:8.5f} 0.000000 0.000000 {int(flag)}")
    out.append("end_obs")
    return "\n".join(out) + "\n"
