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
class GlaficExtend:
    """An extended-source component (glafic set_extend)."""
    type: str                        # glafic name: 'sersic' | 'gauss' | ...
    z: float
    params: list[float]              # length 7 (norm, x, y, e, pa, r0, n)
    opt: Optional[list[int]] = None  # length 8 flags (z + p1..p7)


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
    # extended-source additions (populated by the python-script parser)
    extends: list[GlaficExtend] = field(default_factory=list)
    secondary: dict = field(default_factory=dict)  # set_secondary key -> value
    extended_file: Optional[str] = None            # readobs_extend FITS
    mask_file: Optional[str] = None                # readobs_extend mask
    noise_file: Optional[str] = None               # readnoise_extend FITS
    constraint_file: Optional[str] = None          # readobs_point file
    prior_file: Optional[str] = None               # parprior file
    # --- amoeba (glafic `optimize`) emission, used by glade -> glafic export ---
    optimize: bool = False                         # emit `optimize` in start_command
    hvary: int = 0                                 # 1 => optimise the Hubble constant
    # parprior `range` rows: (kind, id, param_no, lo, hi); param_no 1 = z/zs.
    ranges: list = field(default_factory=list)
    # parprior `match` rows: (kind, i, j, ii, jj) ties param (i,j) = 1.0*(ii,jj),
    # used to reproduce a shared user-variable's single search dimension.
    matches: list = field(default_factory=list)
    hubble_range: Optional[tuple] = None           # (lo, hi) when hvary
    # basenames the rendered start_command references (None = omit the line)
    point_constraint_file: Optional[str] = None    # -> readobs_point <file>
    prior_file_name: Optional[str] = None          # -> parprior <file>


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
# python-driver-script parsing (import a `import glafic; glafic.*(...)` script)
# --------------------------------------------------------------------------- #

def looks_like_python(text: str) -> bool:
    """Heuristic: is this a python glafic driver script rather than an input file?"""
    return ("import glafic" in text) or ("glafic." in text)


def _const_eval(node, variables: dict):
    """Evaluate a literal / simple expression node from a glafic call argument."""
    import ast
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _const_eval(node.operand, variables)
        if isinstance(v, (int, float)):
            return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.Name):
        return variables.get(node.id)
    return None


def parse_glafic_python(text: str) -> GlaficModel:
    """Parse a python glafic driver script (the colleague's workflow) via AST.

    Recognises init / set_secondary / startup_setnum / set_lens / set_extend /
    set_point / setopt_* / readobs_extend / readnoise_extend / readobs_point /
    parprior. Module-level string/number assignments are resolved so file-path
    variables (``extended_file = '...'``) flow into the call arguments.
    """
    import ast

    model = GlaficModel()
    variables: dict = {}
    lens_opt: dict[int, list[int]] = {}
    ext_opt: dict[int, list[int]] = {}

    tree = ast.parse(text)

    # 1) collect simple module-level assignments (Name = literal)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            val = _const_eval(node.value, variables)
            if val is not None:
                variables[node.targets[0].id] = val

    # 2) walk every glafic.<fn>(...) call
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        fn = call.func
        if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                and fn.value.id == "glafic"):
            continue
        name = fn.attr
        args = [_const_eval(a, variables) for a in call.args]

        if name in ("init", "set_primary") and len(args) >= 12:
            keys = ["omega", "lambda", "weos", "hubble", "prefix",
                    "xmin", "ymin", "xmax", "ymax", "pix_ext", "pix_poi", "maxlev"]
            for k, v in zip(keys, args[:12]):
                if k == "prefix":
                    model.prefix = str(v)
                elif k == "maxlev":
                    model.primary[k] = int(v)
                else:
                    model.primary[k] = float(v)
        elif name == "set_secondary" and args and isinstance(args[0], str):
            parts = args[0].split()
            if len(parts) >= 2:
                try:
                    model.secondary[parts[0]] = float(parts[1])
                except ValueError:
                    model.secondary[parts[0]] = parts[1]
        elif name == "set_lens" and len(args) >= 3:
            nums = [float(a) for a in args[2:]]
            z = nums[0] if nums else 0.0
            params = (nums[1:] + [0.0] * NPAR)[:NPAR]
            model.lenses.append(GlaficLens(type=str(args[1]), z=z, params=params))
        elif name == "set_extend" and len(args) >= 3:
            nums = [float(a) for a in args[2:]]
            z = nums[0] if nums else 0.0
            params = (nums[1:] + [0.0] * NPAR)[:NPAR]
            model.extends.append(GlaficExtend(type=str(args[1]), z=z, params=params))
        elif name == "set_point" and len(args) >= 4:
            model.source_z = float(args[1])
            model.source_x = float(args[2])
            model.source_y = float(args[3])
        elif name == "setopt_lens" and len(args) >= 2:
            lens_opt[int(args[0])] = [int(a) for a in args[1:]]
        elif name == "setopt_extend" and len(args) >= 2:
            ext_opt[int(args[0])] = [int(a) for a in args[1:]]
        elif name == "setopt_point" and len(args) >= 2:
            model.point_opt = [int(a) for a in args[1:]][:3]
        elif name == "readobs_extend" and args:
            model.extended_file = str(args[0])
            if len(args) >= 2 and isinstance(args[1], str):
                model.mask_file = str(args[1])
        elif name == "readnoise_extend" and args:
            model.noise_file = str(args[0])
        elif name == "readobs_point" and args:
            model.constraint_file = str(args[0])
        elif name == "parprior" and args:
            model.prior_file = str(args[0])

    # attach opt flags (z + p1..p7) by 1-based id
    for k, lens in enumerate(model.lenses, start=1):
        if k in lens_opt:
            lens.opt = (lens_opt[k] + [0] * (NPAR + 1))[: NPAR + 1]
    for k, ext in enumerate(model.extends, start=1):
        if k in ext_opt:
            ext.opt = (ext_opt[k] + [0] * (NPAR + 1))[: NPAR + 1]
    return model


def parse_glafic_any(text: str) -> GlaficModel:
    """Parse either a glafic python driver script or a glafic input file."""
    if looks_like_python(text):
        return parse_glafic_python(text)
    return parse_glafic_input(text)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def _fmt(v: float) -> str:
    return f"{v:.6e}"


def _setopt_rows(model: GlaficModel) -> Optional[list[str]]:
    """The ``start_setopt`` flag rows, or ``None`` when nothing is optimizable.

    One row of 8 flags (z, p1..p7) per lens, then one row of 3 flags (zs, xs, ys)
    for the point source -- exactly glafic's ``start_setopt`` matrix layout.
    """
    lens_rows = [(list(ll.opt) + [0] * (NPAR + 1))[: NPAR + 1] if ll.opt
                 else [0] * (NPAR + 1) for ll in model.lenses]
    point_row = ((list(model.point_opt) + [0, 0, 0])[:3] if model.point_opt
                 else [0, 0, 0])
    if not any(any(r) for r in lens_rows) and not any(point_row):
        return None
    rows = [" ".join(str(int(f)) for f in r) for r in lens_rows]
    if model.source_z is not None:
        rows.append(" ".join(str(int(f)) for f in point_row))
    return rows


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
    if model.hvary:
        out.append(f"{'hvary':<10}{int(model.hvary)}")
    out.append("")

    npoint = 1 if model.source_z is not None else 0
    out.append(f"startup {len(model.lenses)} 0 {npoint}")
    for lens in model.lenses:
        nums = "  ".join(_fmt(v) for v in [lens.z, *lens.params])
        out.append(f"lens   {lens.type:<6} {nums}")
    if model.source_z is not None:
        out.append(f"point  {_fmt(model.source_z)}  {_fmt(model.source_x)}  "
                   f"{_fmt(model.source_y)}")
    out.append("end_startup")
    out.append("")

    # optimization matrix: glafic needs the start_setopt block to know which
    # parameters amoeba may vary (each {lo, hi} in the glade .dat -> a 1 flag).
    rows = _setopt_rows(model)
    if rows is not None:
        out.append("start_setopt")
        out += rows
        out.append("end_setopt")
        out.append("")

    if command:
        out.append("start_command")
        out.append("")
        if model.optimize and not model.point_constraint_file:
            out.append("# NOTE: no point-source observations were provided; add a")
            out.append("#       'readobs_point <file>' line so optimize has constraints.")
        if model.point_constraint_file:
            out.append(f"readobs_point {model.point_constraint_file}")
        if model.prior_file_name:
            out.append(f"parprior {model.prior_file_name}")
        if model.optimize:
            out.append("optimize")
        out.append("findimg")
        out.append(f"writecrit  {_fmt(model.source_z) if model.source_z is not None else '0.0'}")
        if model.optimize:
            out.append("printmodel")
        out.append("")
        out.append("quit")
        out.append("")
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


def render_glafic_point_constraint(obs: GlaficObs,
                                   center_offset: tuple = (0.0, 0.0)) -> str:
    """Render observed images as a glafic ``readobs_point`` constraint file.

    Unlike :func:`render_glafic_obs` (a GLADE round-trip artifact wrapped in
    ``start_obs``/``end_obs``, which the glafic binary cannot read), this is the
    standalone file glafic loads with ``readobs_point`` for ``optimize``: a
    header ``id n zs zserr`` then one row of exactly eight columns per image
    (``x y flux sigma_pos sigma_flux td sigma_td parity``). Positions are shifted
    by ``-center_offset`` into glafic's lens frame, and the (possibly signed)
    magnification becomes ``flux=|mu|`` with the sign carried in ``parity``.
    """
    cox, coy = center_offset
    out = ["# GLADE point-source constraints for glafic optimize (readobs_point)",
           "# columns: x  y  flux  sigma_pos  sigma_flux  td  sigma_td  parity",
           f"1 {len(obs.images)} {obs.zs:.6f} 0.0"]
    for (x, y, mag, spos, smag, _flag) in obs.images:
        parity = -1 if mag < 0 else (1 if mag > 0 else 0)
        out.append(f"  {x - cox:12.6f} {y - coy:12.6f} {abs(mag):12.5f} "
                   f"{spos:10.6f} {abs(smag):8.5f} 0.000000 0.000000 {parity}")
    return "\n".join(out) + "\n"


def render_glafic_prior(ranges: list, hubble_range: Optional[tuple] = None,
                        matches: Optional[list] = None) -> str:
    """Render parprior ``range`` (+ ``match``) rows.

    ``range``: param_no 1 = z/zs, 2.. = p1..p7. ``match <kind> i j ii jj 1.0 0.0``
    hard-ties param (i,j) to (ii,jj) (sig 0 -> glafic copies the value each step,
    no chi2 term), reproducing a shared user-variable's single search dimension.
    """
    out = ["# GLADE parameter ranges for glafic optimize (parprior format)",
           "# range <lens|point> <id> <param_no> <min> <max>   (param 1 = z/zs)"]
    for (kind, idx, j, lo, hi) in ranges:
        out.append(f"range {kind} {int(idx)} {int(j)} {float(lo):.8e} {float(hi):.8e}")
    if hubble_range is not None:
        out.append(f"range hubble {float(hubble_range[0]):.8e} "
                   f"{float(hubble_range[1]):.8e}")
    if matches:
        out.append("# match <lens> <i> <j> <ii> <jj> 1.0 0.0 = tie (i,j) to (ii,jj)"
                   " (shared variable)")
        for (kind, i, j, ii, jj) in matches:
            out.append(f"match {kind} {int(i)} {int(j)} {int(ii)} {int(jj)} 1.0 0.0")
    return "\n".join(out) + "\n"


def looks_like_glafic_input(text: str) -> bool:
    """Heuristic: is *text* a native glafic input file (not a glade ``.dat``)?

    glade ``.dat`` files carry ``'name': (...)`` component tuples and ``=``
    scalar assignments; a glafic input has ``startup`` / ``lens`` lines and a
    ``start_command`` block instead.
    """
    import re
    if looks_like_python(text):
        return False
    if re.search(r"'[^']+'\s*:\s*\(", text):    # glade component tuple
        return False
    markers = ("startup", "start_command", "end_startup", "start_setopt")
    has_marker = any(m in text for m in markers)
    has_lens = re.search(r"(?m)^\s*lens\s+\w", text) is not None
    return has_marker or has_lens
