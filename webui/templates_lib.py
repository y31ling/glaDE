"""Template snippets for the Editor's Template panel.

Lens / sub-structure snippets are generated from ``core.format.schema`` so they
always match the supported models and parameter order. ``$float`` / ``$int`` mark
positions the user fills; ``$float{lower, upper}`` marks an optimizable parameter.
The component index ``N`` and name are renumbered client-side on insert.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.format import schema  # noqa: E402

DISPLAY_NAME = {
    "point": "point-mass", "jaffe": "p-jaffe", "sers": "Sersic", "sie": "SIE",
    "nfw": "NFW", "gnfw": "gNFW", "tnfw": "tNFW", "king": "King", "hern": "Hernquist",
    "ein": "Einasto", "pow": "power-law", "pert": "perturbation (shear)",
    "gaupot": "Gaussian", "anfw": "analytic NFW",
}

CONSTANTS = """# Basic model parameters
omega = $float          # matter density Omega_m, e.g. 0.3, range (0,1)
lambda_cosmo = $float   # dark-energy density Omega_Lambda, e.g. 0.7
weos = $float           # dark-energy equation of state w, e.g. -1.0
hubble = $float         # H0 / 100, e.g. 0.7

xmin, ymin = $float, $float   # grid lower-left [arcsec], e.g. -0.5, -0.5
xmax, ymax = $float, $float   # grid upper-right [arcsec], e.g. 0.5, 0.5
pix_ext = $float        # extended-source pixel size [arcsec], e.g. 0.01
pix_poi = $float        # point-source pixel size [arcsec], e.g. 0.2
maxlev = $int           # max adaptive refinement level, e.g. 5
"""

IMAGES_DATA = """source_z = $float       # source redshift, e.g. 0.4090
lens_z = $float         # (main) lens redshift, e.g. 0.2160
source_x = $float{lower, upper}   # source x: {lo,hi} optimizes, a value locks
source_y = $float{lower, upper}   # source y

# observed image positions [mas] as [[x1,y1],[x2,y2],...]
obs_positions_mas_list = [[$float, $float], [$float, $float], [$float, $float], [$float, $float]]
obs_magnifications_list = [$float, $float, $float, $float]
obs_mag_errors_list = [$float, $float, $float, $float]
obs_pos_sigma_mas_list = [$float, $float, $float, $float]
center_offset_x = $float
center_offset_y = $float
obs_x_flip = True       # True = sky convention, False = math convention
"""

SOURCE_POINT = """source_z = $float
source_x = $float{lower, upper}    # {lo,hi} to optimize, a value to lock
source_y = $float{lower, upper}
"""

DE_CPU = """# Algorithm: Differential Evolution on the CPU (glafic) backend
LOSS_COEF_A = $float        # weight on position chi2, e.g. 4
LOSS_COEF_B = $float        # weight on magnification chi2, e.g. 1
LOSS_PENALTY_PL = $float    # per-image over-tolerance penalty, e.g. 10000
DE_MAXITER = $int           # e.g. 650
DE_POPSIZE = $int           # population multiplier, e.g. 64
DE_SEED = $int              # e.g. 42
EARLY_STOPPING = True
EARLY_STOP_PATIENCE = $int  # e.g. 30
glafic_verified = True      # after the run, independently re-run the glafic binary
                            # to verify the result (figure unaffected; warns on mismatch)
DE_WORKERS = $int           # -1 = all CPU cores
"""

DE_GPU = """# Algorithm: Differential Evolution on the GPU (Rhongomyniad) backend
# Note: the GPU backend supports a subset of models (point, sie, pert, nfw,
# nfwpot, king, jaffe, gaupot, sers) and a single lens plane.
LOSS_COEF_A = $float        # weight on position chi2, e.g. 4
LOSS_COEF_B = $float        # weight on magnification chi2, e.g. 1
LOSS_PENALTY_PL = $float    # per-image over-tolerance penalty, e.g. 10000
DE_MAXITER = $int           # e.g. 650
DE_POPSIZE = $int           # population multiplier, e.g. 64
DE_SEED = $int              # e.g. 42
EARLY_STOPPING = True
EARLY_STOP_PATIENCE = $int  # e.g. 30
glafic_verified = True      # after the run, independently re-run the glafic binary
                            # to verify the result (figure unaffected; warns on mismatch)
"""

MCMC_GENERAL = """# MCMC sampling (emcee). The prior is ALWAYS the DE {lower, upper} bounds of
# every optimizable parameter; mass-like dims are sampled in log10 space.
# Set MCMC_ENABLED = True to also run MCMC after a DE-CPU / DE-GPU run.
# (The FindImage 'MCMC' mode runs MCMC directly with NO DE, ignoring this flag.)
MCMC_ENABLED = True
MCMC_NWALKERS = $int        # e.g. 32 (auto-raised to >= 2*ndim+2)
MCMC_NSTEPS = $int          # e.g. 2000
MCMC_BURNIN = $int          # steps discarded before thinning, e.g. 300
MCMC_THIN = $int            # e.g. 2
MCMC_PERTURBATION = $float  # walker init spread (fraction of bound width), e.g. 0.01
MCMC_WORKERS = $int         # -1 = all CPU cores (ignored on the vectorized GPU path)
MCMC_PROGRESS = True
"""


def _component_snippet(key: str) -> str:
    spec = schema.model(key)
    if spec is None:
        return ""
    params = ", ".join("$float{lower, upper}" for _ in spec.params)
    line = f"'{key}1': (1, '{key}', lens_z, {params})"
    desc = "; ".join(f"{p.name}: {p.desc}" for p in spec.params if p.desc)
    note = "  # GPU-supported" if spec.gpu else "  # CPU/Glafic only"
    out = f"{line}{note}\n"
    if desc:
        out += f"#   {desc}\n"
    if spec.uncertain:
        out += "#   (parameter labels are best-effort; verify order vs glafic docs)\n"
    return out


# main-lens model keys vs sub-structure model keys, by schema category
def _keys(category: str) -> list:
    order = ["sers", "sie", "pow", "hern", "ein", "pert", "gaupot",
             "point", "nfw", "gnfw", "tnfw", "anfw", "king", "jaffe"]
    keys = [k for k in order if schema.model(k) and schema.model(k).category == category]
    # append any others not in the explicit order
    for k, m in schema.MODELS.items():
        if m.category == category and k not in keys and not k.endswith("pot"):
            keys.append(k)
    return keys


def template_tree() -> list:
    def comp_nodes(category):
        return [{"name": DISPLAY_NAME.get(k, k), "key": k,
                 "snippet": _component_snippet(k)} for k in _keys(category)]

    return [
        {"name": "OBS DATA", "children": [
            {"name": "Images Data", "snippet": IMAGES_DATA},
            {"name": "Constants", "snippet": CONSTANTS},
            {"name": "Extend_images", "snippet": "", "disabled": True},
        ]},
        {"name": "Source", "children": [
            {"name": "point", "snippet": SOURCE_POINT},
            {"name": "gauss", "snippet": "", "disabled": True},
            {"name": "tophat", "snippet": "", "disabled": True},
        ]},
        {"name": "Lens", "children": comp_nodes("lens")},
        {"name": "Sub-structure", "children": comp_nodes("substructure")},
        {"name": "Algorithm parameters", "children": [
            {"name": "DE-CPU", "snippet": DE_CPU},
            {"name": "DE-GPU", "snippet": DE_GPU},
        ]},
        {"name": "MCMC", "children": [
            {"name": "MCMC-GeneralConfig", "snippet": MCMC_GENERAL},
        ]},
    ]
