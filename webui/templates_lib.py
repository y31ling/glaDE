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
from core.format import units as _units  # noqa: E402

_INPUT_DIR = os.path.join(_ROOT, "InputFiles")

DISPLAY_NAME = {
    "point": "point-mass", "jaffe": "p-jaffe", "sers": "Sersic", "sie": "SIE",
    "nfw": "NFW", "gnfw": "gNFW", "tnfw": "tNFW", "king": "King", "hern": "Hernquist",
    "ein": "Einasto", "pow": "power-law", "pert": "perturbation (shear)",
    "gaupot": "Gaussian", "anfw": "analytic NFW",
    "extsersic": "Sersic (extend)", "extgauss": "Gaussian (extend)",
    "exttophat": "top-hat (extend)", "extmoffat": "Moffat (extend)",
    "extjaffe": "Jaffe (extend)",
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


def _images_data(labels: dict, unitsetting=None) -> str:
    """The observation/basic snippet, with unit comments following the active
    profile. When a non-default profile is active, a ``UnitSetting`` line is
    prepended so the inserted config actually binds the profile."""
    obs_u, src_u = labels["obs_pos"], labels["src_pos"]
    head = f"UnitSetting = '{unitsetting}'   # unit profile bound to this config\n\n" \
        if unitsetting else ""
    return (
        f"{head}"
        "source_z = $float       # source redshift, e.g. 0.4090\n"
        "lens_z = $float         # (main) lens redshift, e.g. 0.2160\n"
        f"source_x = $float{{lower, upper}}   # source x [{src_u}]: "
        "{lo,hi} optimizes, a value locks\n"
        f"source_y = $float{{lower, upper}}   # source y [{src_u}]\n"
        "\n"
        f"# observed image positions [{obs_u}] as [[x1,y1],[x2,y2],...]\n"
        "obs_positions_mas_list = [[$float, $float], [$float, $float], "
        "[$float, $float], [$float, $float]]\n"
        "obs_magnifications_list = [$float, $float, $float, $float]\n"
        "obs_mag_errors_list = [$float, $float, $float, $float]\n"
        f"obs_pos_sigma_mas_list = [$float, $float, $float, $float]   "
        f"# position sigmas [{obs_u}]\n"
        "center_offset_x = $float\n"
        "center_offset_y = $float\n"
        "obs_x_flip = True       # True = sky convention, False = math convention\n")


def _source_point(labels: dict) -> str:
    src_u = labels["src_pos"]
    return (
        "source_z = $float\n"
        f"source_x = $float{{lower, upper}}    # [{src_u}]; "
        "{lo,hi} to optimize, a value to lock\n"
        f"source_y = $float{{lower, upper}}    # [{src_u}]\n")


DE_CPU = """# Algorithm: Differential Evolution on the CPU (glafic) backend
LOSS_COEF_A = $float        # weight on position chi2, e.g. 4
LOSS_COEF_B = $float        # weight on magnification chi2, e.g. 1
LOSS_PENALTY_PL = $float    # per-image over-tolerance penalty, e.g. 10000
missing_img_penalty = $float # per-missing-image penalty when a candidate forms
                            # FEWER images than observed; 0 = hard-reject (default)
abs_mag = True              # compare |mu| (parity-insensitive, default); False = signed
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
# Note: the GPU backend supports every deflector model except the file-based
# 'gals', on a single lens plane; any optimizable model (and an optimizable
# source position) is evaluated whole-population batched on the GPU, provided
# hubble, component redshifts and zs_fid (p1 of pert/gaupot/pow/...) stay
# fixed — otherwise the run falls back to per-candidate GPU evaluation.
gpu_precision = $int        # 64 = fp64 (default) | 48 = mixed (fp32 fields,
                            # fp64 Newton refine) | 32 = fp32 — 48/32 speed up
                            # Schramm-heavy models (sers/nfw/...) on consumer GPUs
LOSS_COEF_A = $float        # weight on position chi2, e.g. 4
LOSS_COEF_B = $float        # weight on magnification chi2, e.g. 1
LOSS_PENALTY_PL = $float    # per-image over-tolerance penalty, e.g. 10000
missing_img_penalty = $float # per-missing-image penalty when a candidate forms
                            # FEWER images than observed; 0 = hard-reject (default)
abs_mag = True              # compare |mu| (parity-insensitive, default); False = signed
DE_MAXITER = $int           # e.g. 650
DE_POPSIZE = $int           # population multiplier, e.g. 64
DE_SEED = $int              # e.g. 42
EARLY_STOPPING = True
EARLY_STOP_PATIENCE = $int  # e.g. 30
glafic_verified = True      # after the run, independently re-run the glafic binary
                            # to verify the result (figure unaffected; warns on mismatch)
"""

EXTEND_IMAGES = """# Extended-source (FITS) fitting -- CPU only. Setting extended_file switches the
# run to the extended-source path: DE drives glafic's per-pixel chi2 (c2calc).
extended_file = $str        # observed extended-image FITS, e.g. 'host_ER.fits'
# extend_mask_file = $str   # optional pixel mask FITS (1 = ignore pixel)
# noise_file = $str         # optional per-pixel noise FITS (else analytic, below)

# --- optional point-source (SN) constraints behind the ring ---
# constraint_file = $str    # glafic readobs_point file (x y flux pos_sig flux_err td td_err parity)
# prior_file = $str         # glafic parprior file (gauss/range priors on params)

# --- glafic chi2 / noise engine settings (set_secondary) ---
chi2_splane = $int          # point chi2: 0 = image-plane, 1 = source-plane, e.g. 1
chi2_checknimg = $int       # 1 = penalise wrong image count, e.g. 1
chi2_usemag = $int          # flux chi2: 0 = use flux, -1 = use |mag|, e.g. -1
obs_gain = $float           # detector gain [e-/ADU], e.g. 1.6
obs_ncomb = $int            # number of combined frames, e.g. 1
obs_readnoise = $float      # read noise [e-], e.g. 3.08
flag_extnorm = $int         # source norm: 0 = peak brightness, 1 = total flux, e.g. 0
"""

DE_EXTEND = """# Algorithm: Differential Evolution for an extended-source (CPU) run.
# The loss is glafic's c2calc broken into components, each with its own weight;
# all weights at 1.0 reproduce glafic's c2calc exactly. The point-only legacy
# case corresponds to W_POS, W_FLUX non-zero and the rest zero.
W_POS = $float              # image-position chi2 weight, e.g. 1.0
W_FLUX = $float             # flux / magnitude chi2 weight, e.g. 1.0
W_TD = $float               # time-delay chi2 weight, e.g. 1.0
W_EXT = $float              # extended-source pixel chi2 weight, e.g. 1.0
W_PRIOR = $float            # parameter-prior chi2 weight, e.g. 1.0
missing_img_penalty = $float # per-missing-image penalty when the SN forms FEWER
                            # images than observed; 0 = glafic's flat reject (default)
DE_MAXITER = $int           # e.g. 400
DE_POPSIZE = $int           # population multiplier, e.g. 32
DE_SEED = $int              # e.g. 42
EARLY_STOPPING = True
EARLY_STOP_PATIENCE = $int  # e.g. 30
glafic_verified = True      # after the run, independently re-check with the glafic binary
DE_WORKERS = $int           # -1 = all CPU cores
"""

# --- BIPOP-CMA-ES / jSO algorithm snippets (V0.7) ------------------------- #
# The OPTIMIZER key selects the point-source algorithm; the commented keys below
# tune it (0 = auto where noted). Uncommented loss/verify keys are shared with DE.
_ALGO_TAIL = """LOSS_COEF_A = $float        # weight on position chi2, e.g. 4
LOSS_COEF_B = $float        # weight on magnification chi2, e.g. 1
LOSS_PENALTY_PL = $float    # per-image over-tolerance penalty, e.g. 10000
missing_img_penalty = $float # per-missing-image penalty; 0 = hard-reject (default)
abs_mag = True              # compare |mu| (parity-insensitive, default); False = signed
glafic_verified = True      # after the run, independently re-run the glafic binary
"""

_GPU_PRECISION = """gpu_precision = $int        # 64 = fp64 (default) | 48 = mixed (recommended:
                            # fp32 fields + fp64 Newton refine) | 32 = fp32
"""

CMAES_KEYS = """OPTIMIZER = 'BIPOP-CMA-ES'
# CMAES_MAXEVALS = $int    # total evaluation budget; 0 = auto (10000 * ndim)
# CMAES_SIGMA0 = $float    # initial step in normalized [0,1] coords (default 0.3)
# CMAES_POPSIZE = $int     # base population lambda; 0 = auto (4 + floor(3 ln n))
# CMAES_RESTARTS = $int    # BIPOP large-restart limit (default 9)
# CMAES_TOLFUN = $float    # stop tolerance on f (default 1e-10)
# CMAES_TOLX = $float      # stop tolerance on x, relative to sigma0 (default 1e-12)
# CMAES_SEED = $int        # RNG seed (default 42; falls back to DE_SEED)
# CMAES_WORKERS = $int     # process pool; 1 = serial, -1 = all cores (default DE_WORKERS)
"""

JSO_KEYS = """OPTIMIZER = 'jSO'
# JSO_MAXEVALS = $int      # total evaluation budget; 0 = auto (10000 * ndim)
# JSO_NP_INIT = $int       # initial population; 0 = auto (round(25 ln(D) sqrt(D)))
# JSO_NP_MIN = $int        # minimum population under linear reduction (default 4)
# JSO_H = $int             # historical memory size for F/CR (default 5)
# JSO_ARC_RATE = $float    # external-archive size factor (default 1.0)
# JSO_PBEST_MAX = $float   # top fraction for current-to-pBest (default 0.25)
# JSO_SEED = $int          # RNG seed (default 42; falls back to DE_SEED)
# JSO_WORKERS = $int       # process pool; 1 = serial, -1 = all cores (default DE_WORKERS)
"""

CMAES_CPU = ("# Algorithm: BIPOP-CMA-ES on the CPU (glafic) backend.\n"
             + CMAES_KEYS + _ALGO_TAIL)
CMAES_GPU = ("# Algorithm: BIPOP-CMA-ES on the GPU (Rhongomyniad) backend.\n"
             + CMAES_KEYS + _GPU_PRECISION + _ALGO_TAIL)
JSO_CPU = ("# Algorithm: jSO on the CPU (glafic) backend.\n"
           + JSO_KEYS + _ALGO_TAIL)
JSO_GPU = ("# Algorithm: jSO on the GPU (Rhongomyniad) backend.\n"
           + JSO_KEYS + _GPU_PRECISION + _ALGO_TAIL)

FINE_TUNING = """# Fine-tuning: staged macro -> substructure -> joint-polish pipeline.
# Round 1 removes the substructure and fits the macro (main lens + source);
# the top_k diverse basins seed chains. Round 2 freezes each chain's macro and
# fits only the substructure. Round 3 prunes chains >10x worse than the best,
# then re-opens EVERY deflector/source parameter (fixed or {lo,hi} alike) in a
# value*(1 +- perturb) box and polishes. Needs >=1 main-lens AND >=1
# substructure component (use 'Nl'/'Ns' index suffixes to disambiguate);
# algoN = 'DE' | 'BIPOP-CMA-ES' | 'jSO' (amoeba unsupported); AN/BN override
# LOSS_COEF_A/B per round. Falls back to a normal run (with a warning) when a
# precondition fails. See SPEC.md for the full semantics.
#              activate algo1  A1  B1   algo2  A2  B2   algo3  perturb A3  B3
fine_tuning = (True,   'DE',   4,  0,   'DE',  4,  1,   'DE',  0.01,   4,  1)
fine_tuning_top_k = $int      # diverse round-1 basins kept as chains, e.g. 3
fine_tuning_diversity = $float # min normalized L-inf distance between basins, e.g. 0.1
"""

MCMC_GENERAL = """# MCMC sampling (emcee). The prior is ALWAYS the DE {lower, upper} bounds of
# every optimizable parameter; mass-like dims are sampled in log10 space.
# Set MCMC_ENABLED = True to also run MCMC after a DE-CPU / DE-GPU run.
# (The FindImage 'MCMC' / 'MCMC-GPU' modes run MCMC directly with NO DE,
#  ignoring this flag.)
MCMC_ENABLED = True
MCMC_NWALKERS = $int        # e.g. 32 (auto-raised to >= 2*ndim+2);
                            # for MCMC-GPU use 1024+ to saturate the batched
                            # CUDA likelihood (unset = GPU auto-default 1024)
MCMC_NSTEPS = $int          # e.g. 2000
MCMC_BURNIN = $int          # steps discarded before thinning, e.g. 300
MCMC_THIN = $int            # e.g. 2
MCMC_PERTURBATION = $float  # walker init spread (fraction of bound width), e.g. 0.01
MCMC_WORKERS = $int         # default 1; -1 = all CPU cores, only in a foreground terminal (ignored on the vectorized GPU path)
MCMC_PROGRESS = True
"""


def _sub_unit_labels(text: str, labels: dict) -> str:
    """Rewrite the engine-default unit tags in a schema description to the active
    profile's labels: ``[h^-1 Msun]`` -> the mass label, ``[arcsec]`` -> the
    component-position label. Fixed-unit tags ([km/s], [deg]) are left untouched.
    A no-op under the default profile."""
    return (text.replace("[h^-1 Msun]", f"[{labels['mass']}]")
                .replace("[arcsec]", f"[{labels['comp_pos']}]"))


def _component_snippet(key: str, labels: dict) -> str:
    spec = schema.model(key)
    if spec is None:
        return ""
    is_extend = spec.category == schema.EXTEND_CATEGORY
    z_tok = "source_z" if is_extend else "lens_z"
    params = ", ".join("$float{lower, upper}" for _ in spec.params)
    line = f"'{key}1': (1, '{key}', {z_tok}, {params})"
    desc = "; ".join(f"{p.name}: {p.desc}" for p in spec.params if p.desc)
    desc = _sub_unit_labels(desc, labels)
    if is_extend:
        note = "  # extended source (CPU only); requires extended_file"
    else:
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


def _resolve_units(units):
    """Normalize the ``template_tree`` ``units`` argument (a units dict, a profile
    name, or None) into ``(units_dict_or_None, profile_name_or_None)``."""
    profile_name = None
    if isinstance(units, str):
        profile_name = units if units.strip() not in ("", "default") else None
        if profile_name:
            resolved, _issues = _units.resolve_profile(profile_name, [_INPUT_DIR])
            units = resolved
        else:
            units = None
    if isinstance(units, dict):
        path = units.get("__path__")
        if path and profile_name is None:
            base = os.path.basename(path)
            if base.endswith(_units.PROFILE_SUFFIX):
                profile_name = base[:-len(_units.PROFILE_SUFFIX)]
    else:
        units = None
    return units, profile_name


def template_tree(units=None) -> list:
    """Editor Template-panel tree. ``units`` may be a units dict, a unit-profile
    name, or None (engine defaults); template comments render in the active
    profile's units, and the observation snippet binds a non-default profile via
    a ``UnitSetting`` line."""
    units, profile_name = _resolve_units(units)
    labels = _units.unit_labels(units)
    unitsetting = profile_name if not _units.is_default(units) else None

    def comp_nodes(category):
        return [{"name": DISPLAY_NAME.get(k, k), "key": k,
                 "snippet": _component_snippet(k, labels)} for k in _keys(category)]

    extend_keys = ["extsersic", "extgauss", "exttophat", "extmoffat", "extjaffe"]
    extend_nodes = [{"name": DISPLAY_NAME.get(k, k), "key": k,
                     "snippet": _component_snippet(k, labels)}
                    for k in extend_keys if schema.model(k)]

    return [
        {"name": "OBS DATA", "children": [
            {"name": "Images Data", "snippet": _images_data(labels, unitsetting)},
            {"name": "Constants", "snippet": CONSTANTS},
            {"name": "Extend_images", "snippet": EXTEND_IMAGES},
        ]},
        {"name": "Source", "children": [
            {"name": "point", "snippet": _source_point(labels)},
            {"name": "gauss", "snippet": "", "disabled": True},
            {"name": "tophat", "snippet": "", "disabled": True},
        ]},
        {"name": "Lens", "children": comp_nodes("lens")},
        {"name": "Sub-structure", "children": comp_nodes("substructure")},
        {"name": "Extend Source", "children": extend_nodes},
        {"name": "Algorithm parameters", "children": [
            {"name": "CPU-glafic", "children": [
                {"name": "DE", "snippet": DE_CPU},
                {"name": "DE-extend", "snippet": DE_EXTEND},
                {"name": "BIPOP-CMA-ES", "snippet": CMAES_CPU},
                {"name": "jSO", "snippet": JSO_CPU},
            ]},
            {"name": "GPU-rhongomyniad", "children": [
                {"name": "DE", "snippet": DE_GPU},
                {"name": "BIPOP-CMA-ES", "snippet": CMAES_GPU},
                {"name": "jSO", "snippet": JSO_GPU},
            ]},
            {"name": "Fine-tuning (staged)", "snippet": FINE_TUNING},
        ]},
        {"name": "MCMC", "children": [
            {"name": "MCMC-GeneralConfig", "snippet": MCMC_GENERAL},
        ]},
    ]
