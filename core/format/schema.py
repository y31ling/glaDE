"""Per-model parameter schema for glade ``.dat`` components, plus scalar-key
classification and backend capability sets.

Parameter orders follow the glafic ``set_lens(id, type, z, p1..p7)`` convention
and were verified against ``glafic2/mass.c`` and the legacy ``set_lens`` calls
(e.g. ``lens sers z M x y e pa re n`` and ``lens king z M x y e pa rc c``).

``is_mass`` flags the parameter that is optimized in log10 space (mass /
velocity dispersion / shear amplitude). Perturbation-style models (``pert``,
``gaupot``, ``clus3``, ``mpole``, ``crline``) have an irregular layout; their
entries are marked ``uncertain`` and should be treated as best-effort labels —
the numbers are still passed through to the engine unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParamSpec:
    name: str
    is_mass: bool = False
    desc: str = ""


@dataclass(frozen=True)
class ModelSpec:
    key: str                      # glade keyword as written in a tuple
    glafic_key: str               # keyword passed to glafic set_lens
    params: tuple[ParamSpec, ...] # meaningful params after z, in order
    category: str = "lens"        # 'lens' | 'substructure' (default authoring bucket)
    gpu: bool = False             # supported by the Rhongomyniad GPU backend
    required_min: int = 3         # min params the user must supply (mass + x + y)
    uncertain: bool = False       # param labels are best-effort
    desc: str = ""

    @property
    def mass_positions(self) -> tuple[int, ...]:
        return tuple(i for i, p in enumerate(self.params) if p.is_mass)


def _P(name, is_mass=False, desc=""):
    return ParamSpec(name, is_mass, desc)


# Common trailing pairs reused below.
_XY = (_P("x", desc="centre x [arcsec]"), _P("y", desc="centre y [arcsec]"))
_E_PA = (_P("e", desc="ellipticity in [0,1)"), _P("pa", desc="position angle [deg]"))


MODELS: dict[str, ModelSpec] = {}


def _reg(spec: ModelSpec) -> None:
    MODELS[spec.key] = spec


# ---- point mass / sub-structure primitives ---------------------------------
_reg(ModelSpec(
    "point", "point",
    (_P("mass", is_mass=True, desc="point mass [Msun]"), *_XY),
    category="substructure", gpu=True, required_min=3,
    desc="Point mass.",
))

# ---- isothermal / pseudo-Jaffe ---------------------------------------------
_reg(ModelSpec(
    "sie", "sie",
    (_P("sigma", is_mass=True, desc="velocity dispersion [km/s]"), *_XY, *_E_PA,
     _P("rcore", desc="core/softening radius [arcsec]")),
    category="lens", gpu=True, required_min=3,
    desc="Singular isothermal ellipsoid.",
))
_reg(ModelSpec(
    "jaffe", "jaffe",
    (_P("sigma", is_mass=True, desc="velocity dispersion [km/s]"), *_XY, *_E_PA,
     _P("a", desc="outer truncation radius [arcsec]"),
     _P("rco", desc="inner core radius [arcsec]")),
    category="substructure", gpu=True, required_min=3,
    desc="Pseudo-Jaffe (difference of two SIEs).",
))

# ---- Sersic -----------------------------------------------------------------
_reg(ModelSpec(
    "sers", "sers",
    (_P("mass", is_mass=True, desc="total mass [Msun]"), *_XY, *_E_PA,
     _P("re", desc="effective radius [arcsec]"),
     _P("n", desc="Sersic index (0.06..20)")),
    category="lens", gpu=True, required_min=3,
    desc="Deprojected Sersic density.",
))
_reg(ModelSpec(
    "serspot", "serspot",
    MODELS["sers"].params, category="lens", gpu=False, required_min=3,
    desc="Sersic (potential form).",
))

# ---- NFW family -------------------------------------------------------------
_reg(ModelSpec(
    "nfw", "nfw",
    (_P("mass", is_mass=True, desc="virial mass [Msun]"), *_XY, *_E_PA,
     _P("c", desc="concentration")),
    category="substructure", gpu=True, required_min=3,
    desc="NFW (density form).",
))
_reg(ModelSpec(
    "nfwpot", "nfwpot", MODELS["nfw"].params,
    category="substructure", gpu=True, required_min=3,
    desc="NFW (potential form).",
))
_reg(ModelSpec(
    "gnfw", "gnfw",
    (_P("mass", is_mass=True, desc="virial mass [Msun]"), *_XY, *_E_PA,
     _P("c", desc="concentration"),
     _P("alpha", desc="inner slope (NFW=1)")),
    category="substructure", gpu=False, required_min=3,
    desc="Generalized NFW. Used with alpha=1 as the legacy 'nfw' sub-structure.",
))
_reg(ModelSpec(
    "gnfwpot", "gnfwpot", MODELS["gnfw"].params,
    category="substructure", gpu=False, required_min=3,
    desc="Generalized NFW (potential form).",
))
_reg(ModelSpec(
    "tnfw", "tnfw",
    (_P("mass", is_mass=True, desc="halo mass [Msun]"), *_XY, *_E_PA,
     _P("c", desc="concentration"),
     _P("t", desc="truncation parameter")),
    category="substructure", gpu=False, required_min=3,
    desc="Truncated NFW.",
))
_reg(ModelSpec(
    "tnfwpot", "tnfwpot", MODELS["tnfw"].params,
    category="substructure", gpu=False, required_min=3,
    desc="Truncated NFW (potential form).",
))
_reg(ModelSpec(
    "anfw", "anfw",
    (_P("mass", is_mass=True, desc="mass [Msun]"), *_XY, *_E_PA,
     _P("c", desc="concentration")),
    category="substructure", gpu=False, required_min=3,
    desc="Analytic (CSE-approximated) NFW.",
))

# ---- King -------------------------------------------------------------------
_reg(ModelSpec(
    "king", "king",
    (_P("mass", is_mass=True, desc="mass within tidal radius [Msun]"), *_XY, *_E_PA,
     _P("rc", desc="core radius [arcsec]"),
     _P("c", desc="log10(rt/rc), >= 0")),
    category="substructure", gpu=True, required_min=3,
    desc="King (1966) profile.",
))

# ---- Hernquist --------------------------------------------------------------
_reg(ModelSpec(
    "hern", "hern",
    (_P("mass", is_mass=True, desc="total mass [Msun]"), *_XY, *_E_PA,
     _P("rb", desc="scale radius [arcsec]")),
    category="lens", gpu=False, required_min=3,
    desc="Hernquist (density form).",
))
_reg(ModelSpec(
    "hernpot", "hernpot", MODELS["hern"].params,
    category="lens", gpu=False, required_min=3, desc="Hernquist (potential form).",
))
_reg(ModelSpec(
    "ahern", "ahern", MODELS["hern"].params,
    category="lens", gpu=False, required_min=3, desc="Analytic Hernquist.",
))

# ---- Einasto ----------------------------------------------------------------
_reg(ModelSpec(
    "ein", "ein",
    (_P("mass", is_mass=True, desc="mass [Msun]"), *_XY, *_E_PA,
     _P("c", desc="concentration"),
     _P("alpha", desc="Einasto index")),
    category="lens", gpu=False, required_min=3, desc="Einasto (density form).",
))
_reg(ModelSpec(
    "einpot", "einpot", MODELS["ein"].params,
    category="lens", gpu=False, required_min=3, desc="Einasto (potential form).",
))

# ---- power law --------------------------------------------------------------
# pow/powpot are a zs_fid-family model: glafic reads para_lens[i][1] as the
# fiducial source redshift, so 're' lives in slot p6 (verified against
# mass.c:286/290 and the kapgam_pow/powpot signatures).
_reg(ModelSpec(
    "pow", "pow",
    (_P("zs_fid", desc="fiducial source redshift"), *_XY, *_E_PA,
     _P("re", is_mass=True, desc="Einstein radius [arcsec]"),
     _P("gamma", desc="3D density slope in [1,3]")),
    category="lens", gpu=False, required_min=3, uncertain=True,
    desc="Power-law (density form).",
))
_reg(ModelSpec(
    "powpot", "powpot", MODELS["pow"].params,
    category="lens", gpu=False, required_min=3, desc="Power-law (potential form).",
))

# ---- perturbations / external (irregular layout: best-effort labels) -------
_reg(ModelSpec(
    "pert", "pert",
    (_P("zs_fid", desc="fiducial source redshift"), *_XY,
     _P("gamma", is_mass=True, desc="external shear amplitude"),
     _P("theta_gamma", desc="shear position angle [deg]"),
     _P("_unused", desc="(unused)"),
     _P("kappa", is_mass=True, desc="external convergence")),
    category="lens", gpu=True, required_min=3, uncertain=True,
    desc="External shear + convergence.",
))
_reg(ModelSpec(
    "gaupot", "gaupot",
    (_P("zs_fid", desc="fiducial source redshift"), *_XY, *_E_PA,
     _P("sigma", is_mass=True, desc="Gaussian width"),
     _P("kappa0", is_mass=True, desc="central convergence")),
    category="lens", gpu=True, required_min=3, uncertain=True,
    desc="Gaussian potential.",
))
_reg(ModelSpec(
    "clus3", "clus3",
    (_P("zs_fid", desc="fiducial source redshift"), *_XY,
     _P("gamma", is_mass=True, desc="quadrupole strength"),
     _P("theta_gamma", desc="quadrupole position angle [deg]")),
    category="lens", gpu=False, required_min=3, uncertain=True,
    desc="Cluster quadrupole.",
))
_reg(ModelSpec(
    "mpole", "mpole",
    (_P("zs_fid", desc="fiducial source redshift"), *_XY,
     _P("gamma", is_mass=True, desc="multipole amplitude"),
     _P("theta_gamma", desc="multipole position angle [deg]"),
     _P("m", desc="multipole order"),
     _P("n", desc="power-law exponent")),
    category="lens", gpu=False, required_min=3, uncertain=True,
    desc="m-th order multipole.",
))

# ---- catalogue / line (rarely used here) -----------------------------------
_reg(ModelSpec(
    "gals", "gals",
    (_P("scale", desc="catalogue scaling"),),
    category="lens", gpu=False, required_min=1, uncertain=True,
    desc="External galaxy catalogue (file-based).",
))

# ---- extended-source profiles (glafic set_extend) --------------------------
# These describe a *source* surface-brightness profile (the lensed extended
# image / Einstein ring), NOT a deflector. They map to glafic's set_extend and
# live in their own engine stack. The tuple z is the SOURCE redshift; the 7
# params are para_ext[i][1..7] = (norm, x, y, e, pa, r0, n) where source_all()
# interprets r0 as the size (effective radius / sigma / radius) and n as the
# Sersic/Moffat index (unused for gauss/tophat). 'norm' is the peak brightness
# (flag_extnorm=0) or total flux (flag_extnorm=1). Extended sources are CPU-only.
_EXT_NORM = _P("norm", desc="amplitude: peak brightness (flag_extnorm=0) or total flux (=1)")
_EXT_E_PA = (_P("e", desc="ellipticity in [0,1)"), _P("pa", desc="position angle [deg]"))

_reg(ModelSpec(
    "extsersic", "sersic",
    (_EXT_NORM, *_XY, *_EXT_E_PA,
     _P("re", desc="effective radius [arcsec]"),
     _P("n", desc="Sersic index (e.g. 4=bulge, 1=disk)")),
    category="extend", gpu=False, required_min=7,
    desc="Extended source: Sersic surface-brightness profile.",
))
_reg(ModelSpec(
    "extgauss", "gauss",
    (_EXT_NORM, *_XY, *_EXT_E_PA,
     _P("sigma", desc="Gaussian width [arcsec]"),
     _P("_unused", desc="(unused for gauss)")),
    category="extend", gpu=False, required_min=6,
    desc="Extended source: 2D Gaussian profile.",
))
_reg(ModelSpec(
    "exttophat", "tophat",
    (_EXT_NORM, *_XY, *_EXT_E_PA,
     _P("radius", desc="top-hat radius [arcsec]"),
     _P("_unused", desc="(unused for tophat)")),
    category="extend", gpu=False, required_min=6,
    desc="Extended source: uniform elliptical top-hat.",
))
_reg(ModelSpec(
    "extmoffat", "moffat",
    (_EXT_NORM, *_XY, *_EXT_E_PA,
     _P("rd", desc="Moffat scale radius [arcsec]"),
     _P("beta", desc="Moffat beta index")),
    category="extend", gpu=False, required_min=7,
    desc="Extended source: Moffat profile.",
))
_reg(ModelSpec(
    "extjaffe", "jaffe",
    (_EXT_NORM, *_XY, *_EXT_E_PA,
     _P("a", desc="Jaffe scale radius [arcsec]"),
     _P("rco", desc="inner core radius [arcsec]")),
    category="extend", gpu=False, required_min=6,
    desc="Extended source: (pseudo-)Jaffe profile.",
))

# Number of glafic parameter slots after z (p1..p7).
GLAFIC_NPARAM = 7

# Backends and the model keywords each one supports.
BACKENDS = ("cpu", "gpu", "glafic")
GPU_MODELS = frozenset(k for k, m in MODELS.items() if m.gpu)
ALL_MODELS = frozenset(MODELS)

# Extended-source models (glafic set_extend). They form a separate engine stack
# and are only runnable on the CPU/glafic backend.
EXTEND_CATEGORY = "extend"
EXTEND_MODELS = frozenset(k for k, m in MODELS.items() if m.category == EXTEND_CATEGORY)
# Deflector models (set_lens) = everything that is not an extended source.
LENS_MODELS = frozenset(k for k in MODELS if k not in EXTEND_MODELS)


def model(key: str) -> ModelSpec | None:
    return MODELS.get(key)


def is_extend_model(key: str) -> bool:
    """Whether *key* is an extended-source (set_extend) model."""
    spec = MODELS.get(key)
    return spec is not None and spec.category == EXTEND_CATEGORY


def is_backend(name: str) -> bool:
    return name in BACKENDS


def supports(backend: str, model_key: str) -> bool:
    """Whether *backend* can run *model_key*."""
    if backend == "gpu":
        return model_key in GPU_MODELS
    # cpu (glafic bindings) and glafic-direct support every known model.
    return model_key in ALL_MODELS


# ---- scalar-key classification ---------------------------------------------

COSMOLOGY_KEYS = ("omega", "lambda_cosmo", "weos", "hubble")
GRID_KEYS = ("xmin", "ymin", "xmax", "ymax", "pix_ext", "pix_poi", "maxlev")
REDSHIFT_KEYS = ("source_z", "lens_z")
SOURCE_KEYS = ("source_x", "source_y")
OBS_ARRAY_KEYS = (
    "obs_positions_mas_list",
    "obs_magnifications_list",
    "obs_mag_errors_list",
    "obs_pos_sigma_mas_list",
)
OBS_OTHER_KEYS = ("center_offset_x", "center_offset_y", "obs_x_flip")

# ---- extended-source observation keys (CPU extend mode) --------------------
# File-path scalars (strings). extended_file is the observed FITS image fitted
# pixel-by-pixel; the others are optional glafic-native passthroughs.
EXTEND_FILE_KEYS = (
    "extended_file",     # observed extended-image FITS -> readobs_extend
    "extend_mask_file",  # optional pixel mask FITS      -> readobs_extend(mask=)
    "noise_file",        # optional per-pixel noise FITS -> readnoise_extend
    "constraint_file",   # optional glafic point-constraint file -> readobs_point
    "prior_file",        # optional glafic prior file    -> parprior
)
# Optional extra point-source observation columns (glade arrays) that the legacy
# four arrays cannot express; only used in extend mode when point obs are given
# as glade arrays rather than a constraint_file.
OBS_EXTEND_ARRAY_KEYS = ("obs_td_list", "obs_td_err_list", "obs_parity_list")

# glafic set_secondary engine settings, exposed so an extend run can reproduce a
# glafic input exactly. Emitted verbatim as 'set_secondary("<key> <value>")'.
SECONDARY_KEYS = (
    "chi2_splane", "chi2_checknimg", "chi2_restart", "chi2_usemag",
    "ran_seed", "obs_gain", "obs_ncomb", "obs_readnoise", "flag_extnorm",
)
# Per-component chi2 weights for the extend-mode weighted loss. All == 1 exactly
# reproduces glafic c2calc; legacy point-only behaviour = W_POS, W_FLUX only.
WEIGHT_KEYS = ("W_POS", "W_FLUX", "W_TD", "W_EXT", "W_PRIOR")

ALGORITHM_KEYS = (
    "DE_MAXITER", "DE_POPSIZE", "DE_ATOL", "DE_TOL", "DE_SEED", "DE_POLISH",
    "DE_WORKERS", "EARLY_STOPPING", "EARLY_STOP_PATIENCE",
    "LOSS_COEF_A", "LOSS_COEF_B", "LOSS_PENALTY_PL", "missing_img_penalty",
    "CONSTRAINT_SIGMA",
    "PENALTY_COEFFICIENT", "Draw_Graph", "draw_interval", "PRINT_INTERVAL",
    "COMPARE_GRAPH", "SHOW_2SIGMA", "OUTPUT_PREFIX", "glafic_verified",
    "MCMC_ENABLED", "MCMC_NWALKERS", "MCMC_NSTEPS", "MCMC_BURNIN", "MCMC_THIN",
    "MCMC_PERTURBATION", "MCMC_PROGRESS", "MCMC_WORKERS",
) + WEIGHT_KEYS + SECONDARY_KEYS

# Removed in V0.4.1: the MCMC prior is now always the DE {lower, upper} bounds,
# so these custom-range keys are obsolete. Old .dat files that still set them are
# accepted with a warning (see validate.py) rather than rejected.
DEPRECATED_KEYS = frozenset({
    "MCMC_CUSTOM_RANGE", "MCMC_SEARCH_RADIUS", "MCMC_LOG_M_MIN", "MCMC_LOG_M_MAX",
})

# canonical alias resolution (the .dat may use glafic's 'lambda'; an UPPER-case
# spelling of the missing-image penalty is accepted for consistency with the
# other algorithm knobs).
SCALAR_ALIASES = {"lambda": "lambda_cosmo",
                  "MISSING_IMG_PENALTY": "missing_img_penalty"}

# Hard-required for a runnable config (no defaults).
REQUIRED_OBS_KEYS = OBS_ARRAY_KEYS


def classify_scalar(name: str) -> str:
    """Return the section a scalar key belongs to."""
    name = SCALAR_ALIASES.get(name, name)
    if name in COSMOLOGY_KEYS:
        return "cosmology"
    if name in GRID_KEYS:
        return "grid"
    if name in REDSHIFT_KEYS:
        return "redshifts"
    if name in SOURCE_KEYS:
        return "source"
    if (name in OBS_ARRAY_KEYS or name in OBS_OTHER_KEYS
            or name in EXTEND_FILE_KEYS or name in OBS_EXTEND_ARRAY_KEYS):
        return "obs"
    if name in ALGORITHM_KEYS:
        return "algorithm"
    return "other"
