"""
Physical, numerical, and default constants.

Values are taken verbatim from glafic's glafic.h so that outputs of
Rhongomyniad match glafic bit-for-bit.  Do NOT change these without
cross-checking against the reference C header.
"""

from __future__ import annotations

import math

# --- physical constants (glafic.h lines 410-419) -------------------------
ARCSEC2RADIAN = 0.00000484813681109536       # arcsec -> rad
COVERH_MPCH = 2997.92458                      # c/H0 in Mpc/h
FAC_CRITDENS = 5.5467248e+14                  # Sigma_crit normalization
FAC_TDELAY_DAY = 3.571386089689082e12         # time delay factor -> days
MPC2METER = 3.085677581e22
R_SCHWARZ = 2953.339382                        # Schwarzschild radius of M_sun (m)
C_LIGHT_KMS = 2.99792458e5                     # speed of light in km/s
NFW_RS_NORM = 0.00009510361                    # rs scaling for NFW
NFW_B_NORM = 6.34482175e-8                     # Einstein radius scaling for NFW

PI = math.pi

# --- default primary parameters (glafic.h:28-39) -------------------------
DEF_OMEGA = 0.3
DEF_LAMBDA = 0.7
DEF_WEOS = -1.0
DEF_HUBBLE = 0.7
DEF_XMIN = -60.0
DEF_YMIN = -60.0
DEF_XMAX = 60.0
DEF_YMAX = 60.0
DEF_PIX_EXT = 0.2
DEF_PIX_POI = 3.0
DEF_MAXLEV = 5

# --- image-finder tolerances (glafic.h:62-74) ----------------------------
DEF_NMAX_POI_ITE = 10
DEF_MAX_POI_TOL = 1.0e-10
DEF_POI_IMAG_MAX = 5.0
DEF_POI_IMAG_MIN = 0.001
DEF_IMAG_CEIL = 1.0e-10
DEF_SMALLCORE = 1.0e-10
DEF_FLATFIX = 0
DEF_FLAG_HODENSITY = 0
DEF_HODENSITY = 200.0
DEF_NFW_USERS = 0
DEF_GNFW_USETAB = 1
DEF_EIN_USETAB = 1

# --- array limits --------------------------------------------------------
NPAR_LEN = 8
NPAR_POI = 3
NPAR_LMODEL = 8     # ax, ay, td, kap, g1, g2, muinv, rot
NPAR_IMAGE = 4      # x, y, mag, td
NMAX_LEN = 2000
NMAX_POI = 1000
NMAX_POIMG = 50
NMAX_MAXLEV = 20

# --- numerical tolerances -----------------------------------------------
TOL_ZS = 1.0e-6
OFFSET_LOG = 1.0e-300
OFFSET_TDELAY_FAC = 1.0e-300
TOL_CURVATURE = 1.0e-6
GSL_EPSREL_DISTANCE = 1.0e-6
TOL_ROMBERG_JHK = 5.0e-4
ULIM_JHK = 1.0e-8
TDMIN_SET = 1.0e30

# --- lens model string -> integer id (mass.c:11-38) ---------------------
# Keep order identical to glafic so we can share input files verbatim.
LENS_MODEL_NAMES: tuple[str, ...] = (
    "gals",     # 1
    "nfwpot",   # 2
    "sie",      # 3
    "jaffe",    # 4
    "point",    # 5
    "pert",     # 6
    "clus3",    # 7
    "mpole",    # 8
    "hernpot",  # 9
    "nfw",      # 10
    "hern",     # 11
    "powpot",   # 12
    "pow",      # 13
    "gnfwpot",  # 14
    "gnfw",     # 15
    "serspot",  # 16
    "sers",     # 17
    "tnfwpot",  # 18
    "tnfw",     # 19
    "einpot",   # 20
    "ein",      # 21
    "anfw",     # 22
    "ahern",    # 23
    "crline",   # 24
    "gaupot",   # 25
    "king",     # 26
)

LENS_MODEL_ID: dict[str, int] = {name: i + 1 for i, name in enumerate(LENS_MODEL_NAMES)}


def lmodel_to_int(name: str) -> int:
    """Convert a lens model name to glafic's integer id (or 0 if unknown)."""
    return LENS_MODEL_ID.get(name, 0)


def int_to_lmodel(i: int) -> str:
    if i < 1 or i > len(LENS_MODEL_NAMES):
        raise ValueError(f"invalid lens model id {i}")
    return LENS_MODEL_NAMES[i - 1]


# pout output-slot indices (mass.c:55-66)
POUT_AX = 0
POUT_AY = 1
POUT_TD = 2
POUT_KAP = 3
POUT_GAM1 = 4
POUT_GAM2 = 5
POUT_MUINV = 6
POUT_ROT = 7
