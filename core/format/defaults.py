"""Default values for basic configuration variables.

Taken verbatim from the legacy CPU point-mass script
(``legacy/v_pointmass_1_0/version_pointmass_1_0.py``) -- the iPTF16geu setup --
per the V0.4 decision that omitted basics fall back to those values.

The four observation arrays and at least one component are *hard-required* and
intentionally have no default; a run is blocked if they are absent.
"""
from __future__ import annotations

# Cosmology + grid + redshifts + source position + algorithm.
DEFAULTS: dict[str, object] = {
    # cosmology
    "omega": 0.3,
    "lambda_cosmo": 0.7,
    "weos": -1.0,
    "hubble": 0.7,
    # grid
    "xmin": -0.5,
    "ymin": -0.5,
    "xmax": 0.5,
    "ymax": 0.5,
    "pix_ext": 0.01,
    "pix_poi": 0.2,
    "maxlev": 5,
    # redshifts
    "source_z": 0.4090,
    "lens_z": 0.2160,
    # source position (point source)
    "source_x": 2.685497e-03,
    "source_y": 2.443616e-02,
    # observation offsets / convention (the arrays themselves are required)
    "center_offset_x": 0.01535,
    "center_offset_y": 0.0322,
    "obs_x_flip": True,
    # differential evolution
    "DE_MAXITER": 650,
    "DE_POPSIZE": 64,
    "DE_ATOL": 1e-4,
    "DE_TOL": 1e-4,
    "DE_SEED": 42,
    "DE_POLISH": True,
    "DE_WORKERS": -1,
    "EARLY_STOPPING": True,
    "EARLY_STOP_PATIENCE": 30,
    # loss
    "LOSS_COEF_A": 4,
    "LOSS_COEF_B": 1,
    "LOSS_PENALTY_PL": 10000,
    "CONSTRAINT_SIGMA": 1,
    "PENALTY_COEFFICIENT": 1000,
    # plotting / output
    "Draw_Graph": 1,
    "draw_interval": 5,
    "COMPARE_GRAPH": True,
    "SHOW_2SIGMA": False,
    "OUTPUT_PREFIX": "glade_run",
    # mcmc (off by default)
    "MCMC_ENABLED": False,
    "MCMC_NWALKERS": 32,
    "MCMC_NSTEPS": 2000,
    "MCMC_BURNIN": 300,
    "MCMC_THIN": 2,
    "MCMC_PERTURBATION": 0.01,
    "MCMC_PROGRESS": True,
    "MCMC_WORKERS": -1,
}


def default_for(name: str):
    return DEFAULTS.get(name)


def has_default(name: str) -> bool:
    return name in DEFAULTS
