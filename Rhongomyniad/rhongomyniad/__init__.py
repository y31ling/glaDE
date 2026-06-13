"""
Rhongomyniad — GPU calculator for glafic-compatible gravitational-lens modelling.

Public API mirrors glafic's Python module (glafic.init, set_lens, model_init,
point_solve, calcimage, findimg) so code written against glafic runs here
with only a change of `import` lines.
"""

from .api import (
    init,
    set_cosmo,
    set_primary,
    quit,
    startup_setnum,
    set_lens,
    set_point,
    set_extend,
    set_secondary,
    set_psf,
    setopt_point,
    model_init,
    calcimage,
    point_solve,
    findimg,
    findimg_i,
    readobs_extend,
    readnoise_extend,
    readobs_point,
    parprior,
    c2calc,
    c2calc_each,
    get_nxy_ext,
    writeimage,
    getpar_lens,
    getpar_point,
    getpar_extend,
    getpar_sky,
    getpar_omega,
    getpar_lambda,
    getpar_hubble,
    getpar_weos,
    get_device,
    set_device,
    set_dtype,
    set_max_poi_tol,
    set_nmax_poi_ite,
    set_smallcore,
    set_nfw_users,
    set_finder,
    get_finder,
    supported_models,
)

__version__ = "0.5.0"
