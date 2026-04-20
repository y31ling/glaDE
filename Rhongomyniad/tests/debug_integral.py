"""Verify j1 at (x=10, y=0) probe against a high-accuracy scipy integration."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from scipy import integrate

from rhongomyniad.cosmology import Cosmology
from rhongomyniad.lens_models import LensContext, _calc_bbtt_nfw, _kappa_nfw_dl
from rhongomyniad.elliptical import ell_integ_j

# Same config as the test scenario.
cosmo = Cosmology(0.3, 0.7, -1.0, 0.7)
ctx = LensContext.build(cosmo, zl=0.3, zs=1.5)

m, c = 1.0e14, 5.0
bb, tt_base = _calc_bbtt_nfw(m, c, ctx)
e, pa = 0.2, 30.0
q = 1.0 - e
tt = tt_base / math.sqrt(q)
si = math.sin(-pa * math.pi / 180.0)
co = math.cos(-pa * math.pi / 180.0)
tx, ty = 10.0, 0.0
bx = (co * tx - si * ty) / tt
by = (si * tx + co * ty) / tt
print(f"bx={bx:.12e}, by={by:.12e}")

# scipy high-accuracy reference for j1 and j0
def kappa_nfw_np(x):
    # scalar-at-a-time numpy
    if x > 1.0 + 1e-6:
        return 0.5 * (1 - 2*math.atan(math.sqrt((x-1)/(x+1)))/math.sqrt(x*x-1)) / (x*x - 1)
    if x < 1.0 - 1e-6:
        return 0.5 * (2*math.atanh(math.sqrt((1-x)/(1+x)))/math.sqrt(1-x*x) - 1) / (1 - x*x)
    return 0.5 / 3.0

def make_integrand(n, bx, by, q):
    def integrand(u):
        equ = 1.0 - (1.0 - q*q) * u
        xi2 = u * (by*by + bx*bx/equ + 1e-20)
        xi = math.sqrt(xi2)
        k = kappa_nfw_np(xi)
        # equ^(n+1/2) = sqrt(equ) * equ^n
        denom = math.sqrt(equ)
        for _ in range(n):
            denom *= equ
        return k / denom
    return integrand

for n in [0, 1]:
    ref, _ = integrate.quad(make_integrand(n, bx, by, q), 0.0, 1.0,
                           epsabs=0, epsrel=1e-12, limit=500)
    print(f"scipy j{n} = {ref:.15e}")

# My implementation
bx_t = torch.tensor([bx], dtype=torch.float64)
by_t = torch.tensor([by], dtype=torch.float64)
q_t = torch.tensor(q, dtype=torch.float64)
j0 = ell_integ_j(_kappa_nfw_dl, 0, bx_t, by_t, q_t, 1e-10).item()
j1 = ell_integ_j(_kappa_nfw_dl, 1, bx_t, by_t, q_t, 1e-10).item()
print(f"mine  j0 = {j0:.15e}")
print(f"mine  j1 = {j1:.15e}")

# Now let's also compute j0, j1 through glafic.  Since we can't access them
# directly, compute the full ax/ay from glafic and invert.
# We already know glafic gives ax=3.535050120, ay=0.281754100 at (10,0).
# From my derivation: ay = bb*tt*q*(bx*j1*sin(pa) + by*j0*cos(pa)),
# ax = bb*tt*q*(bx*j1*cos(pa) - by*j0*sin(pa))  (pa in deg, tt=tt_ell)
# So two equations with two unknowns (j0, j1).
ax_g = 3.535050120178563
ay_g = 0.2817540995560170

cos_pa = math.cos(pa * math.pi / 180.0)
sin_pa = math.sin(pa * math.pi / 180.0)

# ax = bb*tt*q*(bx*j1*cos_pa + (-by)*j0*sin_pa) -- wait we need to use matrix from
# the actual rotation conventions.  Easier: derive directly via ell_pxpy.
# bpx = q*bx*j1,  bpy = q*by*j0.
# px = bpx*co + bpy*si,  py = -bpx*si + bpy*co, with co=cos(-pa), si=sin(-pa).
# so ax = bb*tt*(bpx*co + bpy*si)
#    ay = bb*tt*(-bpx*si + bpy*co)
# solve for (bpx, bpy):
# [co  si] [bpx]   [ax/(bb*tt)]
# [-si co] [bpy] = [ay/(bb*tt)]
rhs_x = ax_g / (bb * tt)
rhs_y = ay_g / (bb * tt)
det = co * co + si * si  # = 1
bpx_from_glafic = (co * rhs_x - si * rhs_y)
bpy_from_glafic = (si * rhs_x + co * rhs_y)
print(f"bpx (glafic) = {bpx_from_glafic:.15e}")
print(f"bpy (glafic) = {bpy_from_glafic:.15e}")
# j1 = bpx / (q*bx); j0 = bpy / (q*by)
j1_glafic = bpx_from_glafic / (q * bx)
j0_glafic = bpy_from_glafic / (q * by)
print(f"j0 (glafic) = {j0_glafic:.15e}")
print(f"j1 (glafic) = {j1_glafic:.15e}")
