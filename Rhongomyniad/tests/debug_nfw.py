"""Pinpoint NFW error by comparing intermediate values to glafic."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import rhongomyniad as rh
from rhongomyniad.cosmology import Cosmology
from rhongomyniad.lens_models import (
    LensContext, _calc_bbtt_nfw, _kappa_nfw_dl, _dkappa_nfw_dl,
    _dphi_nfw_dl,
)
from rhongomyniad.elliptical import ell_integ_j, ell_integ_k

# Mirror the scenario.
cosmo = Cosmology(0.3, 0.7, -1.0, 0.7)
ctx = LensContext.build(cosmo, zl=0.3, zs=1.5)
print(f"dis_ol={ctx.dis_ol:.12e}")
print(f"dis_os={ctx.dis_os:.12e}")
print(f"dis_ls={ctx.dis_ls:.12e}")
print(f"delome={ctx.delome:.12e}")

m, c = 1.0e14, 5.0
bb, tt = _calc_bbtt_nfw(m, c, ctx)
print(f"bb={bb:.12e}")
print(f"tt={tt:.12e}")

# Probe point (x=10, y=0) with pa=30deg, e=0.2, q=0.8
pa = 30.0
e = 0.2
q = 1.0 - e
tx, ty = 10.0, 0.0
si = math.sin(-pa * math.pi / 180.0)
co = math.cos(-pa * math.pi / 180.0)
tt_ell = tt / math.sqrt(q)
bx = (co * tx - si * ty) / tt_ell
by = (si * tx + co * ty) / tt_ell
print(f"bx={bx:.12e}, by={by:.12e}")

# Evaluate j0, j1 at this point.
bx_t = torch.tensor([bx], dtype=torch.float64)
by_t = torch.tensor([by], dtype=torch.float64)
q_t = torch.tensor(q, dtype=torch.float64)
smallcore = 1e-10

j0 = ell_integ_j(_kappa_nfw_dl, 0, bx_t, by_t, q_t, smallcore).item()
j1 = ell_integ_j(_kappa_nfw_dl, 1, bx_t, by_t, q_t, smallcore).item()
print(f"j0={j0:.15e}")
print(f"j1={j1:.15e}")

# Also evaluate kappa at some points
xs_test = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
print("kappa_nfw_dl:", _kappa_nfw_dl(xs_test).tolist())
print("dkappa_nfw_dl:", _dkappa_nfw_dl(xs_test).tolist())
print("dphi_nfw_dl:", _dphi_nfw_dl(xs_test).tolist())
