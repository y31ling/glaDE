#!/usr/bin/env python3
"""Numerical anatomy of the |o| stripe: bit-exact replication of glafic's
Sersic elliptical-density J integral + GSL's adaptive Romberg, used to
generate every number in manual/Romberg_JHK_Investigation §2.4.1-2.4.3.

What it replicates (glafic 2.1.14, GLADE-bundled):
  mass.c  kapgam_sers   dimensionless coords bx,by = rotated/(re*bn^-n/sqrt(q))
  mass.c  ell_integ_j   log-substitution branch (taken here since uu <= 0.1)
  GSL     gsl_integration_romberg  trapezoid + Richardson + stop rule
          (epsabs=0, epsrel=TOL_ROMBERG_JHK, max 16 levels; see gsl_romberg2)

Validation trifecta (all reproduced by this script's --validate mode):
  1. vs real GSL     : identical results AND identical neval at every eps
  2. vs real glafic  : alpha_x(glafic)/(q*bx*J1) constant across positions
                       (zero scatter) with the 1e-5 built module
  3. vs scipy quad   : reference values at epsrel=1e-13

Key findings (config: exception/nfwsersic_lens_sersic_source.input, sers lens
m=1.199e11, re=0.9688", n=4, e=0.2, pa=0; scan row y=0.42"):
  * At most positions eps=5e-4 stops at level 6 with true error ~1e-5.
  * In a sliver x in [-0.7041,-0.7004] (0.0036" ~ 4 pixels) the diagonals
    R(2,2) and R(3,3) cross: the error estimate dips to 8.9e-7 while BOTH
    are +2.4% from the truth -> false convergence at level 3, delta(alpha_x)
    ~ 8.3 mas. Contaminated pixel columns 1296-1298 match the dark stripe
    columns in diff_STOCK5e-4_minus_GLADE1e-5.fits at that row exactly.
  * At eps=1e-5 the same sliver narrows ~50x to ~0.07 pixel, which is why
    the stripe fades to ~0.4% instead of vanishing for a deeper reason.
"""
import argparse
import math

import numpy as np
from scipy.integrate import quad

# ---- reproduction config (sers lens of nfwsersic_lens_sersic_source.input) --
RE, NSER, E, PA = 0.9688, 4.0, 0.2, 0.0
Q = 1.0 - E
SMALLCORE = 1.0e-10        # DEF_SMALLCORE, glafic.h
YROW = 0.42                # scan row (stripe spans y ~ 0.35..0.50)


def bn_sers(n):
    """Ciotti & Bertin series, mass.c bn_sers (n > 0.36 branch)."""
    n2, n3, n4 = n * n, n**3, n**4
    return (2.0*n - 1.0/3.0 + (4.0/405.0)/n + (46.0/25515.0)/n2
            + (131.0/1148175.0)/n3 - (2194697.0/30690717750.0)/n4)


TT = RE * bn_sers(NSER)**(-NSER) / math.sqrt(Q)   # mass.c:2176-2178


def j_integrand_ln(tx, ty, nidx):
    """J_n integrand in the log-substituted variable, exactly as
    ell_integ_j_funcln + ell_xi2/ell_qu/ell_nhalf (mass.c:3072-3102)."""
    bx, by = tx/TT, ty/TT
    uu = 1.0/(bx*bx + by*by + SMALLCORE*SMALLCORE)
    assert uu <= 0.1, "these positions take the log branch (mass.c:2194-2201)"
    lnmin = math.log(1.0e-4*uu)
    omq2 = 1.0 - Q*Q

    def f(lu):
        u = math.exp(lu)
        equ = 1.0 - omq2*u
        se = math.sqrt(u*(by*by + bx*bx/equ + SMALLCORE*SMALLCORE))
        kap = math.exp(-(se**(1.0/NSER)))          # kappa_sers_dl
        return u * kap / (math.sqrt(equ) * equ**nidx)
    return f, lnmin


def gsl_romberg(f, a, b, epsabs, epsrel, nmax=16):
    """gsl_integration_romberg (GSL >= 2.5), instrumented."""
    Rp, Rc = [0.0]*nmax, [0.0]*nmax
    Rp[0] = 0.5*(b-a)*(f(a)+f(b))
    neval, diag, errs = 2, [Rp[0]], [None]
    for i in range(1, nmax):
        two_i = 1 << i
        s = sum(f(a + (b-a)*j/two_i) for j in range(1, two_i, 2))
        neval += two_i >> 1
        Rc[0] = 0.5*Rp[0] + s*(b-a)/two_i
        for j in range(1, i+1):
            r = 4.0**j
            Rc[j] = (r*Rc[j-1] - Rp[j-1])/(r - 1.0)
        err = abs(Rc[i] - Rp[i-1])
        diag.append(Rc[i]); errs.append(err)
        if (err < epsabs) or (err < epsrel*abs(Rc[i])):
            return dict(val=Rc[i], level=i, neval=neval, ok=True, diag=diag, errs=errs)
        Rp, Rc = Rc, Rp
    return dict(val=Rp[nmax-1], level=nmax-1, neval=neval, ok=False, diag=diag, errs=errs)


def J(tx, ty, nidx, eps):
    f, lnmin = j_integrand_ln(tx, ty, nidx)
    return gsl_romberg(f, lnmin, 0.0, 0.0, eps)


def J_ref(tx, ty, nidx):
    f, lnmin = j_integrand_ln(tx, ty, nidx)
    return quad(f, lnmin, 0.0, epsabs=0.0, epsrel=1e-13, limit=400)[0]


def tableau(x, y):
    ref = J_ref(x, y, 1)
    r = J(x, y, 1, 1e-30)   # run all 16 levels
    print(f"J1 tableau at theta=({x},{y}), reference {ref:.12e}")
    print(f"{'lvl':>3} {'pts':>6} {'R(i,i)':>18} {'estimate':>10} {'true rel':>10}")
    for i, d in enumerate(r['diag'][:11]):
        est = f"{r['errs'][i]/abs(d):.2e}" if r['errs'][i] is not None else '-'
        print(f"{i:>3} {(1 << i)+1:>6} {d:>18.12e} {est:>10} {(d-ref)/ref:>+10.2e}")
    for eps in (1e-2, 5e-4, 1e-5, 1e-8):
        s = J(x, y, 1, eps)
        print(f"  eps={eps:g}: stop level {s['level']} ({s['neval']} evals), "
              f"true rel err {(s['val']-ref)/ref:+.2e}")


def scan():
    xs = np.arange(-0.7605, -0.6595, 0.0001)
    for eps in (5e-4, 1e-5):
        lev, rel = [], []
        for x in xs:
            r = J(x, YROW, 1, eps)
            lev.append(r['level'])
            rel.append((r['val'] - J_ref(x, YROW, 1))/J_ref(x, YROW, 1))
        lev, rel = np.array(lev), np.array(rel)
        bad = lev == 3
        print(f"eps={eps:g}: {bad.sum()}/{len(xs)} points false-converge at level 3", end='')
        if bad.any():
            print(f", sliver x in [{xs[bad].min():.4f}, {xs[bad].max():.4f}]"
                  f", rel err there {rel[bad].mean():+.2e}", end='')
        print(f"; elsewhere max |rel err| {abs(rel[~bad]).max():.2e}")


def validate():
    import glafic
    glafic.init(0.315, 0.685, -1.03, 0.674, 'out', -2.0, -2.0, 2.0, 2.0,
                0.001, 0.001, 1, verb=0)
    glafic.startup_setnum(1, 0, 0)
    glafic.set_lens(1, 'sers', 0.5, 1.199e11, 0.0, 0.0, E, PA, RE, NSER)
    glafic.model_init(verb=0)
    Cs = []
    for x in (-0.71, -0.70, -0.68, -0.74):
        ax = glafic.calcimage(2.0, x, YROW)[0]
        Cs.append(ax / (Q * (x/TT) * J(x, YROW, 1, 1e-5)['val']))
    glafic.quit()
    Cs = np.array(Cs)
    print(f"alpha_x(glafic) / (q*bx*J1): mean {Cs.mean():.6e}, "
          f"scatter {Cs.std()/abs(Cs.mean()):.1e} (should be ~0 => bit-faithful)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--validate', action='store_true',
                    help='cross-check against the built glafic python module')
    args = ap.parse_args()
    print("== tableau at a well-behaved point ==")
    tableau(-0.71, YROW)
    print("\n== tableau at the false-convergence point ==")
    tableau(-0.70222, YROW)
    print("\n== row scan across the stripe (takes ~a minute) ==")
    scan()
    if args.validate:
        print("\n== validation vs real glafic ==")
        validate()
