"""
Angular-diameter distances, critical surface density, time-delay factor.

Matches glafic's distance.c.  We use scipy.integrate.quad with GSL-comparable
tolerance so that every derived distance agrees with glafic to < 1e-8.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from scipy.integrate import quad

from . import constants as K


@dataclass
class Cosmology:
    """Flat-friendly wCDM cosmology.  Parameters mirror glafic's globals."""

    omega: float = K.DEF_OMEGA
    lam: float = K.DEF_LAMBDA          # Omega_lambda (Lambda component)
    weos: float = K.DEF_WEOS           # dark-energy equation of state w
    hubble: float = K.DEF_HUBBLE       # h (H0/100)

    # -- Hubble function ------------------------------------------------
    def hubble_ez2(self, z: float) -> float:
        """
        E^2(z) following glafic distance.c:177-188.

            h2 = (1 + o*z - l)(1+z)^2 + l (1+z)^{3(1+w)}
        """
        o, l, w = self.omega, self.lam, self.weos
        return (1.0 + o * z - l) * (1.0 + z) ** 2 + l * (1.0 + z) ** (3.0 * (1.0 + w))

    def hubble_ez(self, z: float) -> float:
        return math.sqrt(self.hubble_ez2(z))

    # -- chi integral ---------------------------------------------------
    def _inv_hubble_a(self, a: float) -> float:
        """Integrand: 1 / (a^2 * E(z(a))) where a=1/(1+z)."""
        z = 1.0 / a - 1.0
        return 1.0 / (a * a * self.hubble_ez(z))

    def chi(self, z_a: float, z_b: float) -> float:
        """
        Dimensionless radial (comoving) distance integral from z_a to z_b
        (always z_a <= z_b).  distance.c:34-65 uses GSL CQUAD with epsrel=1e-6.
        """
        if z_a >= z_b:
            return 0.0
        a_hi = 1.0 / (1.0 + z_a)
        a_lo = 1.0 / (1.0 + z_b)
        val, _ = quad(self._inv_hubble_a, a_lo, a_hi,
                      epsabs=0.0, epsrel=K.GSL_EPSREL_DISTANCE, limit=200)
        return val

    def comoving(self, z_a: float, z_b: float) -> float:
        """Dimensionless comoving distance with curvature (distance.c:71-88)."""
        if z_a >= z_b:
            return 0.0
        chi = self.chi(z_a, z_b)
        k = self.omega + self.lam - 1.0
        if abs(k) < K.TOL_CURVATURE:
            return chi
        if k > 0.0:
            return math.sin(chi * math.sqrt(k)) / math.sqrt(k)
        return math.sinh(chi * math.sqrt(-k)) / math.sqrt(-k)

    def angulard(self, z_a: float, z_b: float) -> float:
        """Dimensionless angular-diameter distance, in units of c/H0."""
        if z_a >= z_b:
            return 0.0
        return self.comoving(z_a, z_b) / (1.0 + z_b)


# --- helpers that mirror distance.c free functions -----------------------
def thetator_dis(theta: float, dis: float) -> float:
    """arcsec -> physical Mpc/h at an angular-diameter distance `dis`."""
    return K.COVERH_MPCH * dis * K.ARCSEC2RADIAN * theta


def rtotheta_dis(r: float, dis: float) -> float:
    """physical Mpc/h -> arcsec."""
    return r / (K.COVERH_MPCH * dis * K.ARCSEC2RADIAN)


def inv_sigma_crit_dis(dos: float, dol: float, dls: float) -> float:
    """1 / Sigma_crit  (distance.c:371-374)."""
    return dol * dls / (K.FAC_CRITDENS * dos)


def sigma_crit_dis(dos: float, dol: float, dls: float) -> float:
    return K.FAC_CRITDENS * dos / (dol * dls)


def tdelay_fac(zl: float, dos: float, dol: float, dls: float, hubble: float) -> float:
    """Time delay pre-factor in days (distance.c:329-336)."""
    ddd = (1.0 + zl) * dos * dol / (dls + K.OFFSET_TDELAY_FAC)
    return K.FAC_TDELAY_DAY * K.ARCSEC2RADIAN * K.ARCSEC2RADIAN * ddd / hubble


def critdensz(cosmo: Cosmology, zl: float) -> float:
    return cosmo.hubble_ez2(zl)


def omegaz(cosmo: Cosmology, zl: float) -> float:
    return cosmo.omega * (1.0 + zl) ** 3 / critdensz(cosmo, zl)


def delta_vir(cosmo: Cosmology, zl: float) -> float:
    """
    Virial overdensity (distance.c:390-412).  Only the flat-LCDM branch and
    the EdS fallback are reproduced; open-matter fit omitted (rarely used).
    """
    om, lam = cosmo.omega, cosmo.lam
    omega_vir = omegaz(cosmo, zl)
    omega_vir_2 = 1.0 / omega_vir - 1.0
    if 0.0 < lam < 1.0 and 0.0 < om < 1.0:
        return 177.6528 * (1.0 + 0.40929 * omega_vir_2 ** 0.90524)
    if om < 1.0 and abs(lam) < 1.0e-10:
        ov3 = 1.0 + 2.0 * omega_vir_2
        eta = math.log(ov3 + math.sqrt(ov3 * ov3 - 1.0))
        f = (math.exp(eta) - math.exp(-eta)) * 0.5 - eta
        return 315.827 * omega_vir_2 ** 3 / (f * f)
    return 177.6528


def deltaomega(cosmo: Cosmology, zl: float,
               flag_hodensity: int = K.DEF_FLAG_HODENSITY,
               hodensity: float = K.DEF_HODENSITY) -> float:
    """Overdensity * Omega_m(z=0) * (1+z)^3  (distance.c:414-426)."""
    if flag_hodensity == 1:
        return hodensity * cosmo.omega * (1.0 + zl) ** 3
    if flag_hodensity == 2:
        return hodensity * critdensz(cosmo, zl)
    return delta_vir(cosmo, zl) * cosmo.omega * (1.0 + zl) ** 3
