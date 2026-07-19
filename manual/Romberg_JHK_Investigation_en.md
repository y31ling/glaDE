# The Romberg_JHK Investigation
## From the principles of gravitational-lens computation to a dark stripe in glafic's extended images

**Date**: 2026-07-09
**Intended reader**: a lower-year physics undergraduate (high-school physics plus a rough impression of gravitational lensing is enough; every technical term is explained at first use)
**Code concerned**: glafic 2.1.14 (bundled by GLADE under `glafic2/`), GLADE (this repository)
**Headline conclusion**: the vertical dark stripes seen in glafic's extended-source lensed images originate in the **Romberg numerical-integration error of elliptical-density deflection angles**, amplified roughly **500–1000×** by the **finite differences** in `extend.c` and then rendered into the image. The error is governed by one tolerance constant, `TOL_ROMBERG_JHK`: upstream glafic ships `5e-4` (stripe depth ≈ 11.5%, visible to the naked eye); GLADE's bundled build tightens it to `1e-5` (stripe depth ≈ 0.38%, invisible).

---

## Contents

- Part 1: How a gravitational lens is actually *computed* (the principles)
- Part 2: How glafic performs each step (a guided walk through the code)
- Part 3: The stripe incident — reproduction and isolation experiments
- Part 4: The impact of 5e-4 (upstream default) versus 1e-5 (GLADE)
- Appendix A: Glossary
- Appendix B: Index of evidence files and key source locations

---

# Part 1: How a gravitational lens is actually computed

## 1.1 Light deflection and the three actors

General relativity says that mass curves spacetime and light follows the "straight lines" (geodesics) of that curved spacetime — so, seen from far away, light rays appear *bent* by massive objects. For a point mass M, a ray passing at impact parameter b (the closest distance between ray and object) is deflected by

```
α̂ = 4GM / (c² b)
```

— exactly twice the naive Newtonian estimate; the extra factor comes from the curvature of space itself. For galaxy-scale lenses this angle is tiny and is measured in **arcseconds** (arcsec, written ″): 1″ = 1/3600 of a degree. Every "length" in this report is really an **angle on the sky**, in arcsec.

A lens system has three actors, lined up in distance:

```
observer ●———————————— lens galaxy (z_l = 0.5) ———————————— background source galaxy (z_s = 2)
(telescope)             (the mass that bends light)           (the distant object being observed)
```

z is the redshift — loosely, a "distance label": larger z, farther away. Light from the background source is deflected as it passes the lens galaxy, so the observer sees shifted positions, distorted shapes, sometimes several images of one source, or a ring.

## 1.2 The two planes: image plane and source plane

This is the pair of concepts beginners most often confuse, and the most important one. Both are **two-dimensional angular coordinate systems on the sky** — not physical planes floating in space:

- **Image plane**: the sky your telescope actually sees. Coordinates **θ** = (θx, θy). Every pixel of an observed image corresponds to one θ.
- **Source plane**: the sky as it *would* look if the lens galaxy were removed — where the background source really "is". Coordinates **β** = (βx, βy). It is the un-distorted world.

The two are connected by the **lens equation**:

```
β = θ − α(θ)
```

Meaning: the light you see when looking in direction θ actually left the source at position β. Here α(θ) is the **reduced deflection angle** — the physical deflection α̂ multiplied by a distance ratio D_ls/D_os (lens-to-source distance over observer-to-source distance) so that everything in the equation is an angle in the same coordinate system.

One crucial asymmetry dictates the direction of every computation:

- Given θ, find β: **plug into the formula**, done. Each observed direction corresponds to exactly one source position.
- Given β, find θ: **an equation must be solved**, and it may have several solutions — this is precisely what "multiple images" means: light from one source point reaches the observer along several paths, appearing at several θ.

Numerical work therefore always runs "backwards": **start from the image plane and map each pixel back to the source plane** (called inverse ray-shooting). This is why the direction "image plane → source plane" pervades the code.

## 1.3 From mass distribution to deflection: lens potential, convergence

A galaxy is not a point mass. In the *thin-lens approximation* (the lens is thin compared with all the distances involved), squash the entire lens mass along the line of sight onto one plane, obtaining the **surface density** Σ(θ): mass per unit projected area.

Define the dimensionless **convergence**:

```
κ(θ) = Σ(θ) / Σ_cr ,    Σ_cr = c² D_os / (4πG D_ol D_ls)
```

Σ_cr is the **critical surface density**, a constant built from the three distances. Regions with κ ≳ 1 are "dense enough" to make multiple images — that is strong lensing.

Next define the **lens potential** ψ(θ): a two-dimensional scalar field on the image plane, playing the role of a "projected gravitational potential". It satisfies a 2-D Poisson equation (an equation of the shape "sum of second derivatives = source"; here ∇²ψ = ψxx + ψyy, the sum of the two second partials), and its **gradient** (the vector of first partial derivatives, pointing uphill) is exactly the deflection:

```
∇²ψ = 2κ ,      α = ∇ψ    i.e.  αx = ∂ψ/∂θx ,  αy = ∂ψ/∂θy
```

The computational chain is therefore: **mass model → κ(θ) → ψ or directly α(θ) → lens equation**. With several lens components (say a dark-matter halo plus a stellar component), potentials add linearly, so α, κ, etc. simply sum.

## 1.4 Magnification, Jacobian matrix, critical curves and caustics

A lens does not just move images; it reshapes and (de)amplifies them. Linearize the lens equation locally: near a point θ, a small displacement dθ maps to a small displacement dβ in the source plane, related by a 2×2 matrix — the **Jacobian matrix** A (the "matrix of first derivatives of a multivariable map", describing the local linear behaviour):

```
A = ∂β/∂θ = ⎡ 1−ψxx   −ψxy ⎤ = ⎡ 1−κ−γ1    −γ2   ⎤
            ⎣ −ψxy   1−ψyy ⎦   ⎣  −γ2    1−κ+γ1  ⎦
```

Here ψxx = ∂²ψ/∂θx² etc. are **second partial derivatives** of the potential; the matrix of second partials is called the **Hessian matrix** and describes local curvature. Physically they regroup into:

- **convergence** κ = (ψxx + ψyy)/2 : isotropic overall magnification;
- **shear** γ1 = (ψxx − ψyy)/2, γ2 = ψxy : anisotropic stretching that turns circles into ellipses.

**Magnification** is an area ratio: a small patch of source occupies μ times more solid angle (solid angle: the "angular area" a patch subtends on the sky — a two-dimensional version of an angle) in the image plane than in the source plane, with

```
μ = 1 / det A = 1 / [ (1−κ)² − (γ1² + γ2²) ]
```

Lensing obeys an extremely useful conservation law — **surface brightness conservation**: a lens neither creates nor destroys photons, and the brightness received per unit solid angle is unchanged. Hence:

> the pixel value at image-plane position θ = the source's own surface brightness at β(θ).

Images look "brighter" only because they cover more solid angle (μ × area at the same surface brightness = μ × total flux). This conservation law is the entire physical basis of the algorithm in §1.5.

**Critical curves**: the curves in the image plane where det A = 0, so formally μ → ∞. **Caustics**: the images of the critical curves mapped into the source plane through the lens equation. When the source crosses a caustic, the number of images jumps by ±2. When the source sits almost exactly behind the lens centre, its image is stretched into a ring — the **Einstein ring**.

The point that matters most for this report: **near a critical curve everything is hypersensitive.** There (1−κ)² ≈ γ² and det A ≈ 0 — a small quantity sitting in a denominator. Any tiny numerical error in κ or γ (i.e. in the second derivatives of ψ) is violently amplified in μ and in the local mapping. The stripes appear precisely on the bright ring (the high-magnification region). That is no coincidence.

## 1.5 Imaging an extended source = per-pixel inverse ray-shooting

An **extended source** is one with resolvable size (a galaxy), as opposed to a "point source" (quasar, supernova) treated as a point. Given a lens model and the source's surface-brightness distribution S(β), the algorithm for producing a simulated image just strings §§1.2–1.4 together:

```
for every pixel centre θ of the image plane:
    1. compute α(θ)                 ← mass-model integrals; the protagonist of this report
    2. β = θ − α(θ)                 ← lens equation
    3. pixel value = S(β)           ← surface-brightness conservation
```

The reproduced image is 4000×4000 pixels — sixteen million passes through this loop. For some models step 1 has no closed formula and must be integrated numerically. Everything that follows starts there.

---

# Part 2: How glafic performs each step

## 2.1 What glafic is

[glafic](https://github.com/oguri/glafic2) is Masamune Oguri's public gravitational-lensing software (written in C). It can simulate forwards (model → image) and fit inversely (data → model). GLADE bundles its version 2.1.14 sources under [glafic2/](../glafic2/) with a small number of local modifications (see Part 4). You drive it with a `.input` text file declaring cosmological parameters, lens components and source components, followed by commands such as `writeimage` (render the lensed extended image to a FITS file — astronomy's standard image format), `writecrit` (output critical curves and caustics), and `findimg` (solve for the multiple images of a point source).

## 2.2 From model to deflection: 27 lens models, three implementation routes

glafic ships 27 lens mass models (name table at [mass.c:10-39](../glafic2/mass.c#L10-L39)). The master entry point `lensmodel()` ([mass.c:69-95](../glafic2/mass.c#L69-L95)) sums the contributions of all components linearly (`lensmodel_sum`, [mass.c:354-367](../glafic2/mass.c#L354-L367) — ax, ay, κ, γ added term by term, the "potentials superpose" fact of §1.3). Two models matter here:

- **NFW profile** (Navarro–Frenk–White): the standard density profile of dark-matter halos in simulations, ρ(r) ∝ 1/[(r/r_s)(1+r/r_s)²], described by a mass M and a **concentration** c (how strongly the mass is packed toward the centre). Its circular version has closed-form κ and α ([mass.c:968-1001](../glafic2/mass.c#L968-L1001)).
- **Sersic profile**: the standard brightness/mass profile of a galaxy's stellar component. glafic uses it as a surface density:

  ```
  κ(ξ) ∝ exp[ −b_n (ξ/r_e)^(1/n) ]
  ```

  r_e is the **effective radius** (enclosing half the total light/mass) and n the **Sersic index**: n=4 is the classic de Vaucouleurs profile (elliptical galaxies, bulges), n=1 an exponential disc. b_n is the companion constant that makes r_e enclose exactly one half; the code uses the Ciotti & Bertin series b_n ≈ 2n − 1/3 + 4/(405n) + … ([mass.c:2065-2080](../glafic2/mass.c#L2065-L2080)). The dimensionless kernel is just a few lines ([mass.c:2118-2125](../glafic2/mass.c#L2118-L2125)):

  ```c
  double kappa_sers_dl(double x)
  {
    double xx;
    xx = pow(x, 1.0 / n_sers_sav);
    return exp((-1.0) * xx);      /* κ(x) = exp(−x^(1/n)) */
  }
  ```

For a circularly symmetric model, the deflection obeys a Gauss's-law-like one-dimensional formula (α(θ) depends only on the projected mass enclosed within θ) and is mostly analytic. **The real difficulty is ellipticity — real galaxies are elliptical** (the departure from a circle is measured by the **ellipticity** e = 1 − minor axis/major axis; e=0 is a circle). glafic offers three routes for "making a circular profile elliptical", and this choice is exactly the watershed of our case:

### Route 1: elliptical potential — cheap but slightly unphysical

Put the ellipticity into the **potential**: take the circular model's 1-D potential ψ_1D and simply replace its argument by an "elliptical radius"

```
u = √[ (1+ε)X² + (1−ε)Y² ]        (X, Y are coordinates rotated to the principal axes — the ellipse's major/minor axis directions; ε is the input ellipticity e)
```

i.e. set ψ(θ) = ψ_1D(u). Then α, κ, γ all follow analytically from 1-D functions by the chain rule — **no numerical integration**. The code does this in `u_calc` ([mass.c:3006-3034](../glafic2/mass.c#L3006-L3034), comment "for elliptical potential"), which returns u and its first and second partials u[0..5] in one go. Model names carry a `pot` suffix: `nfwpot`, `serspot`, `hernpot`, …

The price: **elliptical equipotentials ≠ elliptical isodensity contours.** The surface density implied by ∇²ψ=2κ becomes unphysically "dumbbell/peanut"-shaped at larger ellipticity. It is a fast approximation, not the physically natural choice.

### Route 2: elliptical density — physical but needs numerical integration

Put the ellipticity into the **density**: κ(θ) = κ_1D(ξ) with ξ the elliptical radius of the isodensity contours. Physically natural (a galaxy's isophotes — its contours of constant brightness — are roughly ellipses), but the potential and deflection then have **no closed form** — they must be integrated numerically. glafic uses the classic result of Schramm (1990) (see also Keeton 2001): the 2-D integral collapses into a **family of one-dimensional integrals** (§2.3). Eight models take this route: `nfw`, `hern`, `pow`†, `gnfw`, `sers`, `tnfw`, `ein`, `king` (the density versions, i.e. names **without** the pot suffix).

> † `pow` (power law) defaults to the analytic Tessore & Metcalf (2015) method (switch `flag_pow_tm15`, default 1, [glafic.h:198](../glafic2/glafic.h#L198)); it only falls into the numerical route if the user sets the flag to 0.

### Route 3: CSE analytic approximation — building an elliptical density out of solvable bricks

Oguri (2021, arXiv:2106.11464) expands profiles such as NFW into a linear combination of "cored steep ellipsoid" (CSE) basis functions, each of which has closed-form elliptical-density lensing quantities ([app_ell.c:314-357](../glafic2/app_ell.c#L314-L357)). Model names carry an `a` prefix: `anfw`, `ahern`, `acnfw`. **Elliptical density AND fully analytic — no numerical integration.** Remember this: the NFW in the reproduction input is `anfw`, which is why the NFW-only image comes out perfectly clean.

## 2.3 The Schramm integrals: the J, K, I family (where the name `TOL_ROMBERG_JHK` comes from)

The core mathematics of elliptical-density models is in glafic's manual §A.2 (Schramm's 1990 homoeoid decomposition; intuition: view the elliptical mass distribution as nested similar-ellipse "shells" — a "homoeoid" is exactly such a shell; the integration variable u ∈ (0,1] sweeps through the shells, each contributing gravity according to its local κ). With q = 1−e the **axis ratio** (minor over major axis) and (x, y) dimensionless principal-axis coordinates (sky angles rotated to the major/minor axis directions, then divided by the model's scale length — see the construction of bx, by in the code), define the u-dependent elliptical radius

```
ζ(u) = √[ u ( x²/(1−(1−q²)u) + y² ) ]
```

Then the potential and its derivatives are combinations of three kinds of 1-D integrals:

```
potential:      φ    = (q/2) · I                 I  = ∫₀¹ ζ·α_1D(ζ) / ( u·√(1−(1−q²)u) ) du
1st derivs:     φx   = q·x·J₁ ,   φy = q·y·J₀    Jₙ = ∫₀¹ κ(ζ) / (1−(1−q²)u)^(n+1/2) du
2nd derivs:     φxx  = 2q·x²·K₂ + q·J₁           Kₙ = ∫₀¹ u·κ′(ζ) / ( 2ζ·(1−(1−q²)u)^(n+1/2) ) du
                φyy  = 2q·y²·K₀ + q·J₀
                φxy  = 2q·x·y·K₁
```

In one sentence: **J integrals (integrand: κ) give the deflection; K integrals (integrand: the derivative κ′) give the second derivatives (hence κ and γ); the I integral (whose integrand contains α_1D, the circular model's one-dimensional deflection function) gives the potential itself.** The macro `TOL_ROMBERG_JHK` is the tolerance of this whole family. One clarification while we are here: there is no integral actually called "H" — the three functions are `ell_integ_i/j/k`; "JHK" is just the collective label.

Code and formulas correspond line by line ([mass.c:3040-3154](../glafic2/mass.c#L3040-L3154)):

```c
double ell_integ_j_func(double u)              /* integrand of Jₙ */
{
  equ = ell_qu(q_integ_sav, u);                /* equ = 1−(1−q²)u          */
  se  = sqrt(ell_xi2(u, equ));                 /* se  = ζ(u) (with smallcore softening at the centre) */
  f   = ell_nhalf(equ, n_integ_sav);           /* f   = equ^(n+1/2)        */
  return func_sav(se) / f;                     /* κ(ζ) / (1−(1−q²)u)^(n+1/2) */
}
```

`func_sav` is the plugged-in 1-D profile kernel — for Sersic it is that one-line `exp(−x^(1/n))`. Assembly, using `kapgam_sers` ([mass.c:2159-2227](../glafic2/mass.c#L2159-L2227)) as the example:

```c
j1 = ell_integ_j(kappa_sers_dl, 1);   bpx = q * bx * j1;    /* φx = q·x·J₁ → αx */
j0 = ell_integ_j(kappa_sers_dl, 0);   bpy = q * by * j0;    /* φy = q·y·J₀ → αy */
...
bpxx = 2.0*q*bx*bx * ell_integ_k(dkappa_sers_dl, 2) + q*j1; /* φxx straight from the K integral (numerical too, but no differencing) */
```

**The cost ledger**: evaluating one image-plane point takes 2 J integrals (deflection); add 3 K integrals if κ, γ are wanted; add 1 I integral if the potential is wanted (used for **time delays**: the several images of one source travel along different paths and arrive at different times; the differences can be used to measure cosmic expansion). Each is one adaptive Romberg run (§2.4).

One more detail we will need later: the integral drivers choose between a **linear or logarithmic substitution** depending on position ([mass.c:2194-2201](../glafic2/mass.c#L2194-L2201)) — far from the lens centre (criterion: the inverse squared dimensionless circular radius `uu = 1/(x²+y²) ≤ 0.1`) the integrand concentrates near u≈0 and the code switches to ln u as integration variable. This is an **algorithmic branch that flips abruptly with position**. Remember it.

## 2.4 Romberg integration: adaptivity, the tolerance, and the ignored convergence status

The J, K, I integrals have no closed form; glafic evaluates them with the **Romberg method**. For readers who have not met it, three steps:

1. **Trapezoid rule**: chop the interval into equal pieces and sum trapezoid areas. At step size h the error expands in powers of h²: error ≈ C₂h² + C₄h⁴ + ⋯
2. **Halve the step + Richardson extrapolation**: recompute with h/2. Both results have a known h²-shaped leading error, so a linear combination cancels it **exactly**: R = (4·T(h/2) − T(h))/3. Repeat halving and extrapolating to fill a triangular table R(i,j); for smooth integrands the diagonal R(i,i) converges extremely fast.
3. **Adaptive stopping**: after each new level, compare neighbouring diagonal entries; if the change is "small enough", stop.

glafic implements this via GSL (the GNU Scientific Library). The stopping rule inside GSL's romberg.c:

```c
err = fabs(Rc[i] - Rp[i - 1]);                    /* |R(i,i) − R(i−1,i−1)| */
if ((err < epsabs) || (err < epsrel * fabs(Rc[i])))
    return GSL_SUCCESS;                            /* declared converged; return R(i,i) */
```

glafic's wrapper `gsl_romberg2` ([gsl_integration.c:113-140](../glafic2/gsl_integration.c#L113-L140)) passes `epsabs = 0.0`, `epsrel = TOL_ROMBERG_JHK` — a purely **relative tolerance**:

> **Precise meaning of `TOL_ROMBERG_JHK`: stop refining, and accept the current value, once the relative change of the Romberg diagonal drops below it.** Upstream glafic sets 5.0e-4; GLADE's bundled build sets 1.0e-5 ([glafic.h:375](../glafic2/glafic.h#L375)).

Three hazards are buried in this machinery:

**(a) The "error" is an estimate, not a bound.** `err` is merely the difference of two successive levels — a heuristic proxy. It is occasionally optimistic: two levels agreeing by chance does not mean both are near the truth.

**(b) The error is a *staircase* function of position — the seed of the stripes.** As the pixel θ moves continuously, the integrand deforms continuously, but "how many refinement levels are needed to pass" is an **integer** — it can only jump. Picture pixel A converging at level 6 while its neighbour B needs level 7: the two returned values suddenly differ in accuracy, and the deflection field α(θ) carries a **step** of order `TOL × |α|` between A and B. The loci where the level count switches form curves across the image plane; the linear/log substitution branch at the end of §2.3 contributes boundaries of the same kind. The true α field is smooth; the numerical one is covered in invisible "terraces".

**(c) Non-convergence is silently accepted.** The workspace allows at most `GSL_ROMBERG_N = 16` levels ([glafic.h:369](../glafic2/glafic.h#L369)); if the tolerance is still unmet, GSL returns the error code `GSL_EMAXITER` — but glafic has the check commented out ([gsl_integration.c:130-135](../glafic2/gsl_integration.c#L130-L135)):

```c
gslstatus = gsl_integration_romberg(&F, a, b, 0.0, eps, &integral, &neval, workspace);
/* if (gslstatus != GSL_SUCCESS){
   fprintf(stderr, "integration failed in gsl_romberg2\n");
   exit(EXIT_FAILURE);
   } */
```

An under-converged result is accepted without a word — one of the nastiest sources of position-dependent error.

## 2.5 The extended-image pipeline `extend.c`: two stages, and an error amplifier

Now watch `writeimage` assemble an image from these ingredients. The pipeline has two stages.

**Stage A: tile the grid with deflections.** `ext_set_table` ([extend.c:42-147](../glafic2/extend.c#L42-L147)) calls `lensmodel(tx, ty, pout, 1, 0)` once per pixel centre — note the 4th argument `alponly=1`: **deflection only, no second derivatives** — and caches the result:

```c
array_ext_def[k]      = pout[0];    /* αx — stored as float (single precision!) */
array_ext_def[k + nn] = pout[1];    /* αy */
```

For elliptical-density models every α here comes out of the Romberg J integrals of §§2.3–2.4, each carrying a relative error of order TOL and the "terrace" structure. (Storing as single-precision float adds a further ~1e-7 relative noise floor — far below 5e-4, so not the culprit here.)

**Stage A, second half: the finite-difference Hessian — enter the amplifier.** Rendering needs ψxx, ψxy, ψyy at every pixel (for magnification and the sub-pixel mapping, Stage B). glafic does **not** use the model's own K-integral second derivatives (§2.3 has them for exactly this! `alponly=1` skipped them) — instead it takes **central differences** (approximating a derivative by the difference of neighbouring values divided by their separation) of the freshly cached α grid ([extend.c:135-142](../glafic2/extend.c#L135-L142)):

```c
/* phi_xx */  array_ext_mag[k]          = (axxp - axxm) / ddx;   /* [αx(x+h)−αx(x−h)] / 2h */
/* phi_xy */  array_ext_mag[k + nn]     = (ayxp - ayxm) / ddy;
/* phi_yx */  array_ext_mag[k + 2 * nn] = (axyp - axym) / ddx;
/* phi_yy */  array_ext_mag[k + 3 * nn] = (ayyp - ayym) / ddy;
```

h = `pix_ext` is the pixel scale, 0.001″ here, so the denominator is 2h = 0.002″. **Dividing the difference of two nearly equal numbers by a small number is the classic danger zone of numerical computing**: the genuine signal in the numerator (the true change of α) is about 2h·ψxx ~ 0.002 (near-critical κ and γ are of order unity — see §1.4 — so ψxx ~ 1), while each endpoint carries an integration error of order TOL·|α|.

Plug in this case's numbers. Note that the relative tolerance acts only on the **component that actually goes through the Romberg integrals**: at the stripe the anfw part is closed-form and carries no integration error; the error-carrying part is the Sersic component, whose deflection is |α_sers| ≈ 0.39″ (measured with the bundled calcimage at the stripe location; both components together give |α| ≈ 0.86″):

```
TOL = 5e-4:  step jump in α    δα ~ 5e-4 × 0.39″ ≈ 2×10⁻⁴ ″
             impact on ψxx     ~ δα / 2h = 2×10⁻⁴ / 0.002 ≈ 0.1   ← dimensionless: not negligible against
                                                                     the ~order-unity second derivatives!
TOL = 1e-5:  same arithmetic → 0.002                              ← a two-per-mille perturbation
```

The amplification factor is 1/(2h) = 500 — and when the two difference endpoints happen to carry errors of opposite sign (adjacent columns sharing the same contaminated sample; see the paired dark/bright-line mechanism in §2.6), the error in the numerator doubles, an effective 1/h = 1000. That is the origin of the "~500–1000× amplification" in the reproduction report. Recall §1.4: the bright ring where the stripe sits is near the critical curve, det A = (1−ψxx)(1−ψyy) − ψxy² ≈ 0 — a spurious 0.1 in ψxx completely rewrites the local magnification and mapping.

> An aside (a boundary-only quirk, unrelated to the stripes): the `phi_xy` line takes its points along x yet divides by `ddy`, and the `phi_yx` line takes its points along y yet divides by `ddx`. For interior pixels ddx = ddy = 2h so nothing changes; they differ only at image borders where one neighbour is missing.

**Stage B: apply the lens equation per pixel and render.** `ext_set_image` ([extend.c:191-210](../glafic2/extend.c#L191-L210)):

```c
sx = x - array_ext_def[k]      * dis_fac_ext[ii];   /* βx = θx − αx (dis_fac rescales α to the source's redshift) */
sy = y - array_ext_def[k + nn] * dis_fac_ext[ii];   /* βy = θy − αy */
...
array_ext_img[k + ii*nn] = (float)sourcemodel(sx, sy, ii, pxx, pxy, pyx, pyy, pix_psf);
```

`dis_fac_ext` is a distance ratio (α ∝ D_ls/D_os depends on the source redshift; the cache is computed at a fiducial — i.e. reference — redshift and rescaled). `sourcemodel` lands in `source_all` ([source.c:236-282](../glafic2/source.c#L236-L282)), which uses the **finite-differenced** pxx…pyy for two things:

```c
muinv = fabs((1.0 - pxx) * (1.0 - pyy) - pxy * pyx + imag_ceil);   /* 1/μ — decides whether to refine */
...
dsx = (1.0 - pxx) * dx - pxy * dy;    /* sub-pixel offset (dx,dy) mapped to the source plane by the local linear map */
dsy = (1.0 - pyy) * dy - pyx * dx;
f  += source_sersic(x + dsx, y + dsy, ...) * hh * hh;   /* average over 5×5 sub-samples (20×20 near the source centre) */
```

This is **sub-pixel anti-aliasing**: a pixel is not an infinitesimal point but a small square; at high magnification its "footprint" in the source plane covers a large patch of the source, and sampling only the centre produces jagged artifacts — so the pixel is subdivided into 5×5 sub-points, each mapped to the source plane through the local linear map (the Jacobian A), and the brightnesses averaged. **The footprint's shape is entirely determined by pxx…pyy** — so a corrupted Hessian miscomputes the footprint and the pixel averages to a systematically low (or high) brightness. The error appears column-wise along the "Romberg level-switch boundary" and is rendered as a dark line cutting the bright ring. Adjacent columns also share the same contaminated α sample with opposite signs in the difference — a dark line often has a bright companion.

(`writeimage_ori` versus `writeimage` is the same function under a flag: the ori variant sets sx=x, sy=y — "no lensing" — painting the source on the same grid as a control.)

## 2.6 Why *vertical* lines, and why the `|o|` shape?

Assemble the clues:

- The error-step boundaries are curves in the image plane. Their exact shape is set by "how many refinement levels the integral needs at this point", which varies with position in a way that has no simple analytic description (it is not simply some elliptical isocontour); empirically, in the difference image `diff_STOCK5e-4_minus_GLADE1e-5.fits`, the boundary at the incident location is very nearly vertical, drifting by barely a column over its 150-pixel height. Both lenses here have **position angle** pa=0 (position angle: the orientation of the major axis on the sky; pa=0 means the principal axes are aligned with the coordinate axes) and share the same centre at the origin (0,0) — the whole system is mirror-symmetric, so error boundaries come in left-right pairs.
- Of the four difference lines, the two that take their points along x (ψxx, ψxy) straddle a near-vertical boundary and spike; the two that take their points along y (ψyx, ψyy) keep both endpoints on the same side and the step cancels. So the spikes pile up in whole columns — a **vertical line segment**.
- Only the portion superposed on the **bright ring** (high μ, near-critical) is visible: on a dark background there is nothing for even a large relative error to darken.

Result: one short vertical dark line on each side, plus the Einstein ring between them = the `|o|` pattern the user saw in ds9 (the standard astronomical FITS viewer).

---

# Part 3: The stripe incident — reproduction and isolation experiments

## 3.1 The incident

While inspecting a glafic-generated NFW+Sersic extended-source lensed image (4000×4000) in ds9, the user noticed **a mirror-symmetric pair of vertical black lines** forming an `|o|` pattern with the bright ring (the one that caught the eye sits at column x≈1294, spanning y≈2350–2500). In angular coordinates: θx ≈ −0.71″, θy ≈ +0.35 to +0.50″ — right on the ring of radius ≈ 0.8″. No such linear feature belongs in this image — it is an artifact, and an investigation was opened.

## 3.2 Reproduction setup

Input file [exception/nfwsersic_lens_sersic_source.input](../exception/nfwsersic_lens_sersic_source.input), engine glafic 2.1.14. Line by line:

```
omega 0.315 / lambda 0.685 / weos -1.03 / hubble 0.674   ← Planck18 cosmology
xmin -2  xmax 2  ymin -2  ymax 2                          ← field of view ±2″
pix_ext 0.001 (pix_poi likewise)                          ← 0.001″ pixels → 4000×4000
startup 2 1 0                                             ← 2 lens components, 1 extended source, 0 point sources

lens anfw 0.5 3.378e13 0 0 0.2 0 4 0
     ↑ analytic CSE NFW: z_l=0.5, mass 3.378×10¹³, centre (0,0), ellipticity e=0.2, pa=0°, concentration c=4
lens sers 0.5 1.199e11 0 0 0.2 0 0.9688 4 0
     ↑ elliptical-density Sersic: z_l=0.5, mass 1.199×10¹¹, centre (0,0), e=0.2, pa=0°, r_e=0.9688″, n=4
extend sersic 2 1 0.02 0.02 0.4 0 0.1609 4
     ↑ source: Sersic light profile, z_s=2, amplitude 1, position (0.02″,0.02″), e=0.4, pa=0°, r_e=0.1609″, n=4

writeimage 0 0        ← render the lensed image (no noise, no sky)
writeimage_ori 0 0    ← render the unlensed source (control)
writecrit 2           ← output critical curves / caustics
```

Note the role assignment: `anfw` takes Route 3 of §2.2 (analytic, no Romberg) while `sers` takes Route 2 (elliptical density, **2 Romberg J integrals per pixel**). The NFW outweighs the Sersic by two orders of magnitude in mass, but the n=4 Sersic is far more centrally concentrated: at the ring the two contribute comparably to the **deflection** (measured ≈ 0.47″ vs 0.39″), while the **convergence** is NFW-dominated (roughly 8:2) — the ring's high magnification is mostly the NFW's doing, and the Sersic supplies the Romberg-error-carrying share of the deflection.

## 3.3 Tolerance sweep and component isolation

Method: edit `TOL_ROMBERG_JHK` at [glafic.h:375](../glafic2/glafic.h#L375), recompile, rerun the same input; plus isolation runs keeping only one lens component. Results (reproduction report [exception/stripe_repro/README.md](../exception/stripe_repro/README.md)):

| Run | Stripe column x | Relative depth | By eye |
|---|---|---|---|
| TOL = 1e-2 (deliberately loose) | multiple | — | strong, multiple stripes |
| **TOL = 5e-4 (upstream default)** | **1295** | **11.5%** | **clear black line (= what the user saw)** |
| **TOL = 1e-5 (GLADE)** | 1294 | **0.38%** | invisible |
| TOL = 1e-8 (reference baseline†) | — | — | clean |
| anfw only (analytic) | none | ~0 | perfectly clean |
| sers only | same position | present, but whole image ~50× fainter | barely perceptible |

("Relative depth" = the deficit at the stripe relative to the local ring brightness. Subtracting the two builds — `diff_STOCK5e-4_minus_GLADE1e-5.fits` — isolates the stripe cleanly.
† The 1e-8 reference image `lensed_both_1e-8_REFERENCE.fits` exists in the reproduction directory and serves as the "no-stripe baseline"; the reproduction README's own results table does not list this row separately.)

Three decisive inferences:

1. **The tolerance controls the stripe monotonically** (1e-2: several → 5e-4: one at 11.5% → 1e-5: down to 0.38%, with the 1e-8 reference image as the clean baseline): the lesion lives in the integrals governed by `TOL_ROMBERG_JHK` and nowhere else.
2. **anfw alone is perfectly clean**: the analytic route has no Romberg — counter-proof that the lesion is the numerical integration itself, not the extend renderer or the source model.
3. **sers alone shows the stripe at the same position but faintly**: the stripe's *shape* is set by the Sersic's Romberg error; its *visibility* is provided by the NFW's bright, high-magnification ring. Only in combination does the error (Sersic's) meet the amplifier and the spotlight (the near-critical ring dominated by the NFW, plus the finite differences), making the artifact bloom.

## 3.4 The causal chain (summary)

```
① Seed      Sersic elliptical deflection = Romberg J integrals (relative tolerance TOL_ROMBERG_JHK)
            adaptive stopping → the α field carries position-dependent "staircase" error, δα ~ TOL·|α_sers|
            (under-converged results are also accepted silently: the gslstatus check is commented out)
                        ↓
② Amplify   extend.c derives the Hessian from the α grid by central differences (÷ 2·pix_ext = ÷ 0.002″)
            staircase step → spike in ψxx, amplified ~500–1000×; at 5e-4 the spike is ~0.1 (not negligible near-critical)
                        ↓
③ Develop   source.c's sub-pixel anti-aliasing uses that Hessian for the pixel footprint and refinement test
            near-critical (det A≈0) footprints are computed wholly wrong → whole columns systematically dimmed
                        ↓
            a vertical dark line on the bright ring (`|o|`), 11.5% deep @ 5e-4 / 0.38% @ 1e-5
```

## 3.5 Evidence files

Reproduction package [exception/stripe_repro/](../exception/stripe_repro/): six lensed-image FITS files (STOCK 5e-4 / GLADE 1e-5 / 1e-2 / 1e-8 reference / Sersic-only / NFW-only) plus the unlensed source and the difference image, and four analysis PNGs (full-resolution close-up at the user's location, tolerance sweep, ring segment in linear stretch, log-stretch comparison). All 4000×4000 float32, matching the user's original observation pixel for pixel.

---

# Part 4: The impact of 5e-4 (upstream default) versus 1e-5 (GLADE)

## 4.1 Who is affected: models and features in the blast radius

`TOL_ROMBERG_JHK` appears at exactly 6 call sites ([mass.c:3046-3111](../glafic2/mass.c#L3046-L3111)), all inside the J/K/I integral drivers. Hence:

- **Affected**: everything the 8 elliptical-density models `nfw / hern / gnfw / sers / tnfw / ein / king / pow (only with flag_pow_tm15=0)` produce — deflections, κ, γ, potentials, magnifications, the image positions and time delays derived from them, and extended-image rendering. There is **no** "skip the integral when ellipticity is zero" shortcut: even at e=0 these models run the Romberg machinery (for speed, explicitly choose the `*pot` or `anfw`-family variants instead).
- **Not affected**: the `*pot` elliptical-potential family, the CSE family (anfw/ahern/acnfw), the analytic isothermal family (sie/jaffe), point masses, external shear/convergence terms. (The gnfw/ein inner radial integrals have their own separate tolerances, `TOL_ROMBERG_GNFW`=3e-4 / `TOL_ROMBERG_EIN`=1e-3, and are replaced by lookup tables by default.)

## 4.2 Impact on extended images: the stripes (this case's main line)

Quantified above: **5e-4 → an 11.5%-deep, naked-eye artifact line; 1e-5 → 0.38%, a ~30× reduction, below visibility.** What this means for science:

- High-resolution extended images rendered at the upstream default tolerance can carry **fake structure at the ten-percent level in near-critical regions** — enough to be misread as "substructure" or "accretion features", or to send someone on a wild chase through the observation pipeline.
- For pixel-level fitting of extended images (GLADE's FITS extended-source fitting is exactly a per-pixel χ² — the weighted sum of squared residuals between data and model), stripe-level errors inject spurious gradients into the χ² surface. Usually drowned by observational noise, but not negligible for high-signal-to-noise data near critical curves.

## 4.3 Impact on point sources: milliarcsecond-level position drift

The same integration error corrupts point-source computations directly, without any finite differences (findimg solves the lens equation using the very same α and the K-integral second derivatives). The repository's internal verification and the letter drafted to the upstream author record the quantitative picture ([manual/GLADE_Manual_en.md](GLADE_Manual_en.md) §10):

- **5e-4**: images close to a critical curve can be off by up to **~5 mas** (milliarcseconds) in position, and noticeably in magnification — near the critical curve the deflection error is further multiplied by |μ|. One internal verification, run on what later turned out to be a stale 5e-4 build of glafic, measured the deflection error of a single Sersic component peaking around ~46 µas — exactly the scale of milliarcsecond-level image shifts (memory.md §7);
- **1e-5**: positional accuracy reaches the µas–mas transition;
- **1e-8**: agreement with an independent high-precision scipy reference (scipy is the standard Python scientific-computing library; here it means a reference computation built independently on its high-precision integration routines) is essentially exact.

For time-delay cosmography (measuring cosmic expansion from the arrival-time differences between multiple images) or sub-milliarcsecond astrometry (ultra-precise measurement of image positions), 5e-4 is clearly insufficient; for routine arcsecond-level modelling it is just about fine — which is presumably the upstream default's trade-off logic.

## 4.4 The performance price: why GLADE stopped at 1e-5 rather than 1e-8

A tighter tolerance = more Romberg refinement levels. Each extra level doubles the number of integrand evaluations (the 16-level cap corresponds to at most 32769 evaluations per integral). The version history records the trade-off explicitly (Update.txt):

- V0.3 ran at **1e-8** outright — impeccable accuracy, noticeable speed cost;
- V0.4.0/V0.4.1 rolled back to **1e-5**: "a compromise between accuracy and speed: more accurate than the old 5.0e-4, much faster than V0.3's 1.0e-8";
- The only quantified number on record: tightening from 1e-5 to 1e-8 makes the affected integrals **~2–4× slower**.
- In honesty: **the repository contains no direct 5e-4 ↔ 1e-5 timing comparison**; the qualitative record is that 1e-5's cost is far below 1e-8's, while the stripes and the milliarcsecond issue are pushed below visibility / to acceptable levels.

## 4.5 GLADE's layered defences

1. **The header patch (the fundamental fix)**: [glafic.h:374-375](../glafic2/glafic.h#L374-L375) with the comment `/* glade local override: tightened from upstream 5.0e-4 for accuracy (see Update.txt) */`. This is a compile-time constant — **there is no runtime knob** (no .dat key, no environment variable, no API); changing it requires recompiling (`cd glafic2 && make clean && make python` — and do not forget the Python binding `glafic.so`; there was once a "build-state gotcha" where the header said 1e-5 but the stale .so still carried 5e-4).
2. **The GPU engine is natively immune**: GLADE's GPU engine, Rhongomyniad, uses the same Schramm formulas but replaces adaptive Romberg with **fixed 256-node Gauss–Legendre quadrature** (quadrature: another family of numerical-integration rules — here a weighted sum over 256 fixed, optimally placed sample points) ([Rhongomyniad/rhongomyniad/elliptical.py](../Rhongomyniad/rhongomyniad/elliptical.py)) — no stopping rule, no level staircase, no position-dependent branches; the error is smooth. Cross-validation against a high-precision scipy reference shows the GPU path reaches ~1e-9″ — **more accurate than the glafic binary**; the recompile-at-1e-8 cross-check performed before the V0.5.0 release (during the V0.4.6 stage) further confirmed that earlier tiny GPU↔glafic discrepancies were glafic-side Romberg noise (the sers deviation metric fell from 2.7e-4 to 1.3e-6).
3. **The verifier says so out loud**: when the glafic loss disagrees with the optimizer's, `core/verify.py` emits an informational warning — "glafic's elliptical-Sersic deflection is Romberg-tolerance-limited; this difference is expected and is NOT a result error" — and defers to the scipy-exact reference as the final judge.
4. **The 1e-8 cross-check protocol**: `tools/verify_gpu_models.py` documents the standard "temporarily set 1e-8 → recompile → compare → restore 1e-5" procedure used for high-accuracy verification before releases.
5. **Documentation**: the user manual ([GLADE_Manual_en.md](GLADE_Manual_en.md)/[zh](GLADE_Manual_zh.md) §10) devotes a chapter to this issue. One known historical leftover: `Rhongomyniad/README.md` and `constants.py` still quote 5e-4 — **stale mirror constants** consumed by no code; do not be misled.

## 4.6 Recommendations to upstream (tiered, as listed in the reproduction report)

1. **Immediate**: tighten the default `TOL_ROMBERG_JHK` from 5e-4 to 1e-5 or beyond — GLADE's measurements show the stripe vanishes with it;
2. **Root fix (design level)**: stop deriving the Hessian in `extend.c` by finite differences — glafic's `kapgam_*` functions already compute κ, γ **directly** via the K integrals (§2.3; still numerical with a ~TOL-level error, but never divided by 2h); using those removes the ~1/h amplification altogether, so residual integration error is never blown up into a visible artifact;
3. **Diagnostics**: restore the `gslstatus` check — at minimum warn when a Romberg integral fails to converge instead of silently returning an under-converged value.

(Context: the GLADE project has an established channel with the upstream author Oguri — two unrelated bugs were filed as oguri/glafic2 issues #4 and #5; a letter draft raising the TOL_ROMBERG_JHK point-source accuracy question predates the stripe discovery; the stripe itself has not yet been formally reported upstream.)

## 4.7 Conclusion

A 150-pixel dark line unravelled a complete numerical-physics causal chain: **the integer-level stopping of adaptive integration carves staircases into a smooth deflection field (the seed); finite differencing amplifies the steps by 1/2h (the amplifier); near-critical hypersensitivity and sub-pixel rendering develop them onto the brightest ring (the photographic developer).** Remove any one link and nothing is visible — analytic anfw has no seed, coarse pixels or a Hessian taken straight from the K integrals remove the amplifier, and faint regions have nothing to develop. GLADE's tightening from 5e-4 to 1e-5 shrinks the seed 30–50× at a mild CPU cost, backed up by an immune GPU engine, scipy-reference verification, and a dedicated manual chapter. Upstream's true cure is more radical and simpler still: replace the finite differences with the K-integral second derivatives the code already knows how to compute.

---

# Appendix A: Glossary

| Term | Meaning |
|---|---|
| image plane | The angular sky coordinates your telescope actually sees; each pixel of an observation is a point θ on it |
| source plane | The angular sky coordinates where the source would appear with the lens removed; coordinates β; connected to the image plane by the lens equation |
| lens equation | β = θ − α(θ): light seen in direction θ left the source at β |
| reduced deflection α | The physical deflection times the distance ratio D_ls/D_os, so the lens equation is homogeneous in angle (arcsec) |
| arcsec (″) | 1/3600 of a degree; the typical angular scale of strong lensing |
| solid angle | The "angular area" a patch subtends on the sky (a two-dimensional version of an angle); surface brightness and magnification are defined with it |
| redshift z | A cosmological distance label; larger is farther (here lens z=0.5, source z=2) |
| surface density Σ | Mass per unit area after projecting along the line of sight |
| critical surface density Σ_cr | A constant built from the three distances; Σ exceeding it (somewhere) suffices for multiple images |
| convergence κ | Σ/Σ_cr, the dimensionless surface density; also (ψxx+ψyy)/2; produces isotropic magnification |
| shear γ | Anisotropic distortion (circles → ellipses); components γ1, γ2 |
| lens potential ψ | 2-D scalar field with ∇ψ = α and ∇²ψ = 2κ |
| Hessian matrix | The matrix of second partial derivatives (ψxx, ψxy, ψyy); local curvature of the potential |
| Jacobian matrix A | First-derivative matrix of the map θ→β = identity − Hessian; the local linear map |
| magnification μ | 1/det A; the solid-angle (area) ratio of image to source |
| surface-brightness conservation | Lensing changes solid angles, not brightness per solid angle — the physical basis of per-pixel rendering |
| critical curve | The image-plane curve where det A = 0; magnification diverges nearby and errors are hyper-amplified |
| caustic | The critical curve mapped into the source plane; image numbers jump when the source crosses it |
| Einstein ring | The ring an image is stretched into when source, lens and observer are nearly aligned |
| extended source | A source of resolvable size (a galaxy), as opposed to a "point source" (quasar etc.) |
| inverse ray-shooting | The imaging algorithm mapping each image-plane pixel back to the source plane (the direction is forced: β→θ is multivalued) |
| ellipticity e / axis ratio q | e = 1 − minor/major axis (e=0 is a circle); q = 1 − e = minor/major axis ratio |
| principal axes / position angle pa | Principal axes = the ellipse's major/minor axis directions; position angle = the major axis's orientation on the sky |
| time delay | The arrival-time difference between the multiple images of one source; usable to measure cosmic expansion (time-delay cosmography) |
| NFW profile | The standard dark-matter halo density profile; parameters mass and concentration c |
| Sersic profile | The standard stellar profile exp[−b_n(r/r_e)^(1/n)]; n=4 is the classic elliptical-galaxy value |
| elliptical density | Ellipticity placed in the isodensity contours (physical, but deflection needs numerical integration) — where this case's lesion lives |
| elliptical potential | Ellipticity placed in the equipotentials (analytic, fast, but the implied density turns unphysically dumbbell-shaped at large e) |
| CSE approximation | Expanding a profile into a sum of ellipsoidal basis functions with closed-form lensing (anfw etc.; analytic AND elliptical-density) |
| Schramm J/K/I integrals | The 1-D integral family for elliptical-density lensing: J→deflection, K→second derivatives, I→potential |
| Romberg integration | Trapezoid rule + successive step-halving + Richardson extrapolation |
| Richardson extrapolation | Combining results at different step sizes to cancel the leading error terms, order by order |
| relative tolerance | The stopping threshold: converged when the relative change between successive levels drops below it (our protagonist, TOL_ROMBERG_JHK) |
| finite difference | Approximating a derivative by differences of neighbouring values; central difference f′≈[f(x+h)−f(x−h)]/2h |
| sub-pixel anti-aliasing | Subdividing a pixel into sub-points mapped to the source plane and averaged, to remove undersampling artifacts |
| FITS | Astronomy's standard image/data format; ds9 is the usual viewer |
| pixel scale pix_ext | The angle per pixel (0.001″ here); also the finite-difference step h |
| χ² | Goodness of fit: the weighted sum of squared data−model residuals |
| mas / µas | milliarcsecond (10⁻³ ″) / microarcsecond (10⁻⁶ ″) |

# Appendix B: Index of evidence and source locations

**Key source code** (all in the bundled glafic 2.1.14):

| Location | Content |
|---|---|
| [glafic.h:375](../glafic2/glafic.h#L375) | `TOL_ROMBERG_JHK 1.0e-5` (GLADE local override; upstream 5.0e-4, preserved in glafic.h.backup) |
| [glafic.h:369](../glafic2/glafic.h#L369) | `GSL_ROMBERG_N 16`: maximum Romberg refinement levels |
| [mass.c:3040-3133](../glafic2/mass.c#L3040-L3133) | `ell_integ_i/j/k`: the J/K/I integral drivers and integrands (the only 6 call sites consuming TOL_ROMBERG_JHK) |
| [mass.c:2159-2227](../glafic2/mass.c#L2159-L2227) | `kapgam_sers`: assembly of the elliptical-density Sersic deflection / second derivatives |
| [mass.c:3006-3034](../glafic2/mass.c#L3006-L3034) | `u_calc`: the coordinate substitution of the elliptical-potential route |
| [gsl_integration.c:113-140](../glafic2/gsl_integration.c#L113-L140) | `gsl_romberg2`: epsabs=0, purely relative tolerance; the `gslstatus` check commented out |
| [extend.c:42-147](../glafic2/extend.c#L42-L147) | Stage A: tiling α over the grid (alponly=1, float cache) + finite-difference Hessian (lines 135-142 = the amplifier) |
| [extend.c:191-210](../glafic2/extend.c#L191-L210) | Stage B: per-pixel lens equation + render call |
| [source.c:236-282](../glafic2/source.c#L236-L282) | `source_all`: the muinv refinement test and the sub-pixel footprint mapping (where the artifact develops) |

**Evidence data**: [exception/stripe_repro/](../exception/stripe_repro/) (reproduction README + 8 FITS + 4 analysis PNGs); input [exception/nfwsersic_lens_sersic_source.input](../exception/nfwsersic_lens_sersic_source.input).

**Related documents**: [GLADE_Manual_en.md](GLADE_Manual_en.md)/[zh](GLADE_Manual_zh.md) §10 (the numerical-accuracy chapter); Update.txt (tolerance history: 5e-4 → 1e-8 (V0.3) → 1e-5 (V0.4+)); memory.md §7, §11 (the Sersic-accuracy verdict and the 1e-8 recompile protocol).

**Theory references**: Schramm (1990, A&A 231, 19); Keeton (2001, astro-ph/0102341); Oguri (2021, arXiv:2106.11464, the CSE method); Ciotti & Bertin (1999, the b_n series); Tessore & Metcalf (2015, the analytic power-law method).
