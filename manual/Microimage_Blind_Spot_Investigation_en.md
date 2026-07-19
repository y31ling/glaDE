# Investigation of the GLADE / glafic Microimage Blind Spot
## How a numerically precise solver can still miss several real images inside one observed image

**Date:** 2026-07-19

**Intended reader:** A first-year physics student with no astronomy specialization. Familiarity with the mechanics of amoeba and differential evolution (DE) is assumed, so the optimizers themselves are not re-explained.

**Scope:** The CPU and GPU point-source DE paths in GLADE, the image-plane and source-plane point-source χ² paths of native glafic amoeba, archived runs, the current working-tree auto_check implementation, and [microimage_auto_check_plan.md](../microimage_auto_check_plan.md).

**Version note:** This report distinguishes the committed HEAD from the current uncommitted working tree. The committed baseline remains vulnerable. The working tree contains point-source auto_check protection, but it has explicit limits and should not be described as a released or absolute guarantee.

[中文版 / Chinese version](Microimage_Blind_Spot_Investigation_zh.md)

---

## One-page conclusion

**Yes. Both amoeba and GLADE's DE have this class of problem.** The shared fault is not the way amoeba or DE searches parameter space. It is the point-source objective that both ultimately trust:

1. A coarse image finder discovers only one root in a microimage cluster.
2. The matcher treats that root as the observed macro-image.
3. The photometric loss uses that root's |μ|.
4. The telescope, however, measures the total flux of all unresolved roots in the same PSF, whose model observable is Σ|μᵢ|.

There are important qualifications:

| Path | Structural risk? | Qualification |
|---|---:|---|
| GLADE DE, CPU point-source | **Yes** | The committed baseline uses one root; the current working tree wires in auto_check |
| GLADE DE, batched GPU point-source | **Yes** | Its finder is independent, but it produces the same wrong observable; the working tree has a separate local GPU check |
| Native glafic amoeba, image-plane χ² | **Yes** | It calls the same glafic findimg; parity and image-count guards reject some, but not all, cases |
| Native glafic amoeba, source-plane χ² | **Yes** | It does not enumerate roots and uses one local Jacobian branch; a finer grid cannot produce Σ|μᵢ| |
| GLADE extended-source DE | **Not covered by this fix** | A pure surface-brightness fit is not the same single-μ bug; an added point-flux constraint still requires a separate audit |

An archived case provides direct evidence. The coarse img4 solution is μ≈−9.09, apparently matching the observed |μ|=9.1. A fine solve finds four roots:

~~~text
+16.16067, −9.09454, +15.78977, −0.17922
Σ|μᵢ| = 41.22420
Σ μᵢ  = 22.67668
~~~

The image was not dimmed to 9.1. It was split into an unresolved cluster with total magnification about 41.2. The corrected magnification χ² of img4 alone is about 852.86, already much larger than the archived nominal total loss of 10.71.

> The shortest statement of the numerical cause is: **a coordinate can be polished to 10⁻¹⁰ arcsec only after a root has received a Newton seed. If the mesh never seeds the companion roots, Newton accuracy cannot reveal them.**

---

## 1. Three meanings of “one image”

### 1.1 Root, microimage cluster, and observed image

The point-source lens equation is

~~~text
β = θ − α(θ)
~~~

Here β is the unlensed angular source position, θ is an image position on the sky, and α is the lens deflection. For one β, the equation may have several θ solutions. Each mathematical solution is an **image root**.

In a smooth galaxy-scale lens, the roots are commonly separated by hundreds of milliarcseconds and are observed as distinct “macro-images.” If a point mass or another compact substructure lies very close to one macro-image, it can create a tiny local critical curve and replace the original root with several **microimages**, with separations ranging from sub-mas to about 11 mas in the archived examples; the point-mass cluster studied in detail spans about 3.4 mas.

This report therefore uses three levels:

- **Root:** one numerical solution of the lens equation.
- **Microimage cluster:** roots near one macro-image, created or rearranged by a local compact perturbation.
- **Observed image:** one PSF-shaped light patch measured by a telescope. When root separations are below instrumental resolution, one observed image contains several roots.

### 1.2 A negative μ does not mean negative photons

For local lens mapping Jacobian A=∂β/∂θ, the signed magnification is

~~~text
μ = 1 / det(A)
~~~

The magnitude |μ| is the area and total-flux amplification. The sign records **parity**, meaning whether the local image is mirror-reversed:

- μ>0: positive parity; local orientation is preserved.
- μ<0: negative parity; local orientation is reversed.

The minus sign is not negative light. For an unresolved cluster, the observed flux is therefore

~~~text
Fobs = Fsource × Σ |μᵢ|
~~~

It is not Fsource×|one chosen μ|, and it is not Fsource×|Σμᵢ|. The signed sum is a useful topology diagnostic, but a detector adds arriving photons and therefore measures Σ|μᵢ|. This is the standard observable used in substructure lensing and microlensing; see [Keeton 2003](https://arxiv.org/abs/astro-ph/0209040) and [Metcalf & Madau 2001](https://arxiv.org/abs/astro-ph/0108224).

### 1.3 How small is “unresolved”?

The iPTF16geu material included with this project gives an HST pixel scale of about 39.2 mas; the PSF is broader than one pixel. The four point-mass microimages in the archived case span at most about 3.41 mas, only about 8.7% of one HST pixel. A telescope convolves them into one light patch and cannot measure them root by root.

![Scale comparison between an HST pixel, the default finder grid, and the real microimage cluster](assets/microimage_scale.svg)

The three scales illustrate an important distinction: **whether an instrument resolves the microimages** and **whether a numerical program discovers them** are separate questions. The first is set by the PSF and detector pixels; the second is set by the root-finding mesh and seed strategy.

---

## 2. The real microimage structure in this case

A fine local solve of the archived model [runs/iptf-nfw-pm-1234](../runs/iptf-nfw-pm-1234/) gives these img4 roots:

| Root | x (arcsec) | y (arcsec) | μ | Parity |
|---|---:|---:|---:|---|
| A | 0.1447388 | 0.2885201 | +16.16067 | Positive |
| B | 0.1434405 | 0.2894134 | −9.09454 | Negative |
| C | 0.1416484 | 0.2899723 | +15.78977 | Positive |
| D | 0.1427556 | 0.2881219 | −0.17922 | Negative |

The flux-weighted centroid is approximately (0.1432601, 0.2892714) arcsec. The maximum span is about 3.41 mas, with nearby separations of roughly 1.46–1.88 mas. The coarse finder returned only root B, whose |−9.09454| accidentally agrees almost perfectly with the observed 9.1.

![The four historical img4 microimages and their signed magnifications](assets/microimage_cluster.svg)

There is also a strong physical consistency check:

~~~text
Signed sum of the four roots: +22.67668
Smooth macro-image without the perturber: about +22.673
~~~

Their near equality shows that the compact perturber mainly redistributes the original macro-image among roots of different parity. It does not turn positive photons into negative photons. The absolute sum rises to 41.22420, which is the flux magnification a detector should receive.

---

## 3. What is pix_poi?

### 3.1 It is neither a detector pixel nor final root accuracy

The glafic manual defines **pix_poi** as the largest point-source grid size, in arcseconds. It controls the initial image-plane squares used to search for roots of the lens equation.

Several “pixel” concepts must be kept separate:

| Quantity | Used for | Meaning |
|---|---|---|
| pix_poi | Point-source findimg | Largest/initial cell edge of the adaptive root-finding mesh |
| pix_ext | Extended-source imaging | Pixel edge used to render surface brightness; it does not have the same point-source recursion meaning |
| Instrument pixel scale | Real data | Angular sky size of a detector pixel |
| max_poi_tol | Newton solving | Residual tolerance for an already-seeded root; it does not discover new roots |

Thus pix_poi=0.2 means an initial point-source grid edge of 0.2 arcsec=200 mas. It does not limit the final coordinate to 200 mas accuracy. Once a triangle supplies a seed, Newton iteration can polish that root extremely accurately. pix_poi controls **whether the house is first marked on the map**; the Newton tolerance controls **how many digits of the marked street address are read**.

### 3.2 What is maxlev?

The actual input keyword is **maxlev**, without an underscore; “max_lev” is only an informal spelling. maxlev is the total number of adaptive mesh levels, and the base level counts as one.

The C source defines the cell size at level lev as

~~~text
dp(lev) = pix_poi / 2^lev,    lev = 0, 1, ..., maxlev−1
~~~

The minimum cell size is therefore

~~~text
dp_min = pix_poi / 2^(maxlev−1)
       = pix_poi × 2^(1−maxlev)
~~~

For pix_poi=0.2 arcsec and maxlev=5:

| Human level number | Source lev | Cell edge |
|---:|---:|---:|
| 1 | 0 | 200 mas |
| 2 | 1 | 100 mas |
| 3 | 2 | 50 mas |
| 4 | 3 | 25 mas |
| 5 | 4 | **12.5 mas** |

Both the glafic manual and [dp_lev in point.c](../glafic2/point.c#L1166) confirm this formula. GLADE's GPU finder also uses pix_poi/2^(maxlev−1), so the default CPU and GPU minimum seed scales are both 12.5 mas.

This corrects an important value in the original plan: its CPU value pix_poi/2^maxlev=6.25 mas is not the glafic convention. The correct value is **12.5 mas**. The current auto_check coarse box uses pix_poi=5×10⁻⁴ arcsec and maxlev=5, so its minimum cell is 0.03125 mas, not the approximately 0.016 mas stated in the original draft.

---

## 4. Why the guard program cannot see the extra images

### 4.1 The adaptive mesh sees only limited information on cell boundaries

glafic first evaluates the lens mapping at grid corners and stores det(A)=μ⁻¹. It refines a square when:

1. Corner signs of det(A) suggest that a critical curve crosses the cell.
2. Corner values of |det(A)| cross configured abnormal thresholds.
3. The cell lies near the center of a lens component.

This guard works well for a large critical curve crossing a cell. It can miss a **small closed microcritical loop lying entirely inside the cell**. If all four corners are outside the loop, they can have the same sign. Every corner guard looks normal even though additional roots exist inside.

A point-mass center does force continued refinement nearby, but recursion must stop at maxlev. A microimage cluster spanning only 3.41 mas can remain hidden inside the final 12.5 mas cell.

### 4.2 Triangles discover roots; Newton only polishes them

Each terminal square is split into triangles. glafic asks whether the source position lies inside the source-plane triangle formed by mapping the three image-plane corners. If so, it places one initial seed in that image-plane triangle. Newton iteration then refines that seed to a lens-equation root.

The order matters:

~~~text
Corner/triangle test → a finite set of seeds → Newton polishing of those seeds
~~~

The mapping inside one grid triangle can be highly folded. If several microimages lie inside it, the corner approximation may supply only one seed. A 10⁻¹⁰ arcsec Newton tolerance cannot manufacture the missing seeds.

### 4.3 The real 1–3 mas roots are not merged by final de-duplication

The plan interpreted [remove same images in point.c](../glafic2/point.c#L554) as a cell-scale merge. Its actual condition is

~~~text
d² / |μᵢ μⱼ| ≤ 10 × max_poi_tol²
~~~

The default max_poi_tol is 10⁻¹⁰ arcsec. For roots with |μ| around 9–16, the equivalent merge radius is only about 3.8×10⁻⁶ mas, far smaller than the true 1–3 mas separations. If the microimages receive separate seeds, this code will not merge them.

The failure occurs earlier, during **seed generation**.

### 4.4 Why the image-count guard fails

glafic's chi2_checknimg and GLADE's select_images can count only the roots returned by the finder. In the historical nfw-pm case the coarse finder reports four macro roots plus one faint central root. Downstream code drops the faint central root and sees the expected four images. It has no information that img4 should contain three additional local roots.

This illustrates a general computational principle: **a verifier that reuses the same lossy observation as the result being verified is not independent verification.** The archived glafic verification loss, 10.7202, differs from the optimizer's 10.7100 by only 0.095%, but both used the same coarse finder. Agreement did not establish physical correctness.

### 4.5 Parity catches only a subset

If an observation requires positive parity for img4 while the coarse finder returns only a μ<0 root, glafic's image-plane χ² can reject the match. GLADE's .dat-to-amoeba conversion does write the sign of observed μ into the parity field. It would therefore be wrong to claim that the historical “+22.7 to −9.09” candidate must pass the default converted amoeba job.

Parity still does not compute Σ|μᵢ|. It misses cases in which:

- the input sets parity=0, meaning no parity restriction;
- the selected main root keeps the observed parity while unseen companions exist;
- the coarse finder also misses the extra image count;
- source-plane χ² is used and no roots are enumerated at all.

Parity is a useful but incomplete logical filter, not a flux guard.

![Relationship between root finding, optimization, and the current auto_check](assets/microimage_pipeline.svg)

---

## 5. Code audit of GLADE DE

### 5.1 CPU path

The approximate CPU point-source call chain is:

~~~text
runner
  → Objective.evaluate_one
  → EngineBackend.compute_images
  → glafic point_solve
  → select_images / matching
  → point_source_loss
~~~

The relevant files are [runner.py](../core/optimize/runner.py), [objective.py](../core/optimize/objective.py), [backends.py](../core/optimize/backends.py), [matching.py](../core/optimize/matching.py), and [loss.py](../core/optimize/loss.py).

The historical pipeline is internally consistent: the matcher selects among the roots supplied by the backend, and the loss uses |μ| for the selected root. The backend list is incomplete, however, and the required physical observable is not “one root” but “the absolute magnification sum of the roots inside this PSF.”

### 5.2 GPU path

The GPU path does not call the same C root loop. [batched.py](../core/optimize/batched.py) uses a fixed grid/triangle seed construction and a batched Newton kernel. Its implementation is separate, but its blind spot is the same:

- the default finest seed spacing is still 12.5 mas;
- a terminal triangle may not supply enough seeds for internal microstructure;
- downstream code still scores one returned root as the observed image's entire flux.

Independent CPU and GPU implementations are therefore not physically independent checks. Both turn an incomplete root list into the same wrong observable.

### 5.3 Why the issue is optimizer-independent

DE only proposes parameter candidates. Amoeba, BIPOP-CMA-ES, jSO, and MCMC also only propose candidates. If the objective incorrectly assigns a low loss to a candidate, any effective search algorithm is likely to find it.

In the current working tree, DE, BIPOP-CMA-ES, and jSO share the point-source objective, while MCMC reuses the corresponding CPU/GPU evaluation paths. The vulnerability and its correction therefore belong at the objective/backend boundary, not under the name of one optimizer.

---

## 6. Code audit of native glafic amoeba

### 6.1 Amoeba does not inspect the physical observable

The command path supplies chi2calc to the simplex routine. Reflection, expansion, and contraction repeatedly ask only for the χ² of a parameter vector. The optimizer cannot know whether the μ inside that χ² describes one root or a cluster.

The chain is:

~~~text
commands.c: optimize
  → opt_lens.c: opt_lens / chi2calc
  → amoeba_opt.c: simplex
  → opt_point.c: chi2calc_opt_point
~~~

### 6.2 Default image-plane χ² shares the findimg blind spot

The default is chi2_splane=0. In this mode, [opt_point.c](../glafic2/opt_point.c) calls findimg and selects one nearest, unused root rr[k] for every observed image. Relative flux, absolute magnification, and magnitude-difference modes all ultimately read rr[k][2] from that single root. No branch sums unresolved companions into Σ|μᵢ|.

Whenever parity or image-count selection does not reject a candidate first, amoeba therefore inherits the same coarse glafic finder blind spot as GLADE's CPU DE.

### 6.3 Source-plane χ²: no grid-missed root, but still no cluster flux

With chi2_splane=1, glafic does not call findimg. It evaluates the local Jacobian μ at each observed image coordinate, estimates the μ gradient by finite differences, and obtains a model magnification for one image branch by linear expansion.

Consequently:

- it does not experience the specific step “the coarse grid failed to seed the second root”;
- it never enumerates that second root, so it still cannot calculate Σ|μᵢ|;
- reducing pix_poi changes an internal finite-difference step but cannot turn a one-branch approximation into a multi-root sum;
- the glafic manual explicitly states that chi2_checknimg does not work for source-plane χ².

The physical-observable problem is even more direct in this mode: one local Jacobian at an observed position is treated as the whole PSF flux.

### 6.4 Direct amoeba reproduction

This investigation supplied the archived SIE+point-mass snapshot to the vendored glafic binary with:

~~~text
pix_poi          0.2
maxlev           5
chi2_usemag     -1
chi2_checknimg   1
observed parity  0
optimized values source x/y only
~~~

Native amoeba/c2calc returned χ²=0.02759996 and four macro-images, using μ≈−9.0926 for img4. Its final source position was (0.002686377, 0.02443593) arcsec. A fine local solve of this **actual final candidate** gives:

~~~text
img4 Σ|μᵢ| = 38.9482
physical magnification χ² of img4 = 736.30
~~~

The values 38.9482/736.30 refer to amoeba's actual final source. The 38.9314/735.46 values in the archived-evidence table below refer to a nearly identical saved snapshot with a slightly different source. This is direct experimental evidence, not only a source-code inference: without a parity restriction, native amoeba can accept the same class of one-root false solution.

To test whether correct parity is sufficient, this investigation constructed a stronger control. All six other lens components of the archived SIE+point-mass model were left unchanged, while the point mass next to img4 was fixed at:

~~~text
M = 5512.6283494
x = 0.141834895409 arcsec
y = 0.287410485008 arcsec
~~~

The four observations were assigned their correct parities [−,+,−,+]. The run still used pix_poi=0.2, maxlev=5, chi2_checknimg=1, and chi2_usemag=−1, and optimized only the source position. Native amoeba converged to:

~~~text
total χ² = 0.1327811
position term = 0.1303010
photometric term = 0.0024801
source = (0.002678009, 0.02443690) arcsec
~~~

The coarse finder reported exactly four images, with magnifications approximately +15.7306, −35.5272, −7.4748, and +9.0897. The coarse img4 root now has the **correct positive parity**, and +9.0897 almost exactly matches the observed +9.1. Both the parity and image-count guards accept it.

A fine local solve of img4 in the same final model gives:

| Root | x (arcsec) | y (arcsec) | μ |
|---|---:|---:|---:|
| 1 | 0.1409116 | 0.2877793 | +12.84818 |
| 2 | 0.1423954 | 0.2869872 | +9.76924 |
| 3 | 0.1417389 | 0.2871566 | −1.53156 |
| 4 | 0.1419258 | 0.2875550 | −0.35040 |

The maximum root span is about 1.68 mas. Σ|μᵢ|=24.49937, while the signed sum is 20.73546; the physical img4 photometric χ² alone is about 195.98.

This correct-parity reproduction rules out the last possible misconception: a parity guard rejects a candidate whose selected root has the wrong sign, but not one whose selected root has the right sign while unseen companions are present.

### 6.5 A separate likelihood difference in GLADE's amoeba conversion

The current .dat-to-amoeba conversion writes observed parity but does not write chi2_usemag=-1. It therefore uses glafic's default chi2_usemag=0, which fits relative fluxes and a common source-flux normalization. The GLADE DE iPTF examples use an absolute SN Ia standard-candle magnification constraint.

The precise conclusion is therefore:

> Amoeba and DE use different search methods and do not always use identical numerical likelihoods, but both can score one root returned by a coarse finder as the entire PSF flux.

It would be inaccurate to say that both optimizers must assign the same loss to the exact historical candidate.

---

## 7. Quantitative evidence

### 7.1 Three archived false solutions

Each run's saved glafic_verify.input model was solved again in a fine local box:

| Archived run | Coarse matched μ | Fine roots | Σ|μᵢ| | Archived nominal total loss | Physical img4 mag χ² alone |
|---|---:|---:|---:|---:|---:|
| iptf-nfw-pm-1234 | −9.0945 | 4 | **41.2242** | 10.7100 | **852.86** |
| iptf-sie-pm-1234 | −9.0497 | 4 | **38.9314** | 0.1097 | **735.46** |
| iptf-sie-king-1234 | −11.6301 | 3 | **45.4291** | 23.2251 | **1090.75** |

The red bars below are not corrected total losses; each is only the corrected img4 photometric term. That one term already dwarfs the blue archived total.

![Archived nominal total loss versus the physical img4 photometric term alone](assets/microimage_losses.svg)

The archived run plot remains diagnostically useful: it shows that only one img4 root was sent downstream, rather than a PSF-scale sum over local microimages.

![Archived nfw-pm run result](../runs/iptf-nfw-pm-1234/result.png)

### 7.2 Valid controls

Models that should not be rejected were also checked:

- iptf-sie-nfw-1234: the target img3 remains one root at fine resolution, μ≈−7.75495.
- iptf-nfw-nfw-1234-loose: the target img3 remains one root, μ≈−7.98901.
- Full audits of iptf-nfw-nfw-1234 and its loose variant find no microimage cluster.

Thus “substructure near an image” does not automatically imply multiple microimages. A more diffuse NFW perturber can dim a saddle image without crossing the local critical condition. A correction should inspect the actual root topology, not ban every near-image substructure.

The plan's claimed NFW example with 4.7–6 mas multiple roots has no committed model snapshot. This report does not present it as independently archived proof.

### 7.3 Direct reevaluation by the current auto_check

Reevaluating saved candidates in the current uncommitted working tree gives:

| Candidate/path | Old loss | With auto_check |
|---|---:|---:|
| nfw-pm CPU | 10.7162 | **863.588** |
| nfw-pm GPU | 10.7100 | **860.545** |
| sie-king GPU | 23.2266 | **1109.862** |
| nfw-pm GPU fp32 | 10.7105 | **851.071** |

These values need not equal “archived total plus corrected img4 χ²” by simple arithmetic: matching, position terms, precision, and the full loss configuration can also differ. The robust conclusion is that the false low basin disappears when Σ|μᵢ| enters the photometric term.

---

## 8. What the current working-tree auto_check does

### 8.1 Two layers of protection

The implementation lives mainly in [core/micro_audit.py](../core/micro_audit.py) and is wired into [objective.py](../core/optimize/objective.py), [batched.py](../core/optimize/batched.py), and [verify.py](../core/verify.py):

1. **In-loop check:** when a compact perturber approaches a matched image, solve a fine local box and replace that image's model magnification with the cluster Σ|μᵢ|.
2. **Final verification audit:** solve local boxes again for the final candidate and report per-image roots, sum_abs_mu, physical_loss, and a fake_solution warning.

The approximate in-loop trigger is

~~~text
d < 10 × theta_scale + 2 mas
~~~

theta_scale is estimated from perturber mass/core size and scales with the object. For a point mass, the Einstein angle obeys θE∝√M. Reducing the mass by 100 therefore shrinks the local check box by 10. This is more robust than imposing a globally tiny, fixed pix_poi: an optimizer cannot evade a mass-adaptive grid merely by lowering the mass until the microimages fall below a fixed scale.

The helper's actual theta_scale dispatch is:

| Primary scale in the schema | theta_scale calculation |
|---|---|
| point or another profile whose scale is a mass in M⊙ | Einstein angle θE of a same-mass point lens |
| a velocity-dispersion model in km/s | SIS Einstein angle 4π(σ/c)²Dls/Ds |
| a model parameterized directly by an Einstein radius in arcsec | that radius |
| King, softened SIE/Jaffe, and similar profiles | the larger of the preceding scale and the explicit core radius |

Every result has a 0.02 mas floor. Components above 100 mas are treated as main-lens scale and excluded from the compact-perturber gate. Extended sources, uncertain schemas, components without a regular center, and models whose primary scale unit cannot be recognized are skipped. This is why custom or irregular models remain a boundary.

Final verification unconditionally runs a coarse image-centered box with 15 mas half-width, pix_poi=5×10⁻⁴ arcsec, and maxlev=5, giving dp_min=0.03125 mas. If the nearest perturber is within 15 mas and theta_scale<0.2 mas, it adds a fine perturber-centered box:

~~~text
half-width = max(20 × theta_scale, 2 × d)
pix_poi    = theta_scale
dp_min     = theta_scale / 16
~~~

Roots from both boxes are de-duplicated at theta_scale/10. The in-loop solve is entered only for d<10×theta_scale+2 mas, whereas the verification coarse box does not depend on that stricter trigger. These details differ slightly from the draft plan; the current source is authoritative.

### 8.2 A microimage cluster does not increase the “macro-image count”

When a local check finds four microimages, it replaces the matched image's μ with Σ|μᵢ| but does not turn a global four-image system into a seven-image system.

This is deliberate observational semantics:

~~~text
Global image count = number of resolvable macro-images
Flux of one macro-image = Σ|μᵢ| inside its PSF
~~~

Otherwise a physically correct but instrumentally unresolved cluster would be rejected by the n_obs guard.

### 8.3 auto_check=False

When auto_check is disabled, both protection layers are bypassed and the old single-root behavior is restored. Normal candidates with no trigger also stay on the original path as far as possible. This switch is useful for regression and diagnosis; **it does not make the legacy behavior physically safe for compact near-image perturbations**.

### 8.4 Present protection boundary

The following limits must remain explicit:

- Standalone native amoeba is launched directly as a glafic binary by the WebUI and is **not wired into** auto_check.
- Extended-source DE is not wired in. Pure extended surface-brightness fitting is a different observable, but an added point-flux constraint needs separate coverage.
- The in-loop check is currently fail-open. The GPU path prints one warning and falls back to the legacy one-root loss after a local-audit exception. The CPU objective currently falls back silently and does not guarantee a warning; a later final verification can expose the problem only if that verification succeeds.
- Only the nearest relevant compact perturber is selected for each macro-image. Several compact perturbers near the same image can exceed the local-box assumptions.
- Protection requires a schema from which center, mass, or core scale can be extracted. Irregular custom models may be skipped.
- calcimage can still report a one-root loss before a later verification audit marks it.
- These changes are uncommitted in the current working tree and are not a released baseline.

A GPU/verification auto_check warning must therefore not be treated as harmless log noise. If local solve failed, audit skipped, or fake_solution appears, inspect the per_image roots and physical_loss. Conversely, the absence of a CPU warning does not prove that the local audit succeeded.

---

## 9. Test completion

This investigation ran:

~~~text
.venv/bin/python -u core/tests/test_micro_audit.py
~~~

The result was **13/13 passing**. Covered behavior includes:

- local root structures and total magnifications of the three archived false solutions;
- two valid NFW controls;
- four-root topology after simultaneously scaling point mass and distance;
- helper logic for triggers, de-duplication, and loss replacement;
- principal report semantics of the verification layer.

Those 13 tests do not yet prove:

- the real small-budget CPU DE rerun requested by T5;
- full trajectory-level bit identity between auto_check=False and the old code;
- complete GPU precision 48/64 coverage from T8;
- the extended-source path;
- completeness with several compact perturbers close to one macro-image.

The correct test claim is therefore “core local solving and archived-candidate reevaluation are verified,” not “every optimization path has been exhaustively verified.”

---

## 10. Corrections now incorporated into microimage_auto_check_plan.md

The plan's central diagnosis and two-layer remedy are sound. The following audit corrections were incorporated into the plan on 2026-07-19:

1. **CPU minimum-cell formula**
   Incorrect: pix_poi/2^maxlev.
   Correct: pix_poi/2^(maxlev−1). The default CPU and GPU values are both 12.5 mas.

2. **auto_check coarse-box resolution**
   pix_poi=5×10⁻⁴ arcsec and maxlev=5 gives 0.03125 mas, not approximately 0.016 mas.

3. **Mechanism of missing roots**
   Real microimages are not merged at cell scale in point.c:554–560. The default merge radius is about six orders of magnitude smaller than their separation. Independent seeds are missing at the triangle stage.

4. **Total magnification in the scale test**
   After M/100 and distance/10, the four-root topology is recovered, but the measured Σ|μ| is about 34.4 rather than exactly 41. The macro-lens Jacobian has a gradient at the displaced local position. Scale invariance controls the point-mass length scale and topology, not an identical total magnification at a changed macro position. The present test accepts 28–55.

5. **Strength of the NFW example**
   The claimed “extreme NFW gives 4.7–6 mas multiple roots” lacks a fixed repository snapshot and should remain a pending reproduction, not archived conviction evidence.

6. **Actual verification trigger**
   Current verification first runs a fixed 15 mas coarse box for every image. Only the finer second box depends on perturber distance/scale.

7. **Old values in source comments (fixed)**
   The module-level text and COARSE_PIX_POI comment in core/micro_audit.py formerly said 6.25 mas and 0.016 mas. They now match the executable maxlev=5 geometry: 12.5 mas and 0.03125 mas. This was a comment-only correction; runtime behavior did not change.

These corrections do not invalidate the design. They sharpen its target: **generate enough scale-adaptive seeds, then convert an unresolved cluster into the correct Σ|μᵢ| observable.**

---

## 11. Practical interpretation checklist

When a near-image substructure produces a surprisingly good loss, answer at least these questions:

1. Is this a point-source flux/magnification constraint or pure extended surface brightness?
2. What are pix_poi, maxlev, and the actual dp_min?
3. Is the local box finer than the perturber's theta_scale, rather than merely finer than the global mesh?
4. How many roots exist near each observed macro-image?
5. Does the photometric comparison use one |μ| or the PSF-cluster Σ|μᵢ|?
6. Is the sign of μ used only as parity, without treating light as canceling?
7. Does image counting refer to resolvable macro-images, rather than incorrectly counting unresolved microimages as new macro-images?
8. Does verification genuinely improve local resolution, or reuse the original finder?
9. Does status/report contain fake_solution, physical_loss, or an audit warning?
10. If standalone amoeba was used, was a separate fine local audit performed?

If questions 4, 5, or 8 cannot be answered, close agreement between optimizer loss and glafic verification loss is not sufficient evidence.

---

## Appendix A: Key source locations

| Topic | Location |
|---|---|
| glafic grid level size | [glafic2/point.c](../glafic2/point.c#L1166) |
| Adaptive mesh corners and refinement | [glafic2/point.c](../glafic2/point.c#L123) |
| Triangle seeding, Newton, de-duplication | [glafic2/point.c](../glafic2/point.c#L431) |
| Amoeba point-source image/source-plane χ² | [glafic2/opt_point.c](../glafic2/opt_point.c) |
| Amoeba objective dispatch | [glafic2/opt_lens.c](../glafic2/opt_lens.c) |
| GLADE CPU objective | [core/optimize/objective.py](../core/optimize/objective.py) |
| GLADE CPU glafic backend | [core/optimize/backends.py](../core/optimize/backends.py) |
| GLADE batched GPU finder | [core/optimize/batched.py](../core/optimize/batched.py) |
| Matching and image-count filtering | [core/optimize/matching.py](../core/optimize/matching.py) |
| Local microimage audit | [core/micro_audit.py](../core/micro_audit.py) |
| Final verification | [core/verify.py](../core/verify.py) |
| .dat-to-amoeba parity conversion | [core/translate/glafic_io.py](../core/translate/glafic_io.py) |

## Appendix B: Reproduction anchors

The authoritative inputs are the saved glafic_verify.input model snapshots, not optimizer console summaries:

- [iptf-nfw-pm-1234](../runs/iptf-nfw-pm-1234/glafic_verify.input)
- [iptf-sie-pm-1234](../runs/iptf-sie-pm-1234/glafic_verify.input)
- [iptf-sie-king-1234](../runs/iptf-sie-king-1234/glafic_verify.input)
- [iptf-nfw-nfw-1234](../runs/iptf-nfw-nfw-1234/glafic_verify.input)
- [iptf-nfw-nfw-1234-loose](../runs/iptf-nfw-nfw-1234-loose/glafic_verify.input)

To reproduce, retain the lens and point/source parameters, shrink the solve region around the target macro-image, and choose a local dp_min well below the estimated theta_scale. Read every root in the local point.dat, spatially assign and de-duplicate them, then calculate Σ|μᵢ|. Comparing one coarse solve with another solve at the same resolution is not an independent check.

## Appendix C: Documentation and literature

- Bundled glafic manual: [man_glafic.txt](../glafic2/manual/man_glafic.txt), especially the pix_poi/maxlev and chi2_checknimg/chi2_splane entries.
- Bundled iPTF16geu paper: [SNIA_Paper1.pdf](../InputFiles/SN_2Sersic_NFW/SNIA_Paper1.pdf).
- Keeton, “Analytic Cross Sections for Substructure Lensing,” [arXiv:astro-ph/0209040](https://arxiv.org/abs/astro-ph/0209040).
- Metcalf & Madau, “Compound Gravitational Lensing as a Probe of Dark Matter Substructure,” [arXiv:astro-ph/0108224](https://arxiv.org/abs/astro-ph/0108224).
- Schechter & Wambsganss, “Quasar Microlensing at High Magnification and the Role of Dark Matter,” [ADS](https://ui.adsabs.harvard.edu/abs/2002ApJ...580..685S/abstract).
- Bradač et al., “B1422+231: The influence of mass substructure on strong lensing,” [arXiv:astro-ph/0112038](https://arxiv.org/abs/astro-ph/0112038).

---

## Final determination

**GLADE point-source DE on both CPU and GPU, and native glafic amoeba in both image-plane and source-plane χ² modes, have this class of physical-observable risk.** Default parity and image-count rules make amoeba more cautious for some exact candidates, but neither rule discovers roots omitted by the mesh or converts one μ into the unresolved-cluster Σ|μᵢ|.

The current working-tree auto_check implements the right protection for GLADE's point-source paths and passes 13 core tests. Standalone amoeba, the extended path, fail-open behavior, multiple perturbers, and full end-to-end regression remain explicitly open boundaries.
