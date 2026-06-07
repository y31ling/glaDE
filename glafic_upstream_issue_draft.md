# Upstream issue drafts for `oguri/glafic2`

While testing the new `relrange` prior I found **two independent bugs** in the
`parprior` / extended-source optimization path. They are unrelated in root cause,
so they are written up as two separate issues below (file them separately, or
together with a cross-reference — your call).

Tested on `glafic` **v2.1.14** (`VERSION "2.1.14"`, `RELEASE_DATE "2026.04.22"`),
built from source with gcc 13, `-O2`, on Linux x86-64. Bug 2 also reproduces on
**v2.1.10**.

---

## Issue 1 — Segfault in extended-source `optimize`/`optextend` when a parameter prior rejects the model

### Summary
Any extended-source fit (`optimize` or `optextend` with one or more `extend`
components) **crashes with SIGSEGV** if a `parprior` constraint rejects the
current model so that the extended-source χ² is never evaluated. This includes a
plain `range` prior — it is **not** specific to the new `relrange` prior, and it
is present at least as far back as v2.1.10.

### Environment
- glafic v2.1.14 (and v2.1.10), source build, Linux x86-64, gcc `-O2`.

### Steps to reproduce (self-contained, no external files)

`1_gen.input` — generate a mock observation:
```
omega 0.3
lambda 0.7
weos -1.0
hubble 0.7
prefix mock
xmin -3.0
ymin -3.0
xmax 3.0
ymax 3.0
pix_ext 0.06
pix_poi 0.5
maxlev 3
obs_gain 3.0
obs_ncomb 1

startup 1 1 0
lens   sie    0.3 250.0 0.0 0.0 0.2 30.0 0.0 0.0
extend sersic 1.0 5.0e4 0.1 0.1 0.2 20.0 0.5 1.5
end_startup
start_command
writeimage 0.0 1.0
quit
```

`2_fit.input` — fit it back, with a prior whose range excludes the initial value:
```
omega 0.3
lambda 0.7
weos -1.0
hubble 0.7
prefix fit
xmin -3.0
ymin -3.0
xmax 3.0
ymax 3.0
pix_ext 0.06
pix_poi 0.5
maxlev 3
obs_gain 3.0
obs_ncomb 1

startup 1 1 0
lens   sie    0.3 250.0 0.0 0.0 0.2 30.0 0.0 0.0
extend sersic 1.0 5.0e4 0.1 0.1 0.2 20.0 0.5 1.5
end_startup
start_setopt
0 0 0 0 0 0 0 0 
0 0 0 0 0 0 1 0 
end_setopt
start_command
readobs_extend mock_image.fits
parprior prior.dat
optimize
quit
```

`prior.dat` — force the Sersic index `n` (extend parameter 7, initial value 1.5)
into a band it does not contain:
```
range extend 1 7 5.0 6.0
```

Run:
```
glafic 1_gen.input        # writes mock_image.fits
glafic 2_fit.input        # SIGSEGV
```

### Actual result
```
######## optimizing lens
 number of parameters = 0 (lens) + 2 (extend) + 0 (point) 

amoeba:  n =     1  y = 1.000000e+30  tol = 0.000000e+00
Segmentation fault (core dumped)
```

`gdb` backtrace (debug build):
```
Program received signal SIGSEGV
#0  opt_lens (flag=0, verb=1) at opt_lens.c:257
        i = 0
257       if(array_ext_mask[i] == 0) nd++;
#1  do_command (...) at commands.c:319
#2  main (...) at glafic.c:65
```

### Expected result
The optimizer should simply treat the prior-rejected model as having a large χ²
(the intended `chi2pen_range` penalty) and continue / report normally, not crash.

### Root cause
In `chi2tot()` (`opt_lens.c`), a model that violates a prior returns the penalty
value **before** the extended-source χ² is computed:

```c
double chi2tot(...)
{
  ...
  if(check_para_lens_all() > 0) return chi2pen_range;   /* opt_lens.c:310 */
  ...
  if(check_para_ext_all() > 0) return chi2pen_range;    /* opt_lens.c:315  <-- early return */
  r = r + chi2calc_extend(chi2min_extend);              /* opt_lens.c:318  <-- skipped */
  ...
}
```

`array_ext_mask` is allocated only inside `chi2calc_extend()` → `ext_set_table()`
(`extend.c:64`). When the model is rejected on every evaluation (e.g. the initial
guess is outside the prior range), `ext_set_table()` is never called, so
`array_ext_mask` is never allocated (remains `NULL`).

The post-amoeba reporting block in `opt_lens()` then dereferences it, guarded only
by `ne > 0` (number of extended components), with no check that the mask was
actually allocated:

```c
if(verb > 0){
  nd = 0;
  if(ne > 0){
    for(i=0;i<(nx_ext*ny_ext);i++){
      if(array_ext_mask[i] == 0) nd++;   /* opt_lens.c:257  <-- NULL deref */
    }
  }
  ...
}
```

(The same `array_ext_mask` read pattern also appears in `opt_extend.c:216`.)

### Scope confirmed
- Triggers with a plain `range` prior (above) and with `relrange`/`range psf`
  priors — anything that makes `check_para_*_all() > 0`.
- Reproduces on stock **v2.1.10** as well as v2.1.14, so it predates the
  `relrange` feature.
- Does **not** trigger when no prior rejects the initial model (normal fits are
  unaffected).

### Suggested fix
Guard the mask read on actual allocation, e.g. track whether the extended χ² was
evaluated for the reported model (only count `nd` when `array_ext_mask != NULL`),
or ensure `array_ext_mask` is allocated for the reporting path regardless of
prior rejection. Minimal guard:

```c
if(ne > 0 && array_ext_mask != NULL){
  for(i=0;i<(nx_ext*ny_ext);i++){
    if(array_ext_mask[i] == 0) nd++;
  }
}
```

---

## Issue 2 — `relrange psf` in `parprior` reads the range values into the wrong variables (PSF relrange bounds never set)

### Summary
For a `relrange psf` prior, `parprior()` parses the two range factors into
`rat`/`sig` but then stores `ral`/`rah` — which are **never assigned in that
branch** — into the PSF relative-range arrays. As a result the user-specified
`lo`/`hi` are silently discarded and the PSF relrange bound is set from
uninitialized stack values. (`relrange lens|extend|point` are correct; only the
`psf` branch is affected.)

### Environment
- glafic v2.1.14 (introduced in v2.1.13, when `relrange` was added).

### Root cause
`init.c`, `parprior()`, the `relrange` → `psf` branch:

```c
if(strcmp(ptype, "relrange") == 0){
  if(strcmp(keyword, "psf") == 0){
    nn = sscanf(buffer, "%s %s %d %d %lf %lf",
                ptype, keyword, &j, &jj, &rat, &sig);   /* reads into rat, sig */
    if(nn != 6) terminator("input file format irrelevant (parprior)");
    if((j > NPAR_PSF) || (j < 1) || (jj > NPAR_PSF) || (jj < 1)){ terminator(...); }
    para_psf_reraj[j - 1] = jj - 1;
    para_psf_reral[j - 1] = ral;   /* <-- ral never assigned in this branch */
    para_psf_rerah[j - 1] = rah;   /* <-- rah never assigned in this branch */
    n++;
  } else {
    nn = sscanf(buffer, "%s %s %d %d %d %d %lf %lf",
                ptype, keyword, &i, &j, &ii, &jj, &ral, &rah);  /* lens/ext/point: correct */
    ...
  }
}
```

`ral`/`rah` are local `double`s declared at the top of `parprior()` and are not
assigned in the `psf` branch. The branch was evidently copied from the sibling
`match` → `psf` branch (which legitimately reads and stores `rat`/`sig`), but the
`sscanf` target was not updated to `ral`/`rah`. The lens/extend/point `relrange`
branch correctly reads into `&ral, &rah`.

Note: because `&ral`/`&rah` have their address taken in the sibling branch, gcc
`-Wall` does **not** warn about the uninitialized use here.

### Effect
`para_psf_reral[j-1]` / `para_psf_rerah[j-1]` receive indeterminate values, so the
relative-range bound enforced in `check_para_psf()` (`opt_extend.c:384`) is
garbage. Consequently the `lo`/`hi` written in the prior file have **no effect**,
and the effective bound is whatever happened to be on the stack.

### Demonstration
Holding everything else fixed and varying only the `lo`/`hi` of a `relrange psf`
line, the current (buggy) binary's behavior is **identical for every value**
(the values are discarded), whereas a binary with the one-line fix below changes
behavior as the values change (e.g. for a PSF whose initial `FWHM2/FWHM1 ≈ 3.6`,
`relrange psf 5 1 3.0 4.0` accepts the model while `relrange psf 5 1 1.0 1.5`
rejects it — the patched binary distinguishes them; the unpatched binary does
not).

(In practice, on current `glafic` this bug is masked by Issue 1: because the
garbage bound usually rejects the model, `relrange psf` tends to crash via the
Issue-1 path. Issue 1 should be fixed first to observe this one cleanly.)

### Suggested fix (one line)
Read the two factors into the variables that are actually used:

```diff
-    nn = sscanf(buffer, "%s %s %d %d %lf %lf", ptype, keyword, &j, &jj, &rat, &sig);
+    nn = sscanf(buffer, "%s %s %d %d %lf %lf", ptype, keyword, &j, &jj, &ral, &rah);
```

(Optionally also initialize `ral`/`rah` at declaration as defensive hygiene.)
