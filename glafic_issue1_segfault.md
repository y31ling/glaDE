### Summary

Any extended-source fit (`optimize` or `optextend` with one or more `extend`
components) **crashes with SIGSEGV** if a `parprior` constraint rejects the
current model so that the extended-source χ² is never evaluated. A plain `range`
prior is enough to trigger it — it is **not** specific to the new `relrange`
prior, and it reproduces at least as far back as v2.1.10.

### Environment

- glafic **v2.1.14** (`RELEASE_DATE 2026.04.22`); also reproduced on **v2.1.10**.
- Built from source, Linux x86-64, gcc 13, `-O2`.

### Steps to reproduce (self-contained, no external data files)

**`1_gen.input`** — generate a mock observation:

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

**`2_fit.input`** — fit it back, with a prior whose range excludes the initial value:

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

**`prior.dat`** — force the Sersic index `n` (extend parameter 7, initial 1.5)
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
#1  do_command (...) at commands.c
#2  main (...) at glafic.c
```

### Expected result

A prior-rejected model should simply receive the intended large `chi2pen_range`
penalty and the run should continue / report normally — not crash.

### Root cause

In `chi2tot()` (`opt_lens.c`), a model that violates a prior returns the penalty
**before** the extended-source χ² is computed:

```c
double chi2tot(...)
{
  ...
  if(check_para_lens_all() > 0) return chi2pen_range;   /* early return */
  ...
  if(check_para_ext_all() > 0) return chi2pen_range;    /* early return */
  r = r + chi2calc_extend(chi2min_extend);              /* skipped */
  ...
}
```

`array_ext_mask` is allocated only inside `chi2calc_extend()` → `ext_set_table()`
(`extend.c`). When the model is rejected on every evaluation (e.g. the initial
guess is outside the prior range), `ext_set_table()` is never called, so
`array_ext_mask` is never allocated (stays `NULL`).

The post-amoeba reporting block in `opt_lens()` then dereferences it, guarded
only by `ne > 0` (number of extended components):

```c
if(verb > 0){
  nd = 0;
  if(ne > 0){
    for(i=0;i<(nx_ext*ny_ext);i++){
      if(array_ext_mask[i] == 0) nd++;   /* opt_lens.c:257 — NULL deref */
    }
  }
  ...
}
```

(The same `array_ext_mask` read pattern also appears later in `opt_extend.c`.)

### Scope confirmed

- Triggers with a plain `range` prior (above) and with `relrange` / `range psf`
  priors — anything that makes `check_para_*_all() > 0` on the reported model.
- Reproduces on stock **v2.1.10** as well as v2.1.14, so it predates the
  `relrange` feature.
- Does **not** trigger when no prior rejects the model (normal fits are
  unaffected).

### Suggested fix

Guard the mask read on actual allocation:

```c
if(ne > 0 && array_ext_mask != NULL){
  for(i=0;i<(nx_ext*ny_ext);i++){
    if(array_ext_mask[i] == 0) nd++;
  }
}
```

(or ensure the reporting path only reads `array_ext_mask` when the extended χ²
was actually evaluated for the reported model).
