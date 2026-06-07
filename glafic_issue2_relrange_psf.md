### Summary

For a `relrange psf` prior, `parprior()` parses the two range factors into
`rat`/`sig`, but then stores `ral`/`rah` — which are **never assigned in that
branch** — into the PSF relative-range arrays. The user-specified `lo`/`hi` are
therefore silently discarded and the PSF relrange bound is set from uninitialized
stack values. The `relrange lens|extend|point` branches are correct; only the
`psf` branch is affected.

Introduced in **v2.1.13** (when the `relrange` prior was added); still present in
**v2.1.14**.

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
assigned anywhere in the `psf` branch. The branch looks copied from the sibling
`match` → `psf` branch (which legitimately reads and uses `rat`/`sig`), but the
`sscanf` target was not changed to `ral`/`rah`. The lens/extend/point `relrange`
branch reads correctly into `&ral, &rah`.

Note: because `&ral`/`&rah` have their address taken in the sibling branch, gcc
`-Wall` does **not** emit an uninitialized-use warning here, which is likely why
it went unnoticed.

### Effect

`para_psf_reral[j-1]` / `para_psf_rerah[j-1]` receive indeterminate values, so the
relative-range bound later enforced in `check_para_psf()` (`opt_extend.c`) is
garbage. The `lo`/`hi` written in the prior file have **no effect**; the effective
bound is whatever happened to be on the stack.

### Demonstration

Holding everything else fixed and varying only the `lo`/`hi` of a `relrange psf`
line, the current binary's behavior is **identical for every value** (they are
discarded), whereas a binary with the one-line fix below changes behavior with
the values. For example, for a PSF whose initial `FWHM2/FWHM1 ≈ 3.6`:

| binary | `relrange psf 5 1 lo hi` | correct band on FWHM2 | initial in band? | outcome |
|---|---|---|---|---|
| patched | `3.0 4.0` | [3.0,4.0]·FWHM1 | yes (3.6) | model accepted |
| patched | `1.0 1.5` | [1.0,1.5]·FWHM1 | no  | model rejected |
| current | `3.0 4.0` | — | — | same as below |
| current | `1.0 1.5` | — | — | identical to `3.0 4.0` |

i.e. the patched binary distinguishes the two ranges; the current binary does not.

(In practice this is currently masked by a separate crash: because the garbage
bound usually rejects the model, `relrange psf` tends to hit the
extended-source-optimize segfault reported in the companion issue. That one
should be fixed first to observe this bug cleanly.)

### Suggested fix (one line)

Read the two factors into the variables that are actually used:

```diff
-    nn = sscanf(buffer, "%s %s %d %d %lf %lf", ptype, keyword, &j, &jj, &rat, &sig);
+    nn = sscanf(buffer, "%s %s %d %d %lf %lf", ptype, keyword, &j, &jj, &ral, &rah);
```

(Optionally also initialize `ral`/`rah` at declaration as defensive hygiene.)
