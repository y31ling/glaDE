# GLADE `core/` — the V0.4 foundation

Backend-agnostic, self-contained modules extracted from the duplicated legacy
`version_*.py` scripts. **No module here imports from `legacy/`** — `legacy/` is
a frozen historical artifact.

```
core/
├── SPEC.md            # the .dat configuration format specification
├── format/            # parse / merge / default / validate the .dat format
│   ├── parser.py        restricted-AST parser ({lo,hi} bounds, $placeholders, refs)
│   ├── schema.py        per-model param tables + GPU capability + scalar groups
│   ├── config.py        GladeConfig + multi-file section-merge (+ global re-index)
│   ├── defaults.py      iPTF16geu defaults (from the legacy point-mass script)
│   ├── validate.py      block unfilled / unknown-model / GPU-unsupported / missing-obs
│   └── api.py           load_config(paths, backend) / lint_text(text)
├── optimize/          # one backend-agnostic Differential Evolution optimizer
│   ├── problem.py       optimizable dims (log10 for mass-like) + scene reconstruction
│   ├── scene.py         concrete Scene + ObsData (mas->arcsec, x-flip)
│   ├── backends.py      EngineBackend drives glafic (cpu) OR rhongomyniad (gpu)
│   ├── matching.py      image selection + Hungarian matching
│   ├── loss.py          A*chi2_pos + B*chi2_mag + penalty
│   ├── objective.py     picklable candidate->loss (for process-pool workers)
│   ├── de.py            DifferentialEvolutionSolver loop + early stopping
│   └── runner.py        optimize(cfg, backend) -> OptResult
├── translate/         # glafic <-> glade .dat, both directions (+ CLI)
│   ├── glafic_io.py     parse/render glafic input + obs files
│   ├── convert.py       opt-matrix<->{lo,hi}; geometric-mean midpoint for mass
│   └── cli.py           python -m core.translate.cli {to-glade|to-glafic}
├── plot/              # self-contained result plotting
│   ├── crit.py          read glafic *_crit.dat
│   ├── triptych.py      3-panel result + baseline/optimized compare
│   ├── iteration.py     DE population scatter (draw_interval frames)
│   └── labels.py        per-model sub-halo marker labels
├── report.py          # OptResult -> triptych (computes images via a backend)
└── tests/             # 56 tests, dependency-free runners
```

## Quick use

```python
from core.format import load_config, has_errors
from core.optimize import optimize, build_obs
from core.report import make_triptych

cfg, issues = load_config(["constants.dat", "images_data.dat", "lens.dat"],
                          backend="cpu")          # or "gpu" / "glafic"
assert not has_errors(issues)
result = optimize(cfg, backend="cpu")             # DifferentialEvolution
make_triptych(result, build_obs(cfg), "result.png")
```

Translate from/to glafic:

```bash
python -m core.translate.cli to-glade  some_model.input -o out/
python -m core.translate.cli to-glafic constants.dat lens.dat obs.dat -o out/run
```

## Run the tests

```bash
source env.sh   # puts glafic2/python + Rhongomyniad on PYTHONPATH for live runs
for t in format optimize translate plot; do .venv/bin/python core/tests/test_$t.py; done
```

The optimize/plot tests use an analytic `FakeBackend`, so they pass without the
glafic/torch engines; with `env.sh` sourced, the same `EngineBackend` drives the
real glafic (CPU) and Rhongomyniad (GPU) engines identically.
