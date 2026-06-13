"""Tests for the FindImage MCMC-GPU rail plumbing (V0.5.0).

Covers the rail -> (engine, mode) mapping in the WebUI and the GPU walker
auto-tuning in the run worker. Pure plumbing — no torch/CUDA required
(``can_batch_gpu`` is config-only logic).

    python -m pytest core/tests/test_mcmc_gpu_rail.py
    python core/tests/test_mcmc_gpu_rail.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.format import lint_text  # noqa: E402
from core.mcmc import MCMCConfig  # noqa: E402

# A point-source config the batched GPU likelihood accepts (since V0.5.0 any
# GPU-supported optimizable model qualifies; hubble must stay fixed).
BATCHABLE_CFG = """
omega = 0.3
lambda_cosmo = 0.7
weos = -1.0
hubble = 0.7
xmin, ymin = -0.5, -0.5
xmax, ymax = 0.5, 0.5
pix_ext = 0.01
pix_poi = 0.2
maxlev = 5
source_z = 0.409
lens_z = 0.216
source_x = 0.03
source_y = 0.024
obs_positions_mas_list = [[-266.0, 0.4], [118.8, -221.9], [238.3, 227.3], [-126.2, 319.7]]
obs_magnifications_list = [-35.6, 15.7, -7.5, 9.1]
obs_mag_errors_list = [2.1, 1.3, 1.0, 1.1]
obs_pos_sigma_mas_list = [0.41, 0.86, 2.23, 3.11]
center_offset_x = 0.0
center_offset_y = 0.0
obs_x_flip = False
'sers1': (1, 'sers', lens_z, 2.1e10, 0.0, 0.0, 0.3, 112.0, 0.39, 1.06)
'point1': (2, 'point', lens_z, {1e5, 1e7}, {-0.30, -0.20}, {-0.05, 0.05})
"""

# Same config, but an optimizable hubble breaks the batched path (the point
# loss never uses time delays, so h would be a dead search dimension).
NONBATCHABLE_CFG = BATCHABLE_CFG.replace("hubble = 0.7",
                                         "hubble = {0.5, 0.9}")

# A free (non-point) main lens: batchable since V0.5.0 via the generalized
# chunked tensor-kernel path — which does NOT auto-raise the walker count
# (the 1024-walker benchmark only holds for the single-pass point pipeline).
GENERAL_BATCHABLE_CFG = BATCHABLE_CFG.replace(
    "'sers1': (1, 'sers', lens_z, 2.1e10, 0.0, 0.0, 0.3, 112.0, 0.39, 1.06)",
    "'sers1': (1, 'sers', lens_z, {1e9, 1e12}, 0.0, 0.0, 0.3, 112.0, 0.39, 1.06)")

# An extended-source config the batched extend GPU path accepts: GPU-capable
# lens, single plane, no point constraints (so chi2_splane is irrelevant).
EXTEND_BATCHABLE_CFG = """
omega = 0.3
lambda_cosmo = 0.7
weos = -1.0
hubble = 0.7
xmin, ymin = -0.5, -0.5
xmax, ymax = 0.5, 0.5
pix_ext = 0.01
pix_poi = 0.2
maxlev = 5
source_z = 0.409
lens_z = 0.216
extended_file = 'obs.fits'
'sie1': (1, 'sie', lens_z, {100, 300}, 0.0, 0.0, 0.2, 30.0)
'ext1': (2, 'extsersic', source_z, {0.5, 2.0}, 0.0, 0.0, 0.2, 10.0, 0.3, 1.0)
"""

# Image-plane point constraints (chi2_splane = 0) force the per-candidate path.
EXTEND_NONBATCHABLE_CFG = EXTEND_BATCHABLE_CFG + """
constraint_file = 'pts.dat'
chi2_splane = 0
"""


def _cfg(text):
    cfg, issues = lint_text(text, backend="gpu", with_defaults=True)
    assert not any(i.is_error for i in issues), [str(i) for i in issues]
    return cfg


def test_rail_engine_mode_mapping():
    try:
        from webui.app import _rail_engine, _resolve_engine_mode
    except ImportError:
        return  # flask not installed in this environment — skip
    cfg = _cfg(BATCHABLE_CFG)
    assert _rail_engine("mcmc") == "cpu"
    assert _rail_engine("mcmc-gpu") == "gpu"
    assert _rail_engine("cpu") == "cpu"
    assert _rail_engine("gpu") == "gpu"
    assert _resolve_engine_mode("mcmc", cfg) == ("cpu", "mcmc")
    assert _resolve_engine_mode("mcmc-gpu", cfg) == ("gpu", "mcmc")
    # plain rails keep DE (MCMC_ENABLED defaults to False)
    assert _resolve_engine_mode("gpu", cfg) == ("gpu", "findimage")


def test_tune_gpu_auto_raises_default_walkers():
    from webui.runjob import _GPU_MCMC_DEFAULT_NWALKERS, _tune_mcmc_for_gpu
    cfg = _cfg(BATCHABLE_CFG)
    assert "MCMC_NWALKERS" in cfg.applied_defaults
    mcfg = MCMCConfig.from_cfg(cfg)
    out, batched = _tune_mcmc_for_gpu(SimpleNamespace(backend="gpu"), cfg, mcfg)
    assert batched is True
    assert out.nwalkers == _GPU_MCMC_DEFAULT_NWALKERS
    # only the ensemble size is tuned
    assert out.nsteps == mcfg.nsteps and out.burnin == mcfg.burnin


def test_tune_gpu_keeps_explicit_walkers():
    from webui.runjob import _tune_mcmc_for_gpu
    cfg = _cfg(BATCHABLE_CFG + "MCMC_NWALKERS = 64\n")
    assert "MCMC_NWALKERS" not in cfg.applied_defaults
    mcfg = MCMCConfig.from_cfg(cfg)
    out, batched = _tune_mcmc_for_gpu(SimpleNamespace(backend="gpu"), cfg, mcfg)
    assert batched is True
    assert out.nwalkers == 64


def test_tune_gpu_general_path_keeps_default_walkers():
    # generalized (chunked) batched path: batched=True but NO auto-1024
    from webui.runjob import _tune_mcmc_for_gpu, gpu_mcmc_auto_walkers
    cfg = _cfg(GENERAL_BATCHABLE_CFG)
    assert "MCMC_NWALKERS" in cfg.applied_defaults
    mcfg = MCMCConfig.from_cfg(cfg)
    out, batched = _tune_mcmc_for_gpu(SimpleNamespace(backend="gpu"), cfg, mcfg)
    assert batched is True
    assert out.nwalkers == mcfg.nwalkers
    assert gpu_mcmc_auto_walkers(cfg) is None


def test_tune_gpu_nonbatchable_falls_back():
    from webui.runjob import _tune_mcmc_for_gpu
    cfg = _cfg(NONBATCHABLE_CFG)
    mcfg = MCMCConfig.from_cfg(cfg)
    out, batched = _tune_mcmc_for_gpu(SimpleNamespace(backend="gpu"), cfg, mcfg)
    assert batched is False
    assert out.nwalkers == mcfg.nwalkers


def test_tune_noop_on_cpu_backend():
    from webui.runjob import _tune_mcmc_for_gpu
    cfg = _cfg(BATCHABLE_CFG)
    mcfg = MCMCConfig.from_cfg(cfg)
    out, batched = _tune_mcmc_for_gpu(SimpleNamespace(backend="cpu"), cfg, mcfg)
    assert out is mcfg and batched is False


def test_tune_gpu_extend_batchable_raises_walkers():
    from webui.runjob import _GPU_MCMC_DEFAULT_NWALKERS, _tune_mcmc_for_gpu
    cfg = _cfg(EXTEND_BATCHABLE_CFG)
    mcfg = MCMCConfig.from_cfg(cfg)
    out, batched = _tune_mcmc_for_gpu(SimpleNamespace(backend="gpu"), cfg, mcfg,
                                      extend=True)
    assert batched is True
    assert out.nwalkers == _GPU_MCMC_DEFAULT_NWALKERS


def test_tune_gpu_extend_nonbatchable_falls_back():
    from webui.runjob import _tune_mcmc_for_gpu
    cfg = _cfg(EXTEND_NONBATCHABLE_CFG)
    mcfg = MCMCConfig.from_cfg(cfg)
    out, batched = _tune_mcmc_for_gpu(SimpleNamespace(backend="gpu"), cfg, mcfg,
                                      extend=True)
    assert batched is False
    assert out.nwalkers == mcfg.nwalkers


def test_gpu_mcmc_auto_walkers_helper():
    from webui.runjob import _GPU_MCMC_DEFAULT_NWALKERS, gpu_mcmc_auto_walkers
    # unset MCMC_NWALKERS + batchable -> the auto value the worker will use
    assert gpu_mcmc_auto_walkers(_cfg(BATCHABLE_CFG)) == _GPU_MCMC_DEFAULT_NWALKERS
    assert (gpu_mcmc_auto_walkers(_cfg(EXTEND_BATCHABLE_CFG), extend=True)
            == _GPU_MCMC_DEFAULT_NWALKERS)
    # explicit MCMC_NWALKERS or non-batchable config -> no auto-raise
    assert gpu_mcmc_auto_walkers(_cfg(BATCHABLE_CFG + "MCMC_NWALKERS = 64\n")) is None
    assert gpu_mcmc_auto_walkers(_cfg(NONBATCHABLE_CFG)) is None
    assert gpu_mcmc_auto_walkers(_cfg(EXTEND_NONBATCHABLE_CFG), extend=True) is None


def _run_all() -> int:
    funcs = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    failures = 0
    for name, fn in funcs:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(funcs) - failures}/{len(funcs)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
