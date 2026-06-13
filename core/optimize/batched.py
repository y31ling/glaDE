"""Batched GPU objective (whole DE population in one CUDA pass).

Generalized in V0.50 from the legacy point-mass-only pipeline to EVERY
GPU-supported deflector model: per DE generation, the optimizable components'
parameters become per-candidate ``(C, 1, 1)`` tensors fed through the
tensorized Rhongomyniad kernels, the LOCKED components' deflection field is
still cached once on the fine grid, the lens equation is solved by a batched
triangle-seed + Newton refine, and the per-candidate loss reuses the exact
same ``point_source_loss`` helper as the CPU path (so the optional
``missing_img_penalty`` behaves identically on GPU).

Configurations whose optimizable components are all point masses with a fixed
source keep the original analytic pipeline untouched (at the default
``gpu_precision = 64`` same-seed DE trajectories stay bit-identical with
pre-V0.50 runs); everything else takes the generalized tensor-kernel path,
chunked over the population (``GLADE_GPU_CHUNK``; default 32 when a
Schramm-quadrature model is optimizable, else 128 — doubled to 64/256 when
the field phase runs fp32, same memory envelope) to bound the 256-node
quadrature memory.

``gpu_precision`` (a .dat algorithm key; 64 default) selects the compute
precision of BOTH batched modes: 64 = fp64 everywhere; 48 = mixed — the
deflection-field + triangle-test phase runs fp32 (where consumer cards are
~64x faster and the Schramm quadrature dominates) while the Newton refine and
magnifications keep fp64; 32 = fp32 everywhere. Candidate decoding (10**log
mass) always happens in fp64 before casting.

Scope (:func:`can_batch_gpu`): single lens plane, every deflector model
GPU-supported, fixed component redshifts and fixed ``zs_fid`` (the kernels
resolve cosmological distances once), fixed ``hubble`` (the point loss never
uses time delays, so h would be a dead search dimension). The source position
MAY be optimizable. Any rejected configuration falls back to the per-candidate
:class:`~core.optimize.objective.Objective` with ``backend='gpu'`` (correct,
slower).
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from ..format import schema
from ..format.config import GladeConfig
from ..format.values import Bounds, Fixed, SharedBounds
from .loss import LossConfig
from .objective import INVALID_LOSS, point_source_loss
from .problem import OptProblem
from .scene import ObsData

# population chunk size for the generalized tensor-kernel path (also consumed
# by core.optimize.batched_extend)
_CHUNK_ENV = "GLADE_GPU_CHUNK"
# models treated as memory-heavy by the chunk heuristic: the Schramm density
# forms materialize (256, C, ny, nx) quadrature intermediates (linear + log
# rules) on the fine grid. 'pow' is actually the closed-form TM15 series (its
# host-synced convergence loop, not memory, is the cost) but is kept in the
# conservative set; the CSE models (anfw/ahern, 44-term) are light.
_SCHRAMM = {"nfw", "king", "sers", "hern", "pow", "gnfw", "tnfw", "ein"}


def _chunk_from_env(heavy: bool, fp32_fields: bool = False) -> int:
    """Population chunk size: ``GLADE_GPU_CHUNK`` override (non-fatal on a
    malformed value) or the 32/128 heavy-model heuristic — doubled when the
    field phase runs fp32 (gpu_precision 48/32 halves the bytes per
    candidate, so 64/256 keeps the same memory envelope and is measurably
    faster: 6.4 -> 4.8 ms/cand on the 20-dim nfw_only benchmark)."""
    raw = os.environ.get(_CHUNK_ENV)
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError:
            print(f"[warn] ignoring malformed {_CHUNK_ENV}={raw!r} "
                  f"(expected an integer)", flush=True)
    base = 32 if heavy else 128
    return base * 2 if fp32_fields else base


def _legacy_eligible(cfg: GladeConfig) -> bool:
    """The original analytic point-mass pipeline applies: every optimizable
    component is a point mass and the source position is fixed. Kept untouched
    so same-seed DE trajectories stay bit-identical with pre-V0.50 runs; also
    consumed by webui.runjob to tune the GPU MCMC walker default."""
    src = cfg.source
    if isinstance(src.get("source_x"), Bounds) or isinstance(src.get("source_y"), Bounds):
        return False
    # shared user variables need the dim-keyed generalized slots (the legacy
    # label-keyed point map cannot express a shared dimension)
    if any(isinstance(p, SharedBounds)
           for comp in cfg.components for p in (comp.z, *comp.params)):
        return False
    return all(comp.type == "point" or not comp.is_optimizable()
               for comp in cfg.components)


def can_batch_gpu(cfg: GladeConfig) -> tuple[bool, str]:
    """Whether the batched whole-population GPU path applies.

    Returns ``(ok, reason)``; ``reason`` explains why not when ``ok`` is False.
    """
    if isinstance(cfg.cosmology.get("hubble"), Bounds):
        # Cosmological distances are h-independent in glafic's Mpc/h units and
        # the point loss never uses time delays: a hubble search dimension
        # would silently have no effect on the loss.
        return False, ("hubble is optimizable (a dead search dimension for the "
                       "point-source loss, which never uses time delays)")
    lens_z = cfg.redshifts.get("lens_z")
    # must mirror _build_cache's resolution (g(rs, "lens_z", 0.216)): a missing
    # lens_z is the default there, NOT the first component's redshift.
    zl = float(lens_z) if isinstance(lens_z, (int, float)) else 0.216
    has_opt = (isinstance(cfg.source.get("source_x"), Bounds)
               or isinstance(cfg.source.get("source_y"), Bounds))
    for comp in cfg.components:
        spec = schema.model(comp.type)
        if not schema.supports("gpu", comp.type):
            return False, f"lens model '{comp.type}' has no GPU kernel"
        if isinstance(comp.z, Bounds):
            return False, f"component '{comp.name}' has an optimizable redshift"
        zc = comp.z.value if isinstance(comp.z, Fixed) else zl
        if abs(zc - zl) > 1e-6:
            return False, "batched GPU is single-plane: components must share lens_z"
        for j, p in enumerate(comp.params):
            if not isinstance(p, Bounds):
                continue
            has_opt = True
            pname = spec.params[j].name if (spec and j < len(spec.params)) else ""
            if pname == "zs_fid":
                # the kernels resolve the fiducial-redshift distance scaling
                # with scipy on the CPU; it must stay a python scalar.
                return False, (f"component '{comp.name}' ({comp.type}) has an "
                               f"optimizable fiducial source redshift (p1)")
    if not has_opt:
        return False, "no optimizable parameters for the batched objective"
    return True, ""


class BatchedGPUObjective:
    """Vectorized objective for ``scipy`` DE (``vectorized=True``).

    Not picklable / single-process by design (CUDA). ``__call__`` receives an
    ``(ndim, popsize)`` array and returns ``(popsize,)`` losses.

    Two internal modes, chosen once at cache-build time:

    * **legacy point-mass mode** — every optimizable component is a point mass
      and the source position is fixed: the original analytic pipeline; at the
      default ``gpu_precision = 64`` it is operation-for-operation untouched,
      so same-seed DE trajectories are bit-identical with pre-V0.50 runs;
    * **generalized mode** — anything else :func:`can_batch_gpu` accepts:
      optimizable components of any GPU-supported model (and an optimizable
      source position) become per-candidate tensors through the tensorized
      kernels, with the locked components' grid deflection still cached once.

    Both modes honor ``gpu_precision`` (64 fp64 / 48 mixed / 32 fp32); in the
    mixed mode the field + triangle-test phase runs fp32 while the Newton
    refine and the magnifications keep fp64.
    """

    def __init__(self, problem: OptProblem, obs: ObsData, loss_cfg: LossConfig):
        self.problem = problem
        self.obs = obs
        self.loss_cfg = loss_cfg
        self._cache = None
        self._torch = None
        self._K = None

    # -- lazy engine import --------------------------------------------------
    def _ensure_engine(self):
        if self._torch is not None:
            return
        import torch  # noqa: PLC0415
        from rhongomyniad import constants as K  # noqa: PLC0415
        from rhongomyniad.cosmology import Cosmology  # noqa: PLC0415
        from rhongomyniad.image_finder import sum_lensmodel  # noqa: PLC0415
        from rhongomyniad.lens_models import LensContext  # noqa: PLC0415
        import rhongomyniad as rh  # noqa: PLC0415
        self._torch = torch
        self._K = K
        self._Cosmology = Cosmology
        self._sum_lensmodel = sum_lensmodel
        self._LensContext = LensContext
        self.device = rh.get_device()
        self.dtype = torch.float64

    # -- one-time cache ------------------------------------------------------
    def _build_cache(self):
        self._ensure_engine()
        torch = self._torch
        cfg = self.problem.cfg
        cos, grid, rs = cfg.cosmology, cfg.grid, cfg.redshifts

        def g(d, k, default):
            v = d.get(k, default)
            return float(v) if isinstance(v, (int, float)) else float(default)

        lens_z = g(rs, "lens_z", 0.216)
        source_z = g(rs, "source_z", 0.409)
        source_x = g(cfg.source, "source_x", 0.0)
        source_y = g(cfg.source, "source_y", 0.0)
        xmin, ymin = g(grid, "xmin", -0.5), g(grid, "ymin", -0.5)
        xmax, ymax = g(grid, "xmax", 0.5), g(grid, "ymax", 0.5)
        pix_poi = g(grid, "pix_poi", 0.2)
        maxlev = int(g(grid, "maxlev", 5))

        # legacy mode keeps the original analytic point-mass pipeline (and its
        # bit-identical same-seed DE trajectories); anything else accepted by
        # can_batch_gpu takes the generalized tensor-kernel path.
        legacy = self._legacy_eligible(cfg)

        # classify components; build the optimizable index maps
        fixed_lenses = []
        points = []  # legacy: per optimizable point: dict(mass, x, y) each = ('dim', row) | ('fix', value)
        opt_lenses = []  # generalized: (glafic_key, [8 slots]); slot = ('fix', v) | ('dim', k, is_log)
        labels = [d.label for d in self.problem.dims]
        dim_of = {d.target: k for k, d in enumerate(self.problem.dims)}
        for comp in cfg.components:
            spec = schema.model(comp.type)
            if spec is None or not schema.supports("gpu", comp.type):
                raise ValueError(f"GPU backend cannot run model '{comp.type}'")
            zc = comp.z.value if isinstance(comp.z, Fixed) else lens_z
            if abs(zc - lens_z) > 1e-6:
                raise ValueError("batched GPU is single-plane: all components must share lens_z")
            if legacy and comp.type == "point" and comp.is_optimizable():
                slot = {}
                for name, j in (("mass", 0), ("x", 1), ("y", 2)):
                    p = comp.params[j] if j < len(comp.params) else Fixed(0.0)
                    lbl = f"{comp.name}.{name}"
                    if isinstance(p, Bounds) and lbl in labels:
                        slot[name] = ("dim", labels.index(lbl))
                    elif isinstance(p, Fixed):
                        slot[name] = ("fix", float(p.value))
                    else:
                        slot[name] = ("fix", 0.0)
                points.append(slot)
            elif not legacy and comp.is_optimizable():
                slots = [("fix", float(zc))]
                for j, p in enumerate(comp.params):
                    if isinstance(p, SharedBounds):
                        k = dim_of[("var", p.name)]
                        slots.append(("dim", k, self.problem.dims[k].log))
                    elif isinstance(p, Bounds):
                        k = dim_of[("comp_param", comp.index, j)]
                        slots.append(("dim", k, self.problem.dims[k].log))
                    elif isinstance(p, Fixed):
                        slots.append(("fix", float(p.value)))
                    else:
                        slots.append(("fix", 0.0))
                slots = (slots + [("fix", 0.0)] * 8)[:8]
                opt_lenses.append((spec.glafic_key, slots))
            else:
                p7 = [pp.value if isinstance(pp, Fixed) else 0.0 for pp in comp.params]
                p7 = (p7 + [0.0] * 7)[:7]
                fixed_lenses.append((spec.glafic_key, (zc, *p7)))

        # generalized mode: source position may be optimizable
        src_slots = []
        for axis, fallback in (("source_x", source_x), ("source_y", source_y)):
            k = dim_of.get(("source", axis))
            src_slots.append(("fix", fallback) if k is None else ("dim", k, False))

        # GPU compute precision (gpu_precision), honored by BOTH the legacy
        # point pipeline and the generalized path. 48 = mixed: the field /
        # triangle-test phase runs fp32, the Newton refine (and the
        # magnifications) keeps fp64. At the default 64 every cast below is a
        # no-op, preserving the legacy pipeline's bit-parity guarantee.
        gp = cfg.algorithm.get("gpu_precision", 64)
        prec = int(gp) if isinstance(gp, (int, float)) and int(gp) in (32, 48, 64) else 64
        dt_grid = torch.float32 if prec in (32, 48) else self.dtype
        dt_newton = torch.float32 if prec == 32 else self.dtype

        chunk = _chunk_from_env(any(name in _SCHRAMM for name, _ in opt_lenses),
                                fp32_fields=(dt_grid == torch.float32))
        if prec != 64:
            desc = ("fp32 fields + fp64 Newton refine" if prec == 48
                    else "full fp32")
            print(f"  [gpu] gpu_precision={prec}: {desc}"
                  + (f", chunk={chunk}." if not legacy else "."), flush=True)

        ctx = self._LensContext.build(
            self._Cosmology(omega=g(cos, "omega", 0.3), lam=g(cos, "lambda_cosmo", 0.7),
                            weos=g(cos, "weos", -1.0), hubble=g(cos, "hubble", 0.7)),
            zl=lens_z, zs=source_z)

        dp = pix_poi / (2 ** (maxlev - 1))
        nx = int(math.ceil((xmax - xmin) / dp)) + 1
        ny = int(math.ceil((ymax - ymin) / dp)) + 1
        xs_ax = torch.linspace(xmin, xmin + (nx - 1) * dp, nx, device=self.device, dtype=dt_grid)
        ys_ax = torch.linspace(ymin, ymin + (ny - 1) * dp, ny, device=self.device, dtype=dt_grid)
        gx, gy = torch.meshgrid(xs_ax, ys_ax, indexing="xy")

        if fixed_lenses:
            ax_f, ay_f, kap_f, g1_f, g2_f, _phi, _ = self._sum_lensmodel(
                ctx, fixed_lenses, gx, gy, need_kg=True, need_phi=False)
        else:
            z = torch.zeros_like(gx)
            ax_f = ay_f = kap_f = g1_f = g2_f = z

        self._cache = dict(
            ctx=ctx, gx=gx, gy=gy, dp=dp, nx=nx, ny=ny,
            ax=ax_f.contiguous(), ay=ay_f.contiguous(), kap=kap_f.contiguous(),
            g1=g1_f.contiguous(), g2=g2_f.contiguous(), fixed_lenses=fixed_lenses,
            points=points, source_x=source_x, source_y=source_y,
            legacy=legacy, opt_lenses=opt_lenses, src_slots=src_slots,
            chunk=chunk, precision=prec, dt_grid=dt_grid, dt_newton=dt_newton)

    @staticmethod
    def _legacy_eligible(cfg: GladeConfig) -> bool:
        return _legacy_eligible(cfg)

    # -- batched physics (ported from legacy v_pointmass_gpu) ----------------
    def _re2_point(self, mass, ctx):
        K = self._K
        d = ctx.dis_ls / (K.COVERH_MPCH * ctx.dis_ol * ctx.dis_os)
        return (2.0 * (K.R_SCHWARZ * mass / K.MPC2METER) * d) / (K.ARCSEC2RADIAN ** 2)

    def _pm_fields(self, sx, sy, log_m, ctx, gx, gy):
        torch = self._torch
        smallcore = self._K.DEF_SMALLCORE
        C, Kk = sx.shape
        ny, nx = gx.shape
        # decode 10**log in the master (fp64) dtype, THEN cast to the field
        # dtype — no-op at gpu_precision=64, ~30x less mass rounding at 48/32
        re2 = self._re2_point(torch.pow(10.0, log_m), ctx).to(sx.dtype)
        sx_b, sy_b, re2_b = sx.view(C, Kk, 1, 1), sy.view(C, Kk, 1, 1), re2.view(C, Kk, 1, 1)
        dx = gx.view(1, 1, ny, nx) - sx_b
        dy = gy.view(1, 1, ny, nx) - sy_b
        r2 = dx * dx + dy * dy
        sc2 = smallcore * smallcore
        rr = re2_b / (r2 + sc2)
        ax = (rr * dx).sum(dim=1)
        ay = (rr * dy).sum(dim=1)
        return ax, ay

    def _tri_contains(self, xs, ys, ax, ay, bx, by, cx, cy):
        d1x, d1y = xs - ax, ys - ay
        d2x, d2y = xs - bx, ys - by
        d3x, d3y = xs - cx, ys - cy
        d12 = d1x * d2y - d1y * d2x
        d23 = d2x * d3y - d2y * d3x
        d31 = d3x * d1y - d3y * d1x
        return (((d12 >= 0) & (d23 >= 0) & (d31 >= 0))
                | ((d12 <= 0) & (d23 <= 0) & (d31 <= 0)))

    def _solve(self, sx_t, sy_t, log_m_t, xs_src, ys_src, max_iter=8):
        torch, K = self._torch, self._K
        c = self._cache
        gx, gy, dp, ctx = c["gx"], c["gy"], c["dp"], c["ctx"]
        dt_grid, dt_newton = c["dt_grid"], c["dt_newton"]
        C, Kk = sx_t.shape

        # field/triangle phase at dt_grid (no-op casts at gpu_precision=64);
        # log_m_t stays fp64 — _pm_fields decodes 10**log first, then casts
        ax_p, ay_p = self._pm_fields(sx_t.to(dt_grid), sy_t.to(dt_grid),
                                     log_m_t, ctx, gx, gy)
        ax = c["ax"].unsqueeze(0) + ax_p
        ay = c["ay"].unsqueeze(0) + ay_p
        sx_grid = gx.unsqueeze(0) - ax
        sy_grid = gy.unsqueeze(0) - ay
        bl_x, bl_y = sx_grid[:, :-1, :-1], sy_grid[:, :-1, :-1]
        br_x, br_y = sx_grid[:, :-1, 1:], sy_grid[:, :-1, 1:]
        tl_x, tl_y = sx_grid[:, 1:, :-1], sy_grid[:, 1:, :-1]
        tr_x, tr_y = sx_grid[:, 1:, 1:], sy_grid[:, 1:, 1:]
        in_A = self._tri_contains(xs_src, ys_src, bl_x, bl_y, tr_x, tr_y, br_x, br_y)
        in_B = self._tri_contains(xs_src, ys_src, bl_x, bl_y, tr_x, tr_y, tl_x, tl_y)
        ox = gx[:-1, :-1].unsqueeze(0).expand_as(in_A)
        oy = gy[:-1, :-1].unsqueeze(0).expand_as(in_A)
        idx_A = torch.nonzero(in_A, as_tuple=False)
        idx_B = torch.nonzero(in_B, as_tuple=False)
        if idx_A.numel() + idx_B.numel() == 0:
            return [[] for _ in range(C)]

        def _seeds(idx, ofx, ofy):
            cfg_i, j, i = idx[:, 0], idx[:, 1], idx[:, 2]
            return cfg_i, ox[cfg_i, j, i] + ofx * dp, oy[cfg_i, j, i] + ofy * dp

        cA, xA, yA = _seeds(idx_A, 0.667, 0.333)
        cB, xB, yB = _seeds(idx_B, 0.333, 0.667)
        cand_cfg = torch.cat([cA, cB])
        cand_x0 = torch.cat([xA, xB]).to(dt_newton)
        cand_y0 = torch.cat([yA, yB]).to(dt_newton)

        # Newton refine at dt_newton (fp64 in mixed mode; per-point params
        # gathered from the fp64 master tensors so the decode stays exact)
        xi, yi = cand_x0.clone(), cand_y0.clone()
        sub_sx = sx_t[cand_cfg].to(dt_newton)
        sub_sy = sy_t[cand_cfg].to(dt_newton)
        sub_re2 = self._re2_point(torch.pow(10.0, log_m_t[cand_cfg]), ctx).to(dt_newton)
        fixed = c["fixed_lenses"]
        sc2 = K.DEF_SMALLCORE ** 2
        kap_t = g1_t = g2_t = None

        for _ in range(max_iter):
            if fixed:
                ax_fix, ay_fix, kap_fix, g1_fix, g2_fix, _, _ = self._sum_lensmodel(
                    ctx, fixed, xi, yi, need_kg=True, need_phi=False)
            else:
                ax_fix = ay_fix = kap_fix = g1_fix = g2_fix = torch.zeros_like(xi)
            dx = xi.unsqueeze(1) - sub_sx
            dy = yi.unsqueeze(1) - sub_sy
            r2 = dx * dx + dy * dy
            r2s = torch.clamp(r2, min=sc2)
            rr = sub_re2 / r2s
            ax_c = (rr * dx).sum(dim=1)
            ay_c = (rr * dy).sum(dim=1)
            inv_r4 = 1.0 / (r2s * r2s)
            g1_c = (sub_re2 * (dy * dy - dx * dx) * inv_r4).sum(dim=1)
            g2_c = (sub_re2 * (-2.0 * dx * dy) * inv_r4).sum(dim=1)
            ax_t, ay_t = ax_fix + ax_c, ay_fix + ay_c
            kap_t, g1_t, g2_t = kap_fix, g1_fix + g1_c, g2_fix + g2_c
            pxx, pyy, pxy = kap_t + g1_t, kap_t - g1_t, g2_t
            ff = xs_src - xi + ax_t
            gg = ys_src - yi + ay_t
            mm = (1.0 - pxx) * (1.0 - pyy) - pxy * pxy
            xi = xi + ((1.0 - pyy) * ff + pxy * gg) / mm
            yi = yi + ((1.0 - pxx) * gg + pxy * ff) / mm

        muinv = (1.0 - kap_t) ** 2 - (g1_t * g1_t + g2_t * g2_t)
        mag = 1.0 / (muinv + K.DEF_IMAG_CEIL)
        dist2 = (xi - cand_x0) ** 2 + (yi - cand_y0) ** 2
        keep = dist2 <= (2.0 * dp * dp)

        cfg_cpu = cand_cfg.cpu().numpy()
        xi_cpu, yi_cpu, mag_cpu = xi.cpu().numpy(), yi.cpu().numpy(), mag.cpu().numpy()
        keep_cpu = keep.cpu().numpy()
        # fp32 Newton iterates only converge to ~ulp of the root: duplicate
        # seeds land 1-10 ulp apart, far above the fp64-scaled mag tolerance.
        # Floor the merge test with an absolute ulp-scale distance; 0.0 at
        # fp64 keeps the gpu_precision=64/48 predicate bit-identical.
        pos_tol2 = ((128.0 * torch.finfo(dt_newton).eps) ** 2
                    if dt_newton == torch.float32 else 0.0)
        out = [[] for _ in range(C)]
        for i in range(len(cfg_cpu)):
            if not keep_cpu[i]:
                continue
            ci = int(cfg_cpu[i])
            x, y, m = float(xi_cpu[i]), float(yi_cpu[i]), float(mag_cpu[i])
            dup = False
            for xj, yj, mj in out[ci]:
                d2 = (x - xj) ** 2 + (y - yj) ** 2
                s = max(1.0, abs(x), abs(y))
                if (d2 <= pos_tol2 * s * s
                        or d2 / max(abs(m * mj), 1e-300) <= 10.0 * K.DEF_MAX_POI_TOL ** 2):
                    dup = True
                    break
            if not dup:
                out[ci].append((x, y, m))
        return out

    # -- generalized path: any GPU model as per-candidate tensors -------------
    def _slot_tensor(self, slot, arr_t):
        """slot -> float or (C,) tensor from the (ndim, C) candidate tensor.
        Log (mass-like) dims are searched in log10 and decoded to linear."""
        kind = slot[0]
        if kind == "fix":
            return slot[1]
        _, dim_idx, is_log = slot
        col = arr_t[dim_idx]
        return self._torch.pow(10.0, col) if is_log else col

    def _opt_lenses_t(self, arr_t):
        """[(glafic_key, params)] for the optimizable components; optimizable
        slots become per-candidate ``(C, 1, 1)`` tensors (broadcast against the
        grid by the kernels), locked slots stay python floats."""
        torch = self._torch
        C = arr_t.shape[1]
        out = []
        for name, slots in self._cache["opt_lenses"]:
            params = []
            for s in slots:
                v = self._slot_tensor(s, arr_t)
                params.append(v.view(C, 1, 1) if torch.is_tensor(v) else v)
            out.append((name, tuple(params)))
        return out

    def _source_xy_t(self, arr_t):
        """Per-candidate ``(C,)`` source-position tensors (fixed values are
        broadcast; optimizable ones come from their search dimension)."""
        torch = self._torch
        C = arr_t.shape[1]
        xy = []
        for s in self._cache["src_slots"]:
            v = self._slot_tensor(s, arr_t)
            if not torch.is_tensor(v):
                v = torch.full((C,), float(v), device=self.device, dtype=self.dtype)
            xy.append(v)
        return xy[0], xy[1]

    def _solve_general(self, arr_t, max_iter=8):
        """Batched lens-equation solve for the generalized path: the cached
        locked-component field + optimizable components through the tensorized
        kernels, per-candidate source positions. Same seed/Newton/dedup scheme
        (and contract) as :meth:`_solve`."""
        torch, K = self._torch, self._K
        c = self._cache
        gx, gy, dp, ctx = c["gx"], c["gy"], c["dp"], c["ctx"]
        dt_grid, dt_newton = c["dt_grid"], c["dt_newton"]
        C = arr_t.shape[1]

        # decode in fp64 (exact 10**log), cast per phase (no-op at 64). The
        # casts are mandatory in fp32 mode: a single fp64 parameter tensor
        # would silently promote the whole field phase back to fp64.
        opt = self._opt_lenses_t(arr_t)
        opt_grid = [(name, tuple(v.to(dt_grid) if torch.is_tensor(v) else v
                                 for v in p)) for name, p in opt]
        ax_o, ay_o, *_ = self._sum_lensmodel(ctx, opt_grid, gx, gy,
                                             need_kg=False, need_phi=False)
        if ax_o.dim() == 2:                    # every optimizable slot locked
            ax_o = ax_o.unsqueeze(0).expand(C, -1, -1)
            ay_o = ay_o.unsqueeze(0).expand(C, -1, -1)
        ax = c["ax"].unsqueeze(0) + ax_o
        ay = c["ay"].unsqueeze(0) + ay_o
        sx_grid = gx.unsqueeze(0) - ax
        sy_grid = gy.unsqueeze(0) - ay
        xs_src, ys_src = self._source_xy_t(arr_t)
        xsv = xs_src.to(dt_grid).view(C, 1, 1)
        ysv = ys_src.to(dt_grid).view(C, 1, 1)

        bl_x, bl_y = sx_grid[:, :-1, :-1], sy_grid[:, :-1, :-1]
        br_x, br_y = sx_grid[:, :-1, 1:], sy_grid[:, :-1, 1:]
        tl_x, tl_y = sx_grid[:, 1:, :-1], sy_grid[:, 1:, :-1]
        tr_x, tr_y = sx_grid[:, 1:, 1:], sy_grid[:, 1:, 1:]
        in_A = self._tri_contains(xsv, ysv, bl_x, bl_y, tr_x, tr_y, br_x, br_y)
        in_B = self._tri_contains(xsv, ysv, bl_x, bl_y, tr_x, tr_y, tl_x, tl_y)
        ox = gx[:-1, :-1].unsqueeze(0).expand_as(in_A)
        oy = gy[:-1, :-1].unsqueeze(0).expand_as(in_A)
        idx_A = torch.nonzero(in_A, as_tuple=False)
        idx_B = torch.nonzero(in_B, as_tuple=False)
        if idx_A.numel() + idx_B.numel() == 0:
            return [[] for _ in range(C)]

        def _seeds(idx, ofx, ofy):
            cfg_i, j, i = idx[:, 0], idx[:, 1], idx[:, 2]
            return cfg_i, ox[cfg_i, j, i] + ofx * dp, oy[cfg_i, j, i] + ofy * dp

        cA, xA, yA = _seeds(idx_A, 0.667, 0.333)
        cB, xB, yB = _seeds(idx_B, 0.333, 0.667)
        cand_cfg = torch.cat([cA, cB])
        cand_x0 = torch.cat([xA, xB]).to(dt_newton)
        cand_y0 = torch.cat([yA, yB]).to(dt_newton)

        # one combined stack at the seed points: locked params stay floats,
        # per-candidate tensors are gathered per seed from the fp64 decode
        # and cast to the Newton dtype (fp64 in mixed mode)
        lenses_pts = list(c["fixed_lenses"])
        for name, p in opt:
            q = tuple(v.view(-1)[cand_cfg].to(dt_newton) if torch.is_tensor(v)
                      else v for v in p)
            lenses_pts.append((name, q))
        xs_tgt = xs_src[cand_cfg].to(dt_newton)
        ys_tgt = ys_src[cand_cfg].to(dt_newton)

        xi, yi = cand_x0.clone(), cand_y0.clone()
        kap = g1 = g2 = None
        for _ in range(max_iter):
            ax_t, ay_t, kap, g1, g2, _phi, _mi = self._sum_lensmodel(
                ctx, lenses_pts, xi, yi, need_kg=True, need_phi=False)
            pxx, pyy, pxy = kap + g1, kap - g1, g2
            ff = xs_tgt - xi + ax_t
            gg = ys_tgt - yi + ay_t
            mm = (1.0 - pxx) * (1.0 - pyy) - pxy * pxy
            xi = xi + ((1.0 - pyy) * ff + pxy * gg) / mm
            yi = yi + ((1.0 - pxx) * gg + pxy * ff) / mm

        muinv = (1.0 - kap) ** 2 - (g1 * g1 + g2 * g2)
        mag = 1.0 / (muinv + K.DEF_IMAG_CEIL)
        dist2 = (xi - cand_x0) ** 2 + (yi - cand_y0) ** 2
        keep = dist2 <= (2.0 * dp * dp)

        cfg_cpu = cand_cfg.cpu().numpy()
        xi_cpu, yi_cpu, mag_cpu = xi.cpu().numpy(), yi.cpu().numpy(), mag.cpu().numpy()
        keep_cpu = keep.cpu().numpy()
        tol2 = 10.0 * K.DEF_MAX_POI_TOL ** 2
        # see _solve: absolute ulp-scale dedup floor for fp32 Newton iterates
        pos_tol2 = ((128.0 * torch.finfo(dt_newton).eps) ** 2
                    if dt_newton == torch.float32 else 0.0)
        out = [[] for _ in range(C)]
        for i in range(len(cfg_cpu)):
            if not keep_cpu[i]:
                continue
            ci = int(cfg_cpu[i])
            x, y, m = float(xi_cpu[i]), float(yi_cpu[i]), float(mag_cpu[i])
            dup = False
            for xj, yj, mj in out[ci]:
                d2 = (x - xj) ** 2 + (y - yj) ** 2
                s = max(1.0, abs(x), abs(y))
                if (d2 <= pos_tol2 * s * s
                        or d2 / max(abs(m * mj), 1e-300) <= tol2):
                    dup = True
                    break
            if not dup:
                out[ci].append((x, y, m))
        return out

    # -- objective -----------------------------------------------------------
    def __call__(self, params_arr):
        try:
            return self._evaluate(params_arr)
        except Exception as exc:  # noqa: BLE001 - never let DE die on a GPU hiccup
            # ...but say WHY once: with the generalized path a config/memory
            # problem (e.g. CUDA OOM on a huge grid) would otherwise surface
            # only as a run stuck at loss 1e15 with no diagnostic.
            if not getattr(self, "_warned", False):
                self._warned = True
                print(f"  [warn] batched GPU objective failed "
                      f"({type(exc).__name__}: {exc}); returning INVALID_LOSS "
                      f"for the whole population. If this is a GPU "
                      f"out-of-memory, lower {_CHUNK_ENV} (default 32/128 at "
                      f"fp64, 64/256 with fp32 fields) or shrink the grid.",
                      flush=True)
            arr = np.asarray(params_arr)
            popsize = arr.shape[1] if arr.ndim == 2 else 1
            return np.full(popsize, INVALID_LOSS)

    def _batch_tensors(self, arr):
        """Build (sx, sy, log_m) torch tensors of shape (popsize, K) from an
        ``(ndim, popsize)`` candidate array."""
        torch = self._torch
        popsize = arr.shape[1]
        points = self._cache["points"]
        Kk = len(points)

        def col(slot_key, k):
            kind, val = points[k][slot_key]
            return arr[val] if kind == "dim" else np.full(popsize, val)

        sx = np.stack([col("x", k) for k in range(Kk)], axis=1)
        sy = np.stack([col("y", k) for k in range(Kk)], axis=1)
        lm = np.stack([_as_log10(points[k]["mass"], col("mass", k)) for k in range(Kk)], axis=1)
        dev, dt = self.device, self.dtype
        return (torch.tensor(sx, device=dev, dtype=dt),
                torch.tensor(sy, device=dev, dtype=dt),
                torch.tensor(lm, device=dev, dtype=dt))

    def images_for(self, candidate):
        """Images ``[(x, y, mag), ...]`` for a single candidate, via the SAME
        batched solver that drives the optimization (so a result figure matches
        what the optimizer saw)."""
        if self._cache is None:
            self._build_cache()
        arr = np.asarray(candidate, dtype=float).reshape(-1, 1)
        if not self._cache["legacy"]:
            arr_t = self._torch.tensor(arr, dtype=self.dtype, device=self.device)
            return self._solve_general(arr_t)[0]
        sx_t, sy_t, lm_t = self._batch_tensors(arr)
        out = self._solve(sx_t, sy_t, lm_t,
                          float(self._cache["source_x"]), float(self._cache["source_y"]))
        return out[0]

    def _evaluate(self, params_arr):
        if self._cache is None:
            self._build_cache()
        arr = np.asarray(params_arr, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        popsize = arr.shape[1]
        obs = self.obs
        loss = np.empty(popsize)

        if not self._cache["legacy"]:
            torch, chunk = self._torch, self._cache["chunk"]
            for s in range(0, popsize, chunk):
                sub = arr[:, s:s + chunk]
                arr_t = torch.tensor(sub, dtype=self.dtype, device=self.device)
                images = self._solve_general(arr_t)
                for c in range(sub.shape[1]):
                    loss[s + c] = point_source_loss(images[c], obs, self.loss_cfg)
            return loss

        sx_t, sy_t, lm_t = self._batch_tensors(arr)
        all_images = self._solve(sx_t, sy_t, lm_t,
                                 float(self._cache["source_x"]),
                                 float(self._cache["source_y"]))
        for c in range(popsize):
            loss[c] = point_source_loss(all_images[c], obs, self.loss_cfg)
        return loss


def _as_log10(mass_slot, values):
    # mass dim candidates are already log10 (is_mass dims search in log space);
    # a fixed/locked mass is linear and must be converted.
    kind, _ = mass_slot
    if kind == "dim":
        return values
    return np.log10(np.maximum(values, 1e-300))
