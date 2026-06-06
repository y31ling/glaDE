"""Batched GPU objective (whole DE population in one CUDA pass).

A faithful port of the legacy ``v_pointmass_gpu`` batched pipeline, generalized to
the new config model: the FIXED (non-optimized) components' deflection field is
cached once on a fine grid; the OPTIMIZED point-mass sub-structures are evaluated
for the entire population in one batched kernel, the lens equation is solved by a
batched triangle-seed + Newton refine, and the per-candidate loss reuses the
exact same ``select_images`` / ``match_images`` / ``ml_loss`` as the CPU path.

Scope: the fast batched path requires every *optimizable* component to be a
point mass (``'point'``) and the source position to be fixed (matching the legacy
GPU optimizers). For any other combination, callers fall back to the per-candidate
:class:`~core.optimize.objective.Objective` with ``backend='gpu'`` (correct, slower).
Use :func:`can_batch_gpu` to decide.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..format import schema
from ..format.config import GladeConfig
from ..format.values import Bounds, Fixed
from .loss import LossConfig, ml_loss
from .matching import match_images, select_images
from .objective import INVALID_LOSS
from .problem import OptProblem
from .scene import ObsData


def can_batch_gpu(cfg: GladeConfig) -> tuple[bool, str]:
    """Whether the fast batched point-mass GPU path applies.

    Returns ``(ok, reason)``; ``reason`` explains why not when ``ok`` is False.
    """
    src = cfg.source
    if isinstance(src.get("source_x"), Bounds) or isinstance(src.get("source_y"), Bounds):
        return False, "source position is optimizable (batched GPU needs a fixed source)"
    has_opt_point = False
    for comp in cfg.components:
        optimizable = comp.is_optimizable()
        if optimizable and comp.type != "point":
            return False, f"component '{comp.name}' ({comp.type}) is optimizable but not a point mass"
        if optimizable:
            has_opt_point = True
    if not has_opt_point:
        return False, "no optimizable point-mass sub-structure"
    return True, ""


class BatchedGPUObjective:
    """Vectorized objective for ``scipy`` DE (``vectorized=True``).

    Not picklable / single-process by design (CUDA). ``__call__`` receives an
    ``(ndim, popsize)`` array and returns ``(popsize,)`` losses.
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

        # classify components; build the optimizable point-mass index maps
        fixed_lenses = []
        points = []  # per optimizable point: dict(mass, x, y) each = ('dim', row) | ('fix', value)
        labels = [d.label for d in self.problem.dims]
        for comp in cfg.components:
            spec = schema.model(comp.type)
            if spec is None or not schema.supports("gpu", comp.type):
                raise ValueError(f"GPU backend cannot run model '{comp.type}'")
            zc = comp.z.value if isinstance(comp.z, Fixed) else lens_z
            if abs(zc - lens_z) > 1e-6:
                raise ValueError("batched GPU is single-plane: all components must share lens_z")
            if comp.type == "point" and comp.is_optimizable():
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
            else:
                p7 = [pp.value if isinstance(pp, Fixed) else 0.0 for pp in comp.params]
                p7 = (p7 + [0.0] * 7)[:7]
                fixed_lenses.append((spec.glafic_key, (zc, *p7)))

        ctx = self._LensContext.build(
            self._Cosmology(omega=g(cos, "omega", 0.3), lam=g(cos, "lambda_cosmo", 0.7),
                            weos=g(cos, "weos", -1.0), hubble=g(cos, "hubble", 0.7)),
            zl=lens_z, zs=source_z)

        dp = pix_poi / (2 ** (maxlev - 1))
        nx = int(math.ceil((xmax - xmin) / dp)) + 1
        ny = int(math.ceil((ymax - ymin) / dp)) + 1
        xs_ax = torch.linspace(xmin, xmin + (nx - 1) * dp, nx, device=self.device, dtype=self.dtype)
        ys_ax = torch.linspace(ymin, ymin + (ny - 1) * dp, ny, device=self.device, dtype=self.dtype)
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
            points=points, source_x=source_x, source_y=source_y)

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
        re2 = self._re2_point(torch.pow(10.0, log_m), ctx)
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
        C, Kk = sx_t.shape

        ax_p, ay_p = self._pm_fields(sx_t, sy_t, log_m_t, ctx, gx, gy)
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
        cand_x0 = torch.cat([xA, xB])
        cand_y0 = torch.cat([yA, yB])

        xi, yi = cand_x0.clone(), cand_y0.clone()
        sub_sx = sx_t[cand_cfg]
        sub_sy = sy_t[cand_cfg]
        sub_re2 = self._re2_point(torch.pow(10.0, log_m_t[cand_cfg]), ctx)
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
        out = [[] for _ in range(C)]
        for i in range(len(cfg_cpu)):
            if not keep_cpu[i]:
                continue
            ci = int(cfg_cpu[i])
            x, y, m = float(xi_cpu[i]), float(yi_cpu[i]), float(mag_cpu[i])
            dup = False
            for xj, yj, mj in out[ci]:
                if ((x - xj) ** 2 + (y - yj) ** 2) / max(abs(m * mj), 1e-300) <= 10.0 * K.DEF_MAX_POI_TOL ** 2:
                    dup = True
                    break
            if not dup:
                out[ci].append((x, y, m))
        return out

    # -- objective -----------------------------------------------------------
    def __call__(self, params_arr):
        try:
            return self._evaluate(params_arr)
        except Exception:  # noqa: BLE001 - never let DE die on a GPU hiccup
            arr = np.atleast_2d(params_arr)
            popsize = arr.shape[1] if arr.ndim == 2 else 1
            return np.full(popsize, INVALID_LOSS)

    def _evaluate(self, params_arr):
        if self._cache is None:
            self._build_cache()
        torch = self._torch
        arr = np.asarray(params_arr, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
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
        all_images = self._solve(
            torch.tensor(sx, device=dev, dtype=dt),
            torch.tensor(sy, device=dev, dtype=dt),
            torch.tensor(lm, device=dev, dtype=dt),
            float(self._cache["source_x"]), float(self._cache["source_y"]))

        obs = self.obs
        loss = np.full(popsize, INVALID_LOSS)
        for c in range(popsize):
            sel = select_images(all_images[c], obs.n)
            if sel is None:
                continue
            pred_pos = np.array([[im[0], im[1]] for im in sel], dtype=float)
            pred_mag = np.array([im[2] for im in sel], dtype=float)
            _, mm, delta = match_images(obs.positions, pred_pos, pred_mag, obs.center_offset)
            loss[c] = ml_loss(delta, mm, obs.magnifications, obs.mag_errors,
                              obs.pos_sigma_mas, self.loss_cfg)
        return loss


def _as_log10(mass_slot, values):
    # mass dim candidates are already log10 (is_mass dims search in log space);
    # a fixed/locked mass is linear and must be converted.
    kind, _ = mass_slot
    if kind == "dim":
        return values
    return np.log10(np.maximum(values, 1e-300))
