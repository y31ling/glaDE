"""Batched GPU extend objective (whole DE population in one CUDA pass).

The extended-source counterpart of :mod:`core.optimize.batched`: per DE
generation, every candidate's lens/extend parameters become ``(C, 1, 1)``
tensors, the deflection table + finite-difference Jacobian + ray-shoot +
surface-brightness evaluation + pixel chi2 run batched over the whole
population, and the point-source constraint is scored with a vectorised port
of glafic's source-plane chi2 (closed-form linear solve + batched Nelder-Mead
over the inner source position).  The per-candidate weighted loss reuses the
exact ``ExtendLossConfig.combine`` + graded ``missing_img_penalty`` logic of
:class:`core.optimize.extend.ExtendObjective`, so batched and per-candidate
GPU paths score identically.

Scope (``can_batch_extend_gpu``): single lens plane, every lens model
GPU-supported, point constraints (if any) scored in the source plane
(``chi2_splane=1`` — glafic's image-plane mode falls back to the
per-candidate engine).  ``hubble`` may be optimizable: distances are
h-independent in glafic's Mpc/h convention, so only the time-delay factor is
rescaled per candidate (exact).

Cosmology caveat (same as the CPU path): ``omega/lambda/weos`` Bounds are NOT
search dimensions — `OptProblem` only exposes ``hubble`` — so they resolve to
their defaults here exactly as `Scene` does.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from ..format import schema
from ..format.config import GladeConfig
from ..format.values import Bounds, Fixed, SharedBounds
from .batched import _CHUNK_ENV, _SCHRAMM
from .extend import INVALID_LOSS, PENALTY_FLOOR, ExtendSpec
from .loss import ExtendLossConfig
from .problem import OptProblem


def can_batch_extend_gpu(cfg: GladeConfig) -> tuple[bool, str]:
    """Whether the batched extend GPU path applies; ``(ok, reason)``."""
    lens_z = cfg.redshifts.get("lens_z")
    zl = float(lens_z) if isinstance(lens_z, (int, float)) else None
    for comp in cfg.components:
        # an optimizable redshift (incl. a shared user variable; SharedBounds
        # is a Bounds subclass) must be checked for EVERY component: the
        # batched z slots are resolved once, so an extend source's optimizable
        # z would otherwise be silently pinned to lens_z. The per-candidate
        # ExtendObjective handles it correctly via make_scene.
        if isinstance(comp.z, Bounds):
            return False, f"component '{comp.name}' has an optimizable redshift"
        if schema.is_extend_model(comp.type):
            continue
        if not schema.supports("gpu", comp.type):
            return False, f"lens model '{comp.type}' has no GPU kernel"
        mspec = schema.model(comp.type)
        for j, p in enumerate(comp.params):
            if not isinstance(p, Bounds):
                continue
            pname = mspec.params[j].name if (mspec and j < len(mspec.params)) else ""
            if pname == "zs_fid":
                # the kernels resolve the fiducial-redshift distance scaling
                # with scipy on the CPU; it must stay a python scalar.
                return False, (f"component '{comp.name}' ({comp.type}) has an "
                               f"optimizable fiducial source redshift (p1)")
        zc = comp.z.value if isinstance(comp.z, Fixed) else zl
        if zl is None:
            zl = zc
        elif zc is not None and abs(zc - zl) > 1e-6:
            return False, "batched GPU is single-plane: components must share lens_z"
    has_point = ("constraint_file" in cfg.obs) or ("obs_positions_mas_list" in cfg.obs)
    if has_point:
        splane = cfg.get("chi2_splane")
        if not (isinstance(splane, (int, float)) and int(splane) == 1):
            return False, ("point constraints with chi2_splane != 1 (image-plane "
                           "inner solve) run per-candidate")
    return True, ""


class BatchedExtendGPUObjective:
    """Vectorized extend objective for scipy DE (``vectorized=True``).

    ``__call__`` receives an ``(ndim, popsize)`` array, returns ``(popsize,)``
    losses.  Not picklable / single-process by design (CUDA).
    """

    def __init__(self, problem: OptProblem, spec: ExtendSpec,
                 loss_cfg: ExtendLossConfig):
        self.problem = problem
        self.spec = spec
        self.loss_cfg = loss_cfg
        self._static = None
        self._torch = None

    # -- lazy imports / static state -------------------------------------
    def _ensure(self):
        if self._static is not None:
            return
        import torch  # noqa: PLC0415
        import rhongomyniad as rh  # noqa: PLC0415
        from rhongomyniad import constants as RK  # noqa: PLC0415
        from rhongomyniad import extend as rext  # noqa: PLC0415
        from rhongomyniad import sources as rsrc  # noqa: PLC0415
        from rhongomyniad.cosmology import Cosmology, tdelay_fac  # noqa: PLC0415
        from rhongomyniad.image_finder import sum_lensmodel  # noqa: PLC0415
        from rhongomyniad.lens_models import LensContext  # noqa: PLC0415

        self._torch = torch
        self._RK = RK
        self._rext = rext
        self._rsrc = rsrc
        self._sum_lensmodel = sum_lensmodel
        self._LensContext = LensContext
        self._Cosmology = Cosmology
        self._tdelay_fac_fn = tdelay_fac
        self.device = rh.get_device()
        self.dtype = torch.float64

        self._build_static()

    def _fixed_scalar(self, section, name, default):
        v = section.get(name, default)
        return float(v) if isinstance(v, (int, float)) else float(default)

    def _build_static(self):
        torch = self._torch
        rext = self._rext
        cfg = self.problem.cfg
        spec = self.spec
        cos, grid, rs = cfg.cosmology, cfg.grid, cfg.redshifts
        dev, dt = self.device, self.dtype

        omega = self._fixed_scalar(cos, "omega", 0.3)
        lam = self._fixed_scalar(cos, "lambda_cosmo", 0.7)
        weos = self._fixed_scalar(cos, "weos", -1.0)
        h_ref = self._fixed_scalar(cos, "hubble", 0.7)
        cosmo = self._Cosmology(omega=omega, lam=lam, weos=weos, hubble=h_ref)

        xmin = self._fixed_scalar(grid, "xmin", -0.5)
        ymin = self._fixed_scalar(grid, "ymin", -0.5)
        xmax = self._fixed_scalar(grid, "xmax", 0.5)
        ymax = self._fixed_scalar(grid, "ymax", 0.5)
        pix_ext = self._fixed_scalar(grid, "pix_ext", 0.01)
        pix_poi = self._fixed_scalar(grid, "pix_poi", 0.2)
        maxlev = int(self._fixed_scalar(grid, "maxlev", 5))

        # --- per-dimension injection map -------------------------------
        # comp slot source: ('fix', value) | ('dim', dim_index, is_log)
        dim_of = {d.target: k for k, d in enumerate(self.problem.dims)}
        hubble_dim = dim_of.get(("cosmo", "hubble"))

        lens_specs = []      # (glafic_key, [8 slot sources]) in set_lens order
        ext_specs = []       # (source model name, [8 slot sources])
        lens_z = self._fixed_scalar(rs, "lens_z", 0.3)
        for comp in cfg.components:
            mspec = schema.model(comp.type)
            zc = comp.z.value if isinstance(comp.z, Fixed) else lens_z
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
            while len(slots) < 8:
                slots.append(("fix", 0.0))
            slots = slots[:8]
            if schema.is_extend_model(comp.type):
                ext_specs.append((mspec.glafic_key, slots))
            else:
                lens_specs.append((mspec.glafic_key, slots))

        zl = float(lens_specs[0][1][0][1]) if lens_specs else lens_z

        # --- observed extended data ------------------------------------
        from astropy.io import fits  # noqa: PLC0415
        nx, ny = rext.grid_nxy(xmin, xmax, ymin, ymax, pix_ext)
        obs = np.asarray(fits.getdata(spec.extended_file), dtype=np.float64)
        if obs.shape != (ny, nx):
            raise ValueError(f"obs fits {obs.shape[::-1]} != grid {(nx, ny)}")
        obs_t = torch.tensor(obs, dtype=dt, device=dev)
        mask_t = None
        if spec.mask_file:
            m = np.asarray(fits.getdata(spec.mask_file))
            mask_t = torch.tensor(m > 0, device=dev)
        sec = dict(rext.SECONDARY_DEFAULTS)
        for k, v in (spec.secondary or {}).items():
            if k in sec:
                sec[k] = type(sec[k])(v) if isinstance(sec[k], (int, float)) else v
        noise_t, skymed, skysig = rext.calc_obsnoise(
            obs_t, mask_t, float(sec["noise_clip"]), float(sec["obs_gain"]),
            int(sec["obs_ncomb"]), float(sec["skyfix_value"]))
        if spec.noise_file:
            nz = np.asarray(fits.getdata(spec.noise_file), dtype=np.float64)
            noise_t = torch.tensor(nz, dtype=dt, device=dev)
        gx, gy = rext.pixel_centers(xmin, ymin, pix_ext, nx, ny, dev, dt)
        keep = ~mask_t if mask_t is not None else torch.ones_like(obs_t, dtype=torch.bool)

        # --- point constraints ------------------------------------------
        point_obs = []
        if spec.point_file and spec.n_point > 0:
            point_obs = rext.read_point_obs_file(spec.point_file)

        # --- priors / ranges ---------------------------------------------
        priors = rext.PriorStore()
        if spec.prior_file:
            priors.parse_parprior(spec.prior_file)

        # fiducial extend redshift + per-source distance ratios
        zs_list = [float(s[0][1]) for (_m, s) in ext_specs]
        i_fid = int(np.argmax(zs_list)) if zs_list else 0
        zs_fid = zs_list[i_fid] if zs_list else self._fixed_scalar(rs, "source_z", 1.0)
        ctx_ext = self._LensContext.build(cosmo, zl=zl, zs=zs_fid)
        dis_fac = [rext.disratio(cosmo, zl, zs_fid, z) for z in zs_list]

        ctx_pts = [self._LensContext.build(cosmo, zl=zl, zs=po.zs)
                   for po in point_obs]

        chunk = os.environ.get(_CHUNK_ENV)
        if chunk is not None:
            chunk = max(1, int(chunk))
        else:
            heavy = any(name in _SCHRAMM for name, _ in lens_specs)
            chunk = 32 if heavy else 128

        self._static = dict(
            cosmo=cosmo, h_ref=h_ref, zl=zl,
            lens_specs=lens_specs, ext_specs=ext_specs,
            hubble_dim=hubble_dim,
            obs=obs_t, mask=mask_t, keep=keep, noise=noise_t,
            skymed=skymed, sec=sec,
            gx=gx, gy=gy, nx=nx, ny=ny, pix_ext=pix_ext,
            pix_poi=pix_poi, maxlev=maxlev,
            xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
            point_obs=point_obs, ctx_ext=ctx_ext, ctx_pts=ctx_pts,
            dis_fac=dis_fac, priors=priors, chunk=chunk,
            free_src=[(spec.point_sources[k].free_x, spec.point_sources[k].free_y)
                      for k in range(len(point_obs))],
            init_src=[(spec.point_sources[k].init_x, spec.point_sources[k].init_y)
                      for k in range(len(point_obs))],
            nimg_obs=[po.nimg for po in point_obs],
        )

    # -- candidate decoding ------------------------------------------------
    def _slot_tensor(self, slot, arr_t):
        """slot -> float or (C,) tensor from the (ndim, C) candidate tensor."""
        kind = slot[0]
        if kind == "fix":
            return slot[1]
        _, dim_idx, is_log = slot
        col = arr_t[dim_idx]
        return self._torch.pow(10.0, col) if is_log else col

    def _build_lenses(self, arr_t, shape_suffix=(1, 1)):
        """[(name, params)] with per-candidate tensors of shape (C, *suffix)."""
        torch = self._torch
        C = arr_t.shape[1]
        out = []
        for name, slots in self._static["lens_specs"]:
            params = []
            for s in slots:
                v = self._slot_tensor(s, arr_t)
                if torch.is_tensor(v):
                    params.append(v.view(C, *([1] * len(shape_suffix))))
                else:
                    params.append(v)
            out.append((name, tuple(params)))
        return out

    # -- range checks (check_para_lens_all / check_para_ext_all) -----------
    def _range_violation(self, arr_t) -> "torch.Tensor":
        torch = self._torch
        st = self._static
        C = arr_t.shape[1]
        bad = torch.zeros(C, dtype=torch.bool, device=arr_t.device)

        def check(kind, i, j, val):
            nonlocal bad
            lo, hi = st["priors"].rng(kind, i, j)
            if torch.is_tensor(val):
                bad |= (val < lo) | (val > hi)
            elif val < lo or val > hi:
                bad |= torch.ones_like(bad)

        for i, (_n, slots) in enumerate(st["lens_specs"]):
            for j, s in enumerate(slots):
                check("lens", i, j, self._slot_tensor(s, arr_t))
        for i, (_n, slots) in enumerate(st["ext_specs"]):
            for j, s in enumerate(slots):
                check("ext", i, j, self._slot_tensor(s, arr_t))
        # cosmo box (glafic.h INIT_*; only hubble can vary here)
        if st["hubble_dim"] is not None:
            h = arr_t[st["hubble_dim"]]
            bad |= (h < 0.0) | (h > 3.0)
        return bad

    def _prior_chi2(self, arr_t, kind, specs) -> "torch.Tensor":
        """Vectorised gaussian/log parameter priors (rarely used)."""
        torch = self._torch
        st = self._static
        C = arr_t.shape[1]
        c2 = torch.zeros(C, dtype=self.dtype, device=arr_t.device)
        store = st["priors"]
        for i, (_n, slots) in enumerate(specs):
            for j, s in enumerate(slots):
                pr = store.pri(kind, i, j)
                if pr is None:
                    continue
                med, sig = pr
                v = self._slot_tensor(s, arr_t)
                if not torch.is_tensor(v):
                    v = torch.full((C,), float(v), dtype=self.dtype,
                                   device=arr_t.device)
                if sig > 0.0:
                    c2 = c2 + (v - med) ** 2 / (sig * sig)
                elif sig < 0.0:
                    badlog = (v < 0.0) | (med < 0.0)
                    lp = torch.log10(torch.clamp(v, min=1e-300))
                    term = (lp - math.log10(max(med, 1e-300))) ** 2 / (sig * sig)
                    c2 = c2 + torch.where(badlog,
                                          torch.full_like(term, PENALTY_FLOOR * 10),
                                          term)
        return c2

    # -- extended pixel chi2 ------------------------------------------------
    def _pixel_chi2(self, arr_t) -> "torch.Tensor":
        torch = self._torch
        st = self._static
        rext, rsrc = self._rext, self._rsrc
        C = arr_t.shape[1]
        gx, gy = st["gx"], st["gy"]

        lenses = self._build_lenses(arr_t)
        ax, ay, pxx, pxy, pyx, pyy = rext.deflection_table(
            self._sum_lensmodel, st["ctx_ext"], lenses, gx, gy,
            st["mask"], self._RK.DEF_SMALLCORE)
        if ax.dim() == 2:                      # all-fixed lens stack
            ax = ax.unsqueeze(0).expand(C, -1, -1)
            ay = ay.unsqueeze(0).expand(C, -1, -1)
            pxx = pxx.unsqueeze(0).expand(C, -1, -1)
            pxy = pxy.unsqueeze(0).expand(C, -1, -1)
            pyx = pyx.unsqueeze(0).expand(C, -1, -1)
            pyy = pyy.unsqueeze(0).expand(C, -1, -1)

        sec = st["sec"]
        model = torch.zeros_like(ax)
        for k, (name, slots) in enumerate(st["ext_specs"]):
            fac = st["dis_fac"][k]
            sx = gx - ax * fac
            sy = gy - ay * fac
            p = [self._slot_tensor(s, arr_t) for s in slots]
            p = [v.view(C, 1, 1) if torch.is_tensor(v) else v for v in p]
            f = rsrc.source_all(
                rsrc.SOURCE_IDS[name], sx, sy,
                p[2], p[3], p[4], p[5], p[6], p[7],
                pxx * fac, pxy * fac, pyx * fac, pyy * fac,
                pix=st["pix_ext"], pix_ext=st["pix_ext"],
                imag_ceil=self._RK.DEF_IMAG_CEIL,
                flag_extref=int(sec["flag_extref"]),
                source_refr0=float(sec["source_refr0"]),
                num_pixint=int(sec["num_pixint"]),
                flag_extnorm=int(sec["flag_extnorm"]),
                smallcore=self._RK.DEF_SMALLCORE)
            model = model + p[1] * f

        skymed = st["skymed"]
        if int(sec.get("skyfix", 0)) == 1:
            skymed = float(sec.get("skyfix_value", skymed))
        model = model + skymed
        keep = st["keep"]
        diff = (model - st["obs"]) * keep
        sig = torch.where(keep, st["noise"], torch.ones_like(st["noise"]))
        return ((diff * diff) / (sig * sig)).sum(dim=(-2, -1))

    # -- batched source-plane point chi2 -------------------------------------
    def _lensmodel_pts(self, lenses, ctx, x: np.ndarray, y: np.ndarray,
                       need_phi: bool):
        """Batched calcimage at a few obs points: returns (C, n) tensors."""
        torch = self._torch
        tx = torch.tensor(x, dtype=self.dtype, device=self.device).view(1, -1)
        ty = torch.tensor(y, dtype=self.dtype, device=self.device).view(1, -1)
        # lens params reshaped (C, 1) for this call
        lenses2 = []
        for name, p in lenses:
            q = tuple(v.view(-1, 1) if torch.is_tensor(v) else v for v in p)
            lenses2.append((name, q))
        ax, ay, kap, g1, g2, phi, muinv = self._sum_lensmodel(
            ctx, lenses2, tx, ty, need_kg=True, need_phi=need_phi,
            smallcore=self._RK.DEF_SMALLCORE)
        td = None
        if need_phi and phi is not None:
            td = ctx.tdelay_fac * (0.5 * (ax * ax + ay * ay) - phi)
        return ax, ay, kap, g1, g2, muinv, td

    def _point_chi2(self, arr_t):
        """Vectorised splane chi2: returns (pos, flux, td, prior) each a (C,)
        numpy array plus the per-source solved positions [(xs(C,), ys(C,)), ...]
        (numpy).

        The per-image caches (5 batched lensmodel sweeps) are computed on the
        GPU; the inner Nelder-Mead source solve then runs vectorised in numpy —
        the cached arrays are only (C, nimg), so CPU vector math beats tens of
        thousands of tiny CUDA kernel launches by orders of magnitude."""
        torch = self._torch
        st = self._static
        C = arr_t.shape[1]
        out = [np.zeros(C) for _ in range(4)]
        solved = []
        if not st["point_obs"]:
            return out, solved

        lenses = self._build_lenses(arr_t, shape_suffix=(1,))
        sec = st["sec"]
        usemag = int(sec.get("chi2_usemag", 0))
        imag_ceil = self._RK.DEF_IMAG_CEIL
        hh_fd = st["pix_poi"] * (0.5 ** (st["maxlev"] - 1)) * 0.1
        hh_st = st["pix_poi"] * (0.5 ** (st["maxlev"] - 1))
        PEN_RANGE = self._rext.CHI2PEN_RANGE
        PEN_NIMG = self._rext.CHI2PEN_NIMG
        PEN_PARITY = self._rext.CHI2PEN_PARITY

        # per-candidate time-delay factor: distances are h-independent in
        # glafic's Mpc/h units, td scales exactly as 1/h
        if st["hubble_dim"] is not None:
            td_scale = (st["h_ref"] / arr_t[st["hubble_dim"]].cpu().numpy()
                        ).reshape(C, 1)
        else:
            td_scale = np.ones((C, 1))

        def _np(t):
            a = t.cpu().numpy()
            return np.broadcast_to(a, (C, a.shape[1])).copy() if a.shape[0] != C else a

        for s_i, po in enumerate(st["point_obs"]):
            ctx = st["ctx_pts"][s_i]
            obs_x, obs_y = po.rows[:, 0], po.rows[:, 1]
            nimg = po.nimg
            ax_t, ay_t, kap_t, g1_t, g2_t, muinv_t, td_t = self._lensmodel_pts(
                lenses, ctx, obs_x, obs_y, need_phi=True)
            m1x = self._lensmodel_pts(lenses, ctx, obs_x + 0.5 * hh_fd, obs_y, False)[5]
            m2x = self._lensmodel_pts(lenses, ctx, obs_x - 0.5 * hh_fd, obs_y, False)[5]
            m1y = self._lensmodel_pts(lenses, ctx, obs_x, obs_y + 0.5 * hh_fd, False)[5]
            m2y = self._lensmodel_pts(lenses, ctx, obs_x, obs_y - 0.5 * hh_fd, False)[5]

            ax, ay = _np(ax_t), _np(ay_t)
            kap, g1, g2 = _np(kap_t), _np(g1_t), _np(g2_t)
            muinv, td = _np(muinv_t), _np(td_t) * td_scale
            m1x, m2x, m1y, m2y = _np(m1x), _np(m2x), _np(m1y), _np(m2y)

            norm = 1.0 / (muinv + imag_ceil)
            n2 = norm * norm
            a00 = n2 * ((1.0 - kap + g1) ** 2 + g2 * g2)
            a01 = 2.0 * n2 * (g2 * (1.0 - kap))
            a11 = n2 * ((1.0 - kap - g1) ** 2 + g2 * g2)
            mu00 = norm * (1.0 - kap + g1)
            mu01 = norm * g2
            mu10 = norm * g2
            mu11 = norm * (1.0 - kap - g1)
            uobs_x = obs_x[None, :] - ax
            uobs_y = obs_y[None, :] - ay
            mag = norm
            dmudx = -mag * mag * (m1x - m2x) / hh_fd
            dmudy = -mag * mag * (m1y - m2y) / hh_fd

            perr = po.rows[None, :, 3]
            ferr = po.rows[None, :, 4]
            flux = po.rows[None, :, 2]
            tdo = po.rows[None, :, 5]
            tderr = po.rows[None, :, 6]
            parity = po.rows[None, :, 7]
            use_pos = perr > 0.0
            use_flux = ferr > 0.0
            use_td = tderr > 0.0
            inv_p2 = np.where(use_pos, 1.0 / np.where(use_pos, perr, 1.0) ** 2, 0.0)
            inv_f2 = np.where(use_flux, 1.0 / np.where(use_flux, ferr, 1.0) ** 2, 0.0)
            inv_t2 = np.where(use_td, 1.0 / np.where(use_td, tderr, 1.0) ** 2, 0.0)

            # closed-form source position (priors on poi x/y included)
            AA00 = (a00 * inv_p2).sum(axis=1)
            AA01 = (a01 * inv_p2).sum(axis=1)
            AA11 = (a11 * inv_p2).sum(axis=1)
            BB0 = ((a00 * uobs_x + a01 * uobs_y) * inv_p2).sum(axis=1)
            BB1 = ((a01 * uobs_x + a11 * uobs_y) * inv_p2).sum(axis=1)
            for j, idx in ((1, 0), (2, 1)):
                pr = st["priors"].pri("poi", s_i, j)
                if pr is not None and pr[1] > 0.0:
                    if idx == 0:
                        AA00 = AA00 + 1.0 / (pr[1] ** 2)
                        BB0 = BB0 + pr[0] / (pr[1] ** 2)
                    else:
                        AA11 = AA11 + 1.0 / (pr[1] ** 2)
                        BB1 = BB1 + pr[0] / (pr[1] ** 2)
            det = AA00 * AA11 - AA01 * AA01
            ux0 = (AA11 * BB0 - AA01 * BB1) / det
            uy0 = (AA00 * BB1 - AA01 * BB0) / det

            lo1, hi1 = st["priors"].rng("poi", s_i, 1)
            lo2, hi2 = st["priors"].rng("poi", s_i, 2)
            tdfac = ctx.tdelay_fac * td_scale
            pri_x = st["priors"].pri("poi", s_i, 1)
            pri_y = st["priors"].pri("poi", s_i, 2)

            def f_eval(xs, ys):
                """splane components at (xs, ys) each (C,); returns
                (tot, pos, fluxc, tdc, prior) all (C,) numpy."""
                ux = uobs_x - xs[:, None]
                uy = uobs_y - ys[:, None]
                cc = (a00 * ux * ux + (a01 + a01) * ux * uy + a11 * uy * uy) * inv_p2
                cc = np.where(cc < 0.0, PEN_NIMG, cc)
                pos = cc.sum(axis=1)
                dxm = mu00 * ux + mu01 * uy
                dym = mu10 * ux + mu11 * uy
                mumod = mag - (dmudx * dxm + dmudy * dym)
                tdmod = td - tdfac * ((uobs_x - obs_x[None, :]) * ux
                                      + (uobs_y - obs_y[None, :]) * uy)
                absmu = np.abs(mumod)
                logmu = 2.5 * np.log10(np.maximum(absmu, 1e-300))
                if usemag == 0:
                    f1 = (np.abs(flux * mumod) * inv_f2).sum(axis=1)
                    f2 = (mumod * mumod * inv_f2).sum(axis=1)
                else:
                    f1 = ((flux + logmu) * inv_f2).sum(axis=1)
                    f2 = inv_f2.sum(axis=1) * np.ones_like(f1)
                t1 = ((tdo - tdmod) * inv_t2).sum(axis=1)
                t2 = inv_t2.sum(axis=1) * np.ones(C)
                flux0 = np.where(f2 > 0.0, f1 / np.maximum(f2, 1e-300), 1.0)[:, None]
                td0 = np.where(t2 > 0.0, t1 / np.maximum(t2, 1e-300), 0.0)[:, None]
                if usemag == 0:
                    fterm = (np.abs(flux) - absmu * flux0) ** 2 * inv_f2
                elif usemag == -1:
                    fterm = (np.abs(flux) - absmu) ** 2 * inv_f2
                else:
                    fterm = (flux + logmu - flux0) ** 2 * inv_f2
                fluxc = fterm.sum(axis=1)
                bad_par = ((parity != 0.0) & (parity * mumod < 0.0)).any(axis=1)
                fluxc = np.where(bad_par, PEN_PARITY, fluxc)
                tdc = ((tdo - tdmod - td0) ** 2 * inv_t2).sum(axis=1)
                prior = np.zeros(C)
                if pri_x is not None and pri_x[1] > 0.0:
                    prior = prior + (pri_x[0] - xs) ** 2 / (pri_x[1] ** 2)
                if pri_y is not None and pri_y[1] > 0.0:
                    prior = prior + (pri_y[0] - ys) ** 2 / (pri_y[1] ** 2)
                # out-of-range source position: glafic returns c2[0]=c2[4]=pen,
                # c2[1..3]=0 (opt_point.c:429-433)
                oob = (xs < lo1) | (xs > hi1) | (ys < lo2) | (ys > hi2)
                tot = np.where(oob, PEN_RANGE, pos + fluxc + tdc + prior)
                pos = np.where(oob, 0.0, pos)
                fluxc = np.where(oob, 0.0, fluxc)
                tdc = np.where(oob, 0.0, tdc)
                prior = np.where(oob, PEN_RANGE, prior)
                return tot, pos, fluxc, tdc, prior

            free_x, free_y = st["free_src"][s_i]
            init_x, init_y = st["init_src"][s_i]
            nd = int(free_x) + int(free_y)

            if nd > 0 and nimg >= 2:
                xs_b, ys_b = self._nm2(f_eval, ux0, uy0, hh_st, free_x, free_y,
                                       init_x, init_y)
            elif nimg == 1:
                xs_b = uobs_x[:, 0] if free_x else np.full(C, init_x)
                ys_b = uobs_y[:, 0] if free_y else np.full(C, init_y)
            else:
                xs_b = np.full(C, init_x)
                ys_b = np.full(C, init_y)

            tot, pos, fluxc, tdc, prior = f_eval(xs_b, ys_b)
            out[0] = out[0] + pos
            out[1] = out[1] + fluxc
            out[2] = out[2] + tdc
            out[3] = out[3] + prior
            solved.append((xs_b, ys_b))
        return out, solved

    def _nm2(self, f_eval, ux0, uy0, hh, free_x, free_y, init_x, init_y):
        """Vectorised (numpy) Nelder-Mead over the 1-2 dim source position,
        mirroring glafic's amoeba (same coefficients and convergence rule;
        per-candidate freeze on convergence)."""
        C = ux0.shape[0]
        nd = int(free_x) + int(free_y)

        def full_xy(p2):
            # p2: (C, nd) -> xs, ys (C,)
            if nd == 2:
                return p2[:, 0], p2[:, 1]
            if free_x:
                return p2[:, 0], np.full(C, init_y)
            return np.full(C, init_x), p2[:, 0]

        nv = nd + 1
        v = np.zeros((C, nv, nd))
        if nd == 2:
            v[:, 0, 0], v[:, 0, 1] = ux0, uy0
            v[:, 1, 0], v[:, 1, 1] = ux0 + hh, uy0
            v[:, 2, 0], v[:, 2, 1] = ux0, uy0 + hh
        elif free_x:
            v[:, 0, 0] = ux0
            v[:, 1, 0] = ux0 + hh
        else:
            v[:, 0, 0] = uy0
            v[:, 1, 0] = uy0 + hh

        def fv(p2):
            xs, ys = full_xy(p2)
            return f_eval(xs, ys)[0]

        f = np.stack([fv(v[:, r, :]) for r in range(nv)], axis=1)   # (C, nv)
        ALPHA, BETA, GAMMA = 1.0, 0.5, 2.0
        rows = np.arange(C)
        for _ in range(self._rext.NMAX_AMOEBA_POINT):
            vg = np.argmax(f, axis=1)
            vs = np.argmin(f, axis=1)
            fmax = f[rows, vg]
            fmin = f[rows, vs]
            rtol = 2.0 * np.abs(fmax - fmin) / (np.abs(fmax) + np.abs(fmin) + 1e-300)
            active = (rtol >= self._rext.TOL_AMOEBA) &                 (np.abs(fmax) >= 0.1 * self._rext.TOL_AMOEBA)
            if not active.any():
                break
            vgx = v[rows, vg]                                # (C, nd)
            vm = (v.sum(axis=1) - vgx) / nd
            vr = vm + ALPHA * (vm - vgx)
            fr = fv(vr)

            f_sorted = np.sort(f, axis=1)
            f_second_worst = f_sorted[:, -2]
            accept_r = (fr < f_second_worst) & (fr >= fmin) & active
            expand = (fr < fmin) & active
            contract = (fr >= f_second_worst) & active

            ve = vm + GAMMA * (vr - vm)
            fe = fv(ve)
            use_e = expand & (fe < fr)
            new_v = np.where(use_e[:, None], ve,
                             np.where((expand | accept_r)[:, None], vr, vgx))
            new_f = np.where(use_e, fe, np.where(expand | accept_r, fr, fmax))

            vc = np.where((contract & (fr < fmax))[:, None],
                          vm + BETA * (vr - vm),       # outside contraction
                          vm - BETA * (vm - vgx))      # inside contraction
            fc = fv(vc)
            c_ok = contract & (fc < fmax)
            new_v = np.where(c_ok[:, None], vc, new_v)
            new_f = np.where(c_ok, fc, new_f)

            upd = active
            v[rows[upd], vg[upd]] = new_v[upd]
            f[rows[upd], vg[upd]] = new_f[upd]

            shrink = contract & ~c_ok
            if shrink.any():
                vsx = v[rows, vs][:, None, :]                  # (C,1,nd)
                v_shrunk = vsx + (v - vsx) / 2.0
                keep_best = np.zeros((C, nv, 1), dtype=bool)
                keep_best[rows, vs, 0] = True
                v_new = np.where(keep_best, v, v_shrunk)
                v = np.where(shrink[:, None, None], v_new, v)
                f_new = np.stack([fv(v[:, r, :]) for r in range(nv)], axis=1)
                f = np.where(shrink[:, None], f_new, f)

        vs = np.argmin(f, axis=1)
        best = v[rows, vs]
        if nd == 2:
            return best[:, 0], best[:, 1]
        if free_x:
            return best[:, 0], np.full(C, init_y)
        return np.full(C, init_x), best[:, 0]

    # -- batched missing-image counter (uniform-grid finder) ------------------
    def _missing_counts(self, arr_t, solved) -> np.ndarray:
        torch = self._torch
        st = self._static
        C = arr_t.shape[1]
        miss = np.zeros(C, dtype=int)
        if not st["point_obs"] or self.loss_cfg.missing_img_penalty <= 0.0:
            return miss

        dp = st["pix_poi"] / (2 ** (st["maxlev"] - 1))
        nxf = int(math.ceil((st["xmax"] - st["xmin"]) / dp)) + 1
        nyf = int(math.ceil((st["ymax"] - st["ymin"]) / dp)) + 1
        xs_ax = torch.linspace(st["xmin"], st["xmin"] + (nxf - 1) * dp, nxf,
                               device=self.device, dtype=self.dtype)
        ys_ax = torch.linspace(st["ymin"], st["ymin"] + (nyf - 1) * dp, nyf,
                               device=self.device, dtype=self.dtype)
        gxf, gyf = torch.meshgrid(xs_ax, ys_ax, indexing="xy")

        lenses_grid = self._build_lenses(arr_t)            # (C,1,1) params
        sc2 = self._RK.DEF_SMALLCORE ** 2

        for s_i, po in enumerate(st["point_obs"]):
            ctx = st["ctx_pts"][s_i]
            xs_np, ys_np = solved[s_i]
            xs_src = torch.tensor(xs_np, dtype=self.dtype, device=self.device)
            ys_src = torch.tensor(ys_np, dtype=self.dtype, device=self.device)
            ax, ay, *_ = self._sum_lensmodel(ctx, lenses_grid, gxf, gyf,
                                             need_kg=False, need_phi=False,
                                             smallcore=self._RK.DEF_SMALLCORE)
            if ax.dim() == 2:
                ax = ax.unsqueeze(0).expand(C, -1, -1)
                ay = ay.unsqueeze(0).expand(C, -1, -1)
            sxg = gxf.unsqueeze(0) - ax
            syg = gyf.unsqueeze(0) - ay
            xsv = xs_src.view(C, 1, 1)
            ysv = ys_src.view(C, 1, 1)
            bl_x, bl_y = sxg[:, :-1, :-1], syg[:, :-1, :-1]
            br_x, br_y = sxg[:, :-1, 1:], syg[:, :-1, 1:]
            tl_x, tl_y = sxg[:, 1:, :-1], syg[:, 1:, :-1]
            tr_x, tr_y = sxg[:, 1:, 1:], syg[:, 1:, 1:]

            def tri(axx, ayy, bxx, byy, cxx, cyy):
                d1x, d1y = xsv - axx, ysv - ayy
                d2x, d2y = xsv - bxx, ysv - byy
                d3x, d3y = xsv - cxx, ysv - cyy
                d12 = d1x * d2y - d1y * d2x
                d23 = d2x * d3y - d2y * d3x
                d31 = d3x * d1y - d3y * d1x
                return (((d12 >= 0) & (d23 >= 0) & (d31 >= 0))
                        | ((d12 <= 0) & (d23 <= 0) & (d31 <= 0)))

            in_A = tri(bl_x, bl_y, tr_x, tr_y, br_x, br_y)
            in_B = tri(bl_x, bl_y, tr_x, tr_y, tl_x, tl_y)
            ogx = gxf[:-1, :-1].unsqueeze(0).expand_as(in_A)
            ogy = gyf[:-1, :-1].unsqueeze(0).expand_as(in_A)
            idx_A = torch.nonzero(in_A, as_tuple=False)
            idx_B = torch.nonzero(in_B, as_tuple=False)
            if idx_A.numel() + idx_B.numel() == 0:
                miss += st["nimg_obs"][s_i]
                continue

            def seeds(idx, fx, fy):
                ci = idx[:, 0]
                return (ci, ogx[ci, idx[:, 1], idx[:, 2]] + fx * dp,
                        ogy[ci, idx[:, 1], idx[:, 2]] + fy * dp)

            cA, xA, yA = seeds(idx_A, 0.667, 0.333)
            cB, xB, yB = seeds(idx_B, 0.333, 0.667)
            cand = torch.cat([cA, cB])
            xi = torch.cat([xA, xB]).clone()
            yi = torch.cat([yA, yB]).clone()
            x0s, y0s = xi.clone(), yi.clone()

            # gather per-point lens params
            lenses_pts = []
            for name, p in lenses_grid:
                q = tuple(v.view(-1)[cand] if torch.is_tensor(v) else v for v in p)
                lenses_pts.append((name, q))
            xs_tgt = xs_src[cand]
            ys_tgt = ys_src[cand]

            kap = g1 = g2 = None
            for _ in range(8):
                axp, ayp, kap, g1, g2, _phi, _mi = self._sum_lensmodel(
                    ctx, lenses_pts, xi, yi, need_kg=True, need_phi=False,
                    smallcore=self._RK.DEF_SMALLCORE)
                pxx = kap + g1
                pyy = kap - g1
                pxy = g2
                ff = xs_tgt - xi + axp
                gg = ys_tgt - yi + ayp
                mm = (1.0 - pxx) * (1.0 - pyy) - pxy * pxy
                xi = xi + ((1.0 - pyy) * ff + pxy * gg) / mm
                yi = yi + ((1.0 - pxx) * gg + pxy * ff) / mm

            muinv = (1.0 - kap) ** 2 - (g1 * g1 + g2 * g2)
            mag = 1.0 / (muinv + self._RK.DEF_IMAG_CEIL)
            dist2 = (xi - x0s) ** 2 + (yi - y0s) ** 2
            ok = dist2 <= (2.0 * dp * dp)

            cand_np = cand.cpu().numpy()
            xi_np, yi_np, mag_np = xi.cpu().numpy(), yi.cpu().numpy(), mag.cpu().numpy()
            ok_np = ok.cpu().numpy()
            tol2 = 10.0 * self._RK.DEF_MAX_POI_TOL ** 2
            found = [[] for _ in range(C)]
            for kk in range(len(cand_np)):
                if not ok_np[kk]:
                    continue
                ci = int(cand_np[kk])
                x, y, m = float(xi_np[kk]), float(yi_np[kk]), float(mag_np[kk])
                dup = False
                for xj, yj, mj in found[ci]:
                    if ((x - xj) ** 2 + (y - yj) ** 2) / max(abs(m * mj), 1e-300) <= tol2:
                        dup = True
                        break
                if not dup:
                    found[ci].append((x, y, m))
            n_obs = st["nimg_obs"][s_i]
            for ci in range(C):
                miss[ci] += max(0, n_obs - len(found[ci]))
        return miss

    # -- evaluation ----------------------------------------------------------
    def _evaluate_components(self, arr: np.ndarray):
        """(ndim, C) candidates -> (comp (C, 8) ndarray, missing (C,) ints)."""
        self._ensure()
        torch = self._torch
        st = self._static
        arr_t = torch.tensor(arr, dtype=self.dtype, device=self.device)
        C = arr.shape[1]

        comp = np.zeros((C, 8))
        bad_np = self._range_violation(arr_t).cpu().numpy()
        comp[:, 7] = np.where(bad_np, self._rext.CHI2PEN_RANGE, 0.0)

        pixel = self._pixel_chi2(arr_t)
        comp[:, 4] = pixel.cpu().numpy()
        comp[:, 5] = self._prior_chi2(arr_t, "ext", st["ext_specs"]).cpu().numpy()

        (pos, fluxc, tdc, prior_pt), solved = self._point_chi2(arr_t)
        comp[:, 0] = pos
        comp[:, 1] = fluxc
        comp[:, 2] = tdc
        comp[:, 3] = prior_pt

        comp[:, 6] = self._prior_chi2(arr_t, "lens", st["lens_specs"]).cpu().numpy()

        # glafic short-circuits range violations: only out[7] is set
        # (opt_lens.c:346-353); match that for component-level parity.
        comp[bad_np, :7] = 0.0

        missing = self._missing_counts(arr_t, solved)
        missing[bad_np] = 0
        return comp, missing

    def _combine(self, comp: np.ndarray, missing: np.ndarray) -> np.ndarray:
        """Vectorised ExtendObjective.evaluate_one combination logic."""
        lc = self.loss_cfg
        C = comp.shape[0]
        loss = np.empty(C)
        for c in range(C):
            row = comp[c]
            mp = lc.missing_img_penalty
            if mp > 0.0 and missing[c] > 0:
                if any(row[i] >= PENALTY_FLOOR for i in (4, 5, 6, 7)):
                    loss[c] = INVALID_LOSS
                    continue
                base = lc.w_ext * row[4] + lc.w_prior * (row[5] + row[6])
                loss[c] = base + missing[c] * mp
                continue
            if any(v >= PENALTY_FLOOR for v in row):
                loss[c] = INVALID_LOSS
                continue
            loss[c] = lc.combine(tuple(row))
        return loss

    def __call__(self, params_arr) -> np.ndarray:
        arr = np.asarray(params_arr, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        popsize = arr.shape[1]
        try:
            self._ensure()
            chunk = self._static["chunk"]
            loss = np.empty(popsize)
            for s in range(0, popsize, chunk):
                sub = arr[:, s:s + chunk]
                comp, missing = self._evaluate_components(sub)
                loss[s:s + sub.shape[1]] = self._combine(comp, missing)
            return loss
        except Exception:  # noqa: BLE001 - never let DE die on a GPU hiccup
            return np.full(popsize, INVALID_LOSS)

    def components_for(self, candidate) -> Optional[tuple]:
        """c2calc_each-style 8-tuple for one candidate (or None on failure)."""
        try:
            self._ensure()
            arr = np.asarray(candidate, dtype=float).reshape(-1, 1)
            comp, _ = self._evaluate_components(arr)
            return tuple(float(v) for v in comp[0])
        except Exception:  # noqa: BLE001
            return None

    def evaluate_one(self, candidate) -> float:
        return float(self.__call__(np.asarray(candidate, dtype=float))[0])
