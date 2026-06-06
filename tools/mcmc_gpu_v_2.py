#!/usr/bin/env python3
"""
mcmc_gpu_v_2.py
===============
MCMC_GPU.py 的优化版本，应用 A + B + C + D：

A. **牛顿迭代用缓存双线性插值**
   v1 在每次牛顿迭代里都重新调用 `sum_lensmodel(ctx, fixed_lenses, xi, yi, ...)`
   解析求 Sersic + 主透镜的偏角/收敛/剪切。这等于"固定透镜场缓存"在 Newton
   阶段失效。v2 改成对缓存 (ax, ay, κ, γ1, γ2) 网格做双线性插值——子晕场仍解
   析叠加，因而中心发散保持精确。

B. **删除 phi / td 计算**
   loss 仅用位置和放大率，时延 td 不参与；v1 还多调一次
   `sum_lensmodel(..., need_phi=True)`。v2 全部砍掉，td 占位 0。

C. **牛顿早期终止**
   当所有候选的 max(|dx|, |dy|) < newton_tol（默认 1e-8）时立即跳出，
   多数候选 2–3 次就收敛。

D. **更高 GPU 利用率**
   默认 `--nwalkers 64`、`--dtype float32`。GPU 上 fp32 一般快 2–4×；
   位置级精度 ~1e-6 远小于观测 σ（mas 级）。

输入/输出与 MCMC_GPU.py 完全一致。
"""

from __future__ import annotations

import sys
import os
import time
import math
import argparse
import glob
from pathlib import Path
import numpy as np

# ── 触发 v1 的路径 / 运行时环境配置 ─────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# 注：导入 MCMC_GPU 会运行其顶部的 sys.path 配置 + setup_runtime_env，
# 因此此后即可直接 import torch / rhongomyniad。
import MCMC_GPU  # noqa: F401
from MCMC_GPU import (
    OBS_POS, OBS_MAG, OBS_MAG_ERR, OBS_POS_SIGMA,
    CENTER_OFFSET_X, CENTER_OFFSET_Y,
    LOSS_COEF_A, LOSS_COEF_B, LOSS_PENALTY,
    OMEGA, LAMBDA_COSMO, WEOS, HUBBLE,
    SOURCE_Z, SOURCE_X, SOURCE_Y, LENS_Z,
    XMIN, YMIN, XMAX, YMAX, PIX_EXT, PIX_POI, MAXLEV,
    DEFAULT_LENS_PARAMS, DEFAULT_MAIN_LENS_KEY,
    DEFAULT_MCMC_NSTEPS, DEFAULT_MCMC_BURNIN,
    DEFAULT_MCMC_THIN, DEFAULT_MCMC_PERTURBATION, DEFAULT_MCMC_PROGRESS,
    MCMC_SEARCH_RADIUS, MCMC_PM_LOG_M_MIN, MCMC_PM_LOG_M_MAX,
    parse_best_params, load_baseline_lens_params, find_bestfit_dir,
    build_initial_params, build_param_names,
    plot_corner, plot_trace, plot_mass_1d, save_posterior,
    build_fixed_cache, _re2_point, _batched_pointmass_fields, _tri_contains,
)

import torch  # noqa: E402
import rhongomyniad as rh  # noqa: E402
from rhongomyniad import constants as K  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

# ── v2 默认值 ──────────────────────────────────────────────────────────
V2_DEFAULT_NWALKERS = 64       # D：更大 batch
V2_DEFAULT_DTYPE    = "float32"  # D：fp32
V2_NEWTON_TOL       = 1.0e-8   # C：早期终止阈值
V2_MAX_NEWTON_ITER  = 8


# ╔══════════════════════════════════════════════════════════════════════╗
# ║          A：缓存双线性插值（同时取 ax, ay, kap, g1, g2）              ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _bilinear_5(cache, xi, yi):
    """对 (xi, yi) 在缓存网格上做双线性插值，一次返回 5 个场。

    cache 字段：ax, ay, kap, g1, g2，形状 (ny, nx)。
    xi, yi: 任意 1D tensor。返回 5 个 1D tensor，形状与 xi 相同。
    """
    gx0 = cache["gx"]
    gy0 = cache["gy"]
    dp = cache["dp"]
    nx = cache["nx"]
    ny = cache["ny"]

    x0 = gx0[0, 0]   # tensor scalar，避免 .item() 同步
    y0 = gy0[0, 0]
    fx = (xi - x0) / dp
    fy = (yi - y0) / dp
    ix = fx.floor().long().clamp(0, nx - 2)
    iy = fy.floor().long().clamp(0, ny - 2)
    rx = fx - ix.to(fx.dtype)
    ry = fy - iy.to(fy.dtype)
    rx = rx.clamp(0.0, 1.0)
    ry = ry.clamp(0.0, 1.0)
    one_rx = 1.0 - rx
    one_ry = 1.0 - ry
    w00 = one_rx * one_ry
    w10 = rx     * one_ry
    w01 = one_rx * ry
    w11 = rx     * ry

    ix1 = ix + 1
    iy1 = iy + 1

    out = []
    for fld in (cache["ax"], cache["ay"], cache["kap"],
                cache["g1"], cache["g2"]):
        v = (w00 * fld[iy,  ix]  + w10 * fld[iy,  ix1]
             + w01 * fld[iy1, ix]  + w11 * fld[iy1, ix1])
        out.append(v)
    return out  # ax_fix, ay_fix, kap_fix, g1_fix, g2_fix


# ╔══════════════════════════════════════════════════════════════════════╗
# ║       v2 批量求解器：双线性 Newton + 早终止 + 无 phi              ║
# ╚══════════════════════════════════════════════════════════════════════╝

def batched_point_solve_v2(sx_t, sy_t, log_m_t, xs_src, ys_src, cache,
                            max_iter=V2_MAX_NEWTON_ITER,
                            newton_tol=V2_NEWTON_TOL):
    B, Kn = sx_t.shape
    gx, gy = cache["gx"], cache["gy"]
    ny, nx = gx.shape
    dp = cache["dp"]
    ctx = cache["ctx"]
    sc2 = K.DEF_SMALLCORE ** 2

    # ── 1. 构造完整偏角场（缓存 + 子晕扰动）───────────────────────────
    ax_p, ay_p, _, _, _ = _batched_pointmass_fields(
        sx_t, sy_t, log_m_t, ctx, gx, gy)
    ax = cache["ax"].unsqueeze(0) + ax_p
    ay = cache["ay"].unsqueeze(0) + ay_p

    # ── 2. 在源平面寻找包含源的三角形（与 v1 相同）─────────────────────
    sx_grid = gx.unsqueeze(0) - ax
    sy_grid = gy.unsqueeze(0) - ay
    bl_x = sx_grid[:, :-1, :-1]; bl_y = sy_grid[:, :-1, :-1]
    br_x = sx_grid[:, :-1,  1:]; br_y = sy_grid[:, :-1,  1:]
    tl_x = sx_grid[:,  1:, :-1]; tl_y = sy_grid[:,  1:, :-1]
    tr_x = sx_grid[:,  1:,  1:]; tr_y = sy_grid[:,  1:,  1:]
    in_A = _tri_contains(xs_src, ys_src, bl_x, bl_y, tr_x, tr_y, br_x, br_y)
    in_B = _tri_contains(xs_src, ys_src, bl_x, bl_y, tr_x, tr_y, tl_x, tl_y)
    ox_grid = gx[:-1, :-1].unsqueeze(0).expand_as(in_A)
    oy_grid = gy[:-1, :-1].unsqueeze(0).expand_as(in_A)
    idx_A = torch.nonzero(in_A, as_tuple=False)
    idx_B = torch.nonzero(in_B, as_tuple=False)
    if idx_A.numel() + idx_B.numel() == 0:
        return [[] for _ in range(B)]

    def _seeds(idx, off_x, off_y):
        cfg, j, i = idx[:, 0], idx[:, 1], idx[:, 2]
        return cfg, ox_grid[cfg, j, i] + off_x * dp, oy_grid[cfg, j, i] + off_y * dp

    cA, xA, yA = _seeds(idx_A, 0.667, 0.333)
    cB, xB, yB = _seeds(idx_B, 0.333, 0.667)
    cand_cfg = torch.cat([cA, cB])
    cand_x0  = torch.cat([xA, xB])
    cand_y0  = torch.cat([yA, yB])

    xi = cand_x0.clone()
    yi = cand_y0.clone()
    sub_sx  = sx_t[cand_cfg]
    sub_sy  = sy_t[cand_cfg]
    sub_re2 = _re2_point(torch.pow(10.0, log_m_t[cand_cfg]), ctx)

    # 占位（C：早终止时记录最后一次 kap/g1/g2）
    kap_t = None
    g1_t = None
    g2_t = None
    ax_t = None
    ay_t = None

    # ── 3. Newton 迭代（A：双线性插值；C：早终止）─────────────────────
    for _it in range(max_iter):
        # A：缓存插值（一次取 5 个场）
        ax_fix, ay_fix, kap_fix, g1_fix, g2_fix = _bilinear_5(cache, xi, yi)

        # 子晕（解析叠加）
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

        ax_t = ax_fix + ax_c
        ay_t = ay_fix + ay_c
        kap_t = kap_fix
        g1_t  = g1_fix + g1_c
        g2_t  = g2_fix + g2_c

        pxx = kap_t + g1_t
        pyy = kap_t - g1_t
        pxy = g2_t
        ff = xs_src - xi + ax_t
        gg = ys_src - yi + ay_t
        mm = (1.0 - pxx) * (1.0 - pyy) - pxy * pxy
        dxn = ((1.0 - pyy) * ff + pxy * gg) / mm
        dyn = ((1.0 - pxx) * gg + pxy * ff) / mm
        xi = xi + dxn
        yi = yi + dyn

        # C：所有候选都收敛即跳出
        max_step = torch.max(torch.maximum(dxn.abs(), dyn.abs())).item()
        if max_step < newton_tol:
            break

    # ── 4. 解析 Newton 精化（位置 + mag）
    # bilinear Newton 会收敛到 bilinear-α 的不动点，与真不动点略有偏差
    # 在临界曲线附近这个偏差经 mag = 1/((1-κ)²-γ²) 被显著放大
    # 解决：1 步解析 Newton 精化位置 + 1 次最终 kap/g1/g2 评估用于 mag
    from rhongomyniad.image_finder import sum_lensmodel as _sum_lm

    # 4a. 一步解析 Newton（位置精化）
    ax_fix_a, ay_fix_a, kap_fix_a, g1_fix_a, g2_fix_a, _, _ = _sum_lm(
        ctx, cache["fixed_lenses"], xi, yi, need_kg=True, need_phi=False)
    dx_a = xi.unsqueeze(1) - sub_sx
    dy_a = yi.unsqueeze(1) - sub_sy
    r2_a = dx_a * dx_a + dy_a * dy_a
    r2s_a = torch.clamp(r2_a, min=sc2)
    rr_a = sub_re2 / r2s_a
    ax_c_a = (rr_a * dx_a).sum(dim=1)
    ay_c_a = (rr_a * dy_a).sum(dim=1)
    inv_r4_a = 1.0 / (r2s_a * r2s_a)
    g1_c_a = (sub_re2 * (dy_a * dy_a - dx_a * dx_a) * inv_r4_a).sum(dim=1)
    g2_c_a = (sub_re2 * (-2.0 * dx_a * dy_a) * inv_r4_a).sum(dim=1)
    ax_t = ax_fix_a + ax_c_a
    ay_t = ay_fix_a + ay_c_a
    pxx = kap_fix_a + g1_fix_a + g1_c_a
    pyy = kap_fix_a - (g1_fix_a + g1_c_a)
    pxy = g2_fix_a + g2_c_a
    ff = xs_src - xi + ax_t
    gg = ys_src - yi + ay_t
    mm = (1.0 - pxx) * (1.0 - pyy) - pxy * pxy
    xi = xi + ((1.0 - pyy) * ff + pxy * gg) / mm
    yi = yi + ((1.0 - pxx) * gg + pxy * ff) / mm

    # 4b. 在精化后的位置最终评估 kap/g1/g2 用于 mag
    _, _, kap_fix_f, g1_fix_f, g2_fix_f, _, _ = _sum_lm(
        ctx, cache["fixed_lenses"], xi, yi, need_kg=True, need_phi=False)
    dx_f = xi.unsqueeze(1) - sub_sx
    dy_f = yi.unsqueeze(1) - sub_sy
    r2_f = dx_f * dx_f + dy_f * dy_f
    r2s_f = torch.clamp(r2_f, min=sc2)
    inv_r4_f = 1.0 / (r2s_f * r2s_f)
    g1_c_f = (sub_re2 * (dy_f * dy_f - dx_f * dx_f) * inv_r4_f).sum(dim=1)
    g2_c_f = (sub_re2 * (-2.0 * dx_f * dy_f) * inv_r4_f).sum(dim=1)
    kap_t = kap_fix_f
    g1_t = g1_fix_f + g1_c_f
    g2_t = g2_fix_f + g2_c_f
    muinv = (1.0 - kap_t) ** 2 - (g1_t * g1_t + g2_t * g2_t)
    mag = 1.0 / (muinv + K.DEF_IMAG_CEIL)

    # 收敛过滤：候选偏离种子超过 √2 × dp 的视为牛顿跑飞
    dist2 = (xi - cand_x0) ** 2 + (yi - cand_y0) ** 2
    keep = dist2 <= (2.0 * dp * dp)

    # ── 5. CPU 端去重 + 分组 ──────────────────────────────────────────
    cfg_cpu = cand_cfg.cpu().numpy()
    xi_cpu = xi.detach().cpu().numpy()
    yi_cpu = yi.detach().cpu().numpy()
    mag_cpu = mag.detach().cpu().numpy()
    keep_cpu = keep.cpu().numpy()

    out = [[] for _ in range(B)]
    for i in range(len(cfg_cpu)):
        if not keep_cpu[i]:
            continue
        c = int(cfg_cpu[i])
        x, y, m = float(xi_cpu[i]), float(yi_cpu[i]), float(mag_cpu[i])
        dup = False
        for xj, yj, mj, _ in out[c]:
            if ((x - xj) ** 2 + (y - yj) ** 2) / max(abs(m * mj), 1e-300) \
                    <= 10.0 * K.DEF_MAX_POI_TOL ** 2:
                dup = True
                break
        if not dup:
            out[c].append((x, y, m, 0.0))   # B：td 占位 0

    # 5 像 → 剔除中央低放大率像（与 v1 相同）
    result = []
    for imgs in out:
        if len(imgs) == 5:
            central = min(range(5), key=lambda k: abs(imgs[k][2]))
            imgs = [im for k, im in enumerate(imgs) if k != central]
        result.append(imgs)
    return result


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                v2 向量化 log_prob（pointmass）                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

class VectorisedLogProbPM_v2:
    def __init__(self, n_subhalos, active_subhalos, src_x, src_y,
                 cache, device, dtype, prior_radius, log_m_min, log_m_max,
                 newton_iters=V2_MAX_NEWTON_ITER, newton_tol=V2_NEWTON_TOL):
        self.n = n_subhalos
        self.active = list(active_subhalos)
        self.src_x = float(src_x)
        self.src_y = float(src_y)
        self.cache = cache
        self.device = device
        self.dtype = dtype
        self.prior_radius = float(prior_radius)
        self.log_m_min = float(log_m_min)
        self.log_m_max = float(log_m_max)
        self.newton_iters = int(newton_iters)
        self.newton_tol = float(newton_tol)
        self.x_ctr = np.array([OBS_POS[i - 1, 0] for i in self.active])
        self.y_ctr = np.array([OBS_POS[i - 1, 1] for i in self.active])

    def _loss_one(self, imgs):
        if len(imgs) != 4:
            return None
        pred_pos = np.array([[im[0], im[1]] for im in imgs])
        pred_pos[:, 0] += CENTER_OFFSET_X
        pred_pos[:, 1] += CENTER_OFFSET_Y
        pred_mag = np.array([im[2] for im in imgs])
        dists = cdist(OBS_POS, pred_pos)
        row_ind, col_ind = linear_sum_assignment(dists)
        order = col_ind[np.argsort(row_ind)]
        pp = pred_pos[order]
        pm = pred_mag[order]
        delta_mas = np.sqrt(np.sum(((pp - OBS_POS) * 1000) ** 2, axis=1))
        Y = 0.0
        for i in range(4):
            chi2_pos = (delta_mas[i] / OBS_POS_SIGMA[i]) ** 2
            chi2_mag = ((pm[i] - OBS_MAG[i]) / OBS_MAG_ERR[i]) ** 2
            P = 0.0 if delta_mas[i] <= OBS_POS_SIGMA[i] else LOSS_PENALTY * delta_mas[i]
            Y += LOSS_COEF_A * chi2_pos + LOSS_COEF_B * chi2_mag + P
        return Y

    def __call__(self, theta):
        if theta.ndim == 1:
            theta = theta[None, :]
        B = theta.shape[0]

        # 先验（向量化，CPU 上做）
        sx = theta[:, 0::3]
        sy = theta[:, 1::3]
        sm = theta[:, 2::3]
        x_ok = np.all(np.abs(sx - self.x_ctr[None, :]) <= self.prior_radius, axis=1)
        y_ok = np.all(np.abs(sy - self.y_ctr[None, :]) <= self.prior_radius, axis=1)
        m_ok = np.all((sm >= self.log_m_min) & (sm <= self.log_m_max), axis=1)
        valid = x_ok & y_ok & m_ok

        log_p = np.full(B, -np.inf, dtype=np.float64)
        if not np.any(valid):
            return log_p

        sx_v = sx[valid]
        sy_v = sy[valid]
        sm_v = sm[valid]
        sx_t = torch.tensor(sx_v, device=self.device, dtype=self.dtype)
        sy_t = torch.tensor(sy_v, device=self.device, dtype=self.dtype)
        sm_t = torch.tensor(sm_v, device=self.device, dtype=self.dtype)

        all_imgs = batched_point_solve_v2(
            sx_t, sy_t, sm_t,
            self.src_x, self.src_y, self.cache,
            max_iter=self.newton_iters, newton_tol=self.newton_tol,
        )

        valid_idx = np.nonzero(valid)[0]
        for k, c in enumerate(valid_idx):
            loss = self._loss_one(all_imgs[k])
            if loss is None or loss >= 1e10:
                log_p[c] = -np.inf
            else:
                log_p[c] = -0.5 * loss
        return log_p


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                              main                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="MCMC GPU v2（缓存双线性 + 跳 phi + 早终止 + fp32）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("folder", help="包含 *_best_params.txt 的结果文件夹")
    parser.add_argument("--nwalkers", type=int, default=V2_DEFAULT_NWALKERS,
                        help=f"walker 数量 [默认: {V2_DEFAULT_NWALKERS}]")
    parser.add_argument("--nsteps", type=int, default=DEFAULT_MCMC_NSTEPS,
                        help=f"采样步数 [默认: {DEFAULT_MCMC_NSTEPS}]")
    parser.add_argument("--burnin", type=int, default=DEFAULT_MCMC_BURNIN,
                        help=f"burn-in [默认: {DEFAULT_MCMC_BURNIN}]")
    parser.add_argument("--thin", type=int, default=DEFAULT_MCMC_THIN,
                        help=f"thin [默认: {DEFAULT_MCMC_THIN}]")
    parser.add_argument("--perturbation", type=float, default=DEFAULT_MCMC_PERTURBATION,
                        help="初始扰动幅度")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=DEFAULT_MCMC_PROGRESS)
    parser.add_argument("--baseline_dir", type=str, default="",
                        help="含 bestfit.dat 的目录（留空则自动搜索）")
    parser.add_argument("--device", type=str, default="auto",
                        choices=("auto", "cuda", "cpu"))
    parser.add_argument("--dtype", type=str, default=V2_DEFAULT_DTYPE,
                        choices=("float64", "float32"),
                        help=f"张量精度 [默认: {V2_DEFAULT_DTYPE}]")
    parser.add_argument("--newton_tol", type=float, default=V2_NEWTON_TOL,
                        help="Newton 早终止阈值")
    parser.add_argument("--newton_iters", type=int, default=V2_MAX_NEWTON_ITER,
                        help="Newton 最大迭代次数")
    args = parser.parse_args()

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    rh.set_device(device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"[错误] 文件夹不存在: {folder}")
        sys.exit(1)

    print("=" * 70)
    print("mcmc_gpu_v_2.py — A(双线性) + B(无 phi) + C(早终止) + D(fp32/64w)")
    print("=" * 70)
    print(f"  结果文件夹: {folder}")
    print(f"  设备: {device}   精度: {dtype}")
    print(f"  Newton 迭代上限: {args.newton_iters}, tol: {args.newton_tol:g}")

    # 解析 best_params
    cands = glob.glob(os.path.join(folder, "*_best_params.txt"))
    if not cands:
        print(f"[错误] 未找到 *_best_params.txt: {folder}")
        sys.exit(1)
    param_file = sorted(cands)[0]
    print(f"  参数文件: {os.path.basename(param_file)}")
    prefix = os.path.basename(param_file).replace("_best_params.txt", "")

    parsed = parse_best_params(param_file)
    model_type = parsed["model_type"]
    active_subhalos = parsed["active_subhalos"]
    subhalos = parsed["subhalos"]
    print(f"  模型类型: {model_type}")
    print(f"  active_subhalos: {active_subhalos}")
    print(f"  子晕数量: {len(subhalos)}")
    if parsed["chi2_best"]:
        print(f"  DE chi2: {parsed['chi2_best']:.2f}")

    if model_type != "pointmass":
        raise NotImplementedError(
            f"v2 当前仅支持 pointmass 模型，收到: {model_type}"
        )

    # 基础透镜
    lens_params = DEFAULT_LENS_PARAMS
    main_lens_key = DEFAULT_MAIN_LENS_KEY
    src_x, src_y = SOURCE_X, SOURCE_Y

    if args.baseline_dir:
        d_search = [args.baseline_dir]
    else:
        d = find_bestfit_dir(folder)
        d_search = [d] if d else []

    for d in d_search:
        lp, sx, sy, mlk = load_baseline_lens_params(d)
        if lp is not None:
            lens_params = lp
            main_lens_key = mlk
            src_x, src_y = sx, sy
            print(f"  基础透镜: 从 {d} 加载（主透镜: {mlk}）")
            break
    else:
        print(f"  基础透镜: 内置默认（SIE）")

    # 初始参数
    p0 = build_initial_params(model_type, subhalos)
    param_names, corner_labels = build_param_names(model_type, active_subhalos)
    ndim = len(p0)
    print(f"  参数维度: {ndim}")

    # 依赖
    try:
        import emcee
        import corner as _corner  # noqa: F401
        from tqdm import tqdm
    except ImportError as e:
        print(f"[错误] 缺少依赖: {e}\n  请运行: pip install emcee corner tqdm")
        sys.exit(1)

    # 缓存
    print(f"\n构建固定透镜场缓存（仅 Sersic + {main_lens_key}）...")
    cache = build_fixed_cache(lens_params, main_lens_key, src_x, src_y,
                              device, dtype)

    # log_prob
    log_prob = VectorisedLogProbPM_v2(
        n_subhalos=len(active_subhalos),
        active_subhalos=active_subhalos,
        src_x=src_x, src_y=src_y, cache=cache,
        device=device, dtype=dtype,
        prior_radius=MCMC_SEARCH_RADIUS,
        log_m_min=MCMC_PM_LOG_M_MIN,
        log_m_max=MCMC_PM_LOG_M_MAX,
        newton_iters=int(args.newton_iters),
        newton_tol=float(args.newton_tol),
    )

    print(f"\n  先验: 位置半径 ±{MCMC_SEARCH_RADIUS * 1000:.0f} mas")
    print(f"        logM ∈ [{MCMC_PM_LOG_M_MIN}, {MCMC_PM_LOG_M_MAX}] dex")

    # 验证
    lp_test = log_prob(p0[None, :])[0]
    if not np.isfinite(lp_test):
        print(f"[警告] DE 最优解处 log_prob = {lp_test}，初始点可能不可行")
    else:
        print(f"  初始 log_prob = {lp_test:.4f}（chi2 ≈ {-2 * lp_test:.2f}）")

    nwalkers = max(args.nwalkers, 2 * ndim + 2)
    if nwalkers != args.nwalkers:
        print(f"  [调整] nwalkers → {nwalkers}（≥ 2·ndim）")

    rng = np.random.default_rng()
    initial = np.array([
        p0 + rng.normal(0, args.perturbation * (np.abs(p0) + 1e-8), ndim)
        for _ in range(nwalkers)
    ])

    print(f"\n{'=' * 70}")
    print(f"开始向量化 MCMC（nsteps={args.nsteps}, burnin={args.burnin}）")
    print(f"  Walkers: {nwalkers}（每步 GPU 一次性处理整组）")
    print(f"{'=' * 70}")

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, vectorize=True)

    t0 = time.perf_counter()
    if args.progress:
        for _ in tqdm(sampler.sample(initial, iterations=args.nsteps),
                      total=args.nsteps, desc="MCMC采样"):
            pass
    else:
        sampler.run_mcmc(initial, args.nsteps, progress=False)
    elapsed = time.perf_counter() - t0
    print(f"\n  采样耗时: {elapsed:.1f} s "
          f"({elapsed / args.nsteps * 1000:.1f} ms/step)")

    samples = sampler.get_chain(discard=args.burnin, thin=args.thin, flat=True)
    chain = sampler.get_chain()
    print(f"  总样本: {nwalkers * args.nsteps}, 有效样本: {len(samples)}")

    # 输出
    chain_file = os.path.join(folder, f"{prefix}_mcmc_chain.dat")
    np.savetxt(chain_file, samples, header=" ".join(param_names))
    print(f"\n  ✓ MCMC链: {os.path.basename(chain_file)}")

    import matplotlib
    matplotlib.use("Agg")
    print("\n生成图表...")
    plot_corner(samples, corner_labels, p0,
                os.path.join(folder, f"{prefix}_mcmc_corner.png"))
    plot_trace(chain, corner_labels, args.burnin,
               os.path.join(folder, f"{prefix}_mcmc_trace.png"))
    plot_mass_1d(model_type, active_subhalos, samples, subhalos,
                 os.path.join(folder, f"{prefix}_mcmc_mass_1d.png"))

    posterior_file = os.path.join(folder, f"{prefix}_mcmc_posterior.txt")
    save_posterior(samples, param_names, model_type, active_subhalos,
                   subhalos, args, nwalkers, posterior_file)

    print(f"\n{'=' * 70}")
    print(f"完成！输出文件均在: {folder}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
