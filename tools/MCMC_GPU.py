#!/usr/bin/env python3
"""
MCMC_GPU.py
===========
GPU + 向量化版本的 MCMC 后验采样工具，对应 mcmc_from_result.py 的方案 C。

技术要点
--------
1. **固定透镜场缓存**：Sersic + 主透镜（SIE 等）的偏角场在网格上一次计算
   （Rhongomyniad / PyTorch），后续每步 MCMC 只在缓存基础上叠加子晕扰动。
   完全绕开 glafic 的 Hankel-Romberg 数值积分。
2. **向量化 log_prob**：emcee `vectorize=True` 把整组 walker 一次性传入；
   每步只调用一次 GPU kernel，子晕被批维并行处理。
3. **零多进程**：单进程 + GPU batch 替代 multiprocessing.Pool，
   避免 GPU 资源争用，也彻底绕开 fork 状态污染问题。

支持的子晕模型
--------------
- pointmass（已批量化）
- nfw / king / p_jaffe：暂不支持批量。需要为每个模型补一个
  `_batched_*_fields` 才能并行求解。运行时会抛 NotImplementedError。

用法
----
    python tools/MCMC_GPU.py <result_folder>
    python tools/MCMC_GPU.py <result_folder> --nsteps 5000 --nwalkers 64
    python tools/MCMC_GPU.py <result_folder> --device cuda

输出文件（与 mcmc_from_result.py 保持一致）
--------
    <prefix>_mcmc_chain.dat
    <prefix>_mcmc_corner.png
    <prefix>_mcmc_trace.png
    <prefix>_mcmc_mass_1d.png
    <prefix>_mcmc_posterior.txt
"""

from __future__ import annotations

import sys
import os
import time
import math
import argparse
import glob
import numpy as np
from pathlib import Path
from datetime import datetime

# ── 路径配置 ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
GLADE_ROOT = _HERE.parent
sys.path.insert(0, str(GLADE_ROOT))
sys.path.insert(0, str(_HERE))

_RH_DIR = GLADE_ROOT / "Rhongomyniad"
if not _RH_DIR.is_dir():
    raise FileNotFoundError(
        f"找不到 Rhongomyniad 目录：{_RH_DIR}\n"
        f"请确认 Rhongomyniad 与 tools 目录平级（位于 glade/）"
    )
sys.path.insert(0, str(_RH_DIR))

from runtime_env import setup_runtime_env  # noqa: E402
setup_runtime_env(str(GLADE_ROOT.resolve()))

import torch  # noqa: E402
import rhongomyniad as rh  # noqa: E402
from rhongomyniad import constants as K  # noqa: E402
from rhongomyniad.lens_models import LensContext  # noqa: E402
from rhongomyniad.image_finder import sum_lensmodel  # noqa: E402
from rhongomyniad.cosmology import Cosmology  # noqa: E402

from scipy.spatial.distance import cdist  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

# ── 复用 CPU 版本的参数解析 / 绘图 / 后验保存 ───────────────────────────────
# 注意：mcmc_from_result 在导入时会加载 glafic。glafic.so 本身可正常加载，
# 仅 point_solve 慢，因此这里复用没问题。
from mcmc_from_result import (  # noqa: E402
    OBS_POS, OBS_MAG, OBS_MAG_ERR, OBS_POS_SIGMA,
    CENTER_OFFSET_X, CENTER_OFFSET_Y,
    OMEGA, LAMBDA_COSMO, WEOS, HUBBLE,
    SOURCE_Z, SOURCE_X, SOURCE_Y, LENS_Z,
    XMIN, YMIN, XMAX, YMAX, PIX_EXT, PIX_POI, MAXLEV,
    LOSS_COEF_A, LOSS_COEF_B, LOSS_PENALTY,
    DEFAULT_LENS_PARAMS, DEFAULT_MAIN_LENS_KEY,
    DEFAULT_MCMC_NWALKERS, DEFAULT_MCMC_NSTEPS, DEFAULT_MCMC_BURNIN,
    DEFAULT_MCMC_THIN, DEFAULT_MCMC_PERTURBATION, DEFAULT_MCMC_PROGRESS,
    MCMC_SEARCH_RADIUS, MCMC_PM_LOG_M_MIN, MCMC_PM_LOG_M_MAX,
    parse_best_params, load_baseline_lens_params, find_bestfit_dir,
    build_initial_params, build_param_names, params_per_subhalo,
    plot_corner, plot_trace, plot_mass_1d, save_posterior,
)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  固定透镜场缓存（Sersic + 主透镜）                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

def build_fixed_cache(lens_params, main_lens_key, src_x, src_y, device, dtype):
    """构造 Sersic + 主透镜的偏角场缓存（在网格上只算一次）。"""
    rh.set_dtype(dtype)
    rh.init(OMEGA, LAMBDA_COSMO, WEOS, HUBBLE, "_mcmc_gpu",
            XMIN, YMIN, XMAX, YMAX, PIX_EXT, PIX_POI, MAXLEV, verb=0)
    # 在 rh 里登记一遍（某些后端逻辑依赖 set_lens 状态）
    rh.startup_setnum(len(lens_params), 0, 1)
    for _, pv in lens_params.items():
        rh.set_lens(*pv)
    rh.set_point(1, SOURCE_Z, src_x, src_y)
    rh.model_init(verb=0)

    cosmo = Cosmology(omega=OMEGA, lam=LAMBDA_COSMO, weos=WEOS, hubble=HUBBLE)
    ctx = LensContext.build(cosmo, zl=LENS_Z, zs=SOURCE_Z)

    fixed_lenses = []
    for key, pv in lens_params.items():
        if key.startswith("sers") or key == main_lens_key:
            _, model, z, *p7 = pv
            fixed_lenses.append((model, (z, *p7)))

    dp = PIX_POI / (2 ** (MAXLEV - 1))
    nx = int(math.ceil((XMAX - XMIN) / dp)) + 1
    ny = int(math.ceil((YMAX - YMIN) / dp)) + 1
    xs_ax = torch.linspace(XMIN, XMIN + (nx - 1) * dp, nx,
                           device=device, dtype=dtype)
    ys_ax = torch.linspace(YMIN, YMIN + (ny - 1) * dp, ny,
                           device=device, dtype=dtype)
    gx, gy = torch.meshgrid(xs_ax, ys_ax, indexing="xy")

    t0 = time.perf_counter()
    ax_f, ay_f, kap_f, g1_f, g2_f, phi_f, _ = sum_lensmodel(
        ctx, fixed_lenses, gx, gy, need_kg=True, need_phi=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  ✓ 固定透镜场缓存 {nx}×{ny} 完成 "
          f"({(time.perf_counter() - t0) * 1000:.1f} ms)")

    return dict(
        ctx=ctx, gx=gx, gy=gy, dp=dp, nx=nx, ny=ny,
        ax=ax_f.contiguous(), ay=ay_f.contiguous(),
        kap=kap_f.contiguous(), g1=g1_f.contiguous(),
        g2=g2_f.contiguous(), phi=phi_f.contiguous(),
        fixed_lenses=fixed_lenses,
    )


# ╔══════════════════════════════════════════════════════════════════════╗
# ║              批量 pointmass 偏角场 + 图像求解（GPU）                  ║
# ║            从 legacy/v_pointmass_gpu/version_pointmass_gpu.py 移植    ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _re2_point(mass, ctx):
    d = ctx.dis_ls / (K.COVERH_MPCH * ctx.dis_ol * ctx.dis_os)
    return (2.0 * (K.R_SCHWARZ * mass / K.MPC2METER) * d) / (K.ARCSEC2RADIAN ** 2)


def _batched_pointmass_fields(sx, sy, log_m, ctx, gx, gy,
                              smallcore=K.DEF_SMALLCORE):
    """sx, sy, log_m: (B, K)。返回 (ax, ay, g1, g2, phi)，形状 (B, ny, nx)。"""
    B, Kn = sx.shape
    ny, nx = gx.shape
    mass = torch.pow(10.0, log_m)
    re2 = _re2_point(mass, ctx)
    sx_b = sx.view(B, Kn, 1, 1)
    sy_b = sy.view(B, Kn, 1, 1)
    re2_b = re2.view(B, Kn, 1, 1)
    dx = gx.view(1, 1, ny, nx) - sx_b
    dy = gy.view(1, 1, ny, nx) - sy_b
    r2 = dx * dx + dy * dy
    sc2 = smallcore * smallcore
    rr = re2_b / (r2 + sc2)
    ax = (rr * dx).sum(dim=1)
    ay = (rr * dy).sum(dim=1)
    near_center = r2 < sc2
    inv_r4 = 1.0 / torch.where(near_center,
                                torch.full_like(r2, sc2 * sc2),
                                r2 * r2)
    g1_k = re2_b * (dy * dy - dx * dx) * inv_r4
    g2_k = re2_b * (-2.0 * dx * dy) * inv_r4
    g1_k = torch.where(near_center, torch.zeros_like(g1_k), g1_k)
    g2_k = torch.where(near_center, torch.zeros_like(g2_k), g2_k)
    g1 = g1_k.sum(dim=1)
    g2 = g2_k.sum(dim=1)
    phi = (0.5 * re2_b * torch.log(torch.clamp(r2, min=sc2))).sum(dim=1)
    return ax, ay, g1, g2, phi


def _tri_contains(xs, ys, ax, ay, bx, by, cx, cy):
    d1x = xs - ax; d1y = ys - ay
    d2x = xs - bx; d2y = ys - by
    d3x = xs - cx; d3y = ys - cy
    d12 = d1x * d2y - d1y * d2x
    d23 = d2x * d3y - d2y * d3x
    d31 = d3x * d1y - d3y * d1x
    return (((d12 >= 0) & (d23 >= 0) & (d31 >= 0))
            | ((d12 <= 0) & (d23 <= 0) & (d31 <= 0)))


def batched_point_solve(sx_t, sy_t, log_m_t, xs_src, ys_src, cache, max_iter=8):
    """对 B 个 walker 配置一次性求解透镜方程。

    返回长度 B 的 list，每个元素是该 walker 的图像列表 [(x, y, mag, td), ...]。
    """
    B, Kn = sx_t.shape
    gx, gy = cache["gx"], cache["gy"]
    ny, nx = gx.shape
    dp = cache["dp"]
    ctx = cache["ctx"]

    ax_p, ay_p, _, _, _ = _batched_pointmass_fields(
        sx_t, sy_t, log_m_t, ctx, gx, gy)
    ax = cache["ax"].unsqueeze(0) + ax_p
    ay = cache["ay"].unsqueeze(0) + ay_p

    # 在源平面寻找包含源位置的三角形（每个网格元被切成两个三角形）
    sx_grid = gx.unsqueeze(0) - ax
    sy_grid = gy.unsqueeze(0) - ay
    bl_x = sx_grid[:, :-1, :-1]; bl_y = sy_grid[:, :-1, :-1]
    br_x = sx_grid[:, :-1,  1:]; br_y = sy_grid[:, :-1,  1:]
    tl_x = sx_grid[:,  1:, :-1]; tl_y = sy_grid[:,  1:, :-1]
    tr_x = sx_grid[:,  1:,  1:]; tr_y = sy_grid[:,  1:,  1:]
    in_A = _tri_contains(xs_src, ys_src, bl_x, bl_y, tr_x, tr_y, br_x, br_y)
    in_B = _tri_contains(xs_src, ys_src, bl_x, bl_y, tr_x, tr_y, tl_x, tl_y)
    ox = gx[:-1, :-1].unsqueeze(0).expand_as(in_A)
    oy = gy[:-1, :-1].unsqueeze(0).expand_as(in_A)
    idx_A = torch.nonzero(in_A, as_tuple=False)
    idx_B = torch.nonzero(in_B, as_tuple=False)
    if idx_A.numel() + idx_B.numel() == 0:
        return [[] for _ in range(B)]

    def _seeds(idx, off_x, off_y):
        cfg, j, i = idx[:, 0], idx[:, 1], idx[:, 2]
        return cfg, ox[cfg, j, i] + off_x * dp, oy[cfg, j, i] + off_y * dp

    cA, xA, yA = _seeds(idx_A, 0.667, 0.333)
    cB, xB, yB = _seeds(idx_B, 0.333, 0.667)
    cand_cfg = torch.cat([cA, cB])
    cand_x0 = torch.cat([xA, xB])
    cand_y0 = torch.cat([yA, yB])

    xi = cand_x0.clone()
    yi = cand_y0.clone()
    sub_sx = sx_t[cand_cfg]
    sub_sy = sy_t[cand_cfg]
    sub_re2 = _re2_point(torch.pow(10.0, log_m_t[cand_cfg]), ctx)
    fixed_lenses = cache["fixed_lenses"]

    # 牛顿迭代（与 glafic 的 calcimage_i 一致）
    for _ in range(max_iter):
        ax_fix, ay_fix, kap_fix, g1_fix, g2_fix, _, _ = sum_lensmodel(
            ctx, fixed_lenses, xi, yi, need_kg=True, need_phi=False)
        dx = xi.unsqueeze(1) - sub_sx
        dy = yi.unsqueeze(1) - sub_sy
        r2 = dx * dx + dy * dy
        r2s = torch.clamp(r2, min=K.DEF_SMALLCORE ** 2)
        rr = sub_re2 / r2s
        ax_c = (rr * dx).sum(dim=1)
        ay_c = (rr * dy).sum(dim=1)
        inv_r4 = 1.0 / (r2s * r2s)
        g1_c = (sub_re2 * (dy * dy - dx * dx) * inv_r4).sum(dim=1)
        g2_c = (sub_re2 * (-2.0 * dx * dy) * inv_r4).sum(dim=1)
        ax_t = ax_fix + ax_c
        ay_t = ay_fix + ay_c
        kap_t = kap_fix
        g1_t = g1_fix + g1_c
        g2_t = g2_fix + g2_c
        pxx = kap_t + g1_t
        pyy = kap_t - g1_t
        pxy = g2_t
        ff = xs_src - xi + ax_t
        gg = ys_src - yi + ay_t
        mm = (1.0 - pxx) * (1.0 - pyy) - pxy * pxy
        xi = xi + ((1.0 - pyy) * ff + pxy * gg) / mm
        yi = yi + ((1.0 - pxx) * gg + pxy * ff) / mm

    muinv = (1.0 - kap_t) ** 2 - (g1_t * g1_t + g2_t * g2_t)
    mag = 1.0 / (muinv + K.DEF_IMAG_CEIL)

    # 时延（虽然不参与 loss，但保留以便复用 legacy 逻辑）
    _, _, _, _, _, phi_fix, _ = sum_lensmodel(
        ctx, fixed_lenses, xi, yi, need_kg=True, need_phi=True)
    phi_p_c = (0.5 * sub_re2 *
               torch.log(torch.clamp(r2, min=K.DEF_SMALLCORE ** 2))).sum(dim=1)
    phi_total = phi_fix + phi_p_c
    td = ctx.tdelay_fac * (0.5 * (ax_t * ax_t + ay_t * ay_t) - phi_total)

    # 仅保留收敛到本格附近的候选（避免牛顿跑飞）
    dist2 = (xi - cand_x0) ** 2 + (yi - cand_y0) ** 2
    keep = dist2 <= (2.0 * dp * dp)

    cfg_cpu = cand_cfg.cpu().numpy()
    xi_cpu = xi.detach().cpu().numpy()
    yi_cpu = yi.detach().cpu().numpy()
    mag_cpu = mag.detach().cpu().numpy()
    td_cpu = td.detach().cpu().numpy()
    keep_cpu = keep.cpu().numpy()

    out = [[] for _ in range(B)]
    for i in range(len(cfg_cpu)):
        if not keep_cpu[i]:
            continue
        c = int(cfg_cpu[i])
        x, y, m = float(xi_cpu[i]), float(yi_cpu[i]), float(mag_cpu[i])
        # 去重（牛顿可从相邻种子收敛到同一图像）
        dup = False
        for xj, yj, mj, _ in out[c]:
            if ((x - xj) ** 2 + (y - yj) ** 2) / max(abs(m * mj), 1e-300) \
                    <= 10.0 * K.DEF_MAX_POI_TOL ** 2:
                dup = True
                break
        if not dup:
            out[c].append((x, y, m, float(td_cpu[i])))

    # anfw 等会多产生一个低放大率中心像 → 与 CPU 版一致地剔除
    result = []
    for imgs in out:
        if len(imgs) == 5:
            central = min(range(5), key=lambda k: abs(imgs[k][2]))
            imgs = [im for k, im in enumerate(imgs) if k != central]
        result.append(imgs)
    return result


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                       向量化 log_prob                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

class VectorisedLogProbPM:
    """Pointmass 模型的向量化 log_prob。

    输入 theta 形状 (B, ndim)，返回 (B,) np.ndarray。
    """

    def __init__(self, n_subhalos, active_subhalos, src_x, src_y,
                 cache, device, dtype, prior_radius, log_m_min, log_m_max):
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

        # 各 subhalo 的中心位置（先验中心）
        self.x_ctr = np.array([OBS_POS[i - 1, 0] for i in self.active])  # (n,)
        self.y_ctr = np.array([OBS_POS[i - 1, 1] for i in self.active])

    # 单 walker 损失（CPU 端的廉价后处理）
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
        n = self.n

        # ── 先验检查（向量化，CPU）───────────────────────────────────
        sx = theta[:, 0::3]   # (B, n)
        sy = theta[:, 1::3]
        sm = theta[:, 2::3]
        x_ok = np.all(np.abs(sx - self.x_ctr[None, :]) <= self.prior_radius, axis=1)
        y_ok = np.all(np.abs(sy - self.y_ctr[None, :]) <= self.prior_radius, axis=1)
        m_ok = np.all((sm >= self.log_m_min) & (sm <= self.log_m_max), axis=1)
        valid = x_ok & y_ok & m_ok

        log_p = np.full(B, -np.inf, dtype=np.float64)
        if not np.any(valid):
            return log_p

        # ── 对有效 walker 做 GPU 批量求解 ────────────────────────────
        sx_v = sx[valid]
        sy_v = sy[valid]
        sm_v = sm[valid]
        sx_t = torch.tensor(sx_v, device=self.device, dtype=self.dtype)
        sy_t = torch.tensor(sy_v, device=self.device, dtype=self.dtype)
        sm_t = torch.tensor(sm_v, device=self.device, dtype=self.dtype)

        all_imgs = batched_point_solve(
            sx_t, sy_t, sm_t,
            self.src_x, self.src_y, self.cache,
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
        description="GPU + 向量化 MCMC（Rhongomyniad，emcee vectorize=True）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("folder", help="包含 *_best_params.txt 的结果文件夹")
    parser.add_argument("--nwalkers", type=int, default=DEFAULT_MCMC_NWALKERS,
                        help=f"walker 数量 [默认: {DEFAULT_MCMC_NWALKERS}]")
    parser.add_argument("--nsteps", type=int, default=DEFAULT_MCMC_NSTEPS,
                        help=f"采样步数 [默认: {DEFAULT_MCMC_NSTEPS}]")
    parser.add_argument("--burnin", type=int, default=DEFAULT_MCMC_BURNIN,
                        help=f"burn-in [默认: {DEFAULT_MCMC_BURNIN}]")
    parser.add_argument("--thin", type=int, default=DEFAULT_MCMC_THIN,
                        help=f"thin [默认: {DEFAULT_MCMC_THIN}]")
    parser.add_argument("--perturbation", type=float, default=DEFAULT_MCMC_PERTURBATION,
                        help=f"初始 walker 扰动幅度 [默认: {DEFAULT_MCMC_PERTURBATION}]")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="关闭 tqdm 进度条")
    parser.set_defaults(progress=DEFAULT_MCMC_PROGRESS)
    parser.add_argument("--baseline_dir", type=str, default="",
                        help="含 bestfit.dat 的目录（留空则自动搜索）")
    parser.add_argument("--device", type=str, default="auto",
                        choices=("auto", "cuda", "cpu"),
                        help="计算设备 [默认: auto]")
    parser.add_argument("--dtype", type=str, default="float64",
                        choices=("float64", "float32"),
                        help="张量精度 [默认: float64]")
    args = parser.parse_args()

    # ── device ───────────────────────────────────────────────────────
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
    print("MCMC_GPU.py — Rhongomyniad GPU + emcee 向量化 MCMC")
    print("=" * 70)
    print(f"  结果文件夹: {folder}")
    print(f"  设备: {device}   精度: {dtype}")

    # ── 1. 解析 best_params.txt ──────────────────────────────────────
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
            f"MCMC_GPU.py 当前只支持 pointmass 模型，收到: {model_type}\n"
            f"  → nfw / king / p_jaffe 需要为各自模型实现 _batched_*_fields。\n"
            f"  → 临时方案：仍使用 mcmc_from_result.py（CPU），\n"
            f"    或先把 glafic.h 的 TOL_ROMBERG_JHK 恢复到 5.0e-4 并重编译。"
        )

    # ── 2. 加载基础透镜 ──────────────────────────────────────────────
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

    # ── 3. 构造初始参数 ──────────────────────────────────────────────
    p0 = build_initial_params(model_type, subhalos)
    param_names, corner_labels = build_param_names(model_type, active_subhalos)
    ndim = len(p0)
    print(f"  参数维度: {ndim}")

    # ── 4. 检查依赖 ──────────────────────────────────────────────────
    try:
        import emcee
        import corner as _corner  # noqa: F401
        from tqdm import tqdm
    except ImportError as e:
        print(f"[错误] 缺少依赖: {e}\n  请运行: pip install emcee corner tqdm")
        sys.exit(1)

    # ── 5. 构建固定透镜场缓存 ─────────────────────────────────────────
    print(f"\n构建固定透镜场缓存（仅 Sersic + {main_lens_key}）...")
    cache = build_fixed_cache(lens_params, main_lens_key, src_x, src_y,
                              device, dtype)

    # ── 6. 构造向量化 log_prob ────────────────────────────────────────
    log_prob = VectorisedLogProbPM(
        n_subhalos=len(active_subhalos),
        active_subhalos=active_subhalos,
        src_x=src_x, src_y=src_y, cache=cache,
        device=device, dtype=dtype,
        prior_radius=MCMC_SEARCH_RADIUS,
        log_m_min=MCMC_PM_LOG_M_MIN,
        log_m_max=MCMC_PM_LOG_M_MAX,
    )

    print(f"\n  先验范围: 位置半径 ±{MCMC_SEARCH_RADIUS * 1000:.0f} mas")
    print(f"    logM ∈ [{MCMC_PM_LOG_M_MIN}, {MCMC_PM_LOG_M_MAX}] dex")

    # 验证 DE 最优解
    lp_test = log_prob(p0[None, :])[0]
    if not np.isfinite(lp_test):
        print(f"[警告] DE 最优解处 log_prob = {lp_test}，初始点可能不可行")
    else:
        print(f"  初始 log_prob = {lp_test:.4f}（chi2 ≈ {-2 * lp_test:.2f}）")

    # ── 7. 运行 MCMC ─────────────────────────────────────────────────
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

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob, vectorize=True,
    )

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

    # ── 8. 保存 chain ────────────────────────────────────────────────
    chain_file = os.path.join(folder, f"{prefix}_mcmc_chain.dat")
    np.savetxt(chain_file, samples, header=" ".join(param_names))
    print(f"\n  ✓ MCMC链: {os.path.basename(chain_file)}")

    # ── 9. 绘图 ──────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    print("\n生成图表...")
    plot_corner(samples, corner_labels, p0,
                os.path.join(folder, f"{prefix}_mcmc_corner.png"))
    plot_trace(chain, corner_labels, args.burnin,
               os.path.join(folder, f"{prefix}_mcmc_trace.png"))
    plot_mass_1d(model_type, active_subhalos, samples, subhalos,
                 os.path.join(folder, f"{prefix}_mcmc_mass_1d.png"))

    # ── 10. 后验统计 ─────────────────────────────────────────────────
    posterior_file = os.path.join(folder, f"{prefix}_mcmc_posterior.txt")
    save_posterior(samples, param_names, model_type, active_subhalos,
                   subhalos, args, nwalkers, posterior_file)

    print(f"\n{'=' * 70}")
    print(f"完成！输出文件均在: {folder}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
