#!/usr/bin/env python3
"""
Version None 1.0 — No Subhalo: Source / Lens Parameter Optimization
══════════════════════════════════════════════════════════════════════

不添加任何子晕，直接优化基准模型的源位置和/或透镜参数。

优化变量（根据开关动态组合）:
  source_modify = True  → 源面位置 (x, y)
  lens_modify   = True  → 透镜参数

边界模式（fine_tuning 开关切换）:
  fine_tuning = False → 对各非零参数以 modify_percentage 设 ±% 边界
                         source 以 ±source_x/y_delta 为边界
  fine_tuning = True  → 以 lens_optimize_bounds / source_x/y_bounds 为显式边界

lens_optimize_bounds 格式（fine_tuning=True 时生效）:
    {
        'lens_key': [bound_p1, ..., bound_p7],
        # bound: None = 不优化; (lower, upper) = 优化边界
    }
    lens_key 与 load_baseline_lens_params 返回的键名一致
    （如 'sers1', 'sers2', 'sie', 'anfw', ...）
"""

import sys
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'fork':
    multiprocessing.set_start_method('fork', force=True)
import random
import glafic
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, linear_sum_assignment
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
from plot_paper_style import plot_paper_style, plot_paper_style_compare, read_critical_curves

# ══════════════════════════════════════════════════════════
# §1  公共辅助函数（与其他版本保持一致）
# ══════════════════════════════════════════════════════════

def load_baseline_lens_params(directory):
    """
    从指定目录的 bestfit.dat 加载基准透镜参数。
    格式: lens <type> z p1..p7 / point z_s x_s y_s
    至少需要 1 行 lens 和 1 行 point；参数不足 7 个自动补 0。
    """
    bestfit_path = os.path.join(directory, 'bestfit.dat')
    if not os.path.isfile(bestfit_path):
        raise FileNotFoundError(f"未找到: {bestfit_path}")
    lens_lines, point_params = [], None
    with open(bestfit_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            if parts[0] == 'lens':
                lens_lines.append(parts)
            elif parts[0] == 'point':
                point_params = parts
    if len(lens_lines) < 1:
        raise ValueError(f"bestfit.dat 至少需要 1 行 lens: {bestfit_path}")
    if point_params is None:
        raise ValueError(f"bestfit.dat 缺少 point 行: {bestfit_path}")
    params_dict, sers_count, type_counts, main_lens_key = {}, 0, {}, None
    for parts in lens_lines:
        ltype = parts[1]
        z = float(parts[2])
        raw = [float(v) for v in parts[3:]]
        vals = (raw + [0.0] * 7)[:7]
        idx = len(params_dict) + 1
        if ltype == 'sers':
            sers_count += 1
            key = f'sers{sers_count}'
        else:
            type_counts[ltype] = type_counts.get(ltype, 0) + 1
            n = type_counts[ltype]
            key = ltype if n == 1 else f'{ltype}{n}'
            main_lens_key = key
        params_dict[key] = (idx, ltype, z, *vals)
    if main_lens_key is None:
        main_lens_key = list(params_dict.keys())[-1]
    return params_dict, float(point_params[2]), float(point_params[3]), main_lens_key


def find_glafic_bin(default_path=""):
    if os.path.isfile(default_path) and os.access(default_path, os.X_OK):
        return default_path
    try:
        module_dir = os.path.dirname(os.path.abspath(glafic.__file__))
        for rel in ['../glafic', '../../glafic', './glafic', '../bin/glafic']:
            p = os.path.abspath(os.path.join(module_dir, rel))
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    except Exception:
        pass
    import shutil
    return shutil.which('glafic')


print("=" * 70)
print("Version None 1.0: No Subhalo — Source/Lens Parameter Optimization")
print("=" * 70)

# ══════════════════════════════════════════════════════════
# §2  可配置参数区
# ══════════════════════════════════════════════════════════

# ── 0. 基准透镜参数路径 ────────────────────────────────────
BASELINE_LENS_DIR = ''

# ── 1. 约束条件 ────────────────────────────────────────────
CONSTRAINT_SIGMA = 1
PENALTY_COEFFICIENT = 1000

# ── 2. 优化目标 ────────────────────────────────────────────
source_modify = True
lens_modify = True

# fine_tuning=False: 百分比边界 (modify_percentage / source_x/y_delta)
# fine_tuning=True : 显式上下限 (lens_optimize_bounds / source_x/y_bounds)
fine_tuning = False

# --- fine_tuning=False 时 ---
modify_percentage = 0.1
source_x_delta = 0.01
source_y_delta = 0.01

# --- fine_tuning=True 时 ---
# Source 显式边界（相对于 bestfit 中源位置的偏移量，arcsec）
source_x_bounds = [-0.3, 0.3]
source_y_bounds = [-0.3, 0.3]

# 透镜参数显式边界
# 格式: {lens_key: [bound_p1, ..., bound_p7]}
# bound: None=不优化, (lower, upper)=优化边界
lens_optimize_bounds = {}

# ── 3. 损失函数（与 pointmass 一致）──────────────────────────
# Y_i = A*(Δpos/σ_pos)² + B*(Δμ/σ_μ)² + P_i
# P_i = PENALTY_PL * Δpos_mas  (仅当 Δpos > σ_pos 时)
LOSS_COEF_A = 1
LOSS_COEF_B = 1
LOSS_PENALTY_PL = 1000

# ── 4. 差分进化 ────────────────────────────────────────────
DE_MAXITER = 640
DE_POPSIZE = 32
DE_ATOL = 0.0001
DE_TOL = 1e-06
DE_SEED = 42
DE_POLISH = True
DE_WORKERS = -1
EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 64

# ── 5. MCMC ────────────────────────────────────────────────
MCMC_ENABLED = False
MCMC_NWALKERS = 32
MCMC_NSTEPS = 1000
MCMC_BURNIN = 200
MCMC_THIN = 2
MCMC_PERTURBATION = 0.01
MCMC_PROGRESS = True
MCMC_WORKERS = 1

# ── 6. 输出 ────────────────────────────────────────────────
SHOW_2SIGMA = False
OUTPUT_PREFIX = 'v_none_1_0'
COMPARE_GRAPH = True
Draw_Graph = 1
draw_interval = 25
PRINT_INTERVAL = 10

# ── 可注入的观测数据 ────────────────────────────────────────
obs_positions_mas_list = [[-330.461, 0], [330.461, 0], [0, -262.771], [0, 262.771]]
obs_magnifications_list = [2.92052, 2.92052, -1.52908, -1.52908]
obs_mag_errors_list = [0.2, 0.2, 0.2, 0.2]
obs_pos_sigma_mas_list = [0.5, 0.5, 0.691, 0.691]
center_offset_x = 0
center_offset_y = 0
obs_x_flip = True

# 坐标转换：统一取符号，同时作用于观测位置和中心偏移，确保两者始终在同一坐标系下
_x_sign = -1 if obs_x_flip else 1
obs_positions_mas  = np.array(obs_positions_mas_list)
obs_positions      = np.zeros_like(obs_positions_mas)
obs_positions[:, 0] = _x_sign * obs_positions_mas[:, 0] / 1000.0
obs_positions[:, 1] = obs_positions_mas[:, 1] / 1000.0
center_offset_x = 0
obs_magnifications  = np.array(obs_magnifications_list)
obs_mag_errors      = np.array(obs_mag_errors_list)
obs_pos_sigma_mas   = np.array(obs_pos_sigma_mas_list)
n_obs               = len(obs_positions)

# ── glafic 基础参数 ─────────────────────────────────────────
omega        = 0.3
lambda_cosmo = 0.7
weos         = -1.0
hubble       = 0.7
xmin, ymin   = -0.5, -0.5
xmax, ymax   =  0.5,  0.5
pix_ext      = 0.01
pix_poi      = 0.2
maxlev       = 5
source_z     = 0.4090
lens_z       = 0.2160

# ── 默认基准透镜参数 ─────────────────────────────────────────
source_x = 2.685497e-03
source_y = 2.443616e-02
lens_params = {
    'sers1': (1, 'sers', 0.2160, 9.896617e+09, 2.656977e-03, 2.758473e-02,
              2.986760e-01, 1.124730e+02, 3.939718e-01, 1.057760e+00),
    'sers2': (2, 'sers', 0.2160, 2.555580e+10, 2.656977e-03, 2.758473e-02,
              4.242340e-01, 5.396370e+01, 1.538855e+00, 1.000000e+00),
    'sie':   (3, 'sie',  0.2160, 1.183382e+02, 2.656977e-03, 2.758473e-02,
              1.571203e-01, 2.920348e+01, 0.0, 0.0),
}
MAIN_LENS_KEY = 'sie'

if BASELINE_LENS_DIR:
    _loaded, _sx, _sy, _mlk = load_baseline_lens_params(BASELINE_LENS_DIR)
    lens_params   = _loaded
    source_x      = _sx
    source_y      = _sy
    MAIN_LENS_KEY = _mlk
    print(f"[基准透镜] 已从 {BASELINE_LENS_DIR} 加载 (主透镜: {MAIN_LENS_KEY})")
else:
    print("[基准透镜] 使用内置默认 SIE 参数")

lens_params_ref = {k: list(v) for k, v in lens_params.items()}
source_x_ref    = source_x
source_y_ref    = source_y

# ══════════════════════════════════════════════════════════
# §3  构建优化变量映射
# ══════════════════════════════════════════════════════════

def build_optimization():
    """
    返回 (bounds, pmap):
      bounds: list of (lo, hi) 对应各优化变量
      pmap  : list of ('src_x'|'src_y'|'lens', key, param_idx)
    """
    bounds, pmap = [], []

    if source_modify:
        sx, sy = source_x_ref, source_y_ref
        if fine_tuning:
            bounds += [(sx + source_x_bounds[0], sx + source_x_bounds[1]),
                       (sy + source_y_bounds[0], sy + source_y_bounds[1])]
        else:
            bounds += [(sx - source_x_delta, sx + source_x_delta),
                       (sy - source_y_delta, sy + source_y_delta)]
        pmap += [('src_x', None, None), ('src_y', None, None)]

    if lens_modify:
        for key, pv in lens_params_ref.items():
            ps = pv[3:]   # p1..p7
            if fine_tuning:
                kbounds = lens_optimize_bounds.get(key, [None] * 7)
                for pi in range(7):
                    b = kbounds[pi] if pi < len(kbounds) else None
                    if b is not None:
                        bounds.append(tuple(b))
                        pmap.append(('lens', key, pi))
            else:
                for pi, val in enumerate(ps[:7]):
                    if abs(val) > 1e-30:
                        d = abs(val) * modify_percentage
                        bounds.append((val - d, val + d))
                        pmap.append(('lens', key, pi))

    return bounds, pmap


bounds, pmap = build_optimization()
ndim = len(bounds)

print(f"\n优化维度: {ndim}")
print(f"  source_modify = {source_modify}, lens_modify = {lens_modify}, fine_tuning = {fine_tuning}")
if fine_tuning and lens_modify and lens_optimize_bounds:
    for key, blist in lens_optimize_bounds.items():
        active = [(i+1, b) for i, b in enumerate(blist) if b is not None]
        if active:
            print(f"  {key}: " + ", ".join(f"p{i}=[{b[0]:.4g},{b[1]:.4g}]" for i, b in active))

# ══════════════════════════════════════════════════════════
# §4  输出目录
# ══════════════════════════════════════════════════════════

timestamp  = datetime.now().strftime("%y%m%d_%H%M")
output_dir = timestamp
os.makedirs(output_dir, exist_ok=True)
print(f"\n输出目录: {output_dir}")

# ══════════════════════════════════════════════════════════
# §5  核心计算函数
# ══════════════════════════════════════════════════════════

def machine_learning_loss(pred_pos, pred_mag, delta_pos_mas):
    """计算图像位置+放大率 chi^2 损失。
    当 obs_mag_errors[i]==0 时跳过放大率项（仅凭位置优化），避免除以零。
    """
    total = 0.0
    for i in range(n_obs):
        chi2_pos = (delta_pos_mas[i] / obs_pos_sigma_mas[i]) ** 2
        if obs_mag_errors[i] > 0:
            chi2_mag = ((pred_mag[i] - obs_magnifications[i]) / obs_mag_errors[i]) ** 2
        else:
            chi2_mag = 0.0   # 误差未指定时跳过放大率约束
        penalty  = (0.0 if delta_pos_mas[i] <= obs_pos_sigma_mas[i]
                    else LOSS_PENALTY_PL * delta_pos_mas[i])
        total += LOSS_COEF_A * chi2_pos + LOSS_COEF_B * chi2_mag + penalty
    return total


def compute_model(sx, sy, lp_dict, verbose=False):
    """
    计算图像，始终返回有效值——即使像数不足也给出部分匹配结果。

    返回 (matched_pos, matched_mag, delta_mas, loss)
      - matched_pos  : (n_obs, 2) 匹配后的预测位置；未匹配行为 0
      - matched_mag  : (n_obs,)   匹配后的预测放大率；未匹配行为 0
      - delta_mas    : (n_obs,)   位置偏差 [mas]；未匹配行为 1000
      - loss         : mag_chi2 + missing_penalty（每少一个像 +1e5）

    注意：使用进程 PID 作为临时文件后缀，避免并行 worker 间的文件冲突。
    """
    # 每个进程用独立的临时文件前缀，防止 workers=-1 时文件冲突
    _prefix = f'temp_{OUTPUT_PREFIX}_{os.getpid()}'
    glafic.init(omega, lambda_cosmo, weos, hubble,
                _prefix,
                xmin, ymin, xmax, ymax,
                pix_ext, pix_poi, maxlev, verb=0)

    glafic.startup_setnum(len(lp_dict), 0, 1)
    for key, pv in lp_dict.items():
        glafic.set_lens(*pv)
    glafic.set_point(1, source_z, sx, sy)
    glafic.model_init(verb=0)

    result = glafic.point_solve(source_z, sx, sy, verb=0)
    n_img = len(result)

    # 丢弃中心像（NFW 类透镜会产生一个额外的低放大率中心像）
    if n_img == n_obs + 1:
        drop = int(np.argmin([abs(r[2]) for r in result]))
        result = [r for i, r in enumerate(result) if i != drop]
        if verbose:
            print(f"  Info: {n_img} 像 → 丢弃中心像 {drop}")
        n_img = len(result)

    n_missing = max(0, n_obs - n_img)
    extra_penalty = n_missing * 1e5   # 每少一个像 +1e5

    if verbose and n_missing > 0:
        print(f"  Warning: 找到 {n_img} 个像，期望 {n_obs}"
              f"（缺少 {n_missing} 个，penalty +{n_missing:.0e}）")

    # 初始化"未匹配"缺省值：大偏差 + 零放大率
    matched_pos = np.zeros((n_obs, 2))
    matched_mag = np.zeros(n_obs)
    delta_mas   = np.full(n_obs, 1000.0)

    if n_img > 0:
        # 按 |μ| 降序取最多 n_obs 个像进行匹配
        n_use = min(n_img, n_obs)
        result_s = sorted(result, key=lambda r: abs(r[2]), reverse=True)
        pred_pos = np.array([[r[0], r[1]] for r in result_s[:n_use]])
        pred_mag = np.array([r[2]         for r in result_s[:n_use]])
        pred_pos[:, 0] += center_offset_x
        pred_pos[:, 1] += center_offset_y

        # 匈牙利算法：(n_obs × n_use) 矩形代价矩阵
        row_ind, col_ind = linear_sum_assignment(cdist(obs_positions, pred_pos))
        for ri, ci in zip(row_ind, col_ind):
            matched_pos[ri] = pred_pos[ci]
            matched_mag[ri] = pred_mag[ci]
            delta_mas[ri] = np.sqrt(
                ((pred_pos[ci, 0] - obs_positions[ri, 0]) * 1000) ** 2 +
                ((pred_pos[ci, 1] - obs_positions[ri, 1]) * 1000) ** 2
            )

    glafic.quit()
    loss = machine_learning_loss(matched_pos, matched_mag, delta_mas) + extra_penalty
    return matched_pos, matched_mag, delta_mas, loss, extra_penalty


# ══════════════════════════════════════════════════════════
# §6  基准模型评估
# ══════════════════════════════════════════════════════════

_zero_err_idx = [i for i in range(n_obs) if obs_mag_errors[i] == 0]
if _zero_err_idx:
    print(f"\n[警告] 图像 {[i+1 for i in _zero_err_idx]} 的放大率误差为 0，"
          f"mag_chi2 项将被跳过（仅使用位置约束）。"
          f"\n       如需启用放大率约束，请在 obs_mag_errors_list 中填写非零误差。")

print("\n" + "=" * 70)
print("步骤 1: 基准模型（无子晕）")
print("=" * 70)

lp_tuples = {k: tuple(v) for k, v in lens_params_ref.items()}
base_pos, base_mag, base_delta, base_loss, base_missing = compute_model(
    source_x_ref, source_y_ref, lp_tuples, verbose=True)

base_pos_chi2 = sum(LOSS_COEF_A * (base_delta[i] / obs_pos_sigma_mas[i])**2 for i in range(n_obs))
base_mag_chi2 = sum(
    LOSS_COEF_B * ((base_mag[i] - obs_magnifications[i]) / obs_mag_errors[i])**2
    if obs_mag_errors[i] > 0 else 0.0
    for i in range(n_obs))
print(f"\n  位置 RMS  : {np.sqrt(np.mean(base_delta**2)):.3f} mas")
print(f"  最大偏差  : {base_delta.max():.3f} mas")
print(f"  loss      : {base_loss:.4f}  (pos_chi2={base_pos_chi2:.3f}, mag_chi2={base_mag_chi2:.3f})")
if base_missing > 0:
    n_miss_base = int(round(base_missing / 1e5))
    print(f"  [注意] 基准模型像数不足，缺少约 {n_miss_base} 个像，missing_penalty = {base_missing:.2e}")

# ══════════════════════════════════════════════════════════
# §7  目标函数
# ══════════════════════════════════════════════════════════

def objective(x):
    sx = source_x_ref
    sy = source_y_ref
    cur_lp = {k: list(v) for k, v in lens_params_ref.items()}

    for i, (ptype, key, pi) in enumerate(pmap):
        if ptype == 'src_x':   sx = x[i]
        elif ptype == 'src_y': sy = x[i]
        else: cur_lp[key][3 + pi] = x[i]

    lp_t = {k: tuple(v) for k, v in cur_lp.items()}
    _, _, delta_mas, loss, _ = compute_model(sx, sy, lp_t)
    return loss


# ══════════════════════════════════════════════════════════
# §8  迭代绘图函数
# ══════════════════════════════════════════════════════════

def plot_iteration_population(population, iteration_num, output_dir, bounds, pmap):
    """绘制种群参数分布直方图"""
    if Draw_Graph == 0 or ndim == 0:
        return
    if iteration_num % draw_interval != 0 and iteration_num != 0:
        return

    if population.max() <= 1.0 and population.min() >= 0.0:
        population_denorm = np.zeros_like(population)
        for i in range(population.shape[1]):
            lower, upper = bounds[i]
            population_denorm[:, i] = population[:, i] * (upper - lower) + lower
        population = population_denorm

    n_params = population.shape[1]
    ncols = min(n_params, 4)
    nrows = (n_params + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if n_params == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    param_labels = []
    for ptype, key, pi in pmap:
        if ptype == 'src_x':
            param_labels.append('src_x')
        elif ptype == 'src_y':
            param_labels.append('src_y')
        else:
            param_labels.append(f'{key}_p{pi+1}')

    for idx in range(n_params):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.hist(population[:, idx], bins=20, alpha=0.6, color='steelblue')
        ax.set_xlabel(param_labels[idx], fontsize=9)
        ax.set_ylabel('Count', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.set_xlim(bounds[idx])

    for idx in range(n_params, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.suptitle(f'Iteration {iteration_num}: Parameter Distribution',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = os.path.join(output_dir, f'iteration_{iteration_num:04d}.png')
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"    保存迭代图: iteration_{iteration_num:04d}.png")


# ══════════════════════════════════════════════════════════
# §9  DE 优化（DifferentialEvolutionSolver）
# ══════════════════════════════════════════════════════════

if draw_interval < 1:
    draw_interval = 1

print("\n" + "=" * 70)
if ndim == 0:
    print("步骤 2: 无优化参数（source_modify=False, lens_modify=False）")
    print("        直接输出基准模型结果。")
    print("=" * 70)
    best_x    = np.array([])
    best_sx, best_sy = source_x_ref, source_y_ref
    best_lp  = {k: tuple(v) for k, v in lens_params_ref.items()}
    best_loss = base_loss
else:
    print(f"步骤 2: 差分进化优化 (维度={ndim})")
    print("=" * 70)

    from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
    import scipy
    print(f"  Scipy版本: {scipy.__version__}")

    np.random.seed(DE_SEED)
    solver = DifferentialEvolutionSolver(
        objective, bounds,
        maxiter=DE_MAXITER, popsize=DE_POPSIZE,
        atol=DE_ATOL, tol=DE_TOL,
        rng=np.random.default_rng(DE_SEED), polish=DE_POLISH,
        disp=False, workers=DE_WORKERS, updating='deferred')

    if Draw_Graph:
        plot_iteration_population(solver.population.copy(), 0, output_dir, bounds, pmap)

    iteration = 1
    previous_best_energy = np.min(solver.population_energies)
    best_ever_energy = previous_best_energy
    converged_count = 0

    print(f"\n迭代 0: 初始最佳 = {previous_best_energy:.6f}")

    while True:
        try:
            next_gen = solver.__next__()
        except StopIteration:
            print(f"\n  优化收敛！")
            break

        current_best_energy = np.min(solver.population_energies)

        if iteration % PRINT_INTERVAL == 0 or current_best_energy < best_ever_energy:
            if current_best_energy < best_ever_energy:
                print(f"迭代 {iteration}: 最佳 = {current_best_energy:.6f}  (改进)")
                best_ever_energy = current_best_energy
            else:
                print(f"迭代 {iteration}: 最佳 = {current_best_energy:.6f}")

        if Draw_Graph and iteration % draw_interval == 0:
            plot_iteration_population(solver.population.copy(), iteration, output_dir, bounds, pmap)

        abs_change = abs(current_best_energy - previous_best_energy)
        if abs_change < DE_ATOL:
            converged_count += 1
            if EARLY_STOPPING and converged_count >= EARLY_STOP_PATIENCE:
                print(f"\n  早停触发！连续 {converged_count} 次满足容差。")
                break
        else:
            converged_count = 0

        previous_best_energy = current_best_energy
        iteration += 1

        if iteration > DE_MAXITER:
            print(f"\n  达到最大迭代次数 {DE_MAXITER}。")
            break

    print(f"\n总迭代次数: {iteration}  最终最佳值: {best_ever_energy:.6f}")

    best_x = solver.x
    best_loss = np.min(solver.population_energies)

    best_sx, best_sy = source_x_ref, source_y_ref
    best_lp_list = {k: list(v) for k, v in lens_params_ref.items()}
    for i, (ptype, key, pi) in enumerate(pmap):
        if ptype == 'src_x': best_sx = best_x[i]
        elif ptype == 'src_y': best_sy = best_x[i]
        else: best_lp_list[key][3 + pi] = best_x[i]
    best_lp = {k: tuple(v) for k, v in best_lp_list.items()}

    print(f"\n  最优 loss = {best_loss:.6f}")

# ══════════════════════════════════════════════════════════
# §10  最优结果评估
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("步骤 3: 最优结果分析")
print("=" * 70)

opt_pos, opt_mag, opt_delta, opt_loss, opt_missing = compute_model(
    best_sx, best_sy, best_lp, verbose=True)

opt_pos_chi2 = sum(LOSS_COEF_A * (opt_delta[i] / obs_pos_sigma_mas[i])**2 for i in range(n_obs))
opt_mag_chi2 = sum(
    LOSS_COEF_B * ((opt_mag[i] - obs_magnifications[i]) / obs_mag_errors[i])**2
    if obs_mag_errors[i] > 0 else 0.0
    for i in range(n_obs))
print(f"\n  位置 RMS  : {np.sqrt(np.mean(opt_delta**2)):.3f} mas")
print(f"  最大偏差  : {opt_delta.max():.3f} mas")
print(f"  loss      : {opt_loss:.4f}  (pos_chi2={opt_pos_chi2:.3f}, mag_chi2={opt_mag_chi2:.3f})")
improvement = (base_loss - opt_loss) / base_loss * 100 if base_loss > 0 else 0
print(f"  改善      : {improvement:.1f}%  ({base_loss:.4f} → {opt_loss:.4f})")
if opt_missing > 0:
    n_miss_opt = int(round(opt_missing / 1e5))
    print(f"  [注意] 像数不足，缺少约 {n_miss_opt} 个像，missing_penalty = {opt_missing:.2e}")
for i in range(n_obs):
    chi2_pos_i = LOSS_COEF_A * (opt_delta[i] / obs_pos_sigma_mas[i])**2
    chi2_mag_i = (LOSS_COEF_B * ((opt_mag[i] - obs_magnifications[i]) / obs_mag_errors[i])**2
                  if obs_mag_errors[i] > 0 else 0.0)
    print(f"  像 {i+1}: Δpos={opt_delta[i]:.3f} mas (χ²_pos={chi2_pos_i:.2f})  "
          f"μ_pred={opt_mag[i]:.2f}  μ_obs={obs_magnifications[i]:.2f} (χ²_mag={chi2_mag_i:.2f})")

if source_modify:
    print(f"\n  最优源位置: x={best_sx:.6e}, y={best_sy:.6e} arcsec")
if lens_modify:
    print("\n  最优透镜参数:")
    lens_keys_in_pmap = {k for _, k, _ in pmap if k is not None}
    for key, pv in best_lp.items():
        if key in lens_keys_in_pmap:
            print(f"    {key}: {[f'{v:.4g}' for v in pv[3:]]}")

max_pos_deviation_mas = CONSTRAINT_SIGMA * obs_pos_sigma_mas
constraint_satisfied = all(opt_delta[i] <= max_pos_deviation_mas[i] for i in range(n_obs))

# ══════════════════════════════════════════════════════════
# §11  保存结果
# ══════════════════════════════════════════════════════════

best_params_file = os.path.join(output_dir, f"{OUTPUT_PREFIX}_best_params.txt")
with open(best_params_file, 'w') as f:
    f.write(f"# Version None 1.0 — Best Parameters\n")
    f.write(f"# Run: {timestamp}\n")
    f.write(f"# source_modify={source_modify}, lens_modify={lens_modify}, fine_tuning={fine_tuning}\n")
    f.write(f"# best_loss={best_loss:.8f}\n\n")

    f.write(f"Source Optimized: source_x={best_sx:.8e}  source_y={best_sy:.8e}\n\n")

    f.write("Optimized Lens Parameters:\n")
    for key, pv in best_lp.items():
        idx_g, ltype, z = pv[0], pv[1], pv[2]
        ps = pv[3:]
        f.write(f"  lens {ltype} {z:.4f}  " +
                "  ".join(f"{v:.6e}" for v in ps) + f"  # {key}\n")

    f.write(f"\nResults:\n")
    f.write(f"  pos_rms_mas  = {np.sqrt(np.mean(opt_delta**2)):.4f}\n")
    f.write(f"  pos_chi2     = {opt_pos_chi2:.4f}  (A={LOSS_COEF_A})\n")
    f.write(f"  mag_chi2     = {opt_mag_chi2:.4f}  (B={LOSS_COEF_B})\n")
    f.write(f"  total_loss   = {opt_loss:.4f}\n")
    f.write(f"  improvement  = {improvement:.2f}%\n")
    if opt_missing > 0:
        f.write(f"  missing_penalty = {opt_missing:.2e}\n")
    for i in range(n_obs):
        f.write(f"  image_{i+1}: delta_pos={opt_delta[i]:.4f} mas  "
                f"mu_pred={opt_mag[i]:.3f}  mu_obs={obs_magnifications[i]:.3f}\n")

print(f"\n  结果已保存: {best_params_file}")

# ══════════════════════════════════════════════════════════
# §12  MCMC 后验采样
# ══════════════════════════════════════════════════════════

if MCMC_ENABLED and ndim > 0:
    print("\n" + "=" * 70)
    print("步骤 4: MCMC 后验采样（基于DE最优解）")
    print("=" * 70)

    try:
        import emcee
        import corner
        from tqdm import tqdm
        print(f"  emcee, corner, tqdm 已导入")
    except ImportError as e:
        print(f"  缺少依赖库: {e}")
        print(f"    请运行: pip install emcee corner tqdm")
        MCMC_ENABLED = False

if MCMC_ENABLED and ndim > 0:

    def log_probability(params):
        for i, (low, high) in enumerate(bounds):
            if not (low <= params[i] <= high):
                return -np.inf
        loss = objective(params)
        if loss >= 1e10:
            return -np.inf
        return -0.5 * loss

    nw = max(MCMC_NWALKERS, 2 * ndim + 2)
    print(f"\n初始化MCMC采样器:")
    print(f"  参数维度: {ndim}")
    print(f"  Walkers: {nw}")
    print(f"  采样步数: {MCMC_NSTEPS}")
    print(f"  Burn-in: {MCMC_BURNIN}")

    initial_positions = []
    rng = np.random.default_rng(DE_SEED)
    for _ in range(nw):
        perturbation = np.array([
            rng.normal(0, MCMC_PERTURBATION * (bounds[i][1] - bounds[i][0]))
            for i in range(ndim)
        ])
        new_pos = best_x + perturbation
        new_pos = np.clip(new_pos, [b[0] for b in bounds], [b[1] for b in bounds])
        initial_positions.append(new_pos)
    initial_positions = np.array(initial_positions)

    if MCMC_WORKERS == -1:
        mcmc_workers_actual = os.cpu_count() or 1
    else:
        mcmc_workers_actual = MCMC_WORKERS

    print(f"  并行核心数: {mcmc_workers_actual}" + (" (全部CPU)" if MCMC_WORKERS == -1 else ""))

    if mcmc_workers_actual > 1:
        from multiprocessing import Pool
        with Pool(mcmc_workers_actual) as pool:
            sampler = emcee.EnsembleSampler(nw, ndim, log_probability, pool=pool)
            print(f"\n开始MCMC采样（{mcmc_workers_actual}核并行）...")
            if MCMC_PROGRESS:
                for sample in tqdm(sampler.sample(initial_positions, iterations=MCMC_NSTEPS),
                                   total=MCMC_NSTEPS, desc="MCMC采样"):
                    pass
            else:
                sampler.run_mcmc(initial_positions, MCMC_NSTEPS, progress=False)
            samples = sampler.get_chain(discard=MCMC_BURNIN, thin=MCMC_THIN, flat=True)
            chain   = sampler.get_chain()
    else:
        sampler = emcee.EnsembleSampler(nw, ndim, log_probability)
        print(f"\n开始MCMC采样（串行模式）...")
        if MCMC_PROGRESS:
            for sample in tqdm(sampler.sample(initial_positions, iterations=MCMC_NSTEPS),
                               total=MCMC_NSTEPS, desc="MCMC采样"):
                pass
        else:
            sampler.run_mcmc(initial_positions, MCMC_NSTEPS, progress=False)
        samples = sampler.get_chain(discard=MCMC_BURNIN, thin=MCMC_THIN, flat=True)
        chain   = sampler.get_chain()

    print(f"\n采样完成:")
    print(f"  总样本数: {nw * MCMC_NSTEPS}")
    print(f"  有效样本数（去除burn-in）: {len(samples)}")

    param_names = []
    for ptype, key, pi in pmap:
        if ptype == 'src_x':
            param_names.append('src_x')
        elif ptype == 'src_y':
            param_names.append('src_y')
        else:
            param_names.append(f'{key}_p{pi+1}')

    mcmc_chain_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_mcmc_chain.dat')
    np.savetxt(mcmc_chain_file, samples, header=' '.join(param_names))
    print(f"  MCMC链已保存: {mcmc_chain_file}")

    # 参数统计
    print(f"\n" + "=" * 70)
    print("MCMC 后验分布统计")
    print("=" * 70)

    posterior_stats = {}
    print(f"\n参数后验分布 (median +upper_1σ -lower_1σ):")
    for i, name in enumerate(param_names):
        median = np.median(samples[:, i])
        lower  = np.percentile(samples[:, i], 16)
        upper  = np.percentile(samples[:, i], 84)
        posterior_stats[name] = {
            'median': median, 'lower_1sigma': lower, 'upper_1sigma': upper,
            'error_plus': upper - median, 'error_minus': median - lower
        }
        if name.startswith('src_'):
            print(f"  {name}: {median:.6e} +{(upper-median):.3e} -{(median-lower):.3e}")
        else:
            print(f"  {name}: {median:.6e} +{upper-median:.3e} -{median-lower:.3e}")

    # Corner Plot
    print(f"\n生成 Corner Plot...")
    corner_labels = []
    for ptype, key, pi in pmap:
        if ptype == 'src_x':
            corner_labels.append('$x_s$')
        elif ptype == 'src_y':
            corner_labels.append('$y_s$')
        else:
            corner_labels.append(f'${key}\\_p{pi+1}$')

    fig = corner.corner(
        samples,
        labels=corner_labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_fmt='.4f',
        truths=best_x,
        truth_color='red',
        hist_kwargs={'alpha': 0.75},
    )
    _corner_grid = np.array(fig.axes).reshape((ndim, ndim))
    for _ci in range(ndim):
        _ax = _corner_grid[_ci, _ci]
        from matplotlib.ticker import MaxNLocator as _MLoc
        _ylo, _yhi = _ax.get_ylim()
        _ax2 = _ax.twinx()
        _N_corner = len(samples)
        _ax2.set_ylim(_ylo / _N_corner * 100, _yhi / _N_corner * 100)
        _ax2.yaxis.set_major_locator(_MLoc(nbins=4, prune='lower'))
        _ax2.tick_params(axis='y', labelsize=7, length=3, width=0.8)
        _ax2.set_ylabel('%', fontsize=8, rotation=0, labelpad=10, va='center')

    corner_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_corner.png')
    plt.savefig(corner_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Corner plot 已保存: {corner_file}")

    # Trace Plot
    print(f"\n生成 MCMC 链轨迹图...")
    fig, axes = plt.subplots(ndim, figsize=(10, 2 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(chain[:, :, i], alpha=0.3)
        ax.axvline(MCMC_BURNIN, color='red', linestyle='--', label='Burn-in')
        ax.set_ylabel(corner_labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("Step")
    axes[0].legend(loc='upper right')
    trace_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_trace.png')
    plt.savefig(trace_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  轨迹图已保存: {trace_file}")

    # 保存后验统计文件
    posterior_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_posterior.txt')
    with open(posterior_file, 'w') as f:
        f.write("# ============================================================\n")
        f.write("# MCMC Posterior Distribution Summary\n")
        f.write("# Version None 1.0\n")
        f.write("# ============================================================\n\n")
        f.write(f"# Walkers: {nw}, Steps: {MCMC_NSTEPS}, "
                f"Burn-in: {MCMC_BURNIN}, Thin: {MCMC_THIN}\n")
        f.write(f"# Effective samples: {len(samples)}\n\n")
        f.write("# parameter  median  16%_lower  84%_upper  error_plus  error_minus\n\n")
        for i, name in enumerate(param_names):
            st = posterior_stats[name]
            f.write(f"{name}  {st['median']:.10e}  {st['lower_1sigma']:.10e}  "
                    f"{st['upper_1sigma']:.10e}  {st['error_plus']:.10e}  {st['error_minus']:.10e}\n")
    print(f"  后验统计已保存: {posterior_file}")

# ══════════════════════════════════════════════════════════
# §13  三联图（Critical Curves + Paper Style）
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("步骤 5: 生成最终图表")
print("=" * 70)

glafic.init(omega, lambda_cosmo, weos, hubble, f'temp_{OUTPUT_PREFIX}_best',
            xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev, verb=0)
glafic.startup_setnum(len(best_lp), 0, 1)
for key, pv in best_lp.items():
    glafic.set_lens(*pv)
glafic.set_point(1, source_z, best_sx, best_sy)
glafic.model_init(verb=0)
glafic.writecrit(source_z)

crit_file = f'temp_{OUTPUT_PREFIX}_best_crit.dat'
crit_segments, caus_segments = read_critical_curves(crit_file)
glafic.quit()

output_plot_file = os.path.join(output_dir, f"result_{OUTPUT_PREFIX}.png")

if COMPARE_GRAPH and ndim > 0:
    output_plot_file_compare = os.path.join(output_dir, f"result_{OUTPUT_PREFIX}_compare.png")
    plot_paper_style_compare(
        img_numbers=np.arange(1, n_obs + 1),
        delta_pos_mas_baseline=base_delta,
        delta_pos_mas_optimized=opt_delta,
        sigma_pos_mas=obs_pos_sigma_mas,
        mu_obs=obs_magnifications,
        mu_obs_err=obs_mag_errors,
        mu_pred_baseline=base_mag,
        mu_pred_optimized=opt_mag,
        obs_positions_arcsec=obs_positions,
        pred_positions_arcsec=opt_pos,
        crit_segments=crit_segments,
        caus_segments=caus_segments,
        suptitle=f"No Subhalo: Baseline vs Optimized",
        output_file=output_plot_file_compare,
        title_left="Position Offset Comparison",
        title_mid="Magnification Comparison",
        title_right="Image Positions & Critical Curves",
        subhalo_positions=None,
        show_2sigma=SHOW_2SIGMA
    )
    print(f"  比较图已保存: {output_plot_file_compare}")

plot_paper_style(
    img_numbers=np.arange(1, n_obs + 1),
    delta_pos_mas=opt_delta,
    sigma_pos_mas=obs_pos_sigma_mas,
    mu_obs=obs_magnifications,
    mu_obs_err=obs_mag_errors,
    mu_pred=opt_mag,
    mu_at_obs_pred=opt_mag.copy(),
    obs_positions_arcsec=obs_positions,
    pred_positions_arcsec=opt_pos,
    crit_segments=crit_segments,
    caus_segments=caus_segments,
    suptitle=f"No Subhalo: Optimized Model",
    output_file=output_plot_file,
    title_left="Position Offset",
    title_mid="Magnification",
    title_right="Image Positions & Critical Curves",
    subhalo_positions=None,
    show_2sigma=SHOW_2SIGMA
)
print(f"  标准图已保存: {output_plot_file}")

# ══════════════════════════════════════════════════════════
# §14  glafic 命令行验证
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("步骤 6: 验证结果（glafic 命令行 vs Python 接口）")
print("=" * 70)

GLAFIC_BIN = find_glafic_bin()
if GLAFIC_BIN:
    print(f"  glafic 路径: {GLAFIC_BIN}")
else:
    print(f"  警告: 未找到 glafic 可执行文件，跳过验证")

verify_input_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_verify_input.dat')
with open(verify_input_file, 'w') as f:
    f.write(f"omega      {omega}\n")
    f.write(f"lambda     {lambda_cosmo}\n")
    f.write(f"weos       {weos}\n")
    f.write(f"hubble     {hubble}\n\n")
    f.write(f"prefix     {OUTPUT_PREFIX}_verify\n\n")
    f.write(f"xmin       {xmin}\n")
    f.write(f"ymin       {ymin}\n")
    f.write(f"xmax       {xmax}\n")
    f.write(f"ymax       {ymax}\n")
    f.write(f"pix_ext    {pix_ext}\n")
    f.write(f"pix_poi    {pix_poi}\n")
    f.write(f"maxlev     {maxlev}\n\n")
    f.write(f"startup    {len(best_lp)} 0 1\n\n")
    for _key, _pv in best_lp.items():
        f.write(f"lens       {_pv[1]:<10} {_pv[2]}    ")
        f.write("    ".join(f"{_pv[i]:.6e}" for i in range(3, len(_pv))) + "\n")
    f.write(f"\npoint      {source_z}    {best_sx:.6e}    {best_sy:.6e}\n\n")
    f.write("end_startup\n\nstart_command\n\nfindimg\n\nquit\n")

if GLAFIC_BIN:
    try:
        result_verify = subprocess.run(
            [GLAFIC_BIN, os.path.basename(verify_input_file)],
            cwd=output_dir, capture_output=True, text=True, timeout=60)
        if result_verify.returncode == 0:
            print(f"  glafic 运行成功")
        else:
            print(f"  glafic 返回非零代码 {result_verify.returncode}")
    except Exception as e:
        print(f"  glafic 运行出错: {e}")

sys.stdout.flush()
print("\n" + "=" * 70, flush=True)
print("Version None 1.0 完成", flush=True)
print("=" * 70, flush=True)

if MCMC_ENABLED and ndim > 0:
    print(f"  MCMC链: {mcmc_chain_file}")
    print(f"  后验统计: {posterior_file}")
    print(f"  Corner图: {corner_file}")
    print(f"  轨迹图: {trace_file}")
print(f"  结果图: {output_plot_file}")
print(f"  参数文件: {best_params_file}")
