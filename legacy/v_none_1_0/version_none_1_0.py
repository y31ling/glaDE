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
sys.path.insert(0, '/home/luukiaun/glafic251018/glafic2/python')
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

os.environ['LD_LIBRARY_PATH'] = (
    '/home/luukiaun/glafic251018/gsl-2.8/.libs:'
    '/home/luukiaun/glafic251018/fftw-3.3.10/.libs:'
    '/home/luukiaun/glafic251018/cfitsio-4.6.2/.libs'
)

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


def find_glafic_bin(default_path="/home/luukiaun/glafic251018/glafic2/glafic"):
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
BASELINE_LENS_DIR = '/home/luukiaun/glafic251018/work/glade/legacy/v_none_1_0/bestfit_default'

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

_iter_count = [0]
_best_loss  = [float('inf')]
_no_improve = [0]


_best_loss  = [float('inf')]
_no_improve = [0]


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

    # 仅单进程模式下全局计数有效；workers=-1 时更新发生在子进程，主进程看不到
    if loss < _best_loss[0]:
        _best_loss[0] = loss
        _no_improve[0] = 0
    else:
        _no_improve[0] += 1
    return loss


# ══════════════════════════════════════════════════════════
# §8  DE 优化
# ══════════════════════════════════════════════════════════

_iter_count = [0]

_parallel_mode = (DE_WORKERS != 1)   # workers=-1 或 >1 时为并行模式

def de_callback(xk, convergence):
    """迭代回调。
    - 单进程(workers=1): 直接读取全局 _best_loss/_no_improve。
    - 多进程(workers!=1): 全局变量在子进程更新，主进程不可见；
      re-evaluate 一次以获取当前最优 loss，早停依赖 convergence 值。
    """
    _iter_count[0] += 1
    gen = _iter_count[0]

    if _parallel_mode:
        # 并行模式：用 xk（当前最优参数）重新计算一次 loss，用于显示
        cur_loss = objective(xk)
        if cur_loss < _best_loss[0]:
            _best_loss[0] = cur_loss
            _no_improve[0] = 0
        else:
            _no_improve[0] += 1

    if gen % draw_interval == 0:
        print(f"  迭代 {gen:4d}: best={_best_loss[0]:.6f}, "
              f"no_improve={_no_improve[0]}", flush=True)

    if EARLY_STOPPING and _no_improve[0] >= EARLY_STOP_PATIENCE:
        print(f"  [早停] 连续 {EARLY_STOP_PATIENCE} 次无改善，停止优化")
        return True
    return False


print("\n" + "=" * 70)
if ndim == 0:
    print("步骤 2: 无优化参数（source_modify=False, lens_modify=False）")
    print("        直接输出基准模型结果。")
    print("=" * 70)
    best_x    = np.array([])
    best_loss = objective(best_x) if ndim == 0 else float('inf')
    # Recompute baseline
    best_sx, best_sy = source_x_ref, source_y_ref
    best_lp  = {k: tuple(v) for k, v in lens_params_ref.items()}
else:
    print(f"步骤 2: 差分进化优化 (维度={ndim})")
    print("=" * 70)

    try:
        import scipy
        scipy_ver = tuple(int(x) for x in scipy.__version__.split('.')[:2])
        de_kwargs = dict(
            func=objective, bounds=bounds,
            maxiter=DE_MAXITER, popsize=DE_POPSIZE,
            atol=DE_ATOL, tol=DE_TOL,
            polish=DE_POLISH, workers=DE_WORKERS,
            callback=de_callback,
        )
        if scipy_ver >= (1, 7):
            import numpy as _np
            _np.random.seed(DE_SEED)
        else:
            de_kwargs['seed'] = DE_SEED

        sys.stdout.flush()   # fork 前清空缓冲，防止子进程继承并重复输出
        result = differential_evolution(**de_kwargs)
    except Exception as e:
        print(f"  [警告] DE 运行异常: {e}")
        result = type('R', (), {'x': np.array([b[0] for b in bounds]), 'fun': 1e10})()

    best_x    = result.x
    best_loss = result.fun

    # Reconstruct best params
    best_sx, best_sy = source_x_ref, source_y_ref
    best_lp_list = {k: list(v) for k, v in lens_params_ref.items()}
    for i, (ptype, key, pi) in enumerate(pmap):
        if ptype == 'src_x': best_sx = best_x[i]
        elif ptype == 'src_y': best_sy = best_x[i]
        else: best_lp_list[key][3 + pi] = best_x[i]
    best_lp = {k: tuple(v) for k, v in best_lp_list.items()}

    print(f"\n  最优 loss = {best_loss:.6f}")

# ══════════════════════════════════════════════════════════
# §9  最优结果评估
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

# ══════════════════════════════════════════════════════════
# §10  保存结果
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
# §11  结果对比图
# ══════════════════════════════════════════════════════════

def make_result_figure(positions, magnifications, delta_mas, title, filename):
    """Plot image positions and magnification comparison (English labels)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(obs_positions[:, 0] * 1000, obs_positions[:, 1] * 1000,
               s=120, c='black', marker='o', zorder=5, label='Observed')
    ax.scatter(positions[:, 0] * 1000, positions[:, 1] * 1000,
               s=80, c='red', marker='x', zorder=4, label='Predicted')
    for i in range(n_obs):
        ax.plot([obs_positions[i, 0] * 1000, positions[i, 0] * 1000],
                [obs_positions[i, 1] * 1000, positions[i, 1] * 1000],
                'r--', alpha=0.5, linewidth=0.8)
    ax.set_xlabel('x (mas)')
    ax.set_ylabel('y (mas)')
    ax.set_title(f'{title} — Image Positions')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x = np.arange(n_obs)
    ax.bar(x - 0.2, obs_magnifications, 0.4, label='Obs. mu', alpha=0.8)
    ax.bar(x + 0.2, magnifications,     0.4, label='Model mu', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Img {i+1}' for i in range(n_obs)])
    ax.set_ylabel('Magnification mu')
    ax.set_title(f'{title} — Magnification')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure: {out}")


if Draw_Graph:
    make_result_figure(base_pos, base_mag, base_delta,
                       'Baseline', f'{OUTPUT_PREFIX}_baseline.png')
    if COMPARE_GRAPH and ndim > 0:
        make_result_figure(opt_pos, opt_mag, opt_delta,
                           'Optimized', f'{OUTPUT_PREFIX}_optimized.png')

# ══════════════════════════════════════════════════════════
# §12  [可选] MCMC 后验采样
# ══════════════════════════════════════════════════════════

if MCMC_ENABLED and ndim > 0:
    print("\n" + "=" * 70)
    print("步骤 4: MCMC 后验采样")
    print("=" * 70)
    try:
        import emcee

        def log_prob(x):
            for i, (lo, hi) in enumerate(bounds):
                if not (lo <= x[i] <= hi):
                    return -np.inf
            loss = objective(x)
            return -0.5 * loss

        nw = max(MCMC_NWALKERS, 2 * ndim + 2)
        rng = np.random.default_rng(DE_SEED)
        p0  = best_x + rng.normal(0, MCMC_PERTURBATION, (nw, ndim))
        # Clip to bounds
        for i, (lo, hi) in enumerate(bounds):
            p0[:, i] = np.clip(p0[:, i], lo, hi)

        sampler = emcee.EnsembleSampler(nw, ndim, log_prob)
        sampler.run_mcmc(p0, MCMC_NSTEPS, progress=MCMC_PROGRESS)

        flat = sampler.get_chain(discard=MCMC_BURNIN, thin=MCMC_THIN, flat=True)
        print(f"\n  有效样本: {len(flat)}")
        for i, (ptype, key, pi) in enumerate(pmap):
            name = f'src_{ptype[-1]}' if ptype.startswith('src') else f'{key}_p{pi+1}'
            med, std = np.median(flat[:, i]), np.std(flat[:, i])
            print(f"  {name}: {med:.4g} ± {std:.4g}")

        np.save(os.path.join(output_dir, f"{OUTPUT_PREFIX}_mcmc_chain.npy"), flat)
    except ImportError:
        print("  [跳过] emcee 未安装")

sys.stdout.flush()
print("\n" + "=" * 70, flush=True)
print("Version None 1.0 完成", flush=True)
print("=" * 70, flush=True)
