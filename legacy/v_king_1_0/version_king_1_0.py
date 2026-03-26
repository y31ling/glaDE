#!/usr/bin/env python3
"""
Version King 1.0  —  King Profile GC 子晕强引力透镜拟合
══════════════════════════════════════════════════════════════════

King (1962) 面密度轮廓用于拟合球状星团（Globular Cluster, GC）子晕：

    Σ(R) ∝ [1/√(1+(R/rc)²) − 1/√(1+(rt/rc)²)]²   R ≤ rt
          = 0                                          R > rt

glafic 参数格式：
    lens king  zl  M[M☉]  x0  y0  e  PA  rc["]  c

    M   [M☉]      总质量（GC 典型范围：10⁴–10⁸ M☉）
    rc  [arcsec]  King 核半径（典型 ~1–30 pc @z=0.5）
    c              集中度 = log₁₀(rt/rc)（典型 0.8–2.2）

算法流程：
    步骤1  计算基准模型（无 GC 子晕）
    步骤2  差分进化（DE）算法搜索最优 King 参数
    步骤3  分析 DE 最佳结果
    步骤4  [可选] MCMC 后验采样（emcee）
    步骤5  生成论文风格图表
    验证   Python 接口 vs glafic 命令行一致性检验

参数说明：
    每个 King GC 子晕有 5 个自由参数：
        x, y         位置 [arcsec]
        log10(M)     log₁₀(质量 / M☉)  — 在对数空间搜索以覆盖大范围
        rc           核半径 [arcsec]
        c            集中度 log₁₀(rt/rc)

物理注意：
    rc–c 内禀简并：仅靠图像位置约束无法独立约束 rc 和 c，
    但总质量 M 和有效 Einstein 半径 θ_E 受良好约束。
    突破简并需要多背景源、精确流量比或微透镜数据。
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
import matplotlib.pyplot as plt
import subprocess

from plot_paper_style import (
    plot_paper_style_king,
    plot_paper_style_king_compare,
    read_critical_curves,
)

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

os.environ['LD_LIBRARY_PATH'] = (
    '/home/luukiaun/glafic251018/gsl-2.8/.libs:'
    '/home/luukiaun/glafic251018/fftw-3.3.10/.libs:'
    '/home/luukiaun/glafic251018/cfitsio-4.6.2/.libs'
)

# ══════════════════════════════════════════════════════════════════
# §1  基准透镜参数加载函数
# ══════════════════════════════════════════════════════════════════

def load_baseline_lens_params(directory):
    """
    从指定目录的 bestfit.dat 文件加载基准透镜参数。

    bestfit.dat 格式（每行以 lens 或 point 开头）：
        lens  <type>  z  p1  p2  p3  p4  p5  p6  p7
        point  z_s  x_s  y_s

    支持主透镜类型：'sie'（SIE）和 'anfw'（轴对称 NFW）。

    返回：
        (lens_params_dict, source_x, source_y, main_lens_key)
    """
    bestfit_path = os.path.join(directory, 'bestfit.dat')
    if not os.path.isfile(bestfit_path):
        raise FileNotFoundError(f"未找到基准参数文件: {bestfit_path}")

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

    if len(lens_lines) < 3:
        raise ValueError(
            f"bestfit.dat 需要至少3行 lens 参数，"
            f"实际找到 {len(lens_lines)} 行: {bestfit_path}")
    if point_params is None:
        raise ValueError(f"bestfit.dat 缺少 point 行: {bestfit_path}")

    params_dict, sers_count, main_lens_key = {}, 0, None
    for parts in lens_lines:
        lens_type = parts[1]
        z         = float(parts[2])
        vals      = [float(v) for v in parts[3:]]
        if lens_type == 'sers':
            sers_count += 1
            params_dict[f'sers{sers_count}'] = (sers_count, 'sers', z, *vals)
        else:
            main_lens_key = lens_type
            params_dict[lens_type] = (sers_count + 1, lens_type, z, *vals)

    if main_lens_key is None:
        raise ValueError(
            f"bestfit.dat 未找到主透镜 (sie/anfw): {bestfit_path}")

    return (params_dict,
            float(point_params[2]),
            float(point_params[3]),
            main_lens_key)


# ══════════════════════════════════════════════════════════════════
# §2  辅助函数
# ══════════════════════════════════════════════════════════════════

def find_glafic_bin(default_path="/home/luukiaun/glafic251018/glafic2/glafic"):
    """智能查找 glafic 可执行文件"""
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


cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def format_mass(mass_solar):
    """格式化质量显示"""
    if mass_solar < 1e6:
        return f"{mass_solar:.2e} M☉"
    elif mass_solar < 1e9:
        return f"{mass_solar/1e6:.2f}×10⁶ M☉"
    elif mass_solar < 1e12:
        return f"{mass_solar/1e9:.2f}×10⁹ M☉"
    else:
        return f"{mass_solar/1e12:.2f}×10¹² M☉"


# ══════════════════════════════════════════════════════════════════
# §3  可配置参数区域
# ══════════════════════════════════════════════════════════════════

print("=" * 70)
print("Version King 1.0: King Profile GC Sub-halos (DE + MCMC)")
print("=" * 70)

# ── 0. 基准透镜参数路径 ─────────────────────────────────────────
# 设置为含 bestfit.dat 的目录；留空则使用下方内置默认参数
BASELINE_LENS_DIR = ""

# ── 1. 约束条件 ─────────────────────────────────────────────────
CONSTRAINT_SIGMA      = 1.0       # 位置约束σ倍数
PENALTY_COEFFICIENT   = 1000      # 违反约束惩罚系数

# ── 2. 启用的子晕（对应图像编号 1–4）──────────────────────────
active_subhalos = [1,3,4]

# ── 3. 精细调试模式 ─────────────────────────────────────────────
fine_tuning = False

# 通用配置（fine_tuning=False 时使用）
SEARCH_RADIUS = 0.1       # [arcsec] 位置搜索半径
LOGM_MIN      = 3.5        # log10(M/M☉) 下界
LOGM_MAX      = 8.0       # log10(M/M☉) 上界
RC_MIN        = 0.00008      # [arcsec] 核半径下界
RC_MAX        = 0.0080      # [arcsec] 核半径上界
C_MIN         = 0.5        # 集中度下界
C_MAX         = 2.5        # 集中度上界

# 精细配置（fine_tuning=True 时使用）
# 单位：search_radius[arcsec], logM_min/max[dex], rc_min/max[arcsec], c_min/max
fine_tuning_configs = {
    1: {
        'search_radius': 0.10,
        'logM_min': 5.5,  'logM_max': 8.0,
        'rc_min':  0.003, 'rc_max':  0.050,
        'c_min':   0.8,   'c_max':   2.2,
    },
    2: {
        'search_radius': 0.40,
        'logM_min': 5.5,  'logM_max': 8.0,
        'rc_min':  0.003, 'rc_max':  0.060,
        'c_min':   0.8,   'c_max':   2.2,
    },
    3: {
        'search_radius': 0.05,
        'logM_min': 5.0,  'logM_max': 8.0,
        'rc_min':  0.002, 'rc_max':  0.050,
        'c_min':   0.8,   'c_max':   2.2,
    },
    4: {
        'search_radius': 0.05,
        'logM_min': 5.0,  'logM_max': 8.0,
        'rc_min':  0.002, 'rc_max':  0.050,
        'c_min':   0.8,   'c_max':   2.2,
    },
}

# ── 4. 损失函数权重 ─────────────────────────────────────────────
LOSS_COEF_A    = 4.0       # 位置 χ² 权重
LOSS_COEF_B    = 1.0       # 放大率 χ² 权重
LOSS_PENALTY_PL = 1000.0   # 位置超出σ的惩罚系数

# ── 4.1 主透镜 / 源位置微调 ──────────────────────────────────
source_modify     = False   # 是否微调源位置
lens_modify       = False   # 是否微调主透镜参数
modify_percentage = 0.01    # 允许变化百分比

# ── 5. 差分进化配置 ─────────────────────────────────────────────
DE_MAXITER  = 800
DE_POPSIZE  = 75
DE_ATOL     = 1e-5
DE_TOL      = 1e-5
DE_SEED     = random.randint(1, 1000000)
DE_POLISH   = True
DE_WORKERS  = -1            # -1 = 全部 CPU；glafic 非线程安全，建议设为 1

# 早停配置
EARLY_STOPPING      = True
EARLY_STOP_PATIENCE = 55

# ── 6. MCMC 配置 ─────────────────────────────────────────────────
MCMC_ENABLED      = False
MCMC_NWALKERS     = 32
MCMC_NSTEPS       = 2000
MCMC_BURNIN       = 300
MCMC_THIN         = 2
MCMC_PERTURBATION = 0.01   # 初始扰动幅度（相对参数范围）
MCMC_PROGRESS     = True
MCMC_WORKERS      = 1      # MCMC 建议串行（glafic 全局状态不是线程安全的）

# ── 7. 绘图配置 ─────────────────────────────────────────────────
SHOW_2SIGMA   = False
OUTPUT_PREFIX = "v_king_1_0"
COMPARE_GRAPH = True
Draw_Graph    = 1           # 0=不绘制迭代图, 1=绘制
draw_interval = 10

PLOT_KING_PROFILES = True   # 是否绘制 King kappa 剖面图
KING_PROFILE_RMAX  = 0.5    # 剖面绘制最大半径 [arcsec]

# ── 固定观测数据（iPTF16geu 四重像）───────────────────────────
obs_positions_mas = np.array([
    [-266.035, +0.427],
    [+118.835, -221.927],
    [+238.324, +227.270],
    [-126.157, +319.719],
])

obs_positions         = np.zeros_like(obs_positions_mas)
obs_positions[:, 0]   = -obs_positions_mas[:, 0] / 1000.0
obs_positions[:, 1]   =  obs_positions_mas[:, 1] / 1000.0

obs_magnifications    = np.array([-35.6, 15.7, -7.5, 9.1])
obs_mag_errors        = np.array([  2.1,  1.3,  1.0, 1.1])
obs_pos_sigma_mas     = np.array([  0.41, 0.86, 2.23, 3.11])

center_offset_x = -0.01535000
center_offset_y = +0.03220000

# ── 宇宙学 & glafic 网格 ────────────────────────────────────────
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

# ── 默认基准透镜参数（SIE + 2×Sersic）────────────────────────
source_x = 2.685497e-03
source_y = 2.443616e-02

lens_params = {
    'sers1': (1, 'sers', 0.2160, 9.896617e+09,
              2.656977e-03, 2.758473e-02,
              2.986760e-01, 1.124730e+02, 3.939718e-01, 1.057760e+00),
    'sers2': (2, 'sers', 0.2160, 2.555580e+10,
              2.656977e-03, 2.758473e-02,
              4.242340e-01, 5.396370e+01, 1.538855e+00, 1.000000e+00),
    'sie':   (3, 'sie',  0.2160, 1.183382e+02,
              2.656977e-03, 2.758473e-02,
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
    print(f"           source_x={source_x:.6e}, source_y={source_y:.6e}")
else:
    print("[基准透镜] 使用内置默认 SIE 参数")

# ══════════════════════════════════════════════════════════════════
# §4  处理 active_subhalos 和配置
# ══════════════════════════════════════════════════════════════════

active_subhalos = sorted(list(set(active_subhalos)))
for idx in active_subhalos:
    if idx not in [1, 2, 3, 4]:
        raise ValueError(f"active_subhalos 包含无效图像索引: {idx}")

n_active_subhalos = len(active_subhalos)
# 每个 King GC 子晕的 5 个自由参数：x, y, log10(M), rc, c
n_params_subhalo = n_active_subhalos * 5

n_params_source = 2 if source_modify else 0
n_params_lens   = 15 if lens_modify else 0
n_params_extra  = n_params_source + n_params_lens
n_params        = n_params_subhalo + n_params_extra

# 构建每个子晕的搜索配置
subhalo_configs = {}
for img_idx in active_subhalos:
    if fine_tuning and img_idx in fine_tuning_configs:
        cfg = fine_tuning_configs[img_idx]
        subhalo_configs[img_idx] = {
            'search_radius': cfg['search_radius'],
            'logM_min': cfg['logM_min'], 'logM_max': cfg['logM_max'],
            'rc_min':   cfg['rc_min'],   'rc_max':   cfg['rc_max'],
            'c_min':    cfg['c_min'],    'c_max':    cfg['c_max'],
        }
    else:
        subhalo_configs[img_idx] = {
            'search_radius': SEARCH_RADIUS,
            'logM_min': LOGM_MIN, 'logM_max': LOGM_MAX,
            'rc_min':   RC_MIN,   'rc_max':   RC_MAX,
            'c_min':    C_MIN,    'c_max':    C_MAX,
        }

if draw_interval < 1:
    draw_interval = 1

# ══════════════════════════════════════════════════════════════════
# §5  创建时间戳输出目录
# ══════════════════════════════════════════════════════════════════

timestamp  = datetime.now().strftime("%y%m%d_%H%M")
output_dir = timestamp
os.makedirs(output_dir, exist_ok=True)
print(f"\n输出目录: {output_dir}")

# 全局迭代历史
iteration_history = []

# ══════════════════════════════════════════════════════════════════
# §6  打印配置摘要
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("配置摘要")
print("=" * 70)

print(f"\n模型类型: King Profile (King 1962, 球状星团)")
print(f"  每个 sub-halo 参数: x, y, log₁₀M, rc, c (5 个)")
print(f"  注意: rc–c 存在内禀简并；总质量 M 受良好约束")

print(f"\n启用的 Sub-halos:")
print(f"  active_subhalos = {active_subhalos}")
print(f"  启用数量: {n_active_subhalos}")
print(f"  参数维度: {n_params}")

print(f"\nMCMC 后验采样: {'启用' if MCMC_ENABLED else '禁用'}")
if MCMC_ENABLED:
    print(f"  Walkers: {MCMC_NWALKERS}  Steps: {MCMC_NSTEPS}  Burn-in: {MCMC_BURNIN}")

print(f"\n精细调试模式: fine_tuning = {fine_tuning}")

# 固定坐标轴范围（供绘图使用）
FIXED_LOGM_RANGE = []
FIXED_RC_RANGE   = []
FIXED_C_RANGE    = []
for img_idx in active_subhalos:
    cfg = subhalo_configs[img_idx]
    FIXED_LOGM_RANGE.extend([cfg['logM_min'], cfg['logM_max']])
    FIXED_RC_RANGE.extend([cfg['rc_min'],     cfg['rc_max']])
    FIXED_C_RANGE.extend([cfg['c_min'],        cfg['c_max']])

# ══════════════════════════════════════════════════════════════════
# §7  损失函数 & 计算模型
# ══════════════════════════════════════════════════════════════════

def machine_learning_loss(pred_pos_matched, pred_mag_matched, delta_pos_mas):
    """加权损失函数（位置 χ² + 放大率 χ² + 位置越界惩罚）"""
    Y_total = 0.0
    for i in range(4):
        chi2_pos_i = (delta_pos_mas[i] / obs_pos_sigma_mas[i]) ** 2
        chi2_mag_i = ((pred_mag_matched[i] - obs_magnifications[i])
                      / obs_mag_errors[i]) ** 2
        P_i = (0.0 if delta_pos_mas[i] <= obs_pos_sigma_mas[i]
               else LOSS_PENALTY_PL * delta_pos_mas[i])
        Y_total += LOSS_COEF_A * chi2_pos_i + LOSS_COEF_B * chi2_mag_i + P_i
    return Y_total


def compute_model(king_params_list, verbose=False,
                  src_x=None, src_y=None, lens_params_dict=None):
    """
    计算多 King GC 子晕模型并返回像位置/放大率/偏差/放大率χ²。

    参数
    ────
    king_params_list : list of (x, y, M, rc, c)
        每个 King GC 子晕的参数；M 以 M☉ 为单位（非 log）

    返回
    ────
    (pred_pos_matched, pred_mag_matched, delta_pos, mag_chi2)
    或 (None, None, None, 1e10) 如果未找到 4 张像
    """
    use_src_x      = src_x          if src_x          is not None else source_x
    use_src_y      = src_y          if src_y          is not None else source_y
    use_lens_params = lens_params_dict if lens_params_dict is not None else lens_params

    glafic.init(omega, lambda_cosmo, weos, hubble,
                f'temp_{OUTPUT_PREFIX}',
                xmin, ymin, xmax, ymax,
                pix_ext, pix_poi, maxlev, verb=0)

    n_subhalos = len(king_params_list)
    glafic.startup_setnum(3 + n_subhalos, 0, 1)

    glafic.set_lens(*use_lens_params['sers1'])
    glafic.set_lens(*use_lens_params['sers2'])
    glafic.set_lens(*use_lens_params[MAIN_LENS_KEY])

    for i, (x_sub, y_sub, M_sub, rc_sub, c_sub) in enumerate(king_params_list):
        # e=0, PA=0 → 球形 King profile（GC 假设各向同性）
        glafic.set_lens(4 + i, 'king', lens_z,
                        M_sub, x_sub, y_sub,
                        0.0, 0.0, rc_sub, c_sub)

    glafic.set_point(1, source_z, use_src_x, use_src_y)
    glafic.model_init(verb=0)

    result = glafic.point_solve(source_z, use_src_x, use_src_y, verb=0)

    n_images = len(result)
    if n_images == 5:
        abs_mags = [abs(img[2]) for img in result]
        drop_idx = int(np.argmin(abs_mags))
        result   = [img for i, img in enumerate(result) if i != drop_idx]
        if verbose:
            print(f"  Info: 5 images, dropped central (idx={drop_idx}, "
                  f"|μ|={abs_mags[drop_idx]:.4f})")
    elif n_images != 4:
        if verbose:
            print(f"  Warning: Found {n_images} images (expected 4)")
        glafic.quit()
        return None, None, None, 1e10

    pred_positions      = np.array([[img[0], img[1]] for img in result])
    pred_magnifications = np.array([img[2]            for img in result])

    pred_positions[:, 0] += center_offset_x
    pred_positions[:, 1] += center_offset_y

    distances            = cdist(obs_positions, pred_positions)
    row_ind, col_ind     = linear_sum_assignment(distances)
    order                = col_ind[np.argsort(row_ind)]
    pred_pos_matched     = pred_positions[order]
    pred_mag_matched     = pred_magnifications[order]

    delta_pos = np.array([
        np.sqrt(((pred_pos_matched[i, 0] - obs_positions[i, 0]) * 1000) ** 2
                + ((pred_pos_matched[i, 1] - obs_positions[i, 1]) * 1000) ** 2)
        for i in range(len(obs_positions))
    ])

    mag_residuals = (pred_mag_matched - obs_magnifications) / obs_mag_errors
    mag_chi2      = float(np.sum(mag_residuals ** 2))

    glafic.quit()
    return pred_pos_matched, pred_mag_matched, delta_pos, mag_chi2


# ══════════════════════════════════════════════════════════════════
# §8  步骤1：计算基准模型
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("步骤1: 计算基准模型（无 King GC 子晕）")
print("=" * 70)

base_pos, base_mag, base_delta_pos, base_mag_chi2 = compute_model([])

print(f"\n基准模型结果:")
print(f"  位置 RMS:    {np.sqrt(np.mean(base_delta_pos**2)):.3f} mas")
print(f"  放大率 χ²:  {base_mag_chi2:.2f}")

# ══════════════════════════════════════════════════════════════════
# §9  步骤2：差分进化优化
# ══════════════════════════════════════════════════════════════════

def objective_function(params):
    """
    DE 目标函数。
    参数布局（每个 King 子晕 5 个）：
        params[i*5+0] = x
        params[i*5+1] = y
        params[i*5+2] = log10(M/M☉)
        params[i*5+3] = rc [arcsec]
        params[i*5+4] = c = log10(rt/rc)
    """
    king_list = []
    for i in range(n_active_subhalos):
        x       = params[i * 5]
        y       = params[i * 5 + 1]
        log10M  = params[i * 5 + 2]
        rc      = params[i * 5 + 3]
        c       = params[i * 5 + 4]

        # 物理约束：rc > 0，c > 0（搜索范围已保证，此处作为防御性检查）
        if rc <= 0.0 or c <= 0.0:
            return 1e15

        king_list.append((x, y, 10.0 ** log10M, rc, c))

    src_x_opt         = None
    src_y_opt         = None
    lens_params_opt   = None

    idx = n_params_subhalo

    if source_modify:
        src_x_opt = params[idx]
        src_y_opt = params[idx + 1]
        idx += 2

    if lens_modify:
        lens_params_opt = {}
        # sers1
        lens_params_opt['sers1'] = (
            1, 'sers', lens_z,
            params[idx], params[idx+1], params[idx+2],
            params[idx+3], params[idx+4],
            lens_params['sers1'][8], lens_params['sers1'][9])
        idx += 5
        # sers2
        lens_params_opt['sers2'] = (
            2, 'sers', lens_z,
            params[idx], params[idx+1], params[idx+2],
            params[idx+3], params[idx+4],
            lens_params['sers2'][8], lens_params['sers2'][9])
        idx += 5
        # 主透镜
        lens_params_opt[MAIN_LENS_KEY] = (
            3, MAIN_LENS_KEY, lens_z,
            params[idx], params[idx+1], params[idx+2],
            params[idx+3], params[idx+4],
            lens_params[MAIN_LENS_KEY][8], lens_params[MAIN_LENS_KEY][9])

    pos, mag, delta_pos, mag_chi2 = compute_model(
        king_list,
        src_x=src_x_opt,
        src_y=src_y_opt,
        lens_params_dict=lens_params_opt,
    )

    if pos is None:
        return 1e15

    return machine_learning_loss(pos, mag, delta_pos)


print("\n" + "=" * 70)
print(f"步骤2: 差分进化搜索 ({n_params}维参数空间)")
print("=" * 70)

# 构建搜索边界
bounds = []
for img_idx in active_subhalos:
    cfg      = subhalo_configs[img_idx]
    x_center = obs_positions[img_idx - 1, 0]
    y_center = obs_positions[img_idx - 1, 1]
    sr       = cfg['search_radius']

    bounds.append((x_center - sr,         x_center + sr))
    bounds.append((y_center - sr,         y_center + sr))
    bounds.append((cfg['logM_min'],        cfg['logM_max']))
    bounds.append((cfg['rc_min'],          cfg['rc_max']))
    bounds.append((cfg['c_min'],           cfg['c_max']))

if source_modify:
    frac = modify_percentage
    bounds.append((source_x * (1 - frac), source_x * (1 + frac)))
    bounds.append((source_y * (1 - frac), source_y * (1 + frac)))

if lens_modify:
    frac = modify_percentage
    for key in ['sers1', 'sers2']:
        for pi in [3, 4, 5, 6, 7]:
            bounds.append((lens_params[key][pi] * (1 - frac),
                           lens_params[key][pi] * (1 + frac)))
    for pi in [3, 4, 5, 6, 7]:
        bounds.append((lens_params[MAIN_LENS_KEY][pi] * (1 - frac),
                       lens_params[MAIN_LENS_KEY][pi] * (1 + frac)))

print(f"\n搜索空间 ({n_params}维):")
for i, img_idx in enumerate(active_subhalos):
    cfg      = subhalo_configs[img_idx]
    x_center = obs_positions[img_idx - 1, 0]
    y_center = obs_positions[img_idx - 1, 1]
    sr       = cfg['search_radius']
    print(f"  King GC Sub-halo at Image {img_idx}:")
    print(f"    x  ∈ [{x_center-sr:.4f}, {x_center+sr:.4f}] arcsec")
    print(f"    y  ∈ [{y_center-sr:.4f}, {y_center+sr:.4f}] arcsec")
    print(f"    log₁₀M ∈ [{cfg['logM_min']:.1f}, {cfg['logM_max']:.1f}]"
          f"  →  M ∈ [{10**cfg['logM_min']:.0e}, {10**cfg['logM_max']:.0e}] M☉")
    print(f"    rc ∈ [{cfg['rc_min']*1000:.1f}, {cfg['rc_max']*1000:.1f}] mas")
    print(f"    c  ∈ [{cfg['c_min']:.1f}, {cfg['c_max']:.1f}]")

print(f"\n开始优化...")

# ── 迭代绘图函数 ──────────────────────────────────────────────
def plot_iteration_population(population, iteration_num, output_dir, bounds):
    """绘制 DE 种群参数分布（每 draw_interval 次保存一张）"""
    if Draw_Graph == 0:
        return
    if iteration_num % draw_interval != 0 and iteration_num != 0:
        return

    # 反归一化
    if population.max() <= 1.0 and population.min() >= 0.0:
        pop = np.zeros_like(population)
        for k in range(population.shape[1]):
            lo, hi = bounds[k]
            pop[:, k] = population[:, k] * (hi - lo) + lo
        population = pop

    n_halos = n_active_subhalos
    logM, rc, c = [], [], []
    for i in range(n_halos):
        logM.append(population[:, i * 5 + 2])
        rc.append(  population[:, i * 5 + 3])
        c.append(   population[:, i * 5 + 4])

    fig = plt.figure(figsize=(5 * n_halos, 6))
    labels = [f'King {active_subhalos[i]}' for i in range(n_halos)]

    for i in range(n_halos):
        # log10(M) 分布
        ax = plt.subplot(3, n_halos, i + 1)
        ax.hist(logM[i], bins=20, alpha=0.6, color='royalblue')
        ax.set_xlabel(f'{labels[i]} log₁₀M', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.3)

        # rc 分布
        ax = plt.subplot(3, n_halos, n_halos + i + 1)
        ax.hist(rc[i] * 1000, bins=20, alpha=0.6, color='darkorange')
        ax.set_xlabel(f'{labels[i]} rc [mas]', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.3)

        # c 分布
        ax = plt.subplot(3, n_halos, 2 * n_halos + i + 1)
        ax.hist(c[i], bins=20, alpha=0.6, color='green')
        ax.set_xlabel(f'{labels[i]} c', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.3)

    plt.suptitle(f'Iteration {iteration_num}: King GC Parameters',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_f = os.path.join(output_dir, f'iteration_{iteration_num:04d}.png')
    plt.savefig(out_f, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"    保存迭代图: iteration_{iteration_num:04d}.png")


# ── 运行 DE ────────────────────────────────────────────────────
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
import scipy

print(f"  Scipy 版本: {scipy.__version__}")

_solver_kwargs = dict(
    maxiter=DE_MAXITER, popsize=DE_POPSIZE,
    atol=DE_ATOL, tol=DE_TOL,
    polish=DE_POLISH, disp=True,
    workers=DE_WORKERS, updating='deferred',
)

try:
    solver = DifferentialEvolutionSolver(
        objective_function, bounds, seed=DE_SEED, **_solver_kwargs)
    print("  使用 seed 参数初始化成功")
except TypeError:
    try:
        solver = DifferentialEvolutionSolver(
            objective_function, bounds, random_state=DE_SEED, **_solver_kwargs)
        print("  使用 random_state 参数初始化成功")
    except TypeError:
        np.random.seed(DE_SEED)
        solver = DifferentialEvolutionSolver(
            objective_function, bounds, **_solver_kwargs)
        print("  使用 numpy.random.seed() 替代")

print(f"\n迭代 0 (初始种群):")
plot_iteration_population(solver.population.copy(), 0, output_dir, bounds)

iteration            = 1
previous_best_energy = np.min(solver.population_energies)
converged_count      = 0

while True:
    try:
        solver.__next__()
    except StopIteration:
        print(f"\n✓ 优化收敛！")
        break

    current_best_energy = np.min(solver.population_energies)
    print(f"\n迭代 {iteration}:")
    print(f"  当前最佳目标值: {current_best_energy:.6f}")

    plot_iteration_population(solver.population.copy(), iteration, output_dir, bounds)

    abs_change = abs(current_best_energy - previous_best_energy)
    if abs_change < DE_ATOL:
        converged_count += 1
        if EARLY_STOPPING and converged_count >= EARLY_STOP_PATIENCE:
            print(f"\n✓ 早停触发！连续 {converged_count} 次满足容差。")
            break
    else:
        converged_count = 0

    previous_best_energy = current_best_energy
    iteration += 1

    if iteration > DE_MAXITER:
        print(f"\n✓ 达到最大迭代次数 {DE_MAXITER}。")
        break

# ── 解析 DE 最佳结果 ──────────────────────────────────────────
de_result  = solver.x
final_fun  = float(np.min(solver.population_energies))

best_params              = []   # list of (x, y, M, rc, c)  — M 为 M☉ 值
best_params_with_img_idx = []   # list of (img_idx, x, y, M, rc, c)

for i, img_idx in enumerate(active_subhalos):
    x       = de_result[i * 5]
    y       = de_result[i * 5 + 1]
    log10M  = de_result[i * 5 + 2]
    rc      = de_result[i * 5 + 3]
    c       = de_result[i * 5 + 4]
    M       = 10.0 ** log10M
    best_params.append((x, y, M, rc, c))
    best_params_with_img_idx.append((img_idx, x, y, M, rc, c))

best_source_x   = source_x
best_source_y   = source_y
best_lens_params = lens_params.copy()

extra_idx = n_params_subhalo

if source_modify:
    best_source_x = de_result[extra_idx]
    best_source_y = de_result[extra_idx + 1]
    extra_idx += 2

    if lens_modify:
        best_lens_params = {}
        for key, lid in [('sers1', 1), ('sers2', 2)]:
            best_lens_params[key] = (
                lid, 'sers', lens_z,
                de_result[extra_idx], de_result[extra_idx+1], de_result[extra_idx+2],
                de_result[extra_idx+3], de_result[extra_idx+4],
                lens_params[key][8], lens_params[key][9])
            extra_idx += 5
    best_lens_params[MAIN_LENS_KEY] = (
        3, MAIN_LENS_KEY, lens_z,
        de_result[extra_idx], de_result[extra_idx+1], de_result[extra_idx+2],
        de_result[extra_idx+3], de_result[extra_idx+4],
        lens_params[MAIN_LENS_KEY][8], lens_params[MAIN_LENS_KEY][9])

# ══════════════════════════════════════════════════════════════════
# §10  步骤3：分析 DE 最佳结果
# ══════════════════════════════════════════════════════════════════

print(f"\n" + "=" * 70)
print("步骤3: 分析 DE 最佳结果")
print("=" * 70)

best_pos, best_mag, best_delta_pos, best_mag_chi2 = compute_model(
    best_params, verbose=True,
    src_x=best_source_x, src_y=best_source_y,
    lens_params_dict=best_lens_params if lens_modify else None,
)

if best_pos is None:
    print("\n✗ 优化失败：最佳参数无法产生 4 张像！")
    import sys; sys.exit(1)

print(f"\nDE 最佳 {n_active_subhalos} 个 King GC 子晕参数:")
for img_idx, x, y, M, rc, c in best_params_with_img_idx:
    rt = rc * (10.0 ** c)
    print(f"  King GC at Image {img_idx}:")
    print(f"    位置:     ({x:.6f}, {y:.6f}) arcsec")
    print(f"    质量:     M  = {format_mass(M)} ({M:.3e} M☉)")
    print(f"    核半径:   rc = {rc*1000:.2f} mas")
    print(f"    集中度:   c  = {c:.3f}  →  rt = {rt*1000:.1f} mas")
    print(f"    注意: rc 和 c 单独受内禀简并约束（仅 M 和 θ_E 受良好约束）")

improvement = (base_mag_chi2 - best_mag_chi2) / base_mag_chi2 * 100
print(f"\n改善效果:")
print(f"  基准 χ²: {base_mag_chi2:.2f}")
print(f"  最佳 χ²: {best_mag_chi2:.2f}")
print(f"  改善:   {improvement:.1f}%")

max_pos_deviation_mas = CONSTRAINT_SIGMA * obs_pos_sigma_mas
constraint_satisfied  = all(
    best_delta_pos[i] <= max_pos_deviation_mas[i] for i in range(4)
)

# ══════════════════════════════════════════════════════════════════
# §11  步骤4：MCMC 后验采样（可选）
# ══════════════════════════════════════════════════════════════════

if MCMC_ENABLED:
    print("\n" + "=" * 70)
    print("步骤4: MCMC 后验采样（基于 DE 最优解）")
    print("=" * 70)

    try:
        import emcee
        import corner
        from tqdm import tqdm
        print("  ✓ emcee, corner, tqdm 已导入")
    except ImportError as e:
        print(f"  ✗ 缺少依赖库: {e}")
        print("    请运行: pip install emcee corner tqdm")
        MCMC_ENABLED = False

if MCMC_ENABLED:

    def log_probability(params):
        """
        对数概率 = log(prior) + log(likelihood)
        log(likelihood) = -loss/2
        """
        # 边界检查（先验均匀分布）
        for i, (lo, hi) in enumerate(bounds):
            if not (lo <= params[i] <= hi):
                return -np.inf
        # 物理约束
        for i in range(n_active_subhalos):
            if params[i*5+3] <= 0.0 or params[i*5+4] <= 0.0:
                return -np.inf

        loss = objective_function(params)
        if loss >= 1e10:
            return -np.inf
        return -0.5 * loss

    ndim        = n_params
    best_result = de_result

    if MCMC_NWALKERS < 2 * ndim:
        MCMC_NWALKERS = 2 * ndim + 2
        print(f"  [调整] Walkers → {MCMC_NWALKERS}")

    print(f"\n初始化 MCMC: ndim={ndim}  walkers={MCMC_NWALKERS}  steps={MCMC_NSTEPS}")

    initial_positions = []
    for _ in range(MCMC_NWALKERS):
        pert    = np.array([
            np.random.normal(0, MCMC_PERTURBATION * (bounds[k][1] - bounds[k][0]))
            for k in range(ndim)
        ])
        new_pos = np.clip(best_result + pert,
                          [b[0] for b in bounds],
                          [b[1] for b in bounds])
        initial_positions.append(new_pos)
    initial_positions = np.array(initial_positions)

    if MCMC_WORKERS > 1:
        from multiprocessing import Pool
        print(f"  ⚠ 多进程 ({MCMC_WORKERS} 核)：若出错请设置 MCMC_WORKERS=1")
        with Pool(MCMC_WORKERS) as pool:
            sampler = emcee.EnsembleSampler(MCMC_NWALKERS, ndim,
                                            log_probability, pool=pool)
            if MCMC_PROGRESS:
                for _ in tqdm(sampler.sample(initial_positions,
                                             iterations=MCMC_NSTEPS),
                              total=MCMC_NSTEPS, desc="MCMC"):
                    pass
            else:
                sampler.run_mcmc(initial_positions, MCMC_NSTEPS, progress=False)
            samples = sampler.get_chain(discard=MCMC_BURNIN,
                                        thin=MCMC_THIN, flat=True)
            chain   = sampler.get_chain()
    else:
        sampler = emcee.EnsembleSampler(MCMC_NWALKERS, ndim, log_probability)
        if MCMC_PROGRESS:
            for _ in tqdm(sampler.sample(initial_positions,
                                         iterations=MCMC_NSTEPS),
                          total=MCMC_NSTEPS, desc="MCMC"):
                pass
        else:
            sampler.run_mcmc(initial_positions, MCMC_NSTEPS, progress=False)
        samples = sampler.get_chain(discard=MCMC_BURNIN,
                                    thin=MCMC_THIN, flat=True)
        chain   = sampler.get_chain()

    print(f"\n采样完成: 有效样本 {len(samples)}")

    # 保存链
    mcmc_chain_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_mcmc_chain.dat')
    param_names = []
    for img_idx in active_subhalos:
        param_names.extend([f'x_{img_idx}', f'y_{img_idx}',
                            f'logM_{img_idx}', f'rc_{img_idx}', f'c_{img_idx}'])
    if source_modify:
        param_names.extend(['src_x', 'src_y'])
    if lens_modify:
        param_names.extend([
            'sers1_m', 'sers1_x', 'sers1_y', 'sers1_re', 'sers1_pa',
            'sers2_m', 'sers2_x', 'sers2_y', 'sers2_re', 'sers2_pa',
            'ml_p1', 'ml_x', 'ml_y', 'ml_e', 'ml_pa',
        ])
    np.savetxt(mcmc_chain_file, samples, header=' '.join(param_names))
    print(f"  ✓ MCMC 链: {mcmc_chain_file}")

    # 参数统计
    print(f"\n参数后验分布 (median +upper_1σ -lower_1σ):")
    posterior_stats = {}
    for i, name in enumerate(param_names):
        median = np.median(samples[:, i])
        lower  = np.percentile(samples[:, i], 16)
        upper  = np.percentile(samples[:, i], 84)
        posterior_stats[name] = dict(median=median, lower=lower, upper=upper,
                                     ep=upper-median, em=median-lower)
        if name.startswith('x_') or name.startswith('y_'):
            print(f"  {name}: {median:.6f} +{(upper-median)*1000:.3f} "
                  f"-{(median-lower)*1000:.3f} mas")
        elif name.startswith('logM_'):
            print(f"  {name}: {median:.4f} +{upper-median:.4f} "
                  f"-{median-lower:.4f} dex  "
                  f"→ M = {10**median:.3e} M☉")
        elif name.startswith('rc_'):
            print(f"  {name}: {median*1000:.3f} +{(upper-median)*1000:.3f} "
                  f"-{(median-lower)*1000:.3f} mas")
        elif name.startswith('c_'):
            print(f"  {name}: {median:.4f} +{upper-median:.4f} "
                  f"-{median-lower:.4f}")
        else:
            print(f"  {name}: {median:.6e} +{upper-median:.3e} "
                  f"-{median-lower:.3e}")

    # 质量后验（从 log10M 样本导出 M 样本）
    print(f"\n质量后验分布（M [M☉]）:")
    mass_posterior_stats = {}
    for i, img_idx in enumerate(active_subhalos):
        logM_samp = samples[:, i * 5 + 2]
        M_samp    = 10.0 ** logM_samp
        med       = np.median(M_samp)
        lo16      = np.percentile(M_samp, 16)
        hi84      = np.percentile(M_samp, 84)
        mass_posterior_stats[f'mass_{img_idx}'] = dict(
            median=med, lower=lo16, upper=hi84,
            ep=hi84-med, em=med-lo16, samples=M_samp)
        print(f"  mass_{img_idx}: {med:.3e} +{hi84-med:.3e} -{med-lo16:.3e} M☉")

    # Corner plot
    corner_labels = []
    for img_idx in active_subhalos:
        corner_labels.extend([
            f'$x_{img_idx}$', f'$y_{img_idx}$',
            f'$\\log_{{10}}M_{img_idx}$',
            f'$r_{{c,{img_idx}}}$',
            f'$c_{img_idx}$',
        ])

    fig_corner = corner.corner(
        samples[:, :n_params_subhalo],
        labels=corner_labels[:n_params_subhalo],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_fmt='.4f',
        truths=best_result[:n_params_subhalo],
        truth_color='red',
    )
    corner_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_corner.png')
    fig_corner.savefig(corner_file, dpi=150, bbox_inches='tight')
    plt.close(fig_corner)
    print(f"  ✓ Corner plot: {corner_file}")

    # 轨迹图
    fig_trace, axes_trace = plt.subplots(
        n_params_subhalo, figsize=(10, 2 * n_params_subhalo), sharex=True)
    if n_params_subhalo == 1:
        axes_trace = [axes_trace]
    for k in range(n_params_subhalo):
        axes_trace[k].plot(chain[:, :, k], alpha=0.3)
        axes_trace[k].axvline(MCMC_BURNIN, color='red', ls='--')
        axes_trace[k].set_ylabel(corner_labels[k])
    axes_trace[-1].set_xlabel("Step")
    trace_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_trace.png')
    fig_trace.savefig(trace_file, dpi=150, bbox_inches='tight')
    plt.close(fig_trace)
    print(f"  ✓ 轨迹图: {trace_file}")

    # 保存完整后验统计
    posterior_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_posterior.txt')
    with open(posterior_file, 'w') as f:
        f.write("# ============================================================\n")
        f.write("# MCMC Posterior Distribution — Version King 1.0\n")
        f.write("# ============================================================\n\n")
        f.write(f"# Walkers: {MCMC_NWALKERS}  Steps: {MCMC_NSTEPS}  "
                f"Burn-in: {MCMC_BURNIN}  Thin: {MCMC_THIN}\n")
        f.write(f"# Effective samples: {len(samples)}\n\n")
        f.write("# parameter  median  16%  84%  ep  em\n")
        for name in param_names:
            st = posterior_stats[name]
            f.write(f"{name}  {st['median']:.10e}  {st['lower']:.10e}  "
                    f"{st['upper']:.10e}  {st['ep']:.10e}  {st['em']:.10e}\n")
        f.write("\n# King GC mass posterior [M_sun]\n")
        for img_idx in active_subhalos:
            st = mass_posterior_stats[f'mass_{img_idx}']
            f.write(f"mass_{img_idx}  {st['median']:.10e}  "
                    f"{st['lower']:.10e}  {st['upper']:.10e}  "
                    f"{st['ep']:.10e}  {st['em']:.10e}\n")
        f.write("\n# Summary:\n")
        for img_idx in active_subhalos:
            st_x    = posterior_stats[f'x_{img_idx}']
            st_y    = posterior_stats[f'y_{img_idx}']
            st_logM = posterior_stats[f'logM_{img_idx}']
            st_rc   = posterior_stats[f'rc_{img_idx}']
            st_c    = posterior_stats[f'c_{img_idx}']
            st_m    = mass_posterior_stats[f'mass_{img_idx}']
            f.write(f"# Image {img_idx}:\n")
            f.write(f"#   x  = {st_x['median']:.6f} ±{st_x['ep']*1000:.3f} mas\n")
            f.write(f"#   y  = {st_y['median']:.6f} ±{st_y['ep']*1000:.3f} mas\n")
            f.write(f"#   M  = {st_m['median']:.3e} +{st_m['ep']:.3e} -{st_m['em']:.3e} M☉\n")
            f.write(f"#   rc = {st_rc['median']*1000:.3f} ±{st_rc['ep']*1000:.3f} mas\n")
            f.write(f"#   c  = {st_c['median']:.4f} ±{st_c['ep']:.4f}\n\n")
    print(f"  ✓ 后验统计: {posterior_file}")

# ══════════════════════════════════════════════════════════════════
# §12  步骤5：生成最终图表
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("步骤5: 生成最终图表")
print("=" * 70)

# 生成最优模型的临界曲线
glafic.init(omega, lambda_cosmo, weos, hubble,
            f'temp_{OUTPUT_PREFIX}_best',
            xmin, ymin, xmax, ymax,
            pix_ext, pix_poi, maxlev, verb=0)

glafic.startup_setnum(3 + n_active_subhalos, 0, 1)
glafic.set_lens(*best_lens_params['sers1'])
glafic.set_lens(*best_lens_params['sers2'])
glafic.set_lens(*best_lens_params[MAIN_LENS_KEY])
for i, (x_sub, y_sub, M_sub, rc_sub, c_sub) in enumerate(best_params):
    glafic.set_lens(4 + i, 'king', lens_z,
                    M_sub, x_sub, y_sub, 0.0, 0.0, rc_sub, c_sub)
glafic.set_point(1, source_z, best_source_x, best_source_y)
glafic.model_init(verb=0)
glafic.writecrit(source_z)
glafic.quit()

crit_file                   = f'temp_{OUTPUT_PREFIX}_best_crit.dat'
crit_segments, caus_segments = read_critical_curves(crit_file)

output_plot_file = os.path.join(output_dir, f'result_{OUTPUT_PREFIX}.png')

# 构建绘图用的 King 参数列表 (x, y, M, rc, c)
king_plot_params = [(x, y, M, rc, c)
                    for (_, x, y, M, rc, c) in best_params_with_img_idx]

if COMPARE_GRAPH and n_active_subhalos > 0:
    compare_file = os.path.join(output_dir,
                                f'result_{OUTPUT_PREFIX}_compare.png')
    plot_paper_style_king_compare(
        img_numbers              = np.array([1, 2, 3, 4]),
        delta_pos_mas_baseline   = base_delta_pos,
        delta_pos_mas_optimized  = best_delta_pos,
        sigma_pos_mas            = obs_pos_sigma_mas,
        mu_obs                   = obs_magnifications,
        mu_obs_err               = obs_mag_errors,
        mu_pred_baseline         = base_mag,
        mu_pred_optimized        = best_mag,
        obs_positions_arcsec     = obs_positions,
        pred_positions_arcsec    = best_pos,
        crit_segments            = crit_segments,
        caus_segments            = caus_segments,
        king_params              = king_plot_params,
        suptitle                 = (f"iPTF16geu: Baseline vs "
                                    f"{n_active_subhalos} King GC Sub-halos"),
        output_file              = compare_file,
        title_left               = "Position Offset Comparison",
        title_mid                = "Magnification Comparison",
        title_right              = "Image Positions & Critical Curves",
        show_2sigma              = SHOW_2SIGMA,
    )
    print(f"  比较图: {compare_file}")

plot_paper_style_king(
    img_numbers           = np.array([1, 2, 3, 4]),
    delta_pos_mas         = best_delta_pos,
    sigma_pos_mas         = obs_pos_sigma_mas,
    mu_obs                = obs_magnifications,
    mu_obs_err            = obs_mag_errors,
    mu_pred               = best_mag,
    mu_at_obs_pred        = best_mag.copy(),
    obs_positions_arcsec  = obs_positions,
    pred_positions_arcsec = best_pos,
    crit_segments         = crit_segments,
    caus_segments         = caus_segments,
    king_params           = king_plot_params,
    suptitle              = (f"iPTF16geu: {n_active_subhalos} King GC Sub-halos "
                             f"(DE{'+ MCMC' if MCMC_ENABLED else ''})"),
    output_file           = output_plot_file,
    title_left            = "Position Offset",
    title_mid             = "Magnification",
    title_right           = "Image Positions & Critical Curves",
    show_2sigma           = SHOW_2SIGMA,
)

# 可选：绘制 King profile kappa 剖面图
if PLOT_KING_PROFILES and n_active_subhalos > 0:
    from plot_paper_style import plot_king_profiles
    profile_file = os.path.join(output_dir,
                                f'king_profiles_{OUTPUT_PREFIX}.png')
    kp_list = [(M, rc, c) for (_, _, _, M, rc, c) in best_params_with_img_idx]
    kp_labels = [
        f"GC at Im{img_idx}: {format_mass(M)}\n"
        f"rc={rc*1000:.1f}mas c={c:.2f}"
        for (img_idx, _, _, M, rc, c) in best_params_with_img_idx
    ]
    plot_king_profiles(
        king_params_list = kp_list,
        r_min            = 1e-3,
        r_max            = KING_PROFILE_RMAX,
        labels           = kp_labels,
        title            = f"King GC Profile κ(r) — {n_active_subhalos} Sub-halos",
        output_file      = profile_file,
    )
    print(f"  King profiles 图: {profile_file}")

# ══════════════════════════════════════════════════════════════════
# §13  保存 DE 最佳参数文件
# ══════════════════════════════════════════════════════════════════

params_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_best_params.txt')
with open(params_file, 'w') as f:
    f.write(f"# Version King 1.0: King GC Sub-halos (DE + MCMC)\n")
    f.write(f"# active_subhalos = {active_subhalos}\n")
    f.write(f"# fine_tuning     = {fine_tuning}\n")
    f.write(f"# MCMC_ENABLED    = {MCMC_ENABLED}\n\n")
    f.write(f"# z_lens = {lens_z}   z_source = {source_z}\n\n")
    f.write("# King GC Sub-halo Parameters\n")
    total_mass = 0.0
    for img_idx, x, y, M, rc, c in best_params_with_img_idx:
        rt         = rc * (10.0 ** c)
        total_mass += M
        f.write(f"# Image {img_idx}\n")
        f.write(f"x_king{img_idx}   = {x:.10e}  # arcsec\n")
        f.write(f"y_king{img_idx}   = {y:.10e}  # arcsec\n")
        f.write(f"M_king{img_idx}   = {M:.10e}  # M_sun = {format_mass(M)}\n")
        f.write(f"rc_king{img_idx}  = {rc:.10e}  # arcsec = {rc*1000:.4f} mas\n")
        f.write(f"c_king{img_idx}   = {c:.10e}  # log10(rt/rc)\n")
        f.write(f"rt_king{img_idx}  = {rt:.10e}  # arcsec = {rt*1000:.1f} mas\n\n")
    f.write(f"# Total sub-halo mass\n")
    f.write(f"total_mass = {total_mass:.10e}  # M_sun = {format_mass(total_mass)}\n\n")
    f.write(f"# Performance\n")
    f.write(f"chi2_base        = {base_mag_chi2:.4f}\n")
    f.write(f"chi2_best        = {best_mag_chi2:.4f}\n")
    f.write(f"improvement      = {improvement:.2f}%\n")
    f.write(f"constraint_ok    = {constraint_satisfied}\n")

print(f"  DE 参数文件: {params_file}")

# ══════════════════════════════════════════════════════════════════
# §14  验证：Python 接口 vs glafic 命令行
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("验证步骤: Python 接口 vs glafic 命令行")
print("=" * 70)

GLAFIC_BIN = find_glafic_bin()
if GLAFIC_BIN is None:
    print("  警告: 找不到 glafic 可执行文件，跳过验证")
else:
    verify_input_file = os.path.join(output_dir,
                                     f'{OUTPUT_PREFIX}_verify_input.dat')
    verify_prefix     = f"{OUTPUT_PREFIX}_verify"

    with open(verify_input_file, 'w') as f:
        f.write("# Version King 1.0 验证文件\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# active_subhalos: {active_subhalos}\n\n")

        f.write(f"omega     {omega}\n")
        f.write(f"lambda    {lambda_cosmo}\n")
        f.write(f"weos      {weos}\n")
        f.write(f"hubble    {hubble}\n")
        f.write(f"prefix    {verify_prefix}\n")
        f.write(f"xmin      {xmin}\n")
        f.write(f"ymin      {ymin}\n")
        f.write(f"xmax      {xmax}\n")
        f.write(f"ymax      {ymax}\n")
        f.write(f"pix_ext   {pix_ext}\n")
        f.write(f"pix_poi   {pix_poi}\n")
        f.write(f"maxlev    {maxlev}\n\n")

        n_lenses = 3 + n_active_subhalos
        f.write(f"startup   {n_lenses} 0 1\n")

        for key in ['sers1', 'sers2']:
            p = best_lens_params[key]
            f.write(f"lens   sers  {p[2]}  "
                    f"{p[3]:.6e}  {p[4]:.6e}  {p[5]:.6e}  "
                    f"{p[6]:.6e}  {p[7]:.6e}  {p[8]:.6e}  {p[9]:.6e}\n")

        p = best_lens_params[MAIN_LENS_KEY]
        f.write(f"lens   {MAIN_LENS_KEY}  {p[2]}  "
                f"{p[3]:.6e}  {p[4]:.6e}  {p[5]:.6e}  "
                f"{p[6]:.6e}  {p[7]:.6e}  {p[8]:.6e}  {p[9]:.6e}\n")

        for x_s, y_s, M_s, rc_s, c_s in best_params:
            f.write(f"lens   king  {lens_z:.4f}  "
                    f"{M_s:.10e}  {x_s:.10e}  {y_s:.10e}  "
                    f"0.0  0.0  {rc_s:.10e}  {c_s:.10e}\n")

        f.write(f"point  {source_z}  "
                f"{best_source_x:.10e}  {best_source_y:.10e}\n")
        f.write("end_startup\n\nstart_command\nfindimg\nquit\n")

    print(f"  输入文件: {verify_input_file}")

    try:
        result_cmd = subprocess.run(
            [GLAFIC_BIN, os.path.basename(verify_input_file)],
            cwd=output_dir,
            capture_output=True, text=True, timeout=60,
        )
        if result_cmd.returncode == 0:
            print("  glafic 运行成功")
        else:
            print(f"  警告: glafic 返回代码 {result_cmd.returncode}")
    except subprocess.TimeoutExpired:
        print("  警告: glafic 超时（>60s）")
    except (FileNotFoundError, Exception) as e:
        print(f"  警告: {e}")

    verify_point = os.path.join(output_dir, f'{verify_prefix}_point.dat')
    if os.path.exists(verify_point):
        try:
            data = np.loadtxt(verify_point)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            n_imgs_gl = int(data[0, 0])
            print(f"  glafic 找到 {n_imgs_gl} 张像")

            if n_imgs_gl in (4, 5):
                img_data = data[1:n_imgs_gl + 1, :]
                if n_imgs_gl == 5:
                    drop = int(np.argmin(np.abs(img_data[:, 2])))
                    img_data = np.delete(img_data, drop, axis=0)

                gl_pos  = img_data[:, 0:2].copy()
                gl_pos[:, 0] += center_offset_x
                gl_pos[:, 1] += center_offset_y
                gl_mag        = np.abs(img_data[:, 2])

                D2             = cdist(obs_positions, gl_pos)
                ri2, ci2       = linear_sum_assignment(D2)
                gl_pos_m       = gl_pos[ci2[np.argsort(ri2)]]
                gl_mag_m       = gl_mag[ci2[np.argsort(ri2)]]

                print(f"\n  {'Img':<5} {'Py x [mas]':>13} {'GL x [mas]':>13} "
                      f"{'|Δx| [mas]':>12} "
                      f"{'Py y [mas]':>13} {'GL y [mas]':>13} {'|Δy| [mas]':>12}")
                print(f"  {'-'*95}")
                max_pos_diff = 0.0
                for k in range(4):
                    py_x = best_pos[k, 0] * 1000
                    py_y = best_pos[k, 1] * 1000
                    gl_x = gl_pos_m[k, 0] * 1000
                    gl_y = gl_pos_m[k, 1] * 1000
                    dx   = abs(py_x - gl_x)
                    dy   = abs(py_y - gl_y)
                    max_pos_diff = max(max_pos_diff, dx, dy)
                    print(f"  {k+1:<5} {py_x:>13.3f} {gl_x:>13.3f} {dx:>12.3f} "
                          f"{py_y:>13.3f} {gl_y:>13.3f} {dy:>12.3f}")

                max_mag_pct = 0.0
                print(f"\n  {'Img':<5} {'Py |μ|':>13} {'GL |μ|':>13} "
                      f"{'Δ|μ|':>10} {'Δ [%]':>10}")
                print(f"  {'-'*60}")
                for k in range(4):
                    pm  = abs(best_mag[k])
                    gm  = gl_mag_m[k]
                    dm  = abs(pm - gm)
                    dmp = dm / pm * 100 if pm != 0 else 0
                    max_mag_pct = max(max_mag_pct, dmp)
                    print(f"  {k+1:<5} {pm:>13.3f} {gm:>13.3f} "
                          f"{dm:>10.3f} {dmp:>9.3f}%")

                print(f"\n  最大位置差: {max_pos_diff:.6f} mas")
                print(f"  最大放大率差: {max_mag_pct:.6f}%")
                if max_pos_diff < 0.01 and max_mag_pct < 0.1:
                    print("  ✓ 一致性验证通过！")
                elif max_pos_diff < 1.0 and max_mag_pct < 1.0:
                    print("  ✓ 一致性良好（小数值差异）")
                else:
                    print("  ⚠ 较大差异，请检查参数")

                # 保存验证报告
                verify_report = os.path.join(output_dir,
                                             f'{OUTPUT_PREFIX}_verify_report.txt')
                with open(verify_report, 'w') as f:
                    f.write("=" * 70 + "\n")
                    f.write("Python vs glafic 命令行验证报告 — Version King 1.0\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"active_subhalos: {active_subhalos}\n\n")
                    f.write("King GC 参数:\n")
                    for img_idx, x, y, M, rc, c in best_params_with_img_idx:
                        f.write(f"  Image {img_idx}: x={x:.10e} y={y:.10e} "
                                f"M={M:.10e} rc={rc:.10e} c={c:.10e}\n")
                    f.write(f"\n最大位置差:   {max_pos_diff:.6f} mas\n")
                    f.write(f"最大放大率差: {max_mag_pct:.6f}%\n")
        except Exception as e:
            print(f"  读取验证输出出错: {e}")
    else:
        print(f"  警告: 未找到 glafic 输出文件 {verify_point}")

# ══════════════════════════════════════════════════════════════════
# §15  最终摘要
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Version King 1.0 完成")
print("=" * 70)
print(f"\n  结果图:          {output_plot_file}")
print(f"  DE 参数文件:     {params_file}")
if MCMC_ENABLED:
    print(f"  MCMC 链:         {mcmc_chain_file}")
    print(f"  后验统计:        {posterior_file}")
    print(f"  Corner 图:       {corner_file}")
    print(f"  轨迹图:          {trace_file}")
if PLOT_KING_PROFILES:
    print(f"  King profiles:   {profile_file}")
print(f"\n  输出目录: {output_dir}/")
