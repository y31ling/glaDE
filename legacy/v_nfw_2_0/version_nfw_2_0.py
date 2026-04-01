#!/usr/bin/env python3
"""
Version NFW 2.0: Optimized NFW Sub-halos Search
基于 v_nfw_1.0，优化迭代循环以提高多核CPU利用率

优化改进：
1. 精简迭代循环 - 减少主进程阻塞时间
2. 按需打印 - 只在指定间隔打印状态
3. 按需复制 - 只在绘图时才复制种群数组
4. 移除回火机制 - 简化逻辑
5. 保持所有其他功能（NFW模型、宇宙学计算、验证等）
"""

import sys

import random
import glafic
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, linear_sum_assignment
import os
from datetime import datetime
import matplotlib.pyplot as plt
import subprocess
from plot_paper_style import plot_paper_style_nfw, plot_paper_style_nfw_compare, read_critical_curves
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

# ==================== 基准透镜参数加载函数 ====================
def load_baseline_lens_params(directory):
    """
    从指定目录的 bestfit.dat 文件加载基准透镜参数。

    bestfit.dat 格式（每行以 lens 或 point 开头，无索引列）：
        lens  <type>  z  p1  p2  p3  p4  p5  p6  p7
        ...
        point  z_s  x_s  y_s

    支持的主透镜类型: 'sie'（SIE 模型）和 'anfw'（轴对称 NFW 模型）。

    返回:
        (lens_params_dict, source_x, source_y, main_lens_key)
        - lens_params_dict: 含 'sers1', 'sers2', 以及主透镜键（'sie' 或 'anfw'）
        - source_x, source_y: 源位置 [arcsec]
        - main_lens_key: 主透镜的键名字符串
    """
    bestfit_path = os.path.join(directory, 'bestfit.dat')
    if not os.path.isfile(bestfit_path):
        raise FileNotFoundError(f"未找到基准参数文件: {bestfit_path}")

    lens_lines, point_params = [], None
    with open(bestfit_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            if parts[0] == 'lens':
                lens_lines.append(parts)
            elif parts[0] == 'point':
                point_params = parts

    if len(lens_lines) < 1:
        raise ValueError(
            f"bestfit.dat 需要至少1行 lens 参数，实际找到 {len(lens_lines)} 行: {bestfit_path}")
    if point_params is None:
        raise ValueError(f"bestfit.dat 缺少 point 行（源位置参数）: {bestfit_path}")

    params_dict, sers_count, type_counts, main_lens_key = {}, 0, {}, None
    for parts in lens_lines:
        lens_type = parts[1]
        z = float(parts[2])
        raw = [float(v) for v in parts[3:]]
        vals = (raw + [0.0] * 7)[:7]
        idx = len(params_dict) + 1
        if lens_type == 'sers':
            sers_count += 1
            key = f'sers{sers_count}'
        else:
            type_counts[lens_type] = type_counts.get(lens_type, 0) + 1
            n = type_counts[lens_type]
            key = lens_type if n == 1 else f'{lens_type}{n}'
            main_lens_key = key
        params_dict[key] = (idx, lens_type, z, *vals)

    if main_lens_key is None:
        main_lens_key = list(params_dict.keys())[-1]

    return params_dict, float(point_params[2]), float(point_params[3]), main_lens_key

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║              智能查找 glafic 可执行文件                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def find_glafic_bin(default_path=""):
    """
    智能查找 glafic 可执行文件
    
    查找顺序:
    1. 检查指定的默认路径
    2. 从 glafic Python 模块路径推断（支持外部定义的路径）
    3. 返回 None 如果都找不到
    """
    # 1. 首先检查默认路径
    if os.path.isfile(default_path) and os.access(default_path, os.X_OK):
        return default_path
    
    # 2. 尝试从 glafic 模块路径推断
    try:
        glafic_module_file = glafic.__file__
        if glafic_module_file:
            module_dir = os.path.dirname(os.path.abspath(glafic_module_file))
            
            possible_paths = [
                os.path.join(module_dir, '..', 'glafic'),
                os.path.join(module_dir, '..', '..', 'glafic'),
                os.path.join(module_dir, 'glafic'),
                os.path.join(module_dir, '..', 'bin', 'glafic'),
            ]
            
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.isfile(abs_path) and os.access(abs_path, os.X_OK):
                    return abs_path
    except Exception:
        pass
    
    # 3. 最后尝试从 PATH 环境变量查找
    import shutil
    glafic_in_path = shutil.which('glafic')
    if glafic_in_path:
        return glafic_in_path
    
    return None

print("=" * 70)
print("Version NFW 2.0: Optimized NFW Sub-halos Search")
print("=" * 70)

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║              宇宙学计算函数（用于精确计算 r_vir 和 r_s）                ║
# ╚═══════════════════════════════════════════════════════════════════════╝

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)  # H0 [km/s/Mpc], Ωm [dimensionless]

def calculate_r_vir_physical(m_vir_msun, z_lens):
    """
    精确计算维里半径（物理单位）
    
    参数:
        m_vir_msun: 维里质量 [M☉]
        z_lens: 透镜红移 [dimensionless]
    
    返回:
        r_vir_kpc: 维里半径 [kpc]
    """
    rho_crit = cosmo.critical_density(z_lens).to(u.Msun / u.kpc**3).value  # [M☉/kpc³]
    Om_z = cosmo.Om(z_lens)
    x = Om_z - 1
    Delta_vir = 18 * np.pi**2 + 82 * x - 39 * x**2  # Bryan & Norman 1998
    r_vir_kpc = (3 * m_vir_msun / (4 * np.pi * Delta_vir * rho_crit))**(1/3)
    return r_vir_kpc

def r_vir_to_arcsec(r_vir_kpc, z_lens):
    """将维里半径从物理单位转换为角单位"""
    D_A = cosmo.angular_diameter_distance(z_lens).to(u.kpc).value  # [kpc]
    r_vir_arcsec = (r_vir_kpc / D_A) * 206265  # [arcsec]
    return r_vir_arcsec

def calculate_nfw_radii(m_vir_msun, c_vir, z_lens):
    """计算 NFW 模型的特征半径"""
    r_vir_kpc = calculate_r_vir_physical(m_vir_msun, z_lens)
    r_vir_arcsec = r_vir_to_arcsec(r_vir_kpc, z_lens)
    r_s_kpc = r_vir_kpc / c_vir
    r_s_arcsec = r_vir_arcsec / c_vir
    return r_vir_kpc, r_vir_arcsec, r_s_kpc, r_s_arcsec

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                    可配置参数区域                                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# ==================== 0. 基准透镜参数路径配置 ====================
# 设置为包含 bestfit.dat 的目录路径即可加载外部基准透镜参数。
# 支持 sie（SIE 模型）和 anfw（轴对称 NFW）两种主透镜类型。
# 留空字符串 "" 则使用下方内置的 SIE 默认参数。
# 示例: BASELINE_LENS_DIR = "work/SN_2Sersic_NFW"
BASELINE_LENS_DIR = ""

# ==================== 1. 约束条件配置 ====================
CONSTRAINT_SIGMA = 1.0       # 位置约束的σ倍数 [dimensionless]
PENALTY_COEFFICIENT = 1000   # 违反约束的惩罚系数 [dimensionless]

# ==================== 2. Sub-halo 启用配置 ====================
active_subhalos = [1,2,3,4]        # 可修改为任意子集 [1-4]

# ==================== 3. 精细调试模式 ====================
fine_tuning = False          # 是否启用独立配置

# --- 通用配置（当 fine_tuning=False 时使用） ---
SEARCH_RADIUS = 0.1          # 位置搜索半径 [arcsec]
MASS_GUESS = 1.0e6             # 初始质量猜测 [M☉]
MASS_LOG_RANGE = 3.0         # 质量搜索范围 [dex]
CONCENTRATION_GUESS = 20.0   # 浓度参数初始猜测 [dimensionless]
CONCENTRATION_MIN = 5.0      # 浓度参数最小值 [dimensionless]
CONCENTRATION_MAX = 45.0     # 浓度参数最大值 [dimensionless]

# --- 精细配置（当 fine_tuning=True 时使用） ---
fine_tuning_configs = {
    1: {
        'search_radius': 0.1,        # [arcsec]
        'mass_guess': 1.0e5,         # [M☉]
        'mass_log_range': 4.5,       # [dex]
        'concentration_guess': 10.0, # [dimensionless]
        'concentration_min': 2.0,
        'concentration_max': 40.0
    },
    2: {
        'search_radius': 0.070,
        'mass_guess': 5.0e4,
        'mass_log_range': 4.0,
        'concentration_guess': 10.0,
        'concentration_min': 2.0,
        'concentration_max': 40.0
    },
    3: {
        'search_radius': 0.075,
        'mass_guess': 1.0e9,
        'mass_log_range': 5,
        'concentration_guess': 10.0,
        'concentration_min': 2.0,
        'concentration_max': 30.0
    },
    4: {
        'search_radius': 0.065,
        'mass_guess': 1.0e5,
        'mass_log_range': 4,
        'concentration_guess': 10.0,
        'concentration_min': 2.0,
        'concentration_max': 30.0
    }
}

# ==================== 4. 机器学习目标函数参数 ====================
LOSS_COEF_A = 4.0            # 位置chi²的权重系数 [dimensionless]
LOSS_COEF_B = 1              # 放大率chi²的权重系数 [dimensionless]
LOSS_PENALTY_PL = 10000.0    # 位置惩罚系数 [dimensionless]

# ==================== 4.1 透镜和源参数修改配置 ====================
source_modify = False        # 是否优化source位置
lens_modify = False          # 是否优化lens参数
modify_percentage = 0.1     # 参数允许变化的百分比 [fraction]

# ==================== 5. 优化算法配置 ====================
DE_MAXITER = 800             # 最大迭代次数 [count]
DE_POPSIZE = 60             # 种群大小 [count]
DE_ATOL = 1e-6               # 绝对容差 [dimensionless]
DE_TOL = 1e-6                # 相对容差 [dimensionless]
DE_SEED = random.randint(1, 100000)  # 随机种子 [integer]
DE_POLISH = True             # 是否启用局部优化抛光
DE_WORKERS = -1              # 并行核心数 [count], -1=全部CPU

# 早停机制配置
EARLY_STOPPING = True        # 是否启用早停
EARLY_STOP_PATIENCE = 60     # 容忍次数 [count]

# ==================== 6. MCMC 配置 ====================
MCMC_ENABLED = False          # 是否启用MCMC采样
MCMC_NWALKERS = 32           # walker数量 [count]，至少是参数维度的2倍
MCMC_NSTEPS = 2000           # 采样步数 [count]
MCMC_BURNIN = 300            # burn-in 步数 [count]，丢弃前N步
MCMC_THIN = 2                # 稀疏采样 [count]，每N步保留1个
MCMC_PERTURBATION = 0.01     # 初始扰动幅度 [fraction]，相对于参数范围
MCMC_PROGRESS = True         # 是否显示进度条
MCMC_WORKERS = -1            # 并行核心数 [count]，1=串行，-1=全部CPU，>1=指定核心数

# ==================== 6.1 MCMC 先验范围配置 ====================
# MCMC_CUSTOM_RANGE = True  → 使用下方自定义范围作为先验边界
# MCMC_CUSTOM_RANGE = False → 直接沿用 DE 搜索范围（bounds）作为先验边界
MCMC_CUSTOM_RANGE = False

# 自定义先验范围（仅在 MCMC_CUSTOM_RANGE=True 时生效）
MCMC_SEARCH_RADIUS = 0.3     # 位置搜索半径 [arcsec]，以各图像观测位置为中心
MCMC_LOG_M_MIN = 1.0         # log10(M_vir) 下限 [dex]
MCMC_LOG_M_MAX = 14.0        # log10(M_vir) 上限 [dex]
MCMC_C_MIN     = 1.0         # 浓度参数下限 [dimensionless]
MCMC_C_MAX     = 200.0       # 浓度参数上限 [dimensionless]

# ==================== 7. 输出配置 ====================
SHOW_2SIGMA = False          # 是否显示2σ横线
OUTPUT_PREFIX = "v_nfw_2_0"  # 输出文件前缀
COMPARE_GRAPH = True        # 比较模式：生成 baseline vs optimized 对比图（仅在 n_active_subhalos > 0 时生效）

# 绘图配置
Draw_Graph = 1               # 绘图模式: 0=不绘制, 1=绘制
draw_interval = 15            # 绘图间隔 [iterations]

# 打印配置（优化关键！）
PRINT_INTERVAL = 10          # 打印间隔 [iterations]，减少I/O阻塞

# NFW Profile 绘制配置
PLOT_NFW_PROFILES = True     # 是否绘制 NFW density profiles
NFW_PROFILE_RMAX = 0.2       # profile 绘制的最大半径 [arcsec]

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                         固定参数                                       ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# ==================== 可注入的观测数据（WebUI / 注入器可覆盖）====================
obs_positions_mas_list = [[-266.035, +0.427], [+118.835, -221.927], [+238.324, +227.270], [-126.157, +319.719]]
obs_magnifications_list = [-35.6, 15.7, -7.5, 9.1]
obs_mag_errors_list = [2.1, 1.3, 1.0, 1.1]
obs_pos_sigma_mas_list = [0.41, 0.86, 2.23, 3.11]
center_offset_x = +0.01535000   # 与观测数据同坐标系：天球坐标时填RA方向(东=正)，数学坐标时填向右为正
center_offset_y = +0.03220000   # [arcsec]
obs_x_flip = True   # True=输入为天球坐标(RA东向右输入，程序统一取负转数学坐标x向右); False=输入已为数学坐标

# 坐标转换：统一取符号，同时作用于观测位置和中心偏移，确保两者始终在同一坐标系下
_x_sign = -1 if obs_x_flip else 1
obs_positions_mas = np.array(obs_positions_mas_list)
obs_positions = np.zeros_like(obs_positions_mas)
obs_positions[:, 0] = _x_sign * obs_positions_mas[:, 0] / 1000.0
obs_positions[:, 1] = obs_positions_mas[:, 1] / 1000.0
center_offset_x    = _x_sign * center_offset_x   # 输入坐标系 → 模型(数学)坐标

obs_magnifications = np.array(obs_magnifications_list)
obs_mag_errors = np.array(obs_mag_errors_list)
obs_pos_sigma_mas = np.array(obs_pos_sigma_mas_list)

# 宇宙学参数
omega = 0.3                      # Ωm [dimensionless]
lambda_cosmo = 0.7               # ΩΛ [dimensionless]
weos = -1.0                      # w [dimensionless]
hubble = 0.7                     # h [dimensionless]

# glafic 网格设置
xmin, ymin = -0.5, -0.5          # [arcsec]
xmax, ymax = 0.5, 0.5            # [arcsec]
pix_ext = 0.01                   # [arcsec/pixel]
pix_poi = 0.2                    # [arcsec]
maxlev = 5                       # [count]

# 源参数
source_z = 0.4090                # [dimensionless]

# 默认基准透镜参数（SIE 模型，来自 SN_2Sersic_SIE/bestfit.dat）
source_x = 2.685497e-03          # [arcsec]
source_y = 2.443616e-02          # [arcsec]

# 透镜参数
lens_z = 0.2160                  # [dimensionless]

lens_params = {
    'sers1': (1, 'sers', 0.2160, 9.896617e+09, 2.656977e-03, 2.758473e-02,
              2.986760e-01, 1.124730e+02, 3.939718e-01, 1.057760e+00),
    'sers2': (2, 'sers', 0.2160, 2.555580e+10, 2.656977e-03, 2.758473e-02,
              4.242340e-01, 5.396370e+01, 1.538855e+00, 1.000000e+00),
    'sie': (3, 'sie', 0.2160, 1.183382e+02, 2.656977e-03, 2.758473e-02,
            1.571203e-01, 2.920348e+01, 0.0, 0.0)
}
MAIN_LENS_KEY = 'sie'  # 主透镜类型键名（'sie' 或 'anfw'）

# 若指定了外部基准路径，则覆盖上述默认参数
if BASELINE_LENS_DIR:
    _loaded, _sx, _sy, _mlk = load_baseline_lens_params(BASELINE_LENS_DIR)
    lens_params = _loaded
    source_x = _sx
    source_y = _sy
    MAIN_LENS_KEY = _mlk
    print(f"[基准透镜] 已从 {BASELINE_LENS_DIR} 加载参数 (主透镜类型: {MAIN_LENS_KEY})")
    print(f"           source_x={source_x:.6e}, source_y={source_y:.6e}")
else:
    print(f"[基准透镜] 使用内置默认 SIE 参数")

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                处理 active_subhalos 和配置参数                         ║
# ╚═══════════════════════════════════════════════════════════════════════╝

active_subhalos = sorted(list(set(active_subhalos)))
for idx in active_subhalos:
    if idx not in [1, 2, 3, 4]:
        raise ValueError(f"active_subhalos 包含无效的图像索引: {idx}")

n_active_subhalos = len(active_subhalos)
n_params_subhalo = n_active_subhalos * 4  # 每个 NFW 4个参数: x, y, log_m, c

n_params_source = 2 if source_modify else 0
n_params_lens = 15 if lens_modify else 0
n_params_extra = n_params_source + n_params_lens
n_params = n_params_subhalo + n_params_extra

# 构建每个 active sub-halo 的配置
subhalo_configs = {}
for img_idx in active_subhalos:
    if fine_tuning:
        cfg = fine_tuning_configs[img_idx]
        subhalo_configs[img_idx] = {
            'search_radius': cfg['search_radius'],
            'mass_guess': cfg['mass_guess'],
            'mass_log_range': cfg['mass_log_range'],
            'concentration_guess': cfg['concentration_guess'],
            'concentration_min': cfg['concentration_min'],
            'concentration_max': cfg['concentration_max']
        }
    else:
        subhalo_configs[img_idx] = {
            'search_radius': SEARCH_RADIUS,
            'mass_guess': MASS_GUESS,
            'mass_log_range': MASS_LOG_RANGE,
            'concentration_guess': CONCENTRATION_GUESS,
            'concentration_min': CONCENTRATION_MIN,
            'concentration_max': CONCENTRATION_MAX
        }

if draw_interval < 1:
    draw_interval = 1
if PRINT_INTERVAL < 1:
    PRINT_INTERVAL = 1

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                    创建时间戳输出目录                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

timestamp = datetime.now().strftime("%y%m%d_%H%M")
output_dir = timestamp
os.makedirs(output_dir, exist_ok=True)

print(f"\n输出目录: {output_dir}")

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║              迭代可视化函数（优化版：只在需要时调用）                   ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def plot_iteration_population(population, iteration_num, output_dir, bounds):
    """绘制种群分布（只在 Draw_Graph=1 且满足间隔时才实际执行）"""
    # 如果没有 subhalo，跳过绘图
    if n_active_subhalos == 0:
        return
    
    # 反归一化
    if population.max() <= 1.0 and population.min() >= 0.0:
        population_denorm = np.zeros_like(population)
        for i in range(population.shape[1]):
            lower, upper = bounds[i]
            population_denorm[:, i] = population[:, i] * (upper - lower) + lower
        population = population_denorm
    
    # 提取参数
    n_halos = n_active_subhalos
    log_m = []
    conc = []
    for i in range(n_halos):
        log_m.append(population[:, i*4 + 2])
        conc.append(population[:, i*4 + 3])
    
    # 绘制图表
    fig = plt.figure(figsize=(5*n_halos, 4))
    labels = [f'NFW {active_subhalos[i]}' for i in range(n_halos)]
    
    for i in range(n_halos):
        ax = plt.subplot(2, n_halos, i + 1)
        ax.hist(log_m[i], bins=20, alpha=0.6, color='blue')
        ax.set_xlabel(f'{labels[i]} log(M)', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.3)
        
        ax = plt.subplot(2, n_halos, n_halos + i + 1)
        ax.hist(conc[i], bins=20, alpha=0.6, color='green')
        ax.set_xlabel(f'{labels[i]} c_vir', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.3)
    
    plt.suptitle(f'Iteration {iteration_num}: NFW Parameters ({n_halos} Sub-halos)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = os.path.join(output_dir, f'iteration_{iteration_num:04d}.png')
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close()

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                    打印配置摘要                                        ║
# ╚═══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("配置摘要")
print("=" * 70)

print(f"\n模型类型: NFW 暗物质晕 (gnfw profile)")
print(f"  每个 sub-halo 参数: x, y, M_vir, c_vir (4 个)")

print(f"\n启用的 Sub-halos:")
print(f"  active_subhalos = {active_subhalos}")
print(f"  启用数量: {n_active_subhalos}")
print(f"  参数维度: {n_params}")

print(f"\n优化算法配置:")
print(f"  最大迭代: {DE_MAXITER}")
print(f"  种群大小: {DE_POPSIZE}")
print(f"  并行核心: {DE_WORKERS}" + (" (全部CPU)" if DE_WORKERS == -1 else ""))
print(f"  早停: {'启用' if EARLY_STOPPING else '禁用'}, 容忍次数: {EARLY_STOP_PATIENCE}")

print(f"\n性能优化配置:")
print(f"  打印间隔: 每 {PRINT_INTERVAL} 次迭代")
print(f"  绘图模式: {'启用' if Draw_Graph else '禁用'}")
if Draw_Graph:
    print(f"  绘图间隔: 每 {draw_interval} 次迭代")

print(f"\nMCMC 后验采样配置:")
if MCMC_ENABLED:
    print(f"  启用: 是")
    print(f"  Walkers: {MCMC_NWALKERS}")
    print(f"  采样步数: {MCMC_NSTEPS}")
    print(f"  Burn-in: {MCMC_BURNIN}")
    print(f"  有效样本数: ~{MCMC_NWALKERS * (MCMC_NSTEPS - MCMC_BURNIN) // MCMC_THIN}")
    if MCMC_CUSTOM_RANGE:
        print(f"  先验范围: 自定义")
        print(f"    位置半径: ±{MCMC_SEARCH_RADIUS*1000:.0f} mas")
        print(f"    logM: [{MCMC_LOG_M_MIN}, {MCMC_LOG_M_MAX}] dex")
        print(f"    c_vir: [{MCMC_C_MIN}, {MCMC_C_MAX}]")
    else:
        print(f"  先验范围: 沿用 DE 搜索范围（bounds）")
else:
    print(f"  启用: 否")

print(f"\n各 NFW Sub-halo 搜索空间:")
for img_idx in active_subhalos:
    cfg = subhalo_configs[img_idx]
    x_center = obs_positions[img_idx-1, 0]
    y_center = obs_positions[img_idx-1, 1]
    mass_log = np.log10(cfg['mass_guess'])
    mass_log_min = mass_log - cfg['mass_log_range']
    mass_log_max = mass_log + cfg['mass_log_range']
    
    print(f"  NFW Sub-halo {img_idx}:")
    print(f"    位置: ({x_center:.4f} ± {cfg['search_radius']:.4f}) arcsec")
    print(f"    质量: log(M) = [{mass_log_min:.1f}, {mass_log_max:.1f}]")
    print(f"    浓度: c = [{cfg['concentration_min']:.0f}, {cfg['concentration_max']:.0f}]")

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║              机器学习目标函数                                          ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def machine_learning_loss(pred_pos_matched, pred_mag_matched, delta_pos_mas):
    """机器学习损失函数"""
    Y_total = 0.0
    for i in range(4):
        chi2_pos_i = (delta_pos_mas[i] / obs_pos_sigma_mas[i])**2
        chi2_mag_i = ((pred_mag_matched[i] - obs_magnifications[i]) / obs_mag_errors[i])**2
        P_i = 0.0 if delta_pos_mas[i] <= obs_pos_sigma_mas[i] else LOSS_PENALTY_PL * delta_pos_mas[i]
        Y_i = LOSS_COEF_A * chi2_pos_i + LOSS_COEF_B * chi2_mag_i + P_i
        Y_total += Y_i
    return Y_total

# ==================== 每进程一次初始化（消除重复冷启动开销） ====================
# 基于 glafic 官方示例 mock_siex.c：在循环中只 init/quit 一次，
# 每次 evaluation 仅更新 sub-halo 参数后重新调用 model_init + point_solve。

_glafic_worker_pid = None  # 记录已完成 glafic 初始化的进程 PID

def _worker_prefix():
    """为当前进程生成唯一的 glafic 文件前缀，避免多进程写入同一临时文件。"""
    return f'temp_{OUTPUT_PREFIX}_w{os.getpid()}'

def _ensure_worker_init():
    """
    在当前进程中做一次性 glafic 初始化（懒惰式，每进程只执行一次）。
    固定参数（宇宙学、网格、基础透镜、点源）在此设置好，
    后续每次 evaluation 只需更新 sub-halo 参数并重调 model_init。
    """
    global _glafic_worker_pid
    pid = os.getpid()
    if _glafic_worker_pid == pid:
        return
    glafic.init(omega, lambda_cosmo, weos, hubble, _worker_prefix(),
                xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev, verb=0)
    glafic.startup_setnum(3 + n_active_subhalos, 0, 1)
    glafic.set_lens(*lens_params['sers1'])
    glafic.set_lens(*lens_params['sers2'])
    glafic.set_lens(*lens_params[MAIN_LENS_KEY])
    glafic.set_point(1, source_z, source_x, source_y)
    _glafic_worker_pid = pid

def _worker_cleanup():
    """
    清理本进程的 glafic 状态。
    在 workers=1（单进程）模式下，优化结束后须在主进程调用，
    让后续的 compute_model()（带完整 init/quit）能正常工作。
    在 workers>1 时，evaluations 在子进程中运行，主进程无需清理。
    """
    global _glafic_worker_pid
    if _glafic_worker_pid == os.getpid():
        glafic.quit()
        _glafic_worker_pid = None

def _parse_solve_result(result, verbose=False):
    """
    从 glafic.point_solve 结果计算匹配位置、放大率和 chi2。
    提取为独立函数避免在 compute_model 和 compute_model_eval 中重复代码。
    """
    n_images = len(result)
    if n_images == 5:
        abs_mags = [abs(img_data[2]) for img_data in result]
        drop_idx = int(np.argmin(abs_mags))
        result = [img_data for j, img_data in enumerate(result) if j != drop_idx]
        if verbose:
            print(f"  Info: 5 images found, dropped central image "
                  f"(index {drop_idx}, |μ|={abs_mags[drop_idx]:.4f})")
    elif n_images != 4:
        if verbose:
            print(f"  Warning: Found {n_images} images (expected 4)")
        return None, None, None, 1e10

    pred_positions = np.array([[img[0], img[1]] for img in result])
    pred_magnifications = np.array([img[2] for img in result])

    pred_positions[:, 0] += center_offset_x
    pred_positions[:, 1] += center_offset_y

    distances = cdist(obs_positions, pred_positions)
    row_ind, col_ind = linear_sum_assignment(distances)
    sort_idx = col_ind[np.argsort(row_ind)]
    pred_pos_matched = pred_positions[sort_idx]
    pred_mag_matched = pred_magnifications[sort_idx]

    delta_pos = np.array([
        np.sqrt(((pred_pos_matched[i, 0] - obs_positions[i, 0]) * 1000) ** 2 +
                ((pred_pos_matched[i, 1] - obs_positions[i, 1]) * 1000) ** 2)
        for i in range(len(obs_positions))
    ])
    mag_chi2 = np.sum(((pred_mag_matched - obs_magnifications) / obs_mag_errors) ** 2)
    return pred_pos_matched, pred_mag_matched, delta_pos, mag_chi2


def compute_model(nfw_params_list, verbose=False, src_x=None, src_y=None, lens_params_dict=None):
    """
    标准版：每次调用完整的 init → 设参数 → model_init → solve → quit。
    用于基准模型计算（步骤1）和最终结果分析（步骤3），不用于大批量优化。
    """
    use_src_x = src_x if src_x is not None else source_x
    use_src_y = src_y if src_y is not None else source_y
    use_lens_params = lens_params_dict if lens_params_dict is not None else lens_params

    glafic.init(omega, lambda_cosmo, weos, hubble, _worker_prefix(),
                xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev, verb=0)
    n_subhalos = len(nfw_params_list)
    glafic.startup_setnum(3 + n_subhalos, 0, 1)
    glafic.set_lens(*use_lens_params['sers1'])
    glafic.set_lens(*use_lens_params['sers2'])
    glafic.set_lens(*use_lens_params[MAIN_LENS_KEY])
    for i, (x_sub, y_sub, mass_sub, conc_sub) in enumerate(nfw_params_list):
        glafic.set_lens(4 + i, 'gnfw', 0.2160, mass_sub, x_sub, y_sub, 0.0, 0.0, conc_sub, 1.0)
    glafic.set_point(1, source_z, use_src_x, use_src_y)
    glafic.model_init(verb=0)
    result = glafic.point_solve(source_z, use_src_x, use_src_y, verb=0)
    pos, mag, delta_pos, chi2 = _parse_solve_result(result, verbose=verbose)
    glafic.quit()
    if pos is None:
        return None, None, None, 1e10
    return pos, mag, delta_pos, chi2


def compute_model_eval(nfw_params_list, src_x=None, src_y=None, lens_params_dict=None):
    """
    优化专用版：每进程只调用一次 glafic.init()，消除重复冷启动开销。
    每次 evaluation 仅更新变化的参数（sub-halo，以及可选的 source/base lenses），
    直接 model_init + point_solve，不做 init/quit。
    设计依据：glafic 官方示例 mock_siex.c 在循环中重复调用 model_init。
    """
    use_src_x = src_x if src_x is not None else source_x
    use_src_y = src_y if src_y is not None else source_y
    use_lens_params = lens_params_dict if lens_params_dict is not None else lens_params

    _ensure_worker_init()  # 懒惰式：首次调用时初始化，之后直接跳过

    # 只有 lens_modify=True 时基础透镜参数才会变化，需要重设
    if lens_params_dict is not None:
        glafic.set_lens(*use_lens_params['sers1'])
        glafic.set_lens(*use_lens_params['sers2'])
        glafic.set_lens(*use_lens_params[MAIN_LENS_KEY])

    # sub-halo 参数每次 evaluation 都变化，必须更新
    for i, (x_sub, y_sub, mass_sub, conc_sub) in enumerate(nfw_params_list):
        glafic.set_lens(4 + i, 'gnfw', 0.2160, mass_sub, x_sub, y_sub, 0.0, 0.0, conc_sub, 1.0)

    # 只有 source_modify=True 时点源位置才会变化，需要重设
    if src_x is not None or src_y is not None:
        glafic.set_point(1, source_z, use_src_x, use_src_y)

    glafic.model_init(verb=0)
    result = glafic.point_solve(source_z, use_src_x, use_src_y, verb=0)
    return _parse_solve_result(result)

# ==================== 计算基准模型 ====================
print("\n" + "=" * 70)
print("步骤1: 计算基准模型（无sub-halo）")
print("=" * 70)

base_pos, base_mag, base_delta_pos, base_mag_chi2 = compute_model([])

print(f"\n基准模型结果:")
print(f"  位置RMS: {np.sqrt(np.mean(base_delta_pos**2)):.3f} mas")
print(f"  放大率chi2: {base_mag_chi2:.2f}")

# ==================== 定义优化目标 ====================
def objective_function(params):
    """灵活参数优化目标函数"""
    nfw_list = []
    for i in range(n_active_subhalos):
        x = params[i*4]
        y = params[i*4 + 1]
        log_m = params[i*4 + 2]
        c = params[i*4 + 3]
        mass = 10**log_m
        nfw_list.append((x, y, mass, c))
    
    src_x_opt = None
    src_y_opt = None
    lens_params_opt = None
    
    idx = n_params_subhalo
    
    if source_modify:
        src_x_opt = params[idx]
        src_y_opt = params[idx + 1]
        idx += 2
    
    if lens_modify:
        lens_params_opt = {}
        sers1_mass = params[idx]
        sers1_x = params[idx + 1]
        sers1_y = params[idx + 2]
        sers1_re = params[idx + 3]
        sers1_pa = params[idx + 4]
        lens_params_opt['sers1'] = (1, 'sers', 0.2160, sers1_mass, sers1_x, sers1_y,
                                     sers1_re, sers1_pa, lens_params['sers1'][8], lens_params['sers1'][9])
        idx += 5
        
        sers2_mass = params[idx]
        sers2_x = params[idx + 1]
        sers2_y = params[idx + 2]
        sers2_re = params[idx + 3]
        sers2_pa = params[idx + 4]
        lens_params_opt['sers2'] = (2, 'sers', 0.2160, sers2_mass, sers2_x, sers2_y,
                                     sers2_re, sers2_pa, lens_params['sers2'][8], lens_params['sers2'][9])
        idx += 5
        
        # 主透镜（sie 或 anfw）：优化前5个自由参数，第8/9个参数保持原值
        ml_p1 = params[idx]       # sigma (sie) 或 mass (anfw)
        ml_x = params[idx + 1]
        ml_y = params[idx + 2]
        ml_e = params[idx + 3]
        ml_pa = params[idx + 4]
        lens_params_opt[MAIN_LENS_KEY] = (
            3, MAIN_LENS_KEY, 0.2160, ml_p1, ml_x, ml_y,
            ml_e, ml_pa,
            lens_params[MAIN_LENS_KEY][8],
            lens_params[MAIN_LENS_KEY][9])

    pos, mag, delta_pos, mag_chi2 = compute_model_eval(nfw_list, src_x=src_x_opt,
                                                       src_y=src_y_opt,
                                                       lens_params_dict=lens_params_opt)
    
    if pos is None:
        return 1e15
    
    Y = machine_learning_loss(pos, mag, delta_pos)
    return Y

# ==================== 优化搜索 ====================
print("\n" + "=" * 70)
print(f"步骤2: 差分进化算法优化搜索 ({n_params}维空间)")
print("=" * 70)

# 构建bounds
bounds = []
for img_idx in active_subhalos:
    cfg = subhalo_configs[img_idx]
    x_center = obs_positions[img_idx-1, 0]
    y_center = obs_positions[img_idx-1, 1]
    mass_log = np.log10(cfg['mass_guess'])
    mass_log_min = mass_log - cfg['mass_log_range']
    mass_log_max = mass_log + cfg['mass_log_range']
    
    bounds.append((x_center - cfg['search_radius'], x_center + cfg['search_radius']))
    bounds.append((y_center - cfg['search_radius'], y_center + cfg['search_radius']))
    bounds.append((mass_log_min, mass_log_max))
    bounds.append((cfg['concentration_min'], cfg['concentration_max']))

if source_modify:
    bounds.append((source_x * (1 - modify_percentage), source_x * (1 + modify_percentage)))
    bounds.append((source_y * (1 - modify_percentage), source_y * (1 + modify_percentage)))

if lens_modify:
    for key in ['sers1', 'sers2']:
        for i in [3, 4, 5, 6, 7]:
            bounds.append((lens_params[key][i] * (1 - modify_percentage), 
                          lens_params[key][i] * (1 + modify_percentage)))
    # 主透镜参数的 bounds（sie 或 anfw）
    for _i in [3, 4, 5, 6, 7]:
        bounds.append((lens_params[MAIN_LENS_KEY][_i] * (1 - modify_percentage),
                       lens_params[MAIN_LENS_KEY][_i] * (1 + modify_percentage)))

print(f"\n开始优化...")

# 使用 DifferentialEvolutionSolver
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
import scipy

print(f"  Scipy版本: {scipy.__version__}")

# 兼容不同scipy版本
try:
    solver = DifferentialEvolutionSolver(
        objective_function, bounds,
        maxiter=DE_MAXITER, popsize=DE_POPSIZE,
        atol=DE_ATOL, tol=DE_TOL, seed=DE_SEED,
        polish=DE_POLISH, disp=False,  # 关闭内置显示
        workers=DE_WORKERS, updating='deferred'
    )
    print(f"  使用seed参数初始化成功")
except TypeError:
    try:
        solver = DifferentialEvolutionSolver(
            objective_function, bounds,
            maxiter=DE_MAXITER, popsize=DE_POPSIZE,
            atol=DE_ATOL, tol=DE_TOL, random_state=DE_SEED,
            polish=DE_POLISH, disp=False,
            workers=DE_WORKERS, updating='deferred'
        )
        print(f"  使用random_state参数初始化成功")
    except TypeError:
        np.random.seed(DE_SEED)
        solver = DifferentialEvolutionSolver(
            objective_function, bounds,
            maxiter=DE_MAXITER, popsize=DE_POPSIZE,
            atol=DE_ATOL, tol=DE_TOL,
            polish=DE_POLISH, disp=False,
            workers=DE_WORKERS, updating='deferred'
        )
        print(f"  使用numpy.random.seed()替代")

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║              优化迭代循环（精简版 - 最大化CPU利用率）                   ║
# ╚═══════════════════════════════════════════════════════════════════════╝

print(f"\n{'='*50}")
print(f"开始迭代优化（精简模式，每{PRINT_INTERVAL}次打印）")
print(f"{'='*50}")

# 初始状态
if Draw_Graph:
    plot_iteration_population(solver.population.copy(), 0, output_dir, bounds)

iteration = 1
previous_best_energy = np.min(solver.population_energies)
converged_count = 0
best_ever_energy = previous_best_energy

print(f"\n迭代 0: 初始最佳 = {previous_best_energy:.6f}")

while True:
    # =============== 核心并行计算 ===============
    try:
        next_gen = solver.__next__()
    except StopIteration:
        print(f"\n✓ 优化收敛！（算法内部判定）")
        break
    
    current_best_energy = np.min(solver.population_energies)
    
    # =============== 精简的迭代间处理 ===============
    
    # 按需打印（减少I/O阻塞）
    if iteration % PRINT_INTERVAL == 0 or current_best_energy < best_ever_energy:
        if current_best_energy < best_ever_energy:
            print(f"迭代 {iteration}: 最佳 = {current_best_energy:.6f} ★ (改进)")
            best_ever_energy = current_best_energy
        else:
            print(f"迭代 {iteration}: 最佳 = {current_best_energy:.6f}")
    
    # 按需绘图（只在需要时才复制种群）
    if Draw_Graph and iteration % draw_interval == 0:
        plot_iteration_population(solver.population.copy(), iteration, output_dir, bounds)
    
    # 精简的早停检查
    if EARLY_STOPPING:
        abs_change = abs(current_best_energy - previous_best_energy)
        if abs_change < DE_ATOL:
            converged_count += 1
            if converged_count >= EARLY_STOP_PATIENCE:
                print(f"\n✓ 早停触发！连续 {converged_count} 次满足容差。")
                break
        else:
            converged_count = 0
    
    previous_best_energy = current_best_energy
    iteration += 1
    
    if iteration > DE_MAXITER:
        print(f"\n✓ 达到最大迭代次数 {DE_MAXITER}。")
        break

print(f"\n总迭代次数: {iteration}")
print(f"最终最佳值: {np.min(solver.population_energies):.6f}")

# 如果以 workers=1（单进程）运行，evaluations 在主进程中完成，
# glafic 此时处于 "已 init、未 quit" 状态；需先清理，
# 再让后续的 compute_model()（带完整 init/quit）能正常工作。
_worker_cleanup()

# 获取最终结果
result = solver.x
final_fun = np.min(solver.population_energies)

# 解析结果
best_params = []
best_params_with_img_idx = []
for i, img_idx in enumerate(active_subhalos):
    x = result[i*4]
    y = result[i*4 + 1]
    log_m = result[i*4 + 2]
    c = result[i*4 + 3]
    mass = 10**log_m
    best_params.append((x, y, mass, c))
    best_params_with_img_idx.append((img_idx, x, y, mass, c))

best_source_x = source_x
best_source_y = source_y
best_lens_params = lens_params.copy()

idx = n_params_subhalo

if source_modify:
    best_source_x = result[idx]
    best_source_y = result[idx + 1]
    idx += 2

if lens_modify:
    best_lens_params = {}
    sers1_mass = result[idx]
    sers1_x = result[idx + 1]
    sers1_y = result[idx + 2]
    sers1_re = result[idx + 3]
    sers1_pa = result[idx + 4]
    best_lens_params['sers1'] = (1, 'sers', 0.2160, sers1_mass, sers1_x, sers1_y,
                                  sers1_re, sers1_pa, lens_params['sers1'][8], lens_params['sers1'][9])
    idx += 5
    
    sers2_mass = result[idx]
    sers2_x = result[idx + 1]
    sers2_y = result[idx + 2]
    sers2_re = result[idx + 3]
    sers2_pa = result[idx + 4]
    best_lens_params['sers2'] = (2, 'sers', 0.2160, sers2_mass, sers2_x, sers2_y,
                                  sers2_re, sers2_pa, lens_params['sers2'][8], lens_params['sers2'][9])
    idx += 5
    
    # 主透镜（sie 或 anfw）
    ml_p1 = result[idx]
    ml_x = result[idx + 1]
    ml_y = result[idx + 2]
    ml_e = result[idx + 3]
    ml_pa = result[idx + 4]
    best_lens_params[MAIN_LENS_KEY] = (
        3, MAIN_LENS_KEY, 0.2160, ml_p1, ml_x, ml_y,
        ml_e, ml_pa,
        lens_params[MAIN_LENS_KEY][8],
        lens_params[MAIN_LENS_KEY][9])

print(f"\n" + "=" * 70)
print("步骤3: 分析最佳结果")
print("=" * 70)

best_pos, best_mag, best_delta_pos, best_mag_chi2 = compute_model(
    best_params, verbose=True, src_x=best_source_x, src_y=best_source_y,
    lens_params_dict=best_lens_params if lens_modify else None
)

if best_pos is None:
    print("\n✗ 优化失败：最佳参数无法产生4个图像！")
    sys.exit(1)

print(f"\n最佳 {n_active_subhalos} 个 NFW sub-halo 参数:")
for img_idx, x, y, m, c in best_params_with_img_idx:
    r_vir_kpc, r_vir_arcsec, r_s_kpc, r_s_arcsec = calculate_nfw_radii(m, c, lens_z)
    
    print(f"  NFW Sub-halo at Image {img_idx}:")
    print(f"    位置: ({x:.6f}, {y:.6f}) arcsec")
    print(f"    质量: M_vir = {m:.2e} M☉ (log10 = {np.log10(m):.2f})")
    print(f"    浓度: c_vir = {c:.2f}")
    print(f"    维里半径: r_vir = {r_vir_kpc:.3f} kpc = {r_vir_arcsec*1000:.2f} mas")
    print(f"    特征半径: r_s = {r_s_kpc:.3f} kpc = {r_s_arcsec*1000:.2f} mas")

print(f"\n改善效果:")
print(f"  基准chi2: {base_mag_chi2:.2f}")
print(f"  最佳chi2: {best_mag_chi2:.2f}")
improvement = (base_mag_chi2 - best_mag_chi2) / base_mag_chi2 * 100
print(f"  改善: {improvement:.1f}%")

max_pos_deviation_mas = CONSTRAINT_SIGMA * obs_pos_sigma_mas
constraint_satisfied = True
print(f"\n位置约束检查:")
for i in range(4):
    status = "✓ OK" if best_delta_pos[i] <= max_pos_deviation_mas[i] else "✗ 超限"
    if best_delta_pos[i] > max_pos_deviation_mas[i]:
        constraint_satisfied = False
    print(f"  Img {i+1}: {best_delta_pos[i]:.2f} mas < {max_pos_deviation_mas[i]:.2f} mas [{status}]")

if constraint_satisfied:
    print(f"\n✓ 所有约束满足！")
else:
    print(f"\n⚠ 部分图像超限")

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║              步骤4: MCMC 后验采样                                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

if MCMC_ENABLED:
    print("\n" + "=" * 70)
    print("步骤4: MCMC 后验采样（基于DE最优解）")
    print("=" * 70)

    try:
        import emcee
        import corner
        from tqdm import tqdm
        print(f"  ✓ emcee, corner, tqdm 已导入")
    except ImportError as e:
        print(f"  ✗ 缺少依赖库: {e}")
        print(f"    请运行: pip install emcee corner tqdm")
        MCMC_ENABLED = False

if MCMC_ENABLED:

    def log_probability(params):
        """
        对数概率函数 = log(prior) + log(likelihood)

        先验：只保留物理必要约束，不使用 DE 搜索域边界，
        让后验自由向外延伸，避免边界堆积效应。
        """
        if MCMC_CUSTOM_RANGE:
            # 使用自定义先验范围
            for i in range(n_active_subhalos):
                img_idx = active_subhalos[i]
                x_ctr   = obs_positions[img_idx - 1, 0]
                y_ctr   = obs_positions[img_idx - 1, 1]
                x     = params[i*4]
                y     = params[i*4 + 1]
                log_m = params[i*4 + 2]
                c     = params[i*4 + 3]
                if abs(x - x_ctr) > MCMC_SEARCH_RADIUS or abs(y - y_ctr) > MCMC_SEARCH_RADIUS:
                    return -np.inf
                if not (MCMC_LOG_M_MIN <= log_m <= MCMC_LOG_M_MAX):
                    return -np.inf
                if not (MCMC_C_MIN <= c <= MCMC_C_MAX):
                    return -np.inf
        else:
            # 使用 DE 搜索范围（bounds 列表），与 DE 优化保持完全一致
            for i, (low, high) in enumerate(bounds):
                if not (low <= params[i] <= high):
                    return -np.inf

        nfw_list = []
        for i in range(n_active_subhalos):
            x     = params[i*4]
            y     = params[i*4 + 1]
            log_m = params[i*4 + 2]
            c     = params[i*4 + 3]
            nfw_list.append((x, y, 10**log_m, c))

        src_x_opt = None
        src_y_opt = None
        lens_params_opt = None
        idx = n_params_subhalo

        if source_modify:
            src_x_opt = params[idx]
            src_y_opt = params[idx + 1]
            idx += 2

        if lens_modify:
            lens_params_opt = {}
            sers1_mass = params[idx]; sers1_x = params[idx+1]; sers1_y = params[idx+2]
            sers1_re   = params[idx+3]; sers1_pa = params[idx+4]
            lens_params_opt['sers1'] = (1, 'sers', 0.2160, sers1_mass, sers1_x, sers1_y,
                                        sers1_re, sers1_pa, lens_params['sers1'][8], lens_params['sers1'][9])
            idx += 5
            sers2_mass = params[idx]; sers2_x = params[idx+1]; sers2_y = params[idx+2]
            sers2_re   = params[idx+3]; sers2_pa = params[idx+4]
            lens_params_opt['sers2'] = (2, 'sers', 0.2160, sers2_mass, sers2_x, sers2_y,
                                        sers2_re, sers2_pa, lens_params['sers2'][8], lens_params['sers2'][9])
            idx += 5
            ml_p1 = params[idx]; ml_x = params[idx+1]; ml_y = params[idx+2]
            ml_e  = params[idx+3]; ml_pa = params[idx+4]
            lens_params_opt[MAIN_LENS_KEY] = (3, MAIN_LENS_KEY, 0.2160, ml_p1, ml_x, ml_y,
                                              ml_e, ml_pa,
                                              lens_params[MAIN_LENS_KEY][8],
                                              lens_params[MAIN_LENS_KEY][9])

        pos, mag, delta_pos, _ = compute_model(nfw_list, src_x=src_x_opt, src_y=src_y_opt,
                                               lens_params_dict=lens_params_opt)
        if pos is None:
            return -np.inf

        loss = machine_learning_loss(pos, mag, delta_pos)
        if loss >= 1e10:
            return -np.inf

        return -0.5 * loss

    # 初始化 walkers
    ndim = n_params
    best_result = result

    print(f"\n初始化MCMC采样器:")
    print(f"  参数维度: {ndim}")
    print(f"  Walkers: {MCMC_NWALKERS}")
    print(f"  采样步数: {MCMC_NSTEPS}")
    print(f"  Burn-in: {MCMC_BURNIN}")

    if MCMC_NWALKERS < 2 * ndim:
        MCMC_NWALKERS = 2 * ndim + 2
        print(f"  [调整] Walkers 增加到 {MCMC_NWALKERS} (至少为参数维度的2倍)")

    initial_positions = []
    for _ in range(MCMC_NWALKERS):
        perturbation = np.array([
            np.random.normal(0, MCMC_PERTURBATION * (bounds[i][1] - bounds[i][0]))
            for i in range(ndim)
        ])
        new_pos = best_result + perturbation
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
        print(f"  ⚠ 多进程模式：如果出错请设置 MCMC_WORKERS = 1")
        with Pool(mcmc_workers_actual) as pool:
            sampler = emcee.EnsembleSampler(MCMC_NWALKERS, ndim, log_probability, pool=pool)
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
        sampler = emcee.EnsembleSampler(MCMC_NWALKERS, ndim, log_probability)
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
    print(f"  总样本数: {MCMC_NWALKERS * MCMC_NSTEPS}")
    print(f"  有效样本数（去除burn-in）: {len(samples)}")

    # 保存 MCMC 链
    mcmc_chain_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_mcmc_chain.dat')
    param_names = []
    for img_idx in active_subhalos:
        param_names.extend([f'x_{img_idx}', f'y_{img_idx}', f'logM_{img_idx}', f'c_{img_idx}'])
    if source_modify:
        param_names.extend(['src_x', 'src_y'])
    if lens_modify:
        param_names.extend(['sers1_m', 'sers1_x', 'sers1_y', 'sers1_re', 'sers1_pa',
                            'sers2_m', 'sers2_x', 'sers2_y', 'sers2_re', 'sers2_pa',
                            f'{MAIN_LENS_KEY}_p1', f'{MAIN_LENS_KEY}_x', f'{MAIN_LENS_KEY}_y',
                            f'{MAIN_LENS_KEY}_e', f'{MAIN_LENS_KEY}_pa'])
    np.savetxt(mcmc_chain_file, samples, header=' '.join(param_names))
    print(f"  ✓ MCMC链已保存: {mcmc_chain_file}")

    # ==================== 计算参数统计 ====================
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
        if name.startswith('x_') or name.startswith('y_'):
            print(f"  {name}: {median:.6f} +{(upper-median)*1000:.3f} -{(median-lower)*1000:.3f} mas")
        elif name.startswith('logM_'):
            print(f"  {name}: {median:.3f} +{upper-median:.3f} -{median-lower:.3f} dex"
                  f"  (M = {10**median:.3e} M_sun)")
        elif name.startswith('c_'):
            print(f"  {name}: {median:.2f} +{upper-median:.2f} -{median-lower:.2f}")
        else:
            print(f"  {name}: {median:.6e} +{upper-median:.3e} -{median-lower:.3e}")

    # ==================== 绘制 Corner Plot ====================
    if n_params_subhalo > 0:
        print(f"\n生成 Corner Plot...")
        corner_labels = []
        for img_idx in active_subhalos:
            corner_labels.extend([f'$x_{img_idx}$', f'$y_{img_idx}$',
                                   f'$\\log M_{img_idx}$', f'$c_{img_idx}$'])

        _n_corner = n_params_subhalo
        _corner_samples = samples[:, :_n_corner]
        _N_corner = len(_corner_samples)

        fig = corner.corner(
            _corner_samples,
            labels=corner_labels[:_n_corner],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True, title_fmt='.3f',
            truths=best_result[:_n_corner],
            truth_color='red',
            hist_kwargs={'alpha': 0.75},
        )
        _corner_grid = np.array(fig.axes).reshape((_n_corner, _n_corner))
        for _ci in range(_n_corner):
            _ax = _corner_grid[_ci, _ci]
            from matplotlib.ticker import MaxNLocator as _MLoc
            _ylo, _yhi = _ax.get_ylim()
            _ax2 = _ax.twinx()
            _ax2.set_ylim(_ylo / _N_corner * 100, _yhi / _N_corner * 100)
            _ax2.yaxis.set_major_locator(_MLoc(nbins=4, prune='lower'))
            _ax2.tick_params(axis='y', labelsize=7, length=3, width=0.8)
            _ax2.set_ylabel('%', fontsize=8, rotation=0, labelpad=10, va='center')
        corner_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_corner.png')
        plt.savefig(corner_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Corner plot 已保存: {corner_file}")

        # ==================== 绘制轨迹图 ====================
        print(f"\n生成 MCMC 链轨迹图...")
        fig, axes = plt.subplots(n_params_subhalo, figsize=(10, 2*n_params_subhalo), sharex=True)
        if n_params_subhalo == 1:
            axes = [axes]
        for i in range(n_params_subhalo):
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
        print(f"  ✓ 轨迹图已保存: {trace_file}")
    else:
        print(f"\n跳过 Corner Plot 和轨迹图（无 subhalo 参数）")

    # ==================== 计算质量后验分布 ====================
    if n_active_subhalos > 0:
        print(f"\n计算质量后验分布...")
    mass_posterior_stats = {}
    for i, img_idx in enumerate(active_subhalos):
        log_m_samples_i = samples[:, i*4 + 2]
        mass_samples = 10**log_m_samples_i

        median = np.median(mass_samples)
        lower  = np.percentile(mass_samples, 16)
        upper  = np.percentile(mass_samples, 84)
        mass_name = f'mass_{img_idx}'
        mass_posterior_stats[mass_name] = {
            'median': median, 'lower_1sigma': lower, 'upper_1sigma': upper,
            'error_plus': upper - median, 'error_minus': median - lower,
            'samples': mass_samples
        }
        print(f"  mass_{img_idx}: {median:.3e} +{upper-median:.3e} -{median-lower:.3e} M_sun")

    # ==================== 绘制质量一维后验分布图 ====================
    if n_active_subhalos > 0:
        print(f"\n生成质量一维后验分布图（logM）...")
        from scipy.stats import gaussian_kde

        n_mass_halos = len(active_subhalos)
        fig_mass, axes_mass = plt.subplots(1, n_mass_halos, figsize=(5 * n_mass_halos, 4))
        if n_mass_halos == 1:
            axes_mass = [axes_mass]

        for i, img_idx in enumerate(active_subhalos):
            mass_name = f'mass_{img_idx}'
            mass_samples_i = mass_posterior_stats[mass_name]['samples']
            valid_mask = mass_samples_i > 0
            log_mass_samples = np.log10(mass_samples_i[valid_mask])

            # DE 最优解质量
            log_de_mass = np.log10(best_params[i][2])

            kde = gaussian_kde(log_mass_samples, bw_method='scott')
            x_min_log = min(log_mass_samples.min() - 0.3, log_de_mass - 0.3)
            x_max_log = max(log_mass_samples.max() + 0.3, log_de_mass + 0.3)
            x_grid = np.linspace(x_min_log, x_max_log, 500)
            y_kde  = kde(x_grid)

            ax = axes_mass[i]
            ax.plot(x_grid, y_kde, color='steelblue', lw=2)
            ax.fill_between(x_grid, y_kde, alpha=0.25, color='steelblue')

            log_median = np.log10(mass_posterior_stats[mass_name]['median'])
            log_lower  = np.log10(mass_posterior_stats[mass_name]['lower_1sigma'])
            log_upper  = np.log10(mass_posterior_stats[mass_name]['upper_1sigma'])

            ax.axvline(log_median, color='steelblue', lw=1.5, ls='--',
                       label=f'median = {log_median:.2f}')
            ax.axvspan(log_lower, log_upper, alpha=0.15, color='steelblue', label=r'1$\sigma$')
            ax.axvline(log_de_mass, color='tomato', lw=2, ls='-',
                       label=f'DE best = {log_de_mass:.2f}')

            ax.set_xlabel(r'$\log_{10}(M_{\rm vir} / M_\odot)$', fontsize=13)
            ax.set_ylabel('Posterior density', fontsize=13)
            ax.set_title(f'NFW Sub-halo {img_idx} mass posterior', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.set_xlim(x_min_log, x_max_log)
            ax.set_ylim(bottom=0)

        plt.tight_layout()
        mass_1d_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_mass_posterior_1d.png')
        plt.savefig(mass_1d_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 质量一维后验分布图已保存: {mass_1d_file}")
    else:
        print(f"\n跳过质量后验分布图（无 subhalo）")

    # ==================== 保存后验统计文件 ====================
    posterior_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_posterior.txt')
    with open(posterior_file, 'w') as f:
        f.write("# ============================================================\n")
        f.write("# MCMC Posterior Distribution Summary\n")
        f.write("# Version NFW 2.0\n")
        f.write("# ============================================================\n\n")
        f.write(f"# Walkers: {MCMC_NWALKERS}, Steps: {MCMC_NSTEPS}, "
                f"Burn-in: {MCMC_BURNIN}, Thin: {MCMC_THIN}\n")
        f.write(f"# Effective samples: {len(samples)}\n\n")
        f.write("# parameter  median  16%_lower  84%_upper  error_plus  error_minus\n\n")
        for i, name in enumerate(param_names):
            st = posterior_stats[name]
            f.write(f"{name}  {st['median']:.10e}  {st['lower_1sigma']:.10e}  "
                    f"{st['upper_1sigma']:.10e}  {st['error_plus']:.10e}  {st['error_minus']:.10e}\n")
        f.write("\n# Mass posterior statistics (M_sun)\n")
        for img_idx in active_subhalos:
            st = mass_posterior_stats[f'mass_{img_idx}']
            f.write(f"mass_{img_idx}  {st['median']:.10e}  {st['lower_1sigma']:.10e}  "
                    f"{st['upper_1sigma']:.10e}  {st['error_plus']:.10e}  {st['error_minus']:.10e}\n")
        f.write("\n# Summary for paper:\n")
        for img_idx in active_subhalos:
            f.write(f"# NFW Sub-halo at Image {img_idx}:\n")
            xs  = posterior_stats[f'x_{img_idx}']
            ys  = posterior_stats[f'y_{img_idx}']
            lms = posterior_stats[f'logM_{img_idx}']
            cs  = posterior_stats[f'c_{img_idx}']
            ms  = mass_posterior_stats[f'mass_{img_idx}']
            f.write(f"#   x    = {xs['median']:.6f} +{xs['error_plus']*1000:.3f} -{xs['error_minus']*1000:.3f} mas\n")
            f.write(f"#   y    = {ys['median']:.6f} +{ys['error_plus']*1000:.3f} -{ys['error_minus']*1000:.3f} mas\n")
            f.write(f"#   logM = {lms['median']:.3f} +{lms['error_plus']:.3f} -{lms['error_minus']:.3f} dex\n")
            f.write(f"#   c    = {cs['median']:.2f} +{cs['error_plus']:.2f} -{cs['error_minus']:.2f}\n")
            f.write(f"#   M    = {ms['median']:.3e} +{ms['error_plus']:.3e} -{ms['error_minus']:.3e} M_sun\n\n")
    print(f"  ✓ 后验统计已保存: {posterior_file}")

# ==================== 生成图表 ====================
print("\n" + "=" * 70)
print("步骤5: 生成最终图表")
print("=" * 70)

glafic.init(omega, lambda_cosmo, weos, hubble, f'temp_{OUTPUT_PREFIX}_best',
            xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev, verb=0)

glafic.startup_setnum(3 + n_active_subhalos, 0, 1)
glafic.set_lens(*best_lens_params['sers1'])
glafic.set_lens(*best_lens_params['sers2'])
glafic.set_lens(*best_lens_params[MAIN_LENS_KEY])
for i, (x_sub, y_sub, mass_sub, conc_sub) in enumerate(best_params):
    glafic.set_lens(4 + i, 'gnfw', 0.2160, mass_sub, x_sub, y_sub, 0.0, 0.0, conc_sub, 1.0)

glafic.set_point(1, source_z, best_source_x, best_source_y)
glafic.model_init(verb=0)
glafic.writecrit(source_z)

crit_file = f'temp_{OUTPUT_PREFIX}_best_crit.dat'
crit_segments, caus_segments = read_critical_curves(crit_file)

glafic.quit()

output_plot_file = os.path.join(output_dir, f"result_{OUTPUT_PREFIX}.png")

# 根据 COMPARE_GRAPH 选择绘图模式
if COMPARE_GRAPH and n_active_subhalos > 0:
    # 比较模式：baseline vs optimized
    output_plot_file_compare = os.path.join(output_dir, f"result_{OUTPUT_PREFIX}_compare.png")
    
    plot_paper_style_nfw_compare(
        img_numbers=np.array([1, 2, 3, 4]),
        delta_pos_mas_baseline=base_delta_pos,
        delta_pos_mas_optimized=best_delta_pos,
        sigma_pos_mas=obs_pos_sigma_mas,
        mu_obs=obs_magnifications,
        mu_obs_err=obs_mag_errors,
        mu_pred_baseline=base_mag,
        mu_pred_optimized=best_mag,
        obs_positions_arcsec=obs_positions,
        pred_positions_arcsec=best_pos,
        crit_segments=crit_segments,
        caus_segments=caus_segments,
        suptitle=f"iPTF16geu: Baseline vs {n_active_subhalos} NFW Sub-halos",
        output_file=output_plot_file_compare,
        title_left="Position Offset Comparison",
        title_mid="Magnification Comparison",
        title_right="Image Positions & Critical Curves",
        subhalo_positions=best_params,
        show_2sigma=SHOW_2SIGMA
    )
    print(f"  比较图已保存: {output_plot_file_compare}")

# 始终生成标准图
plot_paper_style_nfw(
    img_numbers=np.array([1, 2, 3, 4]),
    delta_pos_mas=best_delta_pos,
    sigma_pos_mas=obs_pos_sigma_mas,
    mu_obs=obs_magnifications,
    mu_obs_err=obs_mag_errors,
    mu_pred=best_mag,
    mu_at_obs_pred=best_mag.copy(),
    obs_positions_arcsec=obs_positions,
    pred_positions_arcsec=best_pos,
    crit_segments=crit_segments,
    caus_segments=caus_segments,
    suptitle=f"iPTF16geu: {n_active_subhalos} NFW Sub-halos Model (v2.0)",
    output_file=output_plot_file,
    title_left="Position Offset",
    title_mid="Magnification",
    title_right="Image Positions & Critical Curves",
    nfw_params=best_params,
    show_2sigma=SHOW_2SIGMA
)

# ==================== 绘制 NFW Profile ====================
if PLOT_NFW_PROFILES:
    print(f"\n生成 NFW density profiles...")
    
    def plot_nfw_profile(x_nfw, y_nfw, m_vir, c_vir, img_idx, output_file):
        """绘制单个 NFW halo 的密度剖面"""
        r_vir_kpc, r_vir_arcsec, r_s_kpc, r_s_arcsec = calculate_nfw_radii(m_vir, c_vir, lens_z)
        
        r_min = max(1e-4, r_s_arcsec / 100)
        r_max = min(NFW_PROFILE_RMAX, r_vir_arcsec * 3)
        r = np.logspace(np.log10(r_min), np.log10(r_max), 300)
        
        x_norm = r / r_s_arcsec
        rho_norm = 1.0 / (x_norm * (1 + x_norm)**2)
        
        fig, ax = plt.subplots(figsize=(7, 5.5))
        
        ax.loglog(r, rho_norm, 'b-', linewidth=2.5, label=f'NFW Profile')
        ax.axvline(r_s_arcsec, color='red', linestyle='--', linewidth=2, 
                  label=f'$r_s$ = {r_s_arcsec*1000:.2f} mas', alpha=0.8)
        ax.axvline(r_vir_arcsec, color='orange', linestyle='-.', linewidth=2, 
                  label=f'$r_{{vir}}$ = {r_vir_arcsec*1000:.2f} mas', alpha=0.8)
        
        ax.set_xlabel('Radius [arcsec]', fontsize=13, fontweight='bold')
        ax.set_ylabel('Relative Density $\\rho/\\rho_0$', fontsize=13, fontweight='bold')
        
        title = f'NFW Sub-halo {img_idx}: $M_{{vir}}$ = {m_vir:.2e} M$_\\odot$, $c_{{vir}}$ = {c_vir:.2f}'
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        
        ax.grid(True, which='both', linestyle=':', alpha=0.4)
        ax.legend(fontsize=9, loc='best', framealpha=0.95)
        
        info_text = f'Position: ({x_nfw:.4f}, {y_nfw:.4f}) arcsec\n$r_s / r_{{vir}}$ = 1 / {c_vir:.2f}'
        ax.text(0.05, 0.05, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 保存: {os.path.basename(output_file)}")
    
    for img_idx, x, y, m, c in best_params_with_img_idx:
        profile_file = os.path.join(output_dir, f"nfw{img_idx}_profile.png")
        plot_nfw_profile(x, y, m, c, img_idx, profile_file)

# 保存参数
params_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_best_params.txt')
with open(params_file, 'w') as f:
    f.write(f"# Version NFW 2.0: Optimized NFW Sub-halos Search\n")
    f.write(f"# Cosmology: FlatLambdaCDM(H0=70, Om0=0.3)\n")
    f.write(f"# active_subhalos = {active_subhalos}\n")
    f.write(f"# fine_tuning = {fine_tuning}\n")
    f.write(f"# DE_SEED = {DE_SEED}\n")
    f.write(f"# Total iterations: {iteration}\n\n")
    
    f.write(f"# NFW Sub-halo Parameters\n")
    for img_idx, x, y, m, c in best_params_with_img_idx:
        r_vir_kpc, r_vir_arcsec, r_s_kpc, r_s_arcsec = calculate_nfw_radii(m, c, lens_z)
        
        f.write(f"# NFW Sub-halo at Image {img_idx} (高精度保存)\n")
        f.write(f"x_nfw{img_idx} = {x:.10e}  # arcsec\n")
        f.write(f"y_nfw{img_idx} = {y:.10e}  # arcsec\n")
        f.write(f"m_vir{img_idx} = {m:.10e}  # Msun\n")
        f.write(f"c_vir{img_idx} = {c:.10e}\n")
        f.write(f"log10_m_vir{img_idx} = {np.log10(m):.10f}\n")
        f.write(f"r_vir{img_idx} = {r_vir_arcsec*1000:.4f} mas = {r_vir_kpc:.6f} kpc\n")
        f.write(f"r_s{img_idx} = {r_s_arcsec*1000:.4f} mas = {r_s_kpc:.6f} kpc\n\n")
    
    f.write(f"# Performance\n")
    f.write(f"chi2_base = {base_mag_chi2:.2f}\n")
    f.write(f"chi2_best = {best_mag_chi2:.2f}\n")
    f.write(f"improvement = {improvement:.1f}%\n")
    f.write(f"constraint_satisfied = {constraint_satisfied}\n")

print(f"\n✓ 完成！")
print(f"  结果图: {output_plot_file}")
print(f"  参数文件: {params_file}")
if MCMC_ENABLED:
    print(f"  MCMC链: {mcmc_chain_file}")
    print(f"  后验统计: {posterior_file}")
    print(f"  Corner图: {corner_file}")
    print(f"  轨迹图: {trace_file}")
    print(f"  质量一维后验分布图: {mass_1d_file}")

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║              步骤5: 验证 - 使用 glafic 命令行工具                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("步骤5: 验证结果（glafic 命令行 vs Python 接口）")
print("=" * 70)

# 使用智能查找 glafic 可执行文件
GLAFIC_BIN = find_glafic_bin()

if GLAFIC_BIN:
    print(f"  glafic 路径: {GLAFIC_BIN}")
else:
    print(f"  警告: 未找到 glafic 可执行文件，将跳过验证步骤")

verify_input_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_verify_input.dat')

with open(verify_input_file, 'w') as f:
    f.write("# ========================================\n")
    f.write("# Version NFW 2.0 验证文件\n")
    f.write("# ========================================\n")
    f.write(f"# 自动生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write(f"omega      {omega}\n")
    f.write(f"lambda     {lambda_cosmo}\n")
    f.write(f"weos       {weos}\n")
    f.write(f"hubble     {hubble}\n\n")
    
    verify_prefix = f"{OUTPUT_PREFIX}_verify"
    f.write(f"prefix     {verify_prefix}\n\n")
    
    f.write(f"xmin       {xmin}\n")
    f.write(f"ymin       {ymin}\n")
    f.write(f"xmax       {xmax}\n")
    f.write(f"ymax       {ymax}\n")
    f.write(f"pix_ext    {pix_ext}\n")
    f.write(f"pix_poi    {pix_poi}\n")
    f.write(f"maxlev     {maxlev}\n\n")
    
    n_lenses = 3 + n_active_subhalos
    f.write(f"startup    {n_lenses} 0 1\n\n")
    
    f.write(f"lens       sers    {best_lens_params['sers1'][2]}    ")
    f.write(f"{best_lens_params['sers1'][3]:.6e}    {best_lens_params['sers1'][4]:.6e}    {best_lens_params['sers1'][5]:.6e}    ")
    f.write(f"{best_lens_params['sers1'][6]:.6e}    {best_lens_params['sers1'][7]:.6e}    {best_lens_params['sers1'][8]:.6e}    {best_lens_params['sers1'][9]:.6e}\n")
    
    f.write(f"lens       sers    {best_lens_params['sers2'][2]}    ")
    f.write(f"{best_lens_params['sers2'][3]:.6e}    {best_lens_params['sers2'][4]:.6e}    {best_lens_params['sers2'][5]:.6e}    ")
    f.write(f"{best_lens_params['sers2'][6]:.6e}    {best_lens_params['sers2'][7]:.6e}    {best_lens_params['sers2'][8]:.6e}    {best_lens_params['sers2'][9]:.6e}\n")
    
    f.write(f"lens       {MAIN_LENS_KEY}     {best_lens_params[MAIN_LENS_KEY][2]}    ")
    f.write(f"{best_lens_params[MAIN_LENS_KEY][3]:.6e}    {best_lens_params[MAIN_LENS_KEY][4]:.6e}    {best_lens_params[MAIN_LENS_KEY][5]:.6e}    ")
    f.write(f"{best_lens_params[MAIN_LENS_KEY][6]:.6e}    {best_lens_params[MAIN_LENS_KEY][7]:.6e}    {best_lens_params[MAIN_LENS_KEY][8]:.6e}    {best_lens_params[MAIN_LENS_KEY][9]:.6e}\n\n")
    
    for x_sub, y_sub, mass_sub, conc_sub in best_params:
        f.write(f"lens       gnfw    0.2160    {mass_sub:.6e}    ")
        f.write(f"{x_sub:.6e}    {y_sub:.6e}    0.0    0.0    {conc_sub:.6e}    1.0\n")
    
    f.write(f"\npoint      {source_z}    {best_source_x:.6e}    {best_source_y:.6e}\n\n")
    f.write("end_startup\n\n")
    f.write("start_command\n\n")
    f.write("findimg\n\n")
    f.write("quit\n")

print(f"  生成验证文件: {verify_input_file}")

# 只有在找到 glafic 可执行文件时才运行验证
if GLAFIC_BIN:
    try:
        result_proc = subprocess.run(
            [GLAFIC_BIN, os.path.basename(verify_input_file)],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result_proc.returncode == 0:
            print(f"  glafic 运行成功")
        else:
            print(f"  警告: glafic 返回非零代码 {result_proc.returncode}")

    except subprocess.TimeoutExpired:
        print(f"  警告: glafic 运行超时（>60秒）")
    except FileNotFoundError:
        print(f"  警告: 找不到 glafic: {GLAFIC_BIN}")
    except Exception as e:
        print(f"  警告: glafic 运行出错: {e}")
else:
    print(f"  跳过 glafic 验证（未找到可执行文件）")

verify_output_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_verify_point.dat')

if os.path.exists(verify_output_file):
    try:
        data = np.loadtxt(verify_output_file)
        
        if len(data.shape) > 1:
            n_images_glafic = int(data[0, 0])
            print(f"  glafic 找到 {n_images_glafic} 个图像")

            if n_images_glafic in (4, 5):
                image_data_glafic = data[1:n_images_glafic + 1, :]

                if n_images_glafic == 5:
                    abs_mags = np.abs(image_data_glafic[:, 2])
                    drop_idx = int(np.argmin(abs_mags))
                    print(f"  Info: 5 images found, dropped central image "
                          f"(index {drop_idx}, |μ|={abs_mags[drop_idx]:.4f})")
                    image_data_glafic = np.delete(image_data_glafic, drop_idx, axis=0)

                glafic_positions = image_data_glafic[:, 0:2].copy()
                glafic_magnifications = np.abs(image_data_glafic[:, 2])

                glafic_positions[:, 0] += center_offset_x
                glafic_positions[:, 1] += center_offset_y

                distances = cdist(obs_positions, glafic_positions)
                row_ind, col_ind = linear_sum_assignment(distances)
                
                glafic_pos_matched = glafic_positions[col_ind[np.argsort(row_ind)]]
                glafic_mag_matched = glafic_magnifications[col_ind[np.argsort(row_ind)]]
                
                max_pos_diff = 0.0
                max_mag_diff_pct = 0.0
                
                for i in range(4):
                    py_x = best_pos[i, 0] * 1000
                    gl_x = glafic_pos_matched[i, 0] * 1000
                    py_y = best_pos[i, 1] * 1000
                    gl_y = glafic_pos_matched[i, 1] * 1000
                    
                    diff_x = abs(py_x - gl_x)
                    diff_y = abs(py_y - gl_y)
                    max_pos_diff = max(max_pos_diff, diff_x, diff_y)
                    
                    py_mag_abs = abs(best_mag[i])
                    gl_mag = glafic_mag_matched[i]
                    diff_mag_pct = abs(py_mag_abs - gl_mag) / py_mag_abs * 100 if py_mag_abs != 0 else 0
                    max_mag_diff_pct = max(max_mag_diff_pct, diff_mag_pct)
                
                print(f"\n验证结果:")
                if max_pos_diff < 0.01 and max_mag_diff_pct < 0.1:
                    print(f"  ✓ 一致性验证通过！")
                elif max_pos_diff < 1.0 and max_mag_diff_pct < 1.0:
                    print(f"  ✓ 一致性良好（小差异）")
                else:
                    print(f"  ⚠ 检测到较大差异")
                print(f"    最大位置差: {max_pos_diff:.6f} mas")
                print(f"    最大放大率差: {max_mag_diff_pct:.6f}%")
            else:
                print(f"  警告: glafic 找到 {n_images_glafic} 个图像（预期4或5个），跳过验证")
    except Exception as e:
        print(f"  读取验证输出时出错: {e}")
else:
    print(f"  警告: 验证输出文件不存在")

print("\n" + "=" * 70)
print("Version NFW 2.0 完成")
print("=" * 70)

