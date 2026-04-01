#!/usr/bin/env python3
"""
Version Pseudo-Jaffe 2.0: Flexible Pseudo-Jaffe Sub-halos Search with MCMC
基于 v_p_jaffe_1.0，新增MCMC后验采样功能

新增特性：
1. 差分进化(DE)算法找最优解
2. 以DE最优解为起点启动MCMC采样
3. 输出完整的后验概率分布
4. 生成Corner Plot可视化参数相关性
5. 计算参数误差 (median ± 1σ)
"""

import sys
sys.path.insert(0, '/home/luukiaun/glafic251018/glafic2/python')

import glafic
import random
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

os.environ['LD_LIBRARY_PATH'] = '/home/luukiaun/glafic251018/gsl-2.8/.libs:/home/luukiaun/glafic251018/fftw-3.3.10/.libs:/home/luukiaun/glafic251018/cfitsio-4.6.2/.libs'

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

def find_glafic_bin(default_path="/home/luukiaun/glafic251018/glafic2/glafic"):
    """
    智能查找 glafic 可执行文件
    
    查找顺序:
    1. 检查指定的默认路径
    2. 从 glafic Python 模块路径推断（支持外部定义的路径）
    3. 返回 None 如果都找不到
    
    参数:
        default_path: 默认的 glafic 可执行文件路径
    
    返回:
        glafic 可执行文件的路径，或 None
    """
    # 1. 首先检查默认路径
    if os.path.isfile(default_path) and os.access(default_path, os.X_OK):
        return default_path
    
    # 2. 尝试从 glafic 模块路径推断
    try:
        glafic_module_file = glafic.__file__
        if glafic_module_file:
            # glafic 模块通常在 .../glafic2/python/glafic.so 或类似路径
            # 可执行文件通常在 .../glafic2/glafic
            module_dir = os.path.dirname(os.path.abspath(glafic_module_file))
            
            # 尝试几个可能的相对位置
            possible_paths = [
                os.path.join(module_dir, '..', 'glafic'),           # ../glafic (最常见)
                os.path.join(module_dir, '..', '..', 'glafic'),     # ../../glafic
                os.path.join(module_dir, 'glafic'),                  # ./glafic
                os.path.join(module_dir, '..', 'bin', 'glafic'),    # ../bin/glafic
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
print("Version Pseudo-Jaffe 2.0: DE + MCMC Posterior Sampling")
print("=" * 70)

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║              Pseudo-Jaffe 模型说明                                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝
# Pseudo-Jaffe (截断等温椭球) 模型参数:
# - sig: 速度弥散 [km/s]
# - a: 截断半径 [arcsec] (必须 > rco)
# - rco: 核心半径 [arcsec] (必须 >= 0)
# - e: 椭率 [0, 1)
# - pa: 位置角 [degrees]

# 定义宇宙学（用于参考）
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║              Pseudo-Jaffe 质量估算函数                                 ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def calculate_jaffe_mass(sigma_km_s, a_arcsec, rco_arcsec, z_lens, z_source):
    """
    计算 Pseudo-Jaffe sub-halo 的质量估算
    
    基于透镜理论，使用速度弥散和特征尺度估算投影质量。
    
    对于 Pseudo-Jaffe 模型:
    - 收敛度参数: κ(r) ∝ (r^2 + rco^2)^(-1) - (r^2 + a^2)^(-1)
    - 投影质量与 σ^2 和角尺度成正比
    
    公式: M = (π σ^2 / G) × D_l × (a - rco) × (D_ls / D_s)
    
    参数:
        sigma_km_s: 速度弥散 [km/s]
        a_arcsec: 截断半径 [arcsec]
        rco_arcsec: 核心半径 [arcsec]
        z_lens: 透镜红移
        z_source: 源红移
    
    返回:
        mass_solar: 质量 [M_sun]
    """
    # 物理常数
    G = 4.302e-6  # 引力常数 [kpc (km/s)^2 / M_sun]
    c = 299792.458  # 光速 [km/s]
    
    # 计算角直径距离
    D_l = cosmo.angular_diameter_distance(z_lens).to(u.kpc).value  # [kpc]
    D_s = cosmo.angular_diameter_distance(z_source).to(u.kpc).value  # [kpc]
    D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).to(u.kpc).value  # [kpc]
    
    # 将角秒转换为弧度，再转换为物理尺度 [kpc]
    arcsec_to_rad = np.pi / (180.0 * 3600.0)
    a_kpc = a_arcsec * arcsec_to_rad * D_l
    rco_kpc = rco_arcsec * arcsec_to_rad * D_l
    
    # 计算临界面密度 [M_sun / kpc^2]
    # Σ_cr = c^2 / (4π G) × D_s / (D_l × D_ls)
    Sigma_cr = (c**2 / (4 * np.pi * G)) * (D_s / (D_l * D_ls))
    
    # 对于 Pseudo-Jaffe 模型，爱因斯坦半径相关的尺度参数
    # b = 4π (σ/c)^2 × (D_ls / D_s) [弧度]
    b_rad = 4 * np.pi * (sigma_km_s / c)**2 * (D_ls / D_s)
    b_kpc = b_rad * D_l
    
    # 投影质量估算 (在截断半径 a 内的质量)
    # M ≈ π × Σ_cr × b × (a - rco)
    # 这里 b 是透镜强度参数，(a - rco) 是有效尺度
    mass_solar = np.pi * Sigma_cr * b_kpc * (a_kpc - rco_kpc)
    
    return mass_solar

def format_mass(mass_solar):
    """
    格式化质量显示
    
    参数:
        mass_solar: 质量 [M_sun]
    
    返回:
        formatted_str: 格式化的字符串
    """
    if mass_solar < 1e6:
        return f"{mass_solar:.2e} M☉"
    elif mass_solar < 1e9:
        return f"{mass_solar/1e6:.2f}×10⁶ M☉"
    elif mass_solar < 1e12:
        return f"{mass_solar/1e9:.2f}×10⁹ M☉"
    else:
        return f"{mass_solar/1e12:.2f}×10¹² M☉"

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
active_subhalos = [1,3,4]        # 可修改为任意子集，对应图像编号 [1-4]

# ==================== 3. 精细调试模式 ====================
fine_tuning = False           # 是否启用独立配置

# --- 通用配置（当 fine_tuning=False 时使用） ---
SEARCH_RADIUS = 0.2          # 位置搜索半径 [arcsec]
SIG_GUESS = 10.0             # 速度弥散初始猜测 [km/s]
SIG_MIN = 0.01               # 速度弥散最小值 [km/s]
SIG_MAX = 30.0               # 速度弥散最大值 [km/s]
A_GUESS = 0.050              # 截断半径初始猜测 [arcsec]
A_MIN = 0.0001                # 截断半径最小值 [arcsec]
A_MAX = 0.300                # 截断半径最大值 [arcsec]
RCO_GUESS = 0.005            # 核心半径初始猜测 [arcsec]
RCO_MIN = 0.0000             # 核心半径最小值 [arcsec]
RCO_MAX = 0.05               # 核心半径最大值 [arcsec]

# --- 精细配置（当 fine_tuning=True 时使用） ---
# 单位说明:
#   search_radius: [arcsec]  位置搜索半径
#   sig_*: [km/s]            速度弥散
#   a_*: [arcsec]            截断半径
#   rco_*: [arcsec]          核心半径
fine_tuning_configs = {
    1: {
        'search_radius': 0.1,   # [arcsec]
        'sig_guess': 5.0,        # [km/s]
        'sig_min': 0.001,         # [km/s]
        'sig_max': 10.0,         # [km/s]
        'a_guess': 0.050,        # [arcsec]
        'a_min': 0.010,         # [arcsec]
        'a_max': 0.200,          # [arcsec]
        'rco_guess': 0.005,      # [arcsec]
        'rco_min': 0.0001,       # [arcsec]
        'rco_max': 0.050         # [arcsec]
    },
    2: {
        'search_radius': 0.4,    # [arcsec]
        'sig_guess': 8.0,       # [km/s]
        'sig_min': 0.01,         # [km/s]
        'sig_max': 50.0,         # [km/s]
        'a_guess': 0.040,        # [arcsec]
        'a_min': 0.005,          # [arcsec]
        'a_max': 0.1,            # [arcsec]
        'rco_guess': 0.004,      # [arcsec]
        'rco_min': 0.0001,       # [arcsec]
        'rco_max': 0.02          # [arcsec]
    },
    3: {
        'search_radius': 0.1,   # [arcsec]
        'sig_guess': 5.0,        # [km/s]
        'sig_min': 1.0,          # [km/s]
        'sig_max': 25.0,         # [km/s]
        'a_guess': 0.045,        # [arcsec]
        'a_min': 0.0050,         # [arcsec]
        'a_max': 0.100,          # [arcsec]
        'rco_guess': 0.005,      # [arcsec]
        'rco_min': 0.001,       # [arcsec]
        'rco_max': 0.012          # [arcsec]
    },
    4: {
        'search_radius': 0.12,   # [arcsec]
        'sig_guess': 0.01,       # [km/s]
        'sig_min': 0.1,         # [km/s]
        'sig_max': 20.0,         # [km/s]
        'a_guess': 0.035,        # [arcsec]
        'a_min': 0.0001,         # [arcsec]
        'a_max': 0.1,           # [arcsec]
        'rco_guess': 0.003,      # [arcsec]
        'rco_min': 0.00005,       # [arcsec]
        'rco_max': 0.005         # [arcsec]
    }
}

# ==================== 4. 机器学习目标函数参数 ====================
LOSS_COEF_A = 4.0            # 位置chi²的权重系数 [dimensionless]
LOSS_COEF_B = 1              # 放大率chi²的权重系数 [dimensionless]
LOSS_PENALTY_PL = 1000.0    # 位置惩罚系数 [dimensionless]

# ==================== 4.1 透镜和源参数修改配置 ====================
source_modify = False        # 是否优化source位置
lens_modify = False          # 是否优化lens参数
modify_percentage = 0.01     # 参数允许变化的百分比 [fraction, 0.01 = 1%]

# ==================== 5. 优化算法配置 ====================
DE_MAXITER = 800             # 最大迭代次数 [count]
DE_POPSIZE = 75              # 种群大小 [count]
DE_ATOL = 1e-5               # 绝对容差 [dimensionless]
DE_TOL = 1e-5                # 相对容差 [dimensionless]
DE_SEED = random.randint(1, 100000)                 # 随机种子 [integer]
DE_POLISH = True             # 是否启用局部优化抛光
DE_WORKERS = -1              # 并行核心数 [count]

# 早停机制配置
EARLY_STOPPING = True        # 是否启用早停
EARLY_STOP_PATIENCE = 60     # 容忍次数 [count]

# 回火机制配置
TEMPER_ENABLED = False       # 是否启用回火机制
TEMPER_COUNT = 3             # 连续回火次数 [count]

# ==================== 6. MCMC 配置 ====================
MCMC_ENABLED = True          # 是否启用MCMC采样
MCMC_NWALKERS = 32           # walker数量 [count]，至少是参数维度的2倍
MCMC_NSTEPS = 3000           # 采样步数 [count]
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
MCMC_SEARCH_RADIUS = 0.2     # 位置搜索半径 [arcsec]，以各图像观测位置为中心
MCMC_SIG_MIN  = 0.01        # 速度弥散下限 [km/s]
MCMC_SIG_MAX  = 100.0        # 速度弥散上限 [km/s]
MCMC_A_MIN    = 0.0001       # 截断半径下限 [arcsec]
MCMC_A_MAX    = 0.500        # 截断半径上限 [arcsec]
MCMC_RCO_MIN  = 0.000001     # 核心半径下限 [arcsec]
MCMC_RCO_MAX  = 0.100        # 核心半径上限 [arcsec]

# ==================== 7. 绘图配置 ====================
SHOW_2SIGMA = False          # 是否显示2σ横线
OUTPUT_PREFIX = "v_p_jaffe_2_0"  # 输出文件前缀
COMPARE_GRAPH = True       # 比较模式：生成 baseline vs optimized 对比图（仅在 n_active_subhalos > 0 时生效）

Draw_Graph = 1               # 绘图模式: 0=不绘制, 1=绘制
draw_interval = 10           # 绘图间隔 [iterations]
PRINT_INTERVAL = 10          # 迭代打印间隔，减少 I/O 阻塞（有改进时仍立即打印）

PLOT_JAFFE_PROFILES = True   # 是否绘制Jaffe density profiles
JAFFE_PROFILE_RMAX = 0.2     # profile绘制的最大半径 [arcsec]

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
omega = 0.3                      # Ωm [dimensionless] 物质密度参数
lambda_cosmo = 0.7               # ΩΛ [dimensionless] 暗能量密度参数
weos = -1.0                      # w [dimensionless] 暗能量状态方程参数
hubble = 0.7                     # h [dimensionless] 约化哈勃常数 H0/(100 km/s/Mpc)

# glafic 网格设置
xmin, ymin = -0.5, -0.5          # [arcsec] 网格范围
xmax, ymax = 0.5, 0.5            # [arcsec] 网格范围
pix_ext = 0.01                   # [arcsec/pixel] 扩展源像素尺寸
pix_poi = 0.2                    # [arcsec] 点源网格尺寸
maxlev = 5                       # [count] 最大细分层级

# 源参数
source_z = 0.4090                # [dimensionless] 源红移

# 默认基准透镜参数（SIE 模型，来自 SN_2Sersic_SIE/bestfit.dat）
source_x = 2.685497e-03          # [arcsec] 源位置 x
source_y = 2.443616e-02          # [arcsec] 源位置 y

# 透镜参数
lens_z = 0.2160                  # [dimensionless] 透镜红移

# lens_params 格式说明:
# sers: (id, model, z, mass[M_sun], x[arcsec], y[arcsec], re[arcsec], pa[deg], e, n)
# sie:  (id, model, z, sigma[km/s], x[arcsec], y[arcsec], e, pa[deg], 0, 0)
# anfw: (id, model, z, mass[M_sun], x[arcsec], y[arcsec], e, pa[deg], rs[arcsec], 0)
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
n_params_subhalo = n_active_subhalos * 5

n_params_source = 0
n_params_lens = 0

if source_modify:
    n_params_source = 2

if lens_modify:
    n_params_lens = 15

n_params_extra = n_params_source + n_params_lens
n_params = n_params_subhalo + n_params_extra

# 构建每个 active sub-halo 的配置
subhalo_configs = {}
for img_idx in active_subhalos:
    if fine_tuning:
        cfg = fine_tuning_configs[img_idx]
        subhalo_configs[img_idx] = {
            'search_radius': cfg['search_radius'],
            'sig_guess': cfg['sig_guess'],
            'sig_min': cfg['sig_min'],
            'sig_max': cfg['sig_max'],
            'a_guess': cfg['a_guess'],
            'a_min': cfg['a_min'],
            'a_max': cfg['a_max'],
            'rco_guess': cfg['rco_guess'],
            'rco_min': cfg['rco_min'],
            'rco_max': cfg['rco_max']
        }
    else:
        subhalo_configs[img_idx] = {
            'search_radius': SEARCH_RADIUS,
            'sig_guess': SIG_GUESS,
            'sig_min': SIG_MIN,
            'sig_max': SIG_MAX,
            'a_guess': A_GUESS,
            'a_min': A_MIN,
            'a_max': A_MAX,
            'rco_guess': RCO_GUESS,
            'rco_min': RCO_MIN,
            'rco_max': RCO_MAX
        }

if draw_interval < 1:
    draw_interval = 1

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                    创建时间戳输出目录                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

timestamp = datetime.now().strftime("%y%m%d_%H%M")
output_dir = timestamp
os.makedirs(output_dir, exist_ok=True)

print(f"\n输出目录: {output_dir}")

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║              全局变量：记录迭代历史                                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝

iteration_history = []

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                    打印配置摘要                                        ║
# ╚═══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("配置摘要")
print("=" * 70)

print(f"\n模型类型: Pseudo-Jaffe (截断等温椭球)")
print(f"  每个 sub-halo 参数: x, y, sig, a, rco (5 个)")

print(f"\n启用的 Sub-halos:")
print(f"  active_subhalos = {active_subhalos}")
print(f"  启用数量: {n_active_subhalos}")
print(f"  参数维度: {n_params}")

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
        print(f"    sig: [{MCMC_SIG_MIN}, {MCMC_SIG_MAX}] km/s")
        print(f"    a:   [{MCMC_A_MIN*1000:.2f}, {MCMC_A_MAX*1000:.0f}] mas")
        print(f"    rco: [{MCMC_RCO_MIN*1000:.3f}, {MCMC_RCO_MAX*1000:.0f}] mas")
    else:
        print(f"  先验范围: 沿用 DE 搜索范围（bounds）")
else:
    print(f"  启用: 否")

print(f"\n精细调试模式: fine_tuning = {fine_tuning}")

# 计算固定坐标轴范围
FIXED_SIG_RANGE = []
FIXED_A_RANGE = []
FIXED_RCO_RANGE = []
for img_idx in active_subhalos:
    cfg = subhalo_configs[img_idx]
    FIXED_SIG_RANGE.extend([cfg['sig_min'], cfg['sig_max']])
    FIXED_A_RANGE.extend([cfg['a_min'], cfg['a_max']])
    FIXED_RCO_RANGE.extend([cfg['rco_min'], cfg['rco_max']])

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

# ==================== 计算模型函数 ====================
def compute_model(jaffe_params_list, verbose=False, src_x=None, src_y=None, lens_params_dict=None):
    """计算多 Pseudo-Jaffe sub-halo 模型"""
    use_src_x = src_x if src_x is not None else source_x
    use_src_y = src_y if src_y is not None else source_y
    use_lens_params = lens_params_dict if lens_params_dict is not None else lens_params
    
    glafic.init(omega, lambda_cosmo, weos, hubble, f'temp_{OUTPUT_PREFIX}',
                xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev, verb=0)
    
    n_subhalos = len(jaffe_params_list)
    glafic.startup_setnum(3 + n_subhalos, 0, 1)
    
    glafic.set_lens(*use_lens_params['sers1'])
    glafic.set_lens(*use_lens_params['sers2'])
    glafic.set_lens(*use_lens_params[MAIN_LENS_KEY])

    for i, (x_sub, y_sub, sig_sub, a_sub, rco_sub) in enumerate(jaffe_params_list):
        glafic.set_lens(4 + i, 'jaffe', 0.2160, sig_sub, x_sub, y_sub, 0.0, 0.0, a_sub, rco_sub)
    
    glafic.set_point(1, source_z, use_src_x, use_src_y)
    glafic.model_init(verb=0)
    
    result = glafic.point_solve(source_z, use_src_x, use_src_y, verb=0)

    n_images = len(result)
    if n_images == 5:
        # anfw 等模型会额外产生一个低放大率中心像，将其剔除
        abs_mags = [abs(img_data[2]) for img_data in result]
        drop_idx = int(np.argmin(abs_mags))
        result = [img_data for i, img_data in enumerate(result) if i != drop_idx]
        if verbose:
            print(f"  Info: 5 images found, dropped central image "
                  f"(index {drop_idx}, |μ|={abs_mags[drop_idx]:.4f})")
    elif n_images != 4:
        if verbose:
            print(f"  Warning: Found {n_images} images (expected 4)")
        glafic.quit()
        return None, None, None, 1e10

    pred_positions = []
    pred_magnifications = []
    for img_data in result:
        x, y = img_data[0], img_data[1]
        mag = img_data[2]
        pred_positions.append([x, y])
        pred_magnifications.append(mag)

    pred_positions = np.array(pred_positions)
    pred_magnifications = np.array(pred_magnifications)

    pred_positions[:, 0] += center_offset_x
    pred_positions[:, 1] += center_offset_y

    distances = cdist(obs_positions, pred_positions)
    row_ind, col_ind = linear_sum_assignment(distances)

    pred_pos_matched = pred_positions[col_ind[np.argsort(row_ind)]]
    pred_mag_matched = pred_magnifications[col_ind[np.argsort(row_ind)]]

    delta_pos = []
    for i in range(len(obs_positions)):
        dx = (pred_pos_matched[i, 0] - obs_positions[i, 0]) * 1000
        dy = (pred_pos_matched[i, 1] - obs_positions[i, 1]) * 1000
        delta = np.sqrt(dx**2 + dy**2)
        delta_pos.append(delta)
    delta_pos = np.array(delta_pos)
    
    mag_residuals = (pred_mag_matched - obs_magnifications) / obs_mag_errors
    mag_chi2 = np.sum(mag_residuals**2)
    
    glafic.quit()
    
    return pred_pos_matched, pred_mag_matched, delta_pos, mag_chi2

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
    jaffe_list = []
    for i in range(n_active_subhalos):
        x = params[i*5]
        y = params[i*5 + 1]
        sig = params[i*5 + 2]
        a = params[i*5 + 3]
        rco = params[i*5 + 4]
        
        if a <= rco:
            return 1e15
        
        jaffe_list.append((x, y, sig, a, rco))
    
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

    pos, mag, delta_pos, mag_chi2 = compute_model(jaffe_list, src_x=src_x_opt, src_y=src_y_opt,
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
    
    bounds.append((x_center - cfg['search_radius'], x_center + cfg['search_radius']))
    bounds.append((y_center - cfg['search_radius'], y_center + cfg['search_radius']))
    bounds.append((cfg['sig_min'], cfg['sig_max']))
    bounds.append((cfg['a_min'], cfg['a_max']))
    bounds.append((cfg['rco_min'], cfg['rco_max']))

if source_modify:
    bounds.append((source_x * (1 - modify_percentage), source_x * (1 + modify_percentage)))
    bounds.append((source_y * (1 - modify_percentage), source_y * (1 + modify_percentage)))

if lens_modify:
    bounds.append((lens_params['sers1'][3] * (1 - modify_percentage), lens_params['sers1'][3] * (1 + modify_percentage)))
    bounds.append((lens_params['sers1'][4] * (1 - modify_percentage), lens_params['sers1'][4] * (1 + modify_percentage)))
    bounds.append((lens_params['sers1'][5] * (1 - modify_percentage), lens_params['sers1'][5] * (1 + modify_percentage)))
    bounds.append((lens_params['sers1'][6] * (1 - modify_percentage), lens_params['sers1'][6] * (1 + modify_percentage)))
    bounds.append((lens_params['sers1'][7] * (1 - modify_percentage), lens_params['sers1'][7] * (1 + modify_percentage)))
    
    bounds.append((lens_params['sers2'][3] * (1 - modify_percentage), lens_params['sers2'][3] * (1 + modify_percentage)))
    bounds.append((lens_params['sers2'][4] * (1 - modify_percentage), lens_params['sers2'][4] * (1 + modify_percentage)))
    bounds.append((lens_params['sers2'][5] * (1 - modify_percentage), lens_params['sers2'][5] * (1 + modify_percentage)))
    bounds.append((lens_params['sers2'][6] * (1 - modify_percentage), lens_params['sers2'][6] * (1 + modify_percentage)))
    bounds.append((lens_params['sers2'][7] * (1 - modify_percentage), lens_params['sers2'][7] * (1 + modify_percentage)))
    
    # 主透镜参数的 bounds（sie 或 anfw）
    for _i in [3, 4, 5, 6, 7]:
        bounds.append((lens_params[MAIN_LENS_KEY][_i] * (1 - modify_percentage),
                       lens_params[MAIN_LENS_KEY][_i] * (1 + modify_percentage)))

print(f"\n搜索参数空间（{n_params}维）:")
for i, img_idx in enumerate(active_subhalos):
    cfg = subhalo_configs[img_idx]
    x_center = obs_positions[img_idx-1, 0]
    y_center = obs_positions[img_idx-1, 1]
    
    print(f"  Pseudo-Jaffe Sub-halo {img_idx}:")
    print(f"    x=[{x_center-cfg['search_radius']:.4f}, {x_center+cfg['search_radius']:.4f}]")
    print(f"    y=[{y_center-cfg['search_radius']:.4f}, {y_center+cfg['search_radius']:.4f}]")
    print(f"    sig=[{cfg['sig_min']:.1f}, {cfg['sig_max']:.1f}] km/s")
    print(f"    a=[{cfg['a_min']*1000:.1f}, {cfg['a_max']*1000:.1f}] mas")
    print(f"    rco=[{cfg['rco_min']*1000:.1f}, {cfg['rco_max']*1000:.1f}] mas")

print(f"\n开始优化...")

# ==================== 迭代绘图函数 ====================
def plot_iteration_population(population, iteration_num, output_dir, bounds):
    """绘制种群参数分布"""
    if Draw_Graph == 0:
        return
    
    # 如果没有 subhalo，跳过绘图
    if n_active_subhalos == 0:
        return
    
    if iteration_num % draw_interval != 0 and iteration_num != 0:
        return
    
    # 反归一化（DE内部使用归一化参数0-1）
    if population.max() <= 1.0 and population.min() >= 0.0:
        population_denorm = np.zeros_like(population)
        for i in range(population.shape[1]):
            lower, upper = bounds[i]
            population_denorm[:, i] = population[:, i] * (upper - lower) + lower
        population = population_denorm
    
    # 提取参数
    n_halos = n_active_subhalos
    sig = []
    a = []
    rco = []
    for i in range(n_halos):
        sig.append(population[:, i*5 + 2])
        a.append(population[:, i*5 + 3])
        rco.append(population[:, i*5 + 4])
    
    # 绘制图表
    fig = plt.figure(figsize=(5*n_halos, 6))
    labels = [f'Jaffe {active_subhalos[i]}' for i in range(n_halos)]
    
    for i in range(n_halos):
        # 速度弥散分布
        ax = plt.subplot(3, n_halos, i + 1)
        ax.hist(sig[i], bins=20, alpha=0.6, color='blue')
        ax.set_xlabel(f'{labels[i]} sig [km/s]', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.3)
        
        # 截断半径分布
        ax = plt.subplot(3, n_halos, n_halos + i + 1)
        ax.hist(a[i] * 1000, bins=20, alpha=0.6, color='green')
        ax.set_xlabel(f'{labels[i]} a [mas]', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.3)
        
        # 核心半径分布
        ax = plt.subplot(3, n_halos, 2*n_halos + i + 1)
        ax.hist(rco[i] * 1000, bins=20, alpha=0.6, color='red')
        ax.set_xlabel(f'{labels[i]} rco [mas]', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.3)
    
    plt.suptitle(f'Iteration {iteration_num}: Pseudo-Jaffe Parameters', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = os.path.join(output_dir, f'iteration_{iteration_num:04d}.png')
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close()

# 使用 DifferentialEvolutionSolver 以获取种群
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
import scipy

print(f"  Scipy版本: {scipy.__version__}")

# 使用try-except处理不同scipy版本的参数兼容性
# 不同scipy版本可能使用不同的随机种子参数名
try:
    # 尝试使用seed参数（scipy某些版本）
    solver = DifferentialEvolutionSolver(
        objective_function,
        bounds,
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        atol=DE_ATOL,
        tol=DE_TOL,
        seed=DE_SEED,
        polish=DE_POLISH,
        disp=False,
        workers=DE_WORKERS,
        updating='deferred'
    )
    print(f"  使用seed参数初始化成功")
except TypeError:
    try:
        solver = DifferentialEvolutionSolver(
            objective_function,
            bounds,
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            atol=DE_ATOL,
            tol=DE_TOL,
            random_state=DE_SEED,
            polish=DE_POLISH,
            disp=False,
            workers=DE_WORKERS,
            updating='deferred'
        )
        print(f"  使用random_state参数初始化成功")
    except TypeError:
        print(f"  scipy不支持seed/random_state参数，使用numpy.random.seed()替代")
        np.random.seed(DE_SEED)
        solver = DifferentialEvolutionSolver(
            objective_function,
            bounds,
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            atol=DE_ATOL,
            tol=DE_TOL,
            polish=DE_POLISH,
            disp=False,
            workers=DE_WORKERS,
            updating='deferred'
        )

# 记录初始种群
if Draw_Graph:
    plot_iteration_population(solver.population.copy(), 0, output_dir, bounds)

# 执行优化迭代
iteration = 1
previous_best_energy = np.min(solver.population_energies)
best_ever_energy = previous_best_energy
converged_count = 0

print(f"\n迭代 0: 初始最佳 = {previous_best_energy:.6f}")

while True:
    try:
        next_gen = solver.__next__()
    except StopIteration:
        print(f"\n✓ 优化收敛！")
        break

    current_best_energy = np.min(solver.population_energies)

    # 按间隔打印，或能量改进时立即打印
    if iteration % PRINT_INTERVAL == 0 or current_best_energy < best_ever_energy:
        if current_best_energy < best_ever_energy:
            print(f"迭代 {iteration}: 最佳 = {current_best_energy:.6f} ★ (改进)")
            best_ever_energy = current_best_energy
        else:
            print(f"迭代 {iteration}: 最佳 = {current_best_energy:.6f}")

    # 按间隔绘图（只在需要时才复制种群）
    if Draw_Graph and iteration % draw_interval == 0:
        plot_iteration_population(solver.population.copy(), iteration, output_dir, bounds)

    # 早停检查
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

print(f"\n总迭代次数: {iteration}  最终最佳值: {best_ever_energy:.6f}")

# 获取最终结果
result = solver.x
final_fun = np.min(solver.population_energies)

# 解析结果
best_params = []
best_params_with_img_idx = []
for i, img_idx in enumerate(active_subhalos):
    x = result[i*5]
    y = result[i*5 + 1]
    sig = result[i*5 + 2]
    a = result[i*5 + 3]
    rco = result[i*5 + 4]
    best_params.append((x, y, sig, a, rco))
    best_params_with_img_idx.append((img_idx, x, y, sig, a, rco))

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
print("步骤3: 分析DE最佳结果")
print("=" * 70)

best_pos, best_mag, best_delta_pos, best_mag_chi2 = compute_model(
    best_params, verbose=True, src_x=best_source_x, src_y=best_source_y,
    lens_params_dict=best_lens_params if lens_modify else None
)

if best_pos is None:
    print("\n✗ 优化失败：最佳参数无法产生4个图像！")
    sys.exit(1)

print(f"\nDE 最佳 {n_active_subhalos} 个 Pseudo-Jaffe sub-halo 参数:")

# 计算每个 sub-halo 的质量估算
subhalo_masses = []
for img_idx, x, y, sig, a, rco in best_params_with_img_idx:
    mass = calculate_jaffe_mass(sig, a, rco, lens_z, source_z)
    subhalo_masses.append(mass)
    print(f"  Pseudo-Jaffe Sub-halo at Image {img_idx}:")
    print(f"    位置: ({x:.6f}, {y:.6f}) arcsec")
    print(f"    速度弥散: sig = {sig:.2f} km/s")
    print(f"    截断半径: a = {a*1000:.2f} mas")
    print(f"    核心半径: rco = {rco*1000:.2f} mas")
    print(f"    质量估算: M = {format_mass(mass)} ({mass:.3e} M_sun)")

print(f"\n改善效果:")
print(f"  基准chi2: {base_mag_chi2:.2f}")
print(f"  最佳chi2: {best_mag_chi2:.2f}")
improvement = (base_mag_chi2 - best_mag_chi2) / base_mag_chi2 * 100
print(f"  改善: {improvement:.1f}%")

max_pos_deviation_mas = CONSTRAINT_SIGMA * obs_pos_sigma_mas
constraint_satisfied = True
for i in range(4):
    if best_delta_pos[i] > max_pos_deviation_mas[i]:
        constraint_satisfied = False

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
        likelihood ∝ exp(-χ²/2)，所以 log(likelihood) = -χ²/2

        先验：只保留物理必要约束，不使用 DE 搜索域边界，
        让后验自由向外延伸，避免边界堆积效应。
        """
        if MCMC_CUSTOM_RANGE:
            # 使用自定义先验范围（MCMC_SEARCH_RADIUS / MCMC_SIG_* / MCMC_A_* 等变量）
            for i in range(n_active_subhalos):
                img_idx = active_subhalos[i]
                x_ctr   = obs_positions[img_idx - 1, 0]
                y_ctr   = obs_positions[img_idx - 1, 1]
                x   = params[i*5]
                y   = params[i*5 + 1]
                sig = params[i*5 + 2]
                a   = params[i*5 + 3]
                rco = params[i*5 + 4]
                if abs(x - x_ctr) > MCMC_SEARCH_RADIUS or abs(y - y_ctr) > MCMC_SEARCH_RADIUS:
                    return -np.inf
                if not (MCMC_SIG_MIN <= sig <= MCMC_SIG_MAX):
                    return -np.inf
                if not (MCMC_A_MIN <= a <= MCMC_A_MAX):
                    return -np.inf
                if not (MCMC_RCO_MIN <= rco <= MCMC_RCO_MAX):
                    return -np.inf
                if a <= rco:
                    return -np.inf
        else:
            # 使用 DE 搜索范围（bounds 列表），与 DE 优化保持完全一致
            for i, (low, high) in enumerate(bounds):
                if not (low <= params[i] <= high):
                    return -np.inf
            # 物理约束：Jaffe 模型要求 a > rco
            for i in range(n_active_subhalos):
                if params[i*5 + 3] <= params[i*5 + 4]:
                    return -np.inf

        # 计算损失函数（即似然）
        loss = objective_function(params)

        if loss >= 1e10:
            return -np.inf

        return -0.5 * loss
    
    # 初始化 walkers：在DE最优解附近添加小扰动
    ndim = n_params
    best_result = result
    
    print(f"\n初始化MCMC采样器:")
    print(f"  参数维度: {ndim}")
    print(f"  Walkers: {MCMC_NWALKERS}")
    print(f"  采样步数: {MCMC_NSTEPS}")
    print(f"  Burn-in: {MCMC_BURNIN}")
    
    # 确保walker数量至少是参数维度的2倍
    if MCMC_NWALKERS < 2 * ndim:
        MCMC_NWALKERS = 2 * ndim + 2
        print(f"  [调整] Walkers 增加到 {MCMC_NWALKERS} (至少为参数维度的2倍)")
    
    # 在最优解周围初始化walkers
    initial_positions = []
    for _ in range(MCMC_NWALKERS):
        perturbation = np.array([
            np.random.normal(0, MCMC_PERTURBATION * (bounds[i][1] - bounds[i][0]))
            for i in range(ndim)
        ])
        new_pos = best_result + perturbation
        # 确保在边界内
        new_pos = np.clip(new_pos, [b[0] for b in bounds], [b[1] for b in bounds])
        # 确保 a > rco
        for i in range(n_active_subhalos):
            a_idx = i*5 + 3
            rco_idx = i*5 + 4
            if new_pos[a_idx] <= new_pos[rco_idx]:
                new_pos[a_idx] = new_pos[rco_idx] * 1.5
        initial_positions.append(new_pos)
    
    initial_positions = np.array(initial_positions)
    
    # 创建采样器（支持多进程并行）
    # 处理 -1 表示使用全部CPU
    if MCMC_WORKERS == -1:
        mcmc_workers_actual = os.cpu_count() or 1
    else:
        mcmc_workers_actual = MCMC_WORKERS
    
    print(f"  并行核心数: {mcmc_workers_actual}" + (" (全部CPU)" if MCMC_WORKERS == -1 else ""))
    
    if mcmc_workers_actual > 1:
        # 使用多进程并行
        from multiprocessing import Pool
        
        # 注意：glafic可能不是进程安全的，如果遇到问题请设置 MCMC_WORKERS = 1
        print(f"  ⚠ 多进程模式：如果出错请设置 MCMC_WORKERS = 1")
        
        with Pool(mcmc_workers_actual) as pool:
            sampler = emcee.EnsembleSampler(MCMC_NWALKERS, ndim, log_probability, pool=pool)
            
            # 运行MCMC
            print(f"\n开始MCMC采样（{mcmc_workers_actual}核并行）...")
            
            if MCMC_PROGRESS:
                for sample in tqdm(sampler.sample(initial_positions, iterations=MCMC_NSTEPS), 
                                  total=MCMC_NSTEPS, desc="MCMC采样"):
                    pass
            else:
                sampler.run_mcmc(initial_positions, MCMC_NSTEPS, progress=False)
            
            # 在pool关闭前获取所有需要的数据
            samples = sampler.get_chain(discard=MCMC_BURNIN, thin=MCMC_THIN, flat=True)
            chain = sampler.get_chain()  # 用于轨迹图
    else:
        # 串行模式
        sampler = emcee.EnsembleSampler(MCMC_NWALKERS, ndim, log_probability)
        
        # 运行MCMC
        print(f"\n开始MCMC采样（串行模式）...")
        
        if MCMC_PROGRESS:
            for sample in tqdm(sampler.sample(initial_positions, iterations=MCMC_NSTEPS), 
                              total=MCMC_NSTEPS, desc="MCMC采样"):
                pass
        else:
            sampler.run_mcmc(initial_positions, MCMC_NSTEPS, progress=False)
        
        # 获取采样结果
        samples = sampler.get_chain(discard=MCMC_BURNIN, thin=MCMC_THIN, flat=True)
        chain = sampler.get_chain()  # 用于轨迹图
    print(f"\n采样完成:")
    print(f"  总样本数: {MCMC_NWALKERS * MCMC_NSTEPS}")
    print(f"  有效样本数（去除burn-in）: {len(samples)}")
    
    # 保存MCMC链
    mcmc_chain_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_mcmc_chain.dat')
    
    # 构建参数名列表
    param_names = []
    for img_idx in active_subhalos:
        param_names.extend([f'x_{img_idx}', f'y_{img_idx}', f'sig_{img_idx}', 
                            f'a_{img_idx}', f'rco_{img_idx}'])
    if source_modify:
        param_names.extend(['src_x', 'src_y'])
    if lens_modify:
        param_names.extend(['sers1_m', 'sers1_x', 'sers1_y', 'sers1_re', 'sers1_pa',
                            'sers2_m', 'sers2_x', 'sers2_y', 'sers2_re', 'sers2_pa',
                            'sie_sig', 'sie_x', 'sie_y', 'sie_e', 'sie_pa'])
    
    header = ' '.join(param_names)
    np.savetxt(mcmc_chain_file, samples, header=header)
    print(f"  ✓ MCMC链已保存: {mcmc_chain_file}")
    
    # ==================== 计算参数统计 ====================
    print(f"\n" + "=" * 70)
    print("MCMC 后验分布统计")
    print("=" * 70)
    
    posterior_stats = {}
    
    print(f"\n参数后验分布 (median +upper_1σ -lower_1σ):")
    for i, name in enumerate(param_names):
        median = np.median(samples[:, i])
        lower = np.percentile(samples[:, i], 16)
        upper = np.percentile(samples[:, i], 84)
        
        posterior_stats[name] = {
            'median': median,
            'lower_1sigma': lower,
            'upper_1sigma': upper,
            'error_plus': upper - median,
            'error_minus': median - lower
        }
        
        # 根据参数类型选择合适的输出格式
        if 'sig' in name and 'sie' not in name:
            print(f"  {name}: {median:.2f} +{upper-median:.2f} -{median-lower:.2f} km/s")
        elif name.startswith('x_') or name.startswith('y_'):
            print(f"  {name}: {median:.6f} +{(upper-median)*1000:.3f} -{(median-lower)*1000:.3f} mas")
        elif name.startswith('a_') or name.startswith('rco_'):
            print(f"  {name}: {median*1000:.3f} +{(upper-median)*1000:.3f} -{(median-lower)*1000:.3f} mas")
        else:
            print(f"  {name}: {median:.6e} +{upper-median:.3e} -{median-lower:.3e}")
    
    # ==================== 绘制 Corner Plot ====================
    if n_params_subhalo > 0:
        print(f"\n生成 Corner Plot...")
        
        # 简化标签用于corner plot
        corner_labels = []
        for img_idx in active_subhalos:
            corner_labels.extend([f'$x_{img_idx}$', f'$y_{img_idx}$', f'$\\sigma_{img_idx}$', 
                                  f'$a_{img_idx}$', f'$r_{{co,{img_idx}}}$'])
        
        _n_corner = n_params_subhalo
        _corner_samples = samples[:, :_n_corner]
        _N_corner = len(_corner_samples)

        fig = corner.corner(
            _corner_samples,
            labels=corner_labels[:_n_corner],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt='.4f',
            truths=best_result[:_n_corner],
            truth_color='red',
            hist_kwargs={'alpha': 0.75},
        )
        # 对角线面板：将 patch 高度转为百分比，并显示纵轴
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
    else:
        print(f"\n跳过 Corner Plot（无 subhalo 参数）")
    
    # ==================== 绘制MCMC链的轨迹图 ====================
    if n_params_subhalo > 0:
        print(f"\n生成 MCMC 链轨迹图...")
        # chain 已在前面获取: shape = (nsteps, nwalkers, ndim)
        
        # 确保 corner_labels 已定义
        if 'corner_labels' not in locals():
            corner_labels = []
            for img_idx in active_subhalos:
                corner_labels.extend([f'$x_{img_idx}$', f'$y_{img_idx}$', f'$\\sigma_{img_idx}$', 
                                      f'$a_{img_idx}$', f'$r_{{co,{img_idx}}}$'])
        
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
        print(f"\n跳过轨迹图（无 subhalo 参数）")
    
    # ==================== 计算质量的后验分布 ====================
    if n_active_subhalos > 0:
        print(f"\n计算质量的后验分布...")
    
    mass_posterior_stats = {}
    for i, img_idx in enumerate(active_subhalos):
        # 从样本中提取每个 sub-halo 的参数
        sig_samples = samples[:, i*5 + 2]
        a_samples = samples[:, i*5 + 3]
        rco_samples = samples[:, i*5 + 4]
        
        # 计算每个样本的质量
        mass_samples = np.array([
            calculate_jaffe_mass(sig, a, rco, lens_z, source_z)
            for sig, a, rco in zip(sig_samples, a_samples, rco_samples)
        ])
        
        # 计算质量的统计量
        median = np.median(mass_samples)
        lower = np.percentile(mass_samples, 16)
        upper = np.percentile(mass_samples, 84)
        
        mass_name = f'mass_{img_idx}'
        mass_posterior_stats[mass_name] = {
            'median': median,
            'lower_1sigma': lower,
            'upper_1sigma': upper,
            'error_plus': upper - median,
            'error_minus': median - lower,
            'samples': mass_samples
        }
        
        print(f"  mass_{img_idx}: {median:.3e} +{upper-median:.3e} -{median-lower:.3e} M_sun")
    
        # ==================== 绘制质量一维后验分布图 ====================
        print(f"\n生成质量一维后验分布图（logM）...")
        
        from scipy.stats import gaussian_kde
        
        n_mass_halos = len(active_subhalos)
        fig_mass, axes_mass = plt.subplots(1, n_mass_halos, figsize=(5 * n_mass_halos, 4))
        
        if n_mass_halos == 1:
            axes_mass = [axes_mass]
        
        for i, img_idx in enumerate(active_subhalos):
            mass_name = f'mass_{img_idx}'
            mass_samples_i = mass_posterior_stats[mass_name]['samples']
            
            # 过滤掉非正质量的样本
            valid_mask = mass_samples_i > 0
            log_mass_samples = np.log10(mass_samples_i[valid_mask])
            
            # DE 最优解质量
            log_de_mass = np.log10(subhalo_masses[i])
            
            # KDE 平滑
            kde = gaussian_kde(log_mass_samples, bw_method='scott')
            x_min_log = min(log_mass_samples.min() - 0.3, log_de_mass - 0.3)
            x_max_log = max(log_mass_samples.max() + 0.3, log_de_mass + 0.3)
            x_grid = np.linspace(x_min_log, x_max_log, 500)
            y_kde = kde(x_grid)
            
            ax = axes_mass[i]
            ax.plot(x_grid, y_kde, color='steelblue', lw=2)
            ax.fill_between(x_grid, y_kde, alpha=0.25, color='steelblue')
            
            # 标记中位数和 1σ 区间
            log_median = np.log10(mass_posterior_stats[mass_name]['median'])
            log_lower  = np.log10(mass_posterior_stats[mass_name]['lower_1sigma'])
            log_upper  = np.log10(mass_posterior_stats[mass_name]['upper_1sigma'])
            
            ax.axvline(log_median, color='steelblue', lw=1.5, ls='--', label=f'median = {log_median:.2f}')
            ax.axvspan(log_lower, log_upper, alpha=0.15, color='steelblue', label=r'1$\sigma$')
            ax.axvline(log_de_mass, color='tomato', lw=2, ls='-', label=f'DE best = {log_de_mass:.2f}')
            
            ax.set_xlabel(r'$\log_{10}(M / M_\odot)$', fontsize=13)
            ax.set_ylabel('Posterior density', fontsize=13)
            ax.set_title(f'Sub-halo {img_idx} mass posterior', fontsize=12)
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
    
    # ==================== 保存完整的后验统计 ====================
    posterior_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_posterior.txt')
    with open(posterior_file, 'w') as f:
        f.write("# ============================================================\n")
        f.write("# MCMC Posterior Distribution Summary\n")
        f.write("# Version Pseudo-Jaffe 2.0\n")
        f.write("# ============================================================\n\n")
        f.write(f"# Configuration:\n")
        f.write(f"#   Walkers: {MCMC_NWALKERS}\n")
        f.write(f"#   Steps: {MCMC_NSTEPS}\n")
        f.write(f"#   Burn-in: {MCMC_BURNIN}\n")
        f.write(f"#   Thin: {MCMC_THIN}\n")
        f.write(f"#   Effective samples: {len(samples)}\n\n")
        
        f.write(f"# Cosmology for mass estimation:\n")
        f.write(f"#   z_lens = {lens_z}\n")
        f.write(f"#   z_source = {source_z}\n")
        f.write(f"#   D_l = {cosmo.angular_diameter_distance(lens_z).to(u.kpc).value:.2f} kpc\n")
        f.write(f"#   D_s = {cosmo.angular_diameter_distance(source_z).to(u.kpc).value:.2f} kpc\n")
        f.write(f"#   D_ls = {cosmo.angular_diameter_distance_z1z2(lens_z, source_z).to(u.kpc).value:.2f} kpc\n\n")
        
        f.write("# Parameter statistics (median, 16th percentile, 84th percentile)\n")
        f.write("# parameter  median  16%_lower  84%_upper  error_plus  error_minus\n\n")
        
        for i, name in enumerate(param_names):
            stats = posterior_stats[name]
            f.write(f"{name}  {stats['median']:.10e}  {stats['lower_1sigma']:.10e}  ")
            f.write(f"{stats['upper_1sigma']:.10e}  {stats['error_plus']:.10e}  {stats['error_minus']:.10e}\n")
        
        f.write("\n# Mass posterior statistics (derived from MCMC samples)\n")
        for img_idx in active_subhalos:
            mass_name = f'mass_{img_idx}'
            stats = mass_posterior_stats[mass_name]
            f.write(f"{mass_name}  {stats['median']:.10e}  {stats['lower_1sigma']:.10e}  ")
            f.write(f"{stats['upper_1sigma']:.10e}  {stats['error_plus']:.10e}  {stats['error_minus']:.10e}\n")
        
        f.write("\n# ============================================================\n")
        f.write("# Summary for paper (Pseudo-Jaffe parameters with mass):\n")
        f.write("# ============================================================\n\n")
        
        for img_idx in active_subhalos:
            f.write(f"# Pseudo-Jaffe Sub-halo at Image {img_idx}:\n")
            
            x_name = f'x_{img_idx}'
            y_name = f'y_{img_idx}'
            sig_name = f'sig_{img_idx}'
            a_name = f'a_{img_idx}'
            rco_name = f'rco_{img_idx}'
            mass_name = f'mass_{img_idx}'
            
            x_stats = posterior_stats[x_name]
            y_stats = posterior_stats[y_name]
            sig_stats = posterior_stats[sig_name]
            a_stats = posterior_stats[a_name]
            rco_stats = posterior_stats[rco_name]
            mass_stats = mass_posterior_stats[mass_name]
            
            f.write(f"#   x = {x_stats['median']:.6f} +{x_stats['error_plus']*1000:.3f} -{x_stats['error_minus']*1000:.3f} mas\n")
            f.write(f"#   y = {y_stats['median']:.6f} +{y_stats['error_plus']*1000:.3f} -{y_stats['error_minus']*1000:.3f} mas\n")
            f.write(f"#   sig = {sig_stats['median']:.2f} +{sig_stats['error_plus']:.2f} -{sig_stats['error_minus']:.2f} km/s\n")
            f.write(f"#   a = {a_stats['median']*1000:.3f} +{a_stats['error_plus']*1000:.3f} -{a_stats['error_minus']*1000:.3f} mas\n")
            f.write(f"#   rco = {rco_stats['median']*1000:.3f} +{rco_stats['error_plus']*1000:.3f} -{rco_stats['error_minus']*1000:.3f} mas\n")
            f.write(f"#   mass = {mass_stats['median']:.3e} +{mass_stats['error_plus']:.3e} -{mass_stats['error_minus']:.3e} M_sun\n\n")
    
    print(f"  ✓ 后验统计已保存: {posterior_file}")

# ==================== 生成最终图表 ====================
print("\n" + "=" * 70)
print("步骤5: 生成最终图表")
print("=" * 70)

glafic.init(omega, lambda_cosmo, weos, hubble, f'temp_{OUTPUT_PREFIX}_best',
            xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev, verb=0)

glafic.startup_setnum(3 + n_active_subhalos, 0, 1)
glafic.set_lens(*best_lens_params['sers1'])
glafic.set_lens(*best_lens_params['sers2'])
glafic.set_lens(*best_lens_params[MAIN_LENS_KEY])
for i, (x_sub, y_sub, sig_sub, a_sub, rco_sub) in enumerate(best_params):
    glafic.set_lens(4 + i, 'jaffe', 0.2160, sig_sub, x_sub, y_sub, 0.0, 0.0, a_sub, rco_sub)

glafic.set_point(1, source_z, best_source_x, best_source_y)
glafic.model_init(verb=0)
glafic.writecrit(source_z)

crit_file = f'temp_{OUTPUT_PREFIX}_best_crit.dat'
crit_segments, caus_segments = read_critical_curves(crit_file)

glafic.quit()

output_plot_file = os.path.join(output_dir, f"result_{OUTPUT_PREFIX}.png")

# 构建带有质量信息的参数列表 (x, y, sig, a, rco, mass)
best_params_with_mass = []
for i, (x, y, sig, a, rco) in enumerate(best_params):
    mass = subhalo_masses[i]
    best_params_with_mass.append((x, y, sig, a, rco, mass))

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
        suptitle=f"iPTF16geu: Baseline vs {n_active_subhalos} Pseudo-Jaffe Sub-halos",
        output_file=output_plot_file_compare,
        title_left="Position Offset Comparison",
        title_mid="Magnification Comparison",
        title_right="Image Positions & Critical Curves",
        subhalo_positions=best_params_with_mass,
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
    suptitle=f"iPTF16geu: {n_active_subhalos} Pseudo-Jaffe Sub-halos (DE + MCMC)",
    output_file=output_plot_file,
    title_left="Position Offset",
    title_mid="Magnification",
    title_right="Image Positions & Critical Curves",
    nfw_params=best_params_with_mass,
    show_2sigma=SHOW_2SIGMA
)

# ==================== 保存DE最佳参数 ====================
params_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_best_params.txt')
with open(params_file, 'w') as f:
    f.write(f"# Version Pseudo-Jaffe 2.0: DE + MCMC\n")
    f.write(f"# Configuration:\n")
    f.write(f"#   active_subhalos = {active_subhalos}\n")
    f.write(f"#   fine_tuning = {fine_tuning}\n")
    f.write(f"#   DE_SEED = {DE_SEED}\n")
    f.write(f"#   MCMC_ENABLED = {MCMC_ENABLED}\n")
    f.write(f"#   MCMC_NSTEPS = {MCMC_NSTEPS}\n")
    f.write(f"#   MCMC_BURNIN = {MCMC_BURNIN}\n\n")
    
    f.write(f"# Cosmology for mass estimation:\n")
    f.write(f"#   z_lens = {lens_z}\n")
    f.write(f"#   z_source = {source_z}\n")
    f.write(f"#   D_l = {cosmo.angular_diameter_distance(lens_z).to(u.kpc).value:.2f} kpc\n")
    f.write(f"#   D_s = {cosmo.angular_diameter_distance(source_z).to(u.kpc).value:.2f} kpc\n")
    f.write(f"#   D_ls = {cosmo.angular_diameter_distance_z1z2(lens_z, source_z).to(u.kpc).value:.2f} kpc\n\n")
    
    f.write(f"# DE Best Pseudo-Jaffe Sub-halo Parameters (高精度保存，避免放大率敏感性问题)\n")
    total_mass = 0.0
    for i, (img_idx, x, y, sig, a, rco) in enumerate(best_params_with_img_idx):
        mass = subhalo_masses[i]
        total_mass += mass
        f.write(f"# Pseudo-Jaffe Sub-halo at Image {img_idx}\n")
        f.write(f"x_jaffe{img_idx} = {x:.10e}  # arcsec\n")
        f.write(f"y_jaffe{img_idx} = {y:.10e}  # arcsec\n")
        f.write(f"sig{img_idx} = {sig:.10e}  # km/s\n")
        f.write(f"a{img_idx} = {a:.10e}  # arcsec = {a*1000:.4f} mas\n")
        f.write(f"rco{img_idx} = {rco:.10e}  # arcsec = {rco*1000:.4f} mas\n")
        f.write(f"rco/a{img_idx} = {rco/a:.10e}\n")
        f.write(f"mass{img_idx} = {mass:.10e}  # M_sun = {format_mass(mass)}\n\n")
    
    f.write(f"# Total sub-halo mass\n")
    f.write(f"total_subhalo_mass = {total_mass:.10e}  # M_sun = {format_mass(total_mass)}\n\n")
    
    f.write(f"# Performance\n")
    f.write(f"chi2_base = {base_mag_chi2:.2f}\n")
    f.write(f"chi2_best = {best_mag_chi2:.2f}\n")
    f.write(f"improvement = {improvement:.1f}%\n")
    f.write(f"constraint_satisfied = {constraint_satisfied}\n")

print(f"\n✓ 完成！")
print(f"  结果图: {output_plot_file}")
print(f"  DE参数文件: {params_file}")
if MCMC_ENABLED:
    print(f"  MCMC链: {mcmc_chain_file}")
    print(f"  后验统计: {posterior_file}")
    print(f"  Corner图: {corner_file}")
    print(f"  轨迹图: {trace_file}")
    print(f"  质量一维后验分布图: {mass_1d_file}")

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║              验证步骤: 使用 glafic 命令行工具                          ║
# ╚═══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("验证步骤: 验证结果（glafic 命令行 vs Python 接口）")
print("=" * 70)

# 使用 find_glafic_bin 找到 glafic 可执行文件
GLAFIC_BIN_VERIFY = find_glafic_bin()
if GLAFIC_BIN_VERIFY is None:
    print("  警告: 找不到 glafic 可执行文件，跳过验证步骤")
else:
    verify_input_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_verify_input.dat')
    print(f"\n生成 glafic 输入文件: {verify_input_file}")

    with open(verify_input_file, 'w') as f:
        f.write("# ========================================\n")
        f.write("# Version Pseudo-Jaffe 2.0 验证文件\n")
        f.write("# ========================================\n")
        f.write(f"# 自动生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# active_subhalos: {active_subhalos}\n")
        f.write(f"# Pseudo-Jaffe Sub-halos 数量: {n_active_subhalos}\n")
        for img_idx, x, y, sig, a, rco in best_params_with_img_idx:
            f.write(f"#   Jaffe at Image {img_idx}: x={x:.10e}, y={y:.10e}, sig={sig:.10e}, a={a:.10e}, rco={rco:.10e}\n")
        f.write("# ========================================\n\n")
        
        # 宇宙学参数
        f.write("# Cosmological parameters\n")
        f.write(f"omega      {omega}\n")
        f.write(f"lambda     {lambda_cosmo}\n")
        f.write(f"weos       {weos}\n")
        f.write(f"hubble     {hubble}\n\n")
        
        # 输出前缀
        f.write("# Output prefix\n")
        verify_prefix = f"{OUTPUT_PREFIX}_verify"
        f.write(f"prefix     {verify_prefix}\n\n")
        
        # 网格设置
        f.write("# Grid settings\n")
        f.write(f"xmin       {xmin}\n")
        f.write(f"ymin       {ymin}\n")
        f.write(f"xmax       {xmax}\n")
        f.write(f"ymax       {ymax}\n")
        f.write(f"pix_ext    {pix_ext}\n")
        f.write(f"pix_poi    {pix_poi}\n")
        f.write(f"maxlev     {maxlev}\n\n")
        
        # 启动设置
        n_lenses = 3 + n_active_subhalos
        f.write(f"# Startup: {n_lenses} lenses, 0 extended, 1 point source\n")
        f.write(f"startup    {n_lenses} 0 1\n\n")
        
        # 基础透镜模型
        f.write("# Base lens model\n")
        f.write(f"lens       sers    {lens_params['sers1'][2]}    ")
        f.write(f"{lens_params['sers1'][3]:.6e}    {lens_params['sers1'][4]:.6e}    {lens_params['sers1'][5]:.6e}    ")
        f.write(f"{lens_params['sers1'][6]:.6e}    {lens_params['sers1'][7]:.6e}    {lens_params['sers1'][8]:.6e}    {lens_params['sers1'][9]:.6e}\n")
        
        f.write(f"lens       sers    {lens_params['sers2'][2]}    ")
        f.write(f"{lens_params['sers2'][3]:.6e}    {lens_params['sers2'][4]:.6e}    {lens_params['sers2'][5]:.6e}    ")
        f.write(f"{lens_params['sers2'][6]:.6e}    {lens_params['sers2'][7]:.6e}    {lens_params['sers2'][8]:.6e}    {lens_params['sers2'][9]:.6e}\n")
        
        f.write(f"lens       {MAIN_LENS_KEY}     {lens_params[MAIN_LENS_KEY][2]}    ")
        f.write(f"{lens_params[MAIN_LENS_KEY][3]:.6e}    {lens_params[MAIN_LENS_KEY][4]:.6e}    {lens_params[MAIN_LENS_KEY][5]:.6e}    ")
        f.write(f"{lens_params[MAIN_LENS_KEY][6]:.6e}    {lens_params[MAIN_LENS_KEY][7]:.6e}    {lens_params[MAIN_LENS_KEY][8]:.6e}    {lens_params[MAIN_LENS_KEY][9]:.6e}\n\n")
        
        # Pseudo-Jaffe Sub-halos (使用高精度)
        f.write(f"# Pseudo-Jaffe Sub-halos ({n_active_subhalos} jaffe perturbations)\n")
        for x_sub, y_sub, sig_sub, a_sub, rco_sub in best_params:
            f.write(f"lens       jaffe   0.2160    {sig_sub:.10e}    ")
            f.write(f"{x_sub:.10e}    {y_sub:.10e}    0.0    0.0    {a_sub:.10e}    {rco_sub:.10e}\n")
        
        f.write("\n")
        
        # 点源
        f.write("# Point source (iPTF16geu supernova)\n")
        f.write(f"point      {source_z}    {best_source_x:.10e}    {best_source_y:.10e}\n\n")
        
        f.write("end_startup\n\n")
        
        f.write("# Commands\n")
        f.write("start_command\n\n")
        f.write("findimg\n\n")
        f.write("quit\n")

    print(f"  生成完成")

    # 运行 glafic
    print(f"\n运行 glafic 命令行工具...")
    print(f"  命令: {GLAFIC_BIN_VERIFY} {verify_input_file}")

    try:
        result = subprocess.run(
            [GLAFIC_BIN_VERIFY, os.path.basename(verify_input_file)],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"  警告: glafic 返回非零代码 {result.returncode}")
            print(f"  stderr: {result.stderr}")
        else:
            print(f"  glafic 运行成功")
        
    except subprocess.TimeoutExpired:
        print(f"  警告: glafic 运行超时（>60秒）")
    except FileNotFoundError:
        print(f"  警告: 找不到 glafic 可执行文件: {GLAFIC_BIN_VERIFY}")
        print(f"  跳过验证步骤")
    except Exception as e:
        print(f"  警告: glafic 运行出错: {e}")

    # 读取 glafic 输出
    verify_output_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_verify_point.dat')

    if os.path.exists(verify_output_file):
        print(f"\n读取 glafic 输出: {verify_output_file}")
        
        try:
            data = np.loadtxt(verify_output_file)
            
            if len(data.shape) == 1:
                n_images_glafic = int(data[0])
                print(f"  警告: glafic 报告 {n_images_glafic} 个图像，但无图像数据")
            else:
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

                    print(f"  已应用中心偏移校正")

                    distances = cdist(obs_positions, glafic_positions)
                    row_ind, col_ind = linear_sum_assignment(distances)

                    glafic_pos_matched = glafic_positions[col_ind[np.argsort(row_ind)]]
                    glafic_mag_matched = glafic_magnifications[col_ind[np.argsort(row_ind)]]
                    
                    print(f"\n对比 Python 接口 vs glafic 命令行:")
                    print(f"  注意: 放大率使用绝对值对比（Python保留正负号，glafic输出绝对值）")
                    print(f"  {'Img':<5} {'Python x [mas]':<15} {'glafic x [mas]':<15} {'Δx [mas]':<12} "
                          f"{'Python y [mas]':<15} {'glafic y [mas]':<15} {'Δy [mas]':<12}")
                    print(f"  {'-'*95}")
                    
                    max_pos_diff = 0.0
                    for i in range(4):
                        py_x = best_pos[i, 0] * 1000
                        py_y = best_pos[i, 1] * 1000
                        gl_x = glafic_pos_matched[i, 0] * 1000
                        gl_y = glafic_pos_matched[i, 1] * 1000
                        
                        diff_x = abs(py_x - gl_x)
                        diff_y = abs(py_y - gl_y)
                        max_pos_diff = max(max_pos_diff, diff_x, diff_y)
                        
                        print(f"  {i+1:<5} {py_x:>13.3f}    {gl_x:>13.3f}    {diff_x:>10.3f}    "
                              f"{py_y:>13.3f}    {gl_y:>13.3f}    {diff_y:>10.3f}")
                    
                    print(f"\n  {'Img':<5} {'Python |μ|':<15} {'glafic |μ|':<15} {'Δ|μ|':<12} {'Δ|μ| [%]':<12}")
                    print(f"  {'-'*65}")
                    
                    max_mag_diff_pct = 0.0
                    for i in range(4):
                        py_mag_abs = abs(best_mag[i])
                        gl_mag = glafic_mag_matched[i]
                        diff_mag = abs(py_mag_abs - gl_mag)
                        diff_mag_pct = (diff_mag / py_mag_abs) * 100 if py_mag_abs != 0 else 0
                        max_mag_diff_pct = max(max_mag_diff_pct, diff_mag_pct)
                        
                        print(f"  {i+1:<5} {py_mag_abs:>13.3f}    {gl_mag:>13.3f}    {diff_mag:>10.3f}    {diff_mag_pct:>10.3f}%")
                    
                    print(f"\n验证结果:")
                    pos_tolerance = 0.01
                    mag_tolerance = 0.1
                    
                    if max_pos_diff < pos_tolerance and max_mag_diff_pct < mag_tolerance:
                        print(f"  ✓ 一致性验证通过！")
                        print(f"    最大位置差: {max_pos_diff:.6f} mas < {pos_tolerance} mas")
                        print(f"    最大放大率差: {max_mag_diff_pct:.6f}% < {mag_tolerance}%")
                    elif max_pos_diff < 1.0 and max_mag_diff_pct < 1.0:
                        print(f"  ✓ 一致性良好（小差异）")
                        print(f"    最大位置差: {max_pos_diff:.6f} mas")
                        print(f"    最大放大率差: {max_mag_diff_pct:.6f}%")
                    else:
                        print(f"  ⚠ 警告: 检测到较大差异")
                        print(f"    最大位置差: {max_pos_diff:.6f} mas")
                        print(f"    最大放大率差: {max_mag_diff_pct:.6f}%")
                    
                    # 保存验证报告
                    verify_report_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_verify_report.txt')
                    with open(verify_report_file, 'w') as f:
                        f.write("=" * 70 + "\n")
                        f.write("Python 接口 vs glafic 命令行 验证报告 (Pseudo-Jaffe Model v2.0)\n")
                        f.write("=" * 70 + "\n\n")
                        f.write(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"active_subhalos: {active_subhalos}\n")
                        f.write(f"Pseudo-Jaffe Sub-halos 数量: {n_active_subhalos}\n")
                        f.write(f"MCMC_ENABLED: {MCMC_ENABLED}\n\n")
                        f.write("注意: 放大率对比使用绝对值，因为 Python 接口保留正负号（宇称），\n")
                        f.write("      而 glafic 命令行输出为绝对值。\n\n")
                        
                        f.write("Pseudo-Jaffe 参数 (高精度):\n")
                        for img_idx, x, y, sig, a, rco in best_params_with_img_idx:
                            f.write(f"  Jaffe {img_idx}: x={x:.10e}, y={y:.10e}, sig={sig:.10e}, a={a:.10e}, rco={rco:.10e}\n")
                        f.write("\n")
                        
                        f.write("位置对比 (mas):\n")
                        f.write(f"{'Img':<5} {'Python x':<15} {'glafic x':<15} {'Δx':<12} "
                                f"{'Python y':<15} {'glafic y':<15} {'Δy':<12}\n")
                        f.write("-" * 95 + "\n")
                        for i in range(4):
                            py_x = best_pos[i, 0] * 1000
                            py_y = best_pos[i, 1] * 1000
                            gl_x = glafic_pos_matched[i, 0] * 1000
                            gl_y = glafic_pos_matched[i, 1] * 1000
                            diff_x = abs(py_x - gl_x)
                            diff_y = abs(py_y - gl_y)
                            f.write(f"{i+1:<5} {py_x:>13.3f}    {gl_x:>13.3f}    {diff_x:>10.3f}    "
                                    f"{py_y:>13.3f}    {gl_y:>13.3f}    {diff_y:>10.3f}\n")
                        
                        f.write(f"\n最大位置差: {max_pos_diff:.6f} mas\n\n")
                        
                        f.write("放大率对比 (使用绝对值):\n")
                        f.write(f"{'Img':<5} {'Python |μ|':<15} {'glafic |μ|':<15} {'Δ|μ|':<12} {'Δ|μ| [%]':<12}\n")
                        f.write("-" * 65 + "\n")
                        for i in range(4):
                            py_mag_abs = abs(best_mag[i])
                            gl_mag = glafic_mag_matched[i]
                            diff_mag = abs(py_mag_abs - gl_mag)
                            diff_mag_pct = (diff_mag / py_mag_abs) * 100 if py_mag_abs != 0 else 0
                            f.write(f"{i+1:<5} {py_mag_abs:>13.3f}    {gl_mag:>13.3f}    {diff_mag:>10.3f}    {diff_mag_pct:>10.3f}%\n")
                        
                        f.write(f"\n最大放大率差: {max_mag_diff_pct:.6f}%\n\n")
                        
                        f.write("验证结论:\n")
                        if max_pos_diff < pos_tolerance and max_mag_diff_pct < mag_tolerance:
                            f.write("一致性验证通过！Python 接口和 glafic 命令行结果高度一致。\n")
                        elif max_pos_diff < 1.0 and max_mag_diff_pct < 1.0:
                            f.write("一致性良好，存在小的数值差异（可能由于数值精度）。\n")
                        else:
                            f.write("警告: 检测到较大差异，建议检查参数设置。\n")
                    
                    print(f"\n  ✓ 验证报告已保存: {verify_report_file}")
                    
                else:
                    print(f"  警告: glafic 找到 {n_images_glafic} 个图像（预期4或5个），跳过验证")

        except Exception as e:
            print(f"  读取或解析 glafic 输出时出错: {e}")

    else:
        print(f"\n警告: glafic 输出文件不存在: {verify_output_file}")
        print(f"  跳过验证步骤")

print("\n" + "=" * 70)
print("Version Pseudo-Jaffe 2.0 完成")
print("=" * 70)

