#!/usr/bin/env python3
"""
Version Point Mass 1.0: Flexible Sub-halos Search
基于 V8.0，新增灵活的控制参数：
1. Draw_Graph 绘图间隔控制
2. active_subhalos 参数：选择启用哪些图像附近的 sub-halo
3. fine_tuning 参数：精细调试模式，独立配置每个 sub-halo
"""

import sys
import random
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'fork':
    multiprocessing.set_start_method('fork', force=True)
import glafic
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, linear_sum_assignment
import os
from datetime import datetime
import matplotlib.pyplot as plt
import subprocess
from plot_paper_style import plot_paper_style, plot_paper_style_compare, read_critical_curves

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
        vals = (raw + [0.0] * 7)[:7]          # 不足7个补0，超出截断
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

    # 全为 sers 时将最后一个视为主透镜
    if main_lens_key is None:
        main_lens_key = list(params_dict.keys())[-1]

    return params_dict, float(point_params[2]), float(point_params[3]), main_lens_key

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║              智能查找 glafic 可执行文件                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def find_glafic_bin(default_path=""):
    """
    智能查找 glafic 可执行文件。

    查找顺序:
    1. 检查指定的默认路径
    2. 从 glafic Python 模块路径推断（支持外部定义的路径）
    3. 从 PATH 环境变量查找
    4. 返回 None 如果都找不到
    """
    if os.path.isfile(default_path) and os.access(default_path, os.X_OK):
        return default_path

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

    import shutil
    glafic_in_path = shutil.which('glafic')
    if glafic_in_path:
        return glafic_in_path

    return None

print("=" * 70)
print("Version Point Mass 1.0: Flexible Sub-halos Search")
print("=" * 70)

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                    可配置参数区域                                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# ==================== 0. 基准透镜参数路径配置 ====================
# 设置为包含 bestfit.dat 的目录路径即可加载外部基准透镜参数。
# 支持 sie（SIE 模型）和 anfw（轴对称 NFW）两种主透镜类型。
# 留空字符串 "" 则使用下方内置的 SIE 默认参数。
# 示例: BASELINE_LENS_DIR = "work/SN_2Sersic_NFW"
BASELINE_LENS_DIR = '/home/luukiaun/glafic251018/work/glade/legacy/v_pointmass_1_0/bestfit_default'

# ==================== 1. 约束条件配置 ====================
CONSTRAINT_SIGMA = 1
PENALTY_COEFFICIENT = 1000

# ==================== 2. Sub-halo 启用配置 (新功能) ====================
# 控制在哪些图像附近拟合 sub-halo
# 例如: [1,2,3,4] 在所有4个图像附近拟合
#      [1,3] 只在图像1和3附近拟合
#      [2,4] 只在图像2和4附近拟合
active_subhalos = [1, 2, 3, 4]

# ==================== 3. 精细调试模式 (新功能) ====================
# 如果为 True，每个 sub-halo 的搜索半径、初始质量、质量范围独立设置
# 如果为 False，所有 sub-halo 使用通用配置
fine_tuning = False

# --- 通用配置（当 fine_tuning=False 时使用） ---
SEARCH_RADIUS = 0.075
MASS_GUESS = 1000000
MASS_LOG_RANGE = 3

# --- 精细配置（当 fine_tuning=True 时使用） ---
# 每个 sub-halo 的独立配置
fine_tuning_configs = {1: {'search_radius': 0.08, 'mass_guess': 100000, 'mass_log_range': 4.5},
 2: {'search_radius': 0.07, 'mass_guess': 50000, 'mass_log_range': 4},
 3: {'search_radius': 0.075, 'mass_guess': 80000, 'mass_log_range': 4.2},
 4: {'search_radius': 0.065, 'mass_guess': 30000, 'mass_log_range': 3.8}}

# ==================== 4. 机器学习目标函数参数 ====================
LOSS_COEF_A = 4
LOSS_COEF_B = 1
LOSS_PENALTY_PL = 10000

# ==================== 4.1 透镜和源参数修改配置 ====================
source_modify = False
lens_modify = False
modify_percentage = 0.2
# 注：两个都为True时，同时优化source和lens参数

# ==================== 5. 优化算法配置 ====================
DE_MAXITER = 650
DE_POPSIZE = 64
DE_ATOL = 0.0001
DE_TOL = 0.0001
DE_SEED = 42
DE_POLISH = True
DE_WORKERS = -1

# 早停机制配置
EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 30

# ==================== 6. MCMC 配置 ====================
MCMC_ENABLED = False
MCMC_NWALKERS = 32
MCMC_NSTEPS = 2000
MCMC_BURNIN = 300
MCMC_THIN = 2
MCMC_PERTURBATION = 0.01
MCMC_PROGRESS = True
MCMC_WORKERS = -1

# ==================== 6.1 MCMC 先验范围配置 ====================
# MCMC_CUSTOM_RANGE = True  → 使用下方自定义范围作为先验边界
# MCMC_CUSTOM_RANGE = False → 直接沿用 DE 搜索范围（bounds）作为先验边界
MCMC_CUSTOM_RANGE = False

# 自定义先验范围（仅在 MCMC_CUSTOM_RANGE=True 时生效）
MCMC_SEARCH_RADIUS = 0.3
MCMC_LOG_M_MIN = 1
MCMC_LOG_M_MAX = 14

# ==================== 7. 绘图配置 (新功能) ====================
SHOW_2SIGMA = False
OUTPUT_PREFIX = 'v_pm_1_0'
COMPARE_GRAPH = True

# Draw_Graph 控制：
# - 0: 不绘制任何迭代图
# - 1: 绘制迭代图，使用 draw_interval 控制间隔
Draw_Graph = 1

# 绘图间隔参数（最小=1）
# 例如：draw_interval=5 表示每5次迭代绘制一次
#      draw_interval=1 表示每次迭代都绘制
draw_interval = 5

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                         固定参数                                       ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# ==================== 可注入的观测数据（WebUI / 注入器可覆盖）====================
obs_positions_mas_list = [[-266.035, 0.427], [118.835, -221.927], [238.324, 227.27], [-126.157, 319.719]]
obs_magnifications_list = [-35.6, 15.7, -7.5, 9.1]
obs_mag_errors_list = [2.1, 1.3, 1, 1.1]
obs_pos_sigma_mas_list = [0.41, 0.86, 2.23, 3.11]
center_offset_x = 0.01535
center_offset_y = 0.0322
obs_x_flip = True

# 坐标转换：统一取符号，同时作用于观测位置和中心偏移，确保两者始终在同一坐标系下
_x_sign = -1 if obs_x_flip else 1
obs_positions_mas = np.array(obs_positions_mas_list)
obs_positions = np.zeros_like(obs_positions_mas)
obs_positions[:, 0] = _x_sign * obs_positions_mas[:, 0] / 1000.0
obs_positions[:, 1] = obs_positions_mas[:, 1] / 1000.0
center_offset_x = 0.01535

obs_magnifications = np.array(obs_magnifications_list)
obs_mag_errors = np.array(obs_mag_errors_list)
obs_pos_sigma_mas = np.array(obs_pos_sigma_mas_list)

# 基础模型参数
omega = 0.3
lambda_cosmo = 0.7
weos = -1.0
hubble = 0.7

xmin, ymin = -0.5, -0.5
xmax, ymax = 0.5, 0.5
pix_ext = 0.01
pix_poi = 0.2
maxlev = 5

source_z = 0.4090
lens_z = 0.2160

# 默认基准透镜参数（SIE 模型，来自 SN_2Sersic_SIE/bestfit.dat）
source_x = 2.685497e-03
source_y = 2.443616e-02

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

# 验证 active_subhalos 参数
active_subhalos = [1, 2, 3, 4]
for idx in active_subhalos:
    if idx not in [1, 2, 3, 4]:
        raise ValueError(f"active_subhalos 包含无效的图像索引: {idx}，必须是 1-4 之间的整数")

n_active_subhalos = len(active_subhalos)
n_params_subhalo = n_active_subhalos * 3  # 每个 sub-halo 3个参数: x, y, log_m

# 计算额外的优化参数维度
n_params_source = 0
n_params_lens = 0

if source_modify:
    n_params_source = 2  # source_x, source_y

if lens_modify:
    # 每个透镜优化 5 个自由参数（p1, x, y, p4, p5），数量由 bestfit.dat 实际透镜数决定
    n_params_lens = len(lens_params) * 5

n_params_extra = n_params_source + n_params_lens
n_params = n_params_subhalo + n_params_extra  # 总参数数量

# 构建每个 active sub-halo 的配置
subhalo_configs = {}
for img_idx in active_subhalos:
    if fine_tuning:
        # 使用精细配置
        cfg = fine_tuning_configs[img_idx]
        subhalo_configs[img_idx] = {
            'search_radius': cfg['search_radius'],
            'mass_guess': cfg['mass_guess'],
            'mass_log_range': cfg['mass_log_range']
        }
    else:
        # 使用通用配置
        subhalo_configs[img_idx] = {
            'search_radius': SEARCH_RADIUS,
            'mass_guess': MASS_GUESS,
            'mass_log_range': MASS_LOG_RANGE
        }

# 验证 draw_interval
if draw_interval < 1:
    print(f"警告: draw_interval={draw_interval} 小于最小值1，已自动调整为1")
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
# ║              迭代可视化函数（适配灵活数量的 sub-halo）                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def plot_iteration_population_flexible(population, iteration_num, output_dir, 
                                       mass_range=None, pos_range=None, bounds=None):
    """
    绘制灵活数量 sub-halo 的种群分布
    
    Draw_Graph 和 draw_interval 控制:
    - Draw_Graph=0: 不绘制任何图
    - Draw_Graph=1: 根据 draw_interval 间隔绘制
    """
    # 根据 Draw_Graph 设置决定是否绘图
    if Draw_Graph == 0:
        return  # 不绘制任何图
    
    # 如果没有 subhalo，跳过绘图
    if n_active_subhalos == 0:
        return  # 没有 subhalo 时不绘制
    
    # 检查是否应该在此迭代绘图
    if iteration_num % draw_interval != 0 and iteration_num != 0:
        return  # 跳过此次绘图
    
    # 检查是否需要反归一化
    if bounds is not None and population.max() <= 1.0 and population.min() >= 0.0:
        if iteration_num == 0:
            print(f"    [调试] 检测到归一化数据，正在反归一化...")
        population_denorm = np.zeros_like(population)
        for i in range(population.shape[1]):
            lower, upper = bounds[i]
            population_denorm[:, i] = population[:, i] * (upper - lower) + lower
        population = population_denorm
    
    # 提取参数
    # population shape: [popsize, n_params]
    # [x1, y1, log_m1, x2, y2, log_m2, ...]
    n_halos = n_active_subhalos
    log_m = []
    x = []
    y = []
    for i in range(n_halos):
        x.append(population[:, i*3])
        y.append(population[:, i*3 + 1])
        log_m.append(population[:, i*3 + 2])
    
    # 绘制图表
    fig = plt.figure(figsize=(4*n_halos, 4*n_halos))
    
    labels = [f'Halo {active_subhalos[i]}' for i in range(n_halos)]
    
    for i in range(n_halos):
        for j in range(n_halos):
            if i < j:
                ax = plt.subplot(n_halos, n_halos, i*n_halos + j + 1)
                scatter = ax.scatter(log_m[j], log_m[i], 
                                   c=np.arange(len(log_m[i])), 
                                   cmap='viridis', alpha=0.5, s=20)
                ax.set_xlabel(f'{labels[j]} log(M)', fontsize=8)
                ax.set_ylabel(f'{labels[i]} log(M)', fontsize=8)
                ax.grid(True, linestyle=':', alpha=0.3)
                if mass_range is not None:
                    ax.set_xlim(mass_range[2*j], mass_range[2*j+1])
                    ax.set_ylim(mass_range[2*i], mass_range[2*i+1])
            elif i == j:
                ax = plt.subplot(n_halos, n_halos, i*n_halos + j + 1)
                ax.hist(log_m[i], bins=20, alpha=0.6, color='blue')
                ax.set_xlabel(f'{labels[i]} log(M)', fontsize=8)
                ax.set_ylabel('Count', fontsize=8)
                ax.grid(True, linestyle=':', alpha=0.3)
                if mass_range is not None:
                    ax.set_xlim(mass_range[2*i], mass_range[2*i+1])
    
    plt.suptitle(f'Iteration {iteration_num}: Mass Distribution ({n_halos} Sub-halos)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = os.path.join(output_dir, f'iteration_{iteration_num:04d}.png')
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"    保存迭代图: iteration_{iteration_num:04d}.png (dpi=120)")

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                    打印配置摘要                                        ║
# ╚═══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("配置摘要")
print("=" * 70)

print(f"\n透镜和源参数优化:")
print(f"  source_modify = {source_modify}")
print(f"  lens_modify = {lens_modify}")
if source_modify or lens_modify:
    print(f"  modify_percentage = {modify_percentage*100:.2f}%")
    if source_modify and lens_modify:
        print(f"  优化范围: source位置 + lens参数 (共{n_params_extra}个)")
    elif source_modify:
        print(f"  优化范围: 仅source位置 (共{n_params_source}个)")
    elif lens_modify:
        print(f"  优化范围: 仅lens参数 (共{n_params_lens}个)")

print(f"\n启用的 Sub-halos (新功能):")
print(f"  active_subhalos = {active_subhalos}")
print(f"  启用数量: {n_active_subhalos}")
print(f"  参数维度: {n_params} (每个sub-halo 3个参数)")

print(f"\n精细调试模式 (新功能):")
print(f"  fine_tuning = {fine_tuning}")

if fine_tuning:
    print(f"  使用独立配置:")
    for img_idx in active_subhalos:
        cfg = subhalo_configs[img_idx]
        print(f"    Sub-halo {img_idx}:")
        print(f"      搜索半径: ±{cfg['search_radius']*1000:.0f} mas")
        print(f"      质量猜测: {cfg['mass_guess']:.2e} M☉")
        print(f"      质量范围: ±{cfg['mass_log_range']:.1f} dex")
else:
    print(f"  使用通用配置:")
    print(f"    搜索半径: ±{SEARCH_RADIUS*1000:.0f} mas")
    print(f"    质量猜测: {MASS_GUESS:.2e} M☉")
    print(f"    质量范围: ±{MASS_LOG_RANGE:.1f} dex")

print(f"\n各 Sub-halo 搜索空间:")
for img_idx in active_subhalos:
    cfg = subhalo_configs[img_idx]
    x_center = obs_positions[img_idx-1, 0]
    y_center = obs_positions[img_idx-1, 1]
    mass_log = np.log10(cfg['mass_guess'])
    mass_log_min = mass_log - cfg['mass_log_range']
    mass_log_max = mass_log + cfg['mass_log_range']
    
    print(f"  Sub-halo {img_idx} (Image {img_idx} 附近):")
    print(f"    中心: ({x_center:.6f}, {y_center:.6f}) arcsec")
    print(f"    范围: x=[{x_center-cfg['search_radius']:.6f}, {x_center+cfg['search_radius']:.6f}]")
    print(f"           y=[{y_center-cfg['search_radius']:.6f}, {y_center+cfg['search_radius']:.6f}]")
    print(f"    质量: log(M) = [{mass_log_min:.2f}, {mass_log_max:.2f}]")
    print(f"           M = [{10**mass_log_min:.2e}, {10**mass_log_max:.2e}] M☉")

print(f"\n约束条件:")
print(f"  σ倍数: {CONSTRAINT_SIGMA}σ")
max_pos_deviation_mas = CONSTRAINT_SIGMA * obs_pos_sigma_mas
for i in range(4):
    print(f"    Image {i+1}: Δpos < {max_pos_deviation_mas[i]:.2f} mas")

print(f"\n优化算法:")
print(f"  最大迭代: {DE_MAXITER}")
print(f"  种群大小: {DE_POPSIZE}")
print(f"  容差: atol={DE_ATOL}, tol={DE_TOL}")
print(f"  并行核心: {DE_WORKERS}")

print(f"\n早停机制:")
if EARLY_STOPPING:
    print(f"  启用: 是")    
    print(f"  容忍次数: {EARLY_STOP_PATIENCE} 次")
else:
    print(f"  启用: 否")

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
    else:
        print(f"  先验范围: 沿用 DE 搜索范围（bounds）")
else:
    print(f"  启用: 否")

print(f"\n输出配置:")
print(f"  显示2σ: {SHOW_2SIGMA}")
print(f"  文件前缀: {OUTPUT_PREFIX}")
print(f"  绘图模式: Draw_Graph={Draw_Graph} (新功能)")
if Draw_Graph == 0:
    print(f"    计算过程中不绘图，只保留最终三联图")
elif Draw_Graph == 1:
    print(f"    绘图间隔: draw_interval={draw_interval}")
    if draw_interval == 1:
        print(f"    每次迭代都绘制")
    else:
        print(f"    每 {draw_interval} 次迭代绘制一次")

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║              计算固定坐标轴范围                                        ║
# ╚═══════════════════════════════════════════════════════════════════════╝

FIXED_MASS_RANGE = []
for img_idx in active_subhalos:
    cfg = subhalo_configs[img_idx]
    mass_log = np.log10(cfg['mass_guess'])
    mass_log_min = mass_log - cfg['mass_log_range']
    mass_log_max = mass_log + cfg['mass_log_range']
    FIXED_MASS_RANGE.extend([mass_log_min, mass_log_max])
FIXED_MASS_RANGE = tuple(FIXED_MASS_RANGE)

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
def compute_model(subhalo_params_list, verbose=False, src_x=None, src_y=None, lens_params_dict=None):
    """
    计算多sub-halo模型
    subhalo_params_list: [(x, y, mass), ...]
    src_x, src_y: 可选的源位置参数（如果为None则使用全局默认值）
    lens_params_dict: 可选的透镜参数字典（如果为None则使用全局默认值）
    """
    # 使用传入的参数或默认值
    use_src_x = src_x if src_x is not None else source_x
    use_src_y = src_y if src_y is not None else source_y
    use_lens_params = lens_params_dict if lens_params_dict is not None else lens_params
    
    # 每个进程用独立的临时文件前缀，避免 workers=-1 时文件冲突
    _prefix = f'temp_{OUTPUT_PREFIX}_{os.getpid()}'
    glafic.init(omega, lambda_cosmo, weos, hubble, _prefix,
                xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev, verb=0)

    n_subhalos = len(subhalo_params_list)
    n_base_lenses = len(use_lens_params)
    glafic.startup_setnum(n_base_lenses + n_subhalos, 0, 1)

    # 动态遍历所有基础透镜（不再硬编码 sers1/sers2/MAIN_LENS_KEY）
    for key, pv in use_lens_params.items():
        glafic.set_lens(*pv)

    for i, (x_sub, y_sub, mass_sub) in enumerate(subhalo_params_list):
        glafic.set_lens(n_base_lenses + 1 + i, 'point', lens_z, mass_sub,
                        x_sub, y_sub, 0.0, 0.0, 0.0, 0.0)
    
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
    """
    灵活参数优化：n_active_subhalos 个 sub-halo
    如果source_modify=True，包括source位置参数
    如果lens_modify=True，包括lens参数
    params = [x1, y1, log_m1, x2, y2, log_m2, ..., (可选: src_x, src_y), (可选: lens参数)]
    """
    # 解析sub-halo参数
    subhalo_list = []
    for i in range(n_active_subhalos):
        x = params[i*3]
        y = params[i*3 + 1]
        log_m = params[i*3 + 2]
        mass = 10**log_m
        subhalo_list.append((x, y, mass))
    
    # 解析额外的源和透镜参数
    src_x_opt = None
    src_y_opt = None
    lens_params_opt = None
    
    idx = n_params_subhalo
    
    # 解析source参数（如果启用source_modify）
    if source_modify:
        src_x_opt = params[idx]
        src_y_opt = params[idx + 1]
        idx += 2
    
    # 解析透镜参数（动态遍历，不硬编码 sers1/sers2/MAIN_LENS_KEY）
    if lens_modify:
        lens_params_opt = {}
        for _key, _pv in lens_params.items():
            new_pv = list(_pv)
            for _pi in [3, 4, 5, 6, 7]:   # 与 bounds 构建顺序一致
                new_pv[_pi] = params[idx]
                idx += 1
            lens_params_opt[_key] = tuple(new_pv)

    pos, mag, delta_pos, mag_chi2 = compute_model(subhalo_list, src_x=src_x_opt, src_y=src_y_opt,
                                                    lens_params_dict=lens_params_opt)
    
    if pos is None:
        return 1e15
    
    Y = machine_learning_loss(pos, mag, delta_pos)
    return Y

# ==================== 优化搜索 ====================
print("\n" + "=" * 70)
print(f"步骤2: 差分进化算法优化搜索 ({n_params}维空间)")
print("=" * 70)

# 构建bounds：n_active_subhalos 个 sub-halo × 3个参数
bounds = []
for img_idx in active_subhalos:
    cfg = subhalo_configs[img_idx]
    x_center = obs_positions[img_idx-1, 0]
    y_center = obs_positions[img_idx-1, 1]
    mass_log = np.log10(cfg['mass_guess'])
    mass_log_min = mass_log - cfg['mass_log_range']
    mass_log_max = mass_log + cfg['mass_log_range']
    
    bounds.append((x_center - cfg['search_radius'], x_center + cfg['search_radius']))  # x
    bounds.append((y_center - cfg['search_radius'], y_center + cfg['search_radius']))  # y
    bounds.append((mass_log_min, mass_log_max))  # log(M)

# 添加额外的优化参数bounds
# 先添加source参数（如果source_modify=True）
if source_modify:
    bounds.append((source_x * (1 - modify_percentage), source_x * (1 + modify_percentage)))
    bounds.append((source_y * (1 - modify_percentage), source_y * (1 + modify_percentage)))

# 再添加lens参数（如果lens_modify=True）——动态遍历所有透镜，不硬编码 sers1/sers2
if lens_modify:
    for _key, _pv in lens_params.items():
        for _pi in [3, 4, 5, 6, 7]:   # p1, x, y, p4, p5
            v = _pv[_pi]
            delta = abs(v) * modify_percentage
            if delta < 1e-10:          # 防止零值导致零宽 bounds
                delta = modify_percentage
            bounds.append((v - delta, v + delta))

print(f"\n搜索参数空间（{n_params}维）:")
for i, img_idx in enumerate(active_subhalos):
    cfg = subhalo_configs[img_idx]
    x_center = obs_positions[img_idx-1, 0]
    y_center = obs_positions[img_idx-1, 1]
    mass_log = np.log10(cfg['mass_guess'])
    mass_log_min = mass_log - cfg['mass_log_range']
    mass_log_max = mass_log + cfg['mass_log_range']
    
    print(f"  Sub-halo {img_idx} (参数索引 {i*3}-{i*3+2}):")
    print(f"    x=[{x_center-cfg['search_radius']:.4f}, {x_center+cfg['search_radius']:.4f}]")
    print(f"    y=[{y_center-cfg['search_radius']:.4f}, {y_center+cfg['search_radius']:.4f}]")
    print(f"    logM=[{mass_log_min:.2f}, {mass_log_max:.2f}]")

print(f"\n开始优化...")

from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
import scipy

print(f"  Scipy版本: {scipy.__version__}")

np.random.seed(DE_SEED)
solver = DifferentialEvolutionSolver(
    objective_function,
    bounds,
    maxiter=DE_MAXITER,
    popsize=DE_POPSIZE,
    atol=DE_ATOL,
    tol=DE_TOL,
    rng=np.random.default_rng(DE_SEED),
    polish=DE_POLISH,
    disp=True,
    workers=DE_WORKERS,
    updating='deferred'
)

# 记录初始种群（迭代0）
print(f"\n迭代 0 (初始种群):")
initial_pop = solver.population.copy()
iteration_history.append({
    'iteration': 0,
    'population': initial_pop
})
plot_iteration_population_flexible(initial_pop, 0, output_dir, 
                                   mass_range=FIXED_MASS_RANGE,
                                   bounds=bounds)

# 执行优化迭代
iteration = 1
previous_best_energy = np.min(solver.population_energies)
converged_count = 0

while True:
    next_gen = solver.__next__()
    
    current_pop = solver.population.copy()
    iteration_history.append({
        'iteration': iteration,
        'population': current_pop
    })
    
    print(f"\n迭代 {iteration}:")
    current_best_energy = np.min(solver.population_energies)
    
    abs_change = abs(current_best_energy - previous_best_energy)
    # 安全计算相对变化，避免除零警告
    if abs(previous_best_energy) > 1e-10 and np.isfinite(previous_best_energy):
        rel_change = abs_change / abs(previous_best_energy)
    else:
        rel_change = float('inf')
    
    print(f"  当前最佳目标值: {current_best_energy:.6f}")
    print(f"  绝对变化: {abs_change:.6f} (容差: {DE_ATOL})")
    print(f"  相对变化: {rel_change:.6e} (容差: {DE_TOL})")
    
    converged_this_iter = False
    if abs_change < DE_ATOL:
        print(f"    满足绝对容差")
        converged_this_iter = True
    if rel_change < DE_TOL:
        print(f"    满足相对容差")
        converged_this_iter = True
    
    if EARLY_STOPPING:
        if converged_this_iter:
            converged_count += 1
            print(f"   连续满足容差: {converged_count}/{EARLY_STOP_PATIENCE} 次")
            
            if converged_count >= EARLY_STOP_PATIENCE:
                print(f"\n 早停触发！连续 {converged_count} 次满足容差。")
                plot_iteration_population_flexible(current_pop, iteration, output_dir,
                                                   mass_range=FIXED_MASS_RANGE,
                                                   bounds=bounds)
                break
        else:
            if converged_count > 0:
                print(f"    未满足容差，重置计数器（之前: {converged_count}）")
            converged_count = 0
    
    plot_iteration_population_flexible(current_pop, iteration, output_dir,
                                       mass_range=FIXED_MASS_RANGE,
                                       bounds=bounds)
    
    previous_best_energy = current_best_energy
    
    if next_gen is None:
        print(f"\n 优化收敛！算法内部判定已收敛。")
        break
    
    iteration += 1
    
    if iteration > DE_MAXITER:
        print(f"\n  达到最大迭代次数 {DE_MAXITER}。")
        break

# 获取最终结果
result = solver.x
final_fun = np.min(solver.population_energies)

# 解析结果
best_params = []
best_params_with_img_idx = []  # 包含图像索引的版本
for i, img_idx in enumerate(active_subhalos):
    x = result[i*3]
    y = result[i*3 + 1]
    log_m = result[i*3 + 2]
    mass = 10**log_m
    best_params.append((x, y, mass))
    best_params_with_img_idx.append((img_idx, x, y, mass))

# 解析额外的优化参数
best_source_x = source_x
best_source_y = source_y
best_lens_params = lens_params.copy()

idx = n_params_subhalo

# 解析source参数（如果启用source_modify）
if source_modify:
    best_source_x = result[idx]
    best_source_y = result[idx + 1]
    idx += 2

# 解析透镜参数（动态遍历，与 objective_function 及 bounds 顺序一致）
if lens_modify:
    best_lens_params = {}
    for _key, _pv in lens_params.items():
        new_pv = list(_pv)
        for _pi in [3, 4, 5, 6, 7]:
            new_pv[_pi] = result[idx]
            idx += 1
        best_lens_params[_key] = tuple(new_pv)

print(f"\n" + "=" * 70)
print("步骤3: 分析最佳结果")
print("=" * 70)

best_pos, best_mag, best_delta_pos, best_mag_chi2 = compute_model(
    best_params, verbose=True, src_x=best_source_x, src_y=best_source_y,
    lens_params_dict=best_lens_params if lens_modify else None
)

if best_pos is None:
    print("\n 优化失败：最佳参数无法产生4个图像！")
    sys.exit(1)

print(f"\n最佳 {n_active_subhalos} 个 sub-halo 参数:")
for img_idx, x, y, m in best_params_with_img_idx:
    print(f"  Sub-halo at Image {img_idx}:")
    print(f"    位置: ({x:.6f}, {y:.6f}) arcsec")
    print(f"    质量: {m:.2e} M☉ (log10 = {np.log10(m):.2f})")

if source_modify:
    print(f"\n优化后的源参数:")
    print(f"  source_x: {source_x:.6e} -> {best_source_x:.6e} (变化: {(best_source_x/source_x - 1)*100:+.3f}%)")
    print(f"  source_y: {source_y:.6e} -> {best_source_y:.6e} (变化: {(best_source_y/source_y - 1)*100:+.3f}%)")

if lens_modify:
    print(f"\n优化后的透镜参数:")
    _param_labels = ['p1', 'x', 'y', 'p4', 'p5']
    for _key in lens_params:
        print(f"  {_key}:")
        for _pi, _label in zip([3, 4, 5, 6, 7], _param_labels):
            orig = lens_params[_key][_pi]
            best = best_lens_params[_key][_pi]
            chg = (best / orig - 1) * 100 if abs(orig) > 1e-12 else float('nan')
            print(f"    {_label}: {orig:.6e} -> {best:.6e} (变化: {chg:+.3f}%)")

print(f"\n改善效果:")
print(f"  基准chi2: {base_mag_chi2:.2f}")
print(f"  最佳chi2: {best_mag_chi2:.2f}")
improvement = (base_mag_chi2 - best_mag_chi2) / base_mag_chi2 * 100
print(f"  改善: {improvement:.1f}%")

print(f"\n位置约束检查:")
constraint_satisfied = True
for i in range(4):
    status = " OK" if best_delta_pos[i] <= max_pos_deviation_mas[i] else " 超限"
    if best_delta_pos[i] > max_pos_deviation_mas[i]:
        constraint_satisfied = False
    print(f"  Img {i+1}: {best_delta_pos[i]:.2f} mas < {max_pos_deviation_mas[i]:.2f} mas [{status}]")

if constraint_satisfied:
    print(f"\n   所有约束满足！")
else:
    print(f"\n    部分图像超限")

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
        点质量模型参数均为 (x, y, log_m)，物理上无额外约束。
        """
        if MCMC_CUSTOM_RANGE:
            # 使用自定义先验范围
            for i in range(n_active_subhalos):
                img_idx = active_subhalos[i]
                x_ctr   = obs_positions[img_idx - 1, 0]
                y_ctr   = obs_positions[img_idx - 1, 1]
                x     = params[i*3]
                y     = params[i*3 + 1]
                log_m = params[i*3 + 2]
                if abs(x - x_ctr) > MCMC_SEARCH_RADIUS or abs(y - y_ctr) > MCMC_SEARCH_RADIUS:
                    return -np.inf
                if not (MCMC_LOG_M_MIN <= log_m <= MCMC_LOG_M_MAX):
                    return -np.inf
        else:
            # 使用 DE 搜索范围（bounds 列表），与 DE 优化保持完全一致
            for i, (low, high) in enumerate(bounds):
                if not (low <= params[i] <= high):
                    return -np.inf

        subhalo_list = []
        for i in range(n_active_subhalos):
            x     = params[i*3]
            y     = params[i*3 + 1]
            log_m = params[i*3 + 2]
            subhalo_list.append((x, y, 10**log_m))

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
            for _key, _pv in lens_params.items():
                new_pv = list(_pv)
                for _pi in [3, 4, 5, 6, 7]:
                    new_pv[_pi] = params[idx]
                    idx += 1
                lens_params_opt[_key] = tuple(new_pv)

        pos, mag, delta_pos, _ = compute_model(subhalo_list, src_x=src_x_opt, src_y=src_y_opt,
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
        param_names.extend([f'x_{img_idx}', f'y_{img_idx}', f'logM_{img_idx}'])
    if source_modify:
        param_names.extend(['src_x', 'src_y'])
    if lens_modify:
        for _key in lens_params:
            param_names.extend([f'{_key}_p1', f'{_key}_x', f'{_key}_y', f'{_key}_p4', f'{_key}_p5'])
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
        else:
            print(f"  {name}: {median:.6e} +{upper-median:.3e} -{median-lower:.3e}")

    # ==================== 绘制 Corner Plot ====================
    if n_params_subhalo > 0:
        print(f"\n生成 Corner Plot...")
        corner_labels = []
        for img_idx in active_subhalos:
            corner_labels.extend([f'$x_{img_idx}$', f'$y_{img_idx}$', f'$\\log M_{img_idx}$'])

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
    else:
        print(f"\n跳过 Corner Plot（无 subhalo 参数）")

    # ==================== 绘制轨迹图 ====================
    if n_params_subhalo > 0:
        print(f"\n生成 MCMC 链轨迹图...")
        corner_labels = []
        for img_idx in active_subhalos:
            corner_labels.extend([f'$x_{img_idx}$', f'$y_{img_idx}$', f'$\\log M_{img_idx}$'])
        
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

    # ==================== 计算质量后验分布 ====================
    if n_active_subhalos > 0:
        print(f"\n计算质量后验分布...")
    mass_posterior_stats = {}
    for i, img_idx in enumerate(active_subhalos):
        log_m_samples_i = samples[:, i*3 + 2]
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

            ax.set_xlabel(r'$\log_{10}(M / M_\odot)$', fontsize=13)
            ax.set_ylabel('Posterior density', fontsize=13)
            ax.set_title(f'Point mass Sub-halo {img_idx} mass posterior', fontsize=12)
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
        f.write("# Version Point Mass 1.0\n")
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
            f.write(f"# Point mass Sub-halo at Image {img_idx}:\n")
            xs  = posterior_stats[f'x_{img_idx}']
            ys  = posterior_stats[f'y_{img_idx}']
            lms = posterior_stats[f'logM_{img_idx}']
            ms  = mass_posterior_stats[f'mass_{img_idx}']
            f.write(f"#   x    = {xs['median']:.6f} +{xs['error_plus']*1000:.3f} -{xs['error_minus']*1000:.3f} mas\n")
            f.write(f"#   y    = {ys['median']:.6f} +{ys['error_plus']*1000:.3f} -{ys['error_minus']*1000:.3f} mas\n")
            f.write(f"#   logM = {lms['median']:.3f} +{lms['error_plus']:.3f} -{lms['error_minus']:.3f} dex\n")
            f.write(f"#   M    = {ms['median']:.3e} +{ms['error_plus']:.3e} -{ms['error_minus']:.3e} M_sun\n\n")
    print(f"  ✓ 后验统计已保存: {posterior_file}")

# ==================== 生成图表 ====================
print("\n" + "=" * 70)
print("步骤5: 生成最终图表")
print("=" * 70)

glafic.init(omega, lambda_cosmo, weos, hubble, f'temp_{OUTPUT_PREFIX}_best',
            xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev, verb=0)

_n_base = len(best_lens_params)
glafic.startup_setnum(_n_base + n_active_subhalos, 0, 1)
for _key, _pv in best_lens_params.items():
    glafic.set_lens(*_pv)
for i, (x_sub, y_sub, mass_sub) in enumerate(best_params):
    glafic.set_lens(_n_base + 1 + i, 'point', lens_z, mass_sub, x_sub, y_sub, 0.0, 0.0, 0.0, 0.0)

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
    
    plot_paper_style_compare(
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
        suptitle=f"iPTF16geu: Baseline vs {n_active_subhalos} Point Mass Sub-halos",
        output_file=output_plot_file_compare,
        title_left="Position Offset Comparison",
        title_mid="Magnification Comparison",
        title_right="Image Positions & Critical Curves",
        subhalo_positions=best_params,
        show_2sigma=SHOW_2SIGMA
    )
    print(f"  比较图已保存: {output_plot_file_compare}")

# 始终生成标准图
plot_paper_style(
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
    suptitle=f"iPTF16geu: {n_active_subhalos} Sub-halos Model",
    output_file=output_plot_file,
    title_left="Position Offset",
    title_mid="Magnification",
    title_right="Image Positions & Critical Curves",
    subhalo_positions=best_params,
    show_2sigma=SHOW_2SIGMA
)

# 保存参数
params_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_best_params.txt')
with open(params_file, 'w') as f:
    f.write(f"# Version Point Mass 1.0: Flexible Sub-halos Search\n")
    f.write(f"# Configuration:\n")
    f.write(f"#   active_subhalos = {active_subhalos}\n")
    f.write(f"#   fine_tuning = {fine_tuning}\n")
    f.write(f"#   source_modify = {source_modify}\n")
    f.write(f"#   lens_modify = {lens_modify}\n")
    if source_modify or lens_modify:
        f.write(f"#   modify_percentage = {modify_percentage}\n")
    f.write(f"#   patience = {EARLY_STOP_PATIENCE}, maxiter = {DE_MAXITER}\n")
    f.write(f"#   draw_interval = {draw_interval}\n")
    f.write(f"#   DE_SEED = {DE_SEED}\n\n")
    
    # 保存源参数（如果优化了）
    if source_modify:
        f.write(f"# Optimized Source Parameters\n")
        f.write(f"source_x_original = {source_x:.10e}\n")
        f.write(f"source_y_original = {source_y:.10e}\n")
        f.write(f"source_x_optimized = {best_source_x:.10e}  # change: {(best_source_x/source_x - 1)*100:+.4f}%\n")
        f.write(f"source_y_optimized = {best_source_y:.10e}  # change: {(best_source_y/source_y - 1)*100:+.4f}%\n\n")
    
    # 保存透镜参数（动态遍历，兼容任意透镜数量）
    if lens_modify:
        f.write(f"# Optimized Lens Parameters\n")
        _pnames = ['p1', 'x', 'y', 'p4', 'p5']
        for _key in lens_params:
            f.write(f"# {_key}\n")
            for _pi, _pn in zip([3, 4, 5, 6, 7], _pnames):
                orig = lens_params[_key][_pi]
                best = best_lens_params[_key][_pi]
                chg = (best / orig - 1) * 100 if abs(orig) > 1e-12 else float('nan')
                f.write(f"{_key}_{_pn}_original = {orig:.10e}\n")
                f.write(f"{_key}_{_pn}_optimized = {best:.10e}  # change: {chg:+.4f}%\n")
            f.write("\n")
    
    f.write(f"# Sub-halo Parameters (高精度保存，避免放大率敏感性问题)\n")
    for img_idx, x, y, m in best_params_with_img_idx:
        f.write(f"# Sub-halo at Image {img_idx}\n")
        f.write(f"x_sub{img_idx} = {x:.10e}  # arcsec\n")
        f.write(f"y_sub{img_idx} = {y:.10e}  # arcsec\n")
        f.write(f"mass_sub{img_idx} = {m:.10e}  # Msun\n\n")
    
    f.write(f"# Performance\n")
    f.write(f"chi2_base = {base_mag_chi2:.2f}\n")
    f.write(f"chi2_best = {best_mag_chi2:.2f}\n")
    f.write(f"improvement = {improvement:.1f}%\n")
    f.write(f"constraint_satisfied = {constraint_satisfied}\n")

print(f"\n 完成！")
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

# 动态查找 glafic 可执行文件
GLAFIC_BIN = find_glafic_bin()
if GLAFIC_BIN:
    print(f"  glafic 路径: {GLAFIC_BIN}")
else:
    print(f"  警告: 未找到 glafic 可执行文件，将跳过验证步骤")

# 生成 glafic input 文件
verify_input_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_verify_input.dat')
print(f"\n生成 glafic 输入文件: {verify_input_file}")

with open(verify_input_file, 'w') as f:
    f.write("# ========================================\n")
    f.write("# Version Point Mass 1.0 验证文件\n")
    f.write("# ========================================\n")
    f.write(f"# 自动生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"# active_subhalos: {active_subhalos}\n")
    f.write(f"# Sub-halos 数量: {n_active_subhalos}\n")
    for img_idx, x, y, m in best_params_with_img_idx:
        f.write(f"#   Sub-halo at Image {img_idx}: x={x:.6f}, y={y:.6f}, M={m:.2e}\n")
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
    
    # 启动设置（动态透镜数量 + N个sub-halos + 1个点源）
    n_lenses = len(lens_params) + n_active_subhalos
    f.write(f"# Startup: {n_lenses} lenses, 0 extended, 1 point source\n")
    f.write(f"startup    {n_lenses} 0 1\n\n")

    # 基础透镜模型（动态遍历）
    f.write("# Base lens model\n")
    for _key, _pv in lens_params.items():
        f.write(f"lens       {_pv[1]:<10} {_pv[2]}    ")
        f.write("    ".join(f"{_pv[i]:.6e}" for i in range(3, 10)) + "\n")
    f.write("\n")
    
    # Sub-halos (点质量模型)
    f.write(f"# Sub-halos ({n_active_subhalos} point mass perturbations)\n")
    for x_sub, y_sub, mass_sub in best_params:
        f.write(f"lens       point   0.2160    {mass_sub:.6e}    ")
        f.write(f"{x_sub:.6e}    {y_sub:.6e}    0.0    0.0    0.0    0.0\n")
    
    f.write("\n")
    
    # 点源（超新星）
    f.write("# Point source (iPTF16geu supernova)\n")
    f.write(f"point      {source_z}    {source_x:.6e}    {source_y:.6e}\n\n")
    
    # 结束startup
    f.write("end_startup\n\n")
    
    # 命令
    f.write("# Commands\n")
    f.write("start_command\n\n")
    f.write("findimg\n\n")
    f.write("quit\n")

print(f"  生成完成")

# 运行 glafic（仅在找到可执行文件时）
if GLAFIC_BIN:
    print(f"\n运行 glafic 命令行工具...")
    print(f"  命令: {GLAFIC_BIN} {verify_input_file}")

    try:
        result = subprocess.run(
            [GLAFIC_BIN, os.path.basename(verify_input_file)],
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
    except Exception as e:
        print(f"  警告: glafic 运行出错: {e}")
else:
    print(f"\n跳过 glafic 验证（未找到可执行文件）")

# 读取 glafic 输出
verify_output_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_verify_point.dat')

if GLAFIC_BIN and os.path.exists(verify_output_file):
    print(f"\n读取 glafic 输出: {verify_output_file}")
    
    try:
        data = np.loadtxt(verify_output_file)
        
        # 检查数据维度
        if len(data.shape) == 1:
            n_images_glafic = int(data[0])
            print(f"  警告: glafic 报告 {n_images_glafic} 个图像，但无图像数据")
        else:
            n_images_glafic = int(data[0, 0])
            print(f"  glafic 找到 {n_images_glafic} 个图像")
            
            if n_images_glafic in (4, 5):
                # 提取图像数据（跳过第一行）
                image_data_glafic = data[1:n_images_glafic + 1, :]

                if n_images_glafic == 5:
                    # anfw 等模型产生额外中心像，剔除 |μ| 最小的一个
                    abs_mags = np.abs(image_data_glafic[:, 2])
                    drop_idx = int(np.argmin(abs_mags))
                    print(f"  Info: 5 images found, dropped central image "
                          f"(index {drop_idx}, |μ|={abs_mags[drop_idx]:.4f})")
                    image_data_glafic = np.delete(image_data_glafic, drop_idx, axis=0)

                glafic_positions = image_data_glafic[:, 0:2].copy()  # 列0,1是x,y位置
                glafic_magnifications = np.abs(image_data_glafic[:, 2])  # 列2是放大率

                # 应用中心偏移校正（与 Python 接口一致）
                glafic_positions[:, 0] += center_offset_x
                glafic_positions[:, 1] += center_offset_y

                print(f"  已应用中心偏移校正")
                
                # 匹配图像（使用最近邻）
                distances = cdist(obs_positions, glafic_positions)
                row_ind, col_ind = linear_sum_assignment(distances)
                
                # 重新排列以匹配观测值
                glafic_pos_matched = glafic_positions[col_ind[np.argsort(row_ind)]]
                glafic_mag_matched = glafic_magnifications[col_ind[np.argsort(row_ind)]]
                
                # 与 Python 接口结果对比
                print(f"\n对比 Python 接口 vs glafic 命令行:")
                print(f"  {'Img':<5} {'Python x [mas]':<15} {'glafic x [mas]':<15} {'Δx [mas]':<12} "
                      f"{'Python y [mas]':<15} {'glafic y [mas]':<15} {'Δy [mas]':<12}")
                print(f"  {'-'*95}")
                
                max_pos_diff = 0.0
                for i in range(4):
                    py_x = best_pos[i, 0] * 1000  # 转换为 mas
                    py_y = best_pos[i, 1] * 1000
                    gl_x = glafic_pos_matched[i, 0] * 1000
                    gl_y = glafic_pos_matched[i, 1] * 1000
                    
                    diff_x = abs(py_x - gl_x)
                    diff_y = abs(py_y - gl_y)
                    max_pos_diff = max(max_pos_diff, diff_x, diff_y)
                    
                    print(f"  {i+1:<5} {py_x:>13.3f}    {gl_x:>13.3f}    {diff_x:>10.3f}    "
                          f"{py_y:>13.3f}    {gl_y:>13.3f}    {diff_y:>10.3f}")
                
                print(f"\n  {'Img':<5} {'Python μ':<15} {'glafic μ':<15} {'Δμ':<12} {'Δμ [%]':<12}")
                print(f"  {'-'*65}")
                
                max_mag_diff_pct = 0.0
                for i in range(4):
                    py_mag = best_mag[i]
                    gl_mag = glafic_mag_matched[i]
                    diff_mag = abs(py_mag - gl_mag)
                    diff_mag_pct = (diff_mag / abs(py_mag)) * 100 if py_mag != 0 else 0
                    max_mag_diff_pct = max(max_mag_diff_pct, diff_mag_pct)
                    
                    print(f"  {i+1:<5} {py_mag:>13.3f}    {gl_mag:>13.3f}    {diff_mag:>10.3f}    {diff_mag_pct:>10.3f}%")
                
                # 验证结果
                print(f"\n验证结果:")
                pos_tolerance = 0.01  # mas
                mag_tolerance = 0.1   # 百分比
                
                if max_pos_diff < pos_tolerance and max_mag_diff_pct < mag_tolerance:
                    print(f"  一致性验证通过！")
                    print(f"    最大位置差: {max_pos_diff:.6f} mas < {pos_tolerance} mas")
                    print(f"    最大放大率差: {max_mag_diff_pct:.6f}% < {mag_tolerance}%")
                elif max_pos_diff < 1.0 and max_mag_diff_pct < 1.0:
                    print(f"  一致性良好（小差异）")
                    print(f"    最大位置差: {max_pos_diff:.6f} mas")
                    print(f"    最大放大率差: {max_mag_diff_pct:.6f}%")
                else:
                    print(f"  警告: 检测到较大差异")
                    print(f"    最大位置差: {max_pos_diff:.6f} mas")
                    print(f"    最大放大率差: {max_mag_diff_pct:.6f}%")
                    print(f"  建议检查参数或数值精度设置")
                
                # 保存验证报告
                verify_report_file = os.path.join(output_dir, f'{OUTPUT_PREFIX}_verify_report.txt')
                with open(verify_report_file, 'w') as f:
                    f.write("=" * 70 + "\n")
                    f.write("Python 接口 vs glafic 命令行 验证报告\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"active_subhalos: {active_subhalos}\n")
                    f.write(f"Sub-halos 数量: {n_active_subhalos}\n\n")
                    
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
                    
                    f.write("放大率对比:\n")
                    f.write(f"{'Img':<5} {'Python μ':<15} {'glafic μ':<15} {'Δμ':<12} {'Δμ [%]':<12}\n")
                    f.write("-" * 65 + "\n")
                    for i in range(4):
                        py_mag = best_mag[i]
                        gl_mag = glafic_mag_matched[i]
                        diff_mag = abs(py_mag - gl_mag)
                        diff_mag_pct = (diff_mag / abs(py_mag)) * 100 if py_mag != 0 else 0
                        f.write(f"{i+1:<5} {py_mag:>13.3f}    {gl_mag:>13.3f}    {diff_mag:>10.3f}    {diff_mag_pct:>10.3f}%\n")
                    
                    f.write(f"\n最大放大率差: {max_mag_diff_pct:.6f}%\n\n")
                    
                    f.write("验证结论:\n")
                    if max_pos_diff < pos_tolerance and max_mag_diff_pct < mag_tolerance:
                        f.write("一致性验证通过！Python 接口和 glafic 命令行结果高度一致。\n")
                    elif max_pos_diff < 1.0 and max_mag_diff_pct < 1.0:
                        f.write("一致性良好，存在小的数值差异（可能由于数值精度）。\n")
                    else:
                        f.write("警告: 检测到较大差异，建议检查参数设置。\n")
                
                print(f"\n  验证报告已保存: {verify_report_file}")
                
            else:
                print(f"  警告: glafic 找到 {n_images_glafic} 个图像（预期4或5个），跳过验证")
    
    except Exception as e:
        print(f"  读取或解析 glafic 输出时出错: {e}")

else:
    print(f"\n警告: glafic 输出文件不存在: {verify_output_file}")
    print(f"  跳过验证步骤")

print("\n" + "=" * 70)
print("Version Point Mass 1.0 完成")
print("=" * 70)

