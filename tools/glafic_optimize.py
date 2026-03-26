#!/usr/bin/env python3
"""
glafic_optimize.py

功能：
1. 从指定文件夹读取 *_best_params.txt 文件
2. 给每个 sub-halo 的参数加上 1% 扰动 (0.99~1.1 随机范围)
3. 生成 glafic 输入文件
4. 运行 glafic 并启用其内置的 optimize 或 MCMC 功能
5. 输出结果，并与源文件比较
6. 生成三联图（调用现有绘图组件）

支持的模型类型：
- pointmass: 点质量模型
- nfw: NFW 模型
- p_jaffe: Pseudo-Jaffe 模型

用法：
    python glafic_optimize.py <input_folder> [--model_type <type>] [--output_dir <dir>]
    python glafic_optimize.py <input_folder> --mcmc  # 使用MCMC采样
                                            --max_restart <N>  # amoeba优化最大重启次数（默认 3，设为0禁用重启，-1无限重启）
                                            --verbose  # 详细输出模式（实时打印glafic输出）
                                            2>&1 # 重定向错误输出到标准输出
作者：自动生成
日期：2026-01-25
"""

import os
import sys
import re
import glob
import subprocess
import argparse
import numpy as np
from datetime import datetime

# 添加 glade 运行时环境
GLADE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GLADE_ROOT)
from runtime_env import setup_runtime_env  # noqa: E402
setup_runtime_env(GLADE_ROOT)

# ==================== 常量定义 ====================

# iPTF16geu 基础透镜模型参数（从 version_p_jaffe_2_0.py 提取）
BASE_LENS_PARAMS = {
    'sers1': (1, 'sers', 0.216, 9.896617e+09, 2.656977e-03, 2.758473e-02, 
              2.986760e-01, 1.124730e+02, 3.939718e-01, 1.057760e+00),
    'sers2': (2, 'sers', 0.216, 2.555580e+10, 2.656977e-03, 2.758473e-02, 
              4.242340e-01, 5.396370e+01, 1.538855e+00, 1.000000e+00),
    'sie': (3, 'sie', 0.216, 1.183382e+02, 2.656977e-03, 2.758473e-02, 
            1.571203e-01, 2.920348e+01, 0.000000e+00, 0.000000e+00)
}

# 观测数据 - 原始观测位置 (mas)
OBS_POSITIONS_MAS = np.array([
    [-266.035, +0.427],    # Image 1
    [+118.835, -221.927],  # Image 2
    [+238.324, +227.270],  # Image 3
    [-126.157, +319.719]   # Image 4
])

# 坐标系转换：X轴取反，转换为arcsec
OBS_POSITIONS = np.zeros_like(OBS_POSITIONS_MAS)
OBS_POSITIONS[:, 0] = -OBS_POSITIONS_MAS[:, 0] / 1000.0  # X轴取反
OBS_POSITIONS[:, 1] = OBS_POSITIONS_MAS[:, 1] / 1000.0

# 中心偏移（glafic坐标系原点与观测坐标系原点的偏移）
CENTER_OFFSET_X = -0.01535000  # arcsec
CENTER_OFFSET_Y = +0.03220000  # arcsec

# glafic坐标系中的观测位置（用于obs_point.dat）
OBS_POSITIONS_GLAFIC = OBS_POSITIONS.copy()
OBS_POSITIONS_GLAFIC[:, 0] -= CENTER_OFFSET_X
OBS_POSITIONS_GLAFIC[:, 1] -= CENTER_OFFSET_Y

# 放大率（注意正负号：负号表示负宇称像）
OBS_MAGNIFICATIONS = np.array([-35.6, 15.7, -7.5, 9.1])  # 带正负号
OBS_MAG_ERRORS = np.array([2.1, 1.3, 1.0, 1.1])  # 修正误差值
OBS_POSITION_ERRORS = np.array([0.00041, 0.00086, 0.00223, 0.00311])  # arcsec (从mas转换)

# 源位置和红移
SOURCE_Z = 0.409
SOURCE_X = 2.685497e-03
SOURCE_Y = 2.443616e-02

# 宇宙学参数
OMEGA = 0.3
LAMBDA_COSMO = 0.7
WEOS = -1.0
HUBBLE = 0.7

# 透镜红移
LENS_Z = 0.216

# 网格设置
XMIN, XMAX = -0.5, 0.5
YMIN, YMAX = -0.5, 0.5
PIX_EXT = 0.01
PIX_POI = 0.2
MAXLEV = 5

# MCMC 默认参数（与 p_jaffe_2.0 类似）
MCMC_NSTEPS = 50000        # 默认采样步数
MCMC_SIGMA_POSITION = 0.005  # 位置步长 (arcsec)
MCMC_SIGMA_MASS_LOG = -0.02  # 质量步长（负号表示log-scale）
MCMC_SIGMA_SIG = 1.0         # σ步长 (km/s)
MCMC_SIGMA_A = 0.005         # a步长 (arcsec)
MCMC_SIGMA_RCO = 0.0005      # rco步长 (arcsec)
MCMC_SIGMA_CVIR = 1.0        # c_vir步长
MCMC_SIGMA_SOURCE = 0.001    # 源位置步长 (arcsec)

# ==================== glafic Initial Guess 配置 ====================
# 若 GLAFIC_GUESS=True，跳过 Python 端随机扰动：
#   - prior.dat 中的 range 范围收紧为 最优解 ± GLAFIC_GUESS_PERTURB
#   - input 文件命令段在 optimize 前插入 randomize（或 opt_explore）
#   - glafic 自行在该范围内随机选取初始点
# 若 GLAFIC_GUESS=False（默认），沿用原有 Python 端随机扰动行为
GLAFIC_GUESS         = True   # 是否启用 glafic 内置 initial guess
GLAFIC_GUESS_PERTURB = 0.10    # 参数范围：最优解 ± 此比例（0.10 = ±10%）
# opt_explore 模式：>0 时使用 opt_explore N c2lim 代替 randomize+optimize
# opt_explore 会在每次 randomize 后 optimize，取 N 次中最优结果
GLAFIC_GUESS_N_EXPLORE = 20     # 0 = 仅 randomize 一次；>0 = opt_explore N 次
GLAFIC_GUESS_C2LIM     = 100  # opt_explore 的 chi2 上限（只记录低于此值的结果）


# ==================== 参数解析函数 ====================

def parse_pointmass_params(content):
    """解析 pointmass 类型的 best_params.txt"""
    params = {}
    subhalos = []
    
    # 匹配 x_sub, y_sub, mass_sub 格式
    pattern_x = r'x_sub(\d+)\s*=\s*([\d.eE+-]+)'
    pattern_y = r'y_sub(\d+)\s*=\s*([\d.eE+-]+)'
    pattern_m = r'mass_sub(\d+)\s*=\s*([\d.eE+-]+)'
    
    x_vals = {int(m.group(1)): float(m.group(2)) for m in re.finditer(pattern_x, content)}
    y_vals = {int(m.group(1)): float(m.group(2)) for m in re.finditer(pattern_y, content)}
    m_vals = {int(m.group(1)): float(m.group(2)) for m in re.finditer(pattern_m, content)}
    
    for idx in sorted(x_vals.keys()):
        if idx in x_vals and idx in y_vals and idx in m_vals:
            subhalos.append({
                'idx': idx,
                'x': x_vals[idx],
                'y': y_vals[idx],
                'mass': m_vals[idx]
            })
    
    # 解析性能指标
    chi2_base_match = re.search(r'chi2_base\s*=\s*([\d.]+)', content)
    chi2_best_match = re.search(r'chi2_best\s*=\s*([\d.]+)', content)
    
    params['chi2_base'] = float(chi2_base_match.group(1)) if chi2_base_match else None
    params['chi2_best'] = float(chi2_best_match.group(1)) if chi2_best_match else None
    params['subhalos'] = subhalos
    params['model_type'] = 'pointmass'
    
    return params


def parse_nfw_params(content):
    """解析 NFW 类型的 best_params.txt"""
    params = {}
    subhalos = []
    
    # 匹配 NFW 参数格式
    pattern_x = r'x_nfw(\d+)\s*=\s*([\d.eE+-]+)'
    pattern_y = r'y_nfw(\d+)\s*=\s*([\d.eE+-]+)'
    pattern_m = r'm_vir(\d+)\s*=\s*([\d.eE+-]+)'
    pattern_c = r'c_vir(\d+)\s*=\s*([\d.eE+-]+)'
    
    x_vals = {int(m.group(1)): float(m.group(2)) for m in re.finditer(pattern_x, content)}
    y_vals = {int(m.group(1)): float(m.group(2)) for m in re.finditer(pattern_y, content)}
    m_vals = {int(m.group(1)): float(m.group(2)) for m in re.finditer(pattern_m, content)}
    c_vals = {int(m.group(1)): float(m.group(2)) for m in re.finditer(pattern_c, content)}
    
    for idx in sorted(x_vals.keys()):
        if idx in x_vals and idx in y_vals and idx in m_vals and idx in c_vals:
            subhalos.append({
                'idx': idx,
                'x': x_vals[idx],
                'y': y_vals[idx],
                'm_vir': m_vals[idx],
                'c_vir': c_vals[idx]
            })
    
    chi2_base_match = re.search(r'chi2_base\s*=\s*([\d.]+)', content)
    chi2_best_match = re.search(r'chi2_best\s*=\s*([\d.]+)', content)
    
    params['chi2_base'] = float(chi2_base_match.group(1)) if chi2_base_match else None
    params['chi2_best'] = float(chi2_best_match.group(1)) if chi2_best_match else None
    params['subhalos'] = subhalos
    params['model_type'] = 'nfw'
    
    return params


def parse_p_jaffe_params(content):
    """解析 Pseudo-Jaffe 类型的 best_params.txt"""
    params = {}
    subhalos = []
    
    # 匹配 Pseudo-Jaffe 参数格式
    pattern_x = r'x_jaffe(\d+)\s*=\s*([\d.eE+-]+)'
    pattern_y = r'y_jaffe(\d+)\s*=\s*([\d.eE+-]+)'
    pattern_sig = r'sig(\d+)\s*=\s*([\d.eE+-]+)'
    pattern_a = r'^a(\d+)\s*=\s*([\d.eE+-]+)'
    pattern_rco = r'rco(\d+)\s*=\s*([\d.eE+-]+)'
    
    x_vals = {int(m.group(1)): float(m.group(2)) for m in re.finditer(pattern_x, content)}
    y_vals = {int(m.group(1)): float(m.group(2)) for m in re.finditer(pattern_y, content)}
    sig_vals = {int(m.group(1)): float(m.group(2)) for m in re.finditer(pattern_sig, content)}
    a_vals = {int(m.group(1)): float(m.group(2)) for m in re.finditer(pattern_a, content, re.MULTILINE)}
    # rco需要排除 rco/a 行
    rco_vals = {}
    for m in re.finditer(pattern_rco, content):
        idx = int(m.group(1))
        # 检查是否是 rco/a 格式
        start = m.start()
        if start > 0 and content[start-1] != '/':
            rco_vals[idx] = float(m.group(2))
    
    for idx in sorted(x_vals.keys()):
        if idx in x_vals and idx in y_vals and idx in sig_vals:
            a_val = a_vals.get(idx, 0.01)  # 默认值
            rco_val = rco_vals.get(idx, 0.001)  # 默认值
            subhalos.append({
                'idx': idx,
                'x': x_vals[idx],
                'y': y_vals[idx],
                'sig': sig_vals[idx],
                'a': a_val,
                'rco': rco_val
            })
    
    chi2_base_match = re.search(r'chi2_base\s*=\s*([\d.]+)', content)
    chi2_best_match = re.search(r'chi2_best\s*=\s*([\d.]+)', content)
    
    params['chi2_base'] = float(chi2_base_match.group(1)) if chi2_base_match else None
    params['chi2_best'] = float(chi2_best_match.group(1)) if chi2_best_match else None
    params['subhalos'] = subhalos
    params['model_type'] = 'p_jaffe'
    
    return params


def detect_model_type(content):
    """自动检测模型类型"""
    if 'x_jaffe' in content or 'Pseudo-Jaffe' in content:
        return 'p_jaffe'
    elif 'x_nfw' in content or 'm_vir' in content or 'NFW' in content:
        return 'nfw'
    elif 'x_sub' in content or 'mass_sub' in content or 'Point Mass' in content:
        return 'pointmass'
    else:
        return None


def parse_best_params(filepath):
    """解析 best_params.txt 文件"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    model_type = detect_model_type(content)
    
    if model_type == 'pointmass':
        return parse_pointmass_params(content)
    elif model_type == 'nfw':
        return parse_nfw_params(content)
    elif model_type == 'p_jaffe':
        return parse_p_jaffe_params(content)
    else:
        raise ValueError(f"无法识别模型类型: {filepath}")


# ==================== 参数扰动函数 ====================

def perturb_params(params, perturb_range=(0.99, 1.1)):
    """
    给每个 sub-halo 的参数加上扰动
    扰动范围: 参数 * random(perturb_min, perturb_max)
    
    如果 perturb_range = (1.0, 1.0)，则不进行任何扰动
    """
    perturbed = {
        'model_type': params['model_type'],
        'chi2_base': params['chi2_base'],
        'chi2_best': params['chi2_best'],
        'subhalos': []
    }
    
    # 检查是否需要扰动
    no_perturbation = (perturb_range[0] == 1.0 and perturb_range[1] == 1.0)
    
    for sh in params['subhalos']:
        new_sh = {'idx': sh['idx']}
        
        if no_perturbation:
            # 不进行扰动，直接复制
            new_sh['x'] = sh['x']
            new_sh['y'] = sh['y']
        else:
            # 位置使用小偏移（基于扰动范围大小）
            # 扰动范围 (0.99, 1.1) -> 偏移范围约 ±5%
            perturbation_scale = (perturb_range[1] - perturb_range[0]) / 2
            pos_perturbation = perturbation_scale * 0.1  # ~1% of arcsec for typical range
            new_sh['x'] = sh['x'] + np.random.uniform(-pos_perturbation, pos_perturbation)
            new_sh['y'] = sh['y'] + np.random.uniform(-pos_perturbation, pos_perturbation)
        
        if params['model_type'] == 'pointmass':
            factor = np.random.uniform(*perturb_range)
            new_sh['mass'] = sh['mass'] * factor
            
        elif params['model_type'] == 'nfw':
            new_sh['m_vir'] = sh['m_vir'] * np.random.uniform(*perturb_range)
            new_sh['c_vir'] = sh['c_vir'] * np.random.uniform(*perturb_range)
            
        elif params['model_type'] == 'p_jaffe':
            new_sh['sig'] = sh['sig'] * np.random.uniform(*perturb_range)
            new_sh['a'] = sh['a'] * np.random.uniform(*perturb_range)
            new_sh['rco'] = sh['rco'] * np.random.uniform(*perturb_range)
        
        perturbed['subhalos'].append(new_sh)
    
    return perturbed


# ==================== glafic 输入文件生成 ====================

def _tight_bounds(val, perturb):
    """计算 initial guess 模式下的紧约束范围 [val*(1-p), val*(1+p)]。
    对位置等带符号参数使用绝对值偏移，避免正负颠倒。"""
    offset = abs(val) * perturb
    if offset == 0:
        offset = 1e-8  # 防止 val=0 时范围退化
    return val - offset, val + offset


def generate_glafic_input(params, output_prefix, output_dir, use_mcmc=False,
                          mcmc_nsteps=MCMC_NSTEPS, verbose=False, max_restart=3,
                          glafic_guess=GLAFIC_GUESS,
                          glafic_guess_perturb=GLAFIC_GUESS_PERTURB,
                          glafic_guess_n_explore=GLAFIC_GUESS_N_EXPLORE,
                          glafic_guess_c2lim=GLAFIC_GUESS_C2LIM):
    """生成 glafic 输入文件
    
    参数:
        max_restart: amoeba优化重启次数限制
            -1 = 无限重启直到收敛
             0 = 不重启，只运行一次
            >0 = 最多重启N次
    
    注意：glafic MCMC 只优化 lens 参数，不优化 point source 的 x,y
    """
    model_type = params['model_type']
    subhalos = params['subhalos']
    n_subhalos = len(subhalos)
    n_lenses = 3 + n_subhalos  # 3个基础透镜 + sub-halos
    
    input_file = os.path.join(output_dir, f'{output_prefix}_optimize.input')
    obs_file = os.path.join(output_dir, 'obs_point.dat')
    prior_file = os.path.join(output_dir, 'prior.dat')
    sigma_file = os.path.join(output_dir, 'mcmc_sigma.dat') if use_mcmc else None
    
    # 生成主输入文件
    with open(input_file, 'w') as f:
        f.write(f"## glafic_optimize.py 自动生成\n")
        f.write(f"## 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"## 模型类型: {model_type}\n")
        f.write(f"## Sub-halos 数量: {n_subhalos}\n")
        f.write(f"## 模式: {'MCMC' if use_mcmc else 'Optimize (Amoeba)'}\n\n")
        
        # 宇宙学参数
        f.write("## Cosmological parameters\n")
        f.write(f"omega     {OMEGA}\n")
        f.write(f"lambda    {LAMBDA_COSMO}\n")
        f.write(f"weos      {WEOS}\n")
        f.write(f"hubble    {HUBBLE}\n")
        f.write(f"prefix    {output_prefix}\n")
        f.write(f"xmin      {XMIN}\n")
        f.write(f"ymin      {YMIN}\n")
        f.write(f"xmax      {XMAX}\n")
        f.write(f"ymax      {YMAX}\n")
        f.write(f"pix_ext   {PIX_EXT}\n")
        f.write(f"pix_poi   {PIX_POI}\n")
        f.write(f"maxlev    {MAXLEV}\n\n")
        
        # 二级参数
        f.write("## Secondary parameters\n")
        f.write("chi2_splane    0\n")       # 0=image plane chi2, 1=source plane chi2
        f.write("chi2_checknimg 1\n")       # 1=要求图像数量匹配
        f.write(f"chi2_restart   {max_restart}\n")  # 优化重启次数: -1=无限, 0=不重启, >0=最多N次
        f.write("chi2_usemag    -1\n")      # -1=直接比较放大率（不是星等）
        f.write("hvary          0\n")
        if verbose:
            f.write("flag_mcmcall   1\n")   # 输出所有MCMC样本（包括reject）
        f.write("\n")
        
        # 透镜和源定义
        f.write(f"## Lens and source definition ({n_lenses} lenses, 0 extended, 1 point)\n")
        f.write(f"startup {n_lenses} 0 1\n\n")
        
        # 基础透镜模型
        f.write("# Base lens model (Sersic + Sersic + SIE)\n")
        for key in ['sers1', 'sers2', 'sie']:
            lens = BASE_LENS_PARAMS[key]
            f.write(f"lens {lens[1]}   {lens[2]}    {lens[3]:.6e}    {lens[4]:.6e}    ")
            f.write(f"{lens[5]:.6e}    {lens[6]:.6e}    {lens[7]:.6e}    {lens[8]:.6e}    {lens[9]:.6e}\n")
        f.write("\n")
        
        # Sub-halos
        f.write(f"# Sub-halos ({model_type})\n")
        for sh in subhalos:
            if model_type == 'pointmass':
                # point lens: z, mass, x, y, e, pa, r0, n (后4个为0)
                f.write(f"lens point  {LENS_Z}    {sh['mass']:.10e}    {sh['x']:.10e}    ")
                f.write(f"{sh['y']:.10e}    0.0    0.0    0.0    0.0\n")
            elif model_type == 'nfw':
                # nfw: z, m_vir, x, y, e, pa, c_vir, n (n未使用)
                f.write(f"lens nfw  {LENS_Z}    {sh['m_vir']:.10e}    {sh['x']:.10e}    ")
                f.write(f"{sh['y']:.10e}    0.0    0.0    {sh['c_vir']:.10e}    0.0\n")
            elif model_type == 'p_jaffe':
                # jaffe: z, sig, x, y, e, pa, a, rco
                f.write(f"lens jaffe  {LENS_Z}    {sh['sig']:.10e}    {sh['x']:.10e}    ")
                f.write(f"{sh['y']:.10e}    0.0    0.0    {sh['a']:.10e}    {sh['rco']:.10e}\n")
        f.write("\n")
        
        # 点源
        f.write("# Point source (iPTF16geu supernova)\n")
        f.write(f"point {SOURCE_Z}    {SOURCE_X:.10e}    {SOURCE_Y:.10e}\n\n")
        
        f.write("end_startup\n\n")
        
        # 优化标志
        f.write("## Optimization flags (setopt)\n")
        f.write("## lens: z, mass/sig, x, y, e, pa, rs/a, n/rco\n")
        f.write("## Format: 0=fixed, 1=optimize\n")
        f.write("start_setopt\n")
        
        # 基础透镜固定
        for _ in range(3):
            f.write("0 0 0 0 0 0 0 0\n")
        
        # Sub-halos 可优化（除了 z, e, pa）
        for _ in range(n_subhalos):
            if model_type == 'pointmass':
                f.write("0 1 1 1 0 0 0 0\n")  # mass, x, y
            elif model_type == 'nfw':
                f.write("0 1 1 1 0 0 1 0\n")  # m_vir, x, y, c_vir
            elif model_type == 'p_jaffe':
                f.write("0 1 1 1 0 0 1 1\n")  # sig, x, y, a, rco
        
        # 点源位置
        # MCMC 模式下不优化点源位置（glafic MCMC 不支持）
        if use_mcmc:
            f.write("0 0 0\n")  # z, x, y 全部固定
        else:
            f.write("0 1 1\n")  # z固定，x,y可优化
        f.write("end_setopt\n\n")
        
        # 命令
        f.write("## Execute commands\n")
        f.write("start_command\n\n")
        f.write(f"readobs_point obs_point.dat\n")
        f.write(f"parprior prior.dat\n\n")
        
        if use_mcmc:
            f.write("# Load MCMC step sizes\n")
            f.write("mcmc_sigma mcmc_sigma.dat\n\n")
            f.write(f"# Run MCMC ({mcmc_nsteps} steps)\n")
            f.write(f"mcmc {mcmc_nsteps}\n\n")
        elif glafic_guess:
            if glafic_guess_n_explore > 0:
                # opt_explore: N 次 randomize+optimize，取最优
                f.write(f"# [glafic_guess] opt_explore: {glafic_guess_n_explore} cycles of randomize+optimize\n")
                f.write(f"opt_explore {glafic_guess_n_explore} {glafic_guess_c2lim:.2e}\n\n")
            else:
                # 单次 randomize + optimize
                f.write("# [glafic_guess] randomize within tight prior range, then optimize\n")
                f.write("randomize\n")
                f.write("optimize\n\n")
        else:
            f.write("# Run optimization (amoeba/simplex)\n")
            f.write("optimize\n\n")
        
        f.write("# Find images with optimized parameters\n")
        f.write("findimg\n\n")
        f.write("# Write critical curves (requires source redshift)\n")
        f.write(f"writecrit {SOURCE_Z}\n\n")
        f.write("# Print final model\n")
        f.write("printmodel\n\n")
        f.write("quit\n")
    
    # 生成观测数据文件
    # glafic obs_point.dat 格式: x y flux sig_pos sig_flux td sig_td parity
    # 注意：
    # 1. 使用glafic坐标系中的观测位置（减去中心偏移）
    # 2. 如果不使用时间延迟，td=0, sig_td=0 (glafic会忽略sig_td=0的约束)
    # 3. 放大率的正负号很重要（负号表示负宇称像）
    with open(obs_file, 'w') as f:
        f.write("# iPTF16geu observed point source images\n")
        f.write("# Format: x y flux sig_pos sig_flux td sig_td parity\n")
        f.write(f"# 注意：位置已转换到glafic坐标系（减去中心偏移）\n")
        f.write(f"1 4 {SOURCE_Z} 0.0\n")
        for i, (pos, mag, mag_err, pos_err) in enumerate(zip(
            OBS_POSITIONS_GLAFIC, OBS_MAGNIFICATIONS, OBS_MAG_ERRORS, OBS_POSITION_ERRORS)):
            # parity: 根据放大率正负号确定
            parity = -1 if mag < 0 else 1
            # x, y, flux(|mag|), sig_pos, sig_flux(mag_err), td=0, sig_td=0, parity
            f.write(f"    {pos[0]:9.6f}     {pos[1]:9.6f}  {abs(mag):.1f} {pos_err:.6f} {mag_err:.1f}   ")
            f.write(f"0.0 0.0    {parity} # {i+1}\n")
    
    # 生成先验约束文件
    with open(prior_file, 'w') as f:
        f.write("# Parameter ranges for optimization\n")
        f.write("# Format: range lens/extend/point id param_no min max\n")
        if glafic_guess:
            f.write(f"# [glafic_guess 模式] 紧约束范围 = 最优解 ± {glafic_guess_perturb*100:.0f}%\n")
        f.write("\n")

        for i, sh in enumerate(subhalos):
            lens_id = 4 + i  # 从第4个开始是sub-halo

            if glafic_guess:
                # ── 紧约束：最优解 ± GLAFIC_GUESS_PERTURB ──────────────
                p = glafic_guess_perturb
                if model_type == 'pointmass':
                    mlo, mhi = _tight_bounds(sh['mass'], p)
                    f.write(f"range lens {lens_id} 2 {mlo:.6e} {mhi:.6e}  # mass\n")
                elif model_type == 'nfw':
                    mlo, mhi = _tight_bounds(sh['m_vir'], p)
                    clo, chi = _tight_bounds(sh['c_vir'], p)
                    f.write(f"range lens {lens_id} 2 {mlo:.6e} {mhi:.6e}  # m_vir\n")
                    f.write(f"range lens {lens_id} 7 {max(clo,1e-3):.6e} {chi:.6e}  # c_vir\n")
                elif model_type == 'p_jaffe':
                    slo, shi_ = _tight_bounds(sh['sig'], p)
                    alo, ahi  = _tight_bounds(sh['a'],   p)
                    rlo, rhi  = _tight_bounds(sh['rco'], p)
                    f.write(f"range lens {lens_id} 2 {max(slo,1e-6):.6e} {shi_:.6e}  # sig\n")
                    f.write(f"range lens {lens_id} 7 {max(alo,1e-7):.6e} {ahi:.6e}   # a\n")
                    f.write(f"range lens {lens_id} 8 {max(rlo,1e-8):.6e} {rhi:.6e}   # rco\n")
                # 位置紧约束
                xlo, xhi = _tight_bounds(sh['x'], p)
                ylo, yhi = _tight_bounds(sh['y'], p)
                f.write(f"range lens {lens_id} 3 {xlo:.6e} {xhi:.6e}  # x\n")
                f.write(f"range lens {lens_id} 4 {ylo:.6e} {yhi:.6e}  # y\n")
            else:
                # ── 宽约束（原有行为）────────────────────────────────────
                if model_type == 'pointmass':
                    f.write(f"range lens {lens_id} 2 1.0e-3 1.0e12\n")
                elif model_type == 'nfw':
                    f.write(f"range lens {lens_id} 2 1.0 1.0e12\n")
                    f.write(f"range lens {lens_id} 7 1.0 100.0\n")
                elif model_type == 'p_jaffe':
                    f.write(f"range lens {lens_id} 2 0.1 50.0\n")
                    f.write(f"range lens {lens_id} 7 1.0e-4 0.5\n")
                    f.write(f"range lens {lens_id} 8 1.0e-6 0.5\n")
                f.write(f"range lens {lens_id} 3 {XMIN} {XMAX}\n")
                f.write(f"range lens {lens_id} 4 {YMIN} {YMAX}\n")

        f.write("\n# Point source position range\n")
        f.write("range point 1 2 -0.1 0.1\n")
        f.write("range point 1 3 -0.1 0.1\n")
    
    # 生成MCMC sigma文件（如果启用MCMC）
    if use_mcmc:
        _generate_mcmc_sigma_file(sigma_file, model_type, n_subhalos)
    
    return input_file, obs_file, prior_file, sigma_file


def _generate_mcmc_sigma_file(sigma_file, model_type, n_subhalos):
    """生成MCMC步长文件
    
    格式：
    第1行：参数总数
    后续行：每个参数的步长（正数=线性步长，负数=log步长）
    
    参数顺序（与setopt对应）：
    1. 各透镜的可优化参数
    
    注意：glafic MCMC 只优化 lens 参数，不优化 point source 的 x,y
    (从 opt_lens_calcndim 和 paratopar 可以看出)
    """
    # 计算总参数数
    # 每个sub-halo的参数数量
    if model_type == 'pointmass':
        params_per_subhalo = 3  # mass, x, y
    elif model_type == 'nfw':
        params_per_subhalo = 4  # m_vir, x, y, c_vir
    elif model_type == 'p_jaffe':
        params_per_subhalo = 5  # sig, x, y, a, rco
    else:
        params_per_subhalo = 3
    
    # Note: glafic MCMC does NOT include point source x,y in parameter vector
    # Only lens parameters are included
    n_params = n_subhalos * params_per_subhalo
    
    with open(sigma_file, 'w') as f:
        f.write(f"# MCMC step sizes for {model_type} model\n")
        f.write(f"# Positive = linear step, Negative = log-scale step\n")
        f.write(f"# Note: glafic MCMC only optimizes lens parameters, not point source x,y\n")
        f.write(f"{n_params}\n")
        
        for i in range(n_subhalos):
            if model_type == 'pointmass':
                f.write(f"{MCMC_SIGMA_MASS_LOG}  # sub-halo {i+1} mass (log)\n")
                f.write(f"{MCMC_SIGMA_POSITION}  # sub-halo {i+1} x\n")
                f.write(f"{MCMC_SIGMA_POSITION}  # sub-halo {i+1} y\n")
            elif model_type == 'nfw':
                f.write(f"{MCMC_SIGMA_MASS_LOG}  # sub-halo {i+1} m_vir (log)\n")
                f.write(f"{MCMC_SIGMA_POSITION}  # sub-halo {i+1} x\n")
                f.write(f"{MCMC_SIGMA_POSITION}  # sub-halo {i+1} y\n")
                f.write(f"{MCMC_SIGMA_CVIR}  # sub-halo {i+1} c_vir\n")
            elif model_type == 'p_jaffe':
                f.write(f"{MCMC_SIGMA_SIG}  # sub-halo {i+1} sig\n")
                f.write(f"{MCMC_SIGMA_POSITION}  # sub-halo {i+1} x\n")
                f.write(f"{MCMC_SIGMA_POSITION}  # sub-halo {i+1} y\n")
                f.write(f"{MCMC_SIGMA_A}  # sub-halo {i+1} a\n")
                f.write(f"{MCMC_SIGMA_RCO}  # sub-halo {i+1} rco\n")
    
    return sigma_file


# ==================== glafic 运行和结果解析 ====================

def find_glafic_bin():
    """查找 glafic 可执行文件"""
    # 尝试常见路径
    glafic_home = os.environ.get("GLAFIC_HOME", os.path.join(GLADE_ROOT, "glafic2"))
    possible_paths = [
        os.path.join(glafic_home, 'glafic'),
        '/usr/local/bin/glafic',
        '/usr/bin/glafic',
        'glafic'
    ]
    
    for path in possible_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    
    # 尝试 which
    try:
        result = subprocess.run(['which', 'glafic'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    return None


def run_glafic(input_file, output_dir, verbose=False, timeout=3000):
    """运行 glafic 并返回结果"""
    glafic_bin = find_glafic_bin()
    
    if glafic_bin is None:
        raise RuntimeError("找不到 glafic 可执行文件")
    
    print(f"  使用 glafic: {glafic_bin}")
    
    # 切换到输出目录运行
    original_dir = os.getcwd()
    os.chdir(output_dir)
    
    try:
        if verbose:
            # 实时打印输出
            print("\n=== glafic 实时输出 ===")
            process = subprocess.Popen(
                [glafic_bin, os.path.basename(input_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            stdout_lines = []
            try:
                for line in process.stdout:
                    print(line, end='')
                    stdout_lines.append(line)
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                print("\n[超时] glafic 运行超时")
            
            stdout = ''.join(stdout_lines)
            stderr = ''
            returncode = process.returncode
        else:
            result = subprocess.run(
                [glafic_bin, os.path.basename(input_file)],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode
            
            # 打印输出
            if stdout:
                print("\n=== glafic stdout ===")
                print(stdout[:2000] if len(stdout) > 2000 else stdout)
            
            if stderr:
                print("\n=== glafic stderr ===")
                print(stderr[:2000] if len(stderr) > 2000 else stderr)
        
        return returncode, stdout, stderr
        
    finally:
        os.chdir(original_dir)


def parse_optresult(output_dir, prefix):
    """解析 glafic 优化结果"""
    result_file = os.path.join(output_dir, f'{prefix}_optresult.dat')
    
    if not os.path.exists(result_file):
        print(f"  警告: 结果文件不存在: {result_file}")
        return None
    
    with open(result_file, 'r') as f:
        content = f.read()
    
    # 解析最后一次优化的 chi^2
    chi2_matches = re.findall(r"chi\^2 = ([\d.eE+-]+)", content)
    final_chi2 = float(chi2_matches[-1]) if chi2_matches else None
    
    # 解析最终的透镜参数
    lens_params = []
    point_params = None
    
    # 从最后一个 block 解析（找倒数第二个分隔线之后的内容）
    blocks = content.split('------------------------------------------')
    if len(blocks) >= 2:
        # 最后一个block通常是空的，取倒数第二个
        last_block = blocks[-2] if blocks[-1].strip() == '' else blocks[-1]
        
        for line in last_block.split('\n'):
            line = line.strip()
            # 匹配 lens 行，格式: lens   type   z   p1   x   y   e   pa   p2   p3
            if line.startswith('lens'):
                parts = line.split()
                if len(parts) >= 10:
                    lens_params.append({
                        'type': parts[1],
                        'z': float(parts[2]),
                        'p1': float(parts[3]),
                        'x': float(parts[4]),
                        'y': float(parts[5]),
                        'e': float(parts[6]),
                        'pa': float(parts[7]),
                        'p2': float(parts[8]),
                        'p3': float(parts[9])
                    })
            # 匹配 point 行，格式: point  z  x  y (不能是 "point no")
            elif line.startswith('point') and not line.startswith('point no'):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        point_params = {
                            'z': float(parts[1]),
                            'x': float(parts[2]),
                            'y': float(parts[3])
                        }
                    except ValueError:
                        pass  # 忽略无法解析的行
    
    return {
        'chi2': final_chi2,
        'lens_params': lens_params,
        'point_params': point_params,
        'raw_content': content
    }


def parse_mcmc_result(output_dir, prefix):
    """解析 glafic MCMC 结果"""
    mcmc_file = os.path.join(output_dir, f'{prefix}_mcmc.dat')
    
    if not os.path.exists(mcmc_file):
        print(f"  警告: MCMC结果文件不存在: {mcmc_file}")
        return None
    
    # 读取MCMC链
    data = []
    with open(mcmc_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            # 跳过 "accept"/"reject" 标记
            if parts[0] in ['accept', 'reject']:
                if parts[0] == 'accept':
                    data.append([float(x) for x in parts[1:]])
            else:
                data.append([float(x) for x in parts])
    
    if not data:
        return None
    
    data = np.array(data)
    
    # 第一列是 chi^2
    chi2_chain = data[:, 0]
    params_chain = data[:, 1:]
    
    # 找到最优参数（最小chi^2）
    best_idx = np.argmin(chi2_chain)
    best_chi2 = chi2_chain[best_idx]
    best_params = params_chain[best_idx]
    
    # 统计
    n_samples = len(chi2_chain)
    
    return {
        'chi2': best_chi2,
        'chi2_chain': chi2_chain,
        'params_chain': params_chain,
        'best_params': best_params,
        'n_samples': n_samples,
        'chi2_mean': np.mean(chi2_chain),
        'chi2_std': np.std(chi2_chain)
    }


def parse_point_result(output_dir, prefix):
    """解析 findimg 结果（_point.dat 文件）
    
    文件格式:
    第1行: n_img  z_s  src_x  src_y
    第2行起: x  y  mag  td
    """
    point_file = os.path.join(output_dir, f'{prefix}_point.dat')
    
    if not os.path.exists(point_file):
        print(f"  警告: 点源结果文件不存在: {point_file}")
        return None
    
    with open(point_file, 'r') as f:
        lines = f.readlines()
    
    images = []
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 4 and i > 0:  # 跳过第一行（总结行）
                images.append({
                    'x': float(parts[0]),
                    'y': float(parts[1]),
                    'mag': float(parts[2]),
                    'td': float(parts[3]) if len(parts) > 3 else 0.0
                })
    
    return images


# ==================== 结果比较 ====================

def compare_results(original_params, optimized_result, perturbed_params, use_mcmc=False):
    """比较原始参数与优化后的结果"""
    print("\n" + "=" * 70)
    print("结果比较")
    print("=" * 70)
    
    print(f"\n原始 chi²: {original_params.get('chi2_best', 'N/A')}")
    
    if use_mcmc and optimized_result:
        print(f"MCMC 最优 chi²: {optimized_result.get('chi2', 'N/A')}")
        print(f"MCMC 采样数: {optimized_result.get('n_samples', 'N/A')}")
        print(f"chi² 均值 ± 标准差: {optimized_result.get('chi2_mean', 0):.4f} ± {optimized_result.get('chi2_std', 0):.4f}")
    else:
        print(f"优化后 chi²: {optimized_result.get('chi2', 'N/A') if optimized_result else 'N/A'}")
    
    if optimized_result and optimized_result.get('chi2') and original_params.get('chi2_best'):
        improvement = (original_params['chi2_best'] - optimized_result['chi2']) / original_params['chi2_best'] * 100
        if improvement > 0:
            print(f"改善: {improvement:.2f}%")
        else:
            print(f"变化: {improvement:.2f}%（无改善或变差）")
    
    print("\nSub-halo 参数比较:")
    print("-" * 70)
    
    model_type = original_params['model_type']
    opt_lenses = optimized_result.get('lens_params', []) if optimized_result else []
    
    # 跳过前3个基础透镜
    subhalo_lenses = opt_lenses[3:] if len(opt_lenses) > 3 else []
    
    for i, (orig, pert) in enumerate(zip(original_params['subhalos'], perturbed_params['subhalos'])):
        print(f"\nSub-halo {orig['idx']}:")
        
        opt = subhalo_lenses[i] if i < len(subhalo_lenses) else None
        
        print(f"  位置 (x, y):")
        print(f"    原始:    ({orig['x']:.6f}, {orig['y']:.6f})")
        print(f"    扰动后:  ({pert['x']:.6f}, {pert['y']:.6f})")
        if opt:
            print(f"    优化后:  ({opt['x']:.6f}, {opt['y']:.6f})")
        
        if model_type == 'pointmass':
            print(f"  质量 (Msun):")
            print(f"    原始:    {orig['mass']:.4e}")
            print(f"    扰动后:  {pert['mass']:.4e}")
            if opt:
                print(f"    优化后:  {opt['p1']:.4e}")
                
        elif model_type == 'nfw':
            print(f"  M_vir (Msun):")
            print(f"    原始:    {orig['m_vir']:.4e}")
            print(f"    扰动后:  {pert['m_vir']:.4e}")
            if opt:
                print(f"    优化后:  {opt['p1']:.4e}")
            print(f"  c_vir:")
            print(f"    原始:    {orig['c_vir']:.4f}")
            print(f"    扰动后:  {pert['c_vir']:.4f}")
            if opt:
                print(f"    优化后:  {opt['p2']:.4f}")
                
        elif model_type == 'p_jaffe':
            print(f"  σ (km/s):")
            print(f"    原始:    {orig['sig']:.4f}")
            print(f"    扰动后:  {pert['sig']:.4f}")
            if opt:
                print(f"    优化后:  {opt['p1']:.4f}")
            print(f"  a (arcsec):")
            print(f"    原始:    {orig['a']:.6f}")
            print(f"    扰动后:  {pert['a']:.6f}")
            if opt:
                print(f"    优化后:  {opt['p2']:.6f}")
            print(f"  rco (arcsec):")
            print(f"    原始:    {orig['rco']:.6f}")
            print(f"    扰动后:  {pert['rco']:.6f}")
            if opt:
                print(f"    优化后:  {opt['p3']:.6f}")


# ==================== 三联图生成 ====================

def generate_triptych(params, optimized_result, output_dir, output_prefix):
    """生成三联图"""
    model_type = params['model_type']
    
    # 尝试导入对应的绘图模块
    # 添加版本目录到 Python 路径
    work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    plot_module = None
    try:
        if model_type == 'p_jaffe':
            jaffe_dir = os.path.join(work_dir, 'legacy', 'v_p_jaffe_2_0')
            if os.path.exists(jaffe_dir):
                sys.path.insert(0, jaffe_dir)
            from plot_paper_style import (
                plot_paper_style_nfw, read_critical_curves
            )
            plot_module = 'p_jaffe'
            print(f"  使用 v_p_jaffe_2.0 的绘图模块: {jaffe_dir}")
        elif model_type == 'nfw':
            nfw_dir = os.path.join(work_dir, 'legacy', 'v_nfw_2_0')
            if os.path.exists(nfw_dir):
                sys.path.insert(0, nfw_dir)
            from plot_paper_style import (
                plot_paper_style_nfw, read_critical_curves
            )
            plot_module = 'nfw'
            print(f"  使用 v_nfw_2.0 的绘图模块: {nfw_dir}")
        else:
            pm_dir = os.path.join(work_dir, 'legacy', 'v_pointmass_1_0')
            if os.path.exists(pm_dir):
                sys.path.insert(0, pm_dir)
            from plot_paper_style import (
                plot_paper_style, read_critical_curves
            )
            plot_module = 'pointmass'
            print(f"  使用 v_pointmass_1.0 的绘图模块: {pm_dir}")
    except ImportError as e:
        print(f"  警告: 无法导入绘图模块: {e}")
        print("  跳过三联图生成")
        return None
    
    # 读取临界曲线
    crit_file = os.path.join(output_dir, f'{output_prefix}_crit.dat')
    crit_segments = []
    caus_segments = []
    
    if os.path.exists(crit_file):
        # 检查文件是否为空
        if os.path.getsize(crit_file) > 0:
            try:
                crit_segments_glafic, caus_segments_glafic = read_critical_curves(crit_file)
                
                # 将临界曲线从 glafic 坐标系转换到偏移后坐标系
                for seg in crit_segments_glafic:
                    new_seg = [[seg[0][0] + CENTER_OFFSET_X, seg[0][1] + CENTER_OFFSET_Y],
                               [seg[1][0] + CENTER_OFFSET_X, seg[1][1] + CENTER_OFFSET_Y]]
                    crit_segments.append(new_seg)
                
                for seg in caus_segments_glafic:
                    new_seg = [[seg[0][0] + CENTER_OFFSET_X, seg[0][1] + CENTER_OFFSET_Y],
                               [seg[1][0] + CENTER_OFFSET_X, seg[1][1] + CENTER_OFFSET_Y]]
                    caus_segments.append(new_seg)
                    
            except Exception as e:
                print(f"  警告: 读取临界曲线失败: {e}")
        else:
            print(f"  警告: 临界曲线文件为空: {crit_file}")
    else:
        print(f"  警告: 临界曲线文件不存在: {crit_file}")
    
    # 解析预测的图像位置
    images = parse_point_result(output_dir, output_prefix)
    if images is None or len(images) < 4:
        print("  警告: 无法获取足够的图像位置")
        return None
    
    # glafic 输出的位置是 glafic 坐标系
    pred_positions_glafic = np.array([[img['x'], img['y']] for img in images[:4]])
    pred_magnifications = np.array([abs(img['mag']) for img in images[:4]])  # 优化只保留绝对值
    
    # 将 glafic 预测位置转换到偏移后坐标系（与观测位置 OBS_POSITIONS 一致）
    pred_positions = pred_positions_glafic.copy()
    pred_positions[:, 0] += CENTER_OFFSET_X
    pred_positions[:, 1] += CENTER_OFFSET_Y
    
    # 计算位置偏差（使用偏移后坐标系）
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    
    distances = cdist(OBS_POSITIONS, pred_positions)
    row_ind, col_ind = linear_sum_assignment(distances)
    
    pred_pos_matched = pred_positions[col_ind[np.argsort(row_ind)]]
    pred_mag_matched = pred_magnifications[col_ind[np.argsort(row_ind)]]
    
    delta_pos_arcsec = np.sqrt(np.sum((OBS_POSITIONS - pred_pos_matched)**2, axis=1))
    delta_pos_mas = delta_pos_arcsec * 1000  # 转换为 mas
    
    # 准备 sub-halo 位置（转换到偏移后坐标系）
    subhalo_positions = []
    for sh in params['subhalos']:
        # 原始位置是 glafic 坐标系，需要加上中心偏移
        x_offset = sh['x'] + CENTER_OFFSET_X
        y_offset = sh['y'] + CENTER_OFFSET_Y
        if model_type == 'pointmass':
            subhalo_positions.append((x_offset, y_offset, sh['mass']))
        elif model_type == 'nfw':
            subhalo_positions.append((x_offset, y_offset, sh['m_vir'], sh['c_vir']))
        elif model_type == 'p_jaffe':
            subhalo_positions.append((x_offset, y_offset, sh['sig'], sh['a'], sh['rco']))
    
    # 生成图片
    output_file = os.path.join(output_dir, f'{output_prefix}_triptych.png')
    
    # 放大率使用绝对值（glafic 优化只保留绝对值）
    obs_mag_abs = np.abs(OBS_MAGNIFICATIONS)
    
    try:
        if model_type == 'pointmass':
            plot_paper_style(
                img_numbers=[1, 2, 3, 4],
                delta_pos_mas=delta_pos_mas,
                sigma_pos_mas=OBS_POSITION_ERRORS * 1000,
                mu_obs=obs_mag_abs,
                mu_obs_err=OBS_MAG_ERRORS,
                mu_pred=pred_mag_matched,
                mu_at_obs_pred=pred_mag_matched,
                obs_positions_arcsec=OBS_POSITIONS,  # 偏移后坐标系
                pred_positions_arcsec=pred_pos_matched,  # 偏移后坐标系
                crit_segments=crit_segments,
                caus_segments=caus_segments if caus_segments else None,
                suptitle=f"glafic Optimized: {model_type.upper()} Model",
                output_file=output_file,
                subhalo_positions=subhalo_positions,
                show_2sigma=True
            )
        else:
            plot_paper_style_nfw(
                img_numbers=[1, 2, 3, 4],
                delta_pos_mas=delta_pos_mas,
                sigma_pos_mas=OBS_POSITION_ERRORS * 1000,
                mu_obs=obs_mag_abs,
                mu_obs_err=OBS_MAG_ERRORS,
                mu_pred=pred_mag_matched,
                mu_at_obs_pred=pred_mag_matched,
                obs_positions_arcsec=OBS_POSITIONS,  # 偏移后坐标系
                pred_positions_arcsec=pred_pos_matched,  # 偏移后坐标系
                crit_segments=crit_segments,
                caus_segments=caus_segments if caus_segments else None,
                suptitle=f"glafic Optimized: {model_type.upper()} Model",
                output_file=output_file,
                nfw_params=subhalo_positions,
                show_2sigma=True
            )
        
        print(f"  ✓ 三联图已保存: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"  警告: 生成三联图失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description='glafic 优化工具：读取 best_params.txt，加扰动后运行 glafic optimize/MCMC'
    )
    parser.add_argument('input_folder', type=str, help='包含 *_best_params.txt 的文件夹路径')
    parser.add_argument('--model_type', type=str, choices=['pointmass', 'nfw', 'p_jaffe'],
                        help='模型类型（自动检测如果不指定）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认在输入文件夹下创建 glafic_optimize）')
    parser.add_argument('--perturb_min', type=float, default=0.99,
                        help='扰动范围下限（默认 0.99）')
    parser.add_argument('--perturb_max', type=float, default=1.1,
                        help='扰动范围上限（默认 1.1）')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子（用于复现）')
    parser.add_argument('--no_plot', action='store_true',
                        help='不生成三联图')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='详细输出模式（实时打印glafic输出）')
    parser.add_argument('--timeout', type=int, default=3000,
                        help='glafic 运行超时时间（秒，默认3000）')
    
    # 优化相关参数
    parser.add_argument('--max_restart', type=int, default=3,
                        help='amoeba优化最大重启次数（默认 3，设为0禁用重启，-1无限重启）')
    
    # MCMC 相关参数
    parser.add_argument('--mcmc', action='store_true',
                        help='使用MCMC采样代替amoeba优化')
    parser.add_argument('--mcmc_nsteps', type=int, default=MCMC_NSTEPS,
                        help=f'MCMC采样步数（默认 {MCMC_NSTEPS}）')

    # glafic Initial Guess 参数
    parser.add_argument('--glafic_guess', action='store_true',
                        default=GLAFIC_GUESS,
                        help=f'启用 glafic 内置 initial guess（randomize 后 optimize）'
                             f'（默认 {GLAFIC_GUESS}）')
    parser.add_argument('--guess_perturb', type=float,
                        default=GLAFIC_GUESS_PERTURB,
                        help=f'glafic_guess 模式的参数范围 ±比例（默认 {GLAFIC_GUESS_PERTURB}，即 ±{GLAFIC_GUESS_PERTURB*100:.0f}%%）')
    parser.add_argument('--guess_n_explore', type=int,
                        default=GLAFIC_GUESS_N_EXPLORE,
                        help=f'glafic_guess 时的 opt_explore 次数（0=randomize+optimize 一次，>0=opt_explore N 次）（默认 {GLAFIC_GUESS_N_EXPLORE}）')
    parser.add_argument('--guess_c2lim', type=float,
                        default=GLAFIC_GUESS_C2LIM,
                        help=f'opt_explore 的 chi2 上限（默认 {GLAFIC_GUESS_C2LIM:.0e}）')

    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"使用随机种子: {args.seed}")
    
    # 检查输入文件夹
    if not os.path.isdir(args.input_folder):
        print(f"错误: 输入文件夹不存在: {args.input_folder}")
        sys.exit(1)
    
    # 查找 best_params.txt 文件
    pattern = os.path.join(args.input_folder, '*_best_params.txt')
    param_files = glob.glob(pattern)
    
    if not param_files:
        print(f"错误: 在 {args.input_folder} 中找不到 *_best_params.txt 文件")
        sys.exit(1)
    
    print(f"找到 {len(param_files)} 个参数文件:")
    for f in param_files:
        print(f"  - {os.path.basename(f)}")
    
    # 使用第一个文件
    param_file = param_files[0]
    print(f"\n使用参数文件: {param_file}")
    
    # 创建输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.input_folder, 'glafic_optimize')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 模式信息
    mode_str = "MCMC" if args.mcmc else "Optimize (Amoeba)"
    print(f"运行模式: {mode_str}")
    if args.mcmc:
        print(f"MCMC 步数: {args.mcmc_nsteps}")
    else:
        if args.max_restart == -1:
            print("优化重启: 无限（直到收敛）")
        elif args.max_restart == 0:
            print("优化重启: 禁用（只运行一次）")
        else:
            print(f"优化重启: 最多 {args.max_restart} 次")
    if args.glafic_guess:
        if args.guess_n_explore > 0:
            print(f"Initial Guess: glafic opt_explore × {args.guess_n_explore}，参数范围 ±{args.guess_perturb*100:.0f}%")
        else:
            print(f"Initial Guess: glafic randomize+optimize，参数范围 ±{args.guess_perturb*100:.0f}%")
    else:
        print(f"Initial Guess: Python 端随机扰动（perturb_range 参数控制）")
    if args.verbose:
        print("详细输出: 启用")
    
    # 解析参数
    print("\n" + "=" * 70)
    print("步骤 1: 解析参数文件")
    print("=" * 70)
    
    original_params = parse_best_params(param_file)
    print(f"  模型类型: {original_params['model_type']}")
    print(f"  Sub-halos 数量: {len(original_params['subhalos'])}")
    print(f"  原始 chi²_best: {original_params.get('chi2_best', 'N/A')}")
    
    # 打印原始参数
    for sh in original_params['subhalos']:
        print(f"\n  Sub-halo {sh['idx']}:")
        print(f"    x = {sh.get('x', 'N/A'):.6f}")
        print(f"    y = {sh.get('y', 'N/A'):.6f}")
        if original_params['model_type'] == 'pointmass':
            print(f"    mass = {sh.get('mass', 'N/A'):.4e}")
        elif original_params['model_type'] == 'nfw':
            print(f"    m_vir = {sh.get('m_vir', 'N/A'):.4e}")
            print(f"    c_vir = {sh.get('c_vir', 'N/A'):.4f}")
        elif original_params['model_type'] == 'p_jaffe':
            print(f"    sig = {sh.get('sig', 'N/A'):.4f}")
            print(f"    a = {sh.get('a', 'N/A'):.6f}")
            print(f"    rco = {sh.get('rco', 'N/A'):.6f}")
    
    # 扰动参数
    print("\n" + "=" * 70)
    print("步骤 2: 应用参数扰动")
    print("=" * 70)

    perturb_range = (args.perturb_min, args.perturb_max)

    if args.glafic_guess:
        # glafic_guess 模式：不在 Python 端扰动，直接使用最优解，
        # 由 glafic randomize 在 prior.dat 的紧约束范围内随机选初始点
        print(f"  [glafic_guess 模式] 跳过 Python 端随机扰动，使用原始最优解作为起点")
        print(f"  glafic 将在 ±{args.guess_perturb*100:.0f}% 范围内随机选取初始点")
        perturbed_params = original_params  # 不扰动，原样传给 generate_glafic_input
    else:
        print(f"  扰动范围: {perturb_range}")
        perturbed_params = perturb_params(original_params, perturb_range)
    
    print("  扰动后的参数:")
    for sh in perturbed_params['subhalos']:
        print(f"\n  Sub-halo {sh['idx']}:")
        print(f"    x = {sh.get('x', 'N/A'):.6f}")
        print(f"    y = {sh.get('y', 'N/A'):.6f}")
        if perturbed_params['model_type'] == 'pointmass':
            print(f"    mass = {sh.get('mass', 'N/A'):.4e}")
        elif perturbed_params['model_type'] == 'nfw':
            print(f"    m_vir = {sh.get('m_vir', 'N/A'):.4e}")
            print(f"    c_vir = {sh.get('c_vir', 'N/A'):.4f}")
        elif perturbed_params['model_type'] == 'p_jaffe':
            print(f"    sig = {sh.get('sig', 'N/A'):.4f}")
            print(f"    a = {sh.get('a', 'N/A'):.6f}")
            print(f"    rco = {sh.get('rco', 'N/A'):.6f}")
    
    # 生成 glafic 输入文件
    print("\n" + "=" * 70)
    print("步骤 3: 生成 glafic 输入文件")
    print("=" * 70)
    
    output_prefix = 'glafic_opt'
    input_file, obs_file, prior_file, sigma_file = generate_glafic_input(
        perturbed_params, output_prefix, output_dir,
        use_mcmc=args.mcmc, mcmc_nsteps=args.mcmc_nsteps, verbose=args.verbose,
        max_restart=args.max_restart,
        glafic_guess=args.glafic_guess,
        glafic_guess_perturb=args.guess_perturb,
        glafic_guess_n_explore=args.guess_n_explore,
        glafic_guess_c2lim=args.guess_c2lim,
    )
    
    print(f"  输入文件: {input_file}")
    print(f"  观测数据: {obs_file}")
    print(f"  先验约束: {prior_file}")
    if sigma_file:
        print(f"  MCMC步长: {sigma_file}")
    
    # 运行 glafic
    print("\n" + "=" * 70)
    print(f"步骤 4: 运行 glafic {mode_str}")
    print("=" * 70)
    
    try:
        returncode, stdout, stderr = run_glafic(
            input_file, output_dir, 
            verbose=args.verbose, 
            timeout=args.timeout
        )
        print(f"\n  glafic 返回码: {returncode}")
    except Exception as e:
        print(f"  错误: 运行 glafic 失败: {e}")
        sys.exit(1)
    
    # 解析结果
    print("\n" + "=" * 70)
    print("步骤 5: 解析优化结果")
    print("=" * 70)
    
    if args.mcmc:
        optimized_result = parse_mcmc_result(output_dir, output_prefix)
        if optimized_result:
            print(f"  MCMC 采样数: {optimized_result.get('n_samples', 'N/A')}")
            print(f"  最优 chi²: {optimized_result.get('chi2', 'N/A')}")
            print(f"  chi² 均值: {optimized_result.get('chi2_mean', 0):.4f}")
            print(f"  chi² 标准差: {optimized_result.get('chi2_std', 0):.4f}")
        else:
            print("  警告: 无法解析MCMC结果")
            optimized_result = {}
        
        # 也尝试解析optresult（可能仍有用）
        opt_result = parse_optresult(output_dir, output_prefix)
        if opt_result and opt_result.get('lens_params'):
            optimized_result['lens_params'] = opt_result['lens_params']
            optimized_result['point_params'] = opt_result.get('point_params')
    else:
        optimized_result = parse_optresult(output_dir, output_prefix)
        
        if optimized_result:
            print(f"  优化后 chi²: {optimized_result.get('chi2', 'N/A')}")
            print(f"  透镜数量: {len(optimized_result.get('lens_params', []))}")
        else:
            print("  警告: 无法解析优化结果")
            optimized_result = {}
    
    # 比较结果
    compare_results(original_params, optimized_result, perturbed_params, use_mcmc=args.mcmc)
    
    # 生成三联图
    if not args.no_plot:
        print("\n" + "=" * 70)
        print("步骤 6: 生成三联图")
        print("=" * 70)
        
        triptych_file = generate_triptych(
            perturbed_params, optimized_result, output_dir, output_prefix
        )
    
    # 保存比较报告
    report_file = os.path.join(output_dir, 'optimization_report.txt')
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"glafic {'MCMC' if args.mcmc else 'Optimization'} Report\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("原始参数文件:\n")
        f.write(f"  {param_file}\n\n")
        
        f.write(f"模型类型: {original_params['model_type']}\n")
        f.write(f"Sub-halos 数量: {len(original_params['subhalos'])}\n")
        f.write(f"运行模式: {mode_str}\n")
        if args.mcmc:
            f.write(f"MCMC 步数: {args.mcmc_nsteps}\n")
        f.write("\n")
        
        f.write("性能对比:\n")
        f.write(f"  原始 chi²: {original_params.get('chi2_best', 'N/A')}\n")
        if args.mcmc and optimized_result:
            f.write(f"  MCMC 最优 chi²: {optimized_result.get('chi2', 'N/A')}\n")
            f.write(f"  chi² 均值 ± 标准差: {optimized_result.get('chi2_mean', 0):.4f} ± {optimized_result.get('chi2_std', 0):.4f}\n")
        else:
            f.write(f"  优化后 chi²: {optimized_result.get('chi2', 'N/A') if optimized_result else 'N/A'}\n")
        
        if optimized_result and optimized_result.get('chi2') and original_params.get('chi2_best'):
            improvement = (original_params['chi2_best'] - optimized_result['chi2']) / original_params['chi2_best'] * 100
            f.write(f"  改变: {improvement:.2f}%\n")
        
        f.write("\n扰动范围: ({:.2f}, {:.2f})\n".format(*perturb_range))
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("参数比较（每个 Sub-halo）\n")
        f.write("=" * 70 + "\n")
        
        opt_lenses = optimized_result.get('lens_params', []) if optimized_result else []
        subhalo_lenses = opt_lenses[3:] if len(opt_lenses) > 3 else []
        
        for i, (orig, pert) in enumerate(zip(original_params['subhalos'], perturbed_params['subhalos'])):
            f.write(f"\nSub-halo {orig['idx']}:\n")
            opt = subhalo_lenses[i] if i < len(subhalo_lenses) else None
            
            f.write(f"  位置 x: 原始={orig['x']:.6f}, 扰动={pert['x']:.6f}")
            if opt: f.write(f", 优化={opt['x']:.6f}")
            f.write("\n")
            
            f.write(f"  位置 y: 原始={orig['y']:.6f}, 扰动={pert['y']:.6f}")
            if opt: f.write(f", 优化={opt['y']:.6f}")
            f.write("\n")
            
            if original_params['model_type'] == 'pointmass':
                f.write(f"  质量: 原始={orig['mass']:.4e}, 扰动={pert['mass']:.4e}")
                if opt: f.write(f", 优化={opt['p1']:.4e}")
                f.write("\n")
            elif original_params['model_type'] == 'nfw':
                f.write(f"  M_vir: 原始={orig['m_vir']:.4e}, 扰动={pert['m_vir']:.4e}")
                if opt: f.write(f", 优化={opt['p1']:.4e}")
                f.write("\n")
                f.write(f"  c_vir: 原始={orig['c_vir']:.4f}, 扰动={pert['c_vir']:.4f}")
                if opt: f.write(f", 优化={opt['p2']:.4f}")
                f.write("\n")
            elif original_params['model_type'] == 'p_jaffe':
                f.write(f"  σ: 原始={orig['sig']:.4f}, 扰动={pert['sig']:.4f}")
                if opt: f.write(f", 优化={opt['p1']:.4f}")
                f.write("\n")
                f.write(f"  a: 原始={orig['a']:.6f}, 扰动={pert['a']:.6f}")
                if opt: f.write(f", 优化={opt['p2']:.6f}")
                f.write("\n")
                f.write(f"  rco: 原始={orig['rco']:.6f}, 扰动={pert['rco']:.6f}")
                if opt: f.write(f", 优化={opt['p3']:.6f}")
                f.write("\n")
    
    print(f"\n✓ 报告已保存: {report_file}")
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print(f"输出文件:")
    print(f"  - {input_file}")
    if args.mcmc:
        print(f"  - {os.path.join(output_dir, f'{output_prefix}_mcmc.dat')}")
    print(f"  - {os.path.join(output_dir, f'{output_prefix}_optresult.dat')}")
    print(f"  - {report_file}")
    if not args.no_plot and 'triptych_file' in dir():
        print(f"  - {triptych_file}")


if __name__ == '__main__':
    main()
