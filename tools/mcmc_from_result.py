#!/usr/bin/env python3
"""
mcmc_from_result.py
===================
读取某次 DE 运行结果文件夹中的 *_best_params.txt，
自动识别子晕模型类型（pointmass / nfw / king / p-jaffe），
以 DE 最优解为起点启动 emcee MCMC 后验采样，
并将全部输出写入同一文件夹。

输出文件：
  <prefix>_mcmc_chain.dat       — 原始后验链
  <prefix>_corner.png           — Corner plot
  <prefix>_trace.png            — Walker 轨迹图
  <prefix>_mass_posterior_1d.png— logM 一维后验 KDE 图
  <prefix>_mcmc_posterior.txt   — 参数统计文件

用法：
  python mcmc_from_result.py <result_folder>
  python mcmc_from_result.py <result_folder> --nsteps 5000 --nwalkers 64
  python mcmc_from_result.py <result_folder> --baseline_dir work/SN_2Sersic_NFW

示例：
  python mcmc_from_result.py results/sie/p-jaffe/260120_0319-3imgs
"""

import sys
import os
import re
import glob
import argparse
import shutil
import numpy as np
from datetime import datetime

# ── glade 运行时环境配置 ───────────────────────────────────────────────────
GLADE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GLADE_ROOT)
from runtime_env import setup_runtime_env  # noqa: E402
setup_runtime_env(os.path.abspath(GLADE_ROOT))

_GLAFIC_PYTHON_PATH = os.path.join(GLADE_ROOT, 'glafic2', 'python')
sys.path.insert(0, _GLAFIC_PYTHON_PATH)

import glafic

# ── 宇宙学（用于 Pseudo-Jaffe 质量估算） ───────────────────────────────────
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
_COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      固定观测数据 / 宇宙学                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

OBS_POS_MAS = np.array([
    [-266.035,  +0.427],
    [+118.835, -221.927],
    [+238.324, +227.270],
    [-126.157, +319.719],
])
OBS_POS = np.column_stack([
    -OBS_POS_MAS[:, 0] / 1000.0,
     OBS_POS_MAS[:, 1] / 1000.0,
])
OBS_MAG       = np.array([-35.6, 15.7, -7.5, 9.1])
OBS_MAG_ERR   = np.array([2.1,   1.3,  1.0,  1.1])
OBS_POS_SIGMA = np.array([0.41,  0.86, 2.23, 3.11])   # [mas]

CENTER_OFFSET_X = -0.01535000   # [arcsec]
CENTER_OFFSET_Y = +0.03220000   # [arcsec]

OMEGA        = 0.3
LAMBDA_COSMO = 0.7
WEOS         = -1.0
HUBBLE       = 0.7
SOURCE_Z     = 0.4090
SOURCE_X     = 2.685497e-03
SOURCE_Y     = 2.443616e-02
LENS_Z       = 0.2160

XMIN, YMIN = -0.5, -0.5
XMAX, YMAX =  0.5,  0.5
PIX_EXT = 0.01
PIX_POI = 0.2
MAXLEV  = 5

LOSS_COEF_A   = 4.0
LOSS_COEF_B   = 1.0
LOSS_PENALTY  = 1000.0   # 统一使用 1000（与 p-jaffe 主脚本一致）

DEFAULT_LENS_PARAMS = {
    'sers1': (1, 'sers', 0.2160, 9.896617e+09, 2.656977e-03, 2.758473e-02,
              2.986760e-01, 1.124730e+02, 3.939718e-01, 1.057760e+00),
    'sers2': (2, 'sers', 0.2160, 2.555580e+10, 2.656977e-03, 2.758473e-02,
              4.242340e-01, 5.396370e+01, 1.538855e+00, 1.000000e+00),
    'sie':   (3, 'sie',  0.2160, 1.183382e+02, 2.656977e-03, 2.758473e-02,
              1.571203e-01, 2.920348e+01, 0.0, 0.0),
}
DEFAULT_MAIN_LENS_KEY = 'sie'

# ╔══════════════════════════════════════════════════════════════════════╗
# ║          默认 MCMC 配置（与 v_p_jaffe_2.0 保持一致）                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
# 命令行未指定时使用以下默认值；如需覆盖，在运行时加上对应的 --xxx 参数。

DEFAULT_MCMC_NWALKERS     = 32       # walker数量 [count]，至少是参数维度的2倍
DEFAULT_MCMC_NSTEPS       = 3000     # 采样步数 [count]
DEFAULT_MCMC_BURNIN       = 300      # burn-in 步数 [count]，丢弃前N步
DEFAULT_MCMC_THIN         = 2        # 稀疏采样 [count]，每N步保留1个
DEFAULT_MCMC_PERTURBATION = 0.01     # 初始扰动幅度 [fraction]，相对于参数绝对值
DEFAULT_MCMC_PROGRESS     = True     # 是否显示进度条
DEFAULT_MCMC_WORKERS      = -1       # 并行核心数，1=串行，-1=全部CPU，>1=指定核心数

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                   MCMC 先验范围配置（在此直接修改）                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
# 位置约束：以各图像观测位置为中心，walker 不超出此半径
MCMC_SEARCH_RADIUS = 0.3       # [arcsec]

# ── Pointmass ──────────────────────────────────────────────────────────
MCMC_PM_LOG_M_MIN  = 1.0       # log10(M) 下限 [dex]
MCMC_PM_LOG_M_MAX  = 10.0      # log10(M) 上限 [dex]

# ── NFW ────────────────────────────────────────────────────────────────
MCMC_NFW_LOG_M_MIN = 1.0       # log10(M_vir) 下限 [dex]
MCMC_NFW_LOG_M_MAX = 14.0      # log10(M_vir) 上限 [dex]
MCMC_NFW_C_MIN     = 1.0       # 浓度参数下限
MCMC_NFW_C_MAX     = 200.0     # 浓度参数上限

# ── Pseudo-Jaffe ───────────────────────────────────────────────────────
MCMC_PJ_SIG_MIN  = 0.01       # 速度弥散下限 [km/s]
MCMC_PJ_SIG_MAX  = 100.0       # 速度弥散上限 [km/s]
MCMC_PJ_A_MIN    = 0.0001      # 截断半径下限 [arcsec]
MCMC_PJ_A_MAX    = 0.300       # 截断半径上限 [arcsec]
MCMC_PJ_RCO_MIN  = 0    # 核心半径下限 [arcsec]
MCMC_PJ_RCO_MAX  = 0.0200       # 核心半径上限 [arcsec]

# ── King ───────────────────────────────────────────────────────────────
MCMC_KING_LOG_M_MIN = 1.0        # log10(M) 下限 [dex]
MCMC_KING_LOG_M_MAX = 10.0       # log10(M) 上限 [dex]
MCMC_KING_RC_MIN    = 0.00001    # rc 下限 [arcsec]
MCMC_KING_RC_MAX    = 0.300      # rc 上限 [arcsec]
MCMC_KING_C_MIN     = 0.1        # c=log10(rt/rc) 下限
MCMC_KING_C_MAX     = 5.0        # c=log10(rt/rc) 上限

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      工具函数                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

def find_glafic_bin(default='/home/luukiaun/glafic251018/glafic2/glafic'):
    if os.path.isfile(default) and os.access(default, os.X_OK):
        return default
    try:
        module_dir = os.path.dirname(os.path.abspath(glafic.__file__))
        for rel in ['../glafic', '../../glafic', './glafic', '../bin/glafic']:
            p = os.path.abspath(os.path.join(module_dir, rel))
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    except Exception:
        pass
    return shutil.which('glafic')


def load_baseline_lens_params(directory):
    """从 bestfit.dat 加载基准透镜参数，格式与主脚本一致。"""
    bestfit_path = os.path.join(directory, 'bestfit.dat')
    if not os.path.isfile(bestfit_path):
        return None, None, None, None

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

    if len(lens_lines) < 3 or point_params is None:
        return None, None, None, None

    params_dict, sers_count, main_key = {}, 0, None
    for parts in lens_lines:
        ltype = parts[1]
        z     = float(parts[2])
        vals  = [float(v) for v in parts[3:]]
        if ltype == 'sers':
            sers_count += 1
            params_dict[f'sers{sers_count}'] = (sers_count, 'sers', z, *vals)
        else:
            main_key = ltype
            params_dict[ltype] = (sers_count + 1, ltype, z, *vals)

    if main_key is None:
        return None, None, None, None

    return params_dict, float(point_params[2]), float(point_params[3]), main_key


def find_bestfit_dir(start_dir):
    """向上遍历目录树，寻找包含 bestfit.dat 的文件夹（最多5层）。"""
    d = os.path.abspath(start_dir)
    for _ in range(6):
        if os.path.isfile(os.path.join(d, 'bestfit.dat')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  best_params.txt 解析                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

def detect_model_type(content):
    if 'x_jaffe' in content or 'Pseudo-Jaffe' in content:
        return 'p_jaffe'
    if 'x_king' in content or 'M_king' in content or 'King' in content:
        return 'king'
    if 'x_nfw' in content or 'm_vir' in content or 'NFW' in content:
        return 'nfw'
    if 'x_sub' in content or 'mass_sub' in content or 'Point Mass' in content:
        return 'pointmass'
    return None


def parse_best_params(filepath):
    """
    解析 *_best_params.txt，返回字典：
      model_type, active_subhalos, subhalos, chi2_base, chi2_best, de_seed
    subhalos 是按图像索引排列的列表，每项视模型类型包含不同字段。
    """
    with open(filepath) as f:
        content = f.read()

    model_type = detect_model_type(content)
    if model_type is None:
        raise ValueError(f"无法识别模型类型：{filepath}")

    # active_subhalos
    m = re.search(r'active_subhalos\s*=\s*(\[[\d,\s]+\])', content)
    active_subhalos = list(map(int, re.findall(r'\d+', m.group(1)))) if m else []

    # DE_SEED
    m = re.search(r'DE_SEED\s*=\s*(\d+)', content)
    de_seed = int(m.group(1)) if m else None

    # chi2
    def _float(pattern):
        m = re.search(pattern, content)
        return float(m.group(1)) if m else None

    chi2_base = _float(r'chi2_base\s*=\s*([\d.]+)')
    chi2_best = _float(r'chi2_best\s*=\s*([\d.]+)')

    subhalos = []

    if model_type == 'pointmass':
        xs  = {int(m.group(1)): float(m.group(2))
               for m in re.finditer(r'x_sub(\d+)\s*=\s*([\d.eE+-]+)', content)}
        ys  = {int(m.group(1)): float(m.group(2))
               for m in re.finditer(r'y_sub(\d+)\s*=\s*([\d.eE+-]+)', content)}
        ms  = {int(m.group(1)): float(m.group(2))
               for m in re.finditer(r'mass_sub(\d+)\s*=\s*([\d.eE+-]+)', content)}
        for idx in sorted(xs):
            if idx in xs and idx in ys and idx in ms:
                subhalos.append({'idx': idx,
                                 'x': xs[idx], 'y': ys[idx], 'mass': ms[idx]})

    elif model_type == 'nfw':
        xs  = {int(m.group(1)): float(m.group(2))
               for m in re.finditer(r'x_nfw(\d+)\s*=\s*([\d.eE+-]+)', content)}
        ys  = {int(m.group(1)): float(m.group(2))
               for m in re.finditer(r'y_nfw(\d+)\s*=\s*([\d.eE+-]+)', content)}
        ms  = {int(m.group(1)): float(m.group(2))
               for m in re.finditer(r'm_vir(\d+)\s*=\s*([\d.eE+-]+)', content)}
        cs  = {int(m.group(1)): float(m.group(2))
               for m in re.finditer(r'c_vir(\d+)\s*=\s*([\d.eE+-]+)', content)}
        for idx in sorted(xs):
            if idx in xs and idx in ys and idx in ms and idx in cs:
                subhalos.append({'idx': idx,
                                 'x': xs[idx], 'y': ys[idx],
                                 'm_vir': ms[idx], 'c_vir': cs[idx]})

    elif model_type == 'p_jaffe':
        xs   = {int(m.group(1)): float(m.group(2))
                for m in re.finditer(r'x_jaffe(\d+)\s*=\s*([\d.eE+-]+)', content)}
        ys   = {int(m.group(1)): float(m.group(2))
                for m in re.finditer(r'y_jaffe(\d+)\s*=\s*([\d.eE+-]+)', content)}
        sigs = {int(m.group(1)): float(m.group(2))
                for m in re.finditer(r'sig(\d+)\s*=\s*([\d.eE+-]+)', content)}
        as_  = {int(m.group(1)): float(m.group(2))
                for m in re.finditer(r'^a(\d+)\s*=\s*([\d.eE+-]+)', content, re.M)}
        # rco 行，排除 rco/a 行
        rcos = {}
        for m in re.finditer(r'rco(\d+)\s*=\s*([\d.eE+-]+)', content):
            idx = int(m.group(1))
            start = m.start()
            if start == 0 or content[start-1] != '/':
                rcos[idx] = float(m.group(2))
        for idx in sorted(xs):
            if idx in xs and idx in ys and idx in sigs:
                subhalos.append({'idx': idx,
                                 'x': xs[idx], 'y': ys[idx],
                                 'sig': sigs[idx],
                                 'a':   as_.get(idx, 0.05),
                                 'rco': rcos.get(idx, 0.005)})

    elif model_type == 'king':
        xs  = {int(m.group(1)): float(m.group(2))
               for m in re.finditer(r'x_king(\d+)\s*=\s*([\d.eE+-]+)', content)}
        ys  = {int(m.group(1)): float(m.group(2))
               for m in re.finditer(r'y_king(\d+)\s*=\s*([\d.eE+-]+)', content)}
        ms  = {int(m.group(1)): float(m.group(2))
               for m in re.finditer(r'M_king(\d+)\s*=\s*([\d.eE+-]+)', content)}
        rcs = {int(m.group(1)): float(m.group(2))
               for m in re.finditer(r'rc_king(\d+)\s*=\s*([\d.eE+-]+)', content)}
        cs  = {int(m.group(1)): float(m.group(2))
               for m in re.finditer(r'c_king(\d+)\s*=\s*([\d.eE+-]+)', content)}
        for idx in sorted(xs):
            if idx in xs and idx in ys and idx in ms and idx in rcs and idx in cs:
                subhalos.append({
                    'idx': idx,
                    'x': xs[idx],
                    'y': ys[idx],
                    'mass': ms[idx],
                    'rc': rcs[idx],
                    'c': cs[idx],
                })

    if not active_subhalos:
        active_subhalos = [s['idx'] for s in subhalos]

    return {
        'model_type':     model_type,
        'active_subhalos': active_subhalos,
        'subhalos':       subhalos,
        'chi2_base':      chi2_base,
        'chi2_best':      chi2_best,
        'de_seed':        de_seed,
    }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                   glafic 模型计算                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _prefix():
    return f'_mcmc_tmp_{os.getpid()}'


def machine_learning_loss(pred_pos, pred_mag, delta_pos):
    Y = 0.0
    for i in range(4):
        chi2_pos = (delta_pos[i] / OBS_POS_SIGMA[i])**2
        chi2_mag = ((pred_mag[i] - OBS_MAG[i]) / OBS_MAG_ERR[i])**2
        P = 0.0 if delta_pos[i] <= OBS_POS_SIGMA[i] else LOSS_PENALTY * delta_pos[i]
        Y += LOSS_COEF_A * chi2_pos + LOSS_COEF_B * chi2_mag + P
    return Y


def compute_model_pointmass(subhalo_list, lens_params, main_lens_key,
                             src_x=SOURCE_X, src_y=SOURCE_Y):
    """subhalo_list: [(x, y, mass), ...]"""
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    glafic.init(OMEGA, LAMBDA_COSMO, WEOS, HUBBLE, _prefix(),
                XMIN, YMIN, XMAX, YMAX, PIX_EXT, PIX_POI, MAXLEV, verb=0)
    n = len(subhalo_list)
    glafic.startup_setnum(3 + n, 0, 1)
    glafic.set_lens(*lens_params['sers1'])
    glafic.set_lens(*lens_params['sers2'])
    glafic.set_lens(*lens_params[main_lens_key])
    for i, (x, y, mass) in enumerate(subhalo_list):
        glafic.set_lens(4 + i, 'point', LENS_Z, mass, x, y, 0.0, 0.0, 0.0, 0.0)
    glafic.set_point(1, SOURCE_Z, src_x, src_y)
    glafic.model_init(verb=0)
    result = glafic.point_solve(SOURCE_Z, src_x, src_y, verb=0)
    glafic.quit()
    return _parse_result(result)


def compute_model_nfw(subhalo_list, lens_params, main_lens_key,
                      src_x=SOURCE_X, src_y=SOURCE_Y):
    """subhalo_list: [(x, y, mass, c_vir), ...]"""
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    glafic.init(OMEGA, LAMBDA_COSMO, WEOS, HUBBLE, _prefix(),
                XMIN, YMIN, XMAX, YMAX, PIX_EXT, PIX_POI, MAXLEV, verb=0)
    n = len(subhalo_list)
    glafic.startup_setnum(3 + n, 0, 1)
    glafic.set_lens(*lens_params['sers1'])
    glafic.set_lens(*lens_params['sers2'])
    glafic.set_lens(*lens_params[main_lens_key])
    for i, (x, y, mass, c) in enumerate(subhalo_list):
        glafic.set_lens(4 + i, 'gnfw', LENS_Z, mass, x, y, 0.0, 0.0, c, 1.0)
    glafic.set_point(1, SOURCE_Z, src_x, src_y)
    glafic.model_init(verb=0)
    result = glafic.point_solve(SOURCE_Z, src_x, src_y, verb=0)
    glafic.quit()
    return _parse_result(result)


def compute_model_p_jaffe(subhalo_list, lens_params, main_lens_key,
                           src_x=SOURCE_X, src_y=SOURCE_Y):
    """subhalo_list: [(x, y, sig, a, rco), ...]"""
    glafic.init(OMEGA, LAMBDA_COSMO, WEOS, HUBBLE, _prefix(),
                XMIN, YMIN, XMAX, YMAX, PIX_EXT, PIX_POI, MAXLEV, verb=0)
    n = len(subhalo_list)
    glafic.startup_setnum(3 + n, 0, 1)
    glafic.set_lens(*lens_params['sers1'])
    glafic.set_lens(*lens_params['sers2'])
    glafic.set_lens(*lens_params[main_lens_key])
    for i, (x, y, sig, a, rco) in enumerate(subhalo_list):
        glafic.set_lens(4 + i, 'jaffe', LENS_Z, sig, x, y, 0.0, 0.0, a, rco)
    glafic.set_point(1, SOURCE_Z, src_x, src_y)
    glafic.model_init(verb=0)
    result = glafic.point_solve(SOURCE_Z, src_x, src_y, verb=0)
    glafic.quit()
    return _parse_result(result)


def compute_model_king(subhalo_list, lens_params, main_lens_key,
                       src_x=SOURCE_X, src_y=SOURCE_Y):
    """subhalo_list: [(x, y, mass, rc, c), ...]"""
    glafic.init(OMEGA, LAMBDA_COSMO, WEOS, HUBBLE, _prefix(),
                XMIN, YMIN, XMAX, YMAX, PIX_EXT, PIX_POI, MAXLEV, verb=0)
    n = len(subhalo_list)
    glafic.startup_setnum(3 + n, 0, 1)
    glafic.set_lens(*lens_params['sers1'])
    glafic.set_lens(*lens_params['sers2'])
    glafic.set_lens(*lens_params[main_lens_key])
    for i, (x, y, mass, rc, c) in enumerate(subhalo_list):
        glafic.set_lens(4 + i, 'king', LENS_Z, mass, x, y, 0.0, 0.0, rc, c)
    glafic.set_point(1, SOURCE_Z, src_x, src_y)
    glafic.model_init(verb=0)
    result = glafic.point_solve(SOURCE_Z, src_x, src_y, verb=0)
    glafic.quit()
    return _parse_result(result)


def _parse_result(result):
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    n = len(result)
    if n == 5:
        abs_mags = [abs(r[2]) for r in result]
        drop = int(np.argmin(abs_mags))
        result = [r for i, r in enumerate(result) if i != drop]
    elif n != 4:
        return None, None, None, 1e10

    pred_pos = np.array([[r[0], r[1]] for r in result])
    pred_mag = np.array([r[2] for r in result])
    pred_pos[:, 0] += CENTER_OFFSET_X
    pred_pos[:, 1] += CENTER_OFFSET_Y

    dists = cdist(OBS_POS, pred_pos)
    row_ind, col_ind = linear_sum_assignment(dists)
    sort = col_ind[np.argsort(row_ind)]
    pred_pos_m = pred_pos[sort]
    pred_mag_m = pred_mag[sort]

    delta_pos = np.array([
        np.sqrt(((pred_pos_m[i, 0] - OBS_POS[i, 0]) * 1000)**2 +
                ((pred_pos_m[i, 1] - OBS_POS[i, 1]) * 1000)**2)
        for i in range(4)
    ])
    mag_chi2 = np.sum(((pred_mag_m - OBS_MAG) / OBS_MAG_ERR)**2)
    return pred_pos_m, pred_mag_m, delta_pos, mag_chi2


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    质量估算（Pseudo-Jaffe）                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

def calculate_jaffe_mass(sigma_km_s, a_arcsec, rco_arcsec):
    G  = 4.302e-6   # kpc (km/s)^2 / M_sun
    c  = 299792.458 # km/s
    D_l  = _COSMO.angular_diameter_distance(LENS_Z).to(u.kpc).value
    D_s  = _COSMO.angular_diameter_distance(SOURCE_Z).to(u.kpc).value
    D_ls = _COSMO.angular_diameter_distance_z1z2(LENS_Z, SOURCE_Z).to(u.kpc).value
    rad  = np.pi / (180.0 * 3600.0)
    a_kpc   = a_arcsec   * rad * D_l
    rco_kpc = rco_arcsec * rad * D_l
    Sigma_cr = (c**2 / (4 * np.pi * G)) * (D_s / (D_l * D_ls))
    b_kpc    = 4 * np.pi * (sigma_km_s / c)**2 * (D_ls / D_s) * D_l
    return np.pi * Sigma_cr * b_kpc * (a_kpc - rco_kpc)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         MCMC 核心                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

class LogProbability:
    """
    可 pickle 的对数概率函数对象（模块级类，multiprocessing.Pool 兼容）。

    Python 的 multiprocessing 要求传递给 Pool 的函数必须可 pickle，
    而局部闭包（local function）无法被 pickle。
    使用模块级类并实现 __call__ 可绕过此限制。
    """

    def __init__(self, model_type, n_subhalos, active_subhalos,
                 lens_params, main_lens_key, prior_ranges):
        self.model_type     = model_type
        self.n              = n_subhalos
        self.active_subhalos = active_subhalos
        self.lens_params    = lens_params
        self.main_lens_key  = main_lens_key
        self.prior          = prior_ranges   # dict，由 build_log_probability 传入

    def __call__(self, params):
        n    = self.n
        lp   = self.lens_params
        mk   = self.main_lens_key
        pr   = self.prior

        if self.model_type == 'pointmass':
            for i in range(n):
                img_idx = self.active_subhalos[i]
                x_ctr   = OBS_POS[img_idx - 1, 0]
                y_ctr   = OBS_POS[img_idx - 1, 1]
                x     = params[i*3]
                y     = params[i*3 + 1]
                log_m = params[i*3 + 2]
                if abs(x - x_ctr) > pr['search_radius'] or \
                   abs(y - y_ctr) > pr['search_radius']:
                    return -np.inf
                if not (pr['pm_log_m_min'] <= log_m <= pr['pm_log_m_max']):
                    return -np.inf
            subhalo_list = [(params[i*3], params[i*3+1], 10**params[i*3+2])
                            for i in range(n)]
            pos, mag, delta, _ = compute_model_pointmass(subhalo_list, lp, mk)

        elif self.model_type == 'nfw':
            for i in range(n):
                img_idx = self.active_subhalos[i]
                x_ctr   = OBS_POS[img_idx - 1, 0]
                y_ctr   = OBS_POS[img_idx - 1, 1]
                x     = params[i*4]
                y     = params[i*4 + 1]
                log_m = params[i*4 + 2]
                c     = params[i*4 + 3]
                if abs(x - x_ctr) > pr['search_radius'] or \
                   abs(y - y_ctr) > pr['search_radius']:
                    return -np.inf
                if not (pr['nfw_log_m_min'] <= log_m <= pr['nfw_log_m_max']):
                    return -np.inf
                if not (pr['nfw_c_min'] <= c <= pr['nfw_c_max']):
                    return -np.inf
            subhalo_list = [(params[i*4], params[i*4+1],
                             10**params[i*4+2], params[i*4+3])
                            for i in range(n)]
            pos, mag, delta, _ = compute_model_nfw(subhalo_list, lp, mk)

        elif self.model_type == 'p_jaffe':
            for i in range(n):
                img_idx = self.active_subhalos[i]
                x_ctr   = OBS_POS[img_idx - 1, 0]
                y_ctr   = OBS_POS[img_idx - 1, 1]
                x   = params[i*5]
                y   = params[i*5 + 1]
                sig = params[i*5 + 2]
                a   = params[i*5 + 3]
                rco = params[i*5 + 4]
                if abs(x - x_ctr) > pr['search_radius'] or \
                   abs(y - y_ctr) > pr['search_radius']:
                    return -np.inf
                if not (pr['pj_sig_min'] <= sig <= pr['pj_sig_max']):
                    return -np.inf
                if not (pr['pj_a_min'] <= a <= pr['pj_a_max']):
                    return -np.inf
                if not (pr['pj_rco_min'] <= rco <= pr['pj_rco_max']):
                    return -np.inf
                if a <= rco:
                    return -np.inf
            subhalo_list = [(params[i*5], params[i*5+1], params[i*5+2],
                             params[i*5+3], params[i*5+4])
                            for i in range(n)]
            pos, mag, delta, _ = compute_model_p_jaffe(subhalo_list, lp, mk)

        elif self.model_type == 'king':
            for i in range(n):
                img_idx = self.active_subhalos[i]
                x_ctr   = OBS_POS[img_idx - 1, 0]
                y_ctr   = OBS_POS[img_idx - 1, 1]
                x     = params[i*5]
                y     = params[i*5 + 1]
                log_m = params[i*5 + 2]
                rc    = params[i*5 + 3]
                c     = params[i*5 + 4]
                if abs(x - x_ctr) > pr['search_radius'] or \
                   abs(y - y_ctr) > pr['search_radius']:
                    return -np.inf
                if not (pr['king_log_m_min'] <= log_m <= pr['king_log_m_max']):
                    return -np.inf
                if not (pr['king_rc_min'] <= rc <= pr['king_rc_max']):
                    return -np.inf
                if not (pr['king_c_min'] <= c <= pr['king_c_max']):
                    return -np.inf
            subhalo_list = [(params[i*5], params[i*5+1], 10**params[i*5+2],
                             params[i*5+3], params[i*5+4])
                            for i in range(n)]
            pos, mag, delta, _ = compute_model_king(subhalo_list, lp, mk)

        else:
            return -np.inf

        if pos is None:
            return -np.inf
        loss = machine_learning_loss(pos, mag, delta)
        return -np.inf if loss >= 1e10 else -0.5 * loss


def build_log_probability(model_type, active_subhalos, lens_params, main_lens_key):
    """返回 LogProbability 实例（可 pickle，multiprocessing 兼容）。"""
    prior_ranges = {
        'search_radius': MCMC_SEARCH_RADIUS,
        # pointmass
        'pm_log_m_min':  MCMC_PM_LOG_M_MIN,
        'pm_log_m_max':  MCMC_PM_LOG_M_MAX,
        # nfw
        'nfw_log_m_min': MCMC_NFW_LOG_M_MIN,
        'nfw_log_m_max': MCMC_NFW_LOG_M_MAX,
        'nfw_c_min':     MCMC_NFW_C_MIN,
        'nfw_c_max':     MCMC_NFW_C_MAX,
        # p_jaffe
        'pj_sig_min':    MCMC_PJ_SIG_MIN,
        'pj_sig_max':    MCMC_PJ_SIG_MAX,
        'pj_a_min':      MCMC_PJ_A_MIN,
        'pj_a_max':      MCMC_PJ_A_MAX,
        'pj_rco_min':    MCMC_PJ_RCO_MIN,
        'pj_rco_max':    MCMC_PJ_RCO_MAX,
        # king
        'king_log_m_min': MCMC_KING_LOG_M_MIN,
        'king_log_m_max': MCMC_KING_LOG_M_MAX,
        'king_rc_min':    MCMC_KING_RC_MIN,
        'king_rc_max':    MCMC_KING_RC_MAX,
        'king_c_min':     MCMC_KING_C_MIN,
        'king_c_max':     MCMC_KING_C_MAX,
    }
    return LogProbability(model_type, len(active_subhalos), active_subhalos,
                          lens_params, main_lens_key, prior_ranges)


def build_initial_params(model_type, subhalos):
    """从解析出的 subhalo 列表构造 MCMC 初始参数向量。"""
    vec = []
    if model_type == 'pointmass':
        for s in subhalos:
            vec += [s['x'], s['y'], np.log10(s['mass'])]
    elif model_type == 'nfw':
        for s in subhalos:
            vec += [s['x'], s['y'], np.log10(s['m_vir']), s['c_vir']]
    elif model_type == 'p_jaffe':
        for s in subhalos:
            vec += [s['x'], s['y'], s['sig'], s['a'], s['rco']]
    elif model_type == 'king':
        for s in subhalos:
            vec += [s['x'], s['y'], np.log10(s['mass']), s['rc'], s['c']]
    return np.array(vec)


def build_param_names(model_type, active_subhalos):
    names, labels = [], []
    if model_type == 'pointmass':
        for idx in active_subhalos:
            names  += [f'x_{idx}',    f'y_{idx}',    f'logM_{idx}']
            labels += [f'$x_{idx}$',  f'$y_{idx}$',  f'$\\log M_{idx}$']
    elif model_type == 'nfw':
        for idx in active_subhalos:
            names  += [f'x_{idx}',    f'y_{idx}',    f'logM_{idx}',         f'c_{idx}']
            labels += [f'$x_{idx}$',  f'$y_{idx}$',  f'$\\log M_{idx}$',    f'$c_{idx}$']
    elif model_type == 'p_jaffe':
        for idx in active_subhalos:
            names  += [f'x_{idx}',   f'y_{idx}',   f'sig_{idx}',
                       f'a_{idx}',   f'rco_{idx}']
            labels += [f'$x_{idx}$', f'$y_{idx}$', f'$\\sigma_{idx}$',
                       f'$a_{idx}$', f'$r_{{co,{idx}}}$']
    elif model_type == 'king':
        for idx in active_subhalos:
            names  += [f'x_{idx}',   f'y_{idx}',   f'logM_{idx}',
                       f'rc_{idx}',  f'c_{idx}']
            labels += [f'$x_{idx}$', f'$y_{idx}$', f'$\\log M_{idx}$',
                       f'$r_{{c,{idx}}}$', f'$c_{idx}$']
    return names, labels


def params_per_subhalo(model_type):
    return {'pointmass': 3, 'nfw': 4, 'p_jaffe': 5, 'king': 5}[model_type]


def run_mcmc(log_prob, p0, args):
    """运行 emcee，返回 (sampler, samples, chain)。"""
    import emcee

    ndim     = len(p0)
    nwalkers = max(args.nwalkers, 2 * ndim + 2)
    if nwalkers != args.nwalkers:
        print(f"  [调整] nwalkers → {nwalkers} (至少为参数维度的2倍)")

    rng = np.random.default_rng()
    initial = np.array([
        p0 + rng.normal(0, args.perturbation * (np.abs(p0) + 1e-8), ndim)
        for _ in range(nwalkers)
    ])

    workers = os.cpu_count() if args.workers == -1 else args.workers
    print(f"  参数维度: {ndim}, Walkers: {nwalkers}, 并行: {workers}")

    def _sample(sampler):
        if args.progress:
            from tqdm import tqdm
            for _ in tqdm(sampler.sample(initial, iterations=args.nsteps),
                          total=args.nsteps, desc='MCMC采样'):
                pass
        else:
            sampler.run_mcmc(initial, args.nsteps, progress=False)
        return (sampler.get_chain(discard=args.burnin, thin=args.thin, flat=True),
                sampler.get_chain())

    if workers > 1:
        from multiprocessing import Pool
        print(f"  ⚠ 多进程模式：如遇崩溃请设置 --workers 1")
        with Pool(workers) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
            samples, chain = _sample(sampler)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
        samples, chain = _sample(sampler)

    return sampler, samples, chain, nwalkers


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        绘图函数                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _corner_to_pct(fig, ndim, n_samples):
    """
    在 corner plot 每个对角线面板右侧用 twinx() 新建百分比纵轴。
    corner 会隐藏原始左侧 y 轴，但不会动 twinx 新建的右侧轴。
    """
    from matplotlib.ticker import MaxNLocator
    grid = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        ax = grid[i, i]
        ylo, yhi = ax.get_ylim()
        ax2 = ax.twinx()
        ax2.set_ylim(ylo / n_samples * 100, yhi / n_samples * 100)
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='lower'))
        ax2.tick_params(axis='y', labelsize=7, length=3, width=0.8)
        ax2.set_ylabel('%', fontsize=8, rotation=0, labelpad=10, va='center')


def plot_corner(samples, labels, p0, output_path):
    import corner
    import matplotlib.pyplot as plt

    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3f',
        truths=p0,
        truth_color='red',
        hist_kwargs={'alpha': 0.75},
    )
    _corner_to_pct(fig, samples.shape[1], len(samples))
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Corner plot: {os.path.basename(output_path)}")


def plot_trace(chain, labels, burnin, output_path):
    import matplotlib.pyplot as plt

    ndim = chain.shape[2]
    fig, axes = plt.subplots(ndim, figsize=(10, 2 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(chain[:, :, i], alpha=0.3, lw=0.5)
        ax.axvline(burnin, color='red', ls='--', label='burn-in')
        ax.set_ylabel(labels[i], fontsize=8)
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel('Step')
    axes[0].legend(loc='upper right', fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 轨迹图: {os.path.basename(output_path)}")


def plot_mass_1d(model_type, active_subhalos, samples, subhalos, output_path):
    """绘制 logM 一维 KDE 后验分布图，并标注 DE 最优解。"""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    n_halos = len(active_subhalos)
    pph = params_per_subhalo(model_type)
    fig, axes = plt.subplots(1, n_halos, figsize=(5 * n_halos, 4))
    if n_halos == 1:
        axes = [axes]

    for i, idx in enumerate(active_subhalos):
        s = subhalos[i]
        ax = axes[i]

        # 质量样本
        if model_type == 'pointmass':
            log_m_samp = samples[:, i * pph + 2]
            mass_samp  = 10**log_m_samp
            de_mass    = s['mass']
        elif model_type == 'nfw':
            log_m_samp = samples[:, i * pph + 2]
            mass_samp  = 10**log_m_samp
            de_mass    = s['m_vir']
        elif model_type == 'king':
            log_m_samp = samples[:, i * pph + 2]
            mass_samp  = 10**log_m_samp
            de_mass    = s['mass']
        else:  # p_jaffe
            sig_samp = samples[:, i * pph + 2]
            a_samp   = samples[:, i * pph + 3]
            rco_samp = samples[:, i * pph + 4]
            mass_samp = np.array([
                calculate_jaffe_mass(sg, a, r)
                for sg, a, r in zip(sig_samp, a_samp, rco_samp)
            ])
            de_mass = calculate_jaffe_mass(s['sig'], s['a'], s['rco'])

        valid = mass_samp > 0
        lms   = np.log10(mass_samp[valid])
        log_de = np.log10(de_mass)

        kde = gaussian_kde(lms, bw_method='scott')
        x_lo = min(lms.min() - 0.3, log_de - 0.3)
        x_hi = max(lms.max() + 0.3, log_de + 0.3)
        xg   = np.linspace(x_lo, x_hi, 500)
        yk   = kde(xg)

        ax.plot(xg, yk, color='steelblue', lw=2)
        ax.fill_between(xg, yk, alpha=0.25, color='steelblue')

        log_med = np.log10(np.median(mass_samp[valid]))
        log_lo  = np.log10(np.percentile(mass_samp[valid], 16))
        log_hi  = np.log10(np.percentile(mass_samp[valid], 84))

        ax.axvline(log_med,  color='steelblue', lw=1.5, ls='--',
                   label=f'median = {log_med:.2f}')
        ax.axvspan(log_lo, log_hi, alpha=0.15, color='steelblue', label=r'1$\sigma$')
        ax.axvline(log_de,  color='tomato',    lw=2,   ls='-',
                   label=f'DE best = {log_de:.2f}')

        xlabel = r'$\log_{10}(M_{\rm vir} / M_\odot)$' if model_type == 'nfw' \
            else r'$\log_{10}(M / M_\odot)$'
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel('Posterior density', fontsize=13)
        title_map = {
            'pointmass': 'Point mass',
            'nfw': 'NFW',
            'p_jaffe': 'Pseudo-Jaffe',
            'king': 'King',
        }
        ax.set_title(f'{title_map[model_type]} Sub-halo {idx} mass posterior', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, ls=':', alpha=0.4)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 质量一维后验分布图: {os.path.basename(output_path)}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        后验统计保存                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

def save_posterior(samples, param_names, model_type, active_subhalos,
                   subhalos, args, nwalkers, output_path):
    pph = params_per_subhalo(model_type)
    stats = {}
    for i, name in enumerate(param_names):
        med = np.median(samples[:, i])
        lo  = np.percentile(samples[:, i], 16)
        hi  = np.percentile(samples[:, i], 84)
        stats[name] = {'median': med, 'lo': lo, 'hi': hi,
                       'ep': hi - med, 'em': med - lo}

    with open(output_path, 'w') as f:
        f.write("# MCMC Posterior Summary  (mcmc_from_result.py)\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Model: {model_type}\n")
        f.write(f"# active_subhalos: {active_subhalos}\n")
        f.write(f"# Walkers: {nwalkers}, Steps: {args.nsteps}, "
                f"Burn-in: {args.burnin}, Thin: {args.thin}\n")
        f.write(f"# Effective samples: {len(samples)}\n\n")
        f.write("# parameter  median  16%  84%  +err  -err\n\n")
        for name, s in stats.items():
            f.write(f"{name}  {s['median']:.10e}  {s['lo']:.10e}  "
                    f"{s['hi']:.10e}  {s['ep']:.10e}  {s['em']:.10e}\n")

        f.write("\n# Summary for paper:\n")
        for i, idx in enumerate(active_subhalos):
            f.write(f"# Sub-halo {idx}:\n")
            for key in param_names[i*pph : (i+1)*pph]:
                s = stats[key]
                if key.startswith('x_') or key.startswith('y_'):
                    f.write(f"#   {key} = {s['median']:.6f}"
                            f" +{s['ep']*1000:.3f} -{s['em']*1000:.3f} mas\n")
                elif key.startswith('logM_'):
                    f.write(f"#   {key} = {s['median']:.3f}"
                            f" +{s['ep']:.3f} -{s['em']:.3f} dex"
                            f"  (M = {10**s['median']:.3e} M_sun)\n")
                elif key.startswith('sig_'):
                    f.write(f"#   {key} = {s['median']:.3f}"
                            f" +{s['ep']:.3f} -{s['em']:.3f} km/s\n")
                elif key.startswith('a_') or key.startswith('rco_') or key.startswith('rc_'):
                    f.write(f"#   {key} = {s['median']*1000:.3f}"
                            f" +{s['ep']*1000:.3f} -{s['em']*1000:.3f} mas\n")
                elif key.startswith('c_'):
                    f.write(f"#   {key} = {s['median']:.4f}"
                            f" +{s['ep']:.4f} -{s['em']:.4f}\n")
                else:
                    f.write(f"#   {key} = {s['median']:.4f}"
                            f" +{s['ep']:.4f} -{s['em']:.4f}\n")

            # 质量
            if model_type in ('pointmass', 'nfw', 'king'):
                lm = stats.get(f'logM_{idx}', {})
                if lm:
                    f.write(f"#   M = {10**lm['median']:.3e}"
                            f" +{(10**(lm['median']+lm['ep'])-10**lm['median']):.3e}"
                            f" -{(10**lm['median']-10**(lm['median']-lm['em'])):.3e}"
                            f" M_sun\n")
            elif model_type == 'p_jaffe':
                s_samp = samples[:, i*pph + 2]
                a_samp = samples[:, i*pph + 3]
                r_samp = samples[:, i*pph + 4]
                ms = np.array([calculate_jaffe_mass(sg, a, r)
                               for sg, a, r in zip(s_samp, a_samp, r_samp)])
                ms = ms[ms > 0]
                if len(ms) > 0:
                    f.write(f"#   M_jaffe = {np.median(ms):.3e}"
                            f" +{np.percentile(ms,84)-np.median(ms):.3e}"
                            f" -{np.median(ms)-np.percentile(ms,16):.3e}"
                            f" M_sun\n")
            f.write("\n")

    print(f"  ✓ 后验统计: {os.path.basename(output_path)}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                            main                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description='从 DE 结果文件夹出发运行 MCMC 后验采样',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('folder', help='包含 *_best_params.txt 的结果文件夹')
    parser.add_argument('--nwalkers',     type=int,   default=DEFAULT_MCMC_NWALKERS,
                        help=f'walker 数量 [默认: {DEFAULT_MCMC_NWALKERS}]')
    parser.add_argument('--nsteps',       type=int,   default=DEFAULT_MCMC_NSTEPS,
                        help=f'采样步数 [默认: {DEFAULT_MCMC_NSTEPS}]')
    parser.add_argument('--burnin',       type=int,   default=DEFAULT_MCMC_BURNIN,
                        help=f'burn-in 步数 [默认: {DEFAULT_MCMC_BURNIN}]')
    parser.add_argument('--thin',         type=int,   default=DEFAULT_MCMC_THIN,
                        help=f'稀疏采样，每N步保留1个 [默认: {DEFAULT_MCMC_THIN}]')
    parser.add_argument('--perturbation', type=float, default=DEFAULT_MCMC_PERTURBATION,
                        help=f'初始 walker 扰动幅度（相对于参数绝对值）[默认: {DEFAULT_MCMC_PERTURBATION}]')
    parser.add_argument('--workers',      type=int,   default=DEFAULT_MCMC_WORKERS,
                        help=f'并行核心数（-1=全部CPU，1=串行）[默认: {DEFAULT_MCMC_WORKERS}]')
    parser.add_argument('--no-progress',  dest='progress', action='store_false',
                        help='关闭 tqdm 进度条（默认显示）')
    parser.set_defaults(progress=DEFAULT_MCMC_PROGRESS)
    parser.add_argument('--baseline_dir', type=str,   default='',
                        help='含 bestfit.dat 的目录（留空则自动搜索）')
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"[错误] 文件夹不存在: {folder}")
        sys.exit(1)

    print("=" * 70)
    print("mcmc_from_result.py — 从 DE 结果出发进行 MCMC 后验采样")
    print("=" * 70)
    print(f"  结果文件夹: {folder}")

    # ── 1. 找到 best_params.txt ────────────────────────────────────────
    cands = glob.glob(os.path.join(folder, '*_best_params.txt'))
    if not cands:
        print(f"[错误] 未在 {folder} 中找到 *_best_params.txt")
        sys.exit(1)
    param_file = sorted(cands)[0]
    print(f"  参数文件: {os.path.basename(param_file)}")
    prefix = os.path.basename(param_file).replace('_best_params.txt', '')

    # ── 2. 解析参数 ────────────────────────────────────────────────────
    parsed = parse_best_params(param_file)
    model_type     = parsed['model_type']
    active_subhalos = parsed['active_subhalos']
    subhalos        = parsed['subhalos']
    print(f"  模型类型: {model_type}")
    print(f"  active_subhalos: {active_subhalos}")
    print(f"  子晕数量: {len(subhalos)}")
    if parsed['de_seed']:
        print(f"  DE_SEED: {parsed['de_seed']}")
    if parsed['chi2_best']:
        print(f"  DE chi2: {parsed['chi2_best']:.2f}")

    # ── 3. 加载基础透镜参数 ────────────────────────────────────────────
    lens_params  = DEFAULT_LENS_PARAMS
    main_lens_key = DEFAULT_MAIN_LENS_KEY
    src_x, src_y = SOURCE_X, SOURCE_Y

    # 优先级：--baseline_dir > 自动向上搜索 > 默认
    search_dirs = []
    if args.baseline_dir:
        search_dirs = [args.baseline_dir]
    else:
        d = find_bestfit_dir(folder)
        if d:
            search_dirs = [d]

    for d in search_dirs:
        lp, sx, sy, mlk = load_baseline_lens_params(d)
        if lp is not None:
            lens_params   = lp
            main_lens_key = mlk
            src_x, src_y  = sx, sy
            print(f"  基础透镜: 从 {d} 加载 (主透镜: {mlk})")
            break
    else:
        print(f"  基础透镜: 使用内置默认参数 (SIE)")

    # ── 4. 构造初始参数向量 ────────────────────────────────────────────
    p0 = build_initial_params(model_type, subhalos)
    param_names, corner_labels = build_param_names(model_type, active_subhalos)
    ndim = len(p0)
    pph  = params_per_subhalo(model_type)
    print(f"  参数维度: {ndim}")

    # ── 5. 检查依赖库 ──────────────────────────────────────────────────
    try:
        import emcee, corner as _corner
        from tqdm import tqdm
    except ImportError as e:
        print(f"[错误] 缺少依赖: {e}\n  请运行: pip install emcee corner tqdm")
        sys.exit(1)

    # ── 6. 构建 log_probability ────────────────────────────────────────
    log_prob = build_log_probability(model_type, active_subhalos,
                                     lens_params, main_lens_key)

    # 打印先验范围摘要
    print(f"\n  先验范围: 位置半径 ±{MCMC_SEARCH_RADIUS*1000:.0f} mas")
    if model_type == 'pointmass':
        print(f"    logM ∈ [{MCMC_PM_LOG_M_MIN}, {MCMC_PM_LOG_M_MAX}] dex")
    elif model_type == 'nfw':
        print(f"    logM ∈ [{MCMC_NFW_LOG_M_MIN}, {MCMC_NFW_LOG_M_MAX}] dex")
        print(f"    c    ∈ [{MCMC_NFW_C_MIN}, {MCMC_NFW_C_MAX}]")
    elif model_type == 'p_jaffe':
        print(f"    sig  ∈ [{MCMC_PJ_SIG_MIN}, {MCMC_PJ_SIG_MAX}] km/s")
        print(f"    a    ∈ [{MCMC_PJ_A_MIN*1000:.3f}, {MCMC_PJ_A_MAX*1000:.0f}] mas")
        print(f"    rco  ∈ [{MCMC_PJ_RCO_MIN*1000:.4f}, {MCMC_PJ_RCO_MAX*1000:.0f}] mas")
    elif model_type == 'king':
        print(f"    logM ∈ [{MCMC_KING_LOG_M_MIN}, {MCMC_KING_LOG_M_MAX}] dex")
        print(f"    rc   ∈ [{MCMC_KING_RC_MIN*1000:.3f}, {MCMC_KING_RC_MAX*1000:.0f}] mas")
        print(f"    c    ∈ [{MCMC_KING_C_MIN}, {MCMC_KING_C_MAX}]")

    # 验证初始点可行性
    lp_test = log_prob(p0)
    if not np.isfinite(lp_test):
        print(f"[警告] DE 最优解处 log_prob = {lp_test}，初始点可能不可行")
    else:
        print(f"  初始 log_prob = {lp_test:.4f}  (对应 chi2 ≈ {-2*lp_test:.2f})")

    # ── 7. 运行 MCMC ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"开始 MCMC 采样（nsteps={args.nsteps}, burnin={args.burnin}）")
    print(f"{'='*70}")

    sampler, samples, chain, nwalkers_actual = run_mcmc(log_prob, p0, args)

    print(f"\n  总样本数: {nwalkers_actual * args.nsteps}")
    print(f"  有效样本数（去除burn-in）: {len(samples)}")

    # ── 8. 保存 chain ──────────────────────────────────────────────────
    chain_file = os.path.join(folder, f'{prefix}_mcmc_chain.dat')
    np.savetxt(chain_file, samples, header=' '.join(param_names))
    print(f"\n  ✓ MCMC链: {os.path.basename(chain_file)}")

    # ── 9. 生成图表 ────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    print(f"\n生成图表...")

    plot_corner(
        samples, corner_labels, p0,
        os.path.join(folder, f'{prefix}_mcmc_corner.png')
    )
    plot_trace(
        chain, corner_labels, args.burnin,
        os.path.join(folder, f'{prefix}_mcmc_trace.png')
    )
    plot_mass_1d(
        model_type, active_subhalos, samples, subhalos,
        os.path.join(folder, f'{prefix}_mcmc_mass_1d.png')
    )

    # ── 10. 后验统计 ───────────────────────────────────────────────────
    posterior_file = os.path.join(folder, f'{prefix}_mcmc_posterior.txt')
    save_posterior(samples, param_names, model_type, active_subhalos,
                   subhalos, args, nwalkers_actual, posterior_file)

    # ── 完成 ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"完成！输出文件均在: {folder}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
