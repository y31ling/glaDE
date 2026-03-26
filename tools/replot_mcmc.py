#!/usr/bin/env python3
"""
replot_mcmc.py
==============
从已有的 MCMC 结果文件夹中读取链数据，重新生成所有图表。

读取文件：
  *_mcmc_chain.dat   — 扁平化的后验样本（header 含参数名）
  *_best_params.txt  — DE 最优解（用于红色标注线）

输出文件（覆盖写入同一文件夹）：
  *_corner.png            — Corner plot（对角线带百分比纵轴）
  *_mass_posterior_1d.png — logM 一维 KDE 后验图（含 DE 最优线）
  *_mcmc_posterior.txt    — 参数统计摘要（如不存在则新建）

用法：
  python replot_mcmc.py <result_folder>
  python replot_mcmc.py <result_folder> --suffix _v2   # 加后缀避免覆盖原图
  python replot_mcmc.py <result_folder> --no_mass      # 跳过质量图
"""

import sys
import os
import re
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from datetime import datetime

# ── 宇宙学（仅 p_jaffe 质量估算需要） ─────────────────────────────────────
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
_COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
LENS_Z   = 0.2160
SOURCE_Z = 0.4090

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      工具函数                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

def calculate_jaffe_mass(sigma_km_s, a_arcsec, rco_arcsec):
    G  = 4.302e-6
    c  = 299792.458
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
# ║                    链文件解析                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

def load_chain(chain_path):
    """
    读取 *_mcmc_chain.dat，返回 (samples, param_names)。
    文件格式：首行 `# name1 name2 ...`，后续行为数值样本。
    """
    param_names = []
    with open(chain_path) as f:
        for line in f:
            if line.startswith('#'):
                names = line.lstrip('#').split()
                if names:
                    param_names = names
                continue
            break   # 找到第一个非注释行就停止

    samples = np.loadtxt(chain_path, comments='#')
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)

    return samples, param_names


def infer_model_type(param_names):
    """从参数名列表推断模型类型。"""
    joined = ' '.join(param_names)
    if 'sig_' in joined or 'rco_' in joined:
        return 'p_jaffe'
    if 'c_' in joined:
        return 'nfw'
    return 'pointmass'


def param_name_to_label(name):
    """将参数名转换为 LaTeX 标签。"""
    m = re.match(r'^([a-zA-Z_]+)_(\d+)$', name)
    if not m:
        return f'${name}$'
    key, idx = m.group(1), m.group(2)
    label_map = {
        'x':    f'$x_{idx}$',
        'y':    f'$y_{idx}$',
        'sig':  f'$\\sigma_{idx}$',
        'a':    f'$a_{idx}$',
        'rco':  f'$r_{{co,{idx}}}$',
        'logM': f'$\\log M_{idx}$',
        'c':    f'$c_{idx}$',
    }
    return label_map.get(key, f'${name}$')


def extract_active_subhalos(param_names, model_type):
    """从参数名提取 active_subhalos 列表和每个 subhalo 的参数索引。"""
    indices = []
    seen = set()
    for name in param_names:
        m = re.match(r'^[a-zA-Z_]+_(\d+)$', name)
        if m:
            idx = int(m.group(1))
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
    return sorted(indices)


def pph(model_type):
    return {'p_jaffe': 5, 'nfw': 4, 'pointmass': 3}[model_type]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                 best_params.txt 解析（DE 最优解）                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

def parse_best_params(folder):
    """在 folder 内寻找 *_best_params.txt 并解析子晕参数。"""
    cands = glob.glob(os.path.join(folder, '*_best_params.txt'))
    if not cands:
        return None, {}

    filepath = sorted(cands)[0]
    with open(filepath) as f:
        content = f.read()

    subhalos = {}

    def _fill(pattern, key, transform=float):
        for m in re.finditer(pattern, content):
            idx = int(m.group(1))
            subhalos.setdefault(idx, {})[key] = transform(m.group(2))

    # ── 位置（三种模型各自的键名）──────────────────────────────────────
    _fill(r'x_jaffe(\d+)\s*=\s*([\d.eE+-]+)', 'x')   # p_jaffe
    _fill(r'y_jaffe(\d+)\s*=\s*([\d.eE+-]+)', 'y')
    _fill(r'x_nfw(\d+)\s*=\s*([\d.eE+-]+)',   'x')   # nfw
    _fill(r'y_nfw(\d+)\s*=\s*([\d.eE+-]+)',   'y')
    _fill(r'x_sub(\d+)\s*=\s*([\d.eE+-]+)',   'x')   # pointmass
    _fill(r'y_sub(\d+)\s*=\s*([\d.eE+-]+)',   'y')

    # ── p_jaffe 专有参数 ────────────────────────────────────────────────
    _fill(r'sig(\d+)\s*=\s*([\d.eE+-]+)', 'sig')
    for m in re.finditer(r'^a(\d+)\s*=\s*([\d.eE+-]+)', content, re.M):
        subhalos.setdefault(int(m.group(1)), {})['a'] = float(m.group(2))
    for m in re.finditer(r'rco(\d+)\s*=\s*([\d.eE+-]+)', content):
        idx = int(m.group(1))
        if m.start() == 0 or content[m.start()-1] != '/':
            subhalos.setdefault(idx, {})['rco'] = float(m.group(2))

    # ── nfw 专有参数 ────────────────────────────────────────────────────
    _fill(r'm_vir(\d+)\s*=\s*([\d.eE+-]+)', 'm_vir')
    _fill(r'c_vir(\d+)\s*=\s*([\d.eE+-]+)', 'c_vir')

    # ── pointmass 专有参数 ──────────────────────────────────────────────
    _fill(r'mass_sub(\d+)\s*=\s*([\d.eE+-]+)', 'mass')

    return filepath, subhalos


def get_de_mass(model_type, sh_dict):
    """从解析的 subhalo 字典计算 DE 最优质量（M_sun）。"""
    if model_type == 'p_jaffe':
        sig = sh_dict.get('sig', 0)
        a   = sh_dict.get('a',   0)
        rco = sh_dict.get('rco', 0)
        if sig > 0 and a > 0 and rco > 0 and a > rco:
            return calculate_jaffe_mass(sig, a, rco)
        return None
    elif model_type == 'nfw':
        m = sh_dict.get('m_vir')
        return m if m and m > 0 else None
    else:  # pointmass
        m = sh_dict.get('mass')
        return m if m and m > 0 else None


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        绘图函数                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

def build_truth_vector(param_names, model_type, subhalos_de):
    """
    将 subhalos_de 字典映射为与 param_names 顺序一致的参数向量，
    用作 corner.corner() 的 truths 参数（红色标注线）。
    找不到对应值的参数填 None（corner 会跳过该参数的标注线）。
    """
    truths = []
    for name in param_names:
        m = re.match(r'^([a-zA-Z_]+)_(\d+)$', name)
        if not m:
            truths.append(None)
            continue
        key, idx = m.group(1), int(m.group(2))
        sh = subhalos_de.get(idx, {})

        if key == 'x':
            truths.append(sh.get('x'))
        elif key == 'y':
            truths.append(sh.get('y'))
        elif key == 'sig':                          # p_jaffe
            truths.append(sh.get('sig'))
        elif key == 'a':                            # p_jaffe
            truths.append(sh.get('a'))
        elif key == 'rco':                          # p_jaffe
            truths.append(sh.get('rco'))
        elif key == 'logM':                         # nfw / pointmass
            if 'm_vir' in sh and sh['m_vir'] > 0:
                truths.append(np.log10(sh['m_vir']))
            elif 'mass' in sh and sh['mass'] > 0:
                truths.append(np.log10(sh['mass']))
            else:
                truths.append(None)
        elif key == 'c':                            # nfw
            truths.append(sh.get('c_vir'))
        else:
            truths.append(None)

    # 若全为 None 则返回 None（不传 truths 给 corner）
    if all(v is None for v in truths):
        return None
    return truths


def make_corner(samples, labels, output_path, truths=None):
    """Corner plot，对角线面板显示百分比纵轴，可选 DE 最优解标注线。"""
    import corner

    N    = len(samples)
    ndim = samples.shape[1]

    corner_kw = dict(
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3f',
        hist_kwargs={'alpha': 0.75},
    )
    if truths is not None:
        corner_kw['truths']      = truths
        corner_kw['truth_color'] = 'red'

    fig = corner.corner(samples, **corner_kw)

    # 对角线面板右侧建百分比纵轴（twinx，corner 不会隐藏它）
    from matplotlib.ticker import MaxNLocator
    grid = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        ax = grid[i, i]
        ylo, yhi = ax.get_ylim()
        ax2 = ax.twinx()
        ax2.set_ylim(ylo / N * 100, yhi / N * 100)
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='lower'))
        ax2.tick_params(axis='y', labelsize=7, length=3, width=0.8)
        ax2.set_ylabel('%', fontsize=8, rotation=0, labelpad=10, va='center')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Corner plot: {os.path.basename(output_path)}")


def make_mass_1d(model_type, active_subhalos, samples, param_names,
                 subhalos_de, output_path):
    """logM 一维 KDE 后验分布图，含 DE 最优解标注。"""
    n_halos = len(active_subhalos)
    n_pp    = pph(model_type)
    fig, axes = plt.subplots(1, n_halos, figsize=(5 * n_halos, 4))
    if n_halos == 1:
        axes = [axes]

    for i, idx in enumerate(active_subhalos):
        ax = axes[i]

        # 从 chain 列提取质量样本
        if model_type == 'p_jaffe':
            sig_col = i * n_pp + 2
            a_col   = i * n_pp + 3
            rco_col = i * n_pp + 4
            if samples.shape[1] <= rco_col:
                ax.text(0.5, 0.5, '数据列不足', ha='center', va='center',
                        transform=ax.transAxes)
                continue
            sig_s = samples[:, sig_col]
            a_s   = samples[:, a_col]
            rco_s = samples[:, rco_col]
            mass_s = np.array([calculate_jaffe_mass(sg, a, r)
                                for sg, a, r in zip(sig_s, a_s, rco_s)])
        else:  # nfw or pointmass: logM column
            lm_col = i * n_pp + 2
            if samples.shape[1] <= lm_col:
                ax.text(0.5, 0.5, '数据列不足', ha='center', va='center',
                        transform=ax.transAxes)
                continue
            mass_s = 10 ** samples[:, lm_col]

        valid = mass_s > 0
        if valid.sum() < 10:
            ax.text(0.5, 0.5, '有效样本不足', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        lms = np.log10(mass_s[valid])
        kde = gaussian_kde(lms, bw_method='scott')

        # DE 最优质量（如果可用）
        de_mass = None
        if idx in subhalos_de:
            de_mass = get_de_mass(model_type, subhalos_de[idx])

        x_lo = lms.min() - 0.3
        x_hi = lms.max() + 0.3
        if de_mass and de_mass > 0:
            log_de = np.log10(de_mass)
            x_lo = min(x_lo, log_de - 0.3)
            x_hi = max(x_hi, log_de + 0.3)

        xg = np.linspace(x_lo, x_hi, 500)
        yk = kde(xg)

        ax.plot(xg, yk, color='steelblue', lw=2)
        ax.fill_between(xg, yk, alpha=0.25, color='steelblue')

        log_med = np.log10(np.median(mass_s[valid]))
        log_lo  = np.log10(np.percentile(mass_s[valid], 16))
        log_hi  = np.log10(np.percentile(mass_s[valid], 84))

        ax.axvline(log_med, color='steelblue', lw=1.5, ls='--',
                   label=f'median = {log_med:.2f}')
        ax.axvspan(log_lo, log_hi, alpha=0.15, color='steelblue',
                   label=r'1$\sigma$')

        if de_mass and de_mass > 0:
            log_de = np.log10(de_mass)
            ax.axvline(log_de, color='tomato', lw=2, ls='-',
                       label=f'DE best = {log_de:.2f}')

        xlabel = (r'$\log_{10}(M_{\rm vir}/M_\odot)$' if model_type == 'nfw'
                  else r'$\log_{10}(M/M_\odot)$')
        title_map = {'p_jaffe': 'Pseudo-Jaffe', 'nfw': 'NFW',
                     'pointmass': 'Point mass'}
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Posterior density', fontsize=12)
        ax.set_title(f'{title_map[model_type]} Sub-halo {idx}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, ls=':', alpha=0.4)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 质量一维图: {os.path.basename(output_path)}")


def make_stats_txt(samples, param_names, model_type, active_subhalos,
                   subhalos_de, output_path):
    """生成参数统计摘要文件。"""
    n_pp = pph(model_type)
    with open(output_path, 'w') as f:
        f.write("# MCMC Posterior Summary (replot_mcmc.py)\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Model: {model_type},  N_samples: {len(samples)}\n\n")
        f.write("# parameter  median  16%  84%  +err  -err\n\n")

        for i, name in enumerate(param_names):
            if i >= samples.shape[1]:
                break
            col = samples[:, i]
            med = np.median(col)
            lo  = np.percentile(col, 16)
            hi  = np.percentile(col, 84)
            f.write(f"{name}  {med:.10e}  {lo:.10e}  {hi:.10e}"
                    f"  {hi-med:.10e}  {med-lo:.10e}\n")

        f.write("\n# Mass posterior (M_sun):\n")
        for i, idx in enumerate(active_subhalos):
            if model_type == 'p_jaffe':
                c0, c1, c2 = i*n_pp+2, i*n_pp+3, i*n_pp+4
                if samples.shape[1] > c2:
                    ms = np.array([calculate_jaffe_mass(sg, a, r)
                                   for sg, a, r in zip(samples[:, c0],
                                                        samples[:, c1],
                                                        samples[:, c2])])
                    ms = ms[ms > 0]
                    if len(ms) > 0:
                        f.write(f"mass_{idx}  {np.median(ms):.10e}"
                                f"  {np.percentile(ms,16):.10e}"
                                f"  {np.percentile(ms,84):.10e}\n")
            else:
                lm_col = i * n_pp + 2
                if samples.shape[1] > lm_col:
                    ms = 10 ** samples[:, lm_col]
                    f.write(f"mass_{idx}  {np.median(ms):.10e}"
                            f"  {np.percentile(ms,16):.10e}"
                            f"  {np.percentile(ms,84):.10e}\n")
    print(f"  ✓ 统计摘要: {os.path.basename(output_path)}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                            main                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description='从已有 MCMC 链数据重新生成图表',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('folder', help='包含 *_mcmc_chain.dat 的文件夹')
    parser.add_argument('--suffix', default='',
                        help='输出文件名后缀（如 _v2），避免覆盖原图')
    parser.add_argument('--no_mass', action='store_true',
                        help='跳过质量一维图')
    parser.add_argument('--no_stats', action='store_true',
                        help='跳过统计摘要文件')
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"[错误] 文件夹不存在: {folder}")
        sys.exit(1)

    print("=" * 70)
    print("replot_mcmc.py — 从已有 MCMC 链重新生成图表")
    print("=" * 70)
    print(f"  文件夹: {folder}")

    # ── 1. 找到 chain 文件 ──────────────────────────────────────────────
    chain_cands = (glob.glob(os.path.join(folder, '*_mcmc_chain.dat')) +
                   glob.glob(os.path.join(folder, '*_chain.dat')))
    if not chain_cands:
        print("[错误] 未找到 *_mcmc_chain.dat 文件")
        sys.exit(1)
    chain_path = sorted(chain_cands)[0]
    prefix = os.path.basename(chain_path).replace('_mcmc_chain.dat', '').replace('_chain.dat', '')
    print(f"  Chain 文件: {os.path.basename(chain_path)}")
    print(f"  输出前缀: {prefix}")

    # ── 2. 加载数据 ─────────────────────────────────────────────────────
    samples, param_names = load_chain(chain_path)
    print(f"  样本数: {len(samples)},  参数数: {samples.shape[1]}")
    if not param_names:
        print("[警告] 未找到参数名，使用 col_0, col_1, ...")
        param_names = [f'col_{i}' for i in range(samples.shape[1])]
    elif len(param_names) < samples.shape[1]:
        param_names += [f'col_{i}' for i in range(len(param_names), samples.shape[1])]
    print(f"  参数: {param_names}")

    # ── 3. 推断模型类型 ─────────────────────────────────────────────────
    model_type = infer_model_type(param_names)
    active_subhalos = extract_active_subhalos(param_names, model_type)
    print(f"  模型类型: {model_type}")
    print(f"  active_subhalos: {active_subhalos}")

    # ── 4. 解析 DE 最优解 ────────────────────────────────────────────────
    bp_file, subhalos_de = parse_best_params(folder)
    if bp_file:
        print(f"  DE 最优解: {os.path.basename(bp_file)}")
    else:
        print("  [提示] 未找到 *_best_params.txt，跳过 DE 标注线")

    # ── 5. 构建标签 ──────────────────────────────────────────────────────
    labels = [param_name_to_label(n) for n in param_names]

    # ── 6. 生成 Corner Plot ──────────────────────────────────────────────
    print(f"\n生成 Corner plot...")
    truths = None
    if subhalos_de:
        truths = build_truth_vector(param_names, model_type, subhalos_de)
        if truths is not None:
            found = sum(v is not None for v in truths)
            print(f"  DE 最优解标注: {found}/{len(truths)} 个参数")
        else:
            print("  [提示] DE 参数与链参数无法对应，跳过 truths")
    corner_path = os.path.join(folder, f'{prefix}_corner{args.suffix}.png')
    make_corner(samples, labels, corner_path, truths=truths)

    # ── 7. 生成质量一维图 ────────────────────────────────────────────────
    if not args.no_mass:
        print(f"\n生成质量一维后验图...")
        mass_path = os.path.join(folder,
                                 f'{prefix}_mass_posterior_1d{args.suffix}.png')
        make_mass_1d(model_type, active_subhalos, samples, param_names,
                     subhalos_de, mass_path)

    # ── 8. 生成统计摘要 ──────────────────────────────────────────────────
    if not args.no_stats:
        print(f"\n生成统计摘要...")
        stats_path = os.path.join(folder,
                                  f'{prefix}_replot_posterior{args.suffix}.txt')
        make_stats_txt(samples, param_names, model_type, active_subhalos,
                       subhalos_de, stats_path)

    print(f"\n{'='*70}")
    print(f"完成！输出文件在: {folder}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
