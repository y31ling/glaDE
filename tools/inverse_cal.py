#!/usr/bin/env python3
"""
inverse_cal.py — 引力透镜正向成像计算工具
==========================================

给定源位置和透镜参数，计算成像数量、位置、放大率和时间延迟。

用法：
  python inverse_cal.py                  # 使用脚本内定义的参数
  python inverse_cal.py --dat path/to/bestfit.dat   # 从 bestfit.dat 加载透镜
  python inverse_cal.py --no-plot        # 不生成图像

配置说明：
  直接修改本文件 §1 CONFIG 节中的参数即可。
"""

import sys
import os
import argparse

# ── 路径设置 ──────────────────────────────────────────────
_GLADE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_GLADE_ROOT, '..', '..', 'glafic2', 'python'))

os.environ.setdefault('LD_LIBRARY_PATH', ':'.join([
    '/home/luukiaun/glafic251018/gsl-2.8/.libs',
    '/home/luukiaun/glafic251018/fftw-3.3.10/.libs',
    '/home/luukiaun/glafic251018/cfitsio-4.6.2/.libs',
]))

import glafic
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ══════════════════════════════════════════════════════════════════
# §1  CONFIG — 在此修改所有参数
# ══════════════════════════════════════════════════════════════════

# ── 1a. 宇宙学参数 ───────────────────────────────────────
OMEGA       = 0.3      # 物质密度参数
LAMBDA_COSMO = 0.7     # 暗能量参数
WEOS        = -1.0     # 暗能量状态方程
HUBBLE      = 0.7      # 哈勃常数（无量纲）

# ── 1b. 计算网格（arcsec） ───────────────────────────────
#   以透镜为中心展开，覆盖所有可能成像区域
XMIN, YMIN  = -2.0, -2.0
XMAX, YMAX  =  2.0,  2.0
PIX_EXT     = 0.02     # 延展源像素尺寸
PIX_POI     = 0.2      # 点源搜索像素
MAXLEV      = 5        # 细分层数（越大越精确，越慢）

# ── 1c. 红移 ─────────────────────────────────────────────
LENS_Z   = 0.2     # 透镜红移
SOURCE_Z = 0.4     # 源红移

# ── 1d. 坐标约定 ─────────────────────────────────────────
#   obs_x_flip = True  → 输入位置为天球坐标（RA 东=正），程序内部取负转数学坐标
#   obs_x_flip = False → 输入位置为数学坐标（右=正），直接使用
OBS_X_FLIP = False

# ── 1e. 源位置（arcsec，与 obs_x_flip 约定一致） ─────────
SOURCE_X =  0    # 源面 x 坐标
SOURCE_Y =  0    # 源面 y 坐标

# ── 1f. 透镜列表 ─────────────────────────────────────────
#   每条记录格式（也是 glafic set_lens 参数顺序）：
#   (类型, 红移, p1, p2, p3, p4, p5, p6, p7)
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │ 类型   │ p1          │ p2/p3       │ p4        │ p5    │ p6   │ p7    │
#   ├─────────────────────────────────────────────────────────────────────────┤
#   │ sie    │ σ [km/s]    │ x,y ["]     │ 椭率 e    │ PA[°] │ 核心 │ 截断  │
#   │ sis    │ σ [km/s]    │ x,y ["]     │ 0         │ 0     │ 0    │ 0     │
#   │ anfw   │ M [M☉]      │ x,y ["]     │ 集中度 c  │ 椭率  │ PA[°]│ 0     │
#   │ gnfw   │ M [M☉]      │ x,y ["]     │ 集中度 c  │ 椭率  │ PA[°]│ 内斜率│
#   │ tnfw   │ M [M☉]      │ x,y ["]     │ 集中度 c  │ 椭率  │ PA[°]│ 截断r │
#   │ hern   │ M [M☉]      │ x,y ["]     │ 有效半径  │ 椭率  │ PA[°]│ 0     │
#   │ jaffe  │ M [M☉]      │ x,y ["]     │ 有效半径  │ 椭率  │ PA[°]│ 0     │
#   │ sers   │ M [M☉]      │ x,y ["]     │ 椭率      │ PA[°] │ Re["]│ n     │
#   │ pgc    │ σ [km/s]    │ x,y ["]     │ 椭率      │ PA[°] │ 核心 │ 截断  │
#   └─────────────────────────────────────────────────────────────────────────┘
#   椭率 e = (a²-b²)/(a²+b²)，PA 为位置角（度，北向东）

LENSES = [
    # name          type     z        p1             p2        p3        p4       p5      p6      p7
    ('sie',       'sie',   0.2,  150,      0, 0, 0.5,   0,  0.0,     0.0   ),
]

# ── 1g. 位置不确定性估算 ──────────────────────────────────
#
#  【模式 A — 观测 sigma（仪器噪声主导）】
#    给出最亮像的参考 sigma（mas），其余各像按 S/N ~ sqrt(|μ|) 推算：
#      σ_obs_i = σ_ref * sqrt(μ_ref / |μ_i|)
#    （图像越暗，astrometric 测量越不精确）
#    设为 None 则跳过此模式。
OBS_SIGMA_REF_MAS = 0.5   # 最亮像的观测位置 sigma [mas]，None=不计算

#  【模式 B — 源面不确定性传播（理论值）】
#    给出源面位置 sigma（mas），映射到像面后：
#      σ_src_i = σ_source * sqrt(|μ_i|)
#    （放大率越高，像面位置对源面偏移越敏感）
#    设为 None 则跳过此模式。
SOURCE_SIGMA_MAS = None    # 源面位置 sigma [mas]，None=不计算

# ── 1h. 输出设置 ──────────────────────────────────────────
SAVE_PLOT   = True                          # 是否保存示意图
PLOT_FILE   = 'inverse_cal_result.png'      # 图片文件名（保存在 tools/ 目录）
SHOW_GRID   = True                          # 网格坐标范围内才显示像
# 中心偏移（与坐标约定一致，仅用于输出显示坐标系对齐，不影响计算）
CENTER_OFFSET_X = 0.0
CENTER_OFFSET_Y = 0.0

# ── 1i. bestfit.dat 路径（None = 使用上方手动配置；命令行 --dat 可覆盖）
BESTFIT_DAT = None

# ══════════════════════════════════════════════════════════════════
# §2  工具函数
# ══════════════════════════════════════════════════════════════════

def load_bestfit(path: str):
    """解析 bestfit.dat，返回 (lenses, source_x, source_y)"""
    lens_rows, point_params = [], None
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            if parts[0] == 'lens':
                lens_rows.append(parts)
            elif parts[0] == 'point':
                point_params = parts
    if not lens_rows:
        raise ValueError(f"bestfit.dat 中未找到 lens 行: {path}")
    if point_params is None:
        raise ValueError(f"bestfit.dat 中未找到 point 行: {path}")

    lenses = []
    sers_count, type_counts = 0, {}
    for i, parts in enumerate(lens_rows):
        ltype = parts[1]
        z = float(parts[2])
        raw = [float(v) for v in parts[3:]]
        p = (raw + [0.0] * 7)[:7]
        if ltype == 'sers':
            sers_count += 1
            name = f'sers{sers_count}'
        else:
            type_counts[ltype] = type_counts.get(ltype, 0) + 1
            n = type_counts[ltype]
            name = ltype if n == 1 else f'{ltype}{n}'
        lenses.append((name, ltype, z, *p))
    sx = float(point_params[2])
    sy = float(point_params[3])
    return lenses, sx, sy


def coord_convert(lenses, source_x, source_y, x_flip: bool):
    """应用坐标翻转（天球坐标 → 数学坐标），不修改原数据，返回新副本"""
    x_sign = -1 if x_flip else 1
    converted = []
    for entry in lenses:
        name, ltype, z = entry[0], entry[1], entry[2]
        p = list(entry[3:])
        p[1] = x_sign * p[1]   # p2 = x 坐标
        converted.append((name, ltype, z, *p))
    return converted, x_sign * source_x, source_y


def run_glafic(lenses, source_x, source_y):
    """调用 glafic 计算成像，返回原始结果列表 [(x, y, mu), ...]"""
    glafic.init(OMEGA, LAMBDA_COSMO, WEOS, HUBBLE,
                '_inverse_cal_tmp',
                XMIN, YMIN, XMAX, YMAX,
                PIX_EXT, PIX_POI, MAXLEV, verb=0)
    glafic.startup_setnum(len(lenses), 0, 1)
    for i, entry in enumerate(lenses):
        name, ltype, z = entry[0], entry[1], entry[2]
        p = list(entry[3:])
        glafic.set_lens(i + 1, ltype, z, *p)
    glafic.set_point(1, SOURCE_Z, source_x, source_y)
    glafic.model_init(verb=0)
    result = glafic.point_solve(SOURCE_Z, source_x, source_y, verb=0)
    glafic.quit()
    return result


def print_results(images, source_x, source_y, lenses, x_flip: bool):
    x_sign = -1 if x_flip else 1
    coord_note = "天球坐标(RA东=正)" if x_flip else "数学坐标(右=正)"

    print()
    print("═" * 80)
    print("  inverse_cal — 成像结果")
    print("═" * 80)
    print(f"  坐标约定 : {coord_note}")
    print(f"  源位置   : x = {source_x * x_sign:+.6f}\"  "
          f"y = {source_y:+.6f}\"  (z = {SOURCE_Z})")
    print(f"  透镜数量 : {len(lenses)}")
    for entry in lenses:
        name, ltype, z, *p = entry
        print(f"    [{name}] {ltype}  z={z}  "
              f"x={p[1]*x_sign:+.5f}\"  y={p[2]:+.5f}\"  "
              f"p1={p[0]:.4g}")
    print()
    print(f"  成像数量 : {len(images)}")
    print()

    # 按 |μ| 降序排列
    images_sorted = sorted(images, key=lambda r: abs(r[2]), reverse=True)
    has_td = len(images_sorted[0]) >= 4 if images_sorted else False
    mus = [abs(r[2]) for r in images_sorted]
    mu_max = mus[0] if mus else 1.0

    # 决定是否输出 sigma 列
    use_obs_sigma = OBS_SIGMA_REF_MAS is not None
    use_src_sigma = SOURCE_SIGMA_MAS is not None

    # 表头
    sigma_hdr = ""
    if use_obs_sigma:
        sigma_hdr += f"  {'σ_obs(mas)':<12}"
    if use_src_sigma:
        sigma_hdr += f"  {'σ_src(mas)':<12}"

    header = (f"  {'#':>3}  {'x (\")':<12} {'y (\")':<12} "
              f"{'x (mas)':<10} {'y (mas)':<10} "
              f"{'|μ|':<9} {'μ':<10} {'奇偶':<4}"
              f"{sigma_hdr}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    total_flux = 0.0
    td_ref = images_sorted[0][3] if has_td else 0.0
    for i, row in enumerate(images_sorted):
        ix, iy, imu = row[0], row[1], row[2]
        td = row[3] if has_td else None
        mu_abs = abs(imu)
        # 成像坐标 + center_offset 后转回用户坐标系
        ox = (ix + CENTER_OFFSET_X) * x_sign
        oy = iy + CENTER_OFFSET_Y
        ox_mas = ox * 1000
        oy_mas = oy * 1000
        parity = "+" if imu >= 0 else "−"
        total_flux += mu_abs

        sigma_vals = ""
        if use_obs_sigma:
            # 仪器噪声主导：亮度越高定位越精确，σ ∝ 1/√|μ|
            s_obs = OBS_SIGMA_REF_MAS * np.sqrt(mu_max / mu_abs)
            sigma_vals += f"  {s_obs:<12.3f}"
        if use_src_sigma:
            # 源面不确定性传播：放大率越高像面越散，σ ∝ √|μ|
            s_src = SOURCE_SIGMA_MAS * np.sqrt(mu_abs)
            sigma_vals += f"  {s_src:<12.3f}"

        td_str = f"  Δt={td - td_ref:+.4f}d" if td is not None else ""
        print(f"  {i+1:>3}  {ox:+.6f}\"  {oy:+.6f}\"  "
              f"{ox_mas:+9.3f}   {oy_mas:+9.3f}   "
              f"{mu_abs:<9.4f} {imu:+.5f}   {parity}"
              f"{sigma_vals}{td_str}")

    print()
    print(f"  总放大率 Σ|μ| = {total_flux:.4f}")
    print()

    # sigma 说明
    if use_obs_sigma or use_src_sigma:
        print("  ─── 位置 sigma 说明 ───────────────────────────────────")
    if use_obs_sigma:
        print(f"  σ_obs : 最亮像参考 σ = {OBS_SIGMA_REF_MAS:.2f} mas，"
              f"按 σ_i = σ_ref × √(|μ_max|/|μ_i|) 推算")
        print(f"          （仪器噪声主导：暗像定位更难，σ 更大）")
    if use_src_sigma:
        print(f"  σ_src : 源面位置 σ = {SOURCE_SIGMA_MAS:.2f} mas，"
              f"按 σ_i = σ_source × √|μ_i| 传播到像面")
        print(f"          （放大倍数越高，像面位置对源面偏移越敏感）")
    if use_obs_sigma or use_src_sigma:
        print()

    # 通量比（以最亮像为基准）
    if len(images_sorted) > 1:
        mus = [abs(r[2]) for r in images_sorted]
        print("  通量比（以像1为基准）: " +
              "  ".join(f"{m/mus[0]:.3f}" for m in mus))
    print()
    print("═" * 80)


def plot_results(images, source_x, source_y, lenses, x_flip: bool, outfile: str):
    x_sign = -1 if x_flip else 1
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')
    ax.set_facecolor('#0d0d1a')
    fig.patch.set_facecolor('#0d0d1a')

    # 绘制爱因斯坦环（近似）
    for entry in lenses:
        name, ltype, z = entry[0], entry[1], entry[2]
        p = list(entry[3:])
        lx = p[1] * x_sign + CENTER_OFFSET_X
        ly = p[2] + CENTER_OFFSET_Y
        ax.plot(lx, ly, 'x', color='#facc15', ms=10, mew=2, zorder=5)
        ax.annotate(name, (lx, ly), textcoords='offset points',
                    xytext=(6, 6), color='#fde68a', fontsize=8)

    # 绘制成像
    images_sorted = sorted(images, key=lambda r: abs(r[2]), reverse=True)
    mus = [abs(r[2]) for r in images_sorted]
    mu_max = max(mus) if mus else 1.0
    colors_pos = ['#38bdf8', '#7dd3fc', '#bae6fd']
    colors_neg = ['#f87171', '#fca5a5', '#fecaca']
    for i, row in enumerate(images_sorted):
        ix, iy, imu = row[0], row[1], row[2]
        ox = (ix + CENTER_OFFSET_X) * x_sign
        oy = iy + CENTER_OFFSET_Y
        size = 60 + 200 * (abs(imu) / mu_max)
        color = colors_pos[i % 3] if imu >= 0 else colors_neg[i % 3]
        ax.scatter(ox, oy, s=size, c=color, zorder=10, edgecolors='white',
                   linewidths=0.8, alpha=0.9)

        # 绘制 sigma 圈（若启用）
        mu_abs = abs(imu)
        if OBS_SIGMA_REF_MAS is not None:
            r_obs = OBS_SIGMA_REF_MAS * np.sqrt(mu_max / mu_abs) / 1000  # → arcsec
            circ = plt.Circle((ox, oy), r_obs, color=color, fill=False,
                               linestyle='--', linewidth=0.8, alpha=0.6, zorder=9)
            ax.add_patch(circ)
        if SOURCE_SIGMA_MAS is not None:
            r_src = SOURCE_SIGMA_MAS * np.sqrt(mu_abs) / 1000  # → arcsec
            circ2 = plt.Circle((ox, oy), r_src, color=color, fill=False,
                                linestyle=':', linewidth=0.8, alpha=0.5, zorder=9)
            ax.add_patch(circ2)

        sigma_note = ""
        if OBS_SIGMA_REF_MAS is not None:
            sigma_note += f"\nσ_obs={OBS_SIGMA_REF_MAS * np.sqrt(mu_max / mu_abs):.2f}mas"
        if SOURCE_SIGMA_MAS is not None:
            sigma_note += f"\nσ_src={SOURCE_SIGMA_MAS * np.sqrt(mu_abs):.2f}mas"
        ax.annotate(f"#{i+1}  |μ|={mu_abs:.2f}{sigma_note}",
                    (ox, oy), textcoords='offset points', xytext=(8, 8),
                    color='white', fontsize=7.5,
                    bbox=dict(boxstyle='round,pad=0.2', fc='#1e293b', alpha=0.7))

    # 绘制源位置（在源平面，用虚线框提示）
    sx_disp = source_x * x_sign + CENTER_OFFSET_X
    sy_disp = source_y + CENTER_OFFSET_Y
    ax.scatter(sx_disp, sy_disp, s=120, c='#a78bfa', zorder=8,
               marker='*', edgecolors='white', linewidths=0.8)
    ax.annotate('Source', (sx_disp, sy_disp), textcoords='offset points',
                xytext=(6, -14), color='#c4b5fd', fontsize=8)

    # 标注
    coord_note = "RA(E+) / Celestial" if x_flip else "Math (x+→right)"
    ax.set_xlabel(f"x  (arcsec, {coord_note})", color='#94a3b8', fontsize=10)
    ax.set_ylabel("y  (arcsec, N+)", color='#94a3b8', fontsize=10)
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    ax.set_title(f"Lensed Images  (n = {len(images)},  Σ|μ| = {sum(mus):.2f})",
                 color='white', fontsize=12, pad=10)

    legend_items = [
        mpatches.Patch(color='#38bdf8', label='Positive parity (+μ)'),
        mpatches.Patch(color='#f87171', label='Negative parity (−μ)'),
        plt.Line2D([0], [0], marker='x', color='#facc15', ms=8, lw=0, label='Lens center'),
        plt.Line2D([0], [0], marker='*', color='#a78bfa', ms=10, lw=0, label='Source'),
    ]
    if OBS_SIGMA_REF_MAS is not None:
        legend_items.append(plt.Line2D([0], [0], color='#94a3b8', ls='--',
                                       lw=1, label=f'σ_obs circle (ref={OBS_SIGMA_REF_MAS}mas)'))
    if SOURCE_SIGMA_MAS is not None:
        legend_items.append(plt.Line2D([0], [0], color='#94a3b8', ls=':',
                                       lw=1, label=f'σ_src circle (src={SOURCE_SIGMA_MAS}mas)'))
    ax.legend(handles=legend_items, loc='upper right', fontsize=8,
              framealpha=0.5, facecolor='#1e293b', edgecolor='#334155',
              labelcolor='#e2e8f0')

    ax.grid(True, color='#1e3a5f', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [图像] 已保存: {outfile}")


# ══════════════════════════════════════════════════════════════════
# §3  主流程
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='引力透镜正向成像计算')
    parser.add_argument('--dat', metavar='PATH', default=None,
                        help='从 bestfit.dat 加载透镜和源位置（覆盖脚本内参数）')
    parser.add_argument('--no-plot', action='store_true',
                        help='不生成图像')
    parser.add_argument('--flip', action='store_true', default=None,
                        help='强制开启天球坐标输入（x 取负）')
    parser.add_argument('--no-flip', action='store_true', default=None,
                        help='强制关闭坐标翻转')
    args = parser.parse_args()

    # 确定翻转设置
    x_flip = OBS_X_FLIP
    if args.flip:
        x_flip = True
    elif args.no_flip:
        x_flip = False

    # 加载透镜参数
    lenses = list(LENSES)
    source_x = SOURCE_X
    source_y = SOURCE_Y

    dat_path = args.dat or BESTFIT_DAT
    if dat_path:
        if not os.path.isfile(dat_path):
            print(f"[错误] 找不到文件: {dat_path}")
            sys.exit(1)
        lenses, source_x, source_y = load_bestfit(dat_path)
        print(f"[载入] {dat_path}  ({len(lenses)} 个透镜)")

    # 坐标转换（统一转为数学坐标传入 glafic）
    lenses_math, sx_math, sy_math = coord_convert(lenses, source_x, source_y, x_flip)

    # 运行 glafic
    print("[计算] 调用 glafic 求解成像...")
    images = run_glafic(lenses_math, sx_math, sy_math)

    # 输出结果
    print_results(images, source_x, source_y, lenses, x_flip)

    # 生成图像
    if SAVE_PLOT and not args.no_plot:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        outfile = os.path.join(script_dir, PLOT_FILE)
        plot_results(images, source_x, source_y, lenses, x_flip, outfile)


if __name__ == '__main__':
    main()
