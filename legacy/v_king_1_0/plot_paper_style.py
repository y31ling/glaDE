#!/usr/bin/env python3
"""
论文风格绘图函数 — King Profile GC 子晕模型
支持：
  plot_paper_style_king         单模型三联图（ΔPos / μ / 临界曲线）
  plot_paper_style_king_compare 比较三联图（baseline vs optimized）
  read_critical_curves          读取 glafic 临界曲线文件
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ══════════════════════════════════════════════════════════════════
# §0  辅助函数
# ══════════════════════════════════════════════════════════════════

def read_critical_curves(crit_file):
    """
    读取 glafic writecrit 输出的临界曲线文件。

    文件格式：每行 8 列
        crit_x1  crit_y1  caus_x1  caus_y1  crit_x2  crit_y2  caus_x2  caus_y2

    返回:
        crit_segments : list of [[x1,y1],[x2,y2]]  临界曲线线段
        caus_segments : list of [[x1,y1],[x2,y2]]  焦散线线段
    """
    data = np.loadtxt(crit_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    crit_segments = [[[row[0], row[1]], [row[4], row[5]]] for row in data]
    caus_segments = [[[row[2], row[3]], [row[6], row[7]]] for row in data]

    return crit_segments, caus_segments


def _format_mass(m):
    """将质量 [M_sun] 格式化为紧凑字符串"""
    if m < 1e6:
        return f"{m:.2e}M☉"
    elif m < 1e9:
        return f"{m/1e6:.2f}×10⁶M☉"
    elif m < 1e12:
        return f"{m/1e9:.2f}×10⁹M☉"
    else:
        return f"{m/1e12:.2f}×10¹²M☉"


def _draw_sigma_lines(ax, img_numbers, sigma_pos_mas, bar_w, show_2sigma):
    """在左图绘制 1σ / 2σ 横线"""
    if np.allclose(sigma_pos_mas, sigma_pos_mas[0]):
        ax.axhline(sigma_pos_mas[0], linestyle="--", linewidth=1.5,
                   color='royalblue', label="1σ", alpha=0.80)
        if show_2sigma:
            ax.axhline(2.0 * sigma_pos_mas[0], linestyle=":", linewidth=1.5,
                       color='crimson', label="2σ", alpha=0.75)
    else:
        for x, s in zip(img_numbers, sigma_pos_mas):
            ax.hlines(s, x - bar_w / 2, x + bar_w / 2,
                      linestyles="--", linewidth=2.0, colors='royalblue', alpha=0.80)
            if show_2sigma:
                ax.hlines(2.0 * s, x - bar_w / 2, x + bar_w / 2,
                          linestyles=":", linewidth=2.0, colors='crimson', alpha=0.75)
        ax.plot([], [], linestyle="--", linewidth=2.0,
                color='royalblue', label="1σ (per image)", alpha=0.80)
        if show_2sigma:
            ax.plot([], [], linestyle=":", linewidth=2.0,
                    color='crimson', label="2σ (per image)", alpha=0.75)


def _draw_critical_curves(ax, crit_segments, caus_segments):
    """在右图绘制临界曲线和焦散线"""
    for seg in crit_segments:
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                'b-', linewidth=1.2, alpha=0.65)
    ax.plot([], [], 'b-', linewidth=1.2, label="Critical curve")

    if caus_segments:
        for seg in caus_segments:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                    'g-', linewidth=0.8, alpha=0.50)
        ax.plot([], [], 'g-', linewidth=0.8, label="Caustics")


def _draw_king_subhalos(ax, king_params):
    """
    在右图标注 King GC 子晕位置。

    king_params: list of tuples，每个 tuple 为以下格式之一：
        (x, y, M, rc, c)          — 无 mass_label
        (x, y, M, rc, c, mass)    — 兼容旧格式，mass==M
    """
    if king_params is None or len(king_params) == 0:
        return

    for i, p in enumerate(king_params, start=1):
        if len(p) >= 5:
            x_sub, y_sub, M_sub, rc_sub, c_sub = p[:5]
        else:
            continue

        ax.scatter(x_sub, y_sub, marker='D', s=160,
                   color='darkorange', edgecolor='black', linewidth=1.8,
                   zorder=10, alpha=0.92)

        lbl = (f"K{i}\n"
               f"{_format_mass(M_sub)}\n"
               f"rc={rc_sub*1000:.1f}mas\n"
               f"c={c_sub:.2f}")
        ax.text(x_sub, y_sub - 0.028, lbl,
                va="top", ha="center", fontsize=7, fontweight='bold',
                color='darkorange',
                bbox=dict(boxstyle='round,pad=0.35',
                          facecolor='white', edgecolor='darkorange',
                          alpha=0.95))

    ax.scatter([], [], marker='D', s=160, color='darkorange',
               edgecolor='black', linewidth=1.8, label='King GC Sub-halo')


# ══════════════════════════════════════════════════════════════════
# §1  单模型三联图
# ══════════════════════════════════════════════════════════════════

def plot_paper_style_king(
    img_numbers,
    delta_pos_mas,
    sigma_pos_mas,
    mu_obs,
    mu_obs_err,
    mu_pred,
    mu_at_obs_pred,
    obs_positions_arcsec,
    pred_positions_arcsec,
    crit_segments,
    caus_segments=None,
    king_params=None,
    title_left="ΔPos vs Image Number",
    title_mid="Magnification (μ)",
    title_right="Positions & Critical Curves",
    suptitle="King GC Sub-halo Model",
    output_file="result_king.png",
    show_2sigma=False,
):
    """
    生成 King profile GC 子晕论文风格三联图。

    参数
    ────
    img_numbers           [4]      图像编号 (1–4)
    delta_pos_mas         [4]      |ΔPos| (mas)
    sigma_pos_mas         float|[4] 位置不确定度 (mas)
    mu_obs                [4]      观测放大率
    mu_obs_err            [4]      观测放大率误差
    mu_pred               [4]      模型预测放大率（在预测位置）
    mu_at_obs_pred        [4]      模型放大率（在观测位置插值）
    obs_positions_arcsec  [4,2]    观测像位置 (arcsec)
    pred_positions_arcsec [4,2]    预测像位置 (arcsec)
    crit_segments         list     临界曲线线段列表
    caus_segments         list|None 焦散线线段列表（可选）
    king_params           list|None King 子晕参数列表，每项 (x,y,M,rc,c) (可选)
    show_2sigma           bool     是否显示 2σ 横线
    output_file           str      输出文件路径
    """
    # 数组化
    img_numbers           = np.asarray(img_numbers)
    delta_pos_mas         = np.asarray(delta_pos_mas)
    mu_obs                = np.asarray(mu_obs)
    mu_obs_err            = np.asarray(mu_obs_err)
    mu_pred               = np.asarray(mu_pred)
    mu_at_obs_pred        = np.asarray(mu_at_obs_pred)
    obs_positions_arcsec  = np.asarray(obs_positions_arcsec)
    pred_positions_arcsec = np.asarray(pred_positions_arcsec)

    if isinstance(sigma_pos_mas, (int, float)):
        sigma_pos_mas = np.full(4, float(sigma_pos_mas))
    else:
        sigma_pos_mas = np.asarray(sigma_pos_mas)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')

    # ── 左图：ΔPos 柱状图 ───────────────────────────────────────
    ax = axes[0]
    bar_w = 0.6
    ax.bar(img_numbers, delta_pos_mas, width=bar_w,
           label="ΔPos (model vs observed)",
           color='lightcoral', edgecolor='black', linewidth=1.5)
    _draw_sigma_lines(ax, img_numbers, sigma_pos_mas, bar_w, show_2sigma)

    ax.set_xlabel("Image Number", fontsize=12, fontweight='bold')
    ax.set_ylabel("ΔPos [mas]",   fontsize=12, fontweight='bold')
    ax.set_xticks(img_numbers)
    ax.set_title(title_left, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
              frameon=True, fontsize=9,
              facecolor='white', edgecolor='gray', framealpha=0.9, ncol=2)

    # ── 中图：放大率柱状图 ──────────────────────────────────────
    ax = axes[1]
    idx = np.arange(len(img_numbers))
    w = 0.22
    ax.bar(idx - w, mu_obs, width=w, yerr=mu_obs_err, capsize=3,
           label="μ_obs",
           color='skyblue', edgecolor='black', linewidth=1.5)
    ax.bar(idx, mu_pred, width=w, hatch='//',
           label="μ_pred",
           color='lightgreen', edgecolor='black', linewidth=1.5)
    ax.errorbar(idx + w, mu_at_obs_pred, yerr=None, fmt='o',
                markersize=8, label="μ@obs_pred",
                color='red', linewidth=2)

    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(i)) for i in img_numbers])
    ax.set_xlabel("Image Number", fontsize=12, fontweight='bold')
    ax.set_ylabel("μ",            fontsize=12, fontweight='bold')
    ax.set_title(title_mid, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
              frameon=True, fontsize=9,
              facecolor='white', edgecolor='gray', framealpha=0.9, ncol=3)

    # ── 右图：像位置 + 临界曲线 ────────────────────────────────
    ax = axes[2]
    _draw_critical_curves(ax, crit_segments, caus_segments)

    ax.scatter(obs_positions_arcsec[:, 0], obs_positions_arcsec[:, 1],
               marker='*', s=200, color='gold', edgecolors='black',
               linewidths=1.5, label="Observed Pos", zorder=5)
    ax.scatter(pred_positions_arcsec[:, 0], pred_positions_arcsec[:, 1],
               marker='x', s=100, color='red', linewidths=2.5,
               label="Predicted Pos", zorder=4)

    for i, (xo, yo) in enumerate(obs_positions_arcsec, start=1):
        ax.text(xo + 0.01, yo + 0.01, f"{i}",
                va="bottom", ha="left", fontsize=11,
                fontweight='bold', color='darkblue')

    _draw_king_subhalos(ax, king_params)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Δx [arcsec]", fontsize=12, fontweight='bold')
    ax.set_ylabel("Δy [arcsec]", fontsize=12, fontweight='bold')
    ax.set_title(title_right, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
              frameon=True, fontsize=9,
              facecolor='white', edgecolor='gray', framealpha=0.95, ncol=2)

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.savefig(output_file, dpi=220, bbox_inches='tight')
    print(f"✓ 保存图片: {output_file}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# §2  比较三联图（baseline vs optimized）
# ══════════════════════════════════════════════════════════════════

def plot_paper_style_king_compare(
    img_numbers,
    delta_pos_mas_baseline,
    delta_pos_mas_optimized,
    sigma_pos_mas,
    mu_obs,
    mu_obs_err,
    mu_pred_baseline,
    mu_pred_optimized,
    obs_positions_arcsec,
    pred_positions_arcsec,
    crit_segments,
    caus_segments=None,
    king_params=None,
    title_left="Position Offset Comparison",
    title_mid="Magnification Comparison",
    title_right="Positions & Critical Curves",
    suptitle="Baseline vs King GC Sub-halos",
    output_file="result_king_compare.png",
    show_2sigma=False,
):
    """
    生成比较模式三联图（无子晕基准 vs King GC 子晕优化后）。

    参数
    ────
    delta_pos_mas_baseline   [4]  基准模型位置偏差 (mas)
    delta_pos_mas_optimized  [4]  优化后位置偏差 (mas)
    mu_pred_baseline         [4]  基准放大率
    mu_pred_optimized        [4]  优化后放大率
    其余参数同 plot_paper_style_king。
    """
    img_numbers              = np.asarray(img_numbers)
    delta_pos_mas_baseline   = np.asarray(delta_pos_mas_baseline)
    delta_pos_mas_optimized  = np.asarray(delta_pos_mas_optimized)
    mu_obs                   = np.asarray(mu_obs)
    mu_obs_err               = np.asarray(mu_obs_err)
    mu_pred_baseline         = np.asarray(mu_pred_baseline)
    mu_pred_optimized        = np.asarray(mu_pred_optimized)
    obs_positions_arcsec     = np.asarray(obs_positions_arcsec)
    pred_positions_arcsec    = np.asarray(pred_positions_arcsec)

    if isinstance(sigma_pos_mas, (int, float)):
        sigma_pos_mas = np.full(4, float(sigma_pos_mas))
    else:
        sigma_pos_mas = np.asarray(sigma_pos_mas)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')

    # ── 左图：位置偏差对比柱状图 ───────────────────────────────
    ax = axes[0]
    idx   = np.arange(len(img_numbers))
    bar_w = 0.35

    ax.bar(idx - bar_w / 2, delta_pos_mas_baseline, width=bar_w,
           label="Baseline (no GC)", color='lightgray',
           edgecolor='black', linewidth=1.5)
    ax.bar(idx + bar_w / 2, delta_pos_mas_optimized, width=bar_w,
           label="Optimized (King GC)", color='lightcoral',
           edgecolor='black', linewidth=1.5)

    if np.allclose(sigma_pos_mas, sigma_pos_mas[0]):
        ax.axhline(sigma_pos_mas[0], linestyle='--', linewidth=1.5,
                   color='royalblue', label="1σ", alpha=0.80)
        if show_2sigma:
            ax.axhline(2.0 * sigma_pos_mas[0], linestyle=':', linewidth=1.5,
                       color='crimson', label="2σ", alpha=0.75)
    else:
        full_w = bar_w * 2.2
        for xi, s in zip(idx, sigma_pos_mas):
            ax.hlines(s, xi - full_w / 2, xi + full_w / 2,
                      linestyles='--', linewidth=2.0, colors='royalblue', alpha=0.80)
            if show_2sigma:
                ax.hlines(2.0 * s, xi - full_w / 2, xi + full_w / 2,
                          linestyles=':', linewidth=2.0, colors='crimson', alpha=0.75)
        ax.plot([], [], linestyle='--', linewidth=2.0,
                color='royalblue', label="1σ", alpha=0.80)
        if show_2sigma:
            ax.plot([], [], linestyle=':', linewidth=2.0,
                    color='crimson', label="2σ", alpha=0.75)

    ax.set_xlabel("Image Number", fontsize=12, fontweight='bold')
    ax.set_ylabel("ΔPos [mas]",   fontsize=12, fontweight='bold')
    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(i)) for i in img_numbers])
    ax.set_title(title_left, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
              frameon=True, fontsize=8,
              facecolor='white', edgecolor='gray', framealpha=0.9, ncol=2)

    # ── 中图：三组放大率对比 ────────────────────────────────────
    ax = axes[1]
    w = 0.25
    ax.bar(idx - w, mu_obs, width=w, yerr=mu_obs_err, capsize=3,
           label="μ_obs",
           color='skyblue', edgecolor='black', linewidth=1.5)
    ax.bar(idx, mu_pred_baseline, width=w,
           label="μ_baseline",
           color='lightgray', edgecolor='black', linewidth=1.5, hatch='\\\\')
    ax.bar(idx + w, mu_pred_optimized, width=w,
           label="μ_optimized",
           color='lightgreen', edgecolor='black', linewidth=1.5, hatch='//')

    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(i)) for i in img_numbers])
    ax.set_xlabel("Image Number", fontsize=12, fontweight='bold')
    ax.set_ylabel("μ",            fontsize=12, fontweight='bold')
    ax.set_title(title_mid, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
              frameon=True, fontsize=8,
              facecolor='white', edgecolor='gray', framealpha=0.9, ncol=3)

    # ── 右图：像位置 + 临界曲线 ────────────────────────────────
    ax = axes[2]
    _draw_critical_curves(ax, crit_segments, caus_segments)

    ax.scatter(obs_positions_arcsec[:, 0], obs_positions_arcsec[:, 1],
               marker='*', s=200, color='gold', edgecolors='black',
               linewidths=1.5, label="Observed Pos", zorder=5)
    ax.scatter(pred_positions_arcsec[:, 0], pred_positions_arcsec[:, 1],
               marker='x', s=100, color='red', linewidths=2.5,
               label="Predicted Pos", zorder=4)

    for i, (xo, yo) in enumerate(obs_positions_arcsec, start=1):
        ax.text(xo + 0.01, yo + 0.01, f"{i}",
                va="bottom", ha="left", fontsize=11,
                fontweight='bold', color='darkblue')

    _draw_king_subhalos(ax, king_params)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Δx [arcsec]", fontsize=12, fontweight='bold')
    ax.set_ylabel("Δy [arcsec]", fontsize=12, fontweight='bold')
    ax.set_title(title_right, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
              frameon=True, fontsize=9,
              facecolor='white', edgecolor='gray', framealpha=0.95, ncol=2)

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.savefig(output_file, dpi=220, bbox_inches='tight')
    print(f"✓ 保存比较图: {output_file}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# §3  kappa 径向轮廓图（用于展示 rc-c 简并）
# ══════════════════════════════════════════════════════════════════

def plot_king_profiles(
    king_params_list,
    r_min=1e-3,
    r_max=1.0,
    n_r=100,
    labels=None,
    title="King Profile κ Radial Profiles",
    output_file="king_profiles.png",
):
    """
    绘制一个或多个 King profile 的 kappa 径向剖面图。

    参数
    ────
    king_params_list  list of (M, rc, c)  各子晕参数
    r_min, r_max      float               径向范围 [arcsec]
    n_r               int                 采样点数
    labels            list of str | None  图例标签
    """
    import sys
    import os
    sys.path.insert(0, '/home/luukiaun/glafic251018/glafic2/python')
    os.environ.setdefault(
        'LD_LIBRARY_PATH',
        '/home/luukiaun/glafic251018/gsl-2.8/.libs:'
        '/home/luukiaun/glafic251018/fftw-3.3.10/.libs:'
        '/home/luukiaun/glafic251018/cfitsio-4.6.2/.libs'
    )
    import glafic

    r_arr = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    colors = plt.cm.tab10(np.linspace(0, 1, len(king_params_list)))

    fig, ax = plt.subplots(figsize=(7, 5))

    for k, (M, rc, c) in enumerate(king_params_list):
        glafic.init(0.3, 0.7, -1.0, 0.7, 'tmp_profile',
                    -2., -2., 2., 2., 0.02, 0.2, 5, verb=0)
        glafic.startup_setnum(1, 0, 1)
        glafic.set_lens(1, 'king', 0.5, M, 0., 0., 0., 0., rc, c)
        glafic.set_point(1, 2.0, 0.01, 0.)
        glafic.model_init(verb=0)

        kap = np.array([glafic.calcimage(2.0, r, 0.)[3] for r in r_arr])
        kap = np.where(kap > 0, kap, np.nan)
        glafic.quit()

        lbl = labels[k] if labels is not None else f"M={_format_mass(M)} rc={rc*1000:.1f}mas c={c:.2f}"
        ax.loglog(r_arr, kap, color=colors[k], lw=2.0, label=lbl)

        # 标注 rc 和 rt
        ax.axvline(rc,         color=colors[k], ls=':', lw=0.9, alpha=0.6)
        ax.axvline(rc*10**c,   color=colors[k], ls='--', lw=0.7, alpha=0.4)

    ax.set_xlabel('r [arcsec]', fontsize=12, fontweight='bold')
    ax.set_ylabel('κ(r)',       fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, which='both', linestyle=':', linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=9, framealpha=0.9)

    # 图例说明
    from matplotlib.lines import Line2D
    extra = [
        Line2D([0], [0], color='gray', ls=':',  lw=1.2, label='rc (core)'),
        Line2D([0], [0], color='gray', ls='--', lw=0.9, label='rt = 10^c · rc (tidal)'),
    ]
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles=handles + extra,
              labels=lbls + [e.get_label() for e in extra],
              fontsize=8.5, framealpha=0.9, loc='lower left')

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✓ 保存 King profiles 图: {output_file}")
    plt.close()


# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("plot_paper_style.py — King GC Sub-halo 绘图模块")
    print("  plot_paper_style_king         : 单模型三联图")
    print("  plot_paper_style_king_compare : 比较三联图 (baseline vs optimized)")
    print("  plot_king_profiles            : King profile kappa 剖面图")
    print("  read_critical_curves          : 读取 glafic 临界曲线文件")
