#!/usr/bin/env python3
"""
论文风格的三联图绘制函数
Fig.6-style triptych for iPTF16geu
支持 Point Mass 和 NFW 模型
"""

import numpy as np
import matplotlib.pyplot as plt

def read_critical_curves(crit_file):
    """
    读取glafic的临界曲线文件
    返回: (crit_x, crit_y, caus_x, caus_y)
    """
    data = np.loadtxt(crit_file)
    
    # 临界曲线：列0,1和列4,5组成线段
    crit_segments = []
    for row in data:
        crit_segments.append([[row[0], row[1]], [row[4], row[5]]])
    
    # 焦散线：列2,3和列6,7组成线段
    caus_segments = []
    for row in data:
        caus_segments.append([[row[2], row[3]], [row[6], row[7]]])
    
    return crit_segments, caus_segments

def plot_paper_style(
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
    title_left="ΔPos vs Img Number",
    title_mid="Magnification (μ)",
    title_right="Positions & Critical Curves",
    suptitle="iPTF16geu: Double Sersic + SIE Model",
    output_file="fig6_reproduction.png",
    subhalo_positions=None,  # list of (x, y, mass) tuples for point mass
    show_2sigma=False  # 是否显示2σ横线
):
    """
    生成论文风格的三联图 (Point Mass 模型)
    
    参数:
    - img_numbers: [4] 图像编号
    - delta_pos_mas: [4] 位置偏差 (mas)
    - sigma_pos_mas: float or [4] 位置不确定度 (mas)
    - mu_obs: [4] 观测放大率
    - mu_obs_err: [4] 观测放大率误差
    - mu_pred: [4] 预测放大率（在预测位置）
    - mu_at_obs_pred: [4] 预测放大率（在观测位置）
    - obs_positions_arcsec: [4,2] 观测位置 (arcsec)
    - pred_positions_arcsec: [4,2] 预测位置 (arcsec)
    - crit_segments: list of [[x1,y1],[x2,y2]] 临界曲线线段
    - caus_segments: list of [[x1,y1],[x2,y2]] 焦散线线段 (可选)
    - subhalo_positions: list of (x, y, mass) tuples for sub-halos (可选)
    - show_2sigma: bool, 是否在左侧面板显示2σ横线 (默认False)
    """
    
    # 数据验证
    img_numbers = np.asarray(img_numbers)
    delta_pos_mas = np.asarray(delta_pos_mas)
    mu_obs = np.asarray(mu_obs)
    mu_obs_err = np.asarray(mu_obs_err)
    mu_pred = np.asarray(mu_pred)
    mu_at_obs_pred = np.asarray(mu_at_obs_pred)
    obs_positions_arcsec = np.asarray(obs_positions_arcsec)
    pred_positions_arcsec = np.asarray(pred_positions_arcsec)
    
    if isinstance(sigma_pos_mas, (int, float)):
        sigma_pos_mas = np.full(4, float(sigma_pos_mas))
    else:
        sigma_pos_mas = np.asarray(sigma_pos_mas)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    
    # ========== 左图: ΔPos vs Img Number ==========
    ax = axes[0]
    bar_w = 0.6
    ax.bar(img_numbers, delta_pos_mas, width=bar_w, 
           label="ΔPos (model vs observed)", color='lightcoral', 
           edgecolor='black', linewidth=1.5)
    
    # 1σ横线（每个图像的具体1σ值）
    if np.allclose(sigma_pos_mas, sigma_pos_mas[0]):
        # 所有像的1σ相同，画一条横线
        ax.axhline(sigma_pos_mas[0], linestyle="--", linewidth=1.5, 
                   color='blue', label="1σ", alpha=0.7)
        if show_2sigma:
            ax.axhline(2.0 * sigma_pos_mas[0], linestyle=":", linewidth=1.5, 
                      color='red', label="2σ", alpha=0.7)
    else:
        # 每个像的1σ不同，画短横线
        for x, s in zip(img_numbers, sigma_pos_mas):
            ax.hlines(s, x - bar_w/2, x + bar_w/2, 
                     linestyles="--", linewidth=2.0, colors='blue', alpha=0.7)
            if show_2sigma:
                ax.hlines(2.0 * s, x - bar_w/2, x + bar_w/2, 
                         linestyles=":", linewidth=2.0, colors='red', alpha=0.7)
        # 添加图例项
        ax.plot([], [], linestyle="--", linewidth=2.0, color='blue', 
               label="1σ (per image)", alpha=0.7)
        if show_2sigma:
            ax.plot([], [], linestyle=":", linewidth=2.0, color='red', 
                   label="2σ (per image)", alpha=0.7)
    
    ax.set_xlabel("Image Number", fontsize=12, fontweight='bold')
    ax.set_ylabel("ΔPos [mas]", fontsize=12, fontweight='bold')
    ax.set_xticks(img_numbers)
    ax.set_title(title_left, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=True, 
             fontsize=9, facecolor='white', edgecolor='gray', framealpha=0.9, ncol=2)
    
    # ========== 中图: μ (obs vs pred vs μ@obs pred) ==========
    ax = axes[1]
    idx = np.arange(len(img_numbers))
    
    w = 0.22
    ax.bar(idx - w, mu_obs, width=w, yerr=mu_obs_err, capsize=3, 
           label="μ_obs", color='skyblue', edgecolor='black', linewidth=1.5)
    ax.bar(idx, mu_pred, width=w, hatch="//", 
           label="μ_pred", color='lightgreen', edgecolor='black', linewidth=1.5)
    ax.errorbar(idx + w, mu_at_obs_pred, yerr=None, fmt="o", 
                markersize=8, label="μ@obs_pred", color='red', linewidth=2)
    
    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(i)) for i in img_numbers])
    ax.set_xlabel("Image Number", fontsize=12, fontweight='bold')
    ax.set_ylabel("μ", fontsize=12, fontweight='bold')
    ax.set_title(title_mid, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=True, 
             fontsize=9, facecolor='white', edgecolor='gray', framealpha=0.9, ncol=3)
    
    # ========== 右图: (Δx, Δy) with critical curves ==========
    ax = axes[2]
    
    # 绘制临界曲线
    for seg in crit_segments:
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], 
               'b-', linewidth=1.2, alpha=0.6)
    ax.plot([], [], 'b-', linewidth=1.2, label="Critical curve")
    
    # 可选：绘制焦散线
    if caus_segments is not None:
        for seg in caus_segments:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], 
                   'g-', linewidth=0.8, alpha=0.5)
        ax.plot([], [], 'g-', linewidth=0.8, label="Caustics")
    
    # 观测和预测位置
    ax.scatter(obs_positions_arcsec[:,0], obs_positions_arcsec[:,1], 
              marker="*", s=200, color='gold', edgecolors='black', 
              linewidths=1.5, label="Observed SN Pos", zorder=5)
    ax.scatter(pred_positions_arcsec[:,0], pred_positions_arcsec[:,1], 
              marker="x", s=100, color='red', linewidths=2.5, 
              label="Predicted SN Pos", zorder=4)
    
    # 标注图像编号
    for i, (xo, yo) in enumerate(obs_positions_arcsec, start=1):
        ax.text(xo + 0.01, yo + 0.01, f"{i}", va="bottom", ha="left", 
               fontsize=11, fontweight='bold', color='darkblue')
    
    # 标注sub-halo位置 (Point Mass)
    if subhalo_positions is not None:
        for i, (x_sub, y_sub, m_sub) in enumerate(subhalo_positions, start=1):
            # 绘制sub-halo位置
            ax.scatter(x_sub, y_sub, marker='D', s=150, 
                      color='red', edgecolor='black', linewidth=2, 
                      zorder=10, alpha=0.9)
            # 标注质量
            mass_label = f"S{i}: {m_sub:.1e}"
            ax.text(x_sub, y_sub - 0.025, mass_label, 
                   va="top", ha="center", fontsize=8, 
                   fontweight='bold', color='red',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='white', edgecolor='red', alpha=0.9))
        # 添加图例
        ax.scatter([], [], marker='D', s=150, color='red', 
                  edgecolor='black', linewidth=2, label='Sub-halo')
    
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Δx [arcsec]", fontsize=12, fontweight='bold')
    ax.set_ylabel("Δy [arcsec]", fontsize=12, fontweight='bold')
    ax.set_title(title_right, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=True, 
             fontsize=9, facecolor='white', edgecolor='gray', framealpha=0.9, ncol=2)
    
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.savefig(output_file, dpi=220, bbox_inches='tight')
    print(f"✓ 保存图片: {output_file}")
    plt.close()


def plot_paper_style_nfw(
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
    title_left="ΔPos vs Img Number",
    title_mid="Magnification (μ)",
    title_right="Positions & Critical Curves",
    suptitle="iPTF16geu: NFW Sub-halos Model",
    output_file="fig_nfw.png",
    nfw_params=None,  # list of (x, y, mass, concentration) tuples
    show_2sigma=False
):
    """
    生成论文风格的三联图 (NFW 模型)
    
    参数:
    - img_numbers: [4] 图像编号
    - delta_pos_mas: [4] 位置偏差 (mas)
    - sigma_pos_mas: float or [4] 位置不确定度 (mas)
    - mu_obs: [4] 观测放大率
    - mu_obs_err: [4] 观测放大率误差
    - mu_pred: [4] 预测放大率（在预测位置）
    - mu_at_obs_pred: [4] 预测放大率（在观测位置）
    - obs_positions_arcsec: [4,2] 观测位置 (arcsec)
    - pred_positions_arcsec: [4,2] 预测位置 (arcsec)
    - crit_segments: list of [[x1,y1],[x2,y2]] 临界曲线线段
    - caus_segments: list of [[x1,y1],[x2,y2]] 焦散线线段 (可选)
    - nfw_params: list of (x, y, mass, concentration) tuples for NFW sub-halos
    - show_2sigma: bool, 是否在左侧面板显示2σ横线
    """
    
    # 数据验证
    img_numbers = np.asarray(img_numbers)
    delta_pos_mas = np.asarray(delta_pos_mas)
    mu_obs = np.asarray(mu_obs)
    mu_obs_err = np.asarray(mu_obs_err)
    mu_pred = np.asarray(mu_pred)
    mu_at_obs_pred = np.asarray(mu_at_obs_pred)
    obs_positions_arcsec = np.asarray(obs_positions_arcsec)
    pred_positions_arcsec = np.asarray(pred_positions_arcsec)
    
    if isinstance(sigma_pos_mas, (int, float)):
        sigma_pos_mas = np.full(4, float(sigma_pos_mas))
    else:
        sigma_pos_mas = np.asarray(sigma_pos_mas)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    
    # ========== 左图: ΔPos vs Img Number ==========
    ax = axes[0]
    bar_w = 0.6
    ax.bar(img_numbers, delta_pos_mas, width=bar_w, 
           label="ΔPos (model vs observed)", color='lightcoral', 
           edgecolor='black', linewidth=1.5)
    
    # 1σ横线
    if np.allclose(sigma_pos_mas, sigma_pos_mas[0]):
        ax.axhline(sigma_pos_mas[0], linestyle="--", linewidth=1.5, 
                   color='blue', label="1σ", alpha=0.7)
        if show_2sigma:
            ax.axhline(2.0 * sigma_pos_mas[0], linestyle=":", linewidth=1.5, 
                      color='red', label="2σ", alpha=0.7)
    else:
        for x, s in zip(img_numbers, sigma_pos_mas):
            ax.hlines(s, x - bar_w/2, x + bar_w/2, 
                     linestyles="--", linewidth=2.0, colors='blue', alpha=0.7)
            if show_2sigma:
                ax.hlines(2.0 * s, x - bar_w/2, x + bar_w/2, 
                         linestyles=":", linewidth=2.0, colors='red', alpha=0.7)
        ax.plot([], [], linestyle="--", linewidth=2.0, color='blue', 
               label="1σ (per image)", alpha=0.7)
        if show_2sigma:
            ax.plot([], [], linestyle=":", linewidth=2.0, color='red', 
                   label="2σ (per image)", alpha=0.7)
    
    ax.set_xlabel("Image Number", fontsize=12, fontweight='bold')
    ax.set_ylabel("ΔPos [mas]", fontsize=12, fontweight='bold')
    ax.set_xticks(img_numbers)
    ax.set_title(title_left, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=True, 
             fontsize=9, facecolor='white', edgecolor='gray', framealpha=0.9, ncol=2)
    
    # ========== 中图: μ (obs vs pred vs μ@obs pred) ==========
    ax = axes[1]
    idx = np.arange(len(img_numbers))
    
    w = 0.22
    ax.bar(idx - w, mu_obs, width=w, yerr=mu_obs_err, capsize=3, 
           label="μ_obs", color='skyblue', edgecolor='black', linewidth=1.5)
    ax.bar(idx, mu_pred, width=w, hatch="//", 
           label="μ_pred", color='lightgreen', edgecolor='black', linewidth=1.5)
    ax.errorbar(idx + w, mu_at_obs_pred, yerr=None, fmt="o", 
                markersize=8, label="μ@obs_pred", color='red', linewidth=2)
    
    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(i)) for i in img_numbers])
    ax.set_xlabel("Image Number", fontsize=12, fontweight='bold')
    ax.set_ylabel("μ", fontsize=12, fontweight='bold')
    ax.set_title(title_mid, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=True, 
             fontsize=9, facecolor='white', edgecolor='gray', framealpha=0.9, ncol=3)
    
    # ========== 右图: (Δx, Δy) with critical curves ==========
    ax = axes[2]
    
    # 绘制临界曲线
    for seg in crit_segments:
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], 
               'b-', linewidth=1.2, alpha=0.6)
    ax.plot([], [], 'b-', linewidth=1.2, label="Critical curve")
    
    # 可选：绘制焦散线
    if caus_segments is not None:
        for seg in caus_segments:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], 
                   'g-', linewidth=0.8, alpha=0.5)
        ax.plot([], [], 'g-', linewidth=0.8, label="Caustics")
    
    # 观测和预测位置
    ax.scatter(obs_positions_arcsec[:,0], obs_positions_arcsec[:,1], 
              marker="*", s=200, color='gold', edgecolors='black', 
              linewidths=1.5, label="Observed SN Pos", zorder=5)
    ax.scatter(pred_positions_arcsec[:,0], pred_positions_arcsec[:,1], 
              marker="x", s=100, color='red', linewidths=2.5, 
              label="Predicted SN Pos", zorder=4)
    
    # 标注图像编号
    for i, (xo, yo) in enumerate(obs_positions_arcsec, start=1):
        ax.text(xo + 0.01, yo + 0.01, f"{i}", va="bottom", ha="left", 
               fontsize=11, fontweight='bold', color='darkblue')
    
    # 标注sub-halo位置（支持NFW或Jaffe）
    if nfw_params is not None:
        for i, params in enumerate(nfw_params, start=1):
            if len(params) == 4:
                # NFW: (x, y, m, c)
                x_sub, y_sub, param1, param2 = params
                sub_label = f"N{i}:\n{param1:.1e}\nc={param2:.1f}"
            elif len(params) == 5:
                # Jaffe: (x, y, sig, a, rco) - 无质量信息
                x_sub, y_sub, sig, a, rco = params
                sub_label = f"J{i}:\nσ={sig:.1f}\na={a*1000:.1f}mas"
            elif len(params) == 6:
                # Jaffe with mass: (x, y, sig, a, rco, mass)
                x_sub, y_sub, sig, a, rco, mass = params
                # 格式化质量显示
                if mass < 1e6:
                    mass_str = f"{mass:.1e}"
                elif mass < 1e9:
                    mass_str = f"{mass/1e6:.1f}e6"
                elif mass < 1e12:
                    mass_str = f"{mass/1e9:.2f}e9"
                else:
                    mass_str = f"{mass/1e12:.2f}e12"
                sub_label = f"J{i}:\nσ={sig:.1f}km/s\nM={mass_str}M☉"
            else:
                continue
            
            # 绘制sub-halo位置
            ax.scatter(x_sub, y_sub, marker='D', s=150, 
                      color='purple', edgecolor='black', linewidth=2, 
                      zorder=10, alpha=0.9)
            # 标注参数
            ax.text(x_sub, y_sub - 0.028, sub_label, 
                   va="top", ha="center", fontsize=7, 
                   fontweight='bold', color='purple',
                   bbox=dict(boxstyle='round,pad=0.35', 
                            facecolor='white', edgecolor='purple', alpha=0.95))
        # 添加图例
        ax.scatter([], [], marker='D', s=150, color='purple', 
                  edgecolor='black', linewidth=2, label='Jaffe Sub-halo')
    
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Δx [arcsec]", fontsize=12, fontweight='bold')
    ax.set_ylabel("Δy [arcsec]", fontsize=12, fontweight='bold')
    ax.set_title(title_right, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=True, 
             fontsize=9, facecolor='white', edgecolor='gray', framealpha=0.95, ncol=2)
    
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.savefig(output_file, dpi=220, bbox_inches='tight')
    print(f"✓ 保存图片: {output_file}")
    plt.close()


def plot_paper_style_nfw_compare(
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
    title_left="Position Offset Comparison",
    title_mid="Magnification Comparison",
    title_right="Positions & Critical Curves",
    suptitle="iPTF16geu: Baseline vs Optimized Comparison",
    output_file="fig_compare.png",
    subhalo_positions=None,
    show_2sigma=False
):
    """
    生成比较模式的三联图（baseline vs optimized）
    
    左图：位置偏移对比（baseline vs optimized）
    中图：放大率对比（observed, baseline, optimized三组）
    右图：位置和临界曲线
    """
    
    img_numbers = np.asarray(img_numbers)
    delta_pos_mas_baseline = np.asarray(delta_pos_mas_baseline)
    delta_pos_mas_optimized = np.asarray(delta_pos_mas_optimized)
    mu_obs = np.asarray(mu_obs)
    mu_obs_err = np.asarray(mu_obs_err)
    mu_pred_baseline = np.asarray(mu_pred_baseline)
    mu_pred_optimized = np.asarray(mu_pred_optimized)
    obs_positions_arcsec = np.asarray(obs_positions_arcsec)
    pred_positions_arcsec = np.asarray(pred_positions_arcsec)
    
    if isinstance(sigma_pos_mas, (int, float)):
        sigma_pos_mas = np.full(4, float(sigma_pos_mas))
    else:
        sigma_pos_mas = np.asarray(sigma_pos_mas)
    
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    
    # ========== 左图: 位置偏移对比 ==========
    ax = axes[0]
    idx = np.arange(len(img_numbers))
    bar_w = 0.35
    
    ax.bar(idx - bar_w/2, delta_pos_mas_baseline, width=bar_w, 
           label="Baseline (no sub-halo)", color='lightgray', 
           edgecolor='black', linewidth=1.5)
    ax.bar(idx + bar_w/2, delta_pos_mas_optimized, width=bar_w, 
           label="Optimized (with sub-halo)", color='lightcoral', 
           edgecolor='black', linewidth=1.5)
    
    # 1σ和2σ横线
    if np.allclose(sigma_pos_mas, sigma_pos_mas[0]):
        ax.axhline(sigma_pos_mas[0], linestyle="--", linewidth=1.5, 
                   color='blue', label="1σ", alpha=0.7)
        if show_2sigma:
            ax.axhline(2.0 * sigma_pos_mas[0], linestyle=":", linewidth=1.5, 
                      color='red', label="2σ", alpha=0.7)
    else:
        for x, s in zip(idx, sigma_pos_mas):
            ax.hlines(s, x - bar_w, x + bar_w, 
                     linestyles="--", linewidth=2.0, colors='blue', alpha=0.7)
            if show_2sigma:
                ax.hlines(2.0 * s, x - bar_w, x + bar_w, 
                         linestyles=":", linewidth=2.0, colors='red', alpha=0.7)
        ax.plot([], [], linestyle="--", linewidth=2.0, color='blue', 
               label="1σ", alpha=0.7)
        if show_2sigma:
            ax.plot([], [], linestyle=":", linewidth=2.0, color='red', 
                   label="2σ", alpha=0.7)
    
    ax.set_xlabel("Image Number", fontsize=12, fontweight='bold')
    ax.set_ylabel("ΔPos [mas]", fontsize=12, fontweight='bold')
    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(i)) for i in img_numbers])
    ax.set_title(title_left, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=True, 
             fontsize=8, facecolor='white', edgecolor='gray', framealpha=0.9, ncol=2)
    
    # ========== 中图: 放大率对比（3组柱状图）==========
    ax = axes[1]
    w = 0.25
    
    ax.bar(idx - w, mu_obs, width=w, yerr=mu_obs_err, capsize=3, 
           label="μ_obs", color='skyblue', edgecolor='black', linewidth=1.5)
    ax.bar(idx, mu_pred_baseline, width=w, 
           label="μ_baseline", color='lightgray', edgecolor='black', linewidth=1.5, hatch='\\\\')
    ax.bar(idx + w, mu_pred_optimized, width=w, 
           label="μ_optimized", color='lightgreen', edgecolor='black', linewidth=1.5, hatch='//')
    
    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(i)) for i in img_numbers])
    ax.set_xlabel("Image Number", fontsize=12, fontweight='bold')
    ax.set_ylabel("μ", fontsize=12, fontweight='bold')
    ax.set_title(title_mid, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=True, 
             fontsize=8, facecolor='white', edgecolor='gray', framealpha=0.9, ncol=3)
    
    # ========== 右图: 位置和临界曲线 ==========
    ax = axes[2]
    
    for seg in crit_segments:
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], 
               'b-', linewidth=1.2, alpha=0.6)
    ax.plot([], [], 'b-', linewidth=1.2, label="Critical curve")
    
    if caus_segments is not None:
        for seg in caus_segments:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], 
                   'g-', linewidth=0.8, alpha=0.5)
        ax.plot([], [], 'g-', linewidth=0.8, label="Caustics")
    
    ax.scatter(obs_positions_arcsec[:,0], obs_positions_arcsec[:,1], 
              marker="*", s=200, color='gold', edgecolors='black', 
              linewidths=1.5, label="Observed SN Pos", zorder=5)
    ax.scatter(pred_positions_arcsec[:,0], pred_positions_arcsec[:,1], 
              marker="x", s=100, color='red', linewidths=2.5, 
              label="Predicted SN Pos", zorder=4)
    
    for i, (xo, yo) in enumerate(obs_positions_arcsec, start=1):
        ax.text(xo + 0.01, yo + 0.01, f"{i}", va="bottom", ha="left", 
               fontsize=11, fontweight='bold', color='darkblue')
    
    # 标注sub-halo位置
    if subhalo_positions is not None:
        for i, params in enumerate(subhalo_positions, start=1):
            if len(params) == 4:
                x_sub, y_sub, param1, param2 = params
                sub_label = f"N{i}:\n{param1:.1e}\nc={param2:.1f}"
            elif len(params) == 5:
                x_sub, y_sub, sig, a, rco = params
                sub_label = f"J{i}:\nσ={sig:.1f}\na={a*1000:.1f}mas"
            elif len(params) == 6:
                # Jaffe with mass: (x, y, sig, a, rco, mass)
                x_sub, y_sub, sig, a, rco, mass = params
                # 格式化质量显示
                if mass < 1e6:
                    mass_str = f"{mass:.1e}"
                elif mass < 1e9:
                    mass_str = f"{mass/1e6:.1f}e6"
                elif mass < 1e12:
                    mass_str = f"{mass/1e9:.2f}e9"
                else:
                    mass_str = f"{mass/1e12:.2f}e12"
                sub_label = f"J{i}:\nσ={sig:.1f}km/s\nM={mass_str}M☉"
            elif len(params) == 3:
                x_sub, y_sub, m = params
                sub_label = f"S{i}: {m:.1e}"
            else:
                continue
            
            ax.scatter(x_sub, y_sub, marker='D', s=150, 
                      color='purple', edgecolor='black', linewidth=2, 
                      zorder=10, alpha=0.9)
            ax.text(x_sub, y_sub - 0.028, sub_label, 
                   va="top", ha="center", fontsize=7, 
                   fontweight='bold', color='purple',
                   bbox=dict(boxstyle='round,pad=0.35', 
                            facecolor='white', edgecolor='purple', alpha=0.95))
        ax.scatter([], [], marker='D', s=150, color='purple', 
                  edgecolor='black', linewidth=2, label='Jaffe Sub-halo')
    
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Δx [arcsec]", fontsize=12, fontweight='bold')
    ax.set_ylabel("Δy [arcsec]", fontsize=12, fontweight='bold')
    ax.set_title(title_right, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=True, 
             fontsize=8, facecolor='white', edgecolor='gray', framealpha=0.95, ncol=2)
    
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.savefig(output_file, dpi=220, bbox_inches='tight')
    print(f"✓ 保存比较图: {output_file}")
    plt.close()


if __name__ == "__main__":
    print("这是一个绘图函数模块，请从主程序中导入使用")
    print("支持 Point Mass 模型 (plot_paper_style) 和 NFW/Jaffe 模型 (plot_paper_style_nfw)")
    print("支持比较模式 (plot_paper_style_nfw_compare)")
