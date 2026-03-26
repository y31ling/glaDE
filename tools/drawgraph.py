#!/usr/bin/env python3
"""
DrawGraph Tool: 从已有数据重新生成结果图
根据 *_best_params.txt 文件自动识别模型类型并生成对应的三联图

用法:
    python drawgraph.py <folder_name> [--compare]
    
参数:
    folder_name: 数据文件夹名称（如 260120_0348）
    --compare: 生成比较图（baseline vs optimized）

示例:
    python drawgraph.py 260120_0348
    python drawgraph.py 260120_0348 --compare
"""

import sys
import os
import re
import glob
import argparse

# 添加 glade 运行时环境
GLADE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GLADE_ROOT)
from runtime_env import setup_runtime_env  # noqa: E402
setup_runtime_env(GLADE_ROOT)
sys.path.insert(0, os.path.join(GLADE_ROOT, 'glafic2', 'python'))

import glafic
import numpy as np

# 导入绘图函数
# 使用 importlib 来分别从不同目录导入
import importlib.util

def load_module_from_path(module_name, file_path):
    """从指定路径加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 从 v_nfw_2.0 导入 NFW 相关函数
nfw_plot_module = load_module_from_path(
    'plot_paper_style_nfw', 
    os.path.join(GLADE_ROOT, 'legacy', 'v_nfw_2_0', 'plot_paper_style.py')
)
plot_paper_style = nfw_plot_module.plot_paper_style
plot_paper_style_nfw = nfw_plot_module.plot_paper_style_nfw
plot_paper_style_nfw_compare = nfw_plot_module.plot_paper_style_nfw_compare
read_critical_curves = nfw_plot_module.read_critical_curves

# 从 v_pointmass_1.0 导入 point mass 比较模式
pm_plot_module = load_module_from_path(
    'plot_paper_style_pm',
    os.path.join(GLADE_ROOT, 'legacy', 'v_pointmass_1_0', 'plot_paper_style.py')
)
plot_paper_style_compare = pm_plot_module.plot_paper_style_compare

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                         固定常量                                       ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# 基础目录（优先 glade 项目根）
GLAFIC_BASE = os.path.abspath(os.path.join(GLADE_ROOT, '..', '..'))

# 观测数据
obs_positions_mas = np.array([
    [-266.035, +0.427],
    [+118.835, -221.927],
    [+238.324, +227.270],
    [-126.157, +319.719],
])

obs_positions = np.zeros_like(obs_positions_mas)
obs_positions[:, 0] = -obs_positions_mas[:, 0] / 1000.0
obs_positions[:, 1] = obs_positions_mas[:, 1] / 1000.0

obs_magnifications = np.array([-35.6, 15.7, -7.5, 9.1])
obs_mag_errors = np.array([2.1, 1.3, 1.0, 1.1])
obs_pos_sigma_mas = np.array([0.41, 0.86, 2.23, 3.11])

# 中心偏移量（透镜中心与图像坐标系原点的偏移）
center_offset_x = -0.01535000   # [arcsec]
center_offset_y = +0.03220000   # [arcsec]

# 模型参数
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
source_x = 2.685497e-03
source_y = 2.443616e-02
lens_z = 0.2160

lens_params = {
    'sers1': (1, 'sers', 0.2160, 9.896617e+09, 2.656977e-03, 2.758473e-02,
              2.986760e-01, 1.124730e+02, 3.939718e-01, 1.057760e+00),
    'sers2': (2, 'sers', 0.2160, 2.555580e+10, 2.656977e-03, 2.758473e-02,
              4.242340e-01, 5.396370e+01, 1.538855e+00, 1.000000e+00),
    'sie': (3, 'sie', 0.2160, 1.183382e+02, 2.656977e-03, 2.758473e-02,
            1.571203e-01, 2.920348e+01, 0.0, 0.0)
}


def find_folder(folder_name):
    """在 glafic251018 目录中查找指定文件夹"""
    # 直接路径
    direct_path = os.path.join(GLAFIC_BASE, folder_name)
    if os.path.isdir(direct_path):
        return direct_path
    
    # 在 work 目录下查找
    work_path = os.path.join(GLAFIC_BASE, 'work', folder_name)
    if os.path.isdir(work_path):
        return work_path
    
    # 递归查找
    for root, dirs, files in os.walk(GLAFIC_BASE):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    
    return None


def detect_model_type(params_file):
    """检测模型类型"""
    with open(params_file, 'r') as f:
        content = f.read()
    
    if 'Version NFW' in content or 'v_nfw' in content:
        return 'nfw'
    elif 'Pseudo-Jaffe' in content or 'v_p_jaffe' in content or 'jaffe' in content.lower():
        return 'p_jaffe'
    elif 'Point Mass' in content or 'v_pm' in content or 'pointmass' in content.lower():
        return 'pointmass'
    else:
        # 尝试从文件名推断
        basename = os.path.basename(params_file)
        if 'nfw' in basename.lower():
            return 'nfw'
        elif 'jaffe' in basename.lower():
            return 'p_jaffe'
        elif 'pm' in basename.lower() or 'point' in basename.lower():
            return 'pointmass'
    
    return None


def parse_nfw_params(params_file):
    """解析 NFW 模型参数 (支持科学计数法)"""
    subhalos = []
    
    with open(params_file, 'r') as f:
        content = f.read()
    
    # 查找所有 NFW sub-halo 参数
    pattern = r'x_nfw(\d+)\s*=\s*([-\d.eE+]+)'
    matches = re.findall(pattern, content)
    
    for img_idx, x_val in matches:
        x_pattern = rf'x_nfw{img_idx}\s*=\s*([-\d.eE+]+)'
        y_pattern = rf'y_nfw{img_idx}\s*=\s*([-\d.eE+]+)'
        m_pattern = rf'm_vir{img_idx}\s*=\s*([-\d.eE+]+)'
        c_pattern = rf'c_vir{img_idx}\s*=\s*([-\d.eE+]+)'
        
        x_match = re.search(x_pattern, content)
        y_match = re.search(y_pattern, content)
        m_match = re.search(m_pattern, content)
        c_match = re.search(c_pattern, content)
        
        if x_match and y_match and m_match and c_match:
            x = float(x_match.group(1))
            y = float(y_match.group(1))
            m = float(m_match.group(1))
            c = float(c_match.group(1))
            subhalos.append((x, y, m, c))
    
    return subhalos


def parse_pointmass_params(params_file):
    """解析 Point Mass 模型参数 (支持科学计数法)"""
    subhalos = []
    
    with open(params_file, 'r') as f:
        content = f.read()
    
    # 查找所有 sub-halo 参数
    pattern = r'x_sub(\d+)\s*=\s*([-\d.eE+]+)'
    matches = re.findall(pattern, content)
    
    for img_idx, x_val in matches:
        x_pattern = rf'x_sub{img_idx}\s*=\s*([-\d.eE+]+)'
        y_pattern = rf'y_sub{img_idx}\s*=\s*([-\d.eE+]+)'
        m_pattern = rf'mass_sub{img_idx}\s*=\s*([-\d.eE+]+)'
        
        x_match = re.search(x_pattern, content)
        y_match = re.search(y_pattern, content)
        m_match = re.search(m_pattern, content)
        
        if x_match and y_match and m_match:
            x = float(x_match.group(1))
            y = float(y_match.group(1))
            m = float(m_match.group(1))
            subhalos.append((x, y, m))
    
    return subhalos


def parse_jaffe_params(params_file):
    """解析 Pseudo-Jaffe 模型参数"""
    subhalos = []
    
    with open(params_file, 'r') as f:
        content = f.read()
    
    # 查找所有 Jaffe sub-halo 参数 (支持科学计数法)
    pattern = r'x_jaffe(\d+)\s*=\s*([-\d.eE+]+)'
    matches = re.findall(pattern, content)
    
    for img_idx, x_val in matches:
        x_pattern = rf'x_jaffe{img_idx}\s*=\s*([-\d.eE+]+)'
        y_pattern = rf'y_jaffe{img_idx}\s*=\s*([-\d.eE+]+)'
        sig_pattern = rf'sig{img_idx}\s*=\s*([-\d.eE+]+)'
        # 使用 (?<![/]) 负向后瞻，避免匹配 rco/a{img_idx}
        a_pattern = rf'(?<![/])a{img_idx}\s*=\s*([-\d.eE+]+)'
        # 使用 (?<![/]) 负向后瞻，避免匹配 rco/a{img_idx}
        rco_pattern = rf'(?<![/])rco{img_idx}\s*=\s*([-\d.eE+]+)'
        
        x_match = re.search(x_pattern, content)
        y_match = re.search(y_pattern, content)
        sig_match = re.search(sig_pattern, content)
        a_match = re.search(a_pattern, content)
        rco_match = re.search(rco_pattern, content)
        
        if x_match and y_match and sig_match and a_match and rco_match:
            x = float(x_match.group(1))
            y = float(y_match.group(1))
            sig = float(sig_match.group(1))
            a = float(a_match.group(1))
            rco = float(rco_match.group(1))
            subhalos.append((x, y, sig, a, rco))
    
    return subhalos


def compute_model_predictions(model_type, subhalos, src_x=None, src_y=None, 
                               lens_params_dict=None, temp_prefix='temp_redraw'):
    """计算模型预测的位置和放大率"""
    if src_x is None:
        src_x = source_x
    if src_y is None:
        src_y = source_y
    if lens_params_dict is None:
        lens_params_dict = lens_params
    
    n_subhalos = len(subhalos)
    
    glafic.init(omega, lambda_cosmo, weos, hubble, temp_prefix,
                xmin, ymin, xmax, ymax, pix_ext, pix_poi, maxlev, verb=0)
    
    glafic.startup_setnum(3 + n_subhalos, 0, 1)
    glafic.set_lens(*lens_params_dict['sers1'])
    glafic.set_lens(*lens_params_dict['sers2'])
    glafic.set_lens(*lens_params_dict['sie'])
    
    # 设置 sub-halos
    for i, params in enumerate(subhalos):
        if model_type == 'nfw':
            x, y, m, c = params
            glafic.set_lens(4 + i, 'gnfw', lens_z, m, x, y, 0.0, 0.0, c, 1.0)
        elif model_type == 'pointmass':
            x, y, m = params
            glafic.set_lens(4 + i, 'point', lens_z, m, x, y, 0.0, 0.0, 0.0, 0.0)
        elif model_type == 'p_jaffe':
            x, y, sig, a, rco = params
            glafic.set_lens(4 + i, 'jaffe', lens_z, sig, x, y, 0.0, 0.0, a, rco)
    
    glafic.set_point(1, source_z, src_x, src_y)
    glafic.model_init(verb=0)
    
    # 计算预测位置和放大率
    # findimg 会将结果写入文件
    glafic.findimg(source_z, src_x, src_y)
    
    # 从输出文件读取结果
    point_file = f'{temp_prefix}_point.dat'
    pred_positions = []
    pred_magnifications = []
    
    try:
        with open(point_file, 'r') as f:
            lines = f.readlines()
        
        # 解析点源文件
        pred_data = []
        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                x = float(parts[0])
                y = float(parts[1])
                mag = float(parts[2])
                pred_data.append([x, y, mag])
        
        if len(pred_data) >= 4:
            pred_arr = np.array([[d[0], d[1]] for d in pred_data])
            pred_mag_arr = np.array([d[2] for d in pred_data])
            
            # 使用最近邻匹配
            from scipy.optimize import linear_sum_assignment
            from scipy.spatial.distance import cdist
            
            dist_matrix = cdist(obs_positions, pred_arr)
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            
            for i in range(4):
                matched_idx = col_ind[i]
                pred_positions.append([pred_arr[matched_idx, 0], pred_arr[matched_idx, 1]])
                pred_magnifications.append(pred_mag_arr[matched_idx])
        else:
            raise ValueError("Not enough predicted images")
    except Exception as e:
        print(f"警告: 无法解析预测结果文件 ({e})，使用默认值")
        for obs_pos in obs_positions:
            pred_positions.append([obs_pos[0], obs_pos[1]])
            pred_magnifications.append(1.0)
    
    # 生成临界曲线
    glafic.writecrit(source_z)
    crit_file = f'{temp_prefix}_crit.dat'
    
    glafic.quit()
    
    pred_positions = np.array(pred_positions)
    pred_magnifications = np.array(pred_magnifications)
    
    # 应用中心偏移量
    pred_positions[:, 0] += center_offset_x
    pred_positions[:, 1] += center_offset_y
    
    delta_pos_mas = np.linalg.norm(pred_positions - obs_positions, axis=1) * 1000
    
    return pred_positions, pred_magnifications, delta_pos_mas, crit_file


def main():
    parser = argparse.ArgumentParser(description='从已有数据重新生成结果图')
    parser.add_argument('folder', type=str, help='数据文件夹名称（如 260120_0348）')
    parser.add_argument('--compare', action='store_true', help='生成比较图（baseline vs optimized）')
    parser.add_argument('--show-2sigma', action='store_true', help='显示2σ横线')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出文件名')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DrawGraph Tool: 从已有数据重新生成结果图")
    print("=" * 70)
    
    # 查找文件夹
    folder_path = find_folder(args.folder)
    if folder_path is None:
        print(f"错误: 找不到文件夹 '{args.folder}'")
        sys.exit(1)
    
    print(f"找到文件夹: {folder_path}")
    
    # 查找 *_best_params.txt 文件
    params_files = glob.glob(os.path.join(folder_path, '*_best_params.txt'))
    if not params_files:
        print(f"错误: 在 {folder_path} 中找不到 *_best_params.txt 文件")
        sys.exit(1)
    
    params_file = params_files[0]
    print(f"参数文件: {os.path.basename(params_file)}")
    
    # 检测模型类型
    model_type = detect_model_type(params_file)
    if model_type is None:
        print(f"错误: 无法检测模型类型")
        sys.exit(1)
    
    print(f"模型类型: {model_type}")
    
    # 解析参数
    if model_type == 'nfw':
        subhalos = parse_nfw_params(params_file)
    elif model_type == 'pointmass':
        subhalos = parse_pointmass_params(params_file)
    elif model_type == 'p_jaffe':
        subhalos = parse_jaffe_params(params_file)
    
    n_subhalos = len(subhalos)
    print(f"Sub-halos 数量: {n_subhalos}")
    
    for i, params in enumerate(subhalos):
        if model_type == 'nfw':
            x, y, m, c = params
            print(f"  Sub-halo {i+1}: x={x:.4f}, y={y:.4f}, M={m:.2e}, c={c:.2f}")
        elif model_type == 'pointmass':
            x, y, m = params
            print(f"  Sub-halo {i+1}: x={x:.4f}, y={y:.4f}, M={m:.2e}")
        elif model_type == 'p_jaffe':
            x, y, sig, a, rco = params
            print(f"  Sub-halo {i+1}: x={x:.4f}, y={y:.4f}, σ={sig:.2f}, a={a*1000:.2f}mas")
    
    # 计算预测结果
    print("\n计算模型预测...")
    pred_pos, pred_mag, delta_pos_mas, crit_file = compute_model_predictions(
        model_type, subhalos)
    
    print(f"位置偏移 (mas): {delta_pos_mas}")
    print(f"预测放大率: {pred_mag}")
    
    # 读取临界曲线
    crit_segments, caus_segments = read_critical_curves(crit_file)
    
    # 确定输出文件名
    if args.output:
        output_file = args.output
    else:
        prefix = os.path.splitext(os.path.basename(params_file))[0].replace('_best_params', '')
        suffix = '_compare' if args.compare else ''
        output_file = os.path.join(folder_path, f"result_{prefix}{suffix}_redraw.png")
    
    # 生成图表
    print(f"\n生成图表: {output_file}")
    
    if args.compare:
        # 比较模式：计算 baseline（无 subhalo）
        print("计算 baseline（无 subhalo）结果...")
        baseline_pos, baseline_mag, baseline_delta_pos, _ = compute_model_predictions(
            model_type, [], temp_prefix='temp_baseline')
        
        if model_type == 'nfw' or model_type == 'p_jaffe':
            plot_paper_style_nfw_compare(
                img_numbers=np.array([1, 2, 3, 4]),
                delta_pos_mas_baseline=baseline_delta_pos,
                delta_pos_mas_optimized=delta_pos_mas,
                sigma_pos_mas=obs_pos_sigma_mas,
                mu_obs=obs_magnifications,
                mu_obs_err=obs_mag_errors,
                mu_pred_baseline=baseline_mag,
                mu_pred_optimized=pred_mag,
                obs_positions_arcsec=obs_positions,
                pred_positions_arcsec=pred_pos,
                crit_segments=crit_segments,
                caus_segments=caus_segments,
                suptitle=f"iPTF16geu: Baseline vs {n_subhalos} {'NFW' if model_type == 'nfw' else 'Pseudo-Jaffe'} Sub-halos",
                output_file=output_file,
                title_left="Position Offset Comparison",
                title_mid="Magnification Comparison",
                title_right="Image Positions & Critical Curves",
                subhalo_positions=subhalos if subhalos else None,
                show_2sigma=args.show_2sigma
            )
        elif model_type == 'pointmass':
            plot_paper_style_compare(
                img_numbers=np.array([1, 2, 3, 4]),
                delta_pos_mas_baseline=baseline_delta_pos,
                delta_pos_mas_optimized=delta_pos_mas,
                sigma_pos_mas=obs_pos_sigma_mas,
                mu_obs=obs_magnifications,
                mu_obs_err=obs_mag_errors,
                mu_pred_baseline=baseline_mag,
                mu_pred_optimized=pred_mag,
                obs_positions_arcsec=obs_positions,
                pred_positions_arcsec=pred_pos,
                crit_segments=crit_segments,
                caus_segments=caus_segments,
                suptitle=f"iPTF16geu: Baseline vs {n_subhalos} Point Mass Sub-halos",
                output_file=output_file,
                title_left="Position Offset Comparison",
                title_mid="Magnification Comparison",
                title_right="Image Positions & Critical Curves",
                subhalo_positions=subhalos if subhalos else None,
                show_2sigma=args.show_2sigma
            )
    else:
        # 标准模式
        if model_type == 'nfw' or model_type == 'p_jaffe':
            plot_paper_style_nfw(
                img_numbers=np.array([1, 2, 3, 4]),
                delta_pos_mas=delta_pos_mas,
                sigma_pos_mas=obs_pos_sigma_mas,
                mu_obs=obs_magnifications,
                mu_obs_err=obs_mag_errors,
                mu_pred=pred_mag,
                mu_at_obs_pred=pred_mag.copy(),
                obs_positions_arcsec=obs_positions,
                pred_positions_arcsec=pred_pos,
                crit_segments=crit_segments,
                caus_segments=caus_segments,
                suptitle=f"iPTF16geu: {n_subhalos} {'NFW' if model_type == 'nfw' else 'Pseudo-Jaffe'} Sub-halos",
                output_file=output_file,
                title_left="Position Offset",
                title_mid="Magnification",
                title_right="Image Positions & Critical Curves",
                nfw_params=subhalos,
                show_2sigma=args.show_2sigma
            )
        elif model_type == 'pointmass':
            plot_paper_style(
                img_numbers=np.array([1, 2, 3, 4]),
                delta_pos_mas=delta_pos_mas,
                sigma_pos_mas=obs_pos_sigma_mas,
                mu_obs=obs_magnifications,
                mu_obs_err=obs_mag_errors,
                mu_pred=pred_mag,
                mu_at_obs_pred=pred_mag.copy(),
                obs_positions_arcsec=obs_positions,
                pred_positions_arcsec=pred_pos,
                crit_segments=crit_segments,
                caus_segments=caus_segments,
                suptitle=f"iPTF16geu: {n_subhalos} Point Mass Sub-halos",
                output_file=output_file,
                title_left="Position Offset",
                title_mid="Magnification",
                title_right="Image Positions & Critical Curves",
                subhalo_positions=subhalos,
                show_2sigma=args.show_2sigma
            )
    
    print(f"\n✓ 完成！图片已保存至: {output_file}")
    
    # 清理临时文件
    for temp_file in glob.glob('temp_*.dat'):
        try:
            os.remove(temp_file)
        except:
            pass


if __name__ == '__main__':
    main()

