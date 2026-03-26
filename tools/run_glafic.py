#!/usr/bin/env python3
"""
Run GLAFIC Tool: 从 best_params 文件生成 glafic 输入并运行

功能：
1. 自动查找文件夹中的 *_best_params.txt 文件
2. 自动识别模型类型（pointmass, nfw, p_jaffe, king）
3. 生成 glafic 输入文件
4. 运行 glafic 并输出计算结果

用法:
    python run_glafic.py <folder_path>
    python run_glafic.py <folder_path> --output <output_dir>
    python run_glafic.py <folder_path> --verbose
    
参数:
    folder_path: 包含 best_params.txt 的文件夹路径
    --output: 输出目录（默认为输入文件夹）
    --verbose: 详细输出模式
    --prefix: 输出文件前缀（默认为 'glafic_run'）

示例:
    python run_glafic.py results/nfw/260120_0344
    python run_glafic.py 260120_0344 --output ./test_output --verbose
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
from runtime_env import setup_runtime_env
setup_runtime_env(GLADE_ROOT)
sys.path.insert(0, os.path.join(GLADE_ROOT, 'glafic2', 'python'))

import glafic

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                         固定常量                                       ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# 基础目录
GLAFIC_BASE = os.path.abspath(os.path.join(GLADE_ROOT, '..', '..'))

# 宇宙学参数
OMEGA = 0.3
LAMBDA_COSMO = 0.7
WEOS = -1.0
HUBBLE = 0.7

# 网格设置
XMIN, XMAX = -0.5, 0.5
YMIN, YMAX = -0.5, 0.5
PIX_EXT = 0.01
PIX_POI = 0.2
MAXLEV = 5

# 源参数
SOURCE_Z = 0.4090
SOURCE_X = 2.685497e-03
SOURCE_Y = 2.443616e-02

# 透镜参数
LENS_Z = 0.2160
LENS_PARAMS = {
    'sers1': (1, 'sers', 0.2160, 9.896617e+09, 2.656977e-03, 2.758473e-02,
              2.986760e-01, 1.124730e+02, 3.939718e-01, 1.057760e+00),
    'sers2': (2, 'sers', 0.2160, 2.555580e+10, 2.656977e-03, 2.758473e-02,
              4.242340e-01, 5.396370e+01, 1.538855e+00, 1.000000e+00),
    'sie': (3, 'sie', 0.2160, 1.183382e+02, 2.656977e-03, 2.758473e-02,
            1.571203e-01, 2.920348e+01, 0.0, 0.0)
}

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                         辅助函数                                       ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def find_glafic_bin():
    """查找 glafic 可执行文件"""
    candidates = [
        os.path.join(GLADE_ROOT, 'glafic2', 'glafic'),
        os.path.join(GLAFIC_BASE, 'glafic2', 'glafic'),
        '/home/luukiaun/glafic251018/glafic2/glafic'
    ]
    
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    
    import shutil
    glafic_in_path = shutil.which('glafic')
    if glafic_in_path:
        return glafic_in_path
    
    return None


def find_params_file(folder_path):
    """在文件夹中查找 best_params.txt 文件"""
    pattern = os.path.join(folder_path, '*_best_params.txt')
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # 如果有多个，优先选择最新的
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]


def detect_model_type(params_file):
    """检测模型类型"""
    with open(params_file, 'r') as f:
        content = f.read()
    
    if 'Version NFW' in content or 'v_nfw' in content or 'm_vir' in content:
        return 'nfw'
    elif 'Pseudo-Jaffe' in content or 'King' in content or 'v_king' in content:
        if 'King' in content or 'king' in content.lower() or 'r_c' in content or 'tidal' in content:
            return 'king'
        return 'p_jaffe'
    elif 'Point Mass' in content or 'v_pm' in content or 'mass_sub' in content:
        return 'pointmass'
    
    # 从文件名推断
    basename = os.path.basename(params_file).lower()
    if 'king' in basename:
        return 'king'
    elif 'nfw' in basename:
        return 'nfw'
    elif 'jaffe' in basename:
        return 'p_jaffe'
    elif 'pm' in basename or 'point' in basename:
        return 'pointmass'
    
    return None


def parse_pointmass_params(params_file):
    """解析 Point Mass 模型参数"""
    subhalos = []
    
    with open(params_file, 'r') as f:
        content = f.read()
    
    pattern = r'x_sub(\d+)\s*=\s*([-\d.eE+]+)'
    matches = re.findall(pattern, content)
    
    for img_idx, _ in matches:
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
            subhalos.append((int(img_idx), x, y, m))
    
    subhalos.sort(key=lambda x: x[0])
    return subhalos


def parse_nfw_params(params_file):
    """解析 NFW 模型参数"""
    subhalos = []
    
    with open(params_file, 'r') as f:
        content = f.read()
    
    pattern = r'x_nfw(\d+)\s*=\s*([-\d.eE+]+)'
    matches = re.findall(pattern, content)
    
    for img_idx, _ in matches:
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
            subhalos.append((int(img_idx), x, y, m, c))
    
    subhalos.sort(key=lambda x: x[0])
    return subhalos


def parse_jaffe_params(params_file):
    """解析 Pseudo-Jaffe 模型参数"""
    subhalos = []
    
    with open(params_file, 'r') as f:
        content = f.read()
    
    pattern = r'x_jaffe(\d+)\s*=\s*([-\d.eE+]+)'
    matches = re.findall(pattern, content)
    
    for img_idx, _ in matches:
        x_pattern = rf'x_jaffe{img_idx}\s*=\s*([-\d.eE+]+)'
        y_pattern = rf'y_jaffe{img_idx}\s*=\s*([-\d.eE+]+)'
        sig_pattern = rf'sig{img_idx}\s*=\s*([-\d.eE+]+)'
        a_pattern = rf'(?<![/])a{img_idx}\s*=\s*([-\d.eE+]+)'
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
            subhalos.append((int(img_idx), x, y, sig, a, rco))
    
    subhalos.sort(key=lambda x: x[0])
    return subhalos


def parse_king_params(params_file):
    """解析 King Profile 模型参数"""
    subhalos = []
    
    with open(params_file, 'r') as f:
        content = f.read()
    
    pattern = r'x_king(\d+)\s*=\s*([-\d.eE+]+)'
    matches = re.findall(pattern, content)
    
    for img_idx, _ in matches:
        x_pattern = rf'x_king{img_idx}\s*=\s*([-\d.eE+]+)'
        y_pattern = rf'y_king{img_idx}\s*=\s*([-\d.eE+]+)'
        m_pattern = rf'(?<!log10_)M{img_idx}\s*=\s*([-\d.eE+]+)'
        rc_pattern = rf'r_c{img_idx}\s*=\s*([-\d.eE+]+)'
        c_pattern = rf'(?<!r_)c{img_idx}\s*=\s*([-\d.eE+]+)'
        
        x_match = re.search(x_pattern, content)
        y_match = re.search(y_pattern, content)
        m_match = re.search(m_pattern, content)
        rc_match = re.search(rc_pattern, content)
        c_match = re.search(c_pattern, content)
        
        if x_match and y_match and m_match and rc_match and c_match:
            x = float(x_match.group(1))
            y = float(y_match.group(1))
            m = float(m_match.group(1))
            rc = float(rc_match.group(1))
            c = float(c_match.group(1))
            subhalos.append((int(img_idx), x, y, m, rc, c))
    
    subhalos.sort(key=lambda x: x[0])
    return subhalos


def generate_glafic_input(model_type, subhalos, output_dir, prefix='glafic_run'):
    """生成 glafic 输入文件"""
    input_file = os.path.join(output_dir, f'{prefix}_input.dat')
    
    with open(input_file, 'w') as f:
        f.write("# ========================================\n")
        f.write(f"# GLAFIC Input File - {model_type.upper()} Model\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Sub-halos: {len(subhalos)}\n")
        f.write("# ========================================\n\n")
        
        # 宇宙学参数
        f.write("# Cosmological parameters\n")
        f.write(f"omega      {OMEGA}\n")
        f.write(f"lambda     {LAMBDA_COSMO}\n")
        f.write(f"weos       {WEOS}\n")
        f.write(f"hubble     {HUBBLE}\n\n")
        
        # 输出前缀
        f.write("# Output prefix\n")
        f.write(f"prefix     {prefix}\n\n")
        
        # 网格设置
        f.write("# Grid settings\n")
        f.write(f"xmin       {XMIN}\n")
        f.write(f"ymin       {YMIN}\n")
        f.write(f"xmax       {XMAX}\n")
        f.write(f"ymax       {YMAX}\n")
        f.write(f"pix_ext    {PIX_EXT}\n")
        f.write(f"pix_poi    {PIX_POI}\n")
        f.write(f"maxlev     {MAXLEV}\n\n")
        
        # 启动设置
        n_lenses = 3 + len(subhalos)
        f.write(f"# Startup: {n_lenses} lenses, 0 extended, 1 point source\n")
        f.write(f"startup    {n_lenses} 0 1\n\n")
        
        # 基础透镜模型
        f.write("# Base lens model\n")
        for key in ['sers1', 'sers2', 'sie']:
            params = LENS_PARAMS[key]
            model_type_name = params[1]
            f.write(f"lens       {model_type_name}    {params[2]:.6e}    ")
            f.write(f"{params[3]:.6e}    {params[4]:.6e}    {params[5]:.6e}    ")
            f.write(f"{params[6]:.6e}    {params[7]:.6e}    {params[8]:.6e}    {params[9]:.6e}\n")
        f.write("\n")
        
        # Sub-halos
        if len(subhalos) > 0:
            f.write(f"# Sub-halos ({len(subhalos)} {model_type} perturbations)\n")
            
            if model_type == 'pointmass':
                for img_idx, x, y, m in subhalos:
                    f.write(f"lens       point   {LENS_Z}    {m:.10e}    ")
                    f.write(f"{x:.10e}    {y:.10e}    0.0    0.0    0.0    0.0\n")
            
            elif model_type == 'nfw':
                for img_idx, x, y, m, c in subhalos:
                    f.write(f"lens       gnfw    {LENS_Z}    {m:.10e}    ")
                    f.write(f"{x:.10e}    {y:.10e}    0.0    0.0    {c:.10e}    1.0\n")
            
            elif model_type == 'p_jaffe':
                for img_idx, x, y, sig, a, rco in subhalos:
                    f.write(f"lens       jaffe   {LENS_Z}    {sig:.10e}    ")
                    f.write(f"{x:.10e}    {y:.10e}    0.0    0.0    {a:.10e}    {rco:.10e}\n")
            
            elif model_type == 'king':
                for img_idx, x, y, m, rc, c in subhalos:
                    f.write(f"lens       pgc     {LENS_Z}    {m:.10e}    ")
                    f.write(f"{x:.10e}    {y:.10e}    0.0    0.0    {rc:.10e}    {c:.10e}\n")
            
            f.write("\n")
        
        # 点源
        f.write("# Point source (iPTF16geu supernova)\n")
        f.write(f"point      {SOURCE_Z}    {SOURCE_X:.10e}    {SOURCE_Y:.10e}\n\n")
        
        f.write("end_startup\n\n")
        
        # 命令
        f.write("# Commands\n")
        f.write("start_command\n\n")
        f.write("findimg\n\n")
        f.write("writecrit\n\n")
        f.write("quit\n")
    
    return input_file


def run_glafic(input_file, output_dir, verbose=False):
    """运行 glafic"""
    glafic_bin = find_glafic_bin()
    
    if not glafic_bin:
        print("❌ 错误: 找不到 glafic 可执行文件")
        return False
    
    print(f"\n运行 glafic...")
    print(f"  glafic 路径: {glafic_bin}")
    print(f"  输入文件: {os.path.basename(input_file)}")
    
    try:
        result = subprocess.run(
            [glafic_bin, os.path.basename(input_file)],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if verbose:
            print("\n--- GLAFIC 输出 ---")
            print(result.stdout)
            if result.stderr:
                print("\n--- GLAFIC 错误 ---")
                print(result.stderr)
            print("--- 输出结束 ---\n")
        
        if result.returncode != 0:
            print(f"  ⚠ glafic 返回非零代码: {result.returncode}")
            if not verbose and result.stderr:
                print(f"  错误信息: {result.stderr[:200]}")
            return False
        else:
            print(f"  ✅ glafic 运行成功")
            return True
            
    except subprocess.TimeoutExpired:
        print(f"  ❌ glafic 运行超时（>120秒）")
        return False
    except Exception as e:
        print(f"  ❌ glafic 运行出错: {e}")
        return False


def display_results(output_dir, prefix='glafic_run'):
    """显示计算结果"""
    point_file = os.path.join(output_dir, f'{prefix}_point.dat')
    
    if not os.path.exists(point_file):
        print(f"\n⚠ 未找到输出文件: {point_file}")
        return
    
    print(f"\n" + "=" * 70)
    print("GLAFIC 计算结果")
    print("=" * 70)
    
    try:
        data = np.loadtxt(point_file)
        
        if len(data.shape) == 1:
            n_images = int(data[0])
            print(f"\n图像数量: {n_images}")
            if n_images == 0:
                print("  （未找到图像）")
        else:
            n_images = int(data[0, 0])
            print(f"\n图像数量: {n_images}")
            
            if n_images > 0:
                image_data = data[1:, :]
                
                print(f"\n{'Img':<5} {'x [arcsec]':<15} {'y [arcsec]':<15} {'μ':<15} {'Time Delay [day]':<20}")
                print("-" * 75)
                
                for i, row in enumerate(image_data, start=1):
                    x, y, mag = row[0], row[1], row[2]
                    td = row[3] if len(row) > 3 else 0.0
                    print(f"{i:<5} {x:>13.6f}    {y:>13.6f}    {mag:>13.4f}    {td:>18.6f}")
                
                print("\n放大率统计:")
                mags = image_data[:, 2]
                print(f"  总数: {len(mags)}")
                print(f"  范围: [{mags.min():.2f}, {mags.max():.2f}]")
                print(f"  绝对值平均: {np.mean(np.abs(mags)):.2f}")
    
    except Exception as e:
        print(f"\n❌ 读取结果文件失败: {e}")
    
    # 显示临界曲线文件信息
    crit_file = os.path.join(output_dir, f'{prefix}_crit.dat')
    if os.path.exists(crit_file):
        crit_size = os.path.getsize(crit_file)
        n_lines = sum(1 for _ in open(crit_file))
        print(f"\n临界曲线文件: {os.path.basename(crit_file)}")
        print(f"  大小: {crit_size} bytes")
        print(f"  线段数: {n_lines}")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                         主函数                                         ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description='从 best_params 文件生成 glafic 输入并运行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s results/nfw/260120_0344
  %(prog)s 260120_0344 --output ./test --verbose
  %(prog)s ../results/jaffe/260302_2036 --prefix my_run
        """
    )
    
    parser.add_argument('folder', help='包含 best_params.txt 的文件夹路径')
    parser.add_argument('--output', help='输出目录（默认为输入文件夹）')
    parser.add_argument('--prefix', default='glafic_run', help='输出文件前缀（默认: glafic_run）')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出')
    
    args = parser.parse_args()
    
    # ==================== 查找文件夹 ====================
    folder_path = args.folder
    
    if not os.path.isabs(folder_path):
        # 尝试相对于 GLADE_ROOT 的路径
        possible_paths = [
            folder_path,
            os.path.join(GLADE_ROOT, folder_path),
            os.path.join(GLADE_ROOT, 'results', folder_path),
            os.path.join(GLAFIC_BASE, folder_path),
            os.path.join(GLAFIC_BASE, 'results', folder_path),
        ]
        
        folder_path = None
        for path in possible_paths:
            if os.path.isdir(path):
                folder_path = os.path.abspath(path)
                break
        
        if not folder_path:
            print(f"❌ 错误: 找不到文件夹 '{args.folder}'")
            sys.exit(1)
    
    print("=" * 70)
    print("Run GLAFIC Tool")
    print("=" * 70)
    print(f"文件夹: {folder_path}")
    
    # ==================== 查找参数文件 ====================
    params_file = find_params_file(folder_path)
    
    if not params_file:
        print(f"❌ 错误: 在 {folder_path} 中未找到 *_best_params.txt 文件")
        sys.exit(1)
    
    print(f"参数文件: {os.path.basename(params_file)}")
    
    # ==================== 识别模型类型 ====================
    model_type = detect_model_type(params_file)
    
    if not model_type:
        print(f"❌ 错误: 无法识别模型类型")
        sys.exit(1)
    
    print(f"模型类型: {model_type}")
    
    # ==================== 解析参数 ====================
    if model_type == 'pointmass':
        subhalos = parse_pointmass_params(params_file)
    elif model_type == 'nfw':
        subhalos = parse_nfw_params(params_file)
    elif model_type == 'p_jaffe':
        subhalos = parse_jaffe_params(params_file)
    elif model_type == 'king':
        subhalos = parse_king_params(params_file)
    else:
        print(f"❌ 错误: 不支持的模型类型 '{model_type}'")
        sys.exit(1)
    
    print(f"Sub-halos 数量: {len(subhalos)}")
    
    if len(subhalos) > 0:
        print("\nSub-halo 参数:")
        for item in subhalos:
            img_idx = item[0]
            x, y = item[1], item[2]
            if model_type == 'pointmass':
                m = item[3]
                print(f"  Sub-halo {img_idx}: x={x:.6f}, y={y:.6f}, M={m:.3e}")
            elif model_type == 'nfw':
                m, c = item[3], item[4]
                print(f"  NFW {img_idx}: x={x:.6f}, y={y:.6f}, M={m:.3e}, c={c:.2f}")
            elif model_type == 'p_jaffe':
                sig, a, rco = item[3], item[4], item[5]
                print(f"  Jaffe {img_idx}: x={x:.6f}, y={y:.6f}, σ={sig:.2f}, a={a*1000:.2f}mas, rco={rco*1000:.2f}mas")
            elif model_type == 'king':
                m, rc, c = item[3], item[4], item[5]
                print(f"  King {img_idx}: x={x:.6f}, y={y:.6f}, M={m:.3e}, rc={rc*1000:.2f}mas, c={c:.2f}")
    else:
        print("\n（无 sub-halos，仅基础透镜模型）")
    
    # ==================== 设置输出目录 ====================
    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = folder_path
    
    print(f"\n输出目录: {output_dir}")
    
    # ==================== 生成 glafic 输入文件 ====================
    print(f"\n生成 glafic 输入文件...")
    input_file = generate_glafic_input(model_type, subhalos, output_dir, args.prefix)
    print(f"  ✅ 输入文件已生成: {os.path.basename(input_file)}")
    
    # ==================== 运行 glafic ====================
    success = run_glafic(input_file, output_dir, args.verbose)
    
    if not success:
        sys.exit(1)
    
    # ==================== 显示结果 ====================
    display_results(output_dir, args.prefix)
    
    print("\n" + "=" * 70)
    print("✅ 完成！")
    print("=" * 70)
    print(f"\n输出文件:")
    print(f"  - {args.prefix}_input.dat   (输入文件)")
    print(f"  - {args.prefix}_point.dat   (图像位置和放大率)")
    print(f"  - {args.prefix}_crit.dat    (临界曲线)")


if __name__ == "__main__":
    main()
