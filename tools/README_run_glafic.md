# Run GLAFIC Tool

## 📖 功能说明

`run_glafic.py` 是一个自动化工具，用于从已有的优化结果重新运行 glafic 计算。

### 主要功能

1. **自动识别模型类型** - 支持 Point Mass, NFW, Pseudo-Jaffe, King Profile
2. **解析参数文件** - 自动读取 `*_best_params.txt` 文件
3. **生成 glafic 输入** - 根据模型类型生成标准 glafic 输入文件
4. **运行 glafic** - 调用 glafic 可执行文件进行计算
5. **显示结果** - 格式化输出图像位置、放大率、时间延迟等

---

## 🚀 使用方法

### 基本用法

```bash
python run_glafic.py <folder_path>
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `folder` | 包含 best_params.txt 的文件夹路径 | 必需 |
| `--output` | 输出目录 | 输入文件夹 |
| `--prefix` | 输出文件前缀 | `glafic_run` |
| `--verbose` | 显示详细输出（包括 glafic 完整输出） | False |

---

## 📝 使用示例

### 示例1: 基本使用

```bash
cd /home/luukiaun/glafic251018/work/glade
python tools/run_glafic.py /path/to/results/folder
```

### 示例2: 指定输出目录

```bash
python tools/run_glafic.py 260322_2137 --output ./test_output
```

### 示例3: 自定义前缀 + 详细输出

```bash
python tools/run_glafic.py results/nfw/260120_0344 --prefix my_test --verbose
```

### 示例4: 使用相对路径

工具会自动在以下位置搜索文件夹：
- 当前目录
- glade 根目录
- glade/results 目录
- glafic251018 根目录
- glafic251018/results 目录

```bash
# 这些都可以工作
python tools/run_glafic.py 260322_2137
python tools/run_glafic.py results/nfw/260120_0344
python tools/run_glafic.py ../260125_0419
```

---

## 📤 输出文件

运行后会在输出目录生成以下文件：

| 文件名 | 说明 |
|--------|------|
| `<prefix>_input.dat` | glafic 输入文件 |
| `<prefix>_point.dat` | 图像位置和放大率 |
| `<prefix>_crit.dat` | 临界曲线数据 |

---

## 🔍 支持的模型类型

### 1. Point Mass
- **识别标志**: `Point Mass`, `v_pm`, `mass_sub`
- **参数**: x, y, mass
- **glafic 模型**: `point`

### 2. NFW
- **识别标志**: `Version NFW`, `v_nfw`, `m_vir`
- **参数**: x, y, M_vir, c_vir
- **glafic 模型**: `gnfw`

### 3. Pseudo-Jaffe
- **识别标志**: `Pseudo-Jaffe`, `v_p_jaffe`, `sig`
- **参数**: x, y, σ, a, r_co
- **glafic 模型**: `jaffe`

### 4. King Profile
- **识别标志**: `King`, `v_king`, `r_c`
- **参数**: x, y, M, r_c, c
- **glafic 模型**: `pgc`

---

## 💡 示例输出

```
======================================================================
Run GLAFIC Tool
======================================================================
文件夹: /home/luukiaun/glafic251018/260125_0419
参数文件: v_p_jaffe_1_0_best_params.txt
模型类型: p_jaffe
Sub-halos 数量: 3

Sub-halo 参数:
  Jaffe 1: x=0.228979, y=-0.042731, σ=3.21, a=66.19mas, rco=13.00mas
  Jaffe 3: x=-0.221212, y=0.193302, σ=8.21, a=54.11mas, rco=3.02mas
  Jaffe 4: x=0.143034, y=0.286894, σ=6.19, a=4.06mas, rco=0.16mas

输出目录: /home/luukiaun/glafic251018/260125_0419

生成 glafic 输入文件...
  ✅ 输入文件已生成: test_jaffe_input.dat

运行 glafic...
  glafic 路径: /home/luukiaun/glafic251018/glafic2/glafic
  输入文件: test_jaffe_input.dat
  ✅ glafic 运行成功

======================================================================
GLAFIC 计算结果
======================================================================

图像数量: 4

Img   x [arcsec]      y [arcsec]      μ               Time Delay [day]    
---------------------------------------------------------------------------
1         -0.103000        -0.254400          15.1002              0.000000
2          0.281500        -0.031900         -30.4364              0.164000
3         -0.223400         0.195100          -7.4840              0.213000
4          0.143100         0.287000           9.1504              0.095000

放大率统计:
  总数: 4
  范围: [-30.44, 15.10]
  绝对值平均: 15.54

临界曲线文件: test_jaffe_crit.dat
  大小: 15318 bytes
  线段数: 142

======================================================================
✅ 完成！
======================================================================

输出文件:
  - test_jaffe_input.dat   (输入文件)
  - test_jaffe_point.dat   (图像位置和放大率)
  - test_jaffe_crit.dat    (临界曲线)
```

---

## 🛠️ 故障排除

### 问题1: 找不到 glafic 可执行文件

**错误信息**: `❌ 错误: 找不到 glafic 可执行文件`

**解决方案**:
- 确保 glafic 已正确编译
- 检查 `glafic2/glafic` 是否存在且有执行权限
- 或将 glafic 添加到 PATH 环境变量

### 问题2: 找不到参数文件

**错误信息**: `❌ 错误: 在 xxx 中未找到 *_best_params.txt 文件`

**解决方案**:
- 确保文件夹路径正确
- 确保文件夹中有 `*_best_params.txt` 文件
- 检查文件名是否符合命名规范

### 问题3: 无法识别模型类型

**错误信息**: `❌ 错误: 无法识别模型类型`

**解决方案**:
- 检查 best_params.txt 文件内容是否包含模型类型标识
- 确保参数文件格式正确

---

## 🔧 高级用法

### 批量处理多个文件夹

```bash
#!/bin/bash
# 批量运行 glafic

for folder in results/*/; do
    echo "处理: $folder"
    python tools/run_glafic.py "$folder" --prefix batch_run
done
```

### 自动化脚本

```bash
#!/bin/bash
# 运行并保存日志

python tools/run_glafic.py "$1" --verbose > glafic_run.log 2>&1
echo "日志已保存到 glafic_run.log"
```

---

## 📋 与其他工具的对比

| 工具 | 功能 | 输出 |
|------|------|------|
| `run_glafic.py` | 仅运行 glafic 计算 | glafic 输出文件 |
| `drawgraph.py` | 运行 glafic + 生成三联图 | 图片 + glafic 输出 |
| `glafic_optimize.py` | 使用 glafic 内置优化 | 优化结果 + 图片 |
| `mcmc_from_result.py` | MCMC 后验采样 | MCMC 链 + 统计 |

---

## 📞 技术支持

如有问题，请查看：
- GLADE 主 README.md
- glafic 官方文档
- GitHub Issues

---

更新日期: 2026-03-27
