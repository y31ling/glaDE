# 🚀 GLADE Tools 快速开始

## 新工具: run_glafic.py

一行命令运行 glafic 计算！

### ⚡ 最简单的用法

```bash
python tools/run_glafic.py <你的结果文件夹>
```

就这么简单！工具会：
1. ✅ 自动找到 best_params.txt
2. ✅ 自动识别模型类型
3. ✅ 生成 glafic 输入文件
4. ✅ 运行 glafic
5. ✅ 显示计算结果

---

## 📖 实例演示

### 示例1: 验证优化结果

```bash
# 假设你刚完成优化，结果在 260327_0123 文件夹
cd /home/luukiaun/glafic251018/work/glade
python tools/run_glafic.py 260327_0123
```

**输出**:
```
======================================================================
Run GLAFIC Tool
======================================================================
文件夹: /home/luukiaun/glafic251018/260327_0123
参数文件: v_nfw_2_0_best_params.txt
模型类型: nfw
Sub-halos 数量: 2

Sub-halo 参数:
  NFW 1: x=0.266035, y=0.000427, M=1.234e+06, c=15.50
  NFW 3: x=-0.238324, y=0.227270, M=5.678e+05, c=20.30

输出目录: /home/luukiaun/glafic251018/260327_0123

生成 glafic 输入文件...
  ✅ 输入文件已生成: glafic_run_input.dat

运行 glafic...
  ✅ glafic 运行成功

======================================================================
GLAFIC 计算结果
======================================================================

图像数量: 4
...（详细结果）...

✅ 完成！
```

### 示例2: 保存到新文件夹

```bash
python tools/run_glafic.py 260327_0123 --output ./verification --prefix verify
```

这会在 `./verification` 文件夹创建：
- `verify_input.dat`
- `verify_point.dat`
- `verify_crit.dat`

### 示例3: 查看详细过程

```bash
python tools/run_glafic.py 260327_0123 --verbose
```

会显示 glafic 的完整输出，包括：
- 参数定义
- 透镜平面信息
- 图像搜索过程
- 临界曲线计算

---

## 🎨 配合其他工具使用

### 完整工作流示例

```bash
# 步骤1: 运行 glafic 验证
python tools/run_glafic.py results/my_run

# 步骤2: 生成可视化图表
python tools/drawgraph.py results/my_run --compare

# 步骤3: MCMC 误差分析（可选）
python tools/mcmc_from_result.py results/my_run --nsteps 2000
```

---

## 📋 支持的模型

| 模型 | 自动识别 | 参数数量 | glafic 类型 |
|------|---------|---------|-------------|
| Point Mass | ✅ | 3/subhalo | `point` |
| NFW | ✅ | 4/subhalo | `gnfw` |
| Pseudo-Jaffe | ✅ | 5/subhalo | `jaffe` |
| King Profile | ✅ | 5/subhalo | `pgc` |

---

## ❓ 常见问题

**Q: 如何找到我的结果文件夹？**

A: 结果文件夹通常在：
- `work/glade/results/`
- `/home/luukiaun/glafic251018/results/`
- 或运行优化时创建的时间戳文件夹（如 `260327_0123`）

**Q: 如果文件夹中没有 best_params.txt 怎么办？**

A: 确保：
1. 优化程序已成功运行完成
2. 文件名符合 `*_best_params.txt` 格式
3. 文件路径正确

**Q: 可以处理没有 subhalo 的结果吗？**

A: 可以！工具会识别并正确处理：
- 仅基础透镜模型（无 subhalo）
- 有 subhalo 的模型

**Q: 输出文件在哪里？**

A: 默认在输入文件夹中，或使用 `--output` 指定位置。

---

## 🔗 相关文档

- **详细文档**: `README_run_glafic.md`
- **工具概览**: `TOOLS_OVERVIEW.md`
- **主文档**: `../README.md`

---

## 🎯 下一步

1. **试运行**: 找一个结果文件夹测试
2. **查看输出**: 检查生成的 glafic 文件
3. **组合使用**: 配合 drawgraph 生成图表
4. **深入分析**: 使用 mcmc_from_result 进行误差分析

---

需要帮助？运行 `python tools/run_glafic.py --help` 查看完整参数说明！
