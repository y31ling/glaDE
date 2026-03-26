# GLADE Tools 工具集概览

## 🛠️ 可用工具

### 1. run_glafic.py ⭐ NEW
**快速运行 glafic 计算**

从 best_params 文件自动生成 glafic 输入并运行，获取图像位置和放大率。

```bash
python tools/run_glafic.py <folder>
python tools/run_glafic.py <folder> --prefix my_run --verbose
```

**输出**: glafic 输入文件、图像位置、临界曲线

**适用场景**: 快速验证结果、批量计算、生成标准 glafic 输出

---

### 2. drawgraph.py
**重新生成结果图**

从 best_params 文件生成三联图（位置偏移/放大率/临界曲线）。

```bash
python tools/drawgraph.py <folder>
python tools/drawgraph.py <folder> --compare  # 比较图模式
```

**输出**: PNG 三联图

**适用场景**: 重新绘图、生成论文图表、比较 baseline vs optimized

---

### 3. glafic_optimize.py
**使用 glafic 内置优化**

加载 best_params，添加扰动，使用 glafic 的 amoeba/mcmc 进一步优化。

```bash
python tools/glafic_optimize.py <folder>
python tools/glafic_optimize.py <folder> --mcmc  # MCMC 模式
```

**输出**: 优化后的参数文件、比较报告、三联图

**适用场景**: 精细调优、验证全局最优、交叉验证

---

### 4. mcmc_from_result.py
**MCMC 后验采样**

从 best_params 启动 MCMC 采样，计算参数后验分布和误差。

```bash
python tools/mcmc_from_result.py <folder>
python tools/mcmc_from_result.py <folder> --nsteps 5000 --nwalkers 64
```

**输出**: MCMC 链、Corner plot、轨迹图、后验统计

**适用场景**: 参数不确定度估计、后验分布分析

---

### 5. replot_mcmc.py
**重新绘制 MCMC 结果**

从已保存的 MCMC 链重新生成可视化图表。

```bash
python tools/replot_mcmc.py <folder>
```

**输出**: Corner plot、轨迹图、质量分布图

**适用场景**: 调整图表样式、生成高分辨率图

---

### 6. king_profile_review_plot.py
**King Profile 参数分析**

分析和可视化 King Profile 模型的参数空间。

```bash
python tools/king_profile_review_plot.py
```

**输出**: 参数关系图、CSV 数据

**适用场景**: 模型验证、参数约束分析

---

## 🔄 工具流程图

```
优化结果文件夹
    ├── *_best_params.txt
    │
    ├──> [run_glafic.py]     --> glafic 输出文件 (快速计算)
    │
    ├──> [drawgraph.py]      --> 三联图 PNG (结果可视化)
    │
    ├──> [glafic_optimize]   --> 优化后的参数 (进一步优化)
    │
    └──> [mcmc_from_result]  --> MCMC 链 + 统计 (误差分析)
```

---

## 📊 工具对比

| 工具 | 运行时间 | 输出 | 使用场景 |
|------|----------|------|----------|
| **run_glafic** | 秒级 | glafic 文件 | 快速验证 |
| **drawgraph** | 秒级 | 图片 | 论文图表 |
| **glafic_optimize** | 分钟级 | 优化结果 | 精细调优 |
| **mcmc_from_result** | 分钟-小时级 | MCMC 统计 | 误差估计 |
| **replot_mcmc** | 秒级 | 图片 | 重新绘图 |

---

## 🎯 使用建议

### 快速工作流

1. **验证结果** → 使用 `run_glafic.py`
   ```bash
   python tools/run_glafic.py results/my_folder
   ```

2. **生成图表** → 使用 `drawgraph.py`
   ```bash
   python tools/drawgraph.py results/my_folder --compare
   ```

3. **误差分析** → 使用 `mcmc_from_result.py`
   ```bash
   python tools/mcmc_from_result.py results/my_folder --nsteps 3000
   ```

### 高级工作流

1. DE 优化 → 得到初步结果
2. `run_glafic.py` → 快速验证
3. `glafic_optimize.py --mcmc` → glafic MCMC 采样
4. `mcmc_from_result.py` → Python MCMC 交叉验证
5. `drawgraph.py --compare` → 生成最终图表

---

## 📚 详细文档

- **run_glafic.py**: 查看 `README_run_glafic.md`
- **drawgraph.py**: 工具内置帮助 `python tools/drawgraph.py --help`
- **其他工具**: 查看各工具源码中的文档字符串

---

## 💡 提示

1. **所有工具都支持 `--help`** - 查看完整参数说明
2. **路径可以是相对或绝对** - 工具会自动搜索
3. **支持所有4个模型** - Point Mass, NFW, Pseudo-Jaffe, King
4. **可以组合使用** - 一个文件夹可以被多个工具处理

---

更新日期: 2026-03-27
