# Version Pseudo-Jaffe 2.0: DE + MCMC 后验采样

## 概述

基于 v_p_jaffe_1.0，新增 MCMC 后验采样功能。

**工作流程：**
1. **差分进化(DE)** - 快速找到最优解
2. **MCMC采样** - 以DE最优解为起点，采样后验概率分布

## 新增功能

### MCMC 后验采样
- 使用 `emcee` 库进行集成采样
- 输出完整的参数概率分布
- 生成 Corner Plot 可视化参数相关性
- 计算参数误差 (median ± 1σ)

## 依赖安装

```bash
pip install emcee corner tqdm
```

## 配置参数

### MCMC 配置（在脚本开头）

```python
MCMC_ENABLED = True          # 是否启用MCMC采样
MCMC_NWALKERS = 32           # walker数量（至少是参数维度的2倍）
MCMC_NSTEPS = 3000           # 采样步数
MCMC_BURNIN = 500            # burn-in 步数（丢弃前N步）
MCMC_THIN = 1                # 稀疏采样
MCMC_PERTURBATION = 0.01     # 初始扰动幅度
```

### 建议的MCMC参数设置

| 目的 | NWALKERS | NSTEPS | BURNIN |
|------|----------|--------|--------|
| 快速测试 | 20 | 1000 | 200 |
| 常规运行 | 32 | 3000 | 500 |
| 论文级别 | 64 | 10000 | 2000 |

## 输出文件

运行后在时间戳目录中生成：

| 文件 | 说明 |
|------|------|
| `*_mcmc_chain.dat` | MCMC采样链（去除burn-in后） |
| `*_posterior.txt` | 参数后验统计 (median ± 1σ) |
| `*_corner.png` | Corner Plot（参数相关性） |
| `*_trace.png` | MCMC链轨迹图 |
| `*_best_params.txt` | DE最佳参数 |
| `result_*.png` | 三联图结果 |

## 运行

```bash
cd /home/luukiaun/glafic251018/work/v_p_jaffe_2.0
chmod +x run.sh
./run.sh
```

或直接：
```bash
python3 version_p_jaffe_2_0.py
```

## 与 glafic MCMC 的对比

| 特性 | glafic MCMC | 本程序 (DE + MCMC) |
|------|-------------|-------------------|
| 起点 | 需要良好初始猜测 | DE自动找最优起点 |
| 收敛速度 | 较慢 | 快（DE预优化） |
| 输出 | 文本链文件 | 链 + 统计 + 可视化 |
| 参数相关性 | 需手动分析 | Corner图自动生成 |

## 后验统计示例

运行完成后，`*_posterior.txt` 会包含：

```
# Pseudo-Jaffe Sub-halo at Image 1:
#   x = 0.266035 +0.123 -0.145 mas
#   y = 0.000427 +0.098 -0.112 mas
#   sig = 15.32 +2.45 -1.87 km/s
#   a = 45.67 +3.21 -2.98 mas
#   rco = 5.23 +0.89 -0.76 mas
```

这正是论文中需要报告的参数误差格式。

