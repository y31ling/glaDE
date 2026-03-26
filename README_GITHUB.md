# GLADE: Gravitational Lensing Analysis and Differential Evolution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**GLADE** 是一个统一的引力透镜建模和优化平台，集成了多种透镜模型和现代优化算法。

## 🌟 特性

- **多模型支持**: Point Mass, NFW, King, Pseudo-Jaffe 等透镜模型
- **现代优化算法**: 差分进化、MCMC采样、贝叶斯优化
- **一键部署**: 自动化Linux环境配置和依赖管理
- **统一接口**: 通过 `main.py` 统一调度所有模型
- **可扩展架构**: 易于添加新的透镜模型和优化方法

## 🚀 快速开始

### 系统要求

- Linux (Ubuntu 18.04+ / CentOS 7+ 推荐)
- Python 3.8+
- GCC 编译器
- 至少 2GB 可用磁盘空间

### 一键安装

```bash
# 克隆仓库
git clone https://github.com/your-username/glade.git
cd glade

# 一键安装所有依赖（首次运行）
bash bootstrap_linux.sh

# 运行项目
./run_glade.sh
```

### 快速配置

编辑 `main.py` 设置您的分析参数：

```python
# 选择透镜模型
model_use = "point_mass"  # 可选: 'nfw', 'king', 'p-jaffe'

# 设置数据目录
source_dir = "work/your_data_directory"

# 配置优化参数
common_overrides = {
    "DE_MAXITER": 800,
    "DE_POPSIZE": 75,
    "MCMC_ENABLED": True,
}
```

## 📁 项目结构

```
glade/
├── main.py              # 主入口文件
├── bootstrap_linux.sh   # 一键安装脚本
├── requirements.txt     # Python依赖
├── glafic2/            # glafic引力透镜计算引擎
├── legacy/             # 各种透镜模型实现
│   ├── v_pointmass_1_0/
│   ├── v_nfw_2_0/
│   ├── v_king_1_0/
│   └── v_p_jaffe_2_0/
├── tools/              # 分析工具集
│   ├── glafic_optimize.py
│   ├── mcmc_from_result.py
│   └── drawgraph.py
└── samples/            # 示例数据
```

## 🔧 支持的透镜模型

| 模型 | 描述 | 参数 | 适用场景 |
|------|------|------|----------|
| **Point Mass** | 点质量模型 | x, y, mass | 子晕检测 |
| **NFW** | Navarro-Frenk-White | x, y, M_vir, c_vir | 暗物质晕 |
| **King** | King模型 | x, y, mass, r_c, c | 球状星团 |
| **Pseudo-Jaffe** | 伪Jaffe模型 | x, y, σ, a, r_co | 椭圆星系 |

## 🎯 使用示例

### 基础透镜建模

```python
# 1. 设置模型
model_use = "point_mass"
source_dir = "work/SN_2Sersic_NFW"

# 2. 配置优化
common_overrides = {
    "active_subhalos": [1, 2, 3, 4],  # 启用所有子晕
    "DE_MAXITER": 650,
    "MCMC_ENABLED": True,
    "MCMC_NSTEPS": 5000,
}

# 3. 运行分析
./run_glade.sh
```

### 高级MCMC采样

```python
model_overrides = {
    "point_mass": {
        "MCMC_ENABLED": True,
        "MCMC_NWALKERS": 64,
        "MCMC_NSTEPS": 10000,
        "MCMC_BURNIN": 1000,
        "fine_tuning": True,  # 精细调试模式
    }
}
```

## 📊 输出结果

运行完成后，结果保存在 `results/<model_name>/` 目录：

- `*_best_params.txt`: 最优参数
- `*_corner.png`: MCMC后验分布图
- `*_triptych.png`: 三联分析图
- `*_mcmc_chain.dat`: MCMC采样链

## 🛠️ 工具集

### glafic优化工具

```bash
# 基于已有结果进行glafic优化
python tools/glafic_optimize.py results/point_mass/260326_1234/

# 启用MCMC采样
python tools/glafic_optimize.py results/point_mass/260326_1234/ --mcmc --mcmc_nsteps 50000
```

### MCMC后处理

```bash
# 从优化结果生成MCMC采样
python tools/mcmc_from_result.py results/point_mass/260326_1234/
```

## 🔬 科学应用

GLADE已成功应用于：

- **强引力透镜系统建模**
- **暗物质子结构检测**  
- **超新星引力透镜分析**
- **星系质量分布研究**

## 📚 技术细节

### 核心算法

- **差分进化 (DE)**: 全局优化主算法
- **MCMC采样**: 贝叶斯参数估计
- **Hungarian算法**: 图像匹配优化
- **自适应网格**: 高精度透镜方程求解

### 性能优化

- 多核并行计算支持
- 早停机制和收敛检测
- 内存高效的数据结构
- 批量处理和缓存机制

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **glafic**: 核心引力透镜计算引擎
- **SciPy**: 科学计算生态系统
- **emcee**: MCMC采样库
- **corner**: 后验分布可视化

## 📞 联系方式

- 问题反馈: [GitHub Issues](https://github.com/your-username/glade/issues)
- 邮箱: your-email@example.com

---

**⭐ 如果这个项目对您有帮助，请给我们一个星标！**