# GLADE 统一入口（可移植 Linux）

`glade` 目录已经内置四个模型的本地副本，并通过 `main.py` 统一调度：

- `point_mass` -> `legacy/v_pointmass_1_0/version_pointmass_1_0.py`
- `nfw` -> `legacy/v_nfw_2_0/version_nfw_2_0.py`
- `king` -> `legacy/v_king_1_0/version_king_1_0.py`
- `p-jaffe` -> `legacy/v_p_jaffe_2_0/version_p_jaffe_2_0.py`

同时已整合：

- `glafic2` 源代码：`glade/glafic2`
- `tools` 工具集：`glade/tools`（含 `mcmc_from_result.py`、`glafic_optimize.py` 等）

## 新机器一键安装（Linux）

首次在全新 Linux 机器上执行：

```bash
cd glade
bash bootstrap_linux.sh
```

安装脚本会自动完成：

- 安装系统依赖（apt）
- 下载并编译 `CFITSIO`、`FFTW`、`GSL`
- 编译 `glafic2` 二进制与 Python 模块
- 创建 `.venv` 并安装 Python 依赖
- 生成 `env.sh` 与 `run_glade.sh`

安装完成后直接运行：

```bash
./run_glade.sh
```

## 使用方法

1. 编辑 `main.py`
   - 设置 `model_use`
   - 设置 `source_dir`（指向含 `bestfit.dat` 的目录）
   - 在 `common_overrides` 或 `model_overrides[对应模型]` 中修改参数（键名与原脚本变量名一致）
2. 运行（推荐）：

```bash
./run_glade.sh
```

## 参数覆盖机制

- 覆盖键名与原模型脚本中的顶层变量名一致，例如：
  - `active_subhalos`
  - `DE_MAXITER`
  - `MCMC_ENABLED`
  - `fine_tuning_configs`
- `glade` 会在运行前生成带覆盖参数的临时脚本：
  - `legacy/<model_dir>/_glade_generated_run.py`

## 输出目录

- 统一输出到：`glade/results/<model_use>/`
- 模型内部仍保留原逻辑（例如时间戳子目录命名）

## Tools 使用

建议先加载环境：

```bash
source env.sh
```

然后运行：

```bash
python tools/mcmc_from_result.py <result_folder>
python tools/replot_mcmc.py <result_folder>
python tools/glafic_optimize.py <result_folder>
python tools/drawgraph.py <result_folder>
```
