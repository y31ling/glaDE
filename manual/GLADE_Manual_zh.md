# GLADE 用户手册

**版本 V0.6.0 · 面向 Linux/WSL2 本地安装 · 2026-07**

GLADE（*Gravitational Lensing Analysis and Differential Evolution*，引力透镜分析
与差分进化）是一个强引力透镜建模工作台。它以本地修改版的 **glafic 2**（Masamune
Oguri 著）作为 CPU 参考引擎，配备 GPU 透镜引擎 **Rhongomyniad**（PyTorch/CUDA），
并用统一的优化器（**差分进化 DE + MCMC**）驱动两者——你可以通过浏览器
（**WebUI**）操作，也可以直接用 Python 或命令行。

本手册面向具备基础编程能力、但并非透镜建模方向的物理专业本科生。手册默认你知道
什么是引力透镜；所有更专门的术语（MCMC、角图、临界线、χ²、宇称……）会在首次出现
的地方解释，并汇总在术语表（附录 A）中。

> **适用范围说明。** 本手册描述的是 **WSL2（Windows 的 Linux 子系统）下的本地
> V0.6.0 代码树**。仓库中个别文档（Rhongomyniad 的 README、`core/SPEC.md` §6）
> 仍停留在 V0.5.0 的描述；凡与本手册冲突之处，以本手册（即 V0.6.0 实际代码
> 行为）为准。

---

## 目录

1. [简介](#1-简介)
2. [支持的系统与环境要求](#2-支持的系统与环境要求)
3. [安装](#3-安装)
4. [快速上手：15 分钟完成第一次拟合](#4-快速上手15-分钟完成第一次拟合)
5. [WebUI](#5-webui)
   — 5.2 [FindImage](#52-findimage-页找像) · 5.3 [Editor](#53-editor-页编辑器) · 5.4 [Clave](#54-clave-页)
6. [`.dat` 配置文件格式](#6-dat-配置文件格式)
7. [拟合引擎与算法](#7-拟合引擎与算法)
   — DE · 损失函数 · 后端与 GPU 批量 · MCMC · 延展源 · 独立验证 · 分轮精调
8. [读懂运行输出](#8-读懂运行输出)
9. [命令行与 Python 库](#9-命令行与-python-库)
10. [数值精度：`TOL_ROMBERG_JHK` 专题注释](#10-数值精度tol_romberg_jhk-专题注释)
11. [故障排查与常见问题](#11-故障排查与常见问题)
12. [附录](#12-附录)
    — A 术语表 · B 版本历史 · C glafic、许可证与引用 · D 环境变量 · E 辅助文件格式

---

# 1 简介

## 1.1 GLADE 做什么

一个强透镜模型 = 一组参数化的质量分布（若干"透镜组件"）+ 一个源。给定观测到的
像位置与放大率，透镜建模就是寻找能复现观测的参数值。GLADE 把这一过程自动化：

1. 你把模型和观测写进一个或多个纯文本 **`.dat` 文件**（第 6 章）。参数写成
   `{lower, upper}` 就表示*参与搜索*；写成普通数值就表示*锁定*。
2. GLADE 用**差分进化（DE）**——一种全局的、不需要导数的优化算法（§7.1）——拟合
   自由参数；每个候选模型由透镜引擎求解透镜方程、预测像位置和放大率来打分。
3. 可选地，**MCMC**（马尔可夫链蒙特卡洛，§7.4）在最优解附近采样，给出参数的
   *不确定度*，而不只是一个最优值。
4. 每次运行的最后还有可选的**独立验证**：最优模型会交给未经改动的 glafic 二进制
   程序和一套 scipy 精确参考计算重新求解（§7.8），让你能够信任结果。

## 1.2 各个组成部分

| 目录 | 是什么 |
|---|---|
| `glafic2/` | 打包在仓库里的本地修改版 **glafic 2.1.14**（C 源码 + `glafic` 可执行文件 + `import glafic` Python 模块）。CPU 参考引擎。GLADE 的本地修改：King 模型（#27）、收紧的 Romberg 容差（第 10 章）、分项 χ² 接口 `c2calc_each`。 |
| `Rhongomyniad/` | **GPU 透镜引擎**（PyTorch/CUDA）。镜像 glafic 的 Python API；实现 **glafic 全部 27 个透镜模型**（V0.6），并把整个 DE/MCMC 种群装进单次 CUDA 调用求值。仅支持单透镜面。 |
| `core/` | 与后端无关的优化器：`.dat` 解析/校验（`core/format`）、DE（`core/optimize`）、MCMC（`core/mcmc`）、结果图（`core/plot`）、glafic↔GLADE 互译（`core/translate`）、验证（`core/verify.py`）。 |
| `webui/` | Flask + Monaco 浏览器工作台：**FindImage**（跑拟合）、**Editor**（编辑 `.dat`），外加 **Clave** 页。 |
| `clave/` | Clave——交互式拖拽算像的透镜计算器（WebUI 第三个页签，也可独立运行）。 |
| `glade/` | `import glade` 库门面（V0.6）：一行 import 即可在任意脚本里使用 `load_config`、`optimize`、`run_mcmc`、`make_triptych` 等。 |
| `InputFiles/` | 你的 `.dat` 输入工作目录——WebUI 的文件管理只能看到这棵目录树。 |
| `runs/` | 每次 WebUI/命令行运行一个目录：日志、图、`status.json`、最优参数。 |
| `tools/`、`legacy/`、`results/` | V0.4 之前的旧流水线及其辅助脚本。仅为保留旧结果而存在；当前流水线**不**使用（§9.6）。 |

## 1.3 如何阅读本手册

- 第 3–4 章是**教程**——按顺序做一遍。
- 第 5–9 章是**参考**——用到什么查什么。
- 第 10 章是**必读警示**——如果你拟合 Sersic/NFW 这类轮廓，或者关心毫角秒级
  精度。
- `等宽字体`表示你要输入的内容或字面的文件/键/按钮名。界面元素按屏幕上的原文
  引用；界面为双语时会同时给出中文串（如 `▶ Run` / `▶ 运行`）。
- 单位：天球上的角位置几乎处处用**角秒**（″）；唯独*观测像位置及其误差*用
  **毫角秒（mas）**（1 mas = 0.001″）。质量单位是太阳质量（M☉）。注意第 6 章
  中的单位提示。

---

# 2 支持的系统与环境要求

## 2.1 操作系统

| 系统 | 状态 |
|---|---|
| **Linux（apt 系，如 Ubuntu）** | 主要目标平台。本手册所有内容在此测试。 |
| **Windows 上的 WSL2** | 开发平台，完整支持。任务通过 **WSLg**（Windows 10 21H2+/11 的 WSL 图形层）在 `gnome-terminal` 窗口中运行；WebUI 借助 WSL2 的自动端口转发，直接在 Windows 浏览器里访问 `http://localhost:6017`。见 §3.5。 |
| **macOS（Apple Silicon 与 Intel）** | 有安装脚本（`bootstrap_macos.sh`），但 macOS 支持**从未在真实硬件上测试过**——从未在真 Mac 上编译或运行。使用风险自负，以安装脚本最后的 `import glafic` 自检为准。macOS 上没有 GPU 运行（仅支持 CUDA）。 |

其他 Linux 发行版只要手工装齐 §3.1 中 apt 包的等价物也能工作。

## 2.2 软件前提

- 官方要求 **Python 3.8+**；项目实际在 **Python 3.12** 上开发并跑 CI——尽量用
  3.10–3.12。
- **C 编译工具链**（`gcc`、`make`）——安装脚本要从源码编译 glafic。
- **CFITSIO、FFTW3、GSL**——编译 glafic 所需。Linux 上安装脚本会自动下载并编译
  固定版本（CFITSIO 4.6.2、FFTW 3.3.10、GSL 2.8）到 `deps/install/`，因此你
  **不需要**系统级安装它们。macOS 上则来自 Homebrew。
- Python 包（自动按 `requirements.txt` 安装，不锁版本）：
  `numpy scipy matplotlib emcee corner tqdm astropy flask`。

## 2.3 GPU 要求（可选）

- 需要 **NVIDIA GPU + CUDA** 和 CUDA 版 **PyTorch**。不支持 AMD/Intel GPU 和
  Apple Metal。
- **安装脚本不会安装 PyTorch**（`torch` 有意不写进 `requirements.txt`，因为该装
  哪个 wheel 取决于你的 CUDA 环境）。这是一步手动操作——见 §3.3。没有 torch 时
  CPU 功能全部可用；选择 GPU 相关按钮会在任务启动时立即报出明确错误。
- WSL2 下 CUDA 通过 Windows 的 NVIDIA 驱动透传（WSL 里不需要装驱动；只要在
  venv 里装 CUDA 版 PyTorch wheel）。
- 显存：批量 GPU 优化器的内存随种群分块大小变化；默认设置在典型配置下 ~6 GB
  以内即可容纳，分块大小可用 `GLADE_GPU_CHUNK` 调节（§7.3）。

## 2.4 磁盘与时间开销

参考 WSL2 安装实测：`deps/` ≈ 350 MB（依赖源码+编译产物）、编译后的
`glafic2/` ≈ 17 MB、`.venv/` 纯 CPU ≈ 300 MB——装了 CUDA 版 PyTorch 后
≈ 5.7 GB。首次安装耗时主要在三个依赖库的 `configure && make`（几分钟，视硬件
而定）；重复运行会跳过它们。

---

# 3 安装

## 3.1 一条命令安装（Linux / WSL2）

```bash
git clone https://github.com/y31ling/glaDE.git
cd glaDE
bash bootstrap_linux.sh
```

脚本是交互式的，提示为中英双语。两个菜单：

1. **操作** —— `[1] Install / 安装`（默认）或 `[2] Uninstall / 卸载`。
2. **安装模式** —— `[1] Virtual environment`（默认；一切隔离在 `.venv/`，推荐）
   或 `[2] Global / System Python`（直接装进系统 Python；Ubuntu 23.04+ 会使用
   `--break-system-packages`）。

安装依次做这些事：

1. 检查所需 apt 包（`build-essential pkg-config python3 python3-dev python3-venv
   python3-pip wget curl tar git libcurl4-openssl-dev zlib1g-dev`），缺什么就用
   `sudo apt-get install` 装。没有 `sudo` 时会停下来请你自行安装。
2. 下载并编译 **CFITSIO 4.6.2 → FFTW 3.3.10 → GSL 2.8** 到 `deps/install/`
   （重复运行时若库已存在则跳过）。每个压缩包都有一串镜像可轮询（其中 GSL 的
   镜像列表额外包含国内镜像——清华、中科大、阿里云；CFITSIO 和 FFTW 只有上游/
   macports/Fedora 镜像）。若*所有*镜像都失败，脚本会明确告诉你手工下载哪个
   压缩包、放到 `deps/src/`，然后重跑。
3. 编译 **glafic2**：重新生成 `glafic2/Makefile`（首次会把原文件备份为
   `Makefile.original`），执行 `make clean && make -j all`，产出 `glafic`
   可执行文件、`libglafic.a` 和 Python 扩展 `glafic2/python/glafic/glafic.so`。
   每次安装 glafic 都会完整重编。
4. 创建 Python 环境并安装 `requirements.txt`。
5. 向 site-packages 写入一个 **`.pth` 文件**（`glafic_glade.pth`），指向
   `glafic2/python`——从此任何 Python 会话都能直接 `import glafic`。
6. 在仓库根目录生成三个启动脚本：**`env.sh`**、**`run_glade.sh`**（旧版
   CLI）、**`run_webui.sh`**。
7. 自检：分别直接导入 `glafic` 和经 `source env.sh` 后导入，打印
   `[OK] glafic import succeeded`。安装以"全部验证通过。"和"下一步"提示结束。

### `env.sh`

`source env.sh`（注意是 *source*，不是执行）会激活 venv，并导出
`GLADE_ROOT`、`GLAFIC_HOME`、`PYTHONPATH`（仓库根、`glafic2/python`、
`Rhongomyniad`、`tools`）、`LD_LIBRARY_PATH`（`deps/install/lib`），同时把
`glafic` 可执行文件加入 `PATH`。

> ⚠️ `env.sh` 开头有 `set -e`，会被 *source 它的* shell 继承：之后任何一条失败
> 的命令都可能直接关掉你的交互终端。遇到这种情况可在 source 后执行 `set +e`，
> 或者干脆用 `run_*.sh` 包装脚本（它们在子 shell 中 source）。

## 3.2 macOS 安装脚本（未经测试）

```bash
bash bootstrap_macos.sh
```

与 Linux 的区别：需要 Xcode Command Line Tools 和 Homebrew；依赖用
`brew install pkg-config gsl cfitsio fftw` 而非源码编译；glafic 用 `cc` 和
macOS 风格链接方式编译；`env.sh` 导出 `DYLD_FALLBACK_LIBRARY_PATH`；多进程用
`spawn` 而非 `fork`（结果一致）。再次强调：**至今没有在真 Mac 上跑过**——如果
你试了，欢迎反馈问题。

## 3.3 为 GPU 安装 PyTorch（手动步骤）

安装脚本不装 PyTorch。启用 GPU 相关功能：

```bash
source env.sh                      # 激活 GLADE 的 venv
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 例：CUDA 12.4 wheel
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

到 <https://pytorch.org/get-started/locally/> 选择匹配你 CUDA 环境的 wheel
（参考机器用的是 `torch 2.6.0+cu124`）。装纯 CPU 版 `pip install torch` 也能
跑通——Rhongomyniad 会静默退回 CPU 张量——但那样 GPU 加速就无从谈起。WSL2 下若
`torch.cuda.is_available()` 打印 `False`，先升级 *Windows 侧* 的 NVIDIA 驱动。

## 3.4 启动

```bash
./run_webui.sh                     # WebUI 在 http://localhost:6017 ；Ctrl+C 停止
GLADE_PORT=8080 ./run_webui.sh     # 换端口
```

> **注意。** 服务器绑定 `0.0.0.0`，也就是说局域网内其他机器同样能访问，而不只
> 是本机。不要在不可信的共享网络上运行——WebUI 能读写 `InputFiles/` 下的文件并
> 启动进程。

命令行运行和库调用见第 9 章。`./run_glade.sh` 启动的是 V0.4 之前的*旧版*流水线
（`main.py`）——只有复现旧 `results/` 结果时才需要它（§9.6）。

## 3.5 WSL2 相关事项

- **终端窗口。** 每个 WebUI 任务都会打开自己的 `gnome-terminal` 窗口，由 WSLg
  像普通 Windows 窗口一样显示。先装一次：`sudo apt install gnome-terminal`。
  即使完全没有可用的终端模拟器，任务也照样运行（后台脱离模式），输出照样流进
  浏览器——只是没有窗口（§5.2.6）。
- **浏览器。** 在 Windows 的浏览器里直接打开 `http://localhost:6017`；WSL2 会
  自动转发 localhost。
- **从 Windows 访问文件。** 仓库在 WSL 文件系统内。Windows 资源管理器中通过
  `\\wsl$\<发行版名>\home\<用户名>\...` 访问——这是把观测 FITS 图像放进
  `InputFiles/` 的实际途径（WebUI 没有上传按钮，§5.3.2）。
- **性能。** 仓库应放在 WSL（ext4）一侧，不要放在 `/mnt/c/…` 下；从 WSL 访问
  Windows 盘的 I/O 慢一个数量级。

## 3.6 `git pull` 之后如何更新

重跑 `bash bootstrap_linux.sh` → Install 即可：依赖编译会被跳过，glafic 重新
编译，Python 依赖重装，启动脚本和 `.pth` 重新生成。若只有 glafic 的 C 源码
变了，`cd glafic2 && make all` 是等效的捷径。

## 3.7 卸载

`bash bootstrap_linux.sh` → `[2] Uninstall` 会先列出将要执行的操作——删除
`deps/`、删除 `.venv/`、清理 glafic 编译产物（恢复 `Makefile.original`）、删除
生成的启动脚本、删除 `.pth` 文件——然后只问一次总确认；确认后全部执行。只有
随后的 shell 配置清理会逐文件询问再删除 GLADE 相关行（每个文件先备份为
`*.glade_uninstall_backup`），最后一步只是报告残留的环境变量路径。源代码、
`.dat` 输入和 `runs/` 结果永远不会被删除。

---

# 4 快速上手：15 分钟完成第一次拟合

本节用仓库自带的示例——对四个观测点源像做"主透镜 + 子结构"拟合——在 CPU 上跑通
整个流程。

**1. 启动服务器，打开界面。**

```bash
./run_webui.sh
```

打开 `http://localhost:6017`。默认落在 **FindImage** 页，深色主题，英文界面
（两者都可以在右上角切换）。

**2. 准备输入文件。** WebUI 只能看到 `InputFiles/` 目录。把自带示例复制过去：

```bash
cp core/examples/constants.dat core/examples/images_data.dat \
   core/examples/lens_and_substructure.dat InputFiles/
```

点击文件面板的 `⟳` 刷新按钮——三个文件出现。

**3. 勾选这三个 `.dat` 文件。** 一个模型可以拆在任意多个文件里（这里是常量 /
观测数据 / 透镜组件三分）；GLADE 会在运行时合并它们（§6.8）。

**4. 选后端。** 左侧栏保持 **CPU**（默认；用 glafic 库为候选模型打分——五个
按钮的含义见 §5.2.1）。

**5. 点 `▶ Run` / `▶ 运行`。** 如果文件里省略了某些基础设置，会弹出标题为
*"Missing basic values — use defaults?"*（缺少基础变量 — 使用默认值?）的对话
框，逐条列出哪些键将回退到什么默认值。**认真读这个列表**（第 6 章会解释原因：
有些默认值是特定巡天项目的遗留值，并不中性）。点 `Confirm` / `确认`。

**6. 观察运行。** 一个新的终端窗口弹出（WSLg 下是 gnome-terminal），其输出同时
流入浏览器的 *Terminal output*（终端输出）面板：可优化维度列表、每隔几代一行的
`iter NNNN best_loss = …`、DE 结果、验证报告。面板标题旁的状态指示从
`running` / `运行中` 变为 `done · loss <x> · <N> iters`。

**7. 查看结果。** 终端面板下方出现结果图（`result.png`，即"三联图"，§8.4）。
磁盘上的全部产物在 `runs/<job_id>/`：`best_params.txt`、`status.json`、
`job.log`、各种图。终端窗口会一直开着，直到你在里面按回车。

**8. 加上 MCMC。** 切到 Editor 页，打开你复制的某个文件，插入模板
`MCMC → MCMC-GeneralConfig`，把 `$int`/`$float` 占位符填上（如 walkers 32、
steps 2000、burn-in 300），保存，回到 FindImage 再次用 **CPU** 运行——文件里有
`MCMC_ENABLED = True` 时会自动变成"先 DE 后 MCMC"，结果区多出角图和迹线图
（§8.6）。

整个闭环就是：*编辑 `.dat` → 勾选 → 运行 → 读图*。手册其余部分都是细节。

---

# 5 WebUI

## 5.1 总览

WebUI 是单页应用，顶部是 40 px 的导航栏：

- 品牌字 `GLADE`，然后三个页签：**`FindImage`** / **`找像`**、**`Editor`** /
  **`编辑器`**、**`Clave`**（Clave 这个名字不翻译）。
- 右上角：**语言按钮**（显示 `EN` 或 `中文`；默认英文）和**主题按钮**（`🌙`
  深色 / `☀️` 浅色；默认深色）。两者都保存在浏览器 localStorage（`glade_lang`、
  `glade_theme`）。切换语言会重绘整个界面文案；已经打印到终端面板里的文本不会
  重新翻译。
- 所有服务器端消息（运行日志、校验错误）只有英文；双语的只是界面本身。

页签切换即时生效；Editor 第一次访问时才加载代码编辑器，Clave 第一次访问时才
加载其页面（内嵌 iframe）。

## 5.2 FindImage 页（找像）

布局从左到右：**后端栏**（五种运行方式）→ **文件选择器**（"Select .dat files" /
"选择 .dat 文件"）→ **运行列**（摘要行、`▶ Run` 按钮、终端面板、结果区）。

### 5.2.1 五个后端按钮

| 按钮 | 副标题 | 实际运行的东西 |
|---|---|---|
| **CPU** | `glafic` | GLADE 的**差分进化**，候选模型由 glafic *库*（编译的 C 扩展）在多进程池中求值。若配置里 `MCMC_ENABLED = True`，DE 之后接着跑 MCMC。 |
| **GPU** | `Rhongomyniad` | 同样的 DE，但候选由 GPU 引擎求值——配置满足条件时整个种群一次 CUDA 调用（§7.3）。同样响应 `MCMC_ENABLED`。 |
| **Glafic** | `amoeba` | **不是 DE。** 运行独立 glafic 二进制程序自带的下山单纯形优化器（`optimize`，"amoeba"）（§7.3.3）。GLADE `.dat` 会先转换成 glafic 输入文件；原生 glafic `.input` 直接运行。 |
| **MCMC** | `emcee only` / `仅 emcee` | **不跑 DE**，直接在 CPU 引擎上做 MCMC。先验就是 `{lo, hi}` 盒子；walker 在盒内均匀起步。忽略 `MCMC_ENABLED`。 |
| **MCMC-GPU** | `emcee · batched CUDA` / `emcee · 批量 CUDA` | GPU 上的纯 MCMC，条件允许时整个 walker 群做批量 CUDA 求值；符合条件且未设定时 walker 数自动提到 1024（§7.5.3）。 |

> ⚠️ **最容易混淆的一对按钮。** *CPU（副标题 "glafic"）* 的意思是"GLADE 的 DE，
> 用 glafic 库打分"；*Glafic（副标题 "amoeba"）* 的意思是"glafic 自带的单纯形
> 优化器，完全不涉及 DE"。两者是不同的算法，输出也不同。

后端选择不会记忆——每次刷新页面都回到 **CPU**。

### 5.2.2 文件选择器

- 显示 `InputFiles/` 目录树（与 Editor 的资源管理器同一棵树）。文件带复选框——
  **支持跨文件夹多选**；文件夹点击展开/收起，但不能整体勾选。
- 勾选多个文件的含义是"把它们合并成一份配置"（§6.8）。摘要行显示
  `N file(s): <路径>` / `N 个文件: <路径>`。
- 列表在这些时机刷新：点 `⟳`、切到本页签、在 Editor 里保存或粘贴文件。被勾选
  的文件如果消失了，勾选会被静默清除。
- 没有文件类型过滤——`InputFiles/` 里可见的任何文件都能勾选，但只有 `.dat`
  （或 Glafic 按钮下的 glafic `.input`）有意义。

### 5.2.3 运行流程与对话框

没有勾选文件时 `▶ Run` 是灰的。点击后有三种结果：

1. **校验错误** —— 弹出 *"Cannot run — configuration errors"* /
   *"无法运行 — 配置错误"*，逐行列出所有问题（每条一行 `✗`；信息会点名出错的
   键或组件，但不带文件名和行号）。运行中止；去 Editor 修文件。完整错误目录见
   §6.10。
2. **默认值确认** —— 弹出 *"Missing basic values — use defaults?"* /
   *"缺少基础变量 — 使用默认值?"*，以 `key = value` 形式列出每个被省略的键及其
   将采用的默认值。`Confirm`/`确认` 继续；`Cancel`/`取消` 中止。若这是一次符合
   walker 自动提升条件的 GPU MCMC 运行，对话框里会直接显示
   `1024 (auto-raised for the batched GPU sampler)`，所见即所得。
3. **成功** —— 终端面板开始滚动输出。

**Glafic** 按钮的特殊情况：只选原生 glafic `.input` 时完全跳过 GLADE 校验
（原样运行；选了多个时只运行*第一个*）；`.dat` 和 `.input` 混选会被明确拒绝。

### 5.2.4 终端面板

- 标题：*"Terminal output — job `<job_id>` (`<terminal>`)"* /
  *"终端输出 — 任务 `<job_id>` (`<terminal>`)"*，其中 `<terminal>` 是任务所在
  的终端程序（`gnome-terminal`、`x-terminal-emulator`、`xterm`、`tmux` 或
  `detached`）。
- 面板内容是任务 `job.log` 的实时尾随（Server-Sent Events 推送），自动滚动到底。
  开始新任务会清空面板——**同一时间只显示一个任务的流，之前任务的输出不保留**
  （完整日志始终在 `runs/<job_id>/job.log`）。
- 状态指示：`running` / `运行中` → 绿色的 `done` / `完成`，后缀
  ` · loss <x> · <N> iters` / ` · 损失 <x> · <N> 次迭代`；跑过 MCMC 还会有
  ` · MCMC accept <a> (<n> samples)` / ` · MCMC 接受率 <a> (<n> 个样本)`；出错
  则红色显示错误状态。
- 浏览器断流（`stream error` / `流错误`）时**任务照常运行**，但界面无法重连。
  去终端窗口或 `runs/<job_id>/` 看。流还有一个设计上的超时：30 分钟无新输出
  自动断开。

### 5.2.5 结果区

任务以 `done` 状态结束后出现在终端面板下方，按顺序显示（只显示存在的）：
**Result** / **结果**（`result.png`）、**MCMC corner** / **MCMC 角图**、
**MCMC trace** / **MCMC 迹线**。各图的读法见第 8 章。其余产物
（`best_params.txt`、迭代帧、验证文件）不在界面显示——从磁盘
`runs/<job_id>/` 拿，或访问
`http://localhost:6017/api/run/<job_id>/result/<文件名>`。

任务可能以 `done` 结束却**没有**结果图（例如最优模型的像数与观测不符；日志里
会有 `[warn] triptych failed: …`）。这时结果区保持隐藏。

### 5.2.6 任务执行模型——终端、停止、并发

- 每次运行获得形如 `260708_153012_a3f1`（`yymmdd_HHMMSS_xxxx`）的 id 和独立
  目录 `runs/<id>/`。可以同时启动多个任务；它们互不共享文件。
- 任务运行在**自己的系统终端窗口**里，标题 `GLADE <job_id>`。终端选择顺序：
  gnome-terminal → x-terminal-emulator → xterm → tmux（无窗口）→ 完全脱离的
  后台进程。最后一种没有窗口，但浏览器的流照常工作。
- **没有停止按钮。** 要中止运行，在它的终端窗口里按 `Ctrl+C`（或直接关窗口）。
  界面的状态检查会发现工作进程已死，报告 `interrupted` 状态并附说明。
- 运行结束后窗口打印 `[GLADE job finished — press Enter to close]` 并等待——
  按回车关闭。
- **重启 WebUI 服务器会忘掉所有任务**：状态变为 `unknown`，图也无法再经界面
  获取——但 `runs/` 下的文件都还在。

## 5.3 Editor 页（编辑器）

一个面向 `InputFiles/` 的迷你 VS Code：50 px 图标栏（**Explorer** /
**资源管理器** 和 **Template** / **模板** 两个面板）、可拖宽的侧栏（拖动分隔
条；120–600 px）、标签栏、Monaco 编辑器，底部一条：左 `🗑 Delete` / `🗑 删除`、
中间当前文件路径、右 `💾 Save` / `💾 保存`。

### 5.3.1 资源管理器

- 根目录固定为 `InputFiles/`——编辑器看不到、也写不了其他位置（越界路径会被
  服务器拒绝）。点号开头的文件不显示。
- 头部小按钮：`＋` 新建文件（默认名 `untitled.dat`）、`🗀` 新建文件夹、`⟳`
  刷新。在"新建文件"里输入 `sub/f.dat` 这样的路径会自动创建中间文件夹。
- **文件右键菜单**：`Open`/`打开`、`Copy`/`复制`、`Paste`/`粘贴`、
  `Import glafic → glade…`/`导入 glafic → glade…`、
  `Export glade → glafic…`/`导出 glade → glafic…`、
  `Import to Clave`/`导入到 Clave`、`Rename…`/`重命名…`、`Delete`/`删除`。
  文件夹右键：`New file…`/`新建文件…`、`New folder…`/`新建文件夹…`、`Copy`、
  `Paste`、`Import glafic → glade…`、`Rename…`、`Delete folder`/`删除文件夹`。
- **复制/粘贴**对文件和*整个文件夹*（递归）都有效。粘贴到右键的文件夹内部，
  或右键文件的旁边。重名自动改为 `name_copy.ext`、`name_copy2.ext`……。文件夹
  不允许粘贴进自己。没有"剪切"。
- **删除**总会确认（*"Delete `<name>`? This cannot be undone."* /
  *"删除 `<name>`?此操作不可撤销。"*）；删除文件夹是递归的——一次确认删掉里面
  的一切。
- 已知小毛病：文件夹右键菜单里也有 `Import glafic → glade…`，但它只对文件有效
  （对文件夹必然弹 "Import failed" 报错）。

### 5.3.2 文件进出

**没有上传/下载按钮。** 要引入外部文件——最常见的是延展源拟合用的观测 FITS
图像——请直接在文件系统里复制进 `<仓库>/InputFiles/`（Windows 下经
`\\wsl$\<发行版>\home\<用户>\...\InputFiles\`），然后点 `⟳`。运行产物只通过
FindImage 的结果区 / 结果 API 提供。

### 5.3.3 编辑器本体

- **Monaco**（VS Code 的编辑器组件），本地内置——多标签页、每个标签独立撤销
  历史、`●` 未保存标记，以及针对 `.dat` 格式的语法高亮：`#` 注释、青色的
  `{lo, hi}` 花括号组、金色加粗的 `$float`/`$int`/`$str` 模板占位符、字符串、
  数字。
- **编辑器内没有语法检查**：错误只会在 FindImage 运行时的校验对话框里暴露。
  自动补全也只有 Monaco 默认的词汇建议。
- **保存**：只有 `💾 Save` 按钮。**没有 Ctrl+S 绑定**（按 Ctrl+S 会弹出浏览器
  自己的保存网页对话框），**没有自动保存**；关闭/刷新浏览器页会静默丢弃所有
  标签的未保存修改。关闭未保存的标签会弹确认——注意确认按钮的文字是
  `Delete` / `删除`，但含义是"放弃修改并关闭"。
- 保存会同时刷新资源管理器和 FindImage 的文件列表，新 `.dat` 立即可跑。
- ⚠️ 目录树里的任何文件都能打开，包括二进制文件。FITS 打开是乱码——**千万不要
  从编辑器保存它**，否则磁盘上的文件会被破坏（读入解码是有损的）。

### 5.3.4 模板面板

模板把现成的、带注释的 `.dat` 片段**插入当前标签页的光标处**（先打开一个
文件）。占位符 `$float` / `$int` / `$str` 表示必须由你填写的值；
`$float{lower, upper}` 表示可优化参数——要么换成真实的 `{lo, hi}` 区间，要么写
普通数值锁定。**占位符没填完的文件在运行时过不了校验。** 模板组名和模板名只有
英文。

组件类模板（Lens / Sub-structure / Extend Source）插入时自动重编号：组件键名
变成 `'nfw2'`、`'nfw3'`……，前导序号取*当前文档*中最大序号加一（把组件拆到多个
文件时要留意这一点）。

模板组（V0.6.0）：

| 组 | 模板 |
|---|---|
| `OBS DATA` | `Images Data`（源/透镜红移、源位置、四个观测数组——位置及位置误差以 **mas** 为单位，放大率及其误差无量纲——`center_offset_*`、`obs_x_flip`）、`Constants`（宇宙学 + 网格）、`Extend_images`（延展源 FITS 相关键 + glafic χ²/噪声设置） |
| `Source` | `point`（其余两个是占位，禁用） |
| `Lens` | `Sersic`、`SIE`、`power-law`、`Hernquist`、`Einasto`、`perturbation (shear)`、`Gaussian`（gaupot）、`ahern`、`clus3`、`mpole`、`crline`、`gals`——全部标注 `# GPU-supported` |
| `Sub-structure` | `point-mass`、`NFW`、`gNFW`、`tNFW`、`analytic NFW`、`King`、`p-jaffe`、`acnfw` |
| `Extend Source` | `Sersic (extend)`、`Gaussian (extend)`、`top-hat (extend)`、`Moffat (extend)`、`Jaffe (extend)`——延展源轮廓，需要 `extended_file` |
| `Algorithm parameters` | `DE-CPU`、`DE-GPU`（多出 `gpu_precision`，去掉 `DE_WORKERS`）、`DE-Extend (CPU)`（`W_*` χ² 权重） |
| `MCMC` | `MCMC-GeneralConfig` |

"Lens"和"Sub-structure"只是*写作上的分类*——两者插入的组件元组格式相同，最终
进入同一个透镜堆栈（§6.3）。每个透镜片段的注释都写明参数含义和顺序；参数标签
属于"尽力标注"的模型会额外注明"请对照 glafic 文档核实顺序"。

### 5.3.5 导入 / 导出 / 导入到 Clave

- **`Import glafic → glade…`（导入）**（右键一个 glafic `.input` 文件）：转换
  成同目录下的 `<名>_model.dat`（有约束时另加 `<名>_obs.dat`），同名文件静默
  覆盖。glafic 优化矩阵里标记为自由的参数会变成**退化区间 `{v, v}`——拟合前必
  须手工拉宽**（零宽度的区间搜不了任何东西）。导入的观测数据一律写
  `obs_x_flip = False`、`center_offset_* = 0`；若你的数据用天球约定，记得改回
  （§6.5）。
- **`Export glade → glafic…`（导出）**（右键 `.dat`）：询问输出名，写到
  `InputFiles/` **根目录**：总是生成 `<名>_model.input`；若选择包含 `{lo, hi}`
  参数，则导出为*可直接优化*的形式——`.input` 带上 `start_setopt` 矩阵和
  `optimize` 命令，另生成 `<名>_obs.dat`（glafic `readobs_point` 约束）和
  `<名>_prior.dat`（glafic `parprior` 范围）。（没有 `{lo, hi}` 但有观测数据时
  也会写 `<名>_obs.dat`——不过那是 GLADE 自用的 `start_obs` 往返格式，glafic
  二进制本身读不了它。）区间坍缩为代表值（质量类取几何
  平均，其余取算术平均）。共享变量（§6.6）导出为 glafic 的 `match` 绑定。
- **`Import to Clave`（导入到 Clave）**（右键 `.dat` *或* `.input`）：把模型
  转成 Clave 场景并切到 Clave 页。`{lo, hi}` 同样坍缩为代表值；只携带一个点源；
  **当前 Clave 场景会被直接替换，没有确认提示**。

## 5.4 Clave 页

Clave 是交互式的**点源透镜计算器**：把透镜和源摆到画布上、用鼠标拖动，像的
位置、放大率和时间延迟由真正的透镜方程求解器（CPU 用 glafic，GPU 用
Rhongomyniad）实时重算。它是建立直觉、快速搭草图的画板——不是拟合工具（不做
优化；没有延展源，也不画临界线）。

运行方式：作为 WebUI 第三个页签（`http://localhost:6017/clave/`），或独立运行
`python -m clave`（自带服务器，默认端口 6019，可用 `CLAVE_PORT` 改）。

两个 Clave 特有的怪癖：它有**自己的语言开关**（工具栏上的 `中/EN`），且
**默认中文**，与主 WebUI 的语言设置互不相干；它**始终是浅色主题**，即使外面的
WebUI 是深色。

### 5.4.1 工具栏

`CLAVE` 标志 · 副标题 · **CPU/GPU 开关**及状态徽章（`CUDA` = GPU 就绪，悬停
可见 GPU 名；`CPU` = Rhongomyniad 可用但没有 CUDA 设备；`N/A` = Rhongomyniad
导入失败，开关禁用）· 状态点/文字（`就绪` / `Ready`、`计算中...` /
`Computing...`、`完成` / `Done`）· `中/EN` · 模式按钮 **`实时模式` /
`Realtime Mode`** ⇄ **`稳定模式` / `Stable Mode`**。

- **实时模式**（默认）：拖动过程中以 10 Hz 节流重算。
- **稳定模式**：开始拖动即隐藏所有像；直到你按下侧栏底部出现的
  `计算像位置` / `Calculate` 按钮才重算。（对话框编辑和增删对象仍然立即计算——
  只有拖动被推迟。）

### 5.4.2 画布

- **平移**：拖动空白处。**缩放**：鼠标滚轮（以光标为中心）。自适应网格的坐标
  标注会随缩放在 ″ 和 mas 之间自动切换。
- **选中**：点击对象（蓝色高亮）；**移动**：拖动已选中的对象，或按住 Alt 拖动
  任意对象。已锁定的对象（侧栏行的小锁）拖不动。
- **旋转**（限透镜）：选中透镜后，其长轴上出现一个小手柄，拖动它设置位置角；
  侧栏实时显示 `PA=…°`。
- 图元：源是蓝色六角星；像是琥珀色五角星，透明度随 |μ| 变化（越亮的像越不
  透明）；透镜是按参数定尺寸的半透明形状（如 SIE 圆盘 ≈ 其爱因斯坦半径；
  NFW 族画成 mas 量级的小光斑，便于子结构工作；King 显示核 + 潮汐半径环）。

### 5.4.3 侧栏、对话框与模型选择器

- `光源 Sources` 和 `透镜 Lenses` 两节，各有 `添加光源` / `ADD SOURCE`、
  `添加透镜` / `ADD LENS` 按钮。行内实时显示坐标；悬停出现锁定（`🔒`）和删除
  （`✕`）；拖 `≡` 手柄排序（顺序即引擎的透镜编号）。双击行打开编辑对话框。
- **添加透镜**对话框有一个可搜索的类型下拉框，内置 15 个精选模型，分三组——
  常用：`sie`、`nfw`、`king`、`gnfw`、`point`、`jaffe`；延展：`sers`、`hern`、
  `pow`、`tnfw`、`ein`、`anfw`、`ahern`；扰动：`pert`、`gaupot`——各有合理默认
  值（如 SIE σ = 200 km/s、z = 0.5）。其他 glafic 模型仍可经 *Import to
  Clave* 进入 Clave。
- 每个对话框都有**文本编辑模式**（`切换文本编辑器`）：把整个对象写成一行
  `type z p1 x y e pa r1 r2`。往第一个输入框粘贴一整行空白分隔的数据也会自动
  分配到各字段。
- `zs_fid` 家族（`pow`、`pert`、`gaupot`）与 §6.4 的约定一致：第一个"z"字段是
  **透镜红移**（默认 0.5），"参考源红移 zs / Fiducial zs"字段是基准源红移
  `zs_fid`（默认 1——应大于透镜红移）。另外：红移填 `0` 会被静默换成默认值
  （透镜 0.5、源 2），非数字输入会变成 0。
- 宇宙学固定为 (Ω_m, Ω_Λ, w, h) = (0.3, 0.7, −1, 0.7)，不可修改；红移逐对象
  可调。

### 5.4.4 像面板与导出

场景中至少有一个透镜和一个源之后，出现只读的 **`像 Images (n)` /
`Images (n)`** 面板，列出每个像：`★j  x=…″ y=…″ μ=…`（多个源时按源标注）。
时间延迟有计算但只出现在导出文件里。`导出数据` / `Export Data` 下载
`clave_export.txt`——glafic 风格的文本文件，含透镜/源行和像表
（x、y、μ、时间延迟）。导出是单向的（Clave 不能重新导入它）；场景不跨刷新
保存。

### 5.4.5 GPU 模式注意事项

GPU 模式要求所有透镜在同一红移面（容差 10⁻³），且模型必须在
`rhongomyniad.supported_models()` 里（V0.6 已是全部 27 个）。第一次 GPU 交互
可能因内核预热而略有停顿。CPU 和 GPU 的初始搜索网格略有不同，临界线附近个别
极暗的像偶尔会有出入。落在自动确定的搜索框（± max(0.4″, 2 × 场景尺度)）之外
的像不会被找到——大分离构型要留意。若编译好的 glafic 模块缺失，CPU 模式会给出
**假数据**（状态显示 `Mock mode`）——只是占位几何，数值毫无意义。

---

# 6 `.dat` 配置文件格式

一个 GLADE 模型由一个或多个 `.dat` 文件描述。语法看起来像 Python，但由一个
**受限的解析器读取，绝不会当作代码执行**：只允许字面量、`{lo, hi}` 二元组、
列表、元组、名字引用、下标，以及 `+ - * / **` 四则/乘方运算。其余一切（函数
调用、比较、字典……）都是语法错误。校验会一次性报出*所有*问题（库/命令行使用时
带文件名和行号；WebUI 对话框只显示信息文本）。

一个最小的完整模型，按惯例拆成三个文件（`core/examples/*.dat` 就是一套可运行
的示例）：

```python
# --- constants.dat -------------------------------------------
omega        = 0.3          # Ω_m
lambda_cosmo = 0.7          # Ω_Λ
weos         = -1.0         # 暗能量状态方程 w
hubble       = 0.7          # H0/100
xmin, ymin   = -0.5, -0.5   # 搜索视场 [角秒]
xmax, ymax   =  0.5,  0.5
pix_ext      = 0.01         # 延展源像素 [角秒]
pix_poi      = 0.2          # 点源搜索网格单元 [角秒]
maxlev       = 5            # 自适应网格细化层数

# --- images_data.dat -----------------------------------------
source_z = 0.4090
lens_z   = 0.2160
source_x = {-0.10, 0.10}    # {lo, hi} = 参与优化；普通数值 = 锁定
source_y = 0.0244

obs_positions_mas_list  = [[229.0, -390.6], [-286.4, -401.7],
                           [-267.4, 393.8], [ 385.8, 274.0]]   # 单位是 mas！
obs_magnifications_list = [ 8.0, 6.1, 4.9, 3.7]
obs_mag_errors_list     = [ 0.8, 0.6, 0.5, 0.4]
obs_pos_sigma_mas_list  = [ 2.0, 2.0, 2.0, 2.0]                # mas
center_offset_x = 0.0
center_offset_y = 0.0
obs_x_flip = True           # True = 天球约定（x 轴取反）

# --- lens_and_substructure.dat -------------------------------
'sie1':   (1, 'sie',   lens_z, {150, 350}, {-0.05, 0.05}, {-0.05, 0.05},
           {0.0, 0.6}, {0, 180}, 0.0)
'point1': (2, 'point', lens_z, {1e5, 1e7}, {-0.30, -0.20}, {-0.05, 0.05})
```

## 6.1 行级语法

- `#` 到行尾是注释（引号字符串内除外）。注释可以出现在跨行的值中间，但**绝不要
  把含有左/右括号的行注释掉**——括号配对发生在剥离注释之后。
- 只要有 `(`、`[` 或 `{` 未闭合，语句就可以跨越多行。
- 支持元组拆包：`xmin, ymin = -0.5, -0.5`。
- 每个标量在一个文件里只能赋值**一次**（在整个多文件选择中也只能一次——§6.8）；
  重复赋值是错误，通过别名重复也算（`lambda` ≡ `lambda_cosmo`）。
- 只有两种语句：标量赋值（`name = value`）和组件条目（`'name': (…)`，§6.3）。
  其他任何写法都是错误。

## 6.2 值的形式

| 写法 | 含义 |
|---|---|
| `0.3`、`-1`、`2.65e-3` | 固定（锁定）的数值 |
| `{lo, hi}` | **参与优化**：一个搜索维度，边界含端点 |
| `[a, b, …]`、`[[…], …]` | 列表（观测数组） |
| `True` / `False` | 布尔值 |
| `'text'` / `"text"` | 字符串（文件路径、输出前缀） |
| `name` | 引用此前定义过的数值标量 |
| `$float`、`$int`、`$str`、`$float{lower, upper}` | 未填写的模板占位符——**填完之前无法运行** |

值得记住的细节：

- `{hi, lo}` 写反了会被静默调整为正序（没有警告）。`{a, b, c}` 元素数 ≠ 2 是
  错误。
- 纯数值运算在解析时就地折算：`re = 0.3 + 0.09` 在哪儿都合法。引用*观测*的
  表达式只允许出现在组件元组内（§6.7）。
- **质量类参数在 log₁₀ 空间搜索**，但你写的永远是*物理值*：`{1e5, 1e7}` 表示
  "在 log₁₀M ∈ [5, 7] 上均匀搜索"。质量槽位的两个边界都必须 > 0。
- 标量上的 `{lo, hi}` 只在受支持的地方产生搜索维度：`source_x`、`source_y`、
  `hubble`、组件的 `z`/参数、以及用户变量。特别地，
  **`lambda_cosmo = {lo, hi}` 会被接受但静默忽略**（回落到固定默认值——这是文档
  写明的限制；宇宙学参数中只有 `hubble` 可以拟合）。

## 6.3 组件元组

```python
'name': (N, 'type', z, p1, p2, ..., pk)
```

- `'name'` —— 引号包起来的自定义标签（`'sie1'`、`'sub_A'`）；它会成为所有输出
  中参数标签的前缀（`sie1.sigma`、`sub_A.mass`）。
- `N` —— 整数序号。**仅供参考**：文件合并时会按文件选择顺序从 1 开始全局重新
  编号。
  - 可选的**分类后缀**：`3l`/`3L` 强制视为*主透镜*，`3s`/`3S` 强制视为
    *子结构*（只影响结果图上红色"Sub-halo"标记/标签——绝不改变物理）。无后缀
    时按模型的类别归类，但**任何带可优化参数的组件默认按子结构处理**。其他
    字母是错误。
- `'type'` —— 引号包起来的模型关键字，见 §6.4。
- `z` —— 组件红移：数值、`{lo, hi}` 或 `lens_z` 之类的引用。对*延展源*轮廓，
  这一槽位是**源**红移。
- `p1…pk` —— 模型参数，**按 glafic `set_lens` 的顺序**（§6.4）。每个槽位可以是
  数值、`{lo, hi}`、引用、表达式（§6.7）或占位符。超出模型最低要求的末尾参数
  可以省略（补 0）——多数偏折体至少要给前三个（质量类、x、y），延展源轮廓要
  6–7 个（`extsersic`/`extmoffat` 必须给全 7 个）；z 之后最多 7 个参数。

主透镜和子结构在**同一个组件堆栈**里——Editor 模板里的"Lens"与"Sub-structure"
只是写作分类。只要所选后端支持，模型可以任意混搭（例如 SIE 主透镜 + King 子
晕 + 点质量子晕）。

## 6.4 模型参考

glafic 的全部 27 个偏折体模型都被识别，且 V0.6 起**全部都能跑 GPU 后端**
（GPU 仅存的例外是*多透镜面*配置——组件位于不同红移）。下表为 z 之后的参数
顺序；**加粗** = 质量类（log₁₀ 搜索）。

**偏折体**（glafic `set_lens`）：

| type | z 之后的参数（按顺序） | 说明 |
|---|---|---|
| `point` | **mass**, x, y | 点质量 [M☉] |
| `sie` | **sigma**, x, y, e, pa, rcore | 奇异等温椭球；σ [km/s] |
| `jaffe` | **sigma**, x, y, e, pa, a, rco | 拟 Jaffe（外截断 a，核 rco） |
| `sers` / `serspot` | **mass**, x, y, e, pa, re, n | Sersic；n ∈ [0.06, 20] |
| `nfw` / `nfwpot` / `anfw` | **mass**, x, y, e, pa, c | NFW（anfw = 解析 CSE 近似） |
| `gnfw` / `gnfwpot` | **mass**, x, y, e, pa, c, alpha | 广义 NFW（内斜率 α，表格范围 [0, 2]） |
| `tnfw` / `tnfwpot` | **mass**, x, y, e, pa, c, t | 截断 NFW |
| `acnfw` | **mass**, x, y, e, pa, c, b | 有核 NFW（CSE）；**b = 核半径/rs ∈ [0, 100)**，严格上界 |
| `king` | **mass**, x, y, e, pa, rc, c | King (1962)；**GLADE 本地模型 #27**；c = log₁₀(rt/rc) ≥ 0 |
| `hern` / `hernpot` / `ahern` | **mass**, x, y, e, pa, rb | Hernquist |
| `ein` / `einpot` | **mass**, x, y, e, pa, c, alpha | Einasto；α 表格范围 [0.02, 1.0] |
| `pow` / `powpot` | zs_fid, x, y, e, pa, **re**, gamma | 幂律——注意质量类槽位是 **re**（p6），p1 是基准源红移 |
| `pert` | zs_fid, x, y, **gamma**, theta_gamma, –, **kappa** | 外剪切 + 收敛 |
| `gaupot` | zs_fid, x, y, e, pa, **sigma**, **kappa0** | 高斯势 |
| `clus3` | zs_fid, x, y, **gamma**, theta_gamma | 星系团扰动 |
| `mpole` | zs_fid, x, y, **gamma**, theta_gamma, m, n | 多极 |
| `crline` | zs_fid, x, y, –, pa, epsilon, kappa | 直线临界线 |
| `gals` | **sigma**, x, y, –, –, a, alpha | 星系表（成员为拟 Jaffe）；需要 `galfile.dat` 星表——见 §9.4 |

常用参数含义：`x, y` 中心 [角秒]；`e` 椭率 ∈ [0, 1)；`pa` 位置角 [度]；尺度类
（`rcore, re, rb, rc, a, …`）单位角秒。`…pot` 变体把椭率放在势而不是密度上。
`zs_fid` 家族（`pow`、`pert`、`gaupot`、`clus3`、`mpole`、`crline`）的 p1 是
用来定归一化的*基准源红移*——转换模型时的常见陷阱。参数标签在 schema 里标为
"尽力标注"的模型（`pow`、`pert`、`gaupot`、`clus3`、`mpole`、`crline`、
`gals`）在被优化时会给出警告：请对照 glafic 手册
（`glafic2/manual/man_glafic.pdf`）核实参数顺序。

**延展源轮廓**（glafic `set_extend`；元组的 z 槽位 = **源**红移；p1 = `norm`，
其含义由 `flag_extnorm` 决定：0 = 峰值面亮度，1 = 总流量）：

| type | z 之后的参数 | |
|---|---|---|
| `extsersic` | norm, x, y, e, pa, re, n | Sersic 轮廓 |
| `extgauss` | norm, x, y, e, pa, sigma | 高斯 |
| `exttophat` | norm, x, y, e, pa, radius | 平顶 |
| `extmoffat` | norm, x, y, e, pa, rd, beta | Moffat |
| `extjaffe` | norm, x, y, e, pa, a, rco | Jaffe |

## 6.5 观测数据（点源路径）

只要拟合点源像，就**必须**给出四个数组（没有默认值）：

```python
obs_positions_mas_list  = [[x1, y1], [x2, y2], ...]  # 像位置 [mas]
obs_magnifications_list = [mu1, mu2, ...]            # 带符号的放大率
obs_mag_errors_list     = [d1, d2, ...]              # 放大率 1σ 误差（> 0）
obs_pos_sigma_mas_list  = [s1, s2, ...]              # 位置 1σ 误差 [mas]（> 0）
```

所有数组长度必须一致；误差必须是有限的正数（零/负误差会让 χ² 发散，直接被
拒绝）。

观测坐标系与引擎（透镜）坐标系之间的映射：

```
x_engine = ±(x_mas/1000 − center_offset_x)     # obs_x_flip = True 时取负号
y_engine =   y_mas/1000 − center_offset_y
```

- `obs_x_flip = True` 表示**天球约定**（x 向东增大 ⇒ 相对数学 x 轴取反）。它是
  *x 的符号翻转*，绝不是 x/y 交换。
- `center_offset_x/y` [角秒] 把观测系原点平移到透镜系原点。

> ⚠️ **GLADE 最危险的一组默认值。** 这些键的内置默认值继承自本项目起家的
> iPTF16geu 分析：`center_offset_x = 0.01535`、`center_offset_y = 0.0322`、
> `obs_x_flip = True`，且 `source_x/y`、`source_z`、`lens_z` 的默认值都是
> iPTF16geu 的最佳拟合值。分析任何新系统时**务必显式写出**（通常
> `center_offset_x = 0`、`center_offset_y = 0`，再加上你自己的 `obs_x_flip`）。
> 默认值确认对话框（§5.2.3）存在的意义正是让你注意到这些。

带符号的放大率与*宇称*：位于时延面极小/极大点的像宇称为正，鞍点像为负——所以
数组里的 μ 带符号。GLADE 默认比较 |μ|（`abs_mag` 键，§7.2），临界线附近的宇称
翻转不会主导你的 χ²。

## 6.6 用户自定义共享变量

任何名字不属于已知配置键的赋值都定义一个**变量**：

- 固定变量（`my_re = 0.39`）：在每个引用处代入其值。
- 被组件元组引用的可优化变量成为**一个共享搜索维度**——所有引用它的槽位拟合到
  *同一个*值：

```python
lens_x = {-0.07, 0.07}
lens_y = {-0.07, 0.07}
'sers1': (1, 'sers', lens_z, {1e9, 1e12}, lens_x, lens_y, {0.01, 0.5}, {0, 180}, {0.08, 0.5}, {0.5, 1.5})
'sers2': (2, 'sers', lens_z, {1e9, 1e12}, lens_x, lens_y, {0.01, 0.5}, {0, 180}, {0.5, 2.0}, {0.5, 1.5})
```

这里两个 Sersic 组件被强制共享同一个中心，`lens_x`、`lens_y` 在角图和拟合输出
中各只出现一次，用变量自己的名字标注。

规则：

- 一个变量不能同时用于质量类槽位*和*线性槽位（错误 `var_mixed_usage`：请定义
  两个变量）。用在质量类槽位时和普通质量一样按 log₁₀ 搜索。
- `{lo, hi}` 变量只能*从组件元组里*引用——不能被其他标量赋值或列表引用。
- 引用可以**跨文件**（多文件选择时在 A 文件定义、B 文件引用）。
- 定义了却从未被引用的 `{lo, hi}` 变量给出警告（`var_unused`）——通常是打错
  字了。
- ⚠️ 推论：**拼错的配置键会被静默当作一个无用的用户变量接受。**
  `DE_MAXITTER = 900` 什么也不做、也没人提醒。当某个设置看起来不生效时，先查
  拼写（另外看 job 日志里的 `[defaults]` 一行，它列出了所有回退到默认值的键）。
- GPU 后端上，用在组件*红移*或 `zs_fid` 槽位的共享变量会静默关闭快速批量路径
  （退回逐候选模式，结果正确但更慢）。

## 6.7 算术与观测表达式（V0.5.3）

在组件元组内部（参数值和区间边界），可以引用观测像位置：

```python
'point1': (4, 'point', lens_z, {1e2, 1e8},
           {img1_x - 0.075, img1_x + 0.075},
           {img1_y - 0.075, img1_y + 0.075})
```

- `imgN_x` / `imgN_y` = 第 N 个观测像（从 1 数）。
  `obs_positions_mas_list[i][j]`（从 0 数；j：0 = x，1 = y）指向同一数据。
- **关键单位规则：** 这些引用*不是*原始 mas 数值——而是已经换算到引擎坐标系的
  位置（mas→角秒、x 翻转、中心平移；即 §6.5 的公式）。所以上面的例子是以像 1
  在透镜系中的真实位置为中心的 ±0.075″ 搜索盒，表达式里的算术常数单位是
  **角秒、引擎坐标系**。
- *其他*观测数组的元素（`obs_magnifications_list[k]` 等）也可以引用，但传入的
  是**原始值**（不做变换）。
- 允许的运算符：`+ - * / **` 和括号。错误（未知名字、下标越界、表达式出现在
  组件元组之外、除零……）会连同组件名和参数槽位一起报告。
- 若 `.dat` 省略了 `center_offset_*`/`obs_x_flip`，表达式会使用*默认值*
  （即 iPTF16geu 的值——§6.5），以保证和优化器实际使用的坐标系一致。

表达式在载入时、所有文件合并之后求值——每个后端（CPU/GPU/amoeba 导出/编辑器
lint）看到的数字完全相同。

## 6.8 多文件合并

在 FindImage 勾选多个文件（或给 `glade.load_config` 传多个路径）时：

- 每个标量在整个选择中至多能定义在**一个**文件里（否则错误 `conflict`，别名
  同样计入）。
- 组件按选择顺序连接，并全局重新编号。
- 跨文件引用没问题：B 文件的元组可以用 A 文件定义的 `lens_z`。
- 合并之后，缺失的基础量按默认值补齐（弹确认对话框）；四个观测数组和"至少一个
  组件"永远不会被默认值顶替。

惯用的拆法是*常量* / *观测* / *透镜模型*三分——这样换透镜假设而不动数据轻而
易举。

## 6.9 配置键完整参考

别名：`lambda` → `lambda_cosmo`，`MISSING_IMG_PENALTY` → `missing_img_penalty`，
`GPU_PRECISION` → `gpu_precision`。"默认值" = 键缺失时（经确认后）采用的值。

**宇宙学与视场**

| 键 | 默认值 | 含义 |
|---|---|---|
| `omega` | 0.3 | Ω_m |
| `lambda_cosmo` | 0.7 | Ω_Λ（写 `{lo,hi}` 会被接受但忽略——不可拟合） |
| `weos` | −1.0 | 暗能量状态方程 w |
| `hubble` | 0.7 | H₀/100。可写 `{lo, hi}` 增加一个拟合维度，**只对带时间延迟的延展源运行有意义**（§7.6）；点源路径上是死维度（批量 GPU 会拒绝）。 |
| `xmin, ymin, xmax, ymax` | ±0.5 | 像面搜索视场 [角秒]——视场外的像不会被找到 |
| `pix_ext` | 0.01 | 延展源像素尺寸 [角秒]；决定 FITS 网格 |
| `pix_poi` | 0.2 | 点源像搜索的粗网格单元 [角秒] |
| `maxlev` | 5 | 自适应网格细化深度（临界线附近丢像时调大它） |

**红移 / 点源**

| 键 | 默认值 | 含义 |
|---|---|---|
| `source_z` | 0.4090 | 源红移（iPTF16geu 默认值！） |
| `lens_z` | 0.2160 | 主透镜红移（iPTF16geu 默认值！）；可在元组中作为引用使用 |
| `source_x`, `source_y` | iPTF16geu 值 | 点源位置 [角秒]；可锁定或 `{lo, hi}`。延展源模式下它们*不是* DE 维度（glafic 逐候选内部求解源位置；`{lo,hi}` 只是标记其自由）。 |

**观测数据** —— 见 §6.5（四个数组、`center_offset_x/y`、`obs_x_flip`）；
延展源模式额外有：`obs_td_list`/`obs_td_err_list`（时间延迟及误差，默认每像
0）、`obs_parity_list`（默认 0）。

**差分进化**

| 键 | 默认值 | 含义 |
|---|---|---|
| `DE_MAXITER` | 650 | 最大代数 |
| `DE_POPSIZE` | 64 | 种群倍数——实际种群 = `DE_POPSIZE × 维度数` |
| `DE_ATOL` / `DE_TOL` | 1e-4 / 1e-4 | 绝对/相对收敛阈值 |
| `DE_SEED` | 42 | 随机种子。同一配置 + 同一种子 ⇒ **任何后端、任何进程数下轨迹完全一致**（同时被 MCMC 初始化复用）。 |
| `DE_POLISH` | True | 被接受但在当前流水线中**无任何效果**（scipy 的抛光步骤根本不会被执行到） |
| `DE_WORKERS` | −1 | CPU 工作进程数；−1 = 全部核心。GPU 后端强制为 1。 |
| `EARLY_STOPPING` | True | 最优损失停滞时提前结束 |
| `EARLY_STOP_PATIENCE` | 30 | 连续多少代在容差内才停止 |

**损失（点源路径）** —— `Y = LOSS_COEF_A·χ²_pos + LOSS_COEF_B·χ²_mag + penalty`（§7.2）

| 键 | 默认值 | 含义 |
|---|---|---|
| `LOSS_COEF_A` | 4 | 位置 χ² 的权重 |
| `LOSS_COEF_B` | 1 | 放大率 χ² 的权重 |
| `LOSS_PENALTY_PL` | 10000 | 位置残差超过其 1σ 的像，每个附加的线性惩罚系数 |
| `missing_img_penalty` | 0.0 | 每缺一个像的惩罚。0 = 产生的像比观测少的候选被硬性拒绝；> 0 = 按已有的像打分，再加 `(n_obs − n_pred) × penalty`，给 DE 一个走向完整像数的梯度。 |
| `abs_mag` | True | 用绝对值比较（和绘制）放大率——对宇称不敏感。`False` 恢复 V0.5 之前的带符号行为（逐位一致）。 |

**分轮精调** —— 宏观 → 子结构 → 联合抛光（§7.9）

| 键 | 默认值 | 含义 |
|---|---|---|
| `fine_tuning` | 缺失 = 关闭 | 11 元组 `(activate, algo1, A1, B1, algo2, A2, B2, algo3, perturb, A3, B3)`，或单个字面量 `False`。启用后以 §7.9 的分轮流水线取代单次全局搜索：第 1 轮删掉所有子结构组件，只拟合用户设为可优化的宏观参数（主透镜 + 源），保留 `fine_tuning_top_k` 个彼此不同的盆地作为独立链；第 2 轮把每条链的宏观参数（含源）冻结在其种子上，只拟合子结构；第 3 轮先淘汰第 2 轮损失超过最优 10× 的链，再把**每一个**透镜/子结构/源参数——无论用户原来固定还是可优化——在链当前值的 `value·(1 ± perturb)` 窄盒内重新放开并抛光；幸存链中损失最低者胜出。`algoN` = `'DE'` / `'BIPOP-CMA-ES'` / `'jSO'`（**不**支持 amoeba）；`AN`/`BN` 只在对应轮次覆盖 `LOSS_COEF_A`/`LOSS_COEF_B`。透镜 vs 子结构的划分：显式 `Nl`/`Ns` 下标后缀优先，否则按模型的 schema 类别。以下情况回退为普通单次运行并给出警告：没有主透镜、没有子结构、第 1 或第 2 轮无可优化参数、延展源（FITS）模式、或某个共享 `{lo,hi}` 变量同时横跨透镜与子结构。第 3 轮细则：红移/`hubble` 保持原有的可优化性；原本固定的精确零保持固定，原本可优化的零回退为 `±perturb·(hi−lo)` 半宽；窄盒还会与用户原始 `{lo,hi}` 求交（对原本可优化的参数），并一律钳制到引擎的硬性定义域（椭率 < 1、Sersic n ∈ [0.06, 20]、幂律 γ ∈ (1, 3) 等）——钳制后塌缩的盒子让该参数保持固定。第 3 轮只可能让链变好：现任解（盒心）会按第 3 轮目标重新打分，抛光打不过就保留现任解；被跳过的链同样重新打分，因此所有链的最终损失都在同一 `A3`/`B3` 口径下可比。各阶段输出写入运行目录下的 `ft_round1/`、`ft_round2_chain<n>/`、`ft_round3_chain<n>/`——各自保留该轮自己的损失口径，其中的 `glade_output_*.dat` 是可直接重跑的*单阶段*输入（带显式 `Nl`/`Ns` 后缀、不含 `fine_tuning` 键）；胜出者的最终损失（status.json、`best_params.txt`、三联图、glafic 验证）会换算回 `.dat` 自身的 `LOSS_COEF_A`/`LOSS_COEF_B` 口径，随后照常走三联图/验证/MCMC 流程（MCMC 种子会被裁剪回用户原始区间内）。这三个键名（连同大写别名）自 V0.7.1 起为保留名——老 `.dat` 若把它们用作自定义变量必须改名。库调用走 `glade.run_fine_tuning(cfg, backend=...)`；普通 `glade.optimize()` 只跑单阶段，遇到激活的 `fine_tuning` 键会给出警告。别名 `FINE_TUNING`。 |
| `fine_tuning_top_k` | 3 | 整数 ≥ 1——第 1 轮保留多少个彼此不同的盆地作为独立链。别名 `FINE_TUNING_TOP_K`。 |
| `fine_tuning_diversity` | 0.1 | 取值 (0, 1]——至少有一个搜索维度相差 ≥ 该维区间宽度的这个比例时，两个候选才算不同盆地（归一化 L∞ 距离）。别名 `FINE_TUNING_DIVERSITY`。 |

**GPU**

| 键 | 默认值 | 含义 |
|---|---|---|
| `gpu_precision` | 64 | 批量 GPU 的浮点精度：`64` = 全程 fp64；`48` = 混合（偏折场用 fp32 + 牛顿修正用 fp64——**推荐**，fp32 的速度、fp64 的精度）；`32` = 全程 fp32（快，但临界线附近有噪声）。CPU 运行和延展源批量路径（恒为 fp64）不受此键影响。其他值是校验错误。 |

**输出 / 验证**

| 键 | 默认值 | 含义 |
|---|---|---|
| `Draw_Graph` | 1 | 向 `runs/<id>/iterations/` 输出 DE 种群角图帧 |
| `draw_interval` | 5 | 每隔多少代输出一帧 |
| `OUTPUT_PREFIX` | `'glade_run'` | 输出前缀（导出 glafic 时用作 `prefix`） |
| `glafic_verified` | True | 运行结束后用独立的 glafic 二进制 + scipy 参考复核最优解（§7.8） |
| `COMPARE_GRAPH` / `SHOW_2SIGMA` / `CONSTRAINT_SIGMA` / `PENALTY_COEFFICIENT` / `PRINT_INTERVAL` | True / False / 1 / 1000 / 无 | 被接受的历史遗留键，当前流水线**不使用**——但仍会补上这些默认值（因此会出现在 `[defaults]` 日志行和确认对话框里） |

**MCMC**（§7.4–7.5）

| 键 | 默认值 | 含义 |
|---|---|---|
| `MCMC_ENABLED` | False | 对 CPU/GPU 按钮：True = DE 之后接着跑 MCMC（"de+mcmc"）。专用的 MCMC/MCMC-GPU 按钮忽略此键。 |
| `MCMC_NWALKERS` | 32 | 系综规模；运行时下限为 `2·ndim + 2`；符合条件的 MCMC-GPU 运行在未设定时自动提到 1024 |
| `MCMC_NSTEPS` | 2000 | 每个 walker 的步数 |
| `MCMC_BURNIN` | 300 | 丢弃的开头步数 |
| `MCMC_THIN` | 2 | burn-in 之后每 N 步保留一步 |
| `MCMC_PERTURBATION` | 0.01 | de+mcmc 模式下 walker 围绕 DE 最优点的初始散布（占各维区间宽度的比例） |
| `MCMC_WORKERS` | 1 | 似然计算的 CPU 池；−1 = 全部核心，**只在真正的前台终端中安全**（§7.7）；批量 GPU 路径忽略 |
| `MCMC_PROGRESS` | True | 进度条（仅库调用；WebUI 运行打印步数行代替） |
| — | — | **没有 `MCMC_SEED`**；`DE_SEED` 只决定 walker 初始化，采样器本身不设种子，因此链不能精确复现 |
| `MCMC_CUSTOM_RANGE`、`MCMC_SEARCH_RADIUS`、`MCMC_LOG_M_MIN`、`MCMC_LOG_M_MAX` | — | V0.4.1 中移除；接受但忽略并给弃用警告——MCMC 先验*永远*是 `{lo, hi}` 区间 |

**延展源模式**（§7.6）——文件路径为字符串，相对路径先按*第一个*选中 `.dat`
所在目录解析，再按工作目录：

| 键 | 默认值 | 含义 |
|---|---|---|
| `extended_file` | —（延展源模式必填） | 观测 FITS 图像 → glafic `readobs_extend`。设置它（或包含任何 `ext*` 组件）就把整个运行切到延展源路径。 |
| `extend_mask_file` | 无 | 可选的像素掩模 FITS（> 0 的像素被忽略） |
| `noise_file` | 无 | 可选的逐像素噪声 FITS（不给则由 `obs_gain`/`obs_readnoise` 等解析推导） |
| `constraint_file` | 无 | glafic 原生 `readobs_point` 点源约束文件（与四个观测数组同时给出时以本文件为准） |
| `prior_file` | 无 | glafic 原生 `parprior` 文件（参数的高斯/区间先验） |
| `W_POS, W_FLUX, W_TD, W_EXT, W_PRIOR` | 各 1.0 | glafic χ² 各分量的权重：`loss = W_POS·pos + W_FLUX·flux + W_TD·td + W_EXT·pixel + W_PRIOR·priors + penalty`。全为 1.0 时与 glafic 的 `c2calc` 完全一致。 |
| `chi2_splane, chi2_checknimg, chi2_restart, chi2_usemag, ran_seed, obs_gain, obs_ncomb, obs_readnoise, flag_extnorm` | glafic 默认 | 原样转发给 glafic `set_secondary`（见 glafic 手册）。存在点源约束时，*批量* GPU 延展路径要求 `chi2_splane 1`（源面点 χ²）。 |

## 6.10 校验

错误（阻止运行）：语法错误；未填的占位符；非法/无法解析的引用；跨文件重复定义
标量；坏表达式；缺观测数组（或延展源模式下四个数组只给了一部分）；缺
`extended_file` / 缺延展源组件；观测数组格式不对（类型/长度/形状/非正误差）；
未知模型类型；组件参数太少/太多；质量边界非正；所选后端不支持的模型；共享变量
同时用于质量与线性槽位；非法 `gpu_precision`；未知后端；一个组件都没有。

警告（继续运行）：弃用键；未被引用的 `{lo,hi}` 变量；观测数组和
`constraint_file` 同时给出（以文件为准）；尽力标注的参数标签；延展源组件上的
分类后缀（忽略）；GPU 后端 + 多个透镜红移（单面引擎可能拒绝）。

在库/命令行使用中，问题以 `[error] file.dat:行号: 信息` 的格式打印；WebUI 的
错误对话框只显示信息文本（不带文件名/行号）。

---

# 7 拟合引擎与算法

## 7.1 差分进化（DE）

**它是什么。** DE 是一种*全局*、不需要导数的优化算法。它维护一整个候选参数向量
的*种群*；每一*代*中，为每个成员构造一个试探解——把两个随机成员的差按比例加到
当前最优解上（`best1bin` 策略），再和父代做交叉混合——谁得分好留谁。种群早期
大范围探索、后期收缩到极小值附近，非常适合透镜建模这种损失面上布满局部极小、
又没有梯度可用的问题。

**GLADE 中 DE 的具体行为**（基于 `scipy.optimize.differential_evolution`）：

- `.dat` 里每个 `{lo, hi}` 是一个维度；质量类维度在 log₁₀ 空间搜索。种群规模 =
  `DE_POPSIZE × 维度数`（默认 64 × 维度数）。
- 策略 `best1bin`、变异系数在 (0.5, 1) 内逐代抖动、交叉率 0.7、拉丁超立方初始
  化——这些是 scipy 默认值，不可配置。
- **可复现性不变量**：所有路径都强制 scipy 的"deferred"更新模式，因此*同一配置
  + 同一 `DE_SEED` 的轨迹完全一致*——无论 CPU 还是 GPU、1 个进程还是 32 个。
- **提前停止**（GLADE 自己的）：某一代最优损失的改善小于 `DE_ATOL`（绝对）或
  `DE_TOL`（相对）就算"收敛代"；连续 `EARLY_STOP_PATIENCE` 个收敛代即结束。
  scipy 自己的种群离散度判据和 `DE_MAXITER` 同样会终止运行。
- 报告的最优解永远是种群成员之一（`DE_POLISH` 无效果）。
- 进度行 `iter NNNN best_loss = …` 每 5 代打印一次。

**如何选区间。** DE 的开销随维度数和区间宽度增长。能锁定的都锁定；其余给出
宽松但物理的 `{lo, hi}`；善用观测表达式（§6.7）把位置搜索盒对准观测像。

## 7.2 点源损失函数

对每个候选，引擎求解透镜方程并给出预测像。GLADE 计算

```
Y = LOSS_COEF_A · χ²_pos + LOSS_COEF_B · χ²_mag + penalty
```

- `χ²_pos = Σ (Δᵢ/σᵢ)²` —— Δᵢ 是观测像 *i* 与其匹配预测像之间的距离 [mas]
  （匹配用匈牙利算法按距离做最优一一配对）；σᵢ 是
  `obs_pos_sigma_mas_list[i]`。
- `χ²_mag = Σ ((μ_pred − μ_obs)/δμ)²`；`abs_mag = True`（默认）时用 |μ| 比较——
  观测 +30 对模型 −29 只差 1 而不是 59；临界线附近的宇称翻转不再受罚。
- `penalty = Σ LOSS_PENALTY_PL · Δᵢ`，对每个 Δᵢ 超过 σᵢ 的像累加——注意它对
  *整个*残差 Δᵢ 是线性的，因此 Δᵢ 一越过 σᵢ 就会跳变式地生效：把位置强力推向
  1σ 以内。

**像数规则。** 模型恰好比观测多一个像时，丢掉最暗的那个（视作尖峰轮廓的中心
退放大像）。多出更多 ⇒ 候选直接判死（损失 10¹⁵）。比观测*少*：默认同样判死，
**除非** `missing_img_penalty > 0`——此时按它已有的像打分，再加
`(n_obs − n_pred) × missing_img_penalty`。这种"分级"模式把悬崖变成坡，让 DE
有梯度可爬向完整像数；当好候选总在早期被整批拒绝时推荐打开。（合适的量级在
点源与延展源路径不同——换模式时需要重调。）

## 7.3 后端

### 7.3.1 CPU（以及"glafic"为什么出现两次）

`cpu` 后端用 **glafic C 扩展**在多进程池（`DE_WORKERS`，默认全核）中为候选打
分。`runjob.py` 里名为 `glafic` 的*后端*是同一个引擎；而 WebUI 的 **Glafic
按钮**运行的是 glafic 的 *amoeba*（§7.3.3）。CPU 是参考基准：其他所有路径都对
它验证。

### 7.3.2 GPU：批量求值、精度、分块

`gpu` 后端用 Rhongomyniad 求值。它的速度来自**批量化**：配置允许时，*整个 DE
种群*在每代一次 CUDA 调用中完成求值。参考数字（RTX 4080 SUPER，20 维 NFW
拟合，每代 180 个候选）：逐个求值 ~21 s/代 → 批量 fp64 ~2.3 s → `gpu_precision
= 48` 时 **~0.86 s**（约 27 倍）。

配置不可批量的情形：某模型没有 GPU 内核（V0.6 已不存在）、组件*红移*或
`zs_fid` 参与优化、组件不在同一透镜面、（点源路径）`hubble` 参与优化。这时
GLADE 打印 `[warn] batched GPU objective unavailable (<原因>)` 并退回逐候选
GPU 求值——结果正确，但通常不比 CPU 池快，此时建议改用 CPU 按钮。确认拿到快速
路径的标志是日志里的 `GPU-batched objective active …`。

- **`gpu_precision`**（§6.9）：64 = fp64（与旧路径逐位一致）；**48 = 推荐**——
  昂贵的偏折场阶段用 fp32、牛顿修正和放大率用 fp64；实测与 fp64 无法区分
  （损失一致到 ~3 × 10⁻¹⁵），速度约 2.7 倍；32 = 全程 fp32——探索够用，但临界
  线附近候选的损失相对噪声可达 ~10⁻⁴–5 × 10⁻²。
- **`GLADE_GPU_CHUNK`**（环境变量）：广义模式下每次 CUDA 调用的候选数。默认
  32（Schramm 积分重的模型：nfw/king/sers/hern/pow/gnfw/tnfw/ein）或 128
  （轻量模型），偏折场为 fp32 时翻倍。显存富余时调大能再快 ~10 %；**CUDA 内存
  不足（OOM）时调小它**。
- ⚠️ 批量 GPU 失败（比如 OOM）**不会终止运行**：整个种群得 10¹⁵ 分，DE 表面上
  继续"跑"，实际什么都没优化。唯一的信号是一行
  `[warn] batched GPU objective failed …`——损失一直停在 1e15 时先读日志。
- GPU 运行永远是单进程（`DE_WORKERS` 被忽略）。

### 7.3.3 Glafic 按钮（amoeba）

WebUI 的 **Glafic** 按钮（CLI：`--mode amoeba`）运行独立 glafic 二进制自带的
**下山单纯形（"amoeba"）优化器**——从代表值出发的局部优化，不是全局搜索。用途：
用一条完全独立的代码路径交叉核对 DE 结果；原样运行现成的 glafic `.input`；
或需要严格按 glafic 的 χ² 定义时。

- GLADE `.dat` 选择会先在任务目录里转换为 `amoeba_model.input` +
  `amoeba_obs.dat`（约束）+ `amoeba_prior.dat`（参数范围）；每个 `{lo, hi}`
  变成自由标志 + 范围先验，起点取几何（质量）/算术平均。`DE_*`/`MCMC_*` 键在
  这条路径上不起作用。
- 原生 `.input` 选择会被*装载*（连同其引用的所有数据文件复制）进任务目录后
  原样运行；若输入里没有 `optimize` 命令，就只执行其中已有的命令。
- 前提：仅点源配置（延展源被拒绝）、至少一个 `{lo, hi}`、有观测数组。墙钟
  限时：环境变量 `GLAFIC_AMOEBA_TIMEOUT`（默认 3600 秒；0 = 不限）——glafic 的
  单纯形在病态模型上可能停不下来。
- 输出：`best_params.txt` 里是 `chi^2` 和像列表；拟合出的参数在 glafic 自己的
  `<prefix>_optresult.dat` 里（§8.3）。

## 7.4 MCMC：它是什么、GLADE 怎么用

DE 给你一个最优解。**MCMC**（马尔可夫链蒙特卡洛）估计的是*后验分布*——哪些参数
值与数据相容、概率几何——从而给出不确定度和参数间的关联。

GLADE 用 **emcee**，一种*系综*采样器：不是一条链，而是一群 **walker**（游走点）
同时在参数空间里移动；每一步中每个 walker 参照其他 walker 的位置构造一次提议
移动（"stretch move"），按似然比接受或拒绝。步数足够后，walker 的位置就是后验
的样本。

GLADE 中的关键机制：

- **先验** = 你的 `{lo, hi}` 区间张成的均匀盒子。永远如此，没有单独的先验设置
  （V0.4.1 之前的先验键会被忽略并警告）。质量维度按 log₁₀ 采样。
- **似然** = `exp(−loss/2)`，其中 loss 正是 DE 的损失（§7.2 / §7.6）——DE 和
  MCMC 看到同一个地形。
- **burn-in**（`MCMC_BURNIN`）：开头若干步（walker 还在从出发点向后验迁移）被
  丢弃。**thinning / 抽稀**（`MCMC_THIN`）：其后每 N 步保留一步，降低样本自
  相关。最终样本数 ≈ `nwalkers × (nsteps − burnin) / thin`（默认
  32 × (2000−300)/2 = 27 200）。
- walker 数无论怎么设都有下限 `2·ndim + 2`。

## 7.5 跑 MCMC 的三种方式

### 7.5.1 `de+mcmc`（推荐）

设 `MCMC_ENABLED = True` 然后用 **CPU** 或 **GPU** 按钮：先跑 DE，然后 walker
在 DE 最优点附近的小高斯球里出发（每维散布 = `MCMC_PERTURBATION` × 区间宽度）
向外探索。角图上用红线标出 DE 最优点。这是稳健的默认选择——采样器从似然高的
地方起步。

### 7.5.2 纯 MCMC（**MCMC** 按钮）

walker 在先验盒内*均匀*起步。诚实，但在高维下危险：如果随机点几乎不可能给出
有效模型（比如像数不对 ⇒ 似然为 0），初始化就会挣扎。GLADE 会在预算内重抽
不可行的 walker，然后用可行 walker 的抖动副本补齐（附警告）；若*一个*可行
walker 都找不到，运行会中止，并明确提示你改用 `de+mcmc` 或收窄区间。纯 MCMC
运行结束后，`result.png` 画的是*后验中位数*模型。

### 7.5.3 MCMC-GPU 与 walker 自动调节

**MCMC-GPU** 按钮用批量 CUDA 调用计算系综似然（emcee 每次更新半个系综）。
32 个 walker 会让 GPU ~97 % 时间闲置，所以当以下条件*全部*成立时 GLADE 会
**把 walker 数自动提到 1024**：你没有设置 `MCMC_NWALKERS`、配置可批量、且这是
延展源运行或纯点质量（"legacy"）点源运行。走广义（分块）路径时默认保持 32 并
附说明——那里多一个 walker 就多一分开销。显式设置的值永远不会被覆盖（只会在
明显浪费 GPU 时打印一条 `[hint]`）。所有决定都会写进日志，默认值确认对话框也
会提前显示提升后的值。

### 7.5.4 判断 MCMC 是否健康

- **接受率**（实时打印并写入摘要）：健康的 emcee 运行大致在 **0.2–0.5**。接近
  0 ⇒ walker 卡死（区间太宽、模型太脆）；接近 1 ⇒ 步子太小、没在探索。
- **迹线图**（§8.6）：红色 burn-in 线之后，每个参数的轨迹应像平坦的"毛毛虫"。
  漂移或分裂成几条带 ⇒ burn-in 不够或后验多峰。
- GLADE **不计算自相关时间**，也只有延展源路径会在接受率 < 0.01 时自动警告——
  收敛与否要你自己判断。
- 耗时量级：点质量模型在 CPU 池上几分钟能采完；Romberg 积分重的轮廓
  （Sersic/NFW……）要几十分钟；延展源 MCMC 在 CPU 上*慢得多*（每个 walker 每步
  一次完整的 glafic 求值）——`MCMC_NSTEPS` 保守一点，或者用 `de+mcmc` + GPU。

## 7.6 延展源拟合

配置里设置了 `extended_file`（观测 FITS 图像）或含任何 `ext*` 组件时，运行
切换到**延展源路径**：不再匹配点像列表，而是由 glafic 把模型源经透镜光线追踪、
在像素网格上渲染出透镜像，χ² 里加入逐像素项。

- 损失是 glafic 的 `c2calc` 按分量拆开、各自加权：`loss = W_POS·pos +
  W_FLUX·flux + W_TD·td + W_EXT·pixel + W_PRIOR·(priors) + penalty`。全部权重
  取 1.0 ⇒ 与 glafic 原生 χ² 严格一致。最优解的分量拆解会打印并存档
  （`status.json → components`）。
- FITS 图像的尺寸必须与 `xmin/xmax/ymin/ymax/pix_ext` 隐含的网格严格一致
  （例如 ±0.5″ 视场、`pix_ext = 0.01` 需要 100×100 的图像）——不一致立即报错。
  图像还必须是 **32 位浮点** FITS（BITPIX −32）：其他像素类型会触发 glafic 的
  `input obs fits must be float` 而中止——写文件前先 `data.astype(np.float32)`
  转换。
- 点源约束（例如宿主星系环前的透镜化超新星）有两个入口：四个观测数组（内部
  转换成临时的 glafic 约束文件）或 glafic 原生 `constraint_file`（两者都给时
  以后者为准）。此模式下 `source_x/y` 由 glafic 逐候选*内部*求解——它们不是
  DE 维度。
- `hubble = {lo, hi}` 在这里**有意义**（时间延迟 ∝ 1/h）——这就是时延宇宙学
  路径。
- 后端：CPU 逐候选驱动 glafic（多进程）；GPU 在符合条件时对整个种群批量求值
  （恒为 fp64）——存在点源约束时额外要求 `chi2_splane = 1`（源面点 χ²）。
  Glafic/amoeba 按钮拒绝延展源配置。
- 延展源路径同样支持 MCMC（注意上文的速度警告）。

## 7.7 多进程注意事项

- Linux/WSL2 用 `fork` 工作池；macOS 用 `spawn`（结果一致——DE 轨迹由种子唯一
  决定）。
- `DE_WORKERS = -1`（全核）是默认值，对 DE 是安全的。
- `MCMC_WORKERS` 有意默认 **1**：在*脱离/后台*终端里，父进程死掉后全核 fork
  池可能留下满负荷空转的孤儿进程。只有在真正的前台终端里才设
  `MCMC_WORKERS = -1`（WebUI 弹出的终端窗口算前台——但若你的任务落在无窗口的
  detached 回退模式，请保持 1）。

## 7.8 独立验证：`glafic_verified` 与 scipy 精确参考

`glafic_verified = True`（默认）时，每个 DE/MCMC 中位数结果都会被独立复核，
日志明确写着：`(verification is informational — the result above is
unchanged)`（验证仅供参考——上面的结果不受影响）。

1. **glafic 二进制交叉核对**：把最优模型写成 `glafic_verify.input`，交给
   *独立的二进制程序*求解（这条代码路径既不经过 Python 绑定、也不经过 GPU）。
   报告：glafic 的像数、它对观测的损失、以及像位置的最大偏差（mas）。
   - 相对损失差超过 50 % 会触发一条自带解释的警告：对 Sersic 类轮廓，*二进制*
     的偏折受 Romberg 容差限制（第 10 章）——这种差异在那里是预期的，**不**表示
     你的拟合有错。`glafic found N image(s); the result assumes M` 之类的警告
     通常意味着临界线附近多出了一个边缘像。
2. **scipy 精确参考（基准真值）**：在观测像位置计算精确偏折（Sersic 用容差
   10⁻¹¹ 的自适应积分；其他模型用 fp64 的 Rhongomyniad 内核）。报告：Rhongomyniad
   （GPU 内核）的 Sersic 偏折相对精确积分的误差（预期 ~10⁻⁹ 角秒；无论哪个
   后端跑的拟合，这项检查都用 fp64 GPU 内核计算——CPU 引擎自己的 Sersic 偏折受
   Romberg 容差限制，见第 10 章）、**源面自洽散布**（mas）——观测像在
   精确模型下反投影回源面的收敛程度，是拟合物理自洽性的品质因子——以及反投影源
   位置与拟合源位置的对比。需要 torch；缺失时打印一行 `[info]` 跳过。

延展源运行的验证类似（`glafic_extverify.input`，二进制的 `c2calc` 总值 vs
GLADE 的总值，相对差 > 5 % 警告）。

更深入的独立核查工具：`tools/verify_gpu_models.py`（每个 GPU 内核 vs glafic）
和 `tools/verify_gpu_precision.py`（64/48/32 三档 vs scipy 精确值）。

## 7.9 分轮精调：`fine_tuning`

组件很多的模型——宏观透镜再加若干子结构团块——常常让单次全局搜索力不从心：
子结构的维度会把宏观盆地淹没。`fine_tuning` 键（§6.9）把一次性搜索换成三轮：

1. **宏观。** 删掉所有子结构组件，只搜索用户设为可优化的主透镜 + 源参数。取
   `fine_tuning_top_k` 个彼此不同的最优盆地（至少一个维度相差
   ≥ `fine_tuning_diversity` 倍区间宽度）各自作为独立链的种子。子结构被移除后，
   宏观模型形成的像可能比观测少——若第 1 轮总被硬性拒绝清空，考虑
   `missing_img_penalty > 0`（或第 1 轮 `B1 = 0`）。
2. **子结构。** 每条链把宏观参数（含源）冻结在种子上，只拟合子结构的
   `{lo,hi}` 参数。
3. **联合抛光。** 先淘汰比最优差 10× 以上的链；幸存者把*所有*透镜/子结构/源
   参数——包括用户原本固定的——在 `value·(1 ± perturb)` 窄盒内重新放开并抛光。
   窄盒带护栏：原本可优化的参数会与你的原始 `{lo,hi}` 求交，并且一律钳制到
   引擎的硬性定义域（椭率 < 1、Sersic n ∈ [0.06, 20]、幂律 γ ∈ (1, 3)）；钳制
   后塌缩的盒子让该参数保持固定。抛光只会让链变好——现任解会按第 3 轮目标重新
   打分，打不过就保留现任解（被跳过的链同样重新打分，保证所有链的最终损失
   可比）。损失最低的链胜出（第二名若在 2× 以内会被点名：数据可能区分不了这
   两个解）。

每一轮可以指定自己的算法（`DE` / `BIPOP-CMA-ES` / `jSO`——不含 amoeba）和损失
权重 `AN`/`BN`；预算仍来自各算法的 `DE_*`/`CMAES_*`/`JSO_*` 键。各阶段结果存档
在 `ft_round1/`、`ft_round2_chain<n>/`、`ft_round3_chain<n>/`——各含
`best_params.txt`（用该轮自己的 `AN`/`BN` 口径）和一份可单独重跑该阶段的
`glade_output_*.dat`（带显式 `Nl`/`Ns` 后缀、不含 `fine_tuning` 键）。胜出者的
最终损失（`status.json`、`best_params.txt`、三联图、glafic 验证）会换算回你
`.dat` 自身的 `LOSS_COEF_A`/`LOSS_COEF_B` 口径，读起来和普通运行没有两样；
MCMC 仍在用户*原始*的 `{lo, hi}` 区间上采样（种子会被裁剪进区间内）。前提条件
不满足时（没有主透镜、没有子结构、第 1 或第 2 轮无可优化参数、FITS 延展源模式、
共享变量横跨透镜与子结构），GLADE 会警告并回退为普通单次运行。两条备注：这三
个键名（连同大写别名）自 V0.7.1 起为保留名，老 `.dat` 若把它们用作自定义变量
必须改名；Python 侧请用 `glade.run_fine_tuning(cfg, backend=...)`——普通
`glade.optimize()` 只跑单阶段，遇到激活的 `fine_tuning` 键只会给出警告。

---

# 8 读懂运行输出

## 8.1 运行目录

每次运行——不管来自 WebUI 还是命令行——都写进一个目录 `runs/<job_id>/`：

| 文件 | 何时产生 | 内容 |
|---|---|---|
| `job.log` | WebUI 运行 | 完整的终端记录（由 WebUI 的终端包装以 `tee` 写出；无头命令行运行需自行重定向输出） |
| `status.json` | 总是 | 机器可读的状态 + 结果（§8.2） |
| `best_params.txt` | DE / 延展源 / amoeba | 最优拟合参数（§8.3） |
| `result.png` | 成功时 | 三联图（点源运行，§8.4）、观测/模型/残差图（延展源运行，§8.5）或 amoeba 三联图 |
| `mcmc_corner.png`、`mcmc_trace.png`、`mcmc_summary.txt` | 跑了 MCMC | 后验图 + 百分位表（§8.6） |
| `iterations/iteration_%04d.png` | `Draw_Graph = 1`（默认） | DE 种群角图帧（§8.7） |
| `best_crit.dat` / `best_ext_crit.dat` | 画图时 | 图中用到的 glafic 临界线线段 |
| `glafic_verify.input`、`glafic_verify_point.dat`（延展源：`glafic_extverify*`） | `glafic_verified = True` | 验证产物（§7.8） |
| `amoeba_model.input`、`amoeba_obs.dat`、`amoeba_prior.dat`、`<prefix>_optresult.dat`、`<prefix>_point.dat`、`<prefix>_crit.dat` | Glafic 按钮 | 转换/装载的 glafic 输入及 glafic 自己的输出 |

成功运行的 `job.log` 最后一行是 `RUN_COMPLETE`。

## 8.2 `status.json`

字段随运行推进累积：`state`（`starting → running → done | error`；WebUI 另外会
合成 `interrupted`——工作进程已死，和 `unknown`——服务器重启过）、`backend`、
`mode`、`files`、`worker_pid`；DE 之后：`loss`、`iterations`、`triptych`、
`fitted`（`标签 → 物理值` 的映射——这里的质量是**线性值**）；MCMC 之后：
`mcmc {acceptance, n_samples, corner, trace, summary}`；延展源运行另有
`c2calc_total` 和八分量拆解 `components {pos, flux, td, prior_pt, pixel,
prior_ext, prior_lens, penalty}`；验证补充 `glafic_verify` + `scipy_reference`
（或 `extend_verify`）。

## 8.3 `best_params.txt`（以及数字都在哪）

点源 DE：

```
# GLADE DE result  backend=gpu  loss=0.50547131
point1.mass = 1558.908622
point1.x = 0.2819156363
...
```

每个拟合维度一行 `标签 = 值`，用**物理单位**（质量是线性的，不是对数）。
延展源运行在注释里追加 `c2calc` 总值和分量拆解。amoeba 运行不一样：
`best_params.txt` 只有 `chi^2 = …` 和像列表——拟合出的透镜参数在 glafic 自己的
`<prefix>_optresult.dat` 里。纯 MCMC 运行完全不写 `best_params.txt`
（看 `mcmc_summary.txt`）。

> ⚠️ **对数质量 vs 线性质量。** `best_params.txt` 和 `status.json → fitted`
> 存的是 M☉（线性）。`mcmc_summary.txt` 和角图坐标轴用的是*搜索空间*，即
> **log₁₀(质量)**；线性中位数在 `status.json` 里叫 `p50_linear`。对比时别
> 搞混。

## 8.4 `result.png` —— 三联图（点源运行）

三个面板；标题带损失值（`GLADE DE result loss=…`、`MCMC posterior-median
model` 或 `glafic amoeba result chi²=…`）。

**左——"Position residuals"（位置残差）。** 每个观测像一根柱：观测位置与模型
匹配像之间的距离 ΔPos [mas]。蓝色虚线是你的 1σ 位置误差（各像 σ 不同则逐柱
画短线）。好的拟合每根柱都低于自己的 1σ 线。

**中——"Magnification"（放大率）。** 每个像三个量：天蓝色柱 = 观测 |μ| 带误差
棒；斜纹绿柱 = |μ_pred|，模型在其自己预测像位置上的放大率；红点 = |μ@obs|，
模型*恰好在观测位置上*的放大率。临界线附近 μ 对位置极其敏感，好模型的 |μ@obs|
也可能和 |μ_pred| 差很多——先比较绿柱和蓝柱，把红点当灵敏度诊断。
`abs_mag = False` 时全部改为带符号显示（负柱 = 鞍点宇称的像）。

**右——"Image plane"（像面）。** 金色星 = 观测像（带编号）；红叉 = 模型像；
蓝色曲线 = **临界线**（像面上形式上放大率无穷的轨迹）；绿色曲线 = **焦散线**
（其在源面的对应，画在同一坐标里）；红色菱形 = **子晕标记**及参数标签
（`S1: 1.0e+06 …`——组件质量和形状参数）。

哪些组件会有子晕标记：被归为子结构的——你写了 `Nl`/`Ns` 后缀就按后缀（§6.3），
否则按模型类别；另外任何带可优化参数的组件默认都会被标记。

若最优模型无法复现观测像数，图会被跳过并记录 `[warn] triptych failed …`——
运行本身仍算 `done`。

## 8.5 `result.png` —— 延展源图

三个面板：**Observed**（你的 FITS）与 **Model (best fit)**（透镜化的模型像）
共享同一亮度标尺；**Residual (model − obs)** 用自己的对称发散标尺
（± max|残差|）——残差面板里的结构就是模型没解释掉的部分。临界线以青色叠加。标题第二行列出 χ² 分量。观测 FITS
读不出来时只画模型面板。

## 8.6 MCMC 图与摘要

**`mcmc_corner.png`** —— *角图*：所有拟合维度的 N×N 矩阵；对角线是每个参数的
一维后验直方图（虚线标出第 16/50/84 百分位——中位数 ± 1σ）；非对角面板是参数
两两的联合二维分布，倾斜/弯曲的等高线揭示参数简并（如质量–聚度）。质量轴标注
为 `log10(...)`。`de+mcmc` 运行中红线标出 DE 最优点——它应当落在后验主体之内。
（样本超过 40 000 时绘图前随机抽样到 40 000。）

**`mcmc_trace.png`** —— 每个参数一行；所有 walker 的取值随步数的轨迹，红色
虚线标 burn-in 结束处。健康：红线之后是平坦、混合良好的"毛毛虫"。趋势性漂移
或分裂成带 ⇒ 链未收敛——加大 burn-in/步数或收窄区间。（显示上限 2000 步 /
256 个 walker。）

**`mcmc_summary.txt`** —— 表头含后端和接受率，然后每参数一行：
`name = p50 [p16, p84]`——后验中位数和 ±1σ 区间，**在搜索空间中（质量是
log₁₀）**。

退化的链（接受率 ≈ 0）可能画不出角图；运行会继续并记录
`[warn] corner plot skipped: …`——这种运行的"后验"应视为无效（§7.5.4）。

## 8.7 迭代帧

`Draw_Graph = 1`（默认）时，每 `draw_interval` 代写一张
`iterations/iteration_%04d.png`：*整个 DE 种群*在所有维度两两组合上的角图式
散点，按损失着色（以色标为准——损失越低越好），当前最优标 `+`。凡是某组件 x/y 维度配对的面板
都会把观测像位置叠加成金色星——可以直观看到子晕候选随代数聚拢到像的位置上。
WebUI **不**显示这些帧；去磁盘 `runs/<id>/iterations/` 浏览。想省一点开销可设
`Draw_Graph = 0`（19 维时约 2.7 秒/帧）。

---

# 9 命令行与 Python 库

WebUI 能做的一切都可以脚本化。三个层次：无头任务运行器（§9.1）、
`import glade` 库（§9.2–9.3）、原始引擎（§9.4）。

## 9.1 无头运行：`webui/runjob.py`

WebUI 的工作进程可以直接使用：

```bash
source env.sh
python webui/runjob.py \
  --backend cpu \                     # cpu | gpu | glafic
  --mode findimage \                  # findimage | de+mcmc | mcmc | amoeba
  --out runs/manual_cpu \
  --files core/examples/constants.dat \
          core/examples/images_data.dat \
          core/examples/lens_and_substructure.dat \
  --force
```

- `--mode findimage` = 只跑 DE；`de+mcmc` = DE 后接 MCMC；`mcmc` = 纯 MCMC；
  `amoeba` = glafic 单纯形（按 WebUI 的映射惯例搭配 `--backend glafic`——该标志
  并不强制；amoeba 模式总是驱动 glafic 二进制）。
- WebUI 按钮与之对应：CPU/GPU → `findimage`/`de+mcmc`（取决于
  `MCMC_ENABLED`），MCMC → `--backend cpu --mode mcmc`，MCMC-GPU →
  `--backend gpu --mode mcmc`，Glafic → `--backend glafic --mode amoeba`。
- 退出码：成功 0（打印 `RUN_COMPLETE`），任何阻塞/错误 2。输出文件与第 8 章
  一致——唯 `job.log` 除外，它来自 WebUI 的 `tee` 包装：想要的话自己接管输出
  （`… | tee runs/manual_cpu/job.log`）。（`--force` 只为接口对称而保留；
  命令行运行不弹确认、直接采用默认值。）
- 先 `source env.sh`（或自行复刻它的 `PYTHONPATH`/`LD_LIBRARY_PATH`）。

## 9.2 `import glade` —— 库门面

仓库根在 `sys.path` 上时（工作目录是仓库时自动满足；或 `source env.sh` 后在
任何位置）：

```python
import glade

cfg, issues = glade.load_config(
    ["InputFiles/constants.dat", "InputFiles/images_data.dat", "InputFiles/lens.dat"],
    backend="gpu")
assert not glade.has_errors(issues)

result = glade.optimize(cfg, backend="gpu")            # DE 拟合
print(result.loss, result.fitted)                      # 物理值

obs = glade.build_obs(cfg)
glade.make_triptych(result, obs, "triptych.png")       # 结果图

from core.optimize.loss import LossConfig     # import glade 之后即可导入
mres = glade.run_mcmc(result.problem, obs, LossConfig.from_cfg(cfg),
                      backend="gpu", best_x=result.x,
                      mcmc_cfg=glade.MCMCConfig.from_cfg(cfg))
glade.plot_mcmc(mres, "out_dir")

report = glade.verify_with_glafic(result.scene, obs, "out_dir", opt_loss=result.loss)
```

`import glade` 会为整棵代码树引导 `sys.path`（此后 `import core`、
`import glafic`、`import rhongomyniad` 都可用），并且是*懒加载*的：重量级模块
（matplotlib、emcee、torch、glafic C 扩展）只在首次用到时载入。

函数速查（核心部分）：

| 函数 | 说明 |
|---|---|
| `load_config(paths, backend=None, with_defaults=True) → (cfg, issues)` | 解析 + 合并 + 补默认 + 校验；文件*内容*有问题不抛异常（语法/校验问题都变成 issues——用 `has_errors(issues)` 检查），但路径不存在/不可读仍会抛 `OSError`。`issues` 打印格式 `[error] file:line: message`。 |
| `lint_text(text, …) → (cfg 或 None, issues)` | 校验一份内存中的文档（等价于编辑器要跑的检查） |
| `optimize(cfg, backend="cpu", on_iteration=None, de_overrides=None, base_dir=None) → OptResult` | DE 拟合；延展源配置自动切到延展源路径（`base_dir` 用来解析相对 FITS 路径）。`de_overrides` 用 *DEConfig 属性名*（如 `{"maxiter": 300, "seed": 7}`），优先于 `.dat`。没有可优化参数时抛 `ValueError`。 |
| `OptResult` | `.x`（最优向量，**搜索空间**——质量为 log₁₀）、`.loss`、`.fitted`（标签 → 物理值）、`.scene`、`.problem`、`.de`（历史）、`.mode`（`"point"`/`"extend"`）、延展源另有 `.extend_components` |
| `build_obs(cfg) → ObsData` | 观测数据换算到引擎单位（角秒，已做翻转/平移） |
| `make_triptych(result, obs, output_file, …)` / `make_extend_figure(result, output_file, …)` | 第 8 章的图；`make_triptych` 在像数对不上时抛异常；两者都会在输出旁写少量 `best*` glafic 辅助文件 |
| `run_mcmc(problem, obs, loss_cfg, backend="cpu", best_x=None, mcmc_cfg=None) → MCMCResult` | `best_x=None` ⇒ 纯 MCMC（均匀初始化）；`best_x=result.x` ⇒ de+mcmc。`MCMCResult`：`.samples`、`.chain`、`.acceptance_fraction`、`.param_names`、`.summary`（p16/p50/p84（+`p50_linear`））。 |
| `plot_mcmc(mres, out_dir) → {"corner": 路径, "trace": 路径}` | 写两张图（某张失败则缺对应键） |
| `verify_with_glafic(scene, obs, out_dir, opt_loss=…) → dict` / `verify_extend(result, out_dir)` / `reference_check(scene, obs)` | §7.8；从不抛异常——检查 `["ok"]` |
| `engine(name)` | §9.3 |

库使用者的注意点：`load_config(..., with_defaults=True)` 才会给你文档所述的
默认值；底层 `DEConfig.from_cfg` 在缺 `DE_WORKERS` 时的回退值是 1（单进程），
而 WebUI 运行从默认表拿到的是 −1（全核）。

## 9.3 命令式引擎 API（`glade.engine`）

`glade.engine("cpu")` 返回 glafic C 扩展；`glade.engine("gpu")` 返回
`rhongomyniad`。两者暴露同一套 glafic 风格 API，探索性代码改一行 import 就能
互换：

```python
eng = glade.engine("cpu")                     # 或 "gpu"
eng.init(0.3, 0.7, -1.0, 0.7, "out", -5.0, -5.0, 5.0, 5.0, 0.2, 3.0, 5, verb=0)
eng.startup_setnum(1, 0, 1)                   # 1 个透镜, 0 个延展源, 1 个点源
eng.set_lens(1, "sie", 0.5, 300.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0)
eng.set_point(1, 2.0, 0.1, 0.05)
eng.model_init(verb=0)
images = eng.point_solve(2.0, 0.1, 0.05, verb=0)   # [(x, y, mag, td_days), ...]
eng.quit()
```

两个引擎共同的基本规则：编号从 1 开始；**每次修改参数后都要重新调用
`model_init()`**；模块是全局状态单例（每个进程同一时刻只有一个模型）；
`init(...)` 会重置一切。`point_solve` 返回带符号的放大率和以天为单位的时间
延迟（相对最早到达的像）。`calcimage(zs, x, y)` 返回八元组
`(αx, αy, td, κ, γ1, γ2, μ⁻¹, rot)`。

只有 glafic 才有的功能包括优化器（`optimize`、`optpoint`、`optextend`）、
`writecrit`/`writelens`/`writemesh`、`calcein_i`、`kappa_ave`/`kappa_cum`、
坐标转换和 `c2calc`/`c2calc_each`。glafic 绑定的两个怪癖：`init` 的参数要按
位置传（上游把第一个关键字拼错成了 `omgea`）；部分调用的越界错误会**直接杀死
整个 Python 进程**而不是抛异常。

## 9.4 直接使用 Rhongomyniad

`import rhongomyniad as rh` 可单独使用 GPU 引擎（参见 `Rhongomyniad/examples/`
和 `tests/test_smoke.py`）：

- `rh.supported_models()` —— V0.6 已是 glafic 全部 27 个模型。唯一的结构性
  限制：**单透镜面**（透镜红移差超过 10⁻⁶ 时 `model_init()` 抛异常）。
- 设备/精度：`rh.set_device("cuda"/"cpu")`（CPU 回退是静默的——觉得慢就查
  `rh.get_device()`），`rh.set_dtype(torch.float64/float32)`。
- 找像器：`rh.set_finder("adaptive")`（默认；glafic 四叉树的 GPU 移植）或
  `"uniform"`（最细层的稠密网格——参考模式；对 NFW 类模型灾难性地慢）。临界线
  极近处丢像时调大 `maxlev`。
- 张量批量参数：任何物理参数都可以是能和网格广播的 torch 张量（GLADE 的批量
  目标函数正是这么用的）。红移和 `zs_fid` 必须保持 Python 浮点数。注意
  **批量参数会跳过合法性检查**——调用方要自行过滤垃圾值。
- `gals` 星表：与 glafic 兼容——*当前工作目录*下的 `galfile.dat`（行格式
  `x y L [e pa]`），首次使用时惰性读取；或显式 `rh.set_galfile(path)` /
  `rh.set_gals(rows)` / `rh.readgals()`。星表数值有意量化到 float32，以与
  glafic 逐位一致。
- 若 `Rhongomyniad/rhongomyniad/_tab_cache/` 下自带的缓存被删，首次用
  `gnfw`/`ein` 会重建查找表（一次性等待）；`acnfw` 的 CSE 表**不可重建**——
  别删它。
- 未实现（会抛异常）：`set_psf`/PSF 卷积、`writeimage` 的噪声模拟（另外它与
  glafic 不同，返回 numpy 数组而不是写 FITS 文件）、点源红移参与优化。

## 9.5 互译器：`python -m core.translate.cli`

```bash
# glafic → GLADE
python -m core.translate.cli to-glade some_model.input -o InputFiles/imported
# （也接受调用 glafic.* 函数的 Python 驱动脚本）

# GLADE → glafic（多个 .dat 的合并规则与运行时一致；-o 是路径前缀）
python -m core.translate.cli to-glafic \
  InputFiles/constants.dat InputFiles/lens.dat InputFiles/images_data.dat \
  -o exported/run
```

`to-glade` 写出 `<名>_model.dat`（+ `<名>_obs.dat`）。需要知道的转换规则：
glafic 的优化标志变成**退化区间 `{v, v}`——必须手工拉宽**；观测位置从角秒换算
到 mas；**硬编码写入 `obs_x_flip = False` 和零 `center_offset_*`**（若原始
坐标是天球约定，要自己改回来）；延展源输入会生成延展源模式的 `.dat`，附上
`W_*` 权重，文件路径只保留文件名（FITS 文件要自己放到 `.dat` 旁边）。

`to-glafic` 写出 `<前缀>_model.input`；只要有可优化参数，还会写
`<前缀>_obs.dat` + `<前缀>_prior.dat` 并接好 `optimize` 命令——在该目录下用
纯 glafic 二进制即可直接运行。区间坍缩为代表值（质量取几何平均）；共享变量变
成 glafic 的 `match` 绑定；`hubble` 区间变成 `hvary 1` + 范围先验。数字按
`%.6e` 精度写出。

## 9.6 旧版入口（`main.py`、`tools/`）

`./run_glade.sh` → `main.py` 是 V0.4 之前的工作流：在 `main.py` 里改
`model_use`（`point_mass`/`nfw`/`king`/`p-jaffe`/`none`）和覆盖字典，它去重跑
`legacy/` 下归档的脚本、写进 `results/`。`tools/` 里的辅助脚本
（`run_glafic.py`、`drawgraph.py`、`mcmc_from_result.py`、`MCMC_GPU.py`、
`replot_mcmc.py`、`glafic_verify.py`）全都针对那个**旧版** `results/` 布局
（`*_best_params.txt` 命名、模型专用表头、iPTF16geu 硬编码）——**它们读不了
现代的 `runs/<id>/` 输出**，而现代运行也不需要它们（图、MCMC、验证都是内置
的）。面向现代流水线的两个例外：`tools/verify_gpu_models.py` 和
`tools/verify_gpu_precision.py`（§7.8）。`tools/inverse_cal.py` 是个自成一体
的正向计算器（编辑其 CONFIG 块使用），偶尔适合做快速的"如果这样会成什么像"
实验。

---

# 10 数值精度：`TOL_ROMBERG_JHK` 专题注释

之所以专门写这一章，是因为 glafic 里这一个编译期常数造成的困惑超过本项目任何
其他数值设置。如果你拟合没有闭式偏折的椭圆轮廓、渲染延展源图像，或者要在毫角
秒量级上下结论，请读完本章。

## 10.1 它是什么

对没有解析偏折公式的椭圆透镜模型，glafic 用 Romberg 数值积分计算
Schramm (1990) 的 **I/J/K 视线积分**。`TOL_ROMBERG_JHK` 就是这套积分的相对
误差容差——`glafic2/glafic.h` 里的一个 `#define`：

```c
/* glade local override: tightened from upstream 5.0e-4 for accuracy */
#define TOL_ROMBERG_JHK 1.0e-5
```

- **上游 glafic 出厂值是 5·10⁻⁴；GLADE 自带的编译版本是 1·10⁻⁵**（精度提高
  30–50 倍，同时远快于 V0.3 短暂用过的 10⁻⁸）。
- 受影响的模型（所有经过 J/K 积分的）：**`nfw`、`gnfw`、`hern`、`pow`、
  `sers`、`tnfw`、`ein`、`king`**。*不*受影响：`sie`、`point`、`anfw`、
  `acnfw`、各 `…pot` 变体等闭式/查表模型。
- **GPU 引擎完全不受影响**——Rhongomyniad 用固定节点的 Gauss–Legendre 求积，
  对这些轮廓比 glafic 二进制*更准*（其 Sersic 偏折与 scipy 精确积分吻合到
  ~10⁻⁹ ″）。

## 10.2 为什么重要：条纹伪影

在上游容差下，Romberg 误差并不平滑——它在内部细化边界处*跳变*，是一种随位置
突变的阶梯误差。两个与 GLADE 直接相关的后果（都在 `exception/stripe_repro/`
中复现并量化过）：

1. **延展源图像上的暗条纹。** 在 NFW+Sersic 系统的 `writeimage` 式透镜像里，
   glafic 用有限差分算放大率，会把偏折的阶梯误差放大约 500–1000 倍，渲染成
   **切过透镜环的竖直暗线**（"|o|"状图案）。实测条纹深度：5·10⁻⁴ 时为环亮度的
   **11.5 %**（清晰可见）；GLADE 的 1·10⁻⁵ 时 **0.4 %**（不可见）。雪上加霜的
   是：glafic 的 Romberg 例程关闭了收敛检查（16 层细化，不收敛也不吭声），
   所以没有任何警告。
2. **临界线附近的点源位置。** 偏折误差会被 |μ| 放大。5·10⁻⁴ 下计算的近临界像
   位置可偏差至 ~5 mas（放大率也明显偏）；1·10⁻⁵ 下精确到大约 µas–mas 过渡
   量级；10⁻⁸ 下与精确参考基本完全一致。

## 10.3 对你的实际意义

- **用自带的编译版本（默认）：什么都不用做。** 1·10⁻⁵ 的折中让条纹不可见、
  点位在常规工作中足够准。
- **以下情况你会看到伪影**：把 GLADE 的结果和*原版*上游 glafic 并排比较（它的
  延展源图像真的和 GLADE 的不一样——那是条纹，不是 GLADE 的 bug）；或者你为了
  提速自己放宽了容差。
- **Sersic 类模型的验证警告是预期现象。** `glafic_verified = True` 时，交叉
  核对可能报告很大的相对损失差，同时附带解释：glafic 的 Sersic 偏折"受
  Romberg 容差限制……该差异是预期的，不是结果错误"。请相信紧随其后打印的
  *scipy 精确参考*——那才是基准真值（§7.8）。
- **连 1·10⁻⁵ 都不够时**（临界线极近处的 µas 级位置结论）：优先用 **GPU
  后端**——它的求积与该容差无关且经过 scipy 验证——而不是把 glafic 重编译到
  10⁻⁸（受影响积分慢 2–4 倍）。

## 10.4 如何修改它（很少需要）

这个常数**没有 `.dat` 键、没有环境变量、没有 API**。修改方法：编辑
`glafic2/glafic.h`，然后**重新编译**——`cd glafic2 && make clean && make all`
（只改头文件不重编等于没改；历史上就发生过 `glafic.so` 还是旧值的乌龙）。
GPU 交叉验证用过的标准流程：改成 `1.0e-8`，`make python`，跑
`tools/verify_gpu_models.py --tol 2e-7`，再改回 `1.0e-5` 重编。

`glafic.h` 里几个相关但不同的容差（均保持上游原值）：`TOL_ROMBERG_GNFW
3.0e-4`、`TOL_ROMBERG_EIN 1.0e-3`（径向轮廓表）、`ULIM_JHK 1.0e-8`（积分下限
截断）。

> 文档陷阱：`Rhongomyniad/README.md` 和
> `Rhongomyniad/rhongomyniad/constants.py` 仍写着 `TOL_ROMBERG_JHK = 5e-4`——
> 两处都是过期的镜像文案，无任何实际作用；实际编译进去的值是
> `glafic2/glafic.h` 里的 `1.0e-5`。

---

# 11 故障排查与常见问题

**安装与启动**

- *依赖压缩包下载失败* → 脚本会指出文件名和目录；手工下载放进 `deps/src/`，
  重跑 bootstrap。
- *重建 venv 或升级 Python 后 `import glafic` 失败* → `.pth` 注册丢了；重跑
  bootstrap（会重新生成 `glafic_glade.pth`）。
- *`source env.sh` 之后随便一条命令出错终端就没了* → `env.sh` 在你的 shell 里
  设了 `set -e`（§3.1）；source 之后执行 `set +e`。
- *WebUI 端口被占* → `GLADE_PORT=8080 ./run_webui.sh`。

**运行任务**

- *`[blocked] the GPU backend needs PyTorch … importing torch failed`* → 在
  venv 里装 CUDA 版 PyTorch（§3.3），或改用 CPU/MCMC 按钮。
- *`[blocked] no optimizable {lo,hi} parameters.`* → `.dat` 里所有参数都被锁定
  了（glafic 导入之后很常见——导入产生的是 `{v, v}` 退化区间，要手工拉宽）。
- *没有终端窗口弹出* → 找不到可用的终端模拟器；任务照样在后台运行并流向浏览
  器。要窗口就 `sudo apt install gnome-terminal`（WSLg）。脱离模式下请保持
  `MCMC_WORKERS = 1`（§7.7）。
- *怎么停止一个运行？* → 在它的终端窗口按 `Ctrl+C`（或关窗口）；界面会显示
  `interrupted`。没有停止按钮。
- *终端面板显示 `stream error` / `流错误`* → 浏览器丢了事件流；任务不受影响。
  刷新也接不回来；去终端窗口或 `runs/<id>/job.log` 看。
- *重启 WebUI 后任务显示 `unknown`、图加载不出* → 任务注册表只在内存里。文件
  都还在 `runs/<id>/`，从文件系统打开即可。
- *默认值对话框列出一堆没见过的键* → 这些键回退到了内置默认；对照 §6.9，并
  记住有些默认值是 iPTF16geu 专属的（§6.5）——请显式设置。

**拟合不对劲**

- *损失卡在 1e15 不动* → 所有候选都被拒绝：区间里根本不存在像数正确的模型，
  或者（GPU）批量目标函数失败过一次、日志里有
  `[warn] batched GPU objective failed …`（OOM → 调小 `GLADE_GPU_CHUNK`）。
  可考虑 `missing_img_penalty > 0` 给 DE 一个梯度（§7.2）。
- *`MCMC cannot start: every initial walker has zero likelihood`* → 均匀初始化
  在高维盒子里找不到有效模型：改用 `de+mcmc`（walker 从 DE 最优点出发），或
  收窄 `{lo, hi}`。
- *MCMC 接受率 ≈ 0* → 不是有效后验（§7.5.4）：用 DE 播种、收窄区间、减少
  维度。
- *`[warn] triptych failed: best-fit model produced N image(s) (expected M)`*
  → 运行结束了，但最优模型的像数不对，所以跳过画图。通常就是拟合不好——调整
  区间——或者有个边缘像贴着临界线（调大 `maxlev`）。
- *glafic 验证警告 `glafic found 5 image(s); the result assumes 4`* → 临界线
  附近多出一个暗像；仅供参考（§7.8）。（措辞相近的
  `(expected M); skipping the result figure` 是上面那条跳过画图的另一条警告。）
- *加的设置没生效* → 先查拼写：未知键会被静默当作用户变量（§6.6）；再看
  `job.log` 里的 `[defaults]` 行。
- *GPU 运行比预期慢* → 找日志里的 `[warn] batched GPU objective unavailable …`
  （§7.3.2）——组件红移/`zs_fid` 参与优化或多透镜面都会关掉批量；同时确认
  `torch.cuda.is_available()`。

**编辑器与文件**

- *在编辑器里打开 FITS 全是乱码* → 二进制文件不可编辑；**千万别保存那个标签
  页**，否则磁盘文件会被破坏。
- *在哪上传文件？* → 没有上传功能；从文件系统复制进 `InputFiles/`（Windows 走
  `\\wsl$`），然后 `⟳`（§5.3.2）。
- *刷新后编辑内容丢了* → 编辑器没有自动保存、关闭页面也没有未保存提醒；勤按
  保存（§5.3.3）。

**Clave**

- *状态显示 `Mock mode`* → glafic 模块导入不了；CPU 结果是占位假数据。重编
  glafic（bootstrap）并重启服务器。
- *`GPU错误: GPU mode requires all lenses on the same redshift plane…`* → GPU
  引擎只支持单透镜面；把所有透镜放到同一红移，或用 CPU 模式。
- *预期的像没出现* → 可能落在自动确定的搜索框之外（§5.4.5），或者你输入的红移
  被悄悄替换了（z = 0 会被换成默认值）。

---

# 12 附录

## 附录 A —— 术语表

- **角秒（″）/ mas**：1″ = 1/3600 度；1 mas = 10⁻³ ″。GLADE 的观测像位置用
  mas，其余几乎都用角秒。
- **burn-in（预热）**：MCMC 开头被丢弃的步数——walker 还在从出发点向后验区域
  迁移。
- **焦散线（caustic）**：临界线映射到源面的曲线；源跨越焦散线时像的数目改变。
- **χ²（卡方）**：以误差归一的残差平方和；GLADE 所有损失函数的基本构件。
- **角图（corner plot）**：拟合参数的矩阵图——对角线是每个参数的一维直方图，
  非对角是两两联合分布；展示多维后验的标准方式。
- **临界线（critical curve）**：像面上放大率形式上发散的轨迹（det J = 0）；
  其附近的像极端且敏感。
- **DE（差分进化）**：基于种群的全局优化算法；§7.1。
- **椭率 e / 位置角 pa**：椭圆轮廓的形状参数；e ∈ [0, 1)，pa 单位度。
- **emcee / walker / stretch move**：GLADE 使用的仿射不变系综 MCMC 采样器；
  *walker* 是采样群中的一个成员。
- **匈牙利匹配**：最优一一配对算法（这里：总距离最小的观测像↔预测像配对）。
- **透镜面 / 多透镜面**：单面 = 所有偏折体同一红移；偏折体分布在多个红移上就
  是多面（GLADE 中仅 CPU 支持）。
- **log₁₀ 搜索**：质量类参数按其以 10 为底的对数来优化/采样，因为它们跨越
  多个数量级。
- **损失（loss）**：优化器最小化的标量；GLADE 的损失是带权 χ² 加惩罚项
  （§7.2、§7.6）。
- **放大率 μ / 宇称**：像的流量放大倍数；符号编码宇称（镜像取向）——鞍点像为
  负。
- **MCMC**：马尔可夫链蒙特卡洛——长期分布收敛于后验的随机游走采样；§7.4。
- **后验（posterior）**：给定数据（和先验）后参数的概率分布；MCMC 估计的
  对象。
- **先验（prior）**：见到数据之前的假设；GLADE 中永远是 `{lo, hi}` 均匀盒。
- **Romberg 积分**：逐级细化的数值积分方案；glafic 用它算椭圆轮廓积分
  （第 10 章）。
- **Schramm (1990) 积分**：把椭圆质量分布的势/偏折/Hessian 表示成的 I/J/K
  线积分。
- **时间延迟（td）**：各像之间的到达时间差 [天]；正比于 1/h，由此而来时延
  宇宙学（§7.6）。
- **三联图（triptych）**：GLADE 的三面板结果图（§8.4）。

## 附录 B —— 版本历史（浓缩）

| 版本 | 要点 |
|---|---|
| 0.1.0 | 原型：glafic + DE |
| 0.2.x | 第一版双语 WebUI；浏览器内参数编辑 |
| 0.3.0 | Rhongomyniad GPU 引擎（beta）、Clave 首次出现、glafic 验证工具 |
| 0.4.0 "ReUnit" | 统一的 `core/`、`{lo, hi}` `.dat` 格式、重写的 `webui/`（FindImage + Editor、Monaco、模板）、glafic↔GLADE 互译 |
| 0.4.1 | 批量 GPU DE/MCMC；纯 MCMC 按钮；MCMC 先验统一为 DE 区间 |
| 0.4.2 | `glafic_verified` 独立验证 + scipy 精确参考 |
| 0.4.3 | 自带 glafic 同步到上游 2.1.14（King → 模型 #27；本地修改保留） |
| 0.4.4 | 延展源（FITS）拟合（经 `c2calc_each`）；`W_*` 权重；可优化 `hubble`；`missing_img_penalty` |
| 0.4.5 | macOS 安装器 + fork/spawn 多进程层（未在真 Mac 上测试） |
| 0.5.0 | GPU/CPU 全对齐（24 个张量内核、延展源管线上 GPU）；广义全种群批量（~27 倍）；`gpu_precision` 64/48/32；用户共享变量；`abs_mag`；`Nl`/`Ns` 后缀；MCMC-GPU 按钮及 walker 自动调节 |
| 0.5.1 | 仓库清理 |
| 0.5.2 | Glafic 按钮 = 原生 amoeba；可直接优化的 glafic 导出（`setopt` + `optimize` + 约束/先验文件；共享变量 → `match` 绑定） |
| 0.5.3 | `.dat` 算术 + 观测像位置表达式（`img1_x ± …`） |
| 0.6.0 | GPU 27/27 透镜模型（`crline`、`acnfw`、`gals`）；Clave 并入为第三页签；`import glade` 库门面；WebUI 深/浅主题 + 中英切换；资源管理器复制/粘贴；Clave 像面板 |
| 0.6.0-GREY | **双语用户手册**——本手册及其英文版加入 `manual/`，经对照代码的对抗式核查；README 末尾新增 GREY 发布图；`update_en.txt` 回填缺失的 V0.6.0 条目 |

完整更新日志：`Update.txt`（中文）和 `update_en.txt`（英文；其 V0.6.0 条目为
浓缩回填）。

## 附录 C —— glafic、许可证与引用

GLADE 打包了本地修改版的 **glafic 2.1.14**（Masamune Oguri 著，GPLv3；上游：
<https://github.com/oguri/glafic2>）。本地修改：King (1962) 轮廓
（模型 #27，`king`）、`TOL_ROMBERG_JHK` 覆盖（第 10 章）、分项 χ² 接口
`c2calc_each`、重新生成的 Makefile（其中 `TOL_ROMBERG_JHK` 与 `c2calc_each`
两处在源码中带 `glade local` 标注；King 代码和 Makefile 没有标注——权威清单见
`Update.txt`）。glafic 官方手册——模型参数、
secondary 设置和文件格式的权威参考——随仓库提供：
**`glafic2/manual/man_glafic.pdf`**（纯文本版：`man_glafic.txt`）。

**用 GLADE 发表研究结果时**，请引用 glafic：

- M. Oguri, *PASJ*, **62**, 1017 (2010) —— 使用（含修改版）glafic 的必引文献。
- M. Oguri, *PASP*, **133**, 074504 (2021) —— 若使用了 `anfw` 或 `ahern`
  模型，请一并引用。

GLADE 本身采用 MIT 许可证；emcee、corner、scipy/numpy/astropy/matplotlib 如在
工作中起显著作用，也各有其引用要求。

## 附录 D —— 环境变量

| 变量 | 默认值 | 作用 |
|---|---|---|
| `GLADE_PORT` | 6017 | WebUI 端口（`GLADE_PORT=8080 ./run_webui.sh`） |
| `CLAVE_PORT` | 6019 | 独立 Clave 端口（`python -m clave`） |
| `GLAFIC_AMOEBA_TIMEOUT` | 3600 | Glafic 按钮的 amoeba 运行被强杀前的秒数；`0` = 不限 |
| `GLADE_GPU_CHUNK` | 32/128（启发式） | 批量 GPU 模式下每次 CUDA 调用的候选数；OOM 时调小，追求速度时调大（§7.3.2） |
| `GLADE_ROOT`、`GLAFIC_HOME`、`GLAFIC_PYTHON_PATH`、`GLAFIC_LIB_PATH` | 由 `env.sh` 设置 | 启动脚本和任务运行器使用的安装路径 |

## 附录 E —— 辅助文件格式

- **glafic 点源约束文件**（`constraint_file`；导出/amoeba 生成的
  `amoeba_obs.dat` / `*_obs.dat` 同格式）：表头 `1 <像数> <z_s> 0.0`，随后每像
  一行 `x y flux pos_sigma flux_err td td_err parity`（角秒/透镜坐标系；GLADE
  写出时 flux = |μ|、符号放在最后的 parity 列）。
- **glafic 先验文件**（`prior_file`、`*_prior.dat`）：形如
  `gauss lens 1 3 0.0327 0.00097`（高斯先验）或 `range lens 2 7 0.5 2.5`
  （范围；`param_no` 1 = z，2…8 = p1…p7）、`range hubble lo hi`，以及
  `match lens i j ii jj 1.0 0.0`（参数硬绑定——GLADE 共享变量的化身）。
- **临界线文件**（`best_crit.dat`、`<prefix>_crit.dat`）：每行 8 列——一段临界
  线线段（x1 y1 → x2 y2，第 0,1,4,5 列）及对应焦散线段（第 2,3,6,7 列）。
- **`galfile.dat`**（`gals` 模型用）：行格式 `x y L [e pa]`，允许 `#` 注释。
- **glafic 像列表**（`<prefix>_point.dat`）：表头 `n zs src_x src_y`，随后每像
  一行 `x y mu tdelay`。

---

*GLADE 手册 · V0.6.0 · 2026-07 生成 · 中文版（English edition:
`GLADE_Manual_en.md`）*
