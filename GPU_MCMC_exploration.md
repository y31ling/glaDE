# GPU-MCMC 可行性探索报告（V0.4.6 调研，2026-06-11）

> 需求背景：「MCMC 是否可以利用 GPU 优化，GPU-MCMC 能否大幅提高 MCMC 的速度」。
> 本报告只做探索与实测基准，结论供后续决策；除 V0.4.6 已落地的
> extend-GPU 似然外，不包含额外代码改动。
>
> **落地状态（V0.4.7）**：FindImage 已新增 MCMC-GPU 档位（engine=gpu、
> mode=mcmc），批量似然可用时自动启用 emcee vectorize=True；
> 未显式设置 MCMC_NWALKERS 时 GPU 路径自动取 1024 walkers（见 §3 结论 2），
> 不可批量时打印回退警告。实现见 webui/app.py + webui/runjob.py。
> 测试环境：RTX 4080 SUPER 16 GB（fp64 = 1/64 速率的消费卡）、torch 2.6 cu124、
> 24 逻辑核 CPU、emcee 3.1.6。

## 1. 现状盘点

| 路径 | 现状 |
|---|---|
| 点源（全部可优化组件为点质量、源位置固定） | **已有** `BatchedGPULogProbability`：emcee `vectorize=True`，整个半族群一次 CUDA 批量 |
| 点源（其它配置） | 逐 walker 走 glafic（CPU fork 池）或逐候选 GPU（慢） |
| 扩展源 | V0.4.6 新增：`BatchedExtendGPULogProbability`（批量 GPU），CPU 路径为 fork 池逐 walker 跑 glafic（**每次调用重读 FITS**） |

**结构性事实（决定收益上限）**
- emcee 默认 StretchMove 按红黑半族群更新：vectorized 似然每次只收到
  `nwalkers/2` 个样本。默认 `MCMC_NWALKERS=32` ⇒ 每次 kernel 只有 16 个候选，
  GPU >97% 时间空转。
- 每个 GPU 批量调用有 ~40–70 ms 固定开销（triangle-test 网格、Python 去重循环、
  kernel 启动），必须靠大 batch 摊薄。
- fp64 超越函数（exp/pow/atanh）在消费卡上是 1/64 速率——这是 SIE/Sersic 等
  解析模型下 GPU 不敌 24 核 CPU 池的根本原因；计算中心 GPU（A100/H100,
  fp64=1/2）会完全改变这一对比。

## 2. 实测基准

### 2.1 点源批量似然（sers×2+SIE 固定主透镜 + 2 个可优化点质量，6 维）

| nwalkers/批 | GPU ms/walker | 对比 CPU 池 (11.5 ms/walker) |
|---|---|---|
| 16（默认 32 walkers 的半族群） | 4.44 | **2.6×** |
| 64 | 1.10 | 10× |
| 256 | 0.30 | 39× |
| 1024 | 0.11 | **~104×** |
| 4096 | 0.096 | **~120×** |

- CPU 基线：glafic 单进程 275 ms/walker（Sersic 主透镜的自适应 Romberg 占
  绝对大头）；24 核池 ≈ 11.5 ms/walker。
- GPU↔CPU log_prob 一致性：max rel 3.1e-6（受 glafic Romberg 1e-5 限制），
  -inf 掩码完全一致。

### 2.2 扩展源批量似然（IvyProject：SIE+2×extsersic，19 维，splane 点约束）

| nwalkers/批 | GPU ms/walker |
|---|---|
| 16 | 2.61 |
| 64 | 0.91 |
| 256 | 0.69 |
| 1024 | 0.67 |

- CPU：glafic 单进程 9.4 ms/walker；24 核池 ≈ 0.39 ms/walker。
- **该配置下 GPU 比 CPU 池慢 ~1.7×**（SIE 是最便宜的透镜模型 + 100×100 小网格，
  fp64 transcendental 墙生效）。一致性 max rel 5.5e-9。

## 3. 结论

1. **GPU-MCMC 能否大幅提速？——能，但有两个前提。**
   (a) walker 数量要大：32→1024+ 后点源路径可达 **~100–120×**（对 24 核池）；
   (b) 或者 CPU 端每次评估本身很贵（含 nfw/king/sers 等需 Romberg 自适应积分的
   模型时 CPU 单核 ~10–275 ms/walker，GPU 固定阶 GL-256 完全向量化）。
   两个条件满足其一，GPU 即明显占优。
2. **默认 32 walkers 几乎吃不到 GPU 收益**（半族群=16/批，仅 ~2.6×）。
   增加 walker 同时改善后验覆盖，对多峰透镜后验本就有益；emcee 的
   StretchMove 在 nwalkers ≫ ndim 时统计性质良好。
3. **便宜解析模型（纯 SIE/点质量小网格）+ 多核 CPU 时，CPU 池仍是合理选择**；
   GLADE 现在两条路径都有且结果一致（DE 轨迹同种子逐位一致），可按配置自由切换。
4. 进一步加速的候选手段（按性价比排序，未实施）：
   - `MCMC_NWALKERS` 调大（零代码成本，立刻见效）；
   - fp32 评估亮度剖面（glafic 内部本就以 float32 存储扩展像，对 chi2 的扰动
     ~1e-7，相当于再换 ~30–60× 的 fp64→fp32 吞吐）——需要作为显式开关；
   - torch.compile / CUDA Graphs 消除小 kernel 启动开销（点源批量的 70 ms
     固定开销可压到 ~10 ms）；
   - 全 GPU 化采样器（把 StretchMove 本身放上 GPU，省去每步 host↔device
     往返）：当前每步传输 ~nwalkers×ndim×8B，微秒级，**不是瓶颈**，收益有限；
   - 换 NUTS/HMC（numpyro）需要可微分似然——findimg 的离散像数与匹配
     不可微，不适用；emcee 的 ensemble 方案仍是正确选择。

## 4. 备注

- emcee 接口已天然支持批量（`vectorize=True`），GLADE 的 MCMC 主循环无需任何
  改动即可吃到上述收益——选择 GPU 后端 + 调大 `MCMC_NWALKERS` 即可。
- CPU extend 路径每次似然调用都让 glafic 重读观测 FITS（`readobs_extend`），
  这部分 I/O 在 GPU 批量路径中只发生一次（启动时缓存为张量）。
- `lambda_cosmo = {lo,hi}` 目前在两条路径上都不会成为采样维度（OptProblem 只
  暴露 hubble），会静默回退默认值——与 DE 行为一致，属既有限制，已另行上报。
