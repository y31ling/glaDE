# GLADE 微像盲区修复计划（`auto_check`）

**文档性质**：实施规格书。写给下一位跟进的智能体：请通读全文后按 §9 的顺序实施。
**日期**：2026-07-18；2026-07-19 按源码与归档测试勘误。**背景调查**：除明确标为“待复现”的项目外，本文实测数字均已在本仓库中独立复现并经对抗性审计核实（见 §1.2 复现方法与附录 A 模板）。

---

## 0. 执行摘要（先读这个）

**问题**：GLADE 点源路径的模型放大率取“求像器报告的单根 μ”。当一个致密扰动体（point / 小核 king / 极端参数 nfw 子结构）落在某个匹配像的 ~几 mas 内时，该像物理上分裂为多个微像（尺度约为扰动体 θ_E；本仓库归档案例的根跨度从亚 mas 到约 11 mas）。glafic findimg 与 GPU 主求像器的最细播种尺度均为 `dp_min = pix_poi/2^(maxlev-1)`；默认 `pix_poi=0.2 arcsec, maxlev=5` 时均为 `0.0125 arcsec = 12.5 mas`（GPU 实现见 [core/optimize/batched.py:289](core/optimize/batched.py#L289)）。求像器可能只为其中一个根产生种子；loss 随后拿这个单根 |μ| 对观测打分。而观测（SN 标准烛光光度）测的是 PSF（~50–100 mas）内**所有微像的总流量 Σ|μ|**。后果：优化器收敛到假的“翻宇称压暗”解——单根报 |μ|≈9 完美命中观测 9.1，真实混合流量却是 39–45（比不加扰动体的基线 22.7 还亮）。

**实测定罪证据**（`pix_poi=5e-4, maxlev=5`，即 `dp_min=0.03125 mas` 的缩放 findimg；vendored `glafic2/glafic`）：

| run | 名义解（单根 μ） | 真实微像结构 | Σ\|μ\| | 名义 loss | 物理重算 img4 项 |
|---|---|---|---|---|---|
| runs/iptf-nfw-pm-1234 | −9.09 | −9.09 / +16.16 / +15.79 / −0.18（4 根，间距 1.3–1.9 mas） | **41.2** | 10.71 | ≈852 |
| runs/iptf-sie-pm-1234 | −9.05 | −9.05 / +14.93 / +14.91 / −0.044 | **38.9** | 0.11 | ≈734 |
| runs/iptf-sie-king-1234 | −11.63 | −11.63 / +15.25 / +18.55（3 根） | **45.4** | 23.2 | ≈1089 |
| （对照）移除扰动体 | +22.67 单根 | 单根 | 22.7 | — | — |

合法解**零误伤**确认：runs/iptf-nfw-nfw-1234 与 -loose（贴像弥散 NFW 压暗鞍点 img1/img3）在 0.003 mas 缩放下每像单根——压暗鞍点不穿临界线、不产生微像，拓扑上安全。修复必须保持这类解不受影响。

**修复方案**（文献标准做法：Keeton 2003 astro-ph/0209040 的 μ_tot=Σ|μᵢ| 观测量；Bradač et al. 2002 的"网格细于扰动体 θ_E 的像数验证"协议）：

- **第一层**：`verify` 层微像审计——最终解出来后，对每个匹配像做局部细网格 findimg，报微像数与 Σ|μ|，用 Σ|μ| 重算 `physical_loss`，超阈值标 `fake_solution`。
- **第二层**：优化循环内触发式局部细解——仅当致密扰动体贴近匹配像时，对该像局部多根求解，Σ|μ| 入 loss。CPU / GPU **各自独立实现**。

**用户硬性要求（不可协商）**：
1. 新增隐藏配置变量 **`auto_check`，默认 `True`**；`False` 时两层全部旁路，行为与旧版 GLADE 逐位一致（bit-identical）。
2. **CPU 与 GPU 独立**：CPU-only 用户（无 GPU、可能未装 torch）必须获得完整修复。第二层 CPU 实现只准用 python-glafic 绑定 + numpy/scipy，禁止 import torch / rhongomyniad。
3. **不要修**以下两件事（用户明确裁定，属人为可规避/特性）：(a) 同名子结构 label 冲突导致 `problem.decode()` 丢参数（problem.py:191-193 一带）；(b) loss 系数不入档 status.json。`ml_loss` 系数可配置是 GLADE 的特性，保持原样。

---

## 1. 问题背景与证据细节

### 1.1 物理机制（为什么假解必然出现、为什么只坑极小值像）

某像处宏观局部本征值 λ_t=1−κ−γ、λ_r=1−κ+γ，μ=1/(λ_t·λ_r)。iPTF16geu img4 是近临界极小值像（λ_t=0.068，μ=+22.7）。贴像致密扰动体把 κ+γ 推过 1 → λ_t 变号 → 翻成鞍点。但 λ_t=0 是临界线：翻转必然生成闭合微临界圈，像只能以"极小-鞍点对"成对产生（Poincaré 指标守恒）——被匹配的 −9 鞍点必然带着亮微极小伴像（Schechter & Wambsganss 2002 定理：宏观极小值内必含 ≥1 个 μ>1 的微极小像）。带符号求和近似守恒（pm 解：+22.68 vs 基线 22.673），流量只是被重排,光度上 Σ|μ| 反而**升高**。鞍点压暗（加 κ 让 det 更负）不穿临界、无新像——所以合法。

因此，针对“致密扰动体 + 贴近匹配像”的触发式检查能覆盖本仓库已归档的这类假解，并让常态候选几乎不增加开销。这里的“覆盖”不是对所有模型的完备性证明：无法从 schema 提取尺度的成分、多扰动体共同作用和 fail-open 回退仍是明确边界（见 §6、§9）。

### 1.2 为什么调 `pix_poi` 没用（别走这条路）

点质量翻转无标度：所需剪切固定 → 微像间距 ∝ √M。质量是自由维度，优化器会把 M 滑到任何固定网格分辨率之下（GLADE 参数先验下限 `1e2 h^-1 M☉` 时，间距 ~0.1 mas，需 pix_poi≈1.6e-3、~1.5 万倍成本）。唯一物理截断在源尺寸（SN 光球 ~0.4 μas ↔ M~0.03 M☉），先验内够不到。**正确做法是让检查分辨率自适应扰动体 θ_E**——θ_E 从候选参数直接算得，因此“固定胞元数、尺寸随 θ_E 缩放”的局部盒在任何质量下都恰好罩住全部微像，成本恒定。

### 1.3 复现命令

任何时候可用附录 A 模板复现定罪证据：把 `runs/iptf-nfw-pm-1234/glafic_verify.input` 的 lens 块套进模板、盒心取 img4 位置 (0.1434, 0.2894)，运行 `glafic2/glafic <input>`，应得 4 根、Σ|μ|=41.2。

---

## 2. 根因（代码级）

1. glafic findimg：自适应网格把终端方格拆成三角形，先由三角形映射产生候选种子，再用 Newton 法抛光到 `max_poi_tol=1e-10`（[glafic2/point.c:431](glafic2/point.c#L431)、[glafic2/point.c:522](glafic2/point.c#L522)）。run 网格 `pix_poi=0.2, maxlev=5` 的最细尺度是 `0.2/2^(5-1)=0.0125 arcsec=12.5 mas`；隐藏微根常在三角形播种阶段就没有得到独立种子。末端去重并非“按胞元尺度合并”，而按 `Δθ²/|μ_i μ_j| <= 10·max_poi_tol²` 判定（[glafic2/point.c:554](glafic2/point.c#L554)）；默认有效尺度约为数个 `10^-6 mas`，不是本案 1–11 mas 真根消失的原因。
2. GPU 批量路径：主求像器的 triangle-seed 间距同样为 `dp = pix_poi/2^(maxlev-1)`（[core/optimize/batched.py:289](core/optimize/batched.py#L289)）= 12.5 mas，因此共享同一种粗播种盲区。
3. 下游全部"如实处理残缺输入"：[matching.select_images](core/optimize/matching.py#L14)（n_obs+1 丢中心像、超出拒绝）、Hungarian 匹配、[loss.ml_loss](core/optimize/loss.py#L89)（abs_mag 下 |−9.09| vs 9.1 → χ²≈0）。规则本身没错，是观测量定义错了。
4. 若求像器如实报 8 根，select_images 会拒绝该候选——所以修复后微像集群**不能进入全局像数**（见 §6.3），否则合法解也会被误杀。

---

## 3. 设计总览

```
auto_check = True (默认, 隐藏键)
│
├── 第二层 in-loop（优化循环内, 每候选）
│     触发检测(in-loop CPU/GPU 共用规则): 任一致密扰动体距任一匹配像 < R_trig?
│     ├─ 否(≫99% 情形) → 原路径, 逐位不变
│     └─ 是 → 对触发像做局部多根求解 → 该像 μ := Σ|μ|(集群) → 进 ml_loss
│           ├─ GPU: batched.py 内 torch 局部种子 + 现有 Newton 核 (§5a)
│           └─ CPU: python-glafic 绑定第二次 init/point_solve 缩放周期 (§5b)
│
└── 第一层 verify（run 结束, 一次）
      对每个匹配像: 无条件粗查；满足距离/尺度条件时追加精查
      (vendored glafic 二进制, 纯 CPU)
      → micro_audit 报告 + physical_loss + fake_solution 标志 (§4)

auto_check = False → 两层全旁路, 行为 = 旧版 GLADE (逐位一致)
```

### 3.1 `auto_check` 配置键接入

- `.dat` 键名 `auto_check`，布尔，默认 True。**必须在 `core/format/schema.py` 注册**（classify 为 algorithm 节；参考现有布尔键如 `EARLY_STOPPING`/`abs_mag` 的注册方式——本仓库新键不注册会被归类为 user variable 或报错）。
- 读取：`cfg.algorithm.get("auto_check", True)`。传播到三处：(a) `LossConfig.from_cfg`（或平行的小 config）→ objective/batched；(b) runner → `verify_with_glafic`；(c) MCMC 复用 objective, 自动继承。
- "隐藏"含义：不写进用户手册的常规键表即可，不需要其它特殊处理。

---

## 4. 第一层规格：verify 微像审计

**位置**：[core/verify.py](core/verify.py) `verify_with_glafic()`，在现有 glafic_loss 计算（L146-152）之后。建议把实现放进新模块 `core/micro_audit.py`，verify.py 只调用——第二层 CPU 版会复用其中的触发/尺度计算。

**算法**（每个匹配像至少运行一次小 glafic 粗查，满足精查条件时再运行一次）：

1. **扰动体关联**：从 `scene.components` 中找出具有可识别中心与尺度、且 `theta_scale <= 100 mas` 的全部成分，不按 locked/optimized 身份或成分序号分类；尺度上限会排除通常的主透镜尺度成分，但保留锁定的致密子结构。对像 i 记录距离最近者；仅当 `d < 15 mas` 时把它记为 verify trigger。无论有没有 trigger，下一步粗查都执行。
2. **两段式缩放 findimg**（用 vendored 二进制,`find_glafic_bin()` 已有）：
   - 粗查（每像无条件执行）：盒心=像位置，半宽 15 mas，`pix_poi=5e-4`；在 `maxlev=5` 时 `dp_min=0.03125 mas`。`pix_ext=盒宽/200`，并加 `outformat_exp 1`。
   - 精查（仅当 verify trigger 存在且其 `theta_scale < 0.2 mas`，即粗查最细胞元可能不够）：第二个盒以扰动体为中心，半宽 `max(20·theta_scale, 2·d)`，`pix_poi=theta_scale`，所以 `maxlev=5` 时 `dp_min=theta_scale/16`。两盒结果取并集，并按 auto_check 自己的 `theta_scale/10` 容差去重；这不是 glafic 默认的末端去重规则。
   - 输入文件生成：复用 [`_write_glafic_input`](core/verify.py#L45)，参数化 xmin/xmax/ymin/ymax/pix_poi/pix_ext/prefix + outformat_exp（需给该函数加可选参数或写个变体）。
3. **集群归属**：盒内所有根都属于像 i 的微像集群（盒半宽 ≤30 mas ≪ 像间距 ~300 mas，不会串台；实现时仍应断言每根到像 i 的距离 < 到其它匹配像的距离）。
4. **重算**：`sum_abs_mu(i) = Σ|μ_root|`；`centroid(i)` = 流量加权质心。用同一 `ml_loss` + 当次 LossConfig 重算 → `physical_loss`（mm[i]:=sum_abs_mu，delta[i]:=质心 vs obs 的 mas 距离）。
5. **输出**（挂在 verify 报告 dict 上，随现有机制进 status.json）：

```python
report["micro_audit"] = {
  "per_image": [ { "n_micro": int, "mu_single": float, "sum_abs_mu": float,
                   "centroid_shift_mas": float, "trigger": None | {"comp_index", "type", "d_mas", "theta_scale_mas"},
                   "roots": [[x, y, mu], ...] }, ... ],
  "physical_loss": float,
  "fake_solution": bool,   # 任一像 |sum_abs_mu - |mu_single|| / max(|mu_single|, 1.0) > 0.05
}
```
   `fake_solution` 为 True 时往 `report["warnings"]` 加一条醒目警告（说明名义 loss 不可信、physical_loss 才是物理值）。
6. **失败安全**：任何异常只记 warning、不抛出——维持 verify.py "never raises" 的既有约定。`auto_check=False` 时整段跳过。

当前 verify 是 **fail-open**：局部求解没有可用根时会写 warning 并保留主求像器的单根值；异常也只写 warning。该候选此时只是“审计未完成”，不能据此宣称已通过物理认证。

**依赖**：仅 vendored glafic 二进制 + numpy/scipy。天然 CPU-only，与 torch/GPU 无关。

---

## 5. 第二层规格：in-loop 触发式 Σ|μ|

### 5a. GPU 路径（core/optimize/batched.py）

1. 主求像与匹配完成后，由 `_losses_for_chunk()` / `_losses_checked()` 追加检查（[core/optimize/batched.py:616](core/optimize/batched.py#L616)）：
   - **触发检测**：逐候选解码 concrete scene，用共享 helper 计算 compact perturber，再对每个匹配像应用 `d < R_trig = 10·theta_scale + 2 mas`；把触发的 `(candidate, image)` 对汇成扁平批次。
   - **触发对局部解**：每对生成两组 21×21 种子——扰动体邻域覆盖 `±20·theta_scale`，像邻域覆盖 `±max(3d, 5·theta_scale)`，共 882 个种子。stage 1 用 `dt_grid` 做 10 次 Newton 与宽松反投影筛选，并按 `theta_scale/10` 合并代表根；stage 2 对少量代表根固定用 fp64 再做 3 次 Newton，以 `|β(θ)-β_src| <= 1e-8 arcsec` 严格验根，重新去重、检查宏像归属并计算 μ；最后令**该像 μ := Σ|μ|**。
2. **不改全局像数逻辑**：微像集群只替换该像的 μ 值,不增减 select_images 看到的像数。
3. **逐位一致保证**：`trigger` 全 False 或 `auto_check=False` 时不碰任何原张量运算——batched.py 文档承诺的 same-seed bit-identical 必须保住（有现成测试约定,见 §8 T6）。
4. 性能边界：触发对通常为 0；有触发时每对处理 882 个 stage-1 种子，代码按模型内存开销切片，再把少量代表根集中做 fp64 stage 2。该成本只出现在触发候选，不能笼统声称为零，但不会扩大到全场网格。
5. **失败回退**：当前 GPU 局部检查抛出异常时只打印一次 warning，并让该 chunk 回退到未审计的单根 loss；若某个触发对没有得到通过验根的根，也保留单根值。这是维持优化连续性的 fail-open 策略，不是 fail-closed 的科学保证。

### 5b. CPU 路径（core/optimize/objective.py + backends.py）——独立实现,禁 torch

现状：`Objective.evaluate_one()`（[objective.py:86](core/optimize/objective.py#L86)）调 `backend.compute_images(scene)` → `point_source_loss()`。`EngineBackend.compute_images()` 每次评估本来就是完整的 `init → set_lens → set_point → model_init → point_solve → quit` 周期（[backends.py:35-53](core/optimize/backends.py#L35-L53)）。

1. **触发检测**：numpy/scipy、无 torch（§6 共享模块），输入 = scene 的成分参数 + 主解的匹配像。
2. **触发时**：对每个触发像,用**同一 glafic python 绑定**再跑一个缩放周期——`m.init(...)` 时把 xmin/xmax/ymin/ymax 设为局部盒(与 §4 同样的两段式盒与分辨率规则)、pix_poi 设为自适应值,其余照旧,`point_solve` 取根。即每个触发候选多花 1–2 次普通候选评估的成本;非触发候选零开销。
   - 注意 init/quit 生命周期不能嵌套：主周期 `quit()` 之后再开缩放周期（compute_images 返回后、loss 之前做）。
   - 多进程 DE（fork pool）下这一切都发生在 worker 进程内、用 worker 自己的引擎实例，与现有 `core/parallel.py` 机制兼容,无需改动并行层。
3. **Σ|μ| 替换与集群语义**：与 GPU 版完全相同（§6.3）。
4. **实现落点建议**：`point_source_loss` 增加可选参数（或新包装函数 `point_source_loss_checked(images, obs, loss_cfg, scene, backend, auto_check)`），evaluate_one 按 auto_check 选择调用；旧签名保留,别的调用方不受影响。
5. **依赖红线**：本路径只准 import numpy/scipy/glafic 绑定。CI/测试须包含"torch 未安装也能跑通"（§8 T7）。
6. **失败回退**：binding 缩放周期失败或没有可用根时保留单根；checked loss 的外层异常也会回退普通 loss。当前 CPU 外层回退不打印 warning，因此“日志里没有 warning”不等于局部审计成功；最终 verify 只有在自身成功时才可能另行暴露该候选。

### 5c. MCMC

CPU MCMC 走 Objective、GPU MCMC 走 batched objective,两边自动继承,无需单独改。

---

## 6. 共享 helper 模块规格（`core/micro_audit.py`，numpy/scipy、无 torch）

### 6.1 致密尺度 `theta_scale(comp)`

- **point**：θ_E = sqrt(4GM/c² · D_ls/(D_l·D_s))。角径距离用 ~30 行 numpy/scipy 平坦 wCDM 数值积分实现（scene 里有 omega/lam/weos/hubble/z_l/z_s；**不得**用 rhongomyniad）。与 glafic 的质量定义对齐：point 的质量参数单位是 `h^-1 M_sun`，角径距离以 `Mpc/h` 表示，式中的 h 因子相消；可用一次 glafic 绑定 findimg 对拍单点验证。
- **king**：max(θ_E_point(M), rc)。
- **nfw / tnfw / 其它弥散剖面**：max(θ_E_point(M), 0.02 mas) —— 当前实现用同质量点透镜的 θ_E 作为保守尺度。原稿所称“NFW 角落解伴像在 4.7–6 mas”没有仓库内固定模型快照，故只能列为待复现观察，不能充当归档定罪证据；§8 T4 只验证 point-mass 缩放，也不构成 NFW 护栏测试。
- 下限护栏：theta_scale ≥ 0.02 mas（源尺寸/数值下限）。

### 6.2 触发规则（in-loop CPU/GPU 共用；verify 另行定义）

优化循环内对 §4.1 所定义的全部 compact perturber 使用 `d(comp, image_i) < R_trig = 10·theta_scale(comp) + 2 mas`，并取满足条件者中距离最近的一项；锁定的致密子结构也会被检查。verify 不以这个门槛决定是否审计：15 mas 粗盒对每像无条件执行，`d < 15 mas` 只决定 trigger 记录以及该扰动体是否有资格追加精查盒。

### 6.3 微像集群语义（三条铁律）

1. 集群内 Σ|μ| 作为该匹配像的模型放大率进 ml_loss（abs_mag 语义不变——Σ|μ| 本来就无宇称）。
2. 集群**不改变**全局像数：select_images / n_obs+1 / missing_img_penalty 的输入照旧是主求像器的像列表。
3. 无触发 / auto_check=False → 一切逐位不变。

---

## 7. 明确不做的事（用户裁定,实施者不得"顺手修"）

1. **不修** problem.py 同名 label 丢参数问题——用户认定人为可规避。
2. **不做** loss 系数入档 status.json——`ml_loss` 系数可配置是 GLADE 特性。
3. 不改 select_images / matching / ml_loss 的任何现有语义;不动 missing_img_penalty;不动 DE/MCMC 算法本身。
4. 历史 run 的批量回溯标记工具是可选加分项,非本次范围（手动用附录 A 模板即可复查）。

---

## 8. 验收测试（全过才算完成）

| # | 测试 | 预期 |
|---|---|---|
| T1 | 第一层跑 runs/iptf-nfw-pm-1234 的模型（从其 glafic_verify.input 读 lens 块构造 scene） | img4 处 n_micro=4、Σ\|μ\|≈41.2±5%、fake_solution=True |
| T2 | 同法跑 sie-pm / sie-king | Σ\|μ\|≈38.9 / 45.4，fake=True |
| T3 | 同法跑 iptf-nfw-nfw-1234 与 -loose | 每像 n_micro=1、physical_loss≈glafic_loss、fake=False（零误伤） |
| T4 | 尺度测试：先以原四根的流量加权质心作为宏像锚点；令点质量 M→M/100，并令扰动体相对该质心的位移→1/10（θ_E→1/10） | 审计仍报 4 微像；宏透镜 Jacobian 梯度使总放大率不严格自相似，当前实测 Σ\|μ\|≈34.4，测试接受 28–55，并要求 fake=True |
| T5 | 第二层 CPU：用 pm_nfw_4sub.dat 同 seed 重跑 DE（小预算即可） | 不再收敛到 \|μ₄\|≈9 的翻转解;触发候选的 loss 里 img4 项按 Σ\|μ\| 计 |
| T6 | 回归：auto_check=False 与旧代码同 seed 同配置 | loss 轨迹逐位一致;auto_check=True 但无触发配置（如纯弥散 NFW run）同样逐位一致 |
| T7 | CPU-only 环境：`pip uninstall torch` 或以 import 屏蔽模拟,backend=cpu 跑 T1+T5 | 全部通过,无 torch import 错误 |
| T8 | 第二层 GPU：gpu_precision ∈ {64, 48} 各跑 T5 的 GPU 版 | 同 T5;且 precision 语义不变 |

**当前覆盖状态（2026-07-19）**：T1–T4 已有直接归档/尺度测试；T6 只有 helper 级的无扰动/远扰动数值等同性与合成根替换测试；T7 只验证屏蔽 torch/rhongomyniad 后模块仍可导入并计算尺度。T5 的真实 CPU DE、T6 的同 seed 完整 loss 轨迹、T7 的真实 CPU backend 端到端运行，以及 T8 的 GPU micro-audit precision 回归仍未完成。因此上表中的 T5–T8 相应内容仍是验收目标，不能当作已经验证的结论。

**测试数据锚点**（本仓库内长期存在）：runs/iptf-nfw-pm-1234、runs/iptf-sie-pm-1234、runs/iptf-sie-king-1234、runs/iptf-nfw-nfw-1234、runs/iptf-nfw-nfw-1234-loose 各自的 `glafic_verify.input`（权威模型快照）与 `glafic_verify_point.dat`（当时单根输出）。

---

## 9. 实施顺序与风险

1. **先做第一层 + §6 共享模块 + T1–T4**（半天量级,~300-400 行,风险低）。这一步能让最终解接受独立局部审计并识别已归档假解，但尚不等于优化循环内已经安全。
2. **再做第二层 CPU（5b）+ T5/T6/T7**——CPU 版逻辑简单（二次绑定周期）,先落地可服务 CPU-only 用户并验证集群语义。
3. **最后第二层 GPU（5a）+ T6/T8**——风险点集中在与 bit-identity / gpu_precision / chunking 的交互,建议单独提交、单独验。
4. 风险提示：(a) glafic 缩放网格下 `pix_ext` 也要同步缩小,否则 ext 网格内存/耗时异常（经验值:盒宽/200）;(b) 缩放盒内可能出现属于**其它**观测像的根——按 §4.3 断言排除;(c) outformat_exp 需确认 vendored 版本支持（本次调查已实测支持）;(d) Newton 验根容差要用现有常量,别自造。

---

## 附录 A：缩放 findimg 输入模板（已实测可用）

盒心 (X0, Y0)=目标像位置,半宽 H,分辨率 P（默认 5e-4）。lens 块从目标 run 的 `glafic_verify.input` 原样拷贝。

```
omega      0.3
lambda     0.7
weos       -1.0
hubble     0.7
prefix     zoom
xmin       {X0-H}
ymin       {Y0-H}
xmax       {X0+H}
ymax       {Y0+H}
pix_ext    {2*H/200}
pix_poi    {P}
maxlev     5
outformat_exp 1

startup    {N} 0 1
lens       ...   （从 runs/<run>/glafic_verify.input 拷贝全部 lens 行）
point      0.409    {source_x}    {source_y}
end_startup

start_command
findimg
quit
```

运行：`cd <scratch> && /path/to/glade/glafic2/glafic zoom.input`,读 `zoom_point.dat`（首行=根数+源位置,之后每行 x y μ delay）。

## 附录 B：本次调查的关键实测存档

- pm 解微像（img4,模型帧）：(0.144739, 0.288520, +16.161)、(0.143441, 0.289413, −9.0945)、(0.141648, 0.289972, +15.790)、(0.142756, 0.288122, −0.179)。带符号和 +22.68 ≈ 无扰动基线 22.673。
- sie-pm：4 根,Σ|μ|=38.93;sie-king：3 根,Σ|μ|=45.43。
- 宏观基线（Lim NFW+stars,收敛网格）：μ = [−451.67, +19.198, −47.296, +22.689, 中心 +0.339]。
- 文献锚点：Keeton 2003 (astro-ph/0209040) §2.2 μ_tot 定义;Bradač+2002 (aah3340) 0.05 mas 网格像数验证协议;Metcalf & Madau 2001 有限源光线追踪观测量;Schechter & Wambsganss 2002 微极小像定理;PyAutoLens 文档 "Demagnified Solutions" 页（同类坑的软件级记载）。
