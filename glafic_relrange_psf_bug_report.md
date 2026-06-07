# glafic 上游 Bug 报告：`relrange psf` 先验读取未初始化变量

> 调查日期：2026-06-07 · 调查对象：上游 glafic2 v2.1.13 / v2.1.14（已随本次升级进入 glade）
> 文件：`glafic2/init.c`，函数 `parprior()`，`relrange` → `psf` 分支

---

## 1. 结论（TL;DR）

- **Bug 真实存在**，已在源码与编译两层确认。
- 位置：`init.c` 的 `parprior()` 中，`relrange psf ...` 这条先验的解析分支把两个数值读进了 `rat`/`sig`，**却把从未赋值的 `ral`/`rah` 写进 PSF 的相对范围数组**。`ral`/`rah` 是未初始化的栈变量 → 写入的是**垃圾值**。
- **引入版本**：v2.1.13（即 `relrange` 先验被加入的那一版）；v2.1.14 原样保留。v2.1.12 及更早没有此代码。
- **对 glade 的影响：零。** glade 从不发出 `relrange`（更不会发 `relrange psf`），唯一出现 "relrange" 的地方是 README 更新日志。该代码路径在 glade 的任何工作流里都不会被触发。
- **对上游/一般用户的影响**：当且仅当用户在 `parprior` 文件里写 `relrange psf ...` 时触发；触发后 PSF 参数的相对范围约束变成不确定值，**很可能导致 PSF 拟合非确定性地失败或约束被静默忽略**。属真实但极冷门路径的缺陷。
- **修复**：1 行——把该分支的 `sscanf` 目标从 `&rat, &sig` 改为 `&ral, &rah`。

---

## 2. 缺陷代码（`glafic2/init.c`，`parprior()`）

变量声明（init.c:937）：

```c
double xx, yy, min, max, med, rat, sig, ral, rah;   /* 全部是局部、未初始化 */
```

`relrange` → `psf` 分支（init.c:1123–1130）：

```c
if(strcmp(ptype, "relrange") == 0){
  if(strcmp(keyword, "psf") == 0){
    nn = sscanf(buffer, "%s %s %d %d %lf %lf", ptype, keyword, &j, &jj, &rat, &sig);  // ← 读进 rat, sig
    if(nn != 6) terminator("input file format irrelevant (parprior)");
    if((j > NPAR_PSF) || (j < 1) || (jj > NPAR_PSF) || (jj < 1)){ terminator("psf id irrelevant (parprior)"); }
    para_psf_reraj[j - 1] = jj - 1;
    para_psf_reral[j - 1] = ral;   // ← BUG：ral 在本分支从未被赋值（未初始化栈值）
    para_psf_rerah[j - 1] = rah;   // ← BUG：rah 同上
    n++;
  } else {
    nn = sscanf(buffer, "%s %s %d %d %d %d %lf %lf", ptype, keyword, &i, &j, &ii, &jj, &ral, &rah);  // 正确：读进 ral, rah
    ...
    para_lens_reral[i-1][j-1] = ral;   // lens/extend/point 分支用的就是 ral/rah（正确）
    para_lens_rerah[i-1][j-1] = rah;
  }
}
```

`sscanf` 把行里两个浮点字段写进了 `rat`、`sig`，随后却把 `ral`、`rah` 存进 `para_psf_reral/rerah`。在这个 `psf` 分支里，`ral`、`rah` **没有任何赋值语句**，因此其值是进入 `parprior()` 时栈上的残留内容——不确定值。

---

## 3. 为什么能确定这是 Bug（而非有意为之）

**对照同函数里写法正确的两个"兄弟"分支：**

(a) 同为 `relrange` 的 lens/extend/point 分支（init.c:1132）——`sscanf(..., &ral, &rah)` 读入 `ral/rah`，再存 `ral/rah`，**自洽**。

(b) 结构完全平行的 `match` → `psf` 分支（init.c:1066–1073）：

```c
if(strcmp(ptype, "match") == 0){
  if(strcmp(keyword, "psf") == 0){
    nn = sscanf(buffer, "%s %s %d %d %lf %lf", ptype, keyword, &j, &jj, &rat, &sig);  // 读 rat, sig
    ...
    para_psf_rat[j - 1] = rat;   // 存 rat（正确）
    para_psf_ras[j - 1] = sig;   // 存 sig（正确）
```

`match`/`psf` 用 `rat/sig` 读、用 `rat/sig` 存——自洽。`relrange`/`psf` 显然是从 `match`/`psf` **复制粘贴**而来：保留了 `&rat, &sig` 的 `sscanf`，但把存储改成了 relrange 专用的 `ral/rah` 数组与变量名，**漏改了 `sscanf` 的目标**。这是典型的复制-改名疏漏，而非设计意图。

---

## 4. 为什么编译器没报警（缺陷为何"静默"）

即便用 `-O2 -Wall` 编译，gcc 也**不会**对 `ral`/`rah` 触发 `-Wmaybe-uninitialized`。原因：在同一函数的兄弟 `else` 分支里出现了 `sscanf(..., &ral, &rah)`，**取了 `ral`/`rah` 的地址**。一旦局部变量被取地址，编译器会保守地认为它"可能已被初始化"，从而抑制未初始化告警。

实证：本次完整重新编译 glafic（binary + lib + python 模块，`-O2 -Wall`）的日志中，`init.c` 没有任何关于 `ral`/`rah` 的告警（仅有与本问题无关的 `mcmc.c` `fscanf`、`point.c` `fptr` 既有告警）。这正解释了它为何能不被注意地随版本发布。

---

## 5. 触发条件

当且仅当 `parprior` 输入文件里出现如下形式的一行：

```
relrange psf <j> <jj> <lo> <hi>
```

即：用户用 `parprior` 命令，对**某个 PSF 参数**施加 **`relrange`（相对范围）**先验（让 PSF 第 j 个参数相对 PSF 第 jj 个参数取值）。

- lens / extend / point 的 `relrange` 不受影响（它们的分支写法正确）。
- `range`、`match` 等其它先验不受影响。
- 因此触发面 = `relrange` 新先验 ∩ `psf` 目标 ∩ `parprior` 用法，非常狭窄。

---

## 6. 运行期影响（一旦触发）

相对范围约束在优化中由 `opt_extend.c` 的 `check_para_psf()` 强制执行（opt_extend.c:384）：

```c
int check_para_psf(int j)
{
  double pl, ph;
  if((para_psf[j] < para_psf_min[j]) || (para_psf[j] > para_psf_max[j])){
    return 1;
  } else if(para_psf_reraj[j] != j){                       // relrange 已设置 → 进入此路径
    pl = para_psf_reral[j] * para_psf[para_psf_reraj[j]];  // 垃圾值 × 参考参数
    ph = para_psf_rerah[j] * para_psf[para_psf_reraj[j]];  // 垃圾值 × 参考参数
    if((para_psf[j] < pl) || (para_psf[j] > ph)){
      return 1;                                            // 判定越界 → 拒绝该模型
    }
  }
  return 0;
}
```

注意：`para_psf_reraj[j-1] = jj-1` 这一句**是正确设置的**，因此 `reraj[j] != j` 的哨兵会成立，相对范围路径**必然被进入**，从而**必然用到**垃圾的 `pl`/`ph`。后果：

- **最可能**：`pl`/`ph` 是无意义边界，几乎对所有试探值都判定"越界" → `check_para_psf` 恒返回 1 → 优化器把每个模型都当作非法 → PSF 拟合**无法推进 / 卡死 / chi² 持续被拒**。
- **或者**：若垃圾恰为极大/极端值，边界形同虚设 → 用户本想施加的 PSF 相对约束被**静默忽略**。
- 由于读的是未初始化栈内存，行为**不确定**：可能随运行、随编译器/优化级别、随此前栈上的内容而变，表现为**时好时坏、难以复现**的 PSF 拟合异常。

（严格地讲，读取未初始化的非静态局部变量在 C 标准下属未定义行为；在 x86 的 `double` 上通常表现为不确定值而非陷阱表示，但其"垃圾且不可复现"的本质不变。）

---

## 7. 严重性评估

| 角度 | 评估 |
|---|---|
| **对 glade** | **零影响**。glade 只发 `range` 先验（`tools/glafic_optimize.py` 写入 `range`），从不发 `relrange`/`relrange psf`；全仓唯一的 "relrange" 是 README 更新日志。缺陷路径永不触发。采用 v2.1.14 不会让 glade 暴露于此 bug。 |
| **对上游/一般用户** | 真实缺陷，但路径极冷门（对 PSF 参数施加相对范围先验）。一旦命中，静默破坏 PSF 先验、很可能令 PSF 拟合非确定性失败。 |
| **可见性** | 静默：无编译告警、无运行报错，只是结果错误/不稳定。 |

---

## 8. 建议修复（1 行）

把 `relrange`/`psf` 分支的 `sscanf` 目标改为它真正使用的变量：

```c
/* init.c:1124 —— 把 &rat, &sig 改为 &ral, &rah */
-   nn = sscanf(buffer, "%s %s %d %d %lf %lf", ptype, keyword, &j, &jj, &rat, &sig);
+   nn = sscanf(buffer, "%s %s %d %d %lf %lf", ptype, keyword, &j, &jj, &ral, &rah);
```

这样 `para_psf_reral/rerah` 就会拿到用户指定的 `lo`/`hi`，与 lens/extend/point 的 relrange 分支保持一致。（可选的防御性写法：在 init.c:937 把 `ral`/`rah` 初始化为 `-1.0e30`/`1.0e30`。）

### 处置选项（待你决定）
- **A. 保持与上游完全一致**（默认，符合"接受所有更新"）：不动这一行；它对 glade 无害，且让未来与上游再合并保持干净。建议向上游 `oguri/glafic2` 提交 issue/PR。
- **B. 本地顺手修掉**：应用上面 1 行补丁。安全、对 glade 行为零影响（glade 不走该路径），但会让该文件与上游产生 1 行偏差。
- **C. A + B**：本地修 + 同时上报上游（最稳妥）。

我可以按你的选择执行（应用补丁 / 起草上游 issue 文案）。

---

## 9. 实机复现补充（2026-06-07）

在用纯 glafic（不经 glade）构造真实触发场景时，确认了上面的 Bug A，并**额外发现了一个独立且更严重的崩溃 bug（记为 Bug B）**。两者根因不同，详见 `glafic_upstream_issue_draft.md`（已起草的上游 issue）。

### Bug A 的实机证据（"lo/hi 被忽略"）
用 `quadhost` 样例改造（仅 FWHM2 自由、`relrange psf 5 1 <lo> <hi>` 把 FWHM2 相对 FWHM1 约束；初始 FWHM2/FWHM1 ≈ 3.6）：

| 二进制 | `relrange psf` 区间 | 正确解释下初始是否在带内 | 结果 |
|---|---|---|---|
| 修复版 | `3.0 4.0` → [0.66,0.88] | 在（0.8∈带） | **正常运行** |
| 修复版 | `1.0 1.5` → [0.22,0.33] | 否 | 崩溃（经 Bug B） |
| 当前版(buggy) | `3.0 4.0` | — | 崩溃 |
| 当前版(buggy) | `1.0 1.5` | — | 崩溃 |
| 当前版(buggy) | `0.001 0.002` | — | 崩溃 |
| 当前版(buggy) | `1000 2000` | — | 崩溃 |

→ **修复版的行为随 lo/hi 改变（3-4 通过 / 1-1.5 拒绝），当前版对任意 lo/hi 行为完全相同** ⇒ 证明当前版把 lo/hi 丢弃（即 Bug A）。该证据与具体"垃圾值"无关，可稳定复现。

### Bug B（新发现，独立、更严重、且早于 2.1.13）
**现象**：扩展源拟合（`optimize`/`optextend`，含 `extend` 分量）时，只要任何 `parprior` 约束（`range` / `relrange` 皆可）拒绝了当前模型、使扩展源 χ² 未被计算，glafic 即 **SIGSEGV**。

**根因**：`chi2tot()`（`opt_lens.c:306`）在 `check_para_lens_all()`/`check_para_ext_all()` 处提前 `return chi2pen_range`（行 310/315），跳过 `chi2calc_extend()`（行 318）；而 `array_ext_mask` 仅在 `chi2calc_extend → ext_set_table()`（`extend.c:64`）中分配。随后 `opt_lens()` 的汇报代码在 `opt_lens.c:257` 仅用 `ne>0` 作守卫就解引用 `array_ext_mask`（此时为 `NULL`）→ 崩溃。

**范围**：用普通 `range` 先验即可触发；在 **v2.1.10** 上同样崩溃，故早于 `relrange` 特性，与 Bug A 无关。自包含复现（生成 mock 观测 → 用越界 `range` 先验拟合 → 崩溃）见 issue 草稿。Bug B 在当前版本里实际上**掩盖**了 Bug A（relrange psf 的垃圾边界通常会拒绝模型→走 Bug B 崩溃）。建议上游先修 Bug B。

**修复**：在 `opt_lens.c:257` 的读取处加 `array_ext_mask != NULL` 守卫（详见 issue 草稿）。
