# NFW+Sersic 扩展像条纹伪影 — 复现与根因（已确认）

输入：`exception/nfwsersic_lens_sersic_source.input`
（`anfw` 3.378e13 + `sers` 1.199e11，z_l=0.5；`sersic` 源 z=2，源位置 (0.02,0.02)；
场 ±2"，pix_ext=0.001 → 4000×4000）。glafic 2.1.14。

## 结论（一句话）
条纹是 **Sersic 椭圆偏转的 Romberg 数值积分**在自适应细化级别/分支边界处产生的
**位置相关阶跃误差**，被 `extend.c` 计算 Hessian 时的**有限差分（步长 pix_ext）放大 ~500–1000×**，
再经 `source.c` 的亚像素抗锯齿渲染成一条切割光环的暗线。
它**由 Romberg 容差 `TOL_ROMBERG_JHK` 控制**：原版 glafic 默认 **5.0e-4**（条纹深 ~11.5%，肉眼可见）；
GLADE 收紧到 **1.0e-5** 后条纹深 ~0.4%（≈30× 更弱，看不见）。

## 决定性证据（用户 ds9 观察吻合）
用户在 STOCK 图 x≈1294、y≈2350–2500 看到对称垂直黑线（`|o|` 形）。复现定量：
| 版本 | 条纹 x | 相对深度 | 可见性 |
|---|---|---|---|
| 1e-2（很松） | 多条 | — | 强（多条）|
| **5e-4（原版默认）** | **1295** | **11.5%** | **可见黑线（=用户所见）** |
| **1e-5（GLADE）** | 1294 | **0.38%** | 不可见 |
| 仅 NFW（解析 anfw） | — | ~0 | 完全干净 |
| 仅 Sersic | 同位置 | 有（但亮度低 ~50×） | 几乎不可察 |

→ NFW(anfw) 偏转是解析闭式（无 Romberg），故单 NFW 干净；单 Sersic 也有条纹但太暗；
合并后 NFW 提供明亮高放大率光环、Sersic 提供带状误差，叠加才出现"明亮的"条纹。

## 关键源码位置
- `mass.c:2203-2206` `kapgam_sers`：Sersic 偏转 = `ell_integ_j(...)`（Schramm J 积分）
- `mass.c:3072-3082` `ell_integ_j` → `gsl_romberg2(..., TOL_ROMBERG_JHK)`
- `glafic.h:375` `TOL_ROMBERG_JHK`（原版 5.0e-4 → GLADE 1.0e-5）
- `gsl_integration.c:116-143` `gsl_romberg2`：workspace 仅 `GSL_ROMBERG_N=16` 层，
  **且 `gslstatus` 返回值被注释忽略** → 不收敛时静默返回欠精度结果
- `extend.c:135-142` 用**有限差分**偏转场算 Hessian（步长 ≈ 2·pix_ext）← 放大器
- `source.c:243,262-265` `source_all`：用该 Hessian 做亚像素抗锯齿足迹 ← 渲染成条纹

## FITS / 图（在 ds9 里看）
- `lensed_both_STOCK_5e-4.fits` / `lensed_both_GLADE_1e-5.fits` / `lensed_both_1e-2_VERYLOOSE.fits`
- `lensed_SERSIConly_5e-4.fits` / `lensed_NFWonly_5e-4.fits` / `source_unlensed.fits`
- `diff_STOCK5e-4_minus_GLADE1e-5.fits`（隔离条纹）
- `STRIPES_userloc_zoom.png`（用户位置全分辨率特写，决定性）
- `STRIPES_tolerance_sweep.png` / `STRIPES_ring_segment_linear.png` / `STRIPES_log_STOCK_vs_GLADE.png`

## 给上游（Oguri）的建议（分层）
1. **即时**：提高 `TOL_ROMBERG_JHK` 默认值（5e-4→1e-5 或更紧）。GLADE 已这么做，条纹基本消失。
2. **根治（设计层）**：`extend.c` 不要用有限差分偏转算 Hessian——glafic 的 `kapgam_*` 本就
   解析输出 kappa/gamma；直接用解析 Hessian 可消除 ~1/h 放大，使残余积分误差不再被放大成条纹。
3. **诊断**：`gsl_integration.c` 不应注释掉 `gslstatus`；Romberg 不收敛时至少应告警
   （现在是静默返回欠精度值，正是位置相关误差的来源；也与给 Oguri 邮件里 TOL_ROMBERG_JHK 那个
   点源精度问题同源）。
