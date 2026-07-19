"use strict";
// ===================== GLADE WebUI front-end =====================
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

// -------------------------------------------------------------------------
(function () {
  const SEEN_KEY = "glade_af_seen";           // flag: starts false (absent)
  let flag = localStorage.getItem(SEEN_KEY) === "1";
  const now = new Date();
  if (flag || now.getMonth() !== 3 || now.getDate() !== 1) return;

  const B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
  let addr = "1";
  for (let i = 0; i < 33; i++) addr += B58[Math.floor(Math.random() * B58.length)];

  const ov = document.createElement("div");
  ov.style.cssText =
    "position:fixed;inset:0;z-index:2147483647;background:#ff0000;color:#000;" +
    "display:flex;flex-direction:column;align-items:center;justify-content:center;" +
    "text-align:center;padding:6vw;gap:3vh;font-family:'Segoe UI',system-ui,sans-serif;" +
    "font-weight:800;cursor:none;user-select:none";
  ov.innerHTML =
    '<div style="font-size:clamp(24px,7vw,84px);line-height:1.1">Your PC has been hacked</div>' +
    '<div style="font-size:clamp(16px,3vw,32px);font-weight:700;max-width:26ch">' +
    "Transfer 1 BTC to the below address to unlock." +
    '</div><div style="font-size:clamp(13px,2.4vw,26px);font-family:Consolas,monospace;' +
    'word-break:break-all;background:#000;color:#ff0000;padding:.5em .8em;border-radius:6px">' +
    addr + "</div>" +
    '<div style="font-size:clamp(10px,1.4vw,14px);font-weight:400;opacity:.55;margin-top:2vh">' +
    "press any key to continue</div>";

  const show = () => {
    document.body.appendChild(ov);
    try {                                     // best-effort real fullscreen
      const r = document.documentElement.requestFullscreen();
      if (r && r.catch) r.catch(() => {});
    } catch (e) { /* no gesture: the overlay already fills the window */ }
  };

  const dismiss = () => {
    window.removeEventListener("keydown", dismiss, true);
    ov.remove();
    if (document.fullscreenElement && document.exitFullscreen)
      document.exitFullscreen().catch(() => {});
    localStorage.setItem(SEEN_KEY, "1");      // flag -> true: never again
    flag = true;
  };

  window.addEventListener("keydown", dismiss, true);
  if (document.body) show();
  else window.addEventListener("DOMContentLoaded", show);
})();
// -------------------------------------------------------------------------

// ===================== i18n =====================
const I18N = {
  en: {
    "nav.findimage": "FindImage",
    "nav.editor": "Editor",
    "nav.clave": "Clave",
    "nav.lang_title": "Switch language (English / 中文)",
    "nav.theme_title": "Toggle dark / light theme",
    "fi.backend": "Backend",
    "fi.mcmc_sub": "emcee only",
    "fi.mcmcgpu_sub": "emcee · batched CUDA",
    "fi.select": "Select .dat files",
    "fi.run": "▶ Run",
    "fi.term_title": "Terminal output",
    "fi.term_title_job": "Terminal output — job {0} ({1})",
    "fi.no_files": "No files in InputFiles/. Create some in the Editor.",
    "fi.none_selected": "No files selected",
    "fi.n_selected": "{0} file(s): {1}",
    "fi.run_failed": "Run failed",
    "fi.cfg_errors": "Cannot run — configuration errors",
    "fi.missing_basics": "Missing basic values — use defaults?",
    "fi.defaults_intro": "These basic variables were not provided and will use defaults:",
    "fi.defaults_q": "Continue with these defaults?",
    "fi.state_running": "running",
    "fi.state_done": "done",
    "fi.loss": "loss",
    "fi.iters": "{0} iters",
    "fi.mcmc_accept": "MCMC accept {0} ({1} samples)",
    "fi.stream_error": "stream error",
    "fi.fig_result": "Result",
    "fi.fig_corner": "MCMC corner",
    "fi.fig_trace": "MCMC trace",
    // --- V0.7 pipeline settings ---
    "fi.settings_title": "Backend / pipeline settings",
    "fi.settings_head": "Pipeline settings",
    "fi.sec_backend": "Backend",
    "fi.sec_purpose": "Purpose",
    "fi.sec_optimizer": "Optimizer",
    "fi.backend_cpu": "CPU",
    "fi.backend_gpu": "GPU",
    "fi.engine_glafic": "glafic",
    "fi.engine_rhon": "Rhongomyniad",
    "fi.hint_cpu": "glafic engine · multiprocess CPU",
    "fi.hint_gpu": "Rhongomyniad engine · batched CUDA (point-source)",
    "fi.purpose_calcimage": "Calcimage",
    "fi.purpose_optimize": "Optimize",
    "fi.purpose_mcmc": "MCMC-only",
    "fi.hint_calcimage": "Compute images at representative values, without optimizing.",
    "fi.hint_optimize": "Search the parameters to fit the observations.",
    "fi.hint_mcmc": "Sample the posterior only (no DE).",
    "fi.opt_amoeba": "amoeba",
    "fi.opt_de": "DE",
    "fi.opt_cmaes": "BIPOP-CMA-ES",
    "fi.opt_jso": "jSO",
    "fi.hint_amoeba": "glafic native downhill simplex.",
    "fi.hint_de": "Differential Evolution.",
    "fi.hint_cmaes": "BIPOP restart CMA-ES.",
    "fi.hint_jso": "jSO — adaptive Differential Evolution.",
    "fi.save": "Save",
    "common.refresh": "Refresh",
    "common.cancel": "Cancel",
    "common.ok": "OK",
    "common.save": "Save",
    "common.confirm": "Confirm",
    "common.delete": "Delete",
    "ed.explorer": "Explorer",
    "ed.template": "Template",
    "ed.explorer_head": "Explorer · InputFiles",
    "ed.templates_head": "Templates",
    "ed.new_file": "New file",
    "ed.new_folder": "New folder",
    "ed.delete": "🗑 Delete",
    "ed.save": "💾 Save",
    "ctx.open": "Open",
    "ctx.new_file": "New file…",
    "ctx.new_folder": "New folder…",
    "ctx.import": "Import glafic → glade…",
    "ctx.export": "Export glade → glafic…",
    "ctx.import_clave": "Import to Clave",
    "ctx.rename": "Rename…",
    "ctx.delete": "Delete",
    "ctx.delete_folder": "Delete folder",
    "ctx.copy": "Copy",
    "ctx.paste": "Paste",
    "ed.prompt_new_file": "New file name",
    "ed.prompt_new_folder": "New folder name",
    "ed.prompt_rename": "Rename",
    "ed.prompt_export": "Export to glafic — output name",
    "ed.confirm_delete_title": "Delete",
    "ed.delete_file_title": "Delete file",
    "ed.confirm_delete": "Delete <b>{0}</b>? This cannot be undone.",
    "ed.confirm_close_title": "Close without saving?",
    "ed.confirm_close": "<b>{0}</b> has unsaved changes.",
    "ed.imported": "Imported",
    "ed.wrote": "Wrote:",
    "ed.exported": "Exported to glafic",
    "ed.export_note_opt": "Found <b>{lo, hi}</b> parameters → added a glafic <code>optimize</code> command + setopt matrix, plus <code>readobs_point</code>/<code>parprior</code> files.",
    "ed.export_note_plain": "No <b>{lo, hi}</b> parameters → findimg-only model.",
    "ed.open_first": "Open or create a file first.",
    "ed.empty_hint": "Open a file from the Explorer, or insert a Template.",
    "ed.save_failed": "Save failed: {0}",
    "ed.import_failed": "Import failed: {0}",
    "ed.export_failed": "Export failed: {0}",
    "ed.paste_failed": "Paste failed: {0}",
    "ed.clave_failed": "Import to Clave failed: {0}",
    // --- V0.7 unit settings ---
    "ed.unitsetting": "Unit settings",
    "unit.title": "Unit settings",
    "unit.profile": "Profile",
    "unit.save_as": "Save as",
    "unit.new_profile": "new profile…",
    "unit.col_category": "Category",
    "unit.col_unit": "Unit",
    "unit.fixed_note": "Fixed units (not adjustable)",
    "unit.custom_note": "Custom {lo, hi} variables are dimensionless — they take the unit of wherever they are inserted.",
    "unit.cat.mass": "Lens masses",
    "unit.cat.obs_pos": "Observed image positions",
    "unit.cat.comp_pos": "Component positions & scale radii",
    "unit.cat.src_pos": "Source position",
    "unit.fixed.velocity": "Velocity dispersion",
    "unit.fixed.angle": "Angles",
    "unit.fixed.grid": "Grid",
    "unit.fixed.center_offset": "Center offset",
    "unit.fixed.time_delay": "Time delay",
    "unit.save_failed": "Save failed: {0}",
    "unit.name_required": "Please enter a profile name.",
  },
  zh: {
    "nav.findimage": "找像",
    "nav.editor": "编辑器",
    "nav.clave": "Clave",
    "nav.lang_title": "切换语言 (English / 中文)",
    "nav.theme_title": "切换深色 / 浅色主题",
    "fi.backend": "后端",
    "fi.mcmc_sub": "仅 emcee",
    "fi.mcmcgpu_sub": "emcee · 批量 CUDA",
    "fi.select": "选择 .dat 文件",
    "fi.run": "▶ 运行",
    "fi.term_title": "终端输出",
    "fi.term_title_job": "终端输出 — 任务 {0} ({1})",
    "fi.no_files": "InputFiles/ 中没有文件,请在编辑器中创建。",
    "fi.none_selected": "未选择文件",
    "fi.n_selected": "{0} 个文件: {1}",
    "fi.run_failed": "运行失败",
    "fi.cfg_errors": "无法运行 — 配置错误",
    "fi.missing_basics": "缺少基础变量 — 使用默认值?",
    "fi.defaults_intro": "以下基础变量未提供,将使用默认值:",
    "fi.defaults_q": "继续使用这些默认值吗?",
    "fi.state_running": "运行中",
    "fi.state_done": "完成",
    "fi.loss": "损失",
    "fi.iters": "{0} 次迭代",
    "fi.mcmc_accept": "MCMC 接受率 {0} ({1} 个样本)",
    "fi.stream_error": "流错误",
    "fi.fig_result": "结果",
    "fi.fig_corner": "MCMC 角图",
    "fi.fig_trace": "MCMC 迹线",
    // --- V0.7 管线设置 ---
    "fi.settings_title": "后端 / 管线设置",
    "fi.settings_head": "管线设置",
    "fi.sec_backend": "后端",
    "fi.sec_purpose": "用途",
    "fi.sec_optimizer": "优化器",
    "fi.backend_cpu": "CPU",
    "fi.backend_gpu": "GPU",
    "fi.engine_glafic": "glafic",
    "fi.engine_rhon": "Rhongomyniad",
    "fi.hint_cpu": "glafic 引擎 · 多进程 CPU",
    "fi.hint_gpu": "Rhongomyniad 引擎 · 批量 CUDA(点源)",
    "fi.purpose_calcimage": "计算像",
    "fi.purpose_optimize": "优化",
    "fi.purpose_mcmc": "仅 MCMC",
    "fi.hint_calcimage": "在代表值处直接计算像,不进行优化。",
    "fi.hint_optimize": "搜索参数以拟合观测。",
    "fi.hint_mcmc": "仅采样后验(不做 DE)。",
    "fi.opt_amoeba": "amoeba",
    "fi.opt_de": "DE",
    "fi.opt_cmaes": "BIPOP-CMA-ES",
    "fi.opt_jso": "jSO",
    "fi.hint_amoeba": "glafic 原生下山单纯形。",
    "fi.hint_de": "差分进化。",
    "fi.hint_cmaes": "BIPOP 重启 CMA-ES。",
    "fi.hint_jso": "jSO — 自适应差分进化。",
    "fi.save": "保存",
    "common.refresh": "刷新",
    "common.cancel": "取消",
    "common.ok": "确定",
    "common.save": "保存",
    "common.confirm": "确认",
    "common.delete": "删除",
    "ed.explorer": "资源管理器",
    "ed.template": "模板",
    "ed.explorer_head": "资源管理器 · InputFiles",
    "ed.templates_head": "模板",
    "ed.new_file": "新建文件",
    "ed.new_folder": "新建文件夹",
    "ed.delete": "🗑 删除",
    "ed.save": "💾 保存",
    "ctx.open": "打开",
    "ctx.new_file": "新建文件…",
    "ctx.new_folder": "新建文件夹…",
    "ctx.import": "导入 glafic → glade…",
    "ctx.export": "导出 glade → glafic…",
    "ctx.import_clave": "导入到 Clave",
    "ctx.rename": "重命名…",
    "ctx.delete": "删除",
    "ctx.delete_folder": "删除文件夹",
    "ctx.copy": "复制",
    "ctx.paste": "粘贴",
    "ed.prompt_new_file": "新文件名",
    "ed.prompt_new_folder": "新文件夹名",
    "ed.prompt_rename": "重命名",
    "ed.prompt_export": "导出到 glafic — 输出名",
    "ed.confirm_delete_title": "删除",
    "ed.delete_file_title": "删除文件",
    "ed.confirm_delete": "删除 <b>{0}</b>?此操作不可撤销。",
    "ed.confirm_close_title": "不保存直接关闭?",
    "ed.confirm_close": "<b>{0}</b> 有未保存的更改。",
    "ed.imported": "已导入",
    "ed.wrote": "已写入:",
    "ed.exported": "已导出到 glafic",
    "ed.export_note_opt": "检测到 <b>{lo, hi}</b> 参数 → 已添加 glafic <code>optimize</code> 命令 + setopt 矩阵,以及 <code>readobs_point</code>/<code>parprior</code> 文件。",
    "ed.export_note_plain": "没有 <b>{lo, hi}</b> 参数 → 仅 findimg 模型。",
    "ed.open_first": "请先打开或创建一个文件。",
    "ed.empty_hint": "从资源管理器打开文件,或插入模板。",
    "ed.save_failed": "保存失败: {0}",
    "ed.import_failed": "导入失败: {0}",
    "ed.export_failed": "导出失败: {0}",
    "ed.paste_failed": "粘贴失败: {0}",
    "ed.clave_failed": "导入 Clave 失败: {0}",
    // --- V0.7 单位设置 ---
    "ed.unitsetting": "单位设置",
    "unit.title": "单位设置",
    "unit.profile": "配置",
    "unit.save_as": "另存为",
    "unit.new_profile": "新配置…",
    "unit.col_category": "类别",
    "unit.col_unit": "单位",
    "unit.fixed_note": "固定单位(不可调整)",
    "unit.custom_note": "自定义 {lo, hi} 变量无量纲 —— 取决于插入位置对应的单位。",
    "unit.cat.mass": "透镜质量",
    "unit.cat.obs_pos": "观测像位置",
    "unit.cat.comp_pos": "组件位置与尺度半径",
    "unit.cat.src_pos": "源位置",
    "unit.fixed.velocity": "速度弥散",
    "unit.fixed.angle": "角度",
    "unit.fixed.grid": "网格",
    "unit.fixed.center_offset": "中心偏移",
    "unit.fixed.time_delay": "时间延迟",
    "unit.save_failed": "保存失败: {0}",
    "unit.name_required": "请输入配置名称。",
  },
};
let LANG = localStorage.getItem("glade_lang") || "en";
if (!(LANG in I18N)) LANG = "en";
const t = (key) => (I18N[LANG] && I18N[LANG][key]) ?? I18N.en[key] ?? key;
const fmt = (s, ...args) => s.replace(/\{(\d+)\}/g, (m, i) => args[+i] ?? m);

function applyLang() {
  document.documentElement.lang = LANG === "zh" ? "zh-CN" : "en";
  $$("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  $$("[data-i18n-title]").forEach((el) => { el.title = t(el.dataset.i18nTitle); });
  $("#btn-lang").textContent = LANG === "zh" ? "中文" : "EN";
  // refresh dynamic texts
  FindImage.refreshTexts && FindImage.refreshTexts();
  const empty = $(".editor-empty"); if (empty) empty.textContent = t("ed.empty_hint");
}

// ===================== theme =====================
function currentTheme() { return document.documentElement.dataset.theme || "dark"; }
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("glade_theme", theme);
  $("#btn-theme").textContent = theme === "dark" ? "🌙" : "☀️";
  if (window.monaco && Editor.useMonaco) {
    monaco.editor.setTheme(theme === "light" ? "glade-light" : "glade-dark");
  }
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  const ct = r.headers.get("content-type") || "";
  const body = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error((body && body.error) || r.statusText);
  return body;
}
const apiJSON = (path, obj) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify(obj) });

// -------- modal + context menu helpers --------
function modal({ title, bodyHTML, actions }) {
  const back = $("#modal"); $("#modal-title").textContent = title;
  $("#modal-body").innerHTML = bodyHTML;
  const ad = $("#modal-actions"); ad.innerHTML = "";
  return new Promise((resolve) => {
    const close = (v) => { back.classList.add("hidden"); resolve(v); };
    actions.forEach((a) => {
      const b = document.createElement("button");
      b.textContent = a.label; if (a.cls) b.className = a.cls;
      b.onclick = () => close(a.value);
      ad.appendChild(b);
    });
    back.classList.remove("hidden");
    const inp = $("#modal-body input");
    if (inp) { inp.focus(); inp.select();
      inp.onkeydown = (e) => { if (e.key === "Enter") { const p = actions.find(a=>a.cls&&a.cls.includes("primary")); if(p) close(p.value);} }; }
  });
}
function prompt2(title, value = "") {
  return modal({ title, bodyHTML: `<input id="m-input" value="${value.replace(/"/g,"&quot;")}" />`,
    actions: [{ label: t("common.cancel"), value: null }, { label: t("common.ok"), value: "OK", cls: "primary" }] })
    .then((v) => v === "OK" ? $("#m-input").value.trim() : null);
}
function confirm2(title, html, danger) {
  return modal({ title, bodyHTML: html, actions: [
    { label: t("common.cancel"), value: false },
    { label: danger ? t("common.delete") : t("common.confirm"), value: true, cls: danger ? "danger" : "primary" }] });
}
function ctxMenu(x, y, items) {
  const m = $("#ctxmenu"); m.innerHTML = "";
  items.forEach((it) => {
    if (it === "-") { const s = document.createElement("div"); s.className = "sep"; m.appendChild(s); return; }
    const d = document.createElement("div"); d.className = "item"; d.textContent = it.label;
    d.onclick = () => { m.classList.add("hidden"); it.action(); }; m.appendChild(d);
  });
  m.style.left = x + "px"; m.style.top = y + "px"; m.classList.remove("hidden");
}
document.addEventListener("click", () => $("#ctxmenu").classList.add("hidden"));

// ===================== page switching =====================
$$(".navtab").forEach((t) => t.onclick = () => {
  $$(".navtab").forEach((x) => x.classList.toggle("active", x === t));
  const page = t.dataset.page;
  $$("main.page").forEach((p) => p.classList.toggle("active", p.id === "page-" + page));
  if (page === "findimage") FindImage.refresh();
  if (page === "editor") Editor.ensureMonaco();
  if (page === "clave") Clave.ensure();
});

// ===================== Clave (embedded lens calculator) =====================
const Clave = {
  loaded: false,
  ensure() {
    if (this.loaded) return;
    $("#clave-frame").src = "/clave/?lang=" + LANG;
    this.loaded = true;
  },
  syncLang() {
    const frame = $("#clave-frame");
    if (this.loaded && frame && frame.contentWindow)
      frame.contentWindow.postMessage({ type: "clave-lang", lang: LANG }, "*");
  },
};

// ===================== FindImage =====================
// pipeline = {backend:'cpu'|'gpu', purpose:'calcimage'|'optimize'|'mcmc',
//             optimizer:'amoeba'|'de'|'cmaes'|'jso'}
const PIPELINE_KEY = "glade_pipeline";
const PIPELINE_DEFAULT = { backend: "cpu", purpose: "optimize", optimizer: "de" };
// optimizer options per backend (amoeba is glafic-native, invalid on GPU)
function optimizersFor(backend) {
  return backend === "gpu" ? ["de", "cmaes", "jso"] : ["amoeba", "de", "cmaes", "jso"];
}
function loadPipeline() {
  // 1) modern key
  let raw = null;
  try { raw = JSON.parse(localStorage.getItem(PIPELINE_KEY) || "null"); } catch (e) { raw = null; }
  let p;
  if (raw && typeof raw === "object" && raw.backend) {
    p = { backend: raw.backend, purpose: raw.purpose || "optimize", optimizer: raw.optimizer || "de" };
  } else {
    // 2) migrate an old backend-selection key, if any
    const old = localStorage.getItem("glade_backend");
    const MAP = {
      cpu: { backend: "cpu", purpose: "optimize", optimizer: "de" },
      gpu: { backend: "gpu", purpose: "optimize", optimizer: "de" },
      glafic: { backend: "cpu", purpose: "optimize", optimizer: "amoeba" },
      mcmc: { backend: "cpu", purpose: "mcmc", optimizer: "de" },
      "mcmc-gpu": { backend: "gpu", purpose: "mcmc", optimizer: "de" },
    };
    p = MAP[old] || { ...PIPELINE_DEFAULT };
  }
  // normalise
  if (p.backend !== "cpu" && p.backend !== "gpu") p.backend = "cpu";
  if (!["calcimage", "optimize", "mcmc"].includes(p.purpose)) p.purpose = "optimize";
  if (!optimizersFor(p.backend).includes(p.optimizer)) p.optimizer = "de";
  return p;
}

const FindImage = {
  pipeline: PIPELINE_DEFAULT, files: [], selected: new Set(),
  es: null, _termInfo: null, settingsOpen: false,
  init() {
    this.pipeline = loadPipeline();
    this.savePipeline();          // persist any migrated / normalised value
    $("#fi-settings-btn").onclick = () => this.toggleSettings();
    this.renderPipeline();
    $("#fi-refresh").onclick = () => this.refresh();
    $("#fi-run").onclick = () => this.run();
    this.refresh();
  },
  savePipeline() {
    localStorage.setItem(PIPELINE_KEY, JSON.stringify(this.pipeline));
  },
  // ---- pipeline display (in the rail) ----
  renderPipeline() {
    const p = this.pipeline;
    const box = $("#fi-pipeline"); if (!box) return;
    const engine = p.backend === "gpu" ? t("fi.engine_rhon") : t("fi.engine_glafic");
    const backendName = p.backend === "gpu" ? t("fi.backend_gpu") : t("fi.backend_cpu");
    const purposeName = { calcimage: "fi.purpose_calcimage", optimize: "fi.purpose_optimize", mcmc: "fi.purpose_mcmc" }[p.purpose];
    const optName = { amoeba: "fi.opt_amoeba", de: "fi.opt_de", cmaes: "fi.opt_cmaes", jso: "fi.opt_jso" }[p.optimizer];
    let html =
      `<div class="pl-node pl-backend"><div class="pl-main">${backendName}</div>` +
      `<div class="pl-sub">${engine}</div></div>` +
      `<div class="pl-bar">|</div>` +
      `<div class="pl-node"><div class="pl-main">${t(purposeName)}</div></div>`;
    if (p.purpose === "optimize") {
      html += `<div class="pl-bar">|</div>` +
        `<div class="pl-node"><div class="pl-main">${t(optName)}</div></div>`;
    }
    box.innerHTML = html;
  },
  // ---- settings panel (over the terminal) ----
  toggleSettings() {
    this.settingsOpen = !this.settingsOpen;
    $("#fi-settings-btn").classList.toggle("active", this.settingsOpen);
    $("#fi-terminal-wrap").classList.toggle("hidden", this.settingsOpen);
    $("#fi-settings-panel").classList.toggle("hidden", !this.settingsOpen);
    if (this.settingsOpen) this.renderSettings();
  },
  _optCard(kind, value, current, labelKey, hintKey) {
    const active = current === value ? " active" : "";
    return `<button class="setting-card${active}" data-kind="${kind}" data-value="${value}">` +
      `<span class="sc-label">${t(labelKey)}</span>` +
      `<span class="sc-hint">${t(hintKey)}</span></button>`;
  },
  renderSettings() {
    const p = this.pipeline;
    const panel = $("#fi-settings-panel");
    let html = `<div class="settings-head">${t("fi.settings_head")}</div>`;
    html += `<div class="settings-scroll">`;
    // Row 1: backend
    html += `<div class="setting-section"><div class="setting-label">${t("fi.sec_backend")}</div>` +
      `<div class="setting-cards">` +
      this._optCard("backend", "cpu", p.backend, "fi.backend_cpu", "fi.hint_cpu") +
      this._optCard("backend", "gpu", p.backend, "fi.backend_gpu", "fi.hint_gpu") +
      `</div></div>`;
    // Row 2: purpose
    html += `<div class="setting-section"><div class="setting-label">${t("fi.sec_purpose")}</div>` +
      `<div class="setting-cards">` +
      this._optCard("purpose", "calcimage", p.purpose, "fi.purpose_calcimage", "fi.hint_calcimage") +
      this._optCard("purpose", "optimize", p.purpose, "fi.purpose_optimize", "fi.hint_optimize") +
      this._optCard("purpose", "mcmc", p.purpose, "fi.purpose_mcmc", "fi.hint_mcmc") +
      `</div></div>`;
    // Row 3: optimizer (only for purpose = optimize)
    if (p.purpose === "optimize") {
      const HINTS = { amoeba: "fi.hint_amoeba", de: "fi.hint_de", cmaes: "fi.hint_cmaes", jso: "fi.hint_jso" };
      const LABELS = { amoeba: "fi.opt_amoeba", de: "fi.opt_de", cmaes: "fi.opt_cmaes", jso: "fi.opt_jso" };
      html += `<div class="setting-section"><div class="setting-label">${t("fi.sec_optimizer")}</div>` +
        `<div class="setting-cards">` +
        optimizersFor(p.backend).map((o) => this._optCard("optimizer", o, p.optimizer, LABELS[o], HINTS[o])).join("") +
        `</div></div>`;
    }
    html += `</div>`; // settings-scroll
    html += `<div class="settings-foot">` +
      `<button id="fi-settings-save" class="run-btn">${t("fi.save")}</button></div>`;
    panel.innerHTML = html;
    // wire cards
    $$(".setting-card", panel).forEach((c) => c.onclick = () => this._pick(c.dataset.kind, c.dataset.value));
    $("#fi-settings-save").onclick = () => this.toggleSettings();
  },
  _pick(kind, value) {
    const p = this.pipeline;
    if (kind === "backend") {
      p.backend = value;
      // amoeba is invalid on GPU -> fall back to DE
      if (!optimizersFor(p.backend).includes(p.optimizer)) p.optimizer = "de";
    } else if (kind === "purpose") {
      p.purpose = value;
    } else if (kind === "optimizer") {
      p.optimizer = value;
    }
    this.savePipeline();          // auto-persist on every change
    this.renderSettings();
    this.renderPipeline();
  },
  async refresh() {
    this.tree = await api("/api/files/tree");
    this.render();
  },
  refreshTexts() {
    this.updateSummary();
    this.renderPipeline();
    if (this.settingsOpen) this.renderSettings();
    if (this._termInfo) {
      $("#term-title").textContent =
        fmt(t("fi.term_title_job"), this._termInfo.jobId, this._termInfo.terminal);
    }
    const empty = $("#fi-filelist .filelist-empty");
    if (empty) empty.textContent = t("fi.no_files");
  },
  render() {
    const list = $("#fi-filelist"); list.innerHTML = "";
    if (!this.tree || !(this.tree.children || []).length) {
      list.innerHTML = `<div class="filelist-empty" style="padding:10px;color:var(--fg-dim);font-size:12px">${t("fi.no_files")}</div>`;
      return this.updateSummary();
    }
    // prune selections that no longer exist
    const present = new Set(); (function walk(n){ n.type === "dir" ? (n.children||[]).forEach(walk) : present.add(n.path); })(this.tree);
    [...this.selected].forEach((p) => { if (!present.has(p)) this.selected.delete(p); });
    list.appendChild(this._node(this.tree, true));
    this.updateSummary();
  },
  _node(node, isRoot) {
    const wrap = document.createElement("div");
    if (node.type === "dir") {
      const row = document.createElement("div");
      row.className = "tree-row" + (isRoot ? " open" : "");
      row.innerHTML = `<span class="twisty">▶</span><span class="ico">${isRoot ? "🗁" : "🗀"}</span><span class="label">${node.name}</span>`;
      const kids = document.createElement("div");
      kids.className = "tree-children" + (isRoot ? " open" : "");
      (node.children || []).forEach((c) => kids.appendChild(this._node(c, false)));
      row.onclick = () => { row.classList.toggle("open"); kids.classList.toggle("open"); };
      wrap.appendChild(row); wrap.appendChild(kids);
    } else {
      const row = document.createElement("label"); row.className = "tree-row";
      row.innerHTML = `<input type="checkbox" class="fcheck" ${this.selected.has(node.path) ? "checked" : ""}/>` +
        `<span class="ico">📄</span><span class="label">${node.name}</span>`;
      row.querySelector("input").onchange = (e) => {
        if (e.target.checked) this.selected.add(node.path); else this.selected.delete(node.path);
        this.updateSummary();
      };
      wrap.appendChild(row);
    }
    return wrap;
  },
  updateSummary() {
    const n = this.selected.size;
    $("#fi-summary").textContent = n ? fmt(t("fi.n_selected"), n, [...this.selected].join(", "))
                                     : t("fi.none_selected");
    $("#fi-run").disabled = n === 0;
  },
  async run() {
    const files = [...this.selected];
    const p = this.pipeline;
    const payload = { backend: p.backend, purpose: p.purpose, optimizer: p.optimizer, files };
    let res;
    try { res = await apiJSON("/api/run", payload); }
    catch (e) { return modal({ title: t("fi.run_failed"), bodyHTML: `<div class="errlist">${e.message}</div>`,
      actions: [{ label: t("common.ok"), value: 1, cls: "primary" }] }); }

    if (res.errors && res.errors.length) {
      return modal({ title: t("fi.cfg_errors"), actions: [{ label: t("common.ok"), value: 1, cls: "primary" }],
        bodyHTML: `<div class="errlist">${res.errors.map((e) => "✗ " + e).join("<br>")}</div>` });
    }
    if (res.needs_confirm) {
      const rows = Object.entries(res.defaulted)
        .map(([k, v]) => `${k} = ${JSON.stringify(v)}`).join("<br>");
      const ok = await confirm2(t("fi.missing_basics"),
        `<p>${t("fi.defaults_intro")}</p>
         <div class="deflist">${rows}</div><p>${t("fi.defaults_q")}</p>`);
      if (!ok) return;
      res = await apiJSON("/api/run", { ...payload, force: true });
    }
    if (res.job_id) this.startStream(res.job_id, res.terminal);
  },
  startStream(jobId, terminal) {
    const term = $("#terminal"); term.textContent = "";
    $("#result-area").classList.add("hidden");
    this._termInfo = { jobId, terminal };
    $("#term-title").textContent = fmt(t("fi.term_title_job"), jobId, terminal);
    const state = $("#term-state"); state.textContent = t("fi.state_running"); state.className = "term-state running";
    if (this.es) this.es.close();
    const es = new EventSource(`/api/run/${jobId}/stream`); this.es = es;
    es.onmessage = (ev) => { term.textContent += ev.data + "\n"; term.scrollTop = term.scrollHeight; };
    es.addEventListener("end", async () => {
      es.close();
      const st = await api(`/api/run/${jobId}/status`).catch(() => ({}));
      if (st.state === "done") {
        let txt = t("fi.state_done");
        if (st.loss != null && isFinite(st.loss)) txt += ` · ${t("fi.loss")} ${Number(st.loss).toFixed(2)}`;
        if (st.iterations) txt += " · " + fmt(t("fi.iters"), st.iterations);
        if (st.mcmc) txt += " · " + fmt(t("fi.mcmc_accept"),
          Number(st.mcmc.acceptance).toFixed(2), st.mcmc.n_samples);
        state.textContent = txt; state.className = "term-state done";
        this.showResults(jobId, st);
      } else { state.textContent = st.state || "ended"; state.className = "term-state error"; }
    });
    es.onerror = () => { state.textContent = t("fi.stream_error"); state.className = "term-state error"; es.close(); };
  },
  showResults(jobId, st) {
    const area = $("#result-area"); area.innerHTML = "";
    const figs = [];
    if (st.triptych) figs.push([t("fi.fig_result"), st.triptych]);
    if (st.mcmc && st.mcmc.corner) figs.push([t("fi.fig_corner"), st.mcmc.corner]);
    if (st.mcmc && st.mcmc.trace) figs.push([t("fi.fig_trace"), st.mcmc.trace]);
    if (!figs.length) { area.classList.add("hidden"); return; }
    figs.forEach(([label, fname]) => {
      const fig = document.createElement("figure"); fig.className = "result-fig";
      fig.innerHTML = `<figcaption>${label}</figcaption>` +
        `<img src="/api/run/${jobId}/result/${fname}?t=${Date.now()}" alt="${label}"/>`;
      area.appendChild(fig);
    });
    area.classList.remove("hidden");
  },
};

// ===================== Editor =====================
const Editor = {
  monacoReady: null, useMonaco: false, mInst: null, mModels: {}, ta: null,
  tabs: [], active: null, changeCb: null, panel: "explorer", clipboard: null,

  init() {
    $$(".icon-btn[data-panel]").forEach((b) => b.onclick = () => this.setPanel(b.dataset.panel));
    $("#icon-unit").onclick = () => UnitSetting.open();
    $("#exp-refresh").onclick = () => this.loadTree();
    $("#exp-new-file").onclick = () => this.newEntry("file", "");
    $("#exp-new-folder").onclick = () => this.newEntry("folder", "");
    $("#btn-save").onclick = () => this.save();
    $("#btn-delete").onclick = () => this.deleteActive();
    this._initResizer();
    this.loadTree(); this.loadTemplates();
    this.renderEmpty();
  },
  setPanel(p) {
    this.panel = p;
    $$(".icon-btn").forEach((b) => b.classList.toggle("active", b.dataset.panel === p));
    $("#explorer-panel").classList.toggle("active", p === "explorer");
    $("#template-panel").classList.toggle("active", p === "template");
  },
  _initResizer() {
    const rz = $("#panel-resizer"), panel = $("#side-panel"); let dragging = false;
    rz.onmousedown = (e) => { dragging = true; e.preventDefault(); document.body.style.cursor = "col-resize"; };
    document.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const w = Math.max(120, Math.min(600, e.clientX - 50));
      panel.style.width = w + "px";
    });
    document.addEventListener("mouseup", () => { dragging = false; document.body.style.cursor = ""; });
  },

  // ---- Monaco / fallback ----
  ensureMonaco() {
    if (this.monacoReady) return this.monacoReady;
    this.monacoReady = new Promise((resolve) => {
      fetch("/static/vendor/monaco/vs/loader.js", { method: "HEAD" }).then((r) => {
        if (!r.ok) throw 0;
        const s = document.createElement("script"); s.src = "/static/vendor/monaco/vs/loader.js";
        s.onload = () => {
          window.require.config({ paths: { vs: "/static/vendor/monaco/vs" } });
          window.require(["vs/editor/editor.main"], () => { this._setupMonaco(); resolve(true); });
        };
        s.onerror = () => { this._setupTextarea(); resolve(false); };
        document.head.appendChild(s);
      }).catch(() => { this._setupTextarea(); resolve(false); });
    });
    return this.monacoReady;
  },
  _setupMonaco() {
    this.useMonaco = true;
    monaco.languages.register({ id: "glade" });
    monaco.languages.setMonarchTokensProvider("glade", {
      tokenizer: { root: [
        [/#.*$/, "comment"],
        [/\$(float|int|str)(\{[^}]*\})?/, "keyword"],
        [/'[^']*'/, "string"], [/"[^"]*"/, "string"],
        [/\{[^}]*\}/, "number.hex"],
        [/[-+]?\d[\d.eE+-]*/, "number"],
        [/[A-Za-z_]\w*/, "identifier"],
      ] } });
    monaco.editor.defineTheme("glade-dark", { base: "vs-dark", inherit: true, rules: [
      { token: "keyword", foreground: "d7ba7d", fontStyle: "bold" },
      { token: "number.hex", foreground: "4ec9b0" } ], colors: {} });
    monaco.editor.defineTheme("glade-light", { base: "vs", inherit: true, rules: [
      { token: "keyword", foreground: "9c6b00", fontStyle: "bold" },
      { token: "number.hex", foreground: "00755f" } ], colors: {} });
  },
  _monacoTheme() { return currentTheme() === "light" ? "glade-light" : "glade-dark"; },
  _setupTextarea() {
    this.useMonaco = false;
    const ta = document.createElement("textarea"); ta.className = "fallback"; ta.spellcheck = false;
    ta.style.display = "none";
    ta.oninput = () => { const t = this._tab(this.active); if (t) { t.content = ta.value; t.dirty = true; this.renderTabs(); this._updateFooter(); } };
    $("#editor-host").appendChild(ta); this.ta = ta;
  },

  // ---- tree rendering ----
  async loadTree() {
    const tree = await api("/api/files/tree");
    const host = $("#file-tree"); host.innerHTML = "";
    host.appendChild(this._renderNode(tree, true));
  },
  _renderNode(node, isRoot) {
    const wrap = document.createElement("div");
    if (node.type === "dir") {
      const row = this._row(node.name, "▶", isRoot ? "🗁" : "🗀", node, "dir");
      const kids = document.createElement("div"); kids.className = "tree-children" + (isRoot ? " open" : "");
      if (isRoot) row.classList.add("open");
      (node.children || []).forEach((c) => kids.appendChild(this._renderNode(c, false)));
      row.onclick = () => { row.classList.toggle("open"); kids.classList.toggle("open"); };
      wrap.appendChild(row); wrap.appendChild(kids);
    } else {
      const row = this._row(node.name, "", "📄", node, "file");
      row.onclick = () => this.openFile(node.path);
      wrap.appendChild(row);
    }
    return wrap;
  },
  _row(label, twisty, ico, node, kind) {
    const row = document.createElement("div"); row.className = "tree-row";
    row.innerHTML = `<span class="twisty">${twisty}</span><span class="ico">${ico}</span><span class="label">${label}</span>`;
    row.oncontextmenu = (e) => { e.preventDefault(); e.stopPropagation(); this._ctx(e, node, kind); };
    return row;
  },
  _ctx(e, node, kind) {
    const items = [];
    if (kind === "dir") {
      items.push({ label: t("ctx.new_file"), action: () => this.newEntry("file", node.path) });
      items.push({ label: t("ctx.new_folder"), action: () => this.newEntry("folder", node.path) });
      if (node.path || this.clipboard) items.push("-");
      if (node.path)
        items.push({ label: t("ctx.copy"), action: () => this.copyEntry(node, kind) });
      if (this.clipboard)
        items.push({ label: t("ctx.paste"), action: () => this.pasteInto(node.path) });
      items.push("-");
      items.push({ label: t("ctx.import"), action: () => this.importFile(node.path) });
      if (node.path) {
        items.push("-");
        items.push({ label: t("ctx.rename"), action: () => this.renameEntry(node) });
        items.push({ label: t("ctx.delete_folder"), action: () => this.deleteEntry(node) });
      }
    } else {
      items.push({ label: t("ctx.open"), action: () => this.openFile(node.path) });
      items.push("-");
      items.push({ label: t("ctx.copy"), action: () => this.copyEntry(node, kind) });
      if (this.clipboard) {
        // paste next to this file, i.e. into its parent folder
        const parent = node.path.includes("/") ? node.path.slice(0, node.path.lastIndexOf("/")) : "";
        items.push({ label: t("ctx.paste"), action: () => this.pasteInto(parent) });
      }
      items.push("-");
      items.push({ label: t("ctx.import"), action: () => this.importFile(node.path) });
      items.push({ label: t("ctx.export"), action: () => this.exportFile(node) });
      items.push({ label: t("ctx.import_clave"), action: () => this.importToClave(node.path) });
      items.push("-");
      items.push({ label: t("ctx.rename"), action: () => this.renameEntry(node) });
      items.push({ label: t("ctx.delete"), action: () => this.deleteEntry(node) });
    }
    ctxMenu(e.clientX, e.clientY, items);
  },
  copyEntry(node, kind) {
    this.clipboard = { path: node.path, name: node.name, type: kind };
  },
  _copyName(base, isFile, attempt) {
    // attempt 0 -> name_copy(.ext), attempt 1 -> name_copy2(.ext), ...
    const suffix = "_copy" + (attempt ? attempt + 1 : "");
    if (isFile && base.includes(".")) {
      const dot = base.lastIndexOf(".");
      return base.slice(0, dot) + suffix + base.slice(dot);
    }
    return base + suffix;
  },
  async pasteInto(dir) {
    const cb = this.clipboard; if (!cb) return;
    let name = cb.name;
    for (let attempt = 0; attempt < 30; attempt++) {
      const dest = dir ? `${dir}/${name}` : name;
      try {
        await apiJSON("/api/files/copy", { path: cb.path, dest });
        await this.loadTree(); FindImage.refresh && FindImage.refresh();
        return;
      } catch (err) {
        if (/exists/i.test(err.message)) {
          name = this._copyName(cb.name, cb.type === "file", attempt);
          continue;
        }
        alert(fmt(t("ed.paste_failed"), err.message)); return;
      }
    }
    alert(fmt(t("ed.paste_failed"), "too many copies"));
  },
  async newEntry(type, dir) {
    const name = await prompt2(type === "folder" ? t("ed.prompt_new_folder") : t("ed.prompt_new_file"),
      type === "file" ? "untitled.dat" : "new_folder");
    if (!name) return;
    const path = dir ? `${dir}/${name}` : name;
    try { await apiJSON("/api/files/create", { path, type }); await this.loadTree();
      if (type === "file") this.openFile(path); }
    catch (e) { alert(e.message); }
  },
  async renameEntry(node) {
    const name = await prompt2(t("ed.prompt_rename"), node.name);
    if (!name || name === node.name) return;
    try { const r = await apiJSON("/api/files/rename", { path: node.path, name }); await this.loadTree();
      const t2 = this._tab(node.path); if (t2) { t2.path = r.path; this.renderTabs(); } }
    catch (e) { alert(e.message); }
  },
  async deleteEntry(node) {
    const ok = await confirm2(t("ed.confirm_delete_title"), fmt(t("ed.confirm_delete"), node.name), true);
    if (!ok) return;
    try { await apiJSON("/api/files/delete", { path: node.path }); this.closeTab(node.path); await this.loadTree(); }
    catch (e) { alert(e.message); }
  },
  async importFile(path) {
    try { const r = await apiJSON("/api/files/import", { path }); await this.loadTree();
      modal({ title: t("ed.imported"), bodyHTML: `${t("ed.wrote")}<br><div class="deflist">${r.written.join("<br>")}</div>`,
        actions: [{ label: t("common.ok"), value: 1, cls: "primary" }] }); }
    catch (e) { alert(fmt(t("ed.import_failed"), e.message)); }
  },
  async importToClave(path) {
    try {
      const scene = await apiJSON("/api/files/import_clave", { path });
      // switch to the Clave tab, then hand the scene to the (same-origin) iframe
      $('.navtab[data-page="clave"]').click();
      const frame = $("#clave-frame");
      const send = () => { frame.contentWindow.claveImport(scene); };
      if (frame.contentWindow && frame.contentWindow.claveImport) send();
      else frame.addEventListener("load", send, { once: true });
    } catch (e) { alert(fmt(t("ed.clave_failed"), e.message)); }
  },
  async exportFile(node) {
    const base = await prompt2(t("ed.prompt_export"),
      node.name.replace(/\.[^.]*$/, "") + "_glafic");
    if (!base) return;
    try { const r = await apiJSON("/api/files/export", { files: [node.path], name: base });
      await this.loadTree();
      const note = r.optimize
        ? `<p class="exportnote-ok">${t("ed.export_note_opt")}</p>`
        : `<p class="exportnote-plain">${t("ed.export_note_plain")}</p>`;
      modal({ title: t("ed.exported"), bodyHTML: `${t("ed.wrote")}<br><div class="deflist">${r.written.join("<br>")}</div>${note}`,
        actions: [{ label: t("common.ok"), value: 1, cls: "primary" }] }); }
    catch (e) { alert(fmt(t("ed.export_failed"), e.message)); }
  },

  // ---- templates ----
  async loadTemplates() {
    // fetch the tree for the active units profile (comments switch units)
    const prof = localStorage.getItem(UNITS_PROFILE_KEY) || "default";
    const url = (prof && prof !== "default")
      ? "/api/templates?units=" + encodeURIComponent(prof)
      : "/api/templates";
    const tree = await api(url);
    const host = $("#template-tree"); host.innerHTML = "";
    // arbitrary nesting: a node with `children` is a group; otherwise a leaf.
    tree.forEach((node) => host.appendChild(this._tmplNode(node, 0)));
  },
  _tmplNode(node, depth) {
    const wrap = document.createElement("div");
    const indent = 8 + depth * 14;   // px; matches .tree indentation feel
    if (Array.isArray(node.children)) {
      const open = depth === 0;      // top-level groups open, deeper ones collapsed
      const row = document.createElement("div");
      row.className = "tree-row" + (open ? " open" : "");
      row.style.paddingLeft = indent + "px";
      row.innerHTML = `<span class="twisty">▶</span><span class="ico">▤</span><span class="label">${node.name}</span>`;
      const kids = document.createElement("div");
      kids.className = "tree-children" + (open ? " open" : "");
      node.children.forEach((c) => kids.appendChild(this._tmplNode(c, depth + 1)));
      row.onclick = () => { row.classList.toggle("open"); kids.classList.toggle("open"); };
      wrap.appendChild(row); wrap.appendChild(kids);
    } else {
      const leaf = node;
      const lr = document.createElement("div");
      lr.className = "tree-row" + (leaf.disabled || !leaf.snippet ? " disabled" : "");
      lr.style.paddingLeft = (indent + 16) + "px";
      lr.innerHTML = `<span class="ico">›</span><span class="label">${leaf.name}</span>`;
      if (!leaf.disabled && leaf.snippet) lr.onclick = () => this.insertTemplate(leaf);
      wrap.appendChild(lr);
    }
    return wrap;
  },
  insertTemplate(leaf) {
    if (!this.active) { alert(t("ed.open_first")); return; }
    let text = leaf.snippet;
    if (leaf.key) {
      const doc = this.getValue();
      let maxName = 0; const nre = new RegExp("'" + leaf.key + "(\\d+)'", "g"); let m;
      while ((m = nre.exec(doc))) maxName = Math.max(maxName, +m[1]);
      let maxIdx = 0; const ire = /^\s*'[^']+'\s*:\s*\(\s*(\d+)/gm; let mi;
      while ((mi = ire.exec(doc))) maxIdx = Math.max(maxIdx, +mi[1]);
      text = text.replace("'" + leaf.key + "1'", "'" + leaf.key + (maxName + 1) + "'")
                 .replace(/\(\s*1\s*,/, "(" + (maxIdx + 1) + ",");
    }
    this.insert(text);
  },

  // ---- tabs / files ----
  _tab(path) { return this.tabs.find((t) => t.path === path); },
  async openFile(path) {
    await this.ensureMonaco();
    if (!this._tab(path)) {
      const r = await api("/api/files/read?path=" + encodeURIComponent(path));
      this.tabs.push({ path, content: r.content, dirty: false });
    }
    this.setActive(path);
    this.renderTabs();
  },
  setActive(path) {
    this.active = path; const t = this._tab(path); if (!t) return;
    $(".editor-empty") && $(".editor-empty").remove();
    if (this.useMonaco) {
      if (!this.mModels[path]) {
        this.mModels[path] = monaco.editor.createModel(t.content, "glade");
        this.mModels[path].onDidChangeContent(() => {
          const tab = this._tab(path); if (tab) { tab.content = this.mModels[path].getValue(); tab.dirty = true; this.renderTabs(); this._updateFooter(); }
        });
      }
      if (!this.mInst) {
        this.mInst = monaco.editor.create($("#editor-host"),
          { model: this.mModels[path], theme: this._monacoTheme(), automaticLayout: true,
            fontSize: 13, minimap: { enabled: false }, scrollBeyondLastLine: false });
      } else this.mInst.setModel(this.mModels[path]);
    } else {
      this.ta.style.display = "block"; this.ta.value = t.content;
    }
    this.renderTabs(); this._updateFooter();
  },
  renderEmpty() {
    if (!this.tabs.length) {
      const h = $("#editor-host");
      if (!$(".editor-empty")) { const d = document.createElement("div"); d.className = "editor-empty";
        d.textContent = t("ed.empty_hint"); h.appendChild(d); }
      if (this.ta) this.ta.style.display = "none";
    }
  },
  renderTabs() {
    const bar = $("#tabbar"); bar.innerHTML = "";
    this.tabs.forEach((t) => {
      const name = t.path.slice(t.path.lastIndexOf("/") + 1);
      const tab = document.createElement("div"); tab.className = "tab" + (t.path === this.active ? " active" : "");
      tab.innerHTML = `<span>${t.dirty ? '<span class="dirty">●</span> ' : ""}${name}</span><span class="close">×</span>`;
      tab.querySelector("span").onclick = () => this.setActive(t.path);
      tab.onclick = (e) => { if (!e.target.classList.contains("close")) this.setActive(t.path); };
      tab.querySelector(".close").onclick = (e) => { e.stopPropagation(); this.closeTab(t.path); };
      bar.appendChild(tab);
    });
  },
  closeTab(path) {
    const t2 = this._tab(path); if (!t2) return;
    const doClose = () => {
      this.tabs = this.tabs.filter((x) => x.path !== path);
      if (this.mModels[path]) { this.mModels[path].dispose(); delete this.mModels[path]; }
      if (this.active === path) {
        const next = this.tabs[this.tabs.length - 1];
        if (next) this.setActive(next.path);
        else { this.active = null; if (this.mInst) this.mInst.setModel(null); if (this.ta) this.ta.style.display = "none"; this.renderEmpty(); }
      }
      this.renderTabs(); this._updateFooter();
    };
    if (t2.dirty) confirm2(t("ed.confirm_close_title"), fmt(t("ed.confirm_close"), path), true).then((ok) => ok && doClose());
    else doClose();
  },
  getValue() {
    if (!this.active) return "";
    return this.useMonaco ? this.mModels[this.active].getValue() : this.ta.value;
  },
  insert(text) {
    if (!this.active) return;
    if (this.useMonaco && this.mInst) {
      const sel = this.mInst.getSelection();
      this.mInst.executeEdits("tmpl", [{ range: sel, text, forceMoveMarkers: true }]);
      this.mInst.focus();
    } else if (this.ta) {
      const s = this.ta.selectionStart, e = this.ta.selectionEnd, v = this.ta.value;
      this.ta.value = v.slice(0, s) + text + v.slice(e);
      this.ta.selectionStart = this.ta.selectionEnd = s + text.length;
      const t2 = this._tab(this.active); if (t2) { t2.content = this.ta.value; t2.dirty = true; }
      this.ta.focus();
    }
    this.renderTabs(); this._updateFooter();
  },
  _updateFooter() {
    $("#btn-save").disabled = !this.active;
    $("#btn-delete").disabled = !this.active;
    $("#editor-path").textContent = this.active || "";
  },
  async save() {
    if (!this.active) return;
    const t2 = this._tab(this.active);
    try { await apiJSON("/api/files/write", { path: t2.path, content: this.getValue() });
      t2.content = this.getValue(); t2.dirty = false; this.renderTabs(); this.loadTree(); FindImage.refresh && FindImage.refresh(); }
    catch (e) { alert(fmt(t("ed.save_failed"), e.message)); }
  },
  async deleteActive() {
    if (!this.active) return;
    const ok = await confirm2(t("ed.delete_file_title"), fmt(t("ed.confirm_delete"), this.active), true);
    if (!ok) return;
    try { await apiJSON("/api/files/delete", { path: this.active }); const p = this.active; this.closeTab(p); this.loadTree(); }
    catch (e) { alert(e.message); }
  },
};

// ===================== Unit settings (Editor) =====================
const UNITS_PROFILE_KEY = "glade_units_profile";
// contract-shaped fallback if GET /api/units is unavailable
const UNITS_FALLBACK = {
  categories: {
    mass: { default: "h^-1 Msun", options: ["h^-1 Msun", "Msun"] },
    obs_pos: { default: "mas", options: ["mas", "arcsec"] },
    comp_pos: { default: "arcsec", options: ["arcsec", "mas"] },
    src_pos: { default: "arcsec", options: ["arcsec", "mas"] },
  },
  fixed: { velocity: "km/s", angle: "deg", grid: "arcsec",
           center_offset: "arcsec", time_delay: "days" },
  profiles: [],
};
// the order category rows are shown in
const UNIT_CAT_ORDER = ["obs_pos", "mass", "comp_pos", "src_pos"];

const UnitSetting = {
  data: null,
  async open() {
    // fetch the unit universe; fall back to the contract-shaped default
    try { this.data = await api("/api/units"); }
    catch (e) { this.data = UNITS_FALLBACK; }
    if (!this.data || !this.data.categories) this.data = UNITS_FALLBACK;
    const active = localStorage.getItem(UNITS_PROFILE_KEY) || "default";
    this._render(active);
  },
  _profileUnits(name) {
    // units map for a named profile ('default' -> {} = each category's default)
    if (!name || name === "default") return {};
    const p = (this.data.profiles || []).find((x) => x.name === name);
    return (p && p.units) || {};
  },
  _catOrder() {
    const cats = this.data.categories || {};
    const ordered = UNIT_CAT_ORDER.filter((k) => k in cats);
    Object.keys(cats).forEach((k) => { if (!ordered.includes(k)) ordered.push(k); });
    return ordered;
  },
  _render(profileName) {
    const cats = this.data.categories || {};
    const fixed = this.data.fixed || {};
    const cur = this._profileUnits(profileName);
    const names = ["default", ...((this.data.profiles || []).map((p) => p.name))];
    const profOpts = names.map((n) =>
      `<option value="${n}"${n === profileName ? " selected" : ""}>${n}</option>`).join("");
    // adjustable category rows
    const catRows = this._catOrder().map((k) => {
      const c = cats[k];
      const chosen = cur[k] || c.default;
      const opts = (c.options || [c.default]).map((o) =>
        `<option value="${o.replace(/"/g, "&quot;")}"${o === chosen ? " selected" : ""}>${o}</option>`).join("");
      return `<tr><td class="ut-label">${t("unit.cat." + k)}</td>` +
        `<td><select class="ut-select" data-cat="${k}">${opts}</select></td></tr>`;
    }).join("");
    // fixed (read-only) rows
    const fixedRows = Object.entries(fixed).map(([k, v]) =>
      `<tr class="ut-fixed"><td class="ut-label">${t("unit.fixed." + k)}</td>` +
      `<td><span class="ut-fixedval">${v}</span></td></tr>`).join("");

    const body =
      `<div class="unit-toprow">` +
        `<label class="unit-field"><span>${t("unit.profile")}</span>` +
          `<select id="u-profile">${profOpts}</select></label>` +
        `<label class="unit-field"><span>${t("unit.save_as")}</span>` +
          `<input id="u-name" value="${profileName === "default" ? "" : profileName}" placeholder="${t("unit.new_profile")}" /></label>` +
      `</div>` +
      `<table class="unit-table"><thead><tr>` +
        `<th>${t("unit.col_category")}</th><th>${t("unit.col_unit")}</th></tr></thead>` +
        `<tbody>${catRows}</tbody></table>` +
      `<div class="unit-fixed-note">${t("unit.fixed_note")}</div>` +
      `<table class="unit-table unit-table-fixed"><tbody>${fixedRows}</tbody></table>` +
      `<div class="unit-custom-note">${t("unit.custom_note")}</div>`;

    const back = $("#modal");
    $("#modal-title").textContent = t("unit.title");
    $("#modal-body").innerHTML = body;
    const ad = $("#modal-actions");
    ad.className = "modal-actions unit-actions";
    ad.innerHTML = "";
    const cancel = document.createElement("button");
    cancel.textContent = t("common.cancel");
    cancel.onclick = () => this._close();
    const save = document.createElement("button");
    save.textContent = t("common.save"); save.className = "primary";
    save.onclick = () => this._save();
    ad.appendChild(cancel); ad.appendChild(save);
    back.classList.remove("hidden");

    // switching the profile selector reloads that profile's values + name
    $("#u-profile").onchange = (e) => this._render(e.target.value);
  },
  _close() {
    $("#modal").classList.add("hidden");
    $("#modal-actions").className = "modal-actions";   // restore shared class
  },
  async _save() {
    let name = ($("#u-name").value || "").trim() || $("#u-profile").value || "default";
    const units = {};
    $$("#modal-body .ut-select").forEach((s) => { units[s.dataset.cat] = s.value; });
    if (name !== "default") {
      try { await apiJSON("/api/units/save", { name, units }); }
      catch (e) { alert(fmt(t("unit.save_failed"), e.message)); return; }
      // reflect the saved profile locally so a reopen shows it
      const list = (this.data.profiles = this.data.profiles || []);
      const ex = list.find((p) => p.name === name);
      if (ex) ex.units = units; else list.push({ name, units });
    }
    localStorage.setItem(UNITS_PROFILE_KEY, name);
    this._close();
    // re-fetch templates so snippet comments switch to the chosen units
    try { await Editor.loadTemplates(); } catch (e) { /* editor may not be mounted */ }
  },
};

// ===================== boot =====================
$("#btn-lang").onclick = () => {
  LANG = LANG === "en" ? "zh" : "en";
  localStorage.setItem("glade_lang", LANG);
  applyLang();
  Clave.syncLang();
};
$("#btn-theme").onclick = () => {
  applyTheme(currentTheme() === "dark" ? "light" : "dark");
};
applyTheme(currentTheme());
applyLang();
FindImage.init();
Editor.init();
