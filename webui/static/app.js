"use strict";
// ===================== GLADE WebUI front-end =====================
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

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
    actions: [{ label: "Cancel", value: null }, { label: "OK", value: "OK", cls: "primary" }] })
    .then((v) => v === "OK" ? $("#m-input").value.trim() : null);
}
function confirm2(title, html, danger) {
  return modal({ title, bodyHTML: html, actions: [
    { label: "Cancel", value: false },
    { label: danger ? "Delete" : "Confirm", value: true, cls: danger ? "danger" : "primary" }] });
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
  $("#page-findimage").classList.toggle("active", page === "findimage");
  $("#page-editor").classList.toggle("active", page === "editor");
  if (page === "findimage") FindImage.refresh();
  if (page === "editor") Editor.ensureMonaco();
});

// ===================== FindImage =====================
const FindImage = {
  backend: "cpu", files: [], selected: new Set(), es: null,
  init() {
    $$(".backend-opt").forEach((b) => b.onclick = () => {
      $$(".backend-opt").forEach((x) => x.classList.toggle("active", x === b));
      this.backend = b.dataset.backend;
    });
    $("#fi-refresh").onclick = () => this.refresh();
    $("#fi-run").onclick = () => this.run();
    this.refresh();
  },
  async refresh() {
    this.tree = await api("/api/files/tree");
    this.render();
  },
  render() {
    const list = $("#fi-filelist"); list.innerHTML = "";
    if (!this.tree || !(this.tree.children || []).length) {
      list.innerHTML = '<div style="padding:10px;color:#777;font-size:12px">No files in InputFiles/. Create some in the Editor.</div>';
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
    $("#fi-summary").textContent = n ? `${n} file(s): ${[...this.selected].join(", ")}` : "No files selected";
    $("#fi-run").disabled = n === 0;
  },
  async run() {
    const files = [...this.selected];
    let res;
    try { res = await apiJSON("/api/run", { backend: this.backend, files }); }
    catch (e) { return modal({ title: "Run failed", bodyHTML: `<div class="errlist">${e.message}</div>`,
      actions: [{ label: "OK", value: 1, cls: "primary" }] }); }

    if (res.errors && res.errors.length) {
      return modal({ title: "Cannot run — configuration errors", actions: [{ label: "OK", value: 1, cls: "primary" }],
        bodyHTML: `<div class="errlist">${res.errors.map((e) => "✗ " + e).join("<br>")}</div>` });
    }
    if (res.needs_confirm) {
      const rows = Object.entries(res.defaulted)
        .map(([k, v]) => `${k} = ${JSON.stringify(v)}`).join("<br>");
      const ok = await confirm2("Missing basic values — use defaults?",
        `<p>These basic variables were not provided and will use defaults:</p>
         <div class="deflist">${rows}</div><p>Continue with these defaults?</p>`);
      if (!ok) return;
      res = await apiJSON("/api/run", { backend: this.backend, files, force: true });
    }
    if (res.job_id) this.startStream(res.job_id, res.terminal);
  },
  startStream(jobId, terminal) {
    const term = $("#terminal"); term.textContent = "";
    $("#result-area").classList.add("hidden");
    $("#term-title").textContent = `Terminal output — job ${jobId} (${terminal})`;
    const state = $("#term-state"); state.textContent = "running"; state.className = "term-state running";
    if (this.es) this.es.close();
    const es = new EventSource(`/api/run/${jobId}/stream`); this.es = es;
    es.onmessage = (ev) => { term.textContent += ev.data + "\n"; term.scrollTop = term.scrollHeight; };
    es.addEventListener("end", async () => {
      es.close();
      const st = await api(`/api/run/${jobId}/status`).catch(() => ({}));
      if (st.state === "done") {
        let txt = "done";
        if (st.loss != null && isFinite(st.loss)) txt += ` · loss ${Number(st.loss).toFixed(2)}`;
        if (st.iterations) txt += ` · ${st.iterations} iters`;
        if (st.mcmc) txt += ` · MCMC accept ${Number(st.mcmc.acceptance).toFixed(2)} (${st.mcmc.n_samples} samples)`;
        state.textContent = txt; state.className = "term-state done";
        this.showResults(jobId, st);
      } else { state.textContent = st.state || "ended"; state.className = "term-state error"; }
    });
    es.onerror = () => { state.textContent = "stream error"; state.className = "term-state error"; es.close(); };
  },
  showResults(jobId, st) {
    const area = $("#result-area"); area.innerHTML = "";
    const figs = [];
    if (st.triptych) figs.push(["Result", st.triptych]);
    if (st.mcmc && st.mcmc.corner) figs.push(["MCMC corner", st.mcmc.corner]);
    if (st.mcmc && st.mcmc.trace) figs.push(["MCMC trace", st.mcmc.trace]);
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
  tabs: [], active: null, changeCb: null, panel: "explorer",

  init() {
    $$(".icon-btn").forEach((b) => b.onclick = () => this.setPanel(b.dataset.panel));
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
  },
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
      items.push({ label: "New file…", action: () => this.newEntry("file", node.path) });
      items.push({ label: "New folder…", action: () => this.newEntry("folder", node.path) });
      items.push("-");
      items.push({ label: "Import glafic → glade…", action: () => this.importFile(node.path) });
      if (node.path) {
        items.push("-");
        items.push({ label: "Rename…", action: () => this.renameEntry(node) });
        items.push({ label: "Delete folder", action: () => this.deleteEntry(node) });
      }
    } else {
      items.push({ label: "Open", action: () => this.openFile(node.path) });
      items.push({ label: "Import glafic → glade…", action: () => this.importFile(node.path) });
      items.push({ label: "Export glade → glafic…", action: () => this.exportFile(node) });
      items.push("-");
      items.push({ label: "Rename…", action: () => this.renameEntry(node) });
      items.push({ label: "Delete", action: () => this.deleteEntry(node) });
    }
    ctxMenu(e.clientX, e.clientY, items);
  },
  async newEntry(type, dir) {
    const name = await prompt2(type === "folder" ? "New folder name" : "New file name",
      type === "file" ? "untitled.dat" : "new_folder");
    if (!name) return;
    const path = dir ? `${dir}/${name}` : name;
    try { await apiJSON("/api/files/create", { path, type }); await this.loadTree();
      if (type === "file") this.openFile(path); }
    catch (e) { alert(e.message); }
  },
  async renameEntry(node) {
    const name = await prompt2("Rename", node.name);
    if (!name || name === node.name) return;
    try { const r = await apiJSON("/api/files/rename", { path: node.path, name }); await this.loadTree();
      const t = this._tab(node.path); if (t) { t.path = r.path; this.renderTabs(); } }
    catch (e) { alert(e.message); }
  },
  async deleteEntry(node) {
    const ok = await confirm2("Delete", `Delete <b>${node.name}</b>? This cannot be undone.`, true);
    if (!ok) return;
    try { await apiJSON("/api/files/delete", { path: node.path }); this.closeTab(node.path); await this.loadTree(); }
    catch (e) { alert(e.message); }
  },
  async importFile(path) {
    try { const r = await apiJSON("/api/files/import", { path }); await this.loadTree();
      modal({ title: "Imported", bodyHTML: `Wrote:<br><div class="deflist">${r.written.join("<br>")}</div>`,
        actions: [{ label: "OK", value: 1, cls: "primary" }] }); }
    catch (e) { alert("Import failed: " + e.message); }
  },
  async exportFile(node) {
    const base = await prompt2("Export to glafic — output name",
      node.name.replace(/\.[^.]*$/, "") + "_glafic");
    if (!base) return;
    try { const r = await apiJSON("/api/files/export", { files: [node.path], name: base });
      await this.loadTree();
      modal({ title: "Exported to glafic", bodyHTML: `Wrote:<br><div class="deflist">${r.written.join("<br>")}</div>`,
        actions: [{ label: "OK", value: 1, cls: "primary" }] }); }
    catch (e) { alert("Export failed: " + e.message); }
  },

  // ---- templates ----
  async loadTemplates() {
    const tree = await api("/api/templates"); const host = $("#template-tree"); host.innerHTML = "";
    tree.forEach((grp) => {
      const row = document.createElement("div"); row.className = "tree-row open";
      row.innerHTML = `<span class="twisty">▶</span><span class="ico">▤</span><span class="label">${grp.name}</span>`;
      const kids = document.createElement("div"); kids.className = "tree-children open";
      (grp.children || []).forEach((leaf) => {
        const lr = document.createElement("div");
        lr.className = "tree-row" + (leaf.disabled || !leaf.snippet ? " disabled" : "");
        lr.style.paddingLeft = "24px";
        lr.innerHTML = `<span class="ico">›</span><span class="label">${leaf.name}</span>`;
        if (!leaf.disabled && leaf.snippet) lr.onclick = () => this.insertTemplate(leaf);
        kids.appendChild(lr);
      });
      row.onclick = () => { row.classList.toggle("open"); kids.classList.toggle("open"); };
      host.appendChild(row); host.appendChild(kids);
    });
  },
  insertTemplate(leaf) {
    if (!this.active) { alert("Open or create a file first."); return; }
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
          { model: this.mModels[path], theme: "glade-dark", automaticLayout: true,
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
        d.textContent = "Open a file from the Explorer, or insert a Template."; h.appendChild(d); }
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
    const t = this._tab(path); if (!t) return;
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
    if (t.dirty) confirm2("Close without saving?", `<b>${path}</b> has unsaved changes.`, true).then((ok) => ok && doClose());
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
      const t = this._tab(this.active); if (t) { t.content = this.ta.value; t.dirty = true; }
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
    const t = this._tab(this.active);
    try { await apiJSON("/api/files/write", { path: t.path, content: this.getValue() });
      t.content = this.getValue(); t.dirty = false; this.renderTabs(); this.loadTree(); FindImage.refresh && FindImage.refresh(); }
    catch (e) { alert("Save failed: " + e.message); }
  },
  async deleteActive() {
    if (!this.active) return;
    const ok = await confirm2("Delete file", `Delete <b>${this.active}</b>? This cannot be undone.`, true);
    if (!ok) return;
    try { await apiJSON("/api/files/delete", { path: this.active }); const p = this.active; this.closeTab(p); this.loadTree(); }
    catch (e) { alert(e.message); }
  },
};

// ===================== boot =====================
FindImage.init();
Editor.init();
