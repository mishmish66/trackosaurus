import { BASE, Data, clearCache, mediaURL } from "./data.js";
import { BAND_LABEL, Chart, DENSITY_AUTO, USE_GL, fmt, fmtDur, fmtSI } from "./plot.js";

const PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
                 "#9c755f", "#bab0ac", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#17becf", "#bcbd22"];
const SIDE_ROW = 24; // px height of a sidebar row
const TIP_ROWS = 14; // value rows shown in the tooltip
const PINNED = "\0pinned"; // section of pinned charts
const SPREAD_SHOWN = 8; // values listed per config key that varies
const FRAME_BUDGET_MS = 12; // chart drawing per frame
const PLAN_IDLE_MS = 250; // tile planning interval while the view is unchanged
const OUTLIERS = [[0, "off"], [0.01, "1–99%"], [0.05, "5–95%"]];
const $ = (s) => document.querySelector(s);
const h = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k in e && k !== "style") e[k] = v;
    else e.setAttribute(k, v);
  }
  e.append(...kids.filter((k) => k != null));
  return e;
};
const esc = (s) => String(s ?? "");
const opt = (value, text, sel) => h("option", { value, textContent: text, selected: sel });

/** [user@]host:path in scp form; its first group is the host. */
const REMOTE = /^((?:[^@/:\s]+@)?(?:\[[^\]\s]+\]|[^@/:\s[\]]+)):(.+)$/;

/** The last `n` characters of `s`, after an ellipsis when cut. */
const tailOf = (s, n) => (s.length > n ? `…${s.slice(1 - n)}` : s);

/** {daemon, roots, history} of the server: its directories when it is the daemon. */
const daemonInfo = async () => (await fetch("/api/daemon", { cache: "no-store" })).json();

/** Group-by field id from a declared default: "subfolder", "parent", "config.<key>", or a bare config key. */
const normalizeField = (f) => (["subfolder", "parent", "dir"].includes(f) || f.startsWith("config.") ? f : `config.${f}`);

function fmtAny(v) {
  if (typeof v === "number") return fmt(v);
  if (v === null || typeof v !== "object") return String(v);
  return JSON.stringify(v);
}

/** Key/value table of a (nested) dict; nested dicts and lists of dicts become collapsible rows. */
function kvTree(o, sort) {
  const keys = Object.keys(o);
  if (sort) keys.sort(cmpNames);
  return h("table", { className: "kv" }, ...keys.map((k) => {
    const v = o[k];
    let cell;
    if (v && typeof v === "object" && (!Array.isArray(v) || v.some((x) => x && typeof x === "object"))) {
      const n = Object.keys(v).length;
      cell = h("details", { open: n <= 6 }, h("summary", { className: "muted", textContent: Array.isArray(v) ? `[${n}]` : `{${n}}` }),
        kvTree(Array.isArray(v) ? Object.fromEntries(v.map((x, i) => [i, x])) : v, sort));
    } else {
      const text = fmtAny(v);
      cell = h("span", { className: text.length > 60 || text.includes("\n") ? "long" : "", textContent: text });
    }
    return h("tr", {}, h("td", { textContent: k }), h("td", {}, cell));
  }));
}
/** Runs' configs as integer codes per key (0: absent, 1: null), so a spread is a typed-array count. */
class ConfigIndex {
  constructor() {
    this.keys = new Map(); // key -> {codes: Map(json -> code), json, value, num (per code), col: Int32Array}
    this.slots = new Map(); // run id -> slot
  }

  encode(r) {
    if (r.cfgOf === r.meta) return;
    r.cfgOf = r.meta;
    let i = this.slots.get(r.id);
    if (i === undefined) this.slots.set(r.id, (i = this.slots.size));
    r.cfgSlot = i;
    const cfg = r.meta.config || {};
    for (const [k, e] of this.keys) if (!(k in cfg) && i < e.col.length) e.col[i] = 0;
    for (const [k, val] of Object.entries(cfg)) {
      let e = this.keys.get(k);
      if (!e) this.keys.set(k, (e = { codes: new Map(), json: ["null", "null"], value: [null, null], num: [NaN, NaN], col: new Int32Array(64) }));
      if (i >= e.col.length) e.col = grown(e.col, i + 1);
      const j = JSON.stringify(val ?? null);
      let code = j === "null" ? 1 : e.codes.get(j);
      if (code === undefined) {
        e.codes.set(j, (code = e.json.length));
        e.json.push(j);
        e.value.push(val);
        e.num.push(typeof val === "number" ? val : NaN);
      }
      e.col[i] = code;
    }
  }

  /** [shared, varies] over the config keys any of `runs` has: its one value, or a summary of its values
   * (absent counts as null). */
  spread(runs) {
    const n = runs.length, slots = new Int32Array(n);
    for (let i = 0; i < n; i++) this.encode(runs[i]), (slots[i] = runs[i].cfgSlot);
    const shared = {}, varies = {}, need = this.slots.size;
    for (const [k, e] of this.keys) {
      if (e.col.length < need) e.col = grown(e.col, need);
      const cnt = new Uint32Array(e.json.length), col = e.col;
      for (let i = 0; i < n; i++) cnt[col[slots[i]]]++;
      if (cnt[0] === n) continue;
      cnt[1] += cnt[0];
      const vals = [];
      for (let c = 1; c < cnt.length; c++) if (cnt[c]) vals.push(c);
      if (vals.length === 1) shared[k] = e.value[vals[0]];
      else {
        // numbers in numeric order, before other values in name order
        const cmp = (a, b) => {
          const x = e.num[a], y = e.num[b];
          if (x === x || y === y) return x === x && y === y ? x - y : x === x ? -1 : 1;
          return cmpNames(e.json[a], e.json[b]) || (e.json[a] < e.json[b] ? -1 : 1);
        };
        varies[k] = `${vals.length} values: ` + smallest(vals, SPREAD_SHOWN, cmp).map((c) => `${e.json[c]} ×${cnt[c]}`).join(", ") +
          (vals.length > SPREAD_SHOWN ? ", …" : "");
      }
    }
    return [shared, varies];
  }
}

/** The k smallest items of `a` under `cmp`, in order (a linear pass when k is small). */
function smallest(a, k, cmp) {
  if (a.length <= 4 * k) return a.slice().sort(cmp).slice(0, k);
  const out = [];
  for (const x of a) {
    if (out.length === k && cmp(x, out[k - 1]) >= 0) continue;
    let i = out.length === k ? k - 1 : out.length;
    while (i > 0 && cmp(x, out[i - 1]) < 0) (out[i] = out[i - 1]), i--;
    out[i] = x;
  }
  return out;
}

/** A copy of typed array `a` with room for at least `n` entries (new entries 0). */
function grown(a, n) {
  const b = new a.constructor(Math.max(2 * a.length, n));
  b.set(a);
  return b;
}

/** What the run filter matches: name, path, tags and key=value config entries. */
function searchText(m) {
  const cfg = Object.entries(m.config || {}).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`);
  return [m.name, m.id, ...(m.dir ? [`dir=${m.dir}`] : []), ...(m.tags || []), ...cfg].join(" ");
}

/** Sorted copy of a section and its subsections: panels by name, subsections by sectionCmp. */
function orderSection(sec) {
  const byName = (a, b) => cmpNames(a[0], b[0]);
  const children = [...sec.children.values()].map(orderSection).sort(sectionCmp);
  return { ...sec, items: sec.items.sort(byName), children, media: sec.items.every(([, kind]) => kind === "media") && children.every((c) => c.media) };
}

/** Sections with charts before media-only ones, then by name. */
function sectionCmp(a, b) {
  return a.media - b.media || cmpNames(a.title, b.title);
}

/** Open page `url` after refetching it past the browser's HTTP cache, so a cached redirect cannot divert it. */
function openPage(url) {
  fetch(url, { cache: "reload" }).catch(() => null).finally(() => (location.href = url));
}

/** Page options from the URL hash. */
function hashOpts(q) {
  return {
    path: q.get("path") || "",
    filter: q.get("filter") || "",
    group: q.get("fgroup") ? [] : (q.get("group") || "").split(",").filter(Boolean),
    focus: focusOpt(q),
    chart: q.get("chart") || "",
    center: q.get("center") || "median",
    band: q.get("band") || "ci",
    keys: q.get("keys") || "",
    sort: q.get("sort") || "created",
    dir: q.get("dir") || "desc",
  };
}
const NAV_OPTS = ["path", "focus", "chart", "group"]; // changed by navigation, restored by back/forward

/** Opened groups, outermost first: [[group-by fields, value], …] from `focus` (JSON), or a single `fgroup` of
 * `group`. */
function focusOpt(q) {
  if (q.get("fgroup")) return [[(q.get("group") || "").split(",").filter(Boolean), q.get("fgroup")]];
  try {
    const f = JSON.parse(q.get("focus") || "[]");
    return Array.isArray(f) ? f.filter((l) => Array.isArray(l?.[0]) && typeof l[1] === "string") : [];
  } catch {
    return [];
  }
}

const cmpNames = (a, b) => a.localeCompare(b, undefined, { numeric: true });
function hashStr(s) {
  let x = 2166136261;
  for (let i = 0; i < s.length; i++) x = Math.imul(x ^ s.charCodeAt(i), 16777619);
  return x >>> 0;
}
/** Call fn now, then at most once per `ms` while calls keep coming (the last call always runs). */
function throttle(fn, ms) {
  let t = null, again = false;
  const tick = () => {
    if (again) (again = false), fn(), (t = setTimeout(tick, ms));
    else t = null;
  };
  return () => {
    if (t) again = true;
    else fn(), (t = setTimeout(tick, ms));
  };
}
const store = {
  get(k, d) {
    try {
      const v = localStorage.getItem(k);
      return v == null ? d : JSON.parse(v);
    } catch {
      return d;
    }
  },
  set(k, v) {
    try {
      localStorage.setItem(k, JSON.stringify(v));
    } catch {}
  },
};

/** Floating popover anchored under a button; closes on outside click or Escape. */
const menu = {
  el: null,
  anchor: null,
  open(anchor, content) {
    this.el = this.el || $("#menu");
    this.anchor = anchor;
    this.el.replaceChildren(content);
    this.el.hidden = false;
    const r = anchor.getBoundingClientRect();
    const w = this.el.offsetWidth;
    this.el.style.left = `${Math.max(4, Math.min(r.left, innerWidth - w - 8))}px`;
    this.el.style.top = `${r.bottom + 4}px`;
    this.el.querySelector("input[type=search]")?.focus();
  },
  close() {
    if (this.el) this.el.hidden = true;
    this.anchor = null;
  },
  /** items: [{label, sub, color, icon, active, onpick}] */
  list(anchor, { title, items, search }) {
    const ul = h("div", { className: "mlist" });
    const render = (q) => {
      const ql = q.toLowerCase();
      ul.replaceChildren(...items.filter((it) => !ql || it.label.toLowerCase().includes(ql)).slice(0, 500).map((it) =>
        h("button", { className: "mitem" + (it.active ? " active" : ""), onclick: () => it.onpick() },
          it.icon ? h("span", { className: "micon", textContent: it.icon })
            : h("span", { className: "sw", style: `background:${it.color || "transparent"}` }),
          h("span", { className: "ml", textContent: it.label }),
          it.sub != null ? h("span", { className: "ms", textContent: it.sub }) : null)));
    };
    render("");
    this.open(anchor, h("div", {},
      title ? h("div", { className: "mtitle", textContent: title }) : null,
      search ? h("input", { type: "search", placeholder: "search…", oninput: (e) => render(e.target.value) }) : null,
      ul));
  },
};
document.addEventListener("mousedown", (e) => {
  if (menu.el && !menu.el.hidden && !menu.el.contains(e.target) && !menu.anchor?.contains(e.target)) menu.close();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Shift" && !e.repeat) window.app?.pinTip();
});
document.addEventListener("keyup", (e) => {
  if (e.key === "Shift") window.app?.unpinTip();
});
window.addEventListener("blur", () => window.app?.unpinTip());
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (menu.el && !menu.el.hidden) return menu.close();
  if (window.app?.opts.chart) window.app.focusChart("");
});

/** Scroll keys move the chart pane unless a text field, select, slider, menu or dialog has focus. */
document.addEventListener("keydown", (e) => {
  if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
  const t = e.target;
  if (t.closest?.("#menu, dialog, select, textarea, [contenteditable]") || (t.matches?.("input") && t.type !== "checkbox")) return;
  const pane = $("#panels");
  const page = pane.clientHeight * 0.9;
  const by = { ArrowDown: 60, ArrowUp: -60, PageDown: page, PageUp: -page, " ": e.shiftKey ? -page : page }[e.key];
  if (by !== undefined) pane.scrollBy({ top: by, behavior: Math.abs(by) > 60 ? "smooth" : "auto" });
  else if (e.key === "Home") pane.scrollTo({ top: 0 });
  else if (e.key === "End") pane.scrollTo({ top: pane.scrollHeight });
  else return;
  e.preventDefault();
});

class App {
  constructor() {
    this.charts = new Map();
    this.mediaPanels = new Map();
    this.xrange = null;
    this.tree = [];
    const q = new URLSearchParams(location.hash.slice(1));
    this.opts = hashOpts(q);
    this.panelCfg = {};
    this.groupFromHash = q.has("group") || q.has("fgroup");
    this.groupSource = null;
    this.data = new Data({
      runs: throttle(() => this.onRuns(), 300),
      data: (keys, r) => this.onData(keys, r),
      keys: throttle(() => this.renderPanels(), 300),
      media: (k) => this.onMedia(k),
      status: (t) => ($("#status").textContent = t),
      replan: () => this.replan(true),
      conn: (live) => {
        $("#conn").className = live ? "live" : "down";
        $("#conn").title = live ? "streaming" : "stream disconnected, retrying";
      },
    });
    this.dirtyMedia = new Set();
    this.mediaThrottle = throttle(() => this.renderMedia(), 500);
    this.io = new IntersectionObserver(
      (es) => {
        for (const e of es) {
          const c = e.target._chart || e.target._media;
          if (c.visible !== e.isIntersecting && e.target._chart) {
            this.replan(true);
            if (!e.isIntersecting && !c.full) c.releaseCanvases();
          }
          c.visible = e.isIntersecting;
          if (c.visible && c.dirty) this.schedule(true);
        }
      },
      { root: $("#panels"), rootMargin: "300px" },
    );
  }

  /** Grouped rendering: group-by fields chosen and the scope is more than one run. */
  get grouped() {
    return this.opts.group.length > 0 && !this.scopeIsRun;
  }

  get scopeIsRun() {
    return this.tree.some(([p]) => p === this.opts.path);
  }

  /** Effective display options for one chart: its overrides on top of the toolbar. */
  panelOpts(key) {
    const o = this.opts, p = this.panelCfg[key] || {};
    return {
      smooth: p.smooth ?? 0,
      xmode: p.x === "runtime" ? 1 : 0,
      logx: p.logx ?? false,
      logy: p.logy ?? false,
      xmin: p.xmin ?? null,
      xmax: p.xmax ?? null,
      ymin: p.ymin ?? null,
      ymax: p.ymax ?? null,
      outliers: p.outliers ?? 0,
      center: p.center ?? o.center,
      band: p.band ?? o.band,
      render: p.render ?? "auto",
    };
  }

  hasPanelOverrides(key) {
    return Object.keys(this.panelCfg[key] || {}).length > 0;
  }

  async start() {
    if (!(await this.enterDaemon())) return;
    await this.data.init();
    const root = this.data.rootKey;
    this.hidden = new Set(store.get(`hidden:${root}`, []));
    this.panelCfg = store.get(`panels:${root}`, {});
    this.collapsed = new Set(store.get(`collapsed:${root}`, []));
    this.pins = store.get(`pins:${root}`, []);
    this.bindControls();
    await this.refreshTree();
    await this.loadScope();
    setInterval(() => this.refreshTree().then(() => this.renderCrumbs()), 15000);
  }

  /** Under the daemon, its state (else null), and the trex brand opening its panel. False when it tracks nothing,
   * after showing the panel in place of the page. */
  async enterDaemon() {
    const d = await daemonInfo();
    this.daemon = d.daemon ? d : null;
    if (!d.daemon) return true;
    const brand = $(".brand");
    brand.classList.add("brandlink");
    brand.title = "workspaces and tracked directories";
    brand.onclick = (e) => this.rootMenu(e.currentTarget);
    if (BASE || d.roots.length) return true;
    this.daemonHome(d);
    return false;
  }

  /** The page of a daemon that tracks nothing: its panel, to add a directory. */
  daemonHome(d) {
    $("#status").textContent = "";
    $("#crumbPath").replaceChildren(h("span", { className: "seg current rootseg" }, h("span", { className: "crumb", textContent: "/" })));
    $("#panels").replaceChildren(h("div", { className: "dhome" }, this.daemonPanel(d, async () => this.daemonHome(await daemonInfo()))));
  }

  async rootMenu(anchor) {
    const d = await daemonInfo();
    this.daemon = d;
    menu.open(anchor, this.daemonPanel(d, () => this.rootMenu(anchor)));
  }

  /** The daemon's workspaces and tracked directories: open, edit or remove one, add a directory (by path, as
   * host:path, or from the remembered ones) or a workspace. `refresh` redraws the panel. */
  daemonPanel(d, refresh) {
    const err = h("div", { className: "merr" });
    const panel = h("div", { className: "dpanel" });
    const show = (...kids) => panel.replaceChildren(...kids.filter((k) => k != null));
    const home = () => show(h("div", { className: "mtitle", textContent: "workspaces" }), ...this.workspaceRows(d, refresh, edit),
      h("button", { className: "mclear", textContent: "+ new workspace", onclick: () => edit(null) }),
      h("div", { className: "mtitle msec", textContent: "tracked directories" }), ...this.trackedRows(d, refresh, err), err,
      this.versionRow(d, err));
    const edit = (ws) => show(this.workspaceEditor(d, ws, home));
    home();
    return panel;
  }

  workspaceRows(d, refresh, edit) {
    return d.workspaces.map((w) => h("div", { className: "mrow" },
      h("button", { className: "mitem" + (w.url === `${BASE}/` ? " active" : ""), onclick: () => openPage(w.url) },
        h("span", { className: "ml", textContent: w.name }), h("span", { className: "ms", textContent: w.members.join(" · ") || "empty" })),
      h("button", { className: "chev", textContent: "✎", title: "edit this workspace", onclick: () => edit(w) }),
      h("button", { className: "chev", textContent: "×", title: "delete this workspace (its directories stay tracked)",
        onclick: () => this.deleteWorkspace(w, refresh) })));
  }

  /** Name and members of a new workspace, or of `ws`; `done` returns to the panel. */
  workspaceEditor(d, ws, done) {
    const err = h("div", { className: "merr" });
    const name = h("input", { type: "text", placeholder: "workspace name", value: ws?.name || "", spellcheck: false });
    const boxes = d.roots.map((r) => h("input", { type: "checkbox", checked: !!ws?.members.includes(r.name), value: r.name }));
    const save = async () => {
      const body = { name: name.value, members: boxes.filter((b) => b.checked).map((b) => b.value), old: ws?.name };
      const r = await fetch("/api/daemon/workspace", { method: "POST", body: JSON.stringify(body) });
      const j = await r.json();
      if (r.ok) openPage(j.url);
      else err.textContent = j.error;
    };
    return h("div", {}, h("div", { className: "mtitle", textContent: ws ? `edit ${ws.name}` : "new workspace" }),
      h("div", { className: "madd" }, name),
      ...d.roots.map((r, i) => h("label", { className: "mitem mcheck", title: r.root }, boxes[i],
        h("span", { className: "ml", textContent: r.name }), h("span", { className: "ms", textContent: tailOf(r.root, 36) }))),
      err, h("div", { className: "mfoot" }, h("button", { textContent: "cancel", onclick: done }), h("button", { textContent: "save", onclick: save })));
  }

  async deleteWorkspace(w, refresh) {
    if (!confirm(`Delete workspace ${w.name}? Its directories stay tracked.`)) return;
    await fetch("/api/daemon/workspace/delete", { method: "POST", body: JSON.stringify({ name: w.name }) });
    if (w.url === `${BASE}/`) openPage("/");
    else refresh();
  }

  /** Rows of the tracked directories, the add box and the remembered directories. */
  trackedRows(d, refresh, err) {
    const note = h("div", { className: "mhint" });
    const add = async (path) => {
      const host = REMOTE.exec(path.trim())?.[1];
      [err.textContent, note.textContent] = ["", host ? `starting trex on ${host}…` : ""];
      const r = await fetch("/api/daemon/add", { method: "POST", body: JSON.stringify({ path: path.trim() }) });
      const j = await r.json();
      note.textContent = "";
      if (r.ok) openPage(j.url);
      else err.textContent = j.error;
    };
    const input = h("input", { type: "text", placeholder: "/path/to/runs, ~/runs or host:path", spellcheck: false,
      onkeydown: (e) => e.key === "Enter" && add(input.value) });
    const served = d.roots.map((r) => h("div", { className: "mrow" },
      h("button", { className: "mitem" + (r.url === `${BASE}/` ? " active" : ""), title: r.error || r.root, onclick: () => openPage(r.url) },
        h("span", { className: "ml", textContent: r.name }),
        h("span", { className: "ms", textContent: tailOf(r.root, 36) + (r.state === "local" || r.state === "connected" ? "" : ` · ${r.state}`) })),
      h("button", { className: "chev", textContent: "×", title: "stop tracking this directory (its files are kept)",
        onclick: () => this.removeRoot(r, refresh) })));
    const recent = d.history.map((path) => h("button", { className: "mitem", title: `track ${path}`, onclick: () => add(path) },
      h("span", { className: "micon", textContent: "+" }), h("span", { className: "ml", textContent: tailOf(path, 56) })));
    const clear = async () => {
      await fetch("/api/daemon/history/clear", { method: "POST", body: "{}" });
      refresh();
    };
    return [served.length ? h("div", { className: "mlist" }, ...served) : h("div", { className: "mhint", textContent: "none tracked" }),
      h("div", { className: "madd" }, input, h("button", { textContent: "add", onclick: () => add(input.value) })), note,
      recent.length ? h("div", { className: "mrecent" }, h("div", { className: "mtitle", textContent: "recent" }),
        h("div", { className: "mlist" }, ...recent), h("button", { className: "mclear", textContent: "clear history", onclick: clear })) : null];
  }

  /** The daemon's trex version, with an update button when it can update itself. */
  versionRow(d, err) {
    const { install: inst, updates: up } = d;
    const label = h("span", { className: "ms", textContent: `trex ${inst.version}${inst.commit ? ` · ${inst.commit.slice(0, 7)}` : ""}`,
      title: up.available ? `updates from ${up.source}` : `no update button: ${up.reason}` });
    const btn = up.available ? h("button", { className: "mclear", textContent: "update", title: `install the newest trex from ${up.source} and restart`,
      onclick: () => this.updateDaemon(btn, err) }) : null;
    return h("div", { className: "mfoot" }, label, btn);
  }

  /** Update the daemon; after it restarts on a new trex, reload the page. */
  async updateDaemon(btn, err) {
    btn.disabled = true;
    btn.textContent = "updating…";
    err.textContent = "";
    const r = await fetch("/api/daemon/update", { method: "POST", body: "{}" });
    const j = await r.json();
    if (!r.ok || !j.updated) {
      btn.disabled = false;
      btn.textContent = "update";
      err.textContent = r.ok ? "already the newest trex" : j.error;
      return;
    }
    btn.textContent = "restarting…";
    for (let i = 0; i < 240; i++) {
      await new Promise((ok) => setTimeout(ok, 500));
      const d = await daemonInfo().catch(() => null);
      if (d?.install && JSON.stringify(d.install) === JSON.stringify(j.to)) break;
    }
    location.reload();
  }

  async removeRoot(r, refresh) {
    if (!confirm(`Stop serving ${r.root}? Its run files are kept.`)) return;
    await fetch("/api/daemon/remove", { method: "POST", body: JSON.stringify({ name: r.name }) });
    if (r.url === `${BASE}/`) openPage("/");
    else refresh();
  }

  async refreshTree() {
    this.tree = await (await fetch(`${BASE}/api/tree`, { cache: "no-store" })).json();
  }

  /** Navigate to folder (or run) `path` relative to the served root. */
  async setPath(path) {
    menu.close();
    this.opts.path = path;
    this.opts.focus = [];
    this.saveHash(true);
    await this.loadScope();
  }

  async loadScope() {
    this.xrange = null;
    for (const c of this.charts.values()) c.el.remove(), c.dispose();
    this.charts.clear();
    for (const m of this.mediaPanels.values()) m.el.remove();
    this.mediaPanels.clear();
    this.saveHash();
    const name = this.data.info?.name || "trex";
    document.title = `${this.opts.path || name} · trex`;
    await this.data.loadScope(this.opts.path);
    if (this.groupFromHash) {
      this.groupFromHash = false;
      this.groupSource = null;
    } else {
      [this.opts.group, this.groupSource] = this.resolveGroup(this.opts.path);
      this.saveHash();
    }
    this.onRuns();
    this.renderPanels();
    if (this.opts.chart && !this.charts.has(this.opts.chart)) this.setOpt("chart", "");
    if (this.opts.chart) this.renderCrumbs();
  }

  /** [fields, source] of the nearest saved or trex_info.json-declared group-by at or above a folder. */
  resolveGroup(path) {
    const saved = store.get(`groupby:${this.data.rootKey}`, {});
    const parts = path ? path.split("/") : [];
    for (let i = parts.length; i >= 0; i--) {
      const p = parts.slice(0, i).join("/");
      if (saved[p]) return [saved[p], { kind: "saved", path: p }];
      const declared = this.data.folders[p]?.trex?.group_by;
      if (Array.isArray(declared)) return [declared.map(normalizeField), { kind: "declared", path: p }];
    }
    return [[], null];
  }

  /** Write the options to the URL hash: a new history entry when `push` (navigation), else in place. */
  saveHash(push = false) {
    const q = new URLSearchParams();
    const defaults = { center: "median", band: "ci", x: "step", sort: "created", dir: "desc" };
    for (const [k, v] of Object.entries(this.opts)) {
      const s = k === "focus" ? (v.length ? JSON.stringify(v) : "") : Array.isArray(v) ? v.join(",") : v === true ? "1" : v;
      if ((s && defaults[k] !== s) || (k === "group" && this.opts.focus.length)) q.set(k, s);
    }
    const url = "#" + q.toString();
    if (push && url !== location.hash) history.pushState(null, "", url);
    else history.replaceState(null, "", url);
  }

  /** Back/forward: return to the folder, group focus and chart focus in the hash, without a reload;
   * a hash that changes other options reloads the page. */
  async restoreFromHash() {
    const q = new URLSearchParams(location.hash.slice(1)), next = hashOpts(q);
    const differs = (k) => JSON.stringify(next[k]) !== JSON.stringify(this.opts[k]);
    if (Object.keys(next).some((k) => !NAV_OPTS.includes(k) && differs(k))) return location.reload();
    menu.close();
    this.unpinTip();
    if (next.path !== this.opts.path) {
      Object.assign(this.opts, { path: next.path, focus: next.focus, chart: next.chart });
      if (q.has("group")) (this.opts.group = next.group), (this.groupFromHash = true);
      return this.loadScope();
    }
    if (differs("focus") || (q.has("group") && differs("group"))) {
      this.opts.focus = next.focus;
      if (q.has("group")) this.opts.group = next.group;
      else [this.opts.group, this.groupSource] = this.resolveGroup(this.opts.path);
      this.onRuns();
    }
    if (next.chart !== this.opts.chart) {
      this.opts.chart = next.chart;
      this.renderPanels();
      this.renderCrumbs();
    }
  }

  setOpt(k, v) {
    this.opts[k] = v;
    this.saveHash();
  }

  saveCollapsed() {
    store.set(`collapsed:${this.data.rootKey}`, [...this.collapsed]);
  }

  bindControls() {
    let sideRaf = 0;
    const side = () => {
      sideRaf = 0;
      const [a, b] = this.sideWin || [0, 0], aside = $("aside");
      const top = $("#runTable").getBoundingClientRect().top - aside.getBoundingClientRect().top + aside.scrollTop;
      const first = (aside.scrollTop - top) / SIDE_ROW, last = first + aside.clientHeight / SIDE_ROW;
      if (first < a + 10 && a > 0 || last > b - 10 && b < (this.sideRows || []).length) this.renderSideWindow();
    };
    $("aside").addEventListener("scroll", () => (sideRaf ||= requestAnimationFrame(side)));
    new ResizeObserver(() => (sideRaf ||= requestAnimationFrame(side))).observe($("aside"));
    const o = this.opts;
    const bind = (sel, key, ev, get, after) => {
      const el = $(sel);
      if (el.type === "checkbox") el.checked = o[key];
      else el.value = o[key];
      el.addEventListener(ev, () => {
        this.setOpt(key, get(el));
        after();
      });
    };
    bind("#runFilter", "filter", "input", (e) => e.value, throttle(() => this.onRuns(), 150));
    bind("#center", "center", "change", (e) => e.value, () => this.redrawAll());
    bind("#band", "band", "change", (e) => e.value, () => this.redrawAll());
    bind("#keyFilter", "keys", "input", (e) => e.value, throttle(() => this.renderPanels(), 150));
    $("#clearCache").addEventListener("click", async () => {
      await clearCache();
      $("#status").textContent = "cache cleared";
    });
    $("#resetZoom").addEventListener("click", () => this.resetZoom());
    $("#groupAdd").addEventListener("click", (e) => this.groupByMenu(e.currentTarget));
    $("#sortBy").addEventListener("change", (e) => {
      this.setOpt("sort", e.target.value);
      this.onRuns();
    });
    $("#sortDir").addEventListener("click", () => {
      this.setOpt("dir", this.opts.dir === "asc" ? "desc" : "asc");
      this.onRuns();
    });
  }

  // ---- path bar ----

  /** Entries directly under folder `prefix`: [{name, path, run (bool), count, state}]. */
  children(prefix) {
    const out = new Map();
    for (const [p, state] of this.tree) {
      if (prefix && !p.startsWith(prefix + "/")) continue;
      const rest = prefix ? p.slice(prefix.length + 1) : p;
      const name = rest.split("/")[0];
      const path = prefix ? `${prefix}/${name}` : name;
      const e = out.get(name) || { name, path, run: false, count: 0, state: null };
      e.count++;
      if (path === p) (e.run = true), (e.state = state);
      out.set(name, e);
    }
    return [...out.values()].sort((a, b) => (a.run - b.run) || cmpNames(a.name, b.name));
  }

  childMenu(anchor, prefix, title) {
    menu.list(anchor, {
      title, search: true,
      items: this.children(prefix).map((e) => ({
        label: e.name, icon: e.run ? "▪" : "▸", sub: e.run ? e.state : `${e.count} runs`, active: e.path === this.opts.path,
        onpick: () => this.setPath(e.path),
      })),
    });
  }

  get rootName() {
    if (this.daemon && !BASE) return "/";
    const here = [...(this.daemon?.workspaces || []), ...(this.daemon?.roots || [])].find((r) => r.url === `${BASE}/`);
    return here?.name || this.data.info?.name || "runs";
  }

  /** Path bar: clicking a segment opens that folder; ▾ switches to a sibling; › opens a child. */
  renderCrumbs() {
    const o = this.opts, depth = o.focus.length;
    const parts = o.path ? o.path.split("/") : [];
    const path = [];
    const sep = () => {
      if (path.length && !(path.length === 1 && this.rootName === "/")) path.push(h("span", { className: "sep", textContent: "/" }));
    };
    const seg = (text, target, { current, siblingsOf, cls }) => {
      const open = target !== o.path ? () => this.setPath(target) : depth ? () => this.focusLevel(0) : () => this.focusChart("");
      const s = h("span", { className: "seg" + (current ? " current" : "") + (cls ? ` ${cls}` : "") },
        h("button", { className: "crumb", title: target || "/", onclick: current ? null : open }, text));
      if (siblingsOf != null) s.append(h("button", { className: "chev", textContent: "▾", title: "switch to a sibling",
        onclick: async (e) => {
          const a = e.currentTarget;
          await this.refreshTree();
          this.childMenu(a, siblingsOf, siblingsOf || this.data.info.name);
        } }));
      sep();
      path.push(s);
    };
    const chart = o.chart && this.charts.has(o.chart);
    seg(this.rootName, "", { current: !parts.length && !depth && !chart, cls: "rootseg" });
    if (this.daemon && BASE) path.unshift(h("span", { className: "seg" }, h("button", { className: "crumb", textContent: "/",
      title: "every tracked directory", onclick: () => openPage("/") })));
    parts.forEach((p, i) => seg(p, parts.slice(0, i + 1).join("/"),
      { current: i === parts.length - 1 && !depth && !chart, siblingsOf: parts.slice(0, i).join("/") }));
    o.focus.forEach((level, i) => {
      const current = i === depth - 1 && !chart, open = i === depth - 1 ? () => this.focusChart("") : () => this.focusLevel(i + 1);
      sep();
      path.push(h("span", { className: "seg fchip" + (current ? " current" : "") },
        h("button", { className: "crumb", title: "an opened group", onclick: current ? null : open }, this.focusLabel(level))));
    });
    if (chart) {
      path.push(h("span", { className: "sep", textContent: "›" }),
        h("span", { className: "seg current fchip" }, h("span", { className: "crumb" }, `chart: ${o.chart}`),
          h("button", { className: "chev", textContent: "×", title: "back to all charts (Esc)", onclick: () => this.focusChart("") })));
    } else if (depth) {
      // an opened group has no children to open
    } else if (!this.scopeIsRun && this.children(o.path).length) {
      path.push(h("button", { className: "chev drill", textContent: "›", title: "open a child folder or run",
        onclick: async (e) => {
          const a = e.currentTarget;
          await this.refreshTree();
          this.childMenu(a, o.path, o.path || this.data.info.name);
        } }));
    }
    $("#crumbPath").replaceChildren(...path);
    this.renderInfo();
  }

  /** Top-of-page panel: a run's info/config/summary, or a folder's or group's info and config spread. */
  renderInfo() {
    const run = this.scopeIsRun && this.data.runs.get(this.opts.path);
    const [header, sections] = run ? this.runInfo(run) : this.scopeInfo();
    const open = store.get("infoOpen", { info: true, folder: true, varies: true });
    const section = ([kind, title, obj, sort]) => h("details", { className: "infosec", open: !!open[kind], ontoggle: (e) => {
      const s = store.get("infoOpen", { info: true, folder: true, varies: true });
      s[kind] = e.target.open;
      store.set("infoOpen", s);
    } }, h("summary", {}, title, h("span", { className: "gcount", textContent: ` ${Object.keys(obj).length}` })), kvTree(obj, sort));
    $("#infoPanel").replaceChildren(h("div", { className: "infohead" }, ...header),
      ...sections.filter(([, , obj]) => obj && Object.keys(obj).length).map(section));
    $("#infoPanel").hidden = false;
  }

  /** [header, sections] of a run's page. A section is [kind, title, dict, sorted]. */
  runInfo(run) {
    const m = run.meta, up = this.opts.path.split("/").slice(0, -1).join("/");
    const header = [h("span", { className: "sw", style: `background:${run.color}` }), h("b", { textContent: m.name }),
      h("span", { className: `st ${m.state}`, textContent: m.state }),
      h("span", { className: "muted", textContent: `${fmtSI(m.summary?._step ?? 0)} steps · ${fmtDur(m.summary?._runtime ?? 0)} · ` +
        `${run.seq} rows · created ${m.created ? new Date(m.created * 1000).toLocaleString() : "?"}` }),
      h("span", { className: "spacer" }),
      h("button", { textContent: `↑ ${up || this.data.info.name}`, onclick: () => this.setPath(up) })];
    return [header, [["info", "info", m.info, false], ["config", "config", m.config, true], ["summary", "summary", m.summary, true]]];
  }

  /** [header, sections] of a folder's or group's page: notes of the folders on its path, and its config spread. */
  scopeInfo() {
    const o = this.opts, members = this.runList.filter((r) => r.match && r.inFocus);
    const where = o.path || this.data.info?.name || "root";
    const outer = [where, ...o.focus.slice(0, -1).map((l) => this.focusLabel(l))].join(" / ");
    const header = [h("b", { textContent: o.focus.length ? this.focusLabel(o.focus.at(-1)) : where }),
      h("span", { className: "muted", textContent: `${o.focus.length ? `group in ${outer} · ` : ""}${members.length} runs` })];
    const parts = o.path ? o.path.split("/") : [];
    const sections = parts.map((_, i) => parts.slice(0, i).join("/")).concat(o.path)
      .filter((p) => this.data.folders[p])
      .map((p) => [p === o.path ? "folder" : "ancestor", `about ${p || this.data.info.name}/`, this.data.folders[p], false]);
    if (members.length > 1) {
      const [shared, varies] = this.configSpread(members);
      sections.push(["varies", `config that varies across these ${members.length} runs`, varies, true],
                    ["shared", "config shared by all of them", shared, true]);
    } else if (members.length === 1) sections.push(["config", "config", members[0].meta.config, true]);
    return [header, sections];
  }


  /** [config keys equal across all runs, {key: "n values: v ×count, …"} for keys that differ]. */
  configSpread(runs) {
    return (this.cfgIndex ||= new ConfigIndex()).spread(runs);
  }


  /** Open a group: grouping by subfolder or parent folder alone makes it a folder (or run), so open that path;
   * otherwise it becomes a level of the path, showing its runs ungrouped. */
  openGroup(name) {
    const [only, more] = this.opts.group, sub = this.opts.path ? `${this.opts.path}/${name}` : name;
    if (!more && only === "subfolder") return this.setPath(sub);
    if (!more && only === "parent" && name !== "." && name !== "∅") return this.setPath(sub);
    menu.close();
    this.opts.focus = [...this.opts.focus, [this.opts.group, name]];
    [this.opts.group, this.groupSource] = [[], null];
    this.saveHash(true);
    this.onRuns();
  }

  /** Go back to the first `depth` opened groups (0: the folder), with the grouping that was on there. */
  focusLevel(depth) {
    menu.close();
    const o = this.opts, inner = o.focus[depth];
    if (inner) {
      o.group = inner[0];
      const [fields, src] = depth ? [null, null] : this.resolveGroup(o.path);
      this.groupSource = JSON.stringify(fields) === JSON.stringify(o.group) ? src : null;
    }
    o.focus = o.focus.slice(0, depth);
    o.chart = "";
    this.saveHash(true);
    this.onRuns();
    this.renderPanels();
  }

  // ---- group by ----

  /** Path relative to the current scope, for display. */
  rel(p) {
    const s = this.opts.path;
    if (!s) return p || ".";
    if (p === s) return ".";
    return p.startsWith(s + "/") ? p.slice(s.length + 1) : p;
  }

  /** Selectable group-by fields: subfolder, parent folder, and config keys that vary across the filtered runs. */
  groupFields() {
    const runs = this.runList.filter((r) => r.match);
    const distinct = (get) => new Set(runs.map((r) => JSON.stringify(get(r) ?? null))).size;
    const fields = [{ id: "subfolder", label: "subfolder", n: distinct((r) => this.subfolder(r)) },
                    { id: "parent", label: "parent folder", n: distinct((r) => r.meta.parent) },
                    ...(runs.some((r) => r.meta.dir) ? [{ id: "dir", label: "tracked dir", n: distinct((r) => r.meta.dir) }] : [])];
    const keys = new Set();
    for (const r of runs) for (const k of Object.keys(r.meta.config || {})) keys.add(k);
    for (const k of [...keys].sort()) {
      const n = distinct((r) => r.meta.config?.[k]);
      if (n > 1) fields.push({ id: `config.${k}`, label: k, n });
    }
    return fields;
  }

  fieldLabel(id) {
    return { subfolder: "subfolder", parent: "parent folder", dir: "tracked dir" }[id] ?? id.slice(7);
  }

  /** First path component of a run below the current folder (the run's own name if it sits directly in it). */
  subfolder(r) {
    return this.rel(r.id).split("/")[0];
  }

  groupFieldLabel(fields = this.opts.group) {
    return fields.map((f) => this.fieldLabel(f)).join(" · ");
  }

  /** "fields: value" of an opened group. */
  focusLabel([fields, value]) {
    return `${this.groupFieldLabel(fields)}: ${value}`;
  }

  groupByMenu(anchor) {
    const sel = this.opts.group;
    menu.list(anchor, {
      title: "group by (click to toggle)", search: true,
      items: this.groupFields().map((f) => ({
        label: f.label, sub: `${f.n} values`, active: sel.includes(f.id),
        onpick: () => {
          this.setGroup(sel.includes(f.id) ? sel.filter((x) => x !== f.id) : [...sel, f.id]);
          this.groupByMenu(anchor);
        },
      })),
    });
  }

  /** Set the group-by: inside an opened group for this view only, else remembered for the current folder (and, by
   * inheritance, its subfolders). */
  setGroup(fields) {
    if (this.opts.focus.length) {
      [this.opts.group, this.groupSource] = [fields, null];
      this.saveHash();
      return this.onRuns();
    }
    const key = `groupby:${this.data.rootKey}`;
    const saved = store.get(key, {});
    saved[this.opts.path] = fields;
    store.set(key, saved);
    this.opts.group = fields;
    this.groupSource = { kind: "saved", path: this.opts.path };
    this.saveHash();
    this.onRuns();
  }

  /** Drop this folder's saved group-by so it inherits again. */
  forgetGroup() {
    const key = `groupby:${this.data.rootKey}`;
    const saved = store.get(key, {});
    delete saved[this.opts.path];
    store.set(key, saved);
    [this.opts.group, this.groupSource] = this.resolveGroup(this.opts.path);
    this.saveHash();
    this.onRuns();
  }

  renderGroupChips() {
    $("#groupChips").replaceChildren(...this.opts.group.map((f) =>
      h("span", { className: "chip" }, this.fieldLabel(f),
        h("button", { textContent: "×", title: "remove", onclick: () => this.setGroup(this.opts.group.filter((x) => x !== f)) }))));
    $("#groupAdd").textContent = this.opts.group.length ? "+" : "+ add field";
    const src = this.groupSource;
    const here = src && src.path === this.opts.path;
    const where = src && (src.path || this.data.info?.name || "root");
    $("#groupSource").replaceChildren(...(!src ? [] : [
      h("span", { textContent: src.kind === "declared" ? `default of ${here ? "this folder" : where + "/"}`
        : here ? "saved for this folder" : `inherited from ${where}/` }),
      src.kind === "saved" && here ? h("button", { className: "linkish", textContent: "reset", title: "forget this folder's choice",
        onclick: () => this.forgetGroup() }) : null,
    ].filter(Boolean)));
  }

  groupValue(r, fields = this.opts.group) {
    if (!fields.length) return null;
    const m = r.meta;
    return fields.map((f) => {
      const v = f === "subfolder" ? this.subfolder(r) : f === "parent" ? this.rel(m.parent) : f === "dir" ? m.dir : m.config?.[f.slice(7)];
      return v == null ? "∅" : typeof v === "object" ? JSON.stringify(v) : String(v);
    }).join(" · ");
  }

  // ---- runs ----

  runFilterFn() {
    const f = this.opts.filter.trim();
    if (!f) return () => true;
    let rx;
    try {
      rx = new RegExp(f, "i");
    } catch {
      const l = f.toLowerCase();
      return (r) => r.search.toLowerCase().includes(l);
    }
    return (r) => rx.test(r.search);
  }

  /** Runs in display order, with filter, group focus, visibility, group and color resolved. */
  computeRuns() {
    const o = this.opts, match = this.runFilterFn(), runs = [...this.data.runs.values()];
    for (const r of runs) {
      if (r.searchOf !== r.meta) (r.searchOf = r.meta), (r.search = searchText(r.meta));
      r.match = match(r);
      r.gval = this.groupValue(r);
      r.focused = o.focus.every(([fields, value]) => this.groupValue(r, fields) === value);
    }
    if (o.focus.length && runs.length && !runs.some((r) => r.focused)) o.focus = [];
    for (const r of runs) {
      r.inFocus = r.match && r.focused;
      r.shown = r.inFocus && (!this.hidden.has(r.id) || this.scopeIsRun);
    }
    runs.sort((a, b) => this.compare(this.sortValue(a), this.sortValue(b)) || (b.meta.created || 0) - (a.meta.created || 0));
    this.groups = o.group.length ? this.buildGroups(runs) : new Map();
    this.assignColors(runs);
    this.runList = runs;
  }

  /** A run's color: its group's, else the next palette color among shown runs (hidden ones by id). */
  assignColors(runs) {
    let ci = 0;
    for (const r of runs) {
      r.color = this.opts.group.length ? this.groups.get(r.gval)?.color ?? "#999"
        : r.shown ? PALETTE[ci++ % PALETTE.length] : PALETTE[hashStr(r.id) % PALETTE.length];
    }
  }

  /** Groups of the matching runs by group value, in sort order; colors follow name order, so they stay put when the sort changes. */
  buildGroups(runs) {
    const groups = new Map();
    const vals = [...new Set(runs.filter((r) => r.match).map((r) => r.gval))].sort(cmpNames);
    vals.forEach((v, i) => groups.set(v, { name: v, color: PALETTE[i % PALETTE.length], runs: [] }));
    for (const r of runs) if (r.match) groups.get(r.gval).runs.push(r);
    const sorted = [...groups.values()].map((g) => [g, this.groupSortValue(g.runs, g.name)])
      .sort((a, b) => this.compare(a[1], b[1]) || cmpNames(a[0].name, b[0].name));
    return new Map(sorted.map(([g]) => [g.name, g]));
  }


  /** Metrics logged by a run that passes the filter and group focus (null: every metric, when all
   * runs pass); true when that set changed. */
  updateScopeKeys() {
    const runs = this.runList.filter((r) => r.inFocus);
    let keys = null;
    if (runs.length < this.runList.length) {
      keys = new Set();
      for (const r of runs) for (const k of r.meta.keys || []) keys.add(k);
    }
    const sig = keys ? [...keys].sort().join("\0") : null;
    if (sig === this.scopeKeySig) return false;
    this.scopeKeySig = sig;
    this.scopeKeys = keys;
    return true;
  }

  /** Series to draw for a metric: one per shown run, or one per group (list of column ids). */
  linesFor(key) {
    const out = [];
    if (!this.grouped) {
      for (const r of this.runList) {
        const c = r.shown ? r.cols.get(key) : undefined;
        if (c !== undefined) out.push({ cols: [c], color: r.color, label: r.meta.name, run: r });
      }
      return out;
    }
    for (const g of this.groups.values()) {
      const cols = [];
      for (const r of g.runs) if (r.shown && r.cols.has(key)) cols.push(r.cols.get(key));
      if (cols.length) out.push({ cols, color: g.color, label: `${g.name} (${cols.length})`, group: g.name });
    }
    return out;
  }

  onRuns() {
    this.computeRuns();
    if (this.updateScopeKeys()) this.renderPanels();
    this.redrawAll();
    this.replan(true);
    this.afterPaint([
      () => this.renderCrumbs(),
      () => this.renderGroupChips(),
      () => this.renderRunTable(),
      () => {
        for (const k of this.mediaPanels.keys()) this.dirtyMedia.add(k);
        this.mediaThrottle();
      },
    ]);
  }

  /** Run fns, one task each, once the next frame (with the charts) has painted; a later call
   * replaces pending fns. */
  afterPaint(fns) {
    this.afterFns = fns;
    if (this.afterRaf) return;
    const next = () => {
      const f = this.afterFns.shift();
      f?.();
      if (this.afterFns.length) setTimeout(next, 0);
      else this.afterRaf = 0;
    };
    this.afterRaf = requestAnimationFrame(() => setTimeout(next, 0));
  }

  /** Sort key of a run under the current sort field; undefined sorts last. */
  sortValue(r) {
    const k = this.opts.sort, m = r.meta, s = m.summary || {};
    const v = k === "created" ? m.created : k === "name" ? m.name : k === "state" ? m.state
      : k === "step" ? s._step : k === "runtime" ? s._runtime : k.startsWith("metric:") ? s[k.slice(7)] : m.created;
    if (typeof v === "string" && k.startsWith("metric:")) return Number(v);
    return v;
  }

  /** Comparator honoring direction, with missing values last in either direction. */
  compare(a, b) {
    const missA = a == null || (typeof a === "number" && Number.isNaN(a));
    const missB = b == null || (typeof b === "number" && Number.isNaN(b));
    if (missA || missB) return missA - missB;
    const c = typeof a === "number" && typeof b === "number" ? a - b : cmpNames(String(a), String(b));
    return this.opts.dir === "asc" ? c : -c;
  }

  /** Sort key of a set of runs (group or folder): its name, its size, or the median of its runs' keys. */
  groupSortValue(runs, name) {
    const k = this.opts.sort;
    if (k === "name") return name;
    if (k === "size") return runs.length;
    const vals = runs.map((r) => this.sortValue(r)).filter((v) => v != null && !(typeof v === "number" && Number.isNaN(v)));
    if (!vals.length) return undefined;
    vals.sort((a, b) => (typeof a === "number" ? a - b : cmpNames(String(a), String(b))));
    return vals[(vals.length - 1) >> 1];
  }

  renderSortOptions() {
    const sel = $("#sortBy");
    const base = [["created", "created"], ["name", "name"], ["state", "state"], ["step", "steps"], ["runtime", "runtime"],
                  ["size", this.opts.group.length ? "group size" : "folder size"]];
    const metrics = [...this.data.keys.keys()].sort(cmpNames);
    const sig = base.map((x) => x.join("=")).join("|") + "#" + metrics.join("|");
    if (sel._sig !== sig) {
      sel._sig = sig;
      sel.replaceChildren(...base.map(([v, t]) => opt(v, t)),
        h("optgroup", { label: "metric (last value)" }, ...metrics.map((k) => opt(`metric:${k}`, k))));
    }
    if (![...sel.options].some((x) => x.value === this.opts.sort)) this.opts.sort = "created";
    sel.value = this.opts.sort;
    $("#sortDir").textContent = this.opts.dir === "asc" ? "↑" : "↓";
    $("#sortDir").title = this.opts.dir === "asc" ? "ascending" : "descending";
  }

  /** Folder tree of `runs` relative to the scope: {name, path, dirs: Map, runs: [], all: []}. */
  buildTree(runs) {
    const scope = this.opts.path;
    const root = { name: "", path: scope, dirs: new Map(), runs: [], all: [] };
    for (const r of runs) {
      const rel = this.rel(r.id);
      const parts = rel === "." ? [] : rel.split("/").slice(0, -1);
      let node = root;
      node.all.push(r);
      for (const part of parts) {
        let next = node.dirs.get(part);
        if (!next) node.dirs.set(part, (next = { name: part, path: node.path ? `${node.path}/${part}` : part, dirs: new Map(), runs: [], all: [] }));
        node = next;
        node.all.push(r);
      }
      node.runs.push(r);
    }
    return root;
  }

  renderRunTable() {
    this.renderSortOptions();
    const rows = []; // row factories in display order
    const metric = this.opts.sort.startsWith("metric:") ? this.opts.sort.slice(7) : null;
    const fmtVal = (v) => (v == null ? "" : typeof v === "number" ? fmt(v) : String(v));
    const saveHidden = () => store.set(`hidden:${this.data.rootKey}`, [...this.hidden]);
    const setHidden = (runs, hide) => {
      for (const r of runs) hide ? this.hidden.add(r.id) : this.hidden.delete(r.id);
      saveHidden();
      this.onRuns();
    };
    const toggle = (key) => {
      this.collapsed.has(key) ? this.collapsed.delete(key) : this.collapsed.add(key);
      this.saveCollapsed();
      this.renderRunTable();
    };
    const runRow = (r, depth) => {
      const s = r.meta.summary || {};
      return h("tr", { className: (r.match ? "" : "nomatch") + (this.sideMark === r.id ? " mark" : "") },
        h("td", { className: "tw" }),
        h("td", {}, h("input", { type: "checkbox", checked: !this.hidden.has(r.id), onchange: (e) => setHidden([r], !e.target.checked) })),
        h("td", {}, h("span", { className: "sw", style: `background:${r.color}` })),
        h("td", { className: "name", title: r.id, style: `padding-left:${4 + depth * 14}px` },
          h("a", { href: "#", textContent: esc(r.meta.name), onclick: (e) => {
            e.preventDefault();
            this.setPath(r.id);
          } })),
        h("td", { className: "stc" }, h("span", { className: `dot ${r.meta.state}`, title: r.meta.state || "" })),
        metric ? h("td", { className: "num val", textContent: fmtVal(this.sortValue(r)) })
          : h("td", { className: "num", textContent: s._step != null ? fmtSI(s._step) : "" }),
        h("td", { className: "num", textContent: s._runtime != null ? fmtDur(s._runtime) : "" }),
      );
    };
    /** A collapsible header row for a folder or group. */
    const headRow = ({ key, label, members, color, depth, open, openTitle, onOpen }) => {
      const vis = members.filter((r) => !this.hidden.has(r.id)).length;
      return h("tr", { className: "grp" + (open ? " open" : "") + (this.sideMark === key ? " mark" : ""), title: open ? "click to collapse" : "click to expand",
        onclick: (e) => !e.target.closest("input, button") && toggle(key) },
        h("td", { className: "tw" }, h("span", { className: "caret", textContent: "▸" })),
        h("td", {}, h("input", { type: "checkbox", checked: vis > 0, indeterminate: vis > 0 && vis < members.length,
          onchange: (e) => setHidden(members, !e.target.checked) })),
        h("td", {}, color ? h("span", { className: "sw", style: `background:${color}` }) : h("span", { className: "folder", textContent: "▤" })),
        h("td", { className: "gname", colSpan: 2, title: label, style: `padding-left:${4 + depth * 14}px` }, label,
          h("span", { className: "gcount", textContent: ` ${members.length}` })),
        h("td", { className: "num val", textContent: metric ? fmtVal(this.groupSortValue(members, label)) : "" }),
        h("td", { className: "num" }, h("button", { className: "gfocus", textContent: "open ›", title: openTitle, onclick: onOpen })));
    };
    const inScope = this.runList.filter((r) => r.match && r.inFocus);
    const heads = [];
    if (this.opts.group.length) {
      const scopeSet = new Set(inScope);
      for (const g of this.groups.values()) {
        const members = g.runs.filter((r) => scopeSet.has(r));
        if (!members.length) continue;
        const key = `group:${g.name}`;
        const open = !this.collapsed.has(key);
        heads.push(key);
        rows.push(Object.assign(() => headRow({ key, label: g.name, members, color: g.color, depth: 0, open,
          openTitle: "open this group", onOpen: () => this.openGroup(g.name) }), { id: key }));
        if (open) for (const r of members) rows.push(Object.assign(() => runRow(r, 0), { id: r.id }));
      }
    } else {
      const walk = (node, depth) => {
        const dirs = [...node.dirs.values()].map((d) => [d, this.groupSortValue(d.all, d.name)])
          .sort((a, b) => this.compare(a[1], b[1]) || cmpNames(a[0].name, b[0].name));
        for (const [d] of dirs) {
          const key = `folder:${d.path}`;
          const open = !this.collapsed.has(key);
          heads.push(key);
          rows.push(Object.assign(() => headRow({ key, label: d.name, members: d.all, depth, open, openTitle: `open ${d.path}`,
            onOpen: () => this.setPath(d.path) }), { id: key }));
          if (open) walk(d, depth + 1);
        }
        for (const r of node.runs) rows.push(Object.assign(() => runRow(r, depth), { id: r.id }));
      };
      walk(this.buildTree(inScope), 0);
    }
    this.sideRows = rows;
    this.renderSideWindow();
    const shown = this.runList.filter((r) => r.shown).length;
    $("#runCount").textContent = `${shown} shown · ${this.runList.length} in ${this.opts.path || this.data.info?.name || "root"}`;
    const ca = $("#collapseAll");
    ca.hidden = !heads.length;
    const anyOpen = heads.some((k) => !this.collapsed.has(k));
    ca.textContent = anyOpen ? "collapse all" : "expand all";
    ca.onclick = () => {
      for (const k of heads) anyOpen ? this.collapsed.add(k) : this.collapsed.delete(k);
      this.saveCollapsed();
      this.renderRunTable();
    };
    const anyVisible = inScope.some((r) => !this.hidden.has(r.id));
    const ha = $("#hideAll");
    ha.textContent = anyVisible ? "hide all" : "show all";
    ha.onclick = () => setHidden(inScope, anyVisible);
  }

  /** Build only the sidebar rows near the scroll position; spacer rows stand in for the rest. */
  renderSideWindow() {
    const aside = $("aside"), table = $("#runTable"), rows = this.sideRows || [];
    const top = table.getBoundingClientRect().top - aside.getBoundingClientRect().top + aside.scrollTop;
    const a = Math.max(0, Math.floor((aside.scrollTop - top) / SIDE_ROW) - 30);
    const b = Math.min(rows.length, a + Math.ceil(aside.clientHeight / SIDE_ROW) + 60);
    const spacer = (n) => h("tr", { className: "spacer" }, h("td", { colSpan: 7, style: `height:${n * SIDE_ROW}px` }));
    const els = [];
    if (a > 0) els.push(spacer(a));
    for (let i = a; i < b; i++) els.push(rows[i]());
    if (b < rows.length) els.push(spacer(rows.length - b));
    $("#runTable tbody").replaceChildren(...els);
    this.sideWin = [a, b];
  }

  // ---- per-chart settings ----

  panelSettings(chart, anchor) {
    const key = chart.key;
    const o = this.opts;
    const cfg = () => this.panelCfg[key] || {};
    const set = (k, v) => {
      const c = { ...cfg() };
      if (v === null || v === undefined || v === "") delete c[k];
      else c[k] = v;
      if (Object.keys(c).length) this.panelCfg[key] = c;
      else delete this.panelCfg[key];
      store.set(`panels:${this.data.rootKey}`, this.panelCfg);
      for (const c of this.chartsOf(key)) c.dirty = true;
      this.schedule(true);
    };
    const num = (k) => h("input", { type: "number", step: "any", placeholder: "auto", value: cfg()[k] ?? "",
      onchange: (e) => set(k, e.target.value === "" ? null : +e.target.value) });
    const sel = (k, def, choices) => h("select", { onchange: (e) => set(k, e.target.value === "" ? null : JSON.parse(e.target.value)) },
      opt("", `default (${def})`, cfg()[k] === undefined),
      ...choices.map(([v, t]) => opt(JSON.stringify(v), t, cfg()[k] === v)));
    const smoothVal = h("span", { className: "muted", textContent: (cfg().smooth ?? 0).toFixed(2) });
    const smoothOn = h("input", { type: "checkbox", checked: cfg().smooth !== undefined });
    const smooth = h("input", { type: "range", min: 0, max: 0.999, step: 0.001, value: cfg().smooth ?? 0, disabled: !smoothOn.checked });
    const applySmooth = () => {
      smooth.disabled = !smoothOn.checked;
      smoothVal.textContent = (+smooth.value).toFixed(2);
      set("smooth", smoothOn.checked ? +smooth.value : null);
    };
    smoothOn.addEventListener("change", applySmooth);
    smooth.addEventListener("input", applySmooth);
    const row = (label, ...els) => h("div", { className: "srow" }, h("label", { textContent: label }), h("div", {}, ...els));
    const form = h("div", { className: "settings" },
      h("div", { className: "mtitle", textContent: key }),
      row("smoothing", h("label", { className: "inl" }, smoothOn, "on"), smooth, smoothVal),
      row("x axis", sel("x", "step", [["step", "step"], ["runtime", "runtime"]]), sel("logx", "linear", [[false, "linear"], [true, "log"]])),
      row("x range", num("xmin"), h("span", { textContent: "to" }), num("xmax")),
      row("y scale", sel("logy", "linear", [[false, "linear"], [true, "log"]])),
      row("y range", num("ymin"), h("span", { textContent: "to" }), num("ymax")),
      row("ignore outliers", sel("outliers", "off", OUTLIERS)),
      row("group line", sel("center", o.center, [["median", "median"], ["mean", "mean"]])),
      row("group band", sel("band", BAND_LABEL[o.band], Object.entries(BAND_LABEL))),
      row("lines", sel("render", "auto", [["lines", "lines"], ["density", "density"]])),
      h("div", { className: "actions" },
        h("button", { textContent: "reset chart", onclick: () => {
          delete this.panelCfg[key];
          store.set(`panels:${this.data.rootKey}`, this.panelCfg);
          for (const c of this.chartsOf(key)) c.dirty = true;
          this.schedule(true);
          this.panelSettings(chart, anchor);
        } }),
        h("button", { textContent: "close", onclick: () => menu.close() })),
      h("div", { className: "muted hint", textContent: "Outlier rejection scales the y axis to the chosen quantiles of the visible values; it does not drop data." }),
      h("div", { className: "muted hint", textContent: `Density draws a heatmap of all runs (WebGL only; ungrouped charts). Auto switches to it above ${DENSITY_AUTO} lines.` }),
    );
    menu.open(anchor, form);
  }

  // ---- panels ----
  // ---- panels ----

  keyFilterFn() {
    const f = this.opts.keys.trim();
    if (!f) return () => true;
    try {
      const rx = new RegExp(f, "i");
      return (k) => rx.test(k);
    } catch {
      return (k) => k.toLowerCase().includes(f.toLowerCase());
    }
  }

  renderPanels() {
    const sections = this.panelSections(), root = $("#panels"), closed = store.get("closedSections", {});
    const sectionBtn = h("button", { className: "sectionsToggle" });
    let nsec = 0;
    const walk = (secs) => secs.forEach((x) => ((nsec += 1), walk(x.children)));
    walk(sections);
    const count = sections.filter((x) => x.id !== PINNED).reduce((n, x) => n + x.n, 0);
    const els = [$("#infoPanel"), h("div", { className: "panelbar" }, sectionBtn,
      h("span", { className: "muted", textContent: `${nsec} sections · ${count} panels` }))];
    els.push(...sections.map((x) => this.sectionEl(x, closed)));
    sectionBtn.addEventListener("click", () => {
      const open = [...root.querySelectorAll("details.section")].some((d) => d.open);
      for (const d of root.querySelectorAll("details.section")) d.open = !open;
    });
    root.replaceChildren(...els);
    const focus = this.opts.chart && this.charts.get(this.opts.chart);
    root.classList.toggle("focus", !!focus);
    if (focus) {
      root.append(h("div", { className: "focusview" }, focus.el));
      root.scrollTop = 0;
    }
    this.updateSectionsToggle();
    this.redrawAll();
    this.renderMedia();
  }

  /** A foldable section: its panels, then its subsections; whether it is folded is remembered by its path. */
  sectionEl(sec, closed) {
    const grid = sec.items.length ? h("div", { className: "grid" }, ...sec.items.map(([key, kind, pinned]) => this.panelEl(key, kind, pinned))) : null;
    return h("details", { className: "section", open: !closed[sec.id], ontoggle: (e) => {
      const c = store.get("closedSections", {});
      c[sec.id] = !e.target.open;
      store.set("closedSections", c);
      this.updateSectionsToggle();
    } }, h("summary", { textContent: `${sec.title} (${sec.n})` }), ...[grid, ...sec.children.map((x) => this.sectionEl(x, closed))].filter(Boolean));
  }

  /** Sections of the panels passing the chart filter, in display order: pinned charts (in pin order; each also
   * stays in its own section), "charts" (keys without a slash), then a section per key prefix, nested by path,
   * media-only sections last at each level. A section is {id (its path), title, items: [[key, kind, pinned]],
   * children, n (panels in it and below)}. */
  panelSections() {
    const kf = this.keyFilterFn(), top = new Map(), pins = [];
    const node = (id, title) => ({ id, title, items: [], children: new Map(), n: 0 });
    const add = (key, kind) => {
      if (!kf(key)) return;
      if (kind === "metric" && this.pins.includes(key)) pins.push([key, kind, true]);
      const parts = key.split("/"), flat = parts.length === 1 || !parts[0], first = flat ? "charts" : parts[0];
      let n = top.get(first) || top.set(first, node(first, first)).get(first);
      n.n++;
      for (let i = 1; !flat && i < parts.length - 1; i++) {
        if (!n.children.has(parts[i])) n.children.set(parts[i], node(parts.slice(0, i + 1).join("/"), parts[i]));
        n = n.children.get(parts[i]);
        n.n++;
      }
      n.items.push([key, kind, false]);
    };
    for (const k of this.data.keys.keys()) if (!this.scopeKeys || this.scopeKeys.has(k)) add(k, "metric");
    for (const k of this.data.media.keys()) add(k, "media");
    const pinOrder = new Map(this.pins.map((k, i) => [k, i]));
    pins.sort((a, b) => pinOrder.get(a[0]) - pinOrder.get(b[0]));
    const sorted = [...top.values()].map(orderSection).sort((a, b) => (a.id !== "charts") - (b.id !== "charts") || sectionCmp(a, b));
    return pins.length ? [{ id: PINNED, title: "📌 pinned", items: pins, children: [], n: pins.length }, ...sorted] : sorted;
  }

  /** The chart or media panel of `key` (the pinned section's own copy when `pinned`), created on first use. */
  panelEl(key, kind, pinned = false) {
    const [map, Make] = kind === "metric" ? [this.charts, Chart] : [this.mediaPanels, MediaPanel];
    const id = pinned ? PINNED + key : key;
    let p = map.get(id);
    if (!p) {
      map.set(id, (p = new Make(this, key)));
      this.io.observe(p.el);
    }
    if (kind === "metric") p.setPinned(this.pins.includes(key));
    return p.el;
  }

  /** The charts of metric `key`: its own and, when pinned, its copy in the pinned section. */
  chartsOf(key) {
    return [this.charts.get(key), this.charts.get(PINNED + key)].filter(Boolean);
  }

  /** Pin a chart to the pinned section at the top (in pin order; it stays in its own section too), or unpin it. */
  togglePin(key) {
    const i = this.pins.indexOf(key);
    if (i >= 0) {
      this.pins.splice(i, 1);
      const copy = this.charts.get(PINNED + key);
      if (copy) this.io.unobserve(copy.el), copy.el.remove(), copy.dispose(), this.charts.delete(PINNED + key);
    } else this.pins.push(key);
    store.set(`pins:${this.data.rootKey}`, this.pins);
    this.renderPanels();
  }

  /** Show one chart filling the chart pane, as a level of the path bar ("" = all charts). */
  focusChart(key) {
    if ((this.opts.chart || "") === key) return;
    this.opts.chart = key;
    this.saveHash(true);
    this.renderPanels();
    this.renderCrumbs();
  }

  /** "collapse all" while any chart section is open, else "expand all". */
  updateSectionsToggle() {
    const btn = $("#panels .sectionsToggle");
    if (!btn) return;
    const secs = [...document.querySelectorAll("#panels details.section")];
    btn.textContent = secs.some((d) => d.open) ? "collapse all sections" : "expand all sections";
    btn.hidden = !secs.length;
  }

  onData(keys, r) {
    if (!keys || !this.runList) return this.redrawAll();
    if (r && !r.shown) return;
    let first = false; // a chart still showing nothing draws on the next frame
    for (const k of keys) {
      for (const c of this.chartsOf(k)) {
        c.dirty = true;
        if ((c.visible || c.full) && !c.view?.lines.length) first = true;
      }
    }
    this.schedule(first);
  }

  redrawAll() {
    for (const c of this.charts.values()) c.dirty = true;
    this.schedule(true);
  }

  /** Draw dirty visible charts on the next frame; streamed updates are coalesced to 4 Hz. */
  schedule(now) {
    if (now) {
      if (!this.raf) this.raf = requestAnimationFrame(() => this.drawDirty());
    } else if (!this.slow) {
      this.slow = setTimeout(() => {
        this.slow = null;
        this.schedule(true);
      }, 250);
    }
  }

  drawDirty() {
    this.raf = null;
    if (!this.runList) return;
    let drew = false;
    const t0 = performance.now();
    for (const c of this.charts.values()) {
      if (!(c.visible || c.full) || !c.dirty) continue;
      // past the frame budget, the remaining charts draw on the next frame
      if (drew && performance.now() - t0 > FRAME_BUDGET_MS) {
        this.schedule(true);
        break;
      }
      c.draw();
      drew = true;
    }
    if (drew) this.replan();
  }

  /** What chart c shows, for `Data.plan`. */
  demand(c, runs) {
    const o = this.panelOpts(c.key), zoom = this.xrange && this.xrange[2] === o.xmode ? this.xrange : null;
    const x0 = o.xmin ?? zoom?.[0] ?? null, x1 = o.xmax ?? zoom?.[1] ?? null;
    return { key: c.key, runs, xmode: o.xmode, zoomed: x0 !== null || x1 !== null, x0: x0 ?? -Infinity, x1: x1 ?? Infinity,
             pw: c.w ? c.pw : 600, densityAbove: this.densityAbove(o) };
  }

  /** Line count above which a chart with options `o` draws a density heatmap. */
  densityAbove(o) {
    if (!USE_GL || this.grouped) return Infinity;
    return o.render === "density" ? 0 : o.render === "auto" ? DENSITY_AUTO : Infinity;
  }

  /** Tell the data layer what the visible charts show: soon if `now`, else within PLAN_IDLE_MS. */
  replan(now = false) {
    const due = performance.now() + (now ? 30 : Math.max(30, PLAN_IDLE_MS - (performance.now() - (this.plannedAt || 0))));
    if (this.planTimer && this.planDue <= due) return;
    clearTimeout(this.planTimer);
    this.planDue = due;
    this.planTimer = setTimeout(() => {
      this.planTimer = null;
      this.plannedAt = performance.now();
      if (!this.runList) return;
      if (this.shownFor !== this.runList) (this.shownFor = this.runList), (this.shown = this.runList.filter((r) => r.shown));
      const demands = new Map(); // one per metric, from its widest visible chart
      for (const c of this.charts.values()) {
        if (!(c.visible || c.full)) continue;
        const d = this.demand(c, this.shown), had = demands.get(d.key);
        if (!had || d.pw > had.pw) demands.set(d.key, d);
      }
      this.data.plan([...demands.values()]);
    }, due - performance.now());
  }

  /** Shared x zoom [x0, x1, xmode], or null. */
  setXRange(r) {
    this.xrange = r;
    this.replan(true);
    this.updateZoomButton();
    this.redrawAll();
  }

  /** The toolbar "reset zoom" shows while any chart is zoomed in x or y. */
  updateZoomButton() {
    $("#resetZoom").hidden = !this.xrange && ![...this.charts.values()].some((c) => c.yzoom);
  }

  resetZoom() {
    for (const c of this.charts.values()) c.yzoom = null;
    this.setXRange(null);
  }

  /** Value tooltip: rows by value, the TIP_ROWS around `near` (the line nearest the pointer). */
  tip(e, chart, xs, rows, near = -1) {
    const t = $("#tip"), key = chart?.key;
    if (!e) return (t.hidden = true);
    const a = near < 0 ? 0 : Math.max(0, Math.min(near - (TIP_ROWS >> 1), rows.length - TIP_ROWS)), b = Math.min(rows.length, a + TIP_ROWS);
    const more = (n, where) => n > 0 && h("div", { className: "tmore", textContent: `${n} more ${where}` });
    t.replaceChildren(...[
      h("div", { className: "th", textContent: `${key} · ${xs}` }),
      more(a, "above"),
      ...rows.slice(a, b).map(({ ln, val, extra }, i) =>
        h("div", { className: "trow" + (a + i === near ? " near" : ""), onmouseenter: () => this.tipRowEnter(chart, ln),
                   onclick: () => this.tipRowOpen(ln) },
          h("span", { className: "sw", style: `background:${ln.color}` }),
          h("span", { className: "tl", textContent: ln.label }), h("b", { textContent: fmt(val) }),
          h("span", { className: "muted", textContent: extra }))),
      more(rows.length - b, "below"),
      h("div", { className: "tf" }),
    ].filter(Boolean));
    this.tipFooter();
    t.hidden = false;
    const W = t.offsetWidth, H = t.offsetHeight;
    let x = e.clientX + 16, y = e.clientY + 12;
    if (x + W > innerWidth) x = e.clientX - W - 16;
    if (y + H > innerHeight) y = Math.max(0, innerHeight - H - 4);
    t.style.transform = `translate(${x}px, ${y}px)`;
  }

  tipFooter() {
    const f = $("#tip .tf");
    if (f) f.textContent = this.tipPinned ? "hover a row to find it in the list · click to open it"
      : "hold shift to pin · drag: zoom x · drag a box: zoom x and y · click: reset";
  }

  /** Freeze the value tooltip where it is, so the pointer can move into it. */
  pinTip() {
    if (this.tipPinned || !this.hovered || $("#tip").hidden) return;
    this.tipPinned = true;
    this.pinned = this.hovered;
    $("#tip").classList.add("pinned");
    this.tipFooter();
  }

  unpinTip() {
    if (!this.tipPinned) return;
    this.tipPinned = false;
    $("#tip").classList.remove("pinned");
    this.tipFooter();
    const c = this.pinned;
    this.pinned = null;
    if (c && this.hovered !== c) c.unhover();
    else c?.highlight(null);
  }

  tipRowEnter(chart, ln) {
    if (!this.tipPinned) return;
    chart?.highlight(ln);
    if (ln.run) this.revealRun(ln.run.id);
    else if (ln.group != null) this.revealSide(`group:${ln.group}`);
  }

  tipRowOpen(ln) {
    if (!this.tipPinned) return;
    this.unpinTip();
    if (ln.run) this.setPath(ln.run.id);
    else if (ln.group != null) this.openGroup(ln.group);
  }

  /** Scroll the sidebar to a run, opening the folders or group that hold it, and mark it. */
  revealRun(id) {
    const r = this.data.runs.get(id);
    if (!r) return;
    let opened = false;
    if (this.opts.group.length) opened = this.collapsed.delete(`group:${r.gval}`);
    else {
      const rel = this.rel(id), parts = rel === "." ? [] : rel.split("/").slice(0, -1);
      let p = this.opts.path;
      for (const part of parts) {
        p = p ? `${p}/${part}` : part;
        if (this.collapsed.delete(`folder:${p}`)) opened = true;
      }
    }
    if (opened) this.saveCollapsed();
    this.revealSide(id, opened);
  }

  /** Scroll the sidebar so the row for `id` (a run id or a "group:" / "folder:" key) is centered and marked. */
  revealSide(id, rebuild = false) {
    if (this.sideMark === id && !rebuild) return;
    this.sideMark = id;
    this.renderRunTable();
    const i = (this.sideRows || []).findIndex((f) => f.id === id);
    if (i < 0) return;
    const aside = $("aside"), table = $("#runTable");
    const top = table.getBoundingClientRect().top - aside.getBoundingClientRect().top + aside.scrollTop;
    aside.scrollTop = top + i * SIDE_ROW - aside.clientHeight / 2 + SIDE_ROW / 2;
    this.renderSideWindow();
  }

  // ---- media ----

  onMedia(key) {
    if (!this.mediaPanels.has(key)) this.data.ui.keys();
    this.dirtyMedia.add(key);
    this.mediaThrottle();
  }

  renderMedia() {
    for (const k of this.dirtyMedia) this.mediaPanels.get(k)?.markDirty();
    this.dirtyMedia.clear();
    for (const m of this.mediaPanels.values()) if (m.visible && m.dirty) m.render();
  }
}

const MAX_MEDIA_RUNS = 16;

/** The last item of `list` (sorted by step) at or before `step`, else its first. */
function atStep(list, step) {
  let lo = 0, hi = list.length - 1, pick = null;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (list[mid].step <= step) (pick = list[mid]), (lo = mid + 1);
    else hi = mid - 1;
  }
  return pick || list[0];
}

class MediaPanel {
  constructor(app, key) {
    this.app = app;
    this.key = key;
    this.dirty = true;
    this._visible = false;
    this.follow = true;
    this.figs = new Map();
    this.slider = h("input", { type: "range", min: 0, max: 0, value: 0, oninput: () => {
      this.follow = +this.slider.value === +this.slider.max;
      this.render();
    } });
    this.stepLabel = h("span", { className: "muted" });
    this.grid = h("div", { className: "mgrid" });
    this.el = h("div", { className: "panel media" },
      h("div", { className: "ptitle" }, h("span", { textContent: key }), this.slider, this.stepLabel), this.grid);
    this.el._media = this;
  }

  get visible() {
    return this._visible;
  }
  set visible(v) {
    this._visible = v;
    if (v && this.dirty) this.render();
  }

  markDirty() {
    this.dirty = true;
  }

  render() {
    this.dirty = false;
    const m = this.app.data.media.get(this.key) || new Map();
    const runs = (this.app.runList || []).filter((r) => r.shown && m.has(r.id)).slice(0, MAX_MEDIA_RUNS);
    const steps = [...new Set(runs.flatMap((r) => m.get(r.id).map((x) => x.step)))].sort((a, b) => a - b);
    this.slider.max = Math.max(0, steps.length - 1);
    if (this.follow) this.slider.value = this.slider.max;
    const step = steps[+this.slider.value];
    this.stepLabel.textContent = step == null ? "" : `step ${step}`;
    const keep = new Set();
    for (const r of runs) {
      const pick = atStep(m.get(r.id), step);
      keep.add(r.id);
      let f = this.figs.get(r.id);
      if (!f) {
        f = { el: h("figure"), file: null };
        this.figs.set(r.id, f);
      }
      this.grid.append(f.el);
      if (f.file === pick.file && f.color === r.color) continue;
      f.file = pick.file;
      f.color = r.color;
      const cap = h("figcaption", {}, h("span", { className: "sw", style: `background:${r.color}` }),
        `${r.meta.name} · step ${pick.step}`);
      f.el.replaceChildren(cap, this.mediaEl(pick, f));
    }
    for (const [id, f] of this.figs) if (!keep.has(id)) (f.el.remove(), this.figs.delete(id));
  }

  /** An image, video, or sandboxed HTML frame (loaded from the cache) for media item `pick` of figure f. */
  mediaEl(pick, f) {
    if (pick.kind === "image") return h("img", { src: mediaURL(pick), loading: "lazy", decoding: "async" });
    if (pick.kind === "video") return h("video", { src: mediaURL(pick), controls: true, muted: true, loop: true, preload: "metadata" });
    const frame = h("iframe", { sandbox: "allow-scripts", loading: "lazy" }), want = pick.file;
    this.app.data.blob(pick).then(
      (buf) => f.file === want && (frame.srcdoc = new TextDecoder().decode(buf)),
      (err) => (frame.srcdoc = `<pre>${String(err).replace(/</g, "&lt;")}</pre>`),
    );
    return frame;
  }
}

addEventListener("popstate", () => app.restoreFromHash());
const app = (window.app = new App());
app.start().catch((e) => {
  console.error(e);
  $("#status").textContent = `error: ${e.message}`;
});
