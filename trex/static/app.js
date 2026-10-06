// The page: its options, runs, panels of charts, sidebar and node panel.
// `allFunctionsCalledOnLoad` has V8 compile a module's functions as it loads, so that none is compiled at its first
// call, during an interaction. It stands before the module's first statement.
//# allFunctionsCalledOnLoad
import { BASE, Data, PROTOCOL, getJSON, hasKey, mediaURL, preload, within } from "./data.js";
import { startWorkers } from "./pool.js";
import { asNumber, compileWhere, completionContext, fieldText, literal, runField, textOf } from "./where.js";
import { BAND_LABEL, Chart, DENSITY_AUTO, drawCharts, fmt, fmtDur, fmtSI } from "./plot.js";
import { X_RUNTIME, X_STEP } from "./kernel.js";
import { renderer } from "./gl.js";
import { gpuQueued, runGpuJobs, warmArrays } from "./gpustats.js";

const PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
                 "#9c755f", "#bab0ac", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#17becf", "#bcbd22"];
const SIDE_ROW = 24; // px height of a sidebar row
const TIP_ROWS = 14; // value rows in view in the tooltip
const TIP_ROW_PX = 18; // height of one (#tip .trow in index.html)
const PINNED = "\0pinned"; // section of pinned charts
const ALONE = "\0alone"; // the chart shown alone: a copy of its own, so the grid's keeps its drawing
const HIDE_ICON = '<svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><path fill="currentColor" d="M2.1 1.1 1 2.2l2.3 2.3C2.1 5.4 1.1 6.6.5 8c1.2 2.9 4 5 7.5 5 1.3 0 2.5-.3 3.6-.9l2.3 2.3 1.1-1.1L2.1 1.1zM8 11.5A3.5 3.5 0 0 1 4.5 8c0-.6.2-1.2.4-1.7l1.2 1.2V8a1.9 1.9 0 0 0 2.4 1.8l1.2 1.2c-.5.3-1.1.5-1.7.5zm7.5-3.5C14.3 5.1 11.5 3 8 3c-.9 0-1.8.2-2.6.5l1.3 1.3c.4-.2.9-.3 1.3-.3A3.5 3.5 0 0 1 11.5 8c0 .5-.1.9-.3 1.3l1.9 1.9c1-.8 1.9-1.9 2.4-3.2z"/></svg>';
const MENU_ICON = '<svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><circle cx="3" cy="8" r="1.5" fill="currentColor"/><circle cx="8" cy="8" r="1.5" fill="currentColor"/><circle cx="13" cy="8" r="1.5" fill="currentColor"/></svg>';
const SPREAD_SHOWN = 8; // values listed per config key that varies
const NEAR_PX = 2400; // charts this near the view are planned and drawn too, so that a scroll finds them drawn...
const NEAR_ROUND = 6; // ...this many a round, the nearest first, once those in view are drawn
const WARM_CHARTS = 30; // charts, the nearest the view, whose zooms' bucket arrays are copied to the GPU ahead
const WARM_BYTES = 4 << 20; // bucket arrays copied to the GPU ahead of a zoom, per idle task
const FRAME_BUDGET_MS = 12; // chart drawing per frame...
const URGENT_BUDGET_MS = 40; // ...or per round drawn for an interaction
const AIM_MS = 40; // between fetches for a zoom being dragged
const AIM_REST_MS = 24; // a dragged zoom at rest this long is binned ahead: more than a frame, so not while it moves
const HOLD_MS = 120; // a chart whose columns are being rebuilt for a new view waits this long for them before drawing
const PLAN_IDLE_MS = 250; // block planning interval while the view is unchanged
const CANVAS_BYTES = 512 << 20; // drawn charts' canvases kept out of view; beyond, the least recently seen are freed
const KEPT_VIEWS = 8; // run tables, lists of shown runs and computed runs kept for views shown again
const UNCOUNTED = new Set(["rows", "media"]); // run fields that change without the run's metadata: no version counts them
// The fields the page keeps on each run (Data.newRun makes them, in this order): computeRuns', buildGroupTree's
// (nodes, part), assignColors' (color), runsAsListed's (listed), ConfigIndex's (cfgOf, cfgSlot) and runNum's (num).
const RUN_FIELDS = Object.freeze({ visible: true, match: true, focused: true, sortBy: undefined, inFocus: false, shown: false, nodes: null,
                                   part: "", color: "", listed: 0, cfgOf: null, cfgSlot: -1, num: 0 });
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
const opt = (value, text, sel) => h("option", { value, textContent: text, selected: sel });

/** Make `kids` the children of `el`, in order, moving only those not in place: the others keep their layout. */
function setChildren(el, kids) {
  const want = new Set(kids);
  for (const c of [...el.children]) if (!want.has(c)) c.remove();
  kids.forEach((k, i) => el.children[i] !== k && el.insertBefore(k, el.children[i] ?? null));
}

/** [user@]host:path in scp form; its first group is the host. */
const REMOTE = /^((?:[^@/:\s]+@)?(?:\[[^\]\s]+\]|[^@/:\s[\]]+)):(.+)$/;

/** The last `n` characters of `s`, after an ellipsis when cut. */
const tailOf = (s, n) => (s.length > n ? `…${s.slice(1 - n)}` : s);

/** `/api/node`: {node, saves, home, dirs, links, workspaces, history, install, updates}. */
const nodeInfo = () => getJSON("/api/node");

const METRIC_SORT = "metric:"; // prefix of a sort key naming a metric
const DEFAULT_GROUP = "run~1"; // a run's directory: runs logged side by side share a line
const RUN_UP = /^run~(\d+)$/; // group-by field: the directory n levels above a run

/** Levels of a group-by expression "a, b / c": levels split on " / ", fields on ","; empty ones are dropped. */
function parseGroupBy(text) {
  return text.split(/\s+\/\s+/).map((l) => l.split(",").map((f) => f.trim()).filter(Boolean)).filter((l) => l.length);
}

const formatGroupBy = (levels) => levels.map((l) => l.join(", ")).join(" / ");

/** The field the caret is in within a group-by expression: {from, to, prefix}. */
function groupContext(text, caret) {
  const before = text.slice(0, caret), level = before.lastIndexOf(" / ");
  let from = Math.max(before.lastIndexOf(",") + 1, level < 0 ? 0 : level + 3);
  while (from < caret && text[from] === " ") from++;
  return { from, to: caret + text.slice(caret).search(/,|\s\/\s|$/), prefix: text.slice(from, caret) };
}

/** A group-tree node's label: its values, or a run~n directory relative to its parent's when that is a directory
 * above it. */
function nodeLabel(fields, values, parent) {
  const dirs = [fields, parent.fields].every((f) => f.length === 1 && RUN_UP.test(f[0]));
  return dirs && values[0].startsWith(parent.values[0] + "/") ? values[0].slice(parent.values[0].length + 1) : values.join(", ");
}

/** State a group's dot shows: running while any member runs, else crashed or failed if any member did, else
 * finished. */
function groupState(runs) {
  const has = new Set(runs.map((r) => r.meta.state));
  return ["running", "crashed", "failed", "finished"].find((s) => has.has(s)) ?? "";
}

/** "2 running · 5 finished": how many runs are in each state. */
function stateCounts(runs) {
  const n = new Map();
  for (const r of runs) n.set(r.meta.state, (n.get(r.meta.state) ?? 0) + 1);
  return [...n].map(([s, k]) => `${k} ${s}`).join(" · ");
}

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
/** A foldable section of the info panel ([kind, title, dict, sorted]), open when `open`; whether it is open is
 * remembered by its kind. Its table is made when it first opens: a run's summary holds a row per metric. */
function infoSection([kind, title, obj, sort], open) {
  let made = open;
  const el = h("details", { className: "infosec", open, ontoggle: (e) => {
    store.patch("infoOpen", INFO_OPEN, { [kind]: e.target.open });
    if (e.target.open && !made) (made = true), el.append(kvTree(obj, sort));
  } }, h("summary", {}, title, h("span", { className: "gcount", textContent: ` ${Object.keys(obj).length}` })));
  if (open) el.append(kvTree(obj, sort));
  return el;
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

/** Tooltip of the filter box. */
const FILTER_HINT = "name search, or a WHERE clause: lr = 0.001 and seed in (0, 1), algo like 'pp%', state = running, visible = true";
/** A click handler that keeps the click from folding the section whose header holds the control. */
const stop = (f) => (e) => (e.preventDefault(), e.stopPropagation(), f(e));

/** A button showing an inline SVG icon. */
function iconButton(className, title, svg, onclick) {
  const b = h("button", { className, title, onclick });
  b.innerHTML = svg;
  return b;
}

/** A panel's name within its section: the key after the section's path. */
const shortName = (key, sec) => (key.startsWith(sec.id + "/") ? key.slice(sec.id.length + 1) : key);

/** Header controls of a section: on one line (cut with an ellipsis), a link to each subsection (opening and scrolling
 * to it) and to each hidden panel (showing it again); then, with subsections, a button folding or unfolding
 * everything inside. */
function sectionControls(sec, subs, show) {
  const open = (d) => {
    d.parentElement.open = true;
    d.open = true;
    requestAnimationFrame(() => d.scrollIntoView({ block: "start" }));
  };
  const all = (e) => foldAll(e.currentTarget.closest("details"));
  const links = [...sec.children.map((x, i) => h("button", { className: "linkish", title: `open ${x.id}`, textContent: `${x.title} (${x.n})`,
                   onclick: stop(() => open(subs[i])) })),
                 ...sec.hidden.map(([key]) => h("button", { className: "linkish hiddenlink", title: `show ${key} (hidden)`,
                   textContent: shortName(key, sec), onclick: stop(() => show(key)) }))];
  return [links.length ? h("span", { className: "sublinks" }, ...links) : null,
          subs.length ? h("button", { className: "subfold", onclick: stop(all) }) : null].filter(Boolean);
}

/** Whether any section inside `el` is open. */
const anyOpen = (el) => [...el.querySelectorAll("details.section")].some((d) => d.open);

/** Fold every section inside `el` while any is open, else unfold them all. */
function foldAll(el) {
  const fold = anyOpen(el);
  for (const d of el.querySelectorAll("details.section")) d.open = !fold;
}

/** Sorted copy of a section and its subsections: panels by name, subsections by sectionCmp. */
function orderSection(sec) {
  const byName = (a, b) => cmpNames(a[0], b[0]);
  const children = [...sec.children.values()].map(orderSection).sort(sectionCmp);
  const panels = [...sec.items, ...sec.hidden];
  return { ...sec, items: sec.items.sort(byName), hidden: sec.hidden.sort(byName), children,
           media: panels.every(([, kind]) => kind === "media") && children.every((c) => c.media) };
}

/** Sections with charts before media-only ones, then by name. */
function sectionCmp(a, b) {
  return a.media - b.media || cmpNames(a.title, b.title);
}

/** POST `body` (as JSON) to the node's `path`: [whether it succeeded, its JSON answer]. */
async function post(path, body = {}) {
  const r = await fetch(path, { method: "POST", body: JSON.stringify(body) });
  return [r.ok, await r.json()];
}

/** Open page `url` after refetching it past the browser's HTTP cache, so a cached redirect cannot divert it. */
function openPage(url) {
  fetch(url, { cache: "reload" }).catch(() => null).finally(() => (location.href = url));
}

/** Page options the URL hash holds when they differ from these (group and focus aside). */
const OPT_DEFAULTS = { path: "", filter: "", chart: "", center: "median", band: "ci", keys: "", sort: "created", dir: "desc" };

/** Page options from the URL hash. */
function hashOpts(q) {
  const o = Object.fromEntries(Object.entries(OPT_DEFAULTS).map(([k, d]) => [k, q.get(k) || d]));
  if (!["median", "mean", "iqm"].includes(o.center)) o.center = "median";
  return { ...o, group: q.get("group") ?? "", focus: focusOpt(q) };
}
const NAV_OPTS = ["path", "focus", "chart", "group"]; // changed by navigation, restored by back/forward
/** A chart's options it does not override (center and band come from the toolbar). */
const PANEL_DEFAULTS = { smooth: 0, logx: false, logy: false, xmin: null, xmax: null, ymin: null, ymax: null, outliers: 0, render: "auto" };

/** Opened groups, outermost first: [[fields of a group-by level, their values], …] from `focus` (JSON). */
function focusOpt(q) {
  try {
    const f = JSON.parse(q.get("focus") || "[]");
    return Array.isArray(f) ? f.filter((l) => Array.isArray(l?.[0]) && Array.isArray(l?.[1])) : [];
  } catch {
    return [];
  }
}

const cmpNames = new Intl.Collator(undefined, { numeric: true }).compare; // as localeCompare with these options, without making a collator per call
// a time as toLocaleString writes it, the formatter made with the page rather than at the first run opened
const dateTime = new Intl.DateTimeFormat(undefined, { year: "numeric", month: "numeric", day: "numeric", hour: "numeric", minute: "numeric", second: "numeric" });
/** Whether shown run r may be drawn for `key` without a column: a finished run logging it, from its buckets. */
const fromBlocks = (r, key) => r.shown && r.meta.state !== "running" && hasKey(r, key);

const fieldTexts = new WeakMap(); // run metadata -> Map(field -> its text)
let runNums = 0;
/** A number of its own for run r. */
const runNum = (r) => (r.num ||= ++runNums);

/** [the shown runs of `runs` in order, a hash of their set]. */
function shownOf(runs) {
  const shown = runs.filter((r) => r.shown);
  let a = shown.length, b = 0;
  for (const r of shown) (a = (a + Math.imul(runNum(r), 0x9e3779b1)) | 0), (b ^= Math.imul(runNum(r), 0x85ebca6b));
  return [shown, `${a}.${b}`];
}

/** A hash of typed arrays' contents. */
function hashArrays(arrays) {
  let a = 2166136261, b = 0;
  for (const t of arrays) {
    a = Math.imul(a ^ t.length, 16777619);
    for (let i = 0; i < t.length; i++) (a = Math.imul(a ^ t[i], 16777619)), (b = (b + Math.imul(t[i], 0x9e3779b1)) | 0);
  }
  return `${a >>> 0}.${b >>> 0}`;
}

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
  /** Merge `part` into the object stored under k (default d). */
  patch(k, d, part) {
    this.set(k, { ...this.get(k, d), ...part });
  },
};
const INFO_OPEN = { info: true, folder: true, varies: true }; // the info panel's sections open by default

/** A menu row ({label, sub, icon}): its icon when set, label, and `sub` beside it when set; class "mitem" + cls. */
function mitem(it, cls, attrs) {
  return h("button", { className: "mitem" + cls, ...attrs }, it.icon != null ? h("span", { className: "micon", textContent: it.icon }) : null,
    h("span", { className: "ml", textContent: it.label }), it.sub != null && it.sub !== "" ? h("span", { className: "ms", textContent: it.sub }) : null);
}

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
  /** items: [{label, sub, icon, active, onpick}] */
  list(anchor, { title, items, search }) {
    const ul = h("div", { className: "mlist" });
    const render = (q) => {
      const ql = q.toLowerCase();
      ul.replaceChildren(...items.filter((it) => !ql || it.label.toLowerCase().includes(ql)).slice(0, 500).map((it) =>
        mitem(it, it.active ? " active" : "", { onclick: () => it.onpick() })));
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
  if (window.app?.opts.chart) window.app.showChartAlone("");
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
    this.lead = null; // metric of the chart last pressed: its demand is planned first
    this.holdUntil = 0; // until when charts wait for columns being rebuilt (`due`)
    this.tipWheel = (e) => this.scrollTip(e); // listens only while the tooltip is pinned, so scrolling never waits for the page
    this.tree = [];
    const q = new URLSearchParams(location.hash.slice(1));
    this.opts = hashOpts(q);
    this.panelCfg = {};
    this.groupFromHash = q.has("group");
    this.panelsSoon = throttle(() => this.renderPanels(), 300);
    this.data = new Data({
      runFields: RUN_FIELDS,
      runs: throttle(() => this.onRuns(), 300),
      data: (keys, streamed) => this.onData(keys, streamed),
      keys: () => this.panelsSoon(),
      media: (k) => this.onMedia(k),
      status: (t) => ($("#status").textContent = t),
      idle: () => (this.nextFrame(), this.warmSoon()),
      ahead: () => this.aheadOf(),
      protocol: (n) => this.showProtocol(n),
      replan: () => this.replan(true),
      aimed: () => this.aim && this.aimZoom(this.aim), // the zoom being dragged is binned again, from its blocks
      conn: (live) => {
        $("#conn").className = live ? "live" : "down";
        $("#conn").title = live ? "streaming" : "stream disconnected, retrying";
      },
    });
    this.mediaThrottle = throttle(() => this.renderMedia(), 500);
    this.io = new IntersectionObserver(
      (es) => {
        let charts = false;
        for (const e of es) {
          const c = e.target._chart || e.target._media;
          if (c.visible !== e.isIntersecting && e.target._chart) (charts = true), (c.seen = performance.now());
          c.visible = e.isIntersecting;
          if (c.visible && c.dirty) e.target._chart ? this.schedule(true) : c.render();
        }
        if (charts) this.planNow(); // before they draw: charts whose blocks are loaded draw them at once
      },
      { root: $("#panels"), rootMargin: `${NEAR_PX}px` },
    );
    this.ioView = new IntersectionObserver((es) => es.forEach((e) => (e.target._chart.inView = e.isIntersecting)), { root: $("#panels") });
  }

  /** Grouped rendering: some group holds several runs, and the scope is more than one run. */
  get grouped() {
    return this.isGrouped && !this.scopeIsRun;
  }

  /** Whether the path opened is a run: then the runs loaded are that run alone. */
  get scopeIsRun() {
    return this.data.runs.has(this.opts.path);
  }

  /** Effective display options for one chart: its overrides on top of the toolbar. */
  panelOpts(key) {
    const p = this.panelCfg[key] || {};
    return { ...PANEL_DEFAULTS, center: this.opts.center, band: this.opts.band, ...p, xmode: p.x === "runtime" ? X_RUNTIME : X_STEP };
  }

  hasPanelOverrides(key) {
    return Object.keys(this.panelCfg[key] || {}).length > 0;
  }

  async start() {
    startWorkers();
    preload(["/api/node", `${BASE}/api/info`]); // not the runs: they are listed once their stream is open (Data.loadScope)
    if (!(await this.enterNode())) return;
    await this.data.init();
    const root = this.data.rootKey;
    this.hidden = new Set(store.get(`hidden:${root}`, []));
    this.hiddenVer = 0; // bumped whenever `hidden` changes
    this.panelCfg = store.get(`panels:${root}`, {});
    this.collapsed = new Set(store.get(`collapsed:${root}`, []));
    this.pins = store.get(`pins:${root}`, []);
    this.hiddenPanels = new Set(store.get(`hiddenPanels:${root}`, []));
    this.bindControls();
    await this.loadScope();
    (window.requestIdleCallback ?? setTimeout)(() => { // its setup blocks, so it runs while blocks load
      const r = renderer();
      if (r) r.onRestore = () => this.redrawAll();
      else $("#status").textContent = "charts need WebGL2, which this browser lacks";
    });
  }

  /** The node's state, and the trex brand opening its panel. False when the node holds nothing, after showing the panel
   * in place of the page. */
  async enterNode() {
    const d = await nodeInfo();
    this.node = d;
    const brand = $(".brand");
    brand.classList.add("brandlink");
    brand.title = "workspaces and directories";
    brand.onclick = (e) => this.rootMenu(e.currentTarget);
    if (BASE || d.dirs.length) return true;
    this.nodeHome(d);
    return false;
  }

  /** The page of a node that holds nothing: its panel, to add a directory. */
  nodeHome(d) {
    $("#status").textContent = "";
    $("#crumbPath").replaceChildren(h("span", { className: "seg current rootseg" }, h("span", { className: "crumb", textContent: "/" })));
    $("#panels").replaceChildren(h("div", { className: "dhome" }, this.nodePanel(d, async () => this.nodeHome(await nodeInfo()))));
  }

  async rootMenu(anchor) {
    const d = await nodeInfo();
    this.node = d;
    menu.open(anchor, this.nodePanel(d, () => this.rootMenu(anchor)));
  }

  /** The node's workspaces, directories and the trex it pulls from: open, edit or remove one, add a directory (by path,
   * as host:path, from the remembered ones) or a trex to pull from, or a workspace. `refresh` redraws the panel. */
  nodePanel(d, refresh) {
    const err = h("div", { className: "merr" });
    const panel = h("div", { className: "dpanel" });
    const show = (...kids) => panel.replaceChildren(...kids.filter((k) => k != null));
    const home = () => show(h("div", { className: "mtitle", textContent: "workspaces" }), ...this.workspaceRows(d, refresh, edit),
      h("button", { className: "mclear", textContent: "+ new workspace", onclick: () => edit(null) }),
      h("div", { className: "mtitle msec", textContent: "tracked directories" }), ...this.trackedRows(d, refresh, err), err,
      d.links.length ? h("div", { className: "mtitle msec", textContent: "pulled from" }) : null, ...this.linkRows(d),
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
        onclick: () => this.forget(`Delete workspace ${w.name}? Its directories stay tracked.`, "/api/node/workspace/delete",
          { name: w.name }, w.url, refresh) })));
  }

  /** Name and members of a new workspace, or of `ws`; `done` returns to the panel. */
  workspaceEditor(d, ws, done) {
    const err = h("div", { className: "merr" });
    const name = h("input", { type: "text", placeholder: "workspace name", value: ws?.name || "", spellcheck: false });
    const boxes = d.dirs.map((r) => h("input", { type: "checkbox", checked: !!ws?.members.includes(r.name), value: r.name }));
    const save = async () => {
      const [ok, j] = await post("/api/node/workspace", { name: name.value, members: boxes.filter((b) => b.checked).map((b) => b.value), old: ws?.name });
      if (ok) openPage(j.url);
      else err.textContent = j.error;
    };
    return h("div", {}, h("div", { className: "mtitle", textContent: ws ? `edit ${ws.name}` : "new workspace" }),
      h("div", { className: "madd" }, name),
      ...d.dirs.map((r, i) => h("label", { className: "mitem mcheck", title: r.root }, boxes[i],
        h("span", { className: "ml", textContent: r.name }), h("span", { className: "ms", textContent: tailOf(r.root, 36) }))),
      err, h("div", { className: "mfoot" }, h("button", { textContent: "cancel", onclick: done }), h("button", { textContent: "save", onclick: save })));
  }

  /** After `question` is confirmed, POST body to the node's `path`; then leave the page for "/" if it showed `url`, else
   * refresh the panel. */
  async forget(question, path, body, url, refresh) {
    if (!confirm(question)) return;
    await post(path, body);
    if (url === `${BASE}/`) openPage("/");
    else refresh();
  }

  /** Rows of the trex this one pulls every directory of; stopping pulling from one opens the root view anew. */
  linkRows(d) {
    const unlink = async (l) => {
      if (!confirm(`Stop pulling from ${l.url}? What only it offers is no longer served here.`)) return;
      await post("/api/node/remove", { name: l.url });
      openPage("/");
    };
    return d.links.map((l) => h("div", { className: "mrow" },
      h("div", { className: "mitem", title: l.error || l.url },
        h("span", { className: "ml", textContent: l.name || l.url }),
        h("span", { className: "ms", textContent: tailOf(l.url, 36) + (l.state === "connected" ? "" : ` · ${l.state}`) })),
      h("button", { className: "chev", textContent: "×", title: "stop pulling from this trex", onclick: () => unlink(l) })));
  }

  /** Rows of the tracked directories (one pulled from another trex is removed with its link), the add box and the
   * remembered directories and links. */
  trackedRows(d, refresh, err) {
    const note = h("div", { className: "mhint" });
    const add = async (path) => {
      const link = /^http:\/\//.test(path.trim()), host = link ? null : REMOTE.exec(path.trim())?.[1];
      [err.textContent, note.textContent] = ["", link ? `asking ${path.trim()}…` : host ? `starting trex on ${host}…` : ""];
      const [ok, j] = await post("/api/node/add", { path: path.trim() });
      note.textContent = "";
      if (ok) openPage(j.url);
      else err.textContent = j.error;
    };
    const input = h("input", { type: "text", placeholder: "/path/to/runs, ~/runs, host:path or http://host:port", spellcheck: false,
      onkeydown: (e) => e.key === "Enter" && add(input.value) });
    const served = d.dirs.map((r) => h("div", { className: "mrow" },
      h("button", { className: "mitem" + (r.url === `${BASE}/` ? " active" : ""), title: r.error || (r.link ? `${r.root}, pulled from ${r.link}` : r.root),
        onclick: () => openPage(r.url) },
        h("span", { className: "ml", textContent: r.name }),
        h("span", { className: "ms", textContent: tailOf(r.root, 36) + (r.state === "local" || r.state === "connected" ? "" : ` · ${r.state}`) })),
      r.link ? null : h("button", { className: "chev", textContent: "×", title: "stop tracking this directory (its files are kept)",
        onclick: () => this.forget(`Stop serving ${r.root}? Its run files are kept.`, "/api/node/remove", { name: r.name }, r.url, refresh) })));
    const recent = d.history.map((path) => h("button", { className: "mitem", title: `track ${path}`, onclick: () => add(path) },
      h("span", { className: "micon", textContent: "+" }), h("span", { className: "ml", textContent: tailOf(path, 56) })));
    const clear = async () => {
      await post("/api/node/history/clear");
      refresh();
    };
    return [served.length ? h("div", { className: "mlist" }, ...served) : h("div", { className: "mhint", textContent: "none tracked" }),
      h("div", { className: "madd" }, input, h("button", { textContent: "add", onclick: () => add(input.value) })), note,
      recent.length ? h("div", { className: "mrecent" }, h("div", { className: "mtitle", textContent: "recent" }),
        h("div", { className: "mlist" }, ...recent), h("button", { className: "mclear", textContent: "clear history", onclick: clear })) : null];
  }

  /** The node's trex version, with an update button when it can update itself. */
  versionRow(d, err) {
    const { install: inst, updates: up } = d;
    const label = h("span", { className: "ms", textContent: `trex ${inst.version}${inst.commit ? ` · ${inst.commit.slice(0, 7)}` : ""}`,
      title: up.available ? `updates from ${up.source}` : `no update button: ${up.reason}` });
    const btn = up.available ? h("button", { className: "mclear", textContent: "update", title: `install the newest trex from ${up.source} and restart`,
      onclick: () => this.updateNode(btn, err) }) : null;
    return h("div", { className: "mfoot" }, label, btn);
  }

  /** Update the node; after it restarts on a new trex, reload the page. */
  async updateNode(btn, err) {
    btn.disabled = true;
    btn.textContent = "updating…";
    err.textContent = "";
    const [ok, j] = await post("/api/node/update");
    if (!ok || !j.updated) {
      btn.disabled = false;
      btn.textContent = "update";
      err.textContent = ok ? "already the newest trex" : j.error;
      return;
    }
    btn.textContent = "restarting…";
    for (let i = 0; i < 240; i++) {
      await new Promise((ok) => setTimeout(ok, 500));
      const d = await nodeInfo().catch(() => null);
      if (d?.install && JSON.stringify(d.install) === JSON.stringify(j.to)) break;
    }
    location.reload();
  }

  async refreshTree() {
    this.tree = await getJSON(`${BASE}/api/tree`);
  }

  /** Navigate to folder (or run) `path` relative to the served root. */
  async setPath(path) {
    renderer()?.wake();
    menu.close();
    this.opts.path = path;
    this.opts.focus = [];
    this.saveHash(true);
    await this.openPath();
  }

  /** Show the folder (or run) the path names: from the runs the page holds when they include it (a folder under the
   * one loaded, once its runs are listed), else by loading its runs. */
  openPath() {
    if (this.data.listed && within(this.opts.path, this.data.scope)) return this.showPath();
    return this.loadScope();
  }

  /** Show the path from the runs loaded: nothing is fetched but the blocks its charts lack. */
  showPath() {
    this.xrange = null;
    for (const c of this.charts.values()) c.yzoom = null;
    document.title = `${this.opts.path || this.data.info?.name || "trex"} · trex`;
    this.data.view = this.opts.path;
    if (this.groupFromHash) this.groupFromHash = false;
    else this.opts.group = this.defaultGroup(this.opts.path);
    this.saveHash();
    this.onRuns();
    this.renderCrumbs(); // with the info panel, whose height moves the charts: before looking where they are
    this.lookNow();
    this.planNow(false);
    this.updateZoomButton();
    if (this.opts.chart && !this.charts.has(this.opts.chart)) this.setOpt("chart", "");
  }

  async loadScope() {
    this.xrange = null;
    for (const c of this.charts.values()) this.io.unobserve(c.el), this.ioView.unobserve(c.el), c.el.remove(), c.dispose();
    this.charts.clear();
    this.round = null;
    this.keptRuns = this.runsNow = null; // of the runs that go
    for (const m of this.mediaPanels.values()) m.el.remove();
    this.mediaPanels.clear();
    this.saveHash();
    const name = this.data.info?.name || "trex";
    document.title = `${this.opts.path || name} · trex`;
    await this.data.loadScope(this.opts.path);
    if (this.groupFromHash) this.groupFromHash = false;
    else (this.opts.group = this.defaultGroup(this.opts.path)), this.saveHash();
    this.onRuns();
    this.renderPanels();
    if (this.opts.chart && !this.charts.has(this.opts.chart)) this.setOpt("chart", "");
    if (this.opts.chart) this.renderCrumbs();
    (window.requestIdleCallback ?? setTimeout)(() => warmFilter(this.data.everyRun()), { timeout: 2000 });
  }

  /** Note which charts are in view, and which near it, as the page is laid out now, as the observers will at the next
   * frame: after a change of the panels, the charts then in view are planned and drawn without waiting for it. One
   * never laid out takes the size of a chart beside it in its grid (they are alike) until its own is known. */
  lookNow() {
    const box = $("#panels").getBoundingClientRect(), near = (r, pad) => !!r && r.height > 0 && r.bottom > box.top - pad && r.top < box.bottom + pad;
    const reaches = new Map(); // a grid of charts -> whether it reaches near the view: the charts of the others are not looked at
    for (const c of this.charts.values()) {
      const grid = c.el.isConnected && !c.alone ? c.el.parentNode : null;
      let reach = !!grid && reaches.get(grid);
      if (grid && reach === undefined) reaches.set(grid, (reach = near(grid.getBoundingClientRect(), NEAR_PX)));
      const r = reach ? c.el.getBoundingClientRect() : null, visible = near(r, NEAR_PX);
      if (visible && !c.visible) c.seen = performance.now();
      (c.visible = visible), (c.inView = near(r, 0));
      const like = c.visible && !c.w ? [...c.el.parentNode.children].find((el) => el._chart?.w)?._chart : null;
      if (like) (c.w = like.w), (c.h = like.h), (c.dirty = true);
    }
  }

  /** Tell the data layer what the visible charts show (those in view only, when not `all`: the others are planned
   * before the rounds that draw them, `afterFrame`, or in a while), and have the columns it then queues for the charts
   * in view built, before they draw: charts whose blocks are loaded draw them in their first round. */
  planNow(all = true) {
    this.plan(all);
    this.data.rebuildNow(new Set([...this.charts.values()].filter((c) => c.inView || c.full).map((c) => c.key)));
  }

  /** The group-by a folder's trex_info.json declares (`group_by`, an expression or a list of fields), or the nearest
   * one above it; else DEFAULT_GROUP. */
  defaultGroup(path) {
    const parts = path ? path.split("/") : [];
    for (let i = parts.length; i >= 0; i--) {
      const d = this.data.folders[parts.slice(0, i).join("/")]?.trex?.group_by;
      if (typeof d === "string") return formatGroupBy(parseGroupBy(d));
      if (Array.isArray(d)) return d.join(", ");
    }
    return DEFAULT_GROUP;
  }

  /** Whether the hash states the grouping: inside an opened group, or when it differs from the folder's default. */
  groupInHash() {
    return this.opts.focus.length > 0 || this.opts.group !== this.defaultGroup(this.opts.path);
  }

  /** Write the options to the URL hash: a new history entry when `push` (navigation), else in place. The browser takes
   * a millisecond or more for it, so it is written at the end of the task, after the charts the task asked for drew. */
  saveHash(push = false) {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(this.opts)) {
      const s = k === "focus" ? (v.length ? JSON.stringify(v) : "") : v;
      if (k === "group" ? this.groupInHash() : s && OPT_DEFAULTS[k] !== s) q.set(k, s);
    }
    const url = "#" + q.toString(), was = this.hashNext; // set while a write is queued
    this.hashNext = { url, push: !!was?.push || (push && url !== (was ? was.url : location.hash)) };
    if (!was) queueMicrotask(() => this.writeHash());
  }

  /** Write the hash `saveHash` was last given, once the draw queued in this task (`drawSoon`), if any, has run. */
  writeHash() {
    if (this.soon) return queueMicrotask(() => this.writeHash());
    const { url, push } = this.hashNext;
    this.hashNext = null;
    if (push) history.pushState(null, "", url);
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
      return this.openPath();
    }
    if (differs("focus") || (q.has("group") && differs("group"))) {
      this.opts.focus = next.focus;
      if (q.has("group")) this.opts.group = next.group;
      else this.opts.group = this.defaultGroup(this.opts.path);
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

  /** The grip between sidebar and charts: dragging sets the sidebar's width (remembered), double-clicking resets it. */
  bindSideGrip() {
    const grip = $("#sideGrip"), main = $("main");
    const set = (w) => w && main.style.setProperty("--side-w", `${Math.round(Math.max(200, Math.min(0.7 * innerWidth, w)))}px`);
    set(store.get("sideWidth", 0));
    grip.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      grip.setPointerCapture(e.pointerId);
      grip.classList.add("drag");
      const x0 = e.clientX, w0 = $("aside").offsetWidth;
      const move = (ev) => set(w0 + ev.clientX - x0);
      grip.addEventListener("pointermove", move);
      grip.addEventListener("pointerup", () => {
        grip.removeEventListener("pointermove", move);
        grip.classList.remove("drag");
        store.set("sideWidth", $("aside").offsetWidth);
      }, { once: true });
    });
    grip.addEventListener("dblclick", () => {
      main.style.removeProperty("--side-w");
      store.set("sideWidth", 0);
    });
  }

  bindControls() {
    this.bindSideGrip();
    let sideRaf = 0;
    const side = () => {
      sideRaf = 0;
      const [a, b] = this.sideWin || [0, 0], aside = $("aside");
      const top = this.sideTop();
      const first = (aside.scrollTop - top) / SIDE_ROW, last = first + aside.clientHeight / SIDE_ROW;
      if (first < a + 10 && a > 0 || last > b - 10 && b < (this.sideRows || []).length) this.renderSideWindow();
    };
    $("aside").addEventListener("scroll", () => (sideRaf ||= requestAnimationFrame(side)));
    new ResizeObserver(() => (sideRaf ||= requestAnimationFrame(side))).observe($("aside"));
    const o = this.opts;
    const bind = (sel, key, ev, after) => {
      const el = $(sel);
      el.value = o[key];
      el.addEventListener(ev, () => {
        this.setOpt(key, el.value);
        after();
      });
    };
    bind("#runFilter", "filter", "input", throttle(() => this.onRuns(), 150));
    this.bindFilterBox();
    bind("#center", "center", "change", () => this.redrawAll());
    bind("#band", "band", "change", () => this.redrawAll());
    bind("#keyFilter", "keys", "input", throttle(() => this.renderPanels(), 150));
    $("#resetZoom").addEventListener("click", () => this.resetZoom());
    this.bindGroupBox();
    this.bindSortBox();
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

  /** The folders and runs directly under `prefix`, to open one, by the tree as it now is. */
  async childMenu(anchor, prefix) {
    await this.refreshTree();
    menu.list(anchor, {
      title: prefix || this.data.info.name, search: true,
      items: this.children(prefix).map((e) => ({
        label: e.name, icon: e.run ? "▪" : "▸", sub: e.run ? e.state : `${e.count} runs`, active: e.path === this.opts.path,
        onpick: () => this.setPath(e.path),
      })),
    });
  }

  get rootName() {
    const node = this.node;
    if (!BASE && !node?.home) return "/";
    const here = BASE ? [...(node?.workspaces || []), ...(node?.dirs || [])].find((r) => r.url === `${BASE}/`)
      : node?.dirs.find((r) => r.id === node.home);
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
      const open = target !== o.path ? () => this.setPath(target) : depth ? () => this.focusLevel(0) : () => this.showChartAlone("");
      const s = h("span", { className: "seg" + (current ? " current" : "") + (cls ? ` ${cls}` : "") },
        h("button", { className: "crumb", title: target || "/", onclick: current ? null : open }, text));
      if (siblingsOf != null) s.append(h("button", { className: "chev", textContent: "▾", title: "switch to a sibling",
        onclick: (e) => this.childMenu(e.currentTarget, siblingsOf) }));
      sep();
      path.push(s);
    };
    const chart = o.chart && this.charts.has(o.chart);
    seg(this.rootName, "", { current: !parts.length && !depth && !chart, cls: "rootseg" });
    if (BASE) path.unshift(h("span", { className: "seg" }, h("button", { className: "crumb", textContent: "/",
      title: "home", onclick: () => openPage("/") })));
    parts.forEach((p, i) => seg(p, parts.slice(0, i + 1).join("/"),
      { current: i === parts.length - 1 && !depth && !chart, siblingsOf: parts.slice(0, i).join("/") }));
    o.focus.forEach((level, i) => {
      const current = i === depth - 1 && !chart, open = i === depth - 1 ? () => this.showChartAlone("") : () => this.focusLevel(i + 1);
      sep();
      path.push(h("span", { className: "seg fchip" + (current ? " current" : "") },
        h("button", { className: "crumb", title: "an opened group", onclick: current ? null : open }, this.focusLabel(level))));
    });
    if (chart) {
      path.push(h("span", { className: "sep", textContent: "›" }),
        h("span", { className: "seg current fchip" }, h("span", { className: "crumb" }, `chart: ${o.chart}`),
          h("button", { className: "chev", textContent: "×", title: "back to all charts (Esc)", onclick: () => this.showChartAlone("") })));
    } else if (depth) {
      // an opened group has no children to open
    } else if (!this.scopeIsRun && this.data.runs.size) {
      path.push(h("button", { className: "chev drill", textContent: "›", title: "open a child folder or run",
        onclick: (e) => this.childMenu(e.currentTarget, o.path) }));
    }
    $("#crumbPath").replaceChildren(...path);
    this.renderInfo();
  }

  /** Top-of-page panel: a run's info/config/summary, or a folder's or group's info and config spread. */
  renderInfo() {
    const run = this.scopeIsRun && this.data.runs.get(this.opts.path);
    const [header, sections] = run ? this.runInfo(run) : this.scopeInfo();
    const open = store.get("infoOpen", INFO_OPEN);
    $("#infoPanel").replaceChildren(h("div", { className: "infohead" }, ...header),
      ...sections.filter(([, , obj]) => obj && Object.keys(obj).length).map((s) => infoSection(s, !!open[s[0]])));
    $("#infoPanel").hidden = false;
  }

  /** [header, sections] of a run's page. A section is [kind, title, dict, sorted]. */
  runInfo(run) {
    const m = run.meta, up = this.opts.path.split("/").slice(0, -1).join("/");
    const header = [h("span", { className: "sw", style: `background:${run.color}` }), h("b", { textContent: m.name }),
      h("span", { className: `st ${m.state}`, textContent: m.state }),
      h("span", { className: "muted", textContent: `${fmtSI(m.summary?._step ?? 0)} steps · ${fmtDur(m.summary?._runtime ?? 0)} · ` +
        `${run.seq} rows · created ${m.created ? dateTime.format(m.created * 1000) : "?"}` }),
      h("span", { className: "spacer" }),
      h("button", { textContent: `↑ ${up || this.data.info.name}`, onclick: () => this.setPath(up) })];
    return [header, [["info", "info", m.info, false], ["config", "config", m.config, true], ["summary", "summary", m.summary, true]]];
  }

  /** [header, sections] of a folder's or group's page: notes of the folders on its path, and its config spread. */
  scopeInfo() {
    const o = this.opts, members = this.runList.filter((r) => r.inFocus);
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


  /** Open a node of the group tree: its directory when every field on the way to it is a run~n, else one level of
   * the path bar per node on the way, each a condition on its level's fields. */
  openGroup(key) {
    const chain = [];
    for (let n = this.nodes?.get(key); n?.parent; n = n.parent) chain.unshift(n);
    if (!chain.length) return;
    menu.close();
    const ups = chain.flatMap((n) => n.fields).map((f) => RUN_UP.exec(f)?.[1]);
    const dir = ups.every(Boolean) && chain.at(-1).runs[0].id.split("/").slice(0, -Math.min(...ups.map(Number))).join("/");
    if (dir !== false && dir !== this.opts.path) return this.setPath(dir);
    this.opts.focus = [...this.opts.focus, ...chain.map((n) => [n.fields, n.values])];
    this.saveHash(true);
    this.onRuns();
  }

  /** Go back to the first `depth` opened groups (0: the folder). */
  focusLevel(depth) {
    menu.close();
    const o = this.opts;
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

  /** A run's value of group-by field f, as text: `run` its id, `run~n` its directory n levels up (relative to the
   * view), anything else the field as filters name it. */
  valueText(r, f) {
    if (f === "run") return r.id;
    if (f.startsWith("run~")) return this.dirText(r, f);
    let known = fieldTexts.get(r.meta);
    if (!known) fieldTexts.set(r.meta, (known = new Map()));
    let t = known.get(f);
    if (t === undefined) {
      const v = runField(r, f);
      known.set(f, (t = v == null ? "∅" : textOf(v)));
    }
    return t;
  }

  /** A run's value of field `run~n` (valueText), kept per run for each path viewed; of any other field starting so,
   * the field as filters name it. */
  dirText(r, f) {
    let dirs = (this.dirs ||= new Map()).get(this.opts.path);
    if (!dirs) this.dirs.set(this.opts.path, (dirs = new WeakMap()));
    let known = dirs.get(r);
    if (!known) dirs.set(r, (known = new Map()));
    let t = known.get(f);
    if (t === undefined) {
      const up = RUN_UP.exec(f), v = up ? null : runField(r, f);
      known.set(f, (t = up ? this.rel(r.id.split("/").slice(0, -Number(up[1])).join("/")) : v == null ? "∅" : textOf(v)));
    }
    return t;
  }

  /** Fields to group by, with how many values each takes among the matching runs: run, run~n for each directory
   * above them, then dir, state and the config keys whose value varies. */
  groupFieldItems() {
    if (this.groupItemsFor === this.runList) return this.groupItems;
    const runs = this.runList.filter((r) => r.match), count = (f) => new Set(runs.map((r) => this.valueText(r, f))).size;
    const depth = Math.max(0, ...runs.map((r) => this.rel(r.id).split("/").length - 1));
    const cfg = new Set();
    for (const r of runs) for (const k of Object.keys(r.meta.config || {})) cfg.add(k);
    const fields = ["run", ...Array.from({ length: depth }, (_, i) => `run~${i + 1}`), ...(runs.some((r) => r.meta.dir) ? ["dir"] : []),
                    "state", ...[...cfg].sort(cmpNames)];
    this.groupItemsFor = this.runList;
    this.groupItems = fields.map((f) => [f, count(f)]).filter(([f, n]) => n > 1 || f.startsWith("run"))
      .map(([f, n]) => ({ label: f, insert: f, tail: "", sub: `${n} value${n === 1 ? "" : "s"}` }));
    return this.groupItems;
  }

  /** "fields: values" of an opened group. */
  focusLabel([fields, values]) {
    return `${fields.join(", ")}: ${values.join(", ")}`;
  }

  /** Group by expression `expr`, kept in its canonical form "a, b / c". */
  setGroup(expr) {
    const g = formatGroupBy(parseGroupBy(expr));
    if (g === this.opts.group) return;
    this.opts.group = g;
    this.saveHash();
    this.onRuns();
  }

  /** The group-by box shows the expression in use, unless it is being edited. */
  renderGroupBox() {
    const box = $("#groupBy");
    if (document.activeElement !== box) box.value = this.opts.group;
  }

  /** Nest the runs in view by the group-by's levels that take more than one value among them; a level holding `run`
   * nests nothing, its runs being lines. Nodes are {key, label, fields, values (of their level), runs, children,
   * parent}, in sort order (this.groupTree, this.nodes by key). When some leaf holds several runs, the leaves are the
   * groups (this.groups), one line each, colored in label order. Returns each level's `levelText` of each run. */
  buildGroupTree(runs) {
    const levels = parseGroupBy(this.opts.group);
    const at = levels.map((fields) => runs.map((r) => this.levelText(r, fields)));
    const multi = at.map((texts) => texts.some((t) => t !== texts[0]));
    const shown = levels.flatMap((l, i) => (multi[i] && !l.includes("run") ? [i] : []));
    const root = { key: "[]", label: "", name: "", fields: [], values: [], runs: [], children: [], parent: null };
    this.nodes = new Map([[root.key, root]]);
    const childOf = new Map(); // node -> Map(its level's values joined -> child)
    runs.forEach((r, j) => {
      let node = root;
      root.runs.push(r);
      r.nodes = [];
      for (const i of shown) {
        let kids = childOf.get(node);
        if (!kids) childOf.set(node, (kids = new Map()));
        let next = kids.get(at[i][j]);
        if (!next) {
          const values = levels[i].map((f) => this.valueText(r, f)), key = JSON.stringify([...JSON.parse(node.key), ...values]);
          const label = nodeLabel(levels[i], values, node);
          next = { key, label, name: node.name ? `${node.name} / ${label}` : label, fields: levels[i], values, runs: [], children: [], parent: node };
          node.children.push(next);
          kids.set(at[i][j], next);
          this.nodes.set(key, next);
        }
        const key = next.key;
        next.runs.push(r);
        r.nodes.push(key);
        node = next;
      }
      r.part = node.key;
    });
    const leaves = [];
    const order = (node) => {
      node.children = node.children.map((c) => [c, this.groupSortValue(c.runs, c.label)])
        .sort((a, b) => this.compare(a[1], b[1]) || cmpNames(a[0].label, b[0].label)).map(([c]) => c);
      node.children.forEach(order);
      if (!node.children.length) leaves.push(node);
    };
    order(root);
    this.groupTree = root;
    this.isGrouped = !levels.some((l, i) => multi[i] && l.includes("run")) && leaves.length > 1 && leaves.some((n) => n.runs.length > 1);
    this.groups = new Map(this.isGrouped ? leaves.map((n) => [n.key, n]) : []);
    this.colorGroups();
    return at;
  }

  /** A run's values of a group-by level's fields, joined: what tells its node among its parent's. */
  levelText(r, fields) {
    return fields.length === 1 ? this.valueText(r, fields[0]) : fields.map((f) => this.valueText(r, f)).join("\u0001");
  }

  /** Give each group its color: by its name's place among the groups' names in order, kept while those names stay. */
  colorGroups() {
    const names = [...this.groups.values()].map((n) => n.name), sig = names.join("\u0001");
    if (sig !== this.groupColors?.sig) {
      const place = new Map([...names].sort(cmpNames).map((name, i) => [name, i]));
      this.groupColors = { sig, place };
    }
    for (const n of this.groups.values()) n.color = PALETTE[this.groupColors.place.get(n.name) % PALETTE.length];
  }

  // ---- runs ----

  /** The filter the box states, compiled (`compileWhere`), kept for its text: null for none, and for a clause that does
   * not parse, which marks the box with its error. */
  where() {
    const box = $("#runFilter"), text = this.opts.filter;
    if (this.whereOf?.text !== text) {
      let w = null, err = "";
      try {
        w = text.trim() ? compileWhere(text) : null;
      } catch (e) {
        err = e.message;
      }
      this.whereOf = { text, w, err };
      if (box) (box.title = err || FILTER_HINT), box.classList.toggle("bad", !!err);
    }
    return this.whereOf.w;
  }

  /** The filter box's test of a run (name search or WHERE clause); a clause that does not parse filters nothing. */
  runFilterFn() {
    const w = this.where();
    if (!w) return () => true;
    let run = null; // the run the getter reads, so that no getter is made per run
    const get = (f) => runField(run, f);
    return (r) => ((run = r), w.test(get));
  }

  /** A key of the runs loaded: another once one comes or goes. */
  get loadedKey() {
    return `${this.data.gen}.${this.data.listVer}`;
  }

  /** The runs of the folder (or run) shown, of those loaded, in the order they were last listed in when all were
   * listed (others after them): sorting them again is then quick. */
  runsAsListed() {
    const path = this.opts.path, of = `${this.loadedKey}\0${path}`;
    if (this.runList && this.listedOf === of) return this.runList.slice(); // the same runs: none came or went
    this.listedOf = of;
    const runs = [], mark = (this.listMark = (this.listMark || 0) + 1), loaded = this.data.runs, prefix = `${path}/`;
    const inPath = path === "" ? () => true : (id) => id === path || id.startsWith(prefix);
    const take = (r) => {
      if (inPath(r.id)) (r.listed = mark), runs.push(r);
      else r.shown = r.inFocus = false;
    };
    const whole = !!this.runOrder && this.orderOf === this.loadedKey; // the order holds the runs loaded, and no other
    let seen = 0;
    for (const r of this.runOrder || this.runList || []) if (whole || loaded.get(r.id) === r) seen++, take(r);
    if (seen < loaded.size) for (const r of loaded.values()) if (r.listed !== mark) take(r); // runs that came since the order was made
    return runs;
  }

  /** Runs in display order, with filter, group focus, visibility, group and color resolved: computed, or as they were
   * computed for the same view before (`runsKey`), while those hold (`runsHold`). */
  computeRuns() {
    const kept = (this.keptRuns ||= new Map()).get(this.runsKey());
    if (kept && this.runsHold(kept)) return this.restoreRuns(kept);
    const runs = this.runsAsListed();
    this.markRuns(runs);
    runs.sort((a, b) => this.compare(a.sortBy, b.sortBy) || (b.meta.created || 0) - (a.meta.created || 0));
    if (runs.length === this.data.runs.size) (this.runOrder = runs), (this.orderOf = this.loadedKey);
    const tree = runs.filter((r) => r.inFocus), texts = this.buildGroupTree(tree);
    this.assignColors(runs);
    this.runList = runs;
    [this.shown, this.shownSig] = this.keptShown(...shownOf(runs));
    this.keepRuns(runs, tree, texts);
  }

  /** Mark each of `runs`: visible (its sidebar checkbox), match (it passes the filter), focused (it lies in the groups
   * opened), its sort key, inFocus (both) and shown. Opened groups that hold no run are closed. */
  markRuns(runs) {
    const o = this.opts, match = this.runFilterFn(), focus = o.focus.length > 0;
    for (const r of runs) {
      r.visible = !this.hidden.has(r.id);
      r.match = match(r);
      r.focused = !focus || this.inFocus(r);
      r.sortBy = this.sortValue(r);
    }
    if (focus && runs.length && !runs.some((r) => r.focused)) o.focus = [];
    const isRun = this.scopeIsRun;
    for (const r of runs) {
      r.inFocus = r.match && r.focused;
      r.shown = r.inFocus && (r.visible || isRun);
    }
  }

  /** Whether a run lies in the groups opened (`opts.focus`): its values of each one's fields are that group's. */
  inFocus(r) {
    return this.opts.focus.every(([fields, values]) => fields.every((f, i) => this.valueText(r, f) === values[i]));
  }

  /** What `computeRuns` computes from besides the runs' metadata, as text: the runs loaded, the runs hidden, the path,
   * the filter, the groups opened, the grouping and the sorting. Null when the filter or the grouping reads a field
   * that changes without its run's metadata (`UNCOUNTED`): such a view is computed each time. */
  runsKey() {
    const o = this.opts;
    if ([...(this.where()?.fields || []), ...parseGroupBy(o.group).flat()].some((f) => UNCOUNTED.has(f))) return null;
    return [this.loadedKey, this.hiddenVer, o.path, o.filter, JSON.stringify(o.focus), o.group, o.sort, o.dir].join("\0");
  }

  /** Keep the runs as just computed, for `computeRuns` of the same view to restore (when the view has a `runsKey`): the
   * list, each run's fields and what `runHolds` compares, the group tree and the shown runs; `linesSig` and
   * `buildRunTable` keep theirs with them. */
  keepRuns(runs, tree, texts) {
    const key = this.runsKey(), d = this.data, n = runs.length, col = () => new Array(n);
    this.runsNow = null;
    if (key === null) return;
    const at = new Int32Array(d.byIdx.length).fill(-1), inTree = new Int32Array(d.byIdx.length).fill(-1);
    const k = { key, ver: d.metaVer, runs, at, inTree, texts, levels: parseGroupBy(this.opts.group), visible: col(), match: col(), focused: col(),
                sortBy: col(), created: col(), inFocus: col(), shown: col(), nodes: col(), part: col(), color: col(),
                view: { groupTree: this.groupTree, nodes: this.nodes, groups: this.groups, isGrouped: this.isGrouped, shown: this.shown, shownSig: this.shownSig },
                linesSig: "", tab: null, tabDone: -1 };
    for (let j = 0; j < n; j++) {
      const r = runs[j];
      at[r.idx] = j;
      (k.visible[j] = r.visible), (k.match[j] = r.match), (k.focused[j] = r.focused), (k.sortBy[j] = r.sortBy), (k.created[j] = r.meta.created || 0);
      (k.inFocus[j] = r.inFocus), (k.shown[j] = r.shown), (k.nodes[j] = r.nodes), (k.part[j] = r.part), (k.color[j] = r.color);
    }
    tree.forEach((r, t) => (inTree[r.idx] = t));
    this.keptRuns.delete(key);
    this.keptRuns.set(key, (this.runsNow = k));
    if (this.keptRuns.size > KEPT_VIEWS) this.keptRuns.delete(this.keptRuns.keys().next().value);
  }

  /** Whether kept runs (`keepRuns`) still hold: each run under their path whose metadata was set since they were
   * computed, or last found to hold, is marked, sorted and grouped as it was (`runHolds`). */
  runsHold(kept) {
    const d = this.data;
    if (kept.ver === d.metaVer) return true;
    const match = this.runFilterFn(), focus = this.opts.focus.length > 0;
    for (const r of d.byIdx) {
      const j = r && r.ver > kept.ver ? kept.at[r.idx] : -1;
      if (j >= 0 && !this.runHolds(kept, r, j, match, focus)) return false;
    }
    kept.ver = d.metaVer;
    kept.runs = kept.runs.slice(); // another list object: what is kept for a list of runs may hold their metadata
    return true;
  }

  /** Whether run `r`, the `j`th of `kept`, still passes the filter or not, lies in the groups opened or not, sorts where
   * it did, and has the values it was grouped by. */
  runHolds(kept, r, j, match, focus) {
    r.visible = kept.visible[j]; // which a filter may read
    if (match(r) !== kept.match[j] || (!focus || this.inFocus(r)) !== kept.focused[j]) return false;
    const sortBy = this.sortValue(r), was = kept.sortBy[j];
    if (!(sortBy === was || (sortBy !== sortBy && was !== was)) || (r.meta.created || 0) !== kept.created[j]) return false;
    const t = kept.inTree[r.idx];
    return t < 0 || kept.levels.every((fields, i) => this.levelText(r, fields) === kept.texts[i][t]);
  }

  /** The runs as `kept` (`keepRuns`) holds them: nothing is written to the runs when they were the last ones computed. */
  restoreRuns(kept) {
    const d = this.data, runs = kept.runs, v = kept.view;
    if (this.runsNow !== kept) {
      for (const r of d.byIdx) if (r && kept.at[r.idx] < 0) r.shown = r.inFocus = false;
      for (let j = 0; j < runs.length; j++) {
        const r = runs[j];
        (r.visible = kept.visible[j]), (r.match = kept.match[j]), (r.focused = kept.focused[j]), (r.sortBy = kept.sortBy[j]);
        (r.inFocus = kept.inFocus[j]), (r.shown = kept.shown[j]), (r.nodes = kept.nodes[j]), (r.part = kept.part[j]), (r.color = kept.color[j]);
      }
    }
    (this.groupTree = v.groupTree), (this.nodes = v.nodes), (this.groups = v.groups), (this.isGrouped = v.isGrouped);
    [this.shown, this.shownSig] = this.keptShown(v.shown, v.shownSig);
    this.runList = runs;
    this.listedOf = `${this.loadedKey}\0${this.opts.path}`;
    if (runs.length === d.runs.size) (this.runOrder = runs), (this.orderOf = this.loadedKey);
    this.keptRuns.delete(kept.key);
    this.keptRuns.set(kept.key, (this.runsNow = kept));
  }

  /** The list `shown` (of signature `sig`) as the same list object it was when last shown, if it was lately: what is
   * kept per list of runs (Data.runsWith and the caches keyed by its lists) serves again. */
  keptShown(shown, sig) {
    const kept = (this.shownLists ||= new Map()), was = kept.get(sig);
    const list = was?.length === shown.length && was.every((r, i) => r === shown[i]) ? was : shown;
    kept.delete(sig);
    kept.set(sig, list);
    if (kept.size > KEPT_VIEWS) kept.delete(kept.keys().next().value);
    return [list, sig];
  }

  /** A run's color: its group's, else the next palette color among shown runs (hidden ones by id). */
  assignColors(runs) {
    let ci = 0;
    for (const r of runs) {
      r.color = this.isGrouped ? this.groups.get(r.part)?.color ?? "#999"
        : r.shown ? PALETTE[ci++ % PALETTE.length] : PALETTE[hashStr(r.id) % PALETTE.length];
    }
  }

  /** Metric and media keys of the shown runs (null: every key, when all runs are shown); true when that set
   * changed. */
  updateScopeKeys() {
    const runs = this.shown, data = this.data;
    let keys = null;
    if (runs.length < data.runs.size) {
      keys = new Set();
      for (const k of data.keys.keys()) if (data.someLog(runs, k)) keys.add(k);
      for (const [k, m] of data.media) if (runs.some((r) => m.has(r.id))) keys.add(k);
      if (keys.size === new Set([...data.keys.keys(), ...data.media.keys()]).size) keys = null; // every key, as of all runs
    }
    const sig = keys ? [...keys].sort().join("\0") : null;
    if (sig === this.scopeKeySig) return false;
    this.scopeKeySig = sig;
    this.scopeKeys = keys;
    return true;
  }

  /** A hash of what charts draw lines of: the shown runs in order with their colors, or the groups' members. Kept with
   * the runs it is of (`keepRuns`). */
  linesSig() {
    const kept = this.runsNow, grouped = this.grouped;
    if (kept?.linesSig) return kept.linesSig;
    let a = grouped ? 1 : 2, b = 3;
    const mix = (x) => ((a = Math.imul(a ^ x, 0x9e3779b1)), (b = Math.imul(b + x, 0x85ebca6b) ^ (b >>> 13)));
    for (const r of this.runList) {
      if (!r.shown) continue;
      mix(runNum(r));
      mix(hashStr(grouped ? r.part : r.color));
    }
    for (const g of this.groups.values()) mix(hashStr(`${g.key}\0${g.name}\0${g.color}`));
    const sig = `${a >>> 0}.${b >>> 0}`;
    if (kept) kept.linesSig = sig;
    return sig;
  }

  /** {cols, color, label, run} per shown run, or {cols, runs, color, label, group} per group, of a metric (a finished
   * run without a column has an undefined one: its buckets draw it); the same list while neither the
   * lines (`linesSig`) nor the data change. */
  linesFor(key) {
    const sig = `${this.drawnSig}|${this.data.keyVersion(key)}`, hit = (this.linesKept ||= new Map()).get(key);
    if (hit?.sig === sig) return hit.lines;
    const lines = this.collectLines(key);
    this.linesKept.set(key, { sig, lines });
    return lines;
  }

  collectLines(key) {
    const out = [];
    if (!this.grouped) {
      for (const r of this.runList) {
        const c = r.shown ? r.cols.get(key) : undefined;
        if (c !== undefined || fromBlocks(r, key)) out.push({ cols: [c], color: r.color, label: r.meta.name, run: r });
      }
      return out;
    }
    for (const g of this.groups.values()) {
      const cols = [], runs = [];
      for (const r of g.runs) if (r.shown && (r.cols.has(key) || fromBlocks(r, key))) cols.push(r.cols.get(key)), runs.push(r);
      if (cols.length) out.push({ cols, runs, color: g.color, label: `${g.name} (${cols.length})`, group: g.key });
    }
    return out;
  }

  onRuns() {
    renderer()?.wake(); // the charts read their ranges back from the GPU once this is done
    this.computeRuns();
    this.buildRunTable();
    if (this.updateScopeKeys()) this.renderPanels();
    const sig = this.linesSig();
    if (sig !== this.drawnSig) (this.drawnSig = sig), this.redrawAll();
    else this.nextFrame(); // charts just created
    this.planSoon = true; // once the charts have drawn
    this.afterPaint([
      () => this.renderCrumbs(),
      () => this.renderGroupBox(),
      () => this.renderRunTable(),
      () => {
        for (const m of this.mediaPanels.values()) m.dirty = true;
        this.mediaThrottle();
      },
    ]);
  }

  /** The runs as the GPU bins them (gpustats.js), by Data's run index (r.idx): each shown run's group (0 for every one
   * when the view is not grouped; -1 when not shown) and whether it is drawn from its column (every running run's,
   * shown or not, so that bin means kept for one set of shown runs hold for any; `columns` a hash of those flags); the
   * groups' members in order, each group's (start, size); and the running runs. Shared by every chart; its `ver`
   * changes only when what it holds does. Kept with the runs it is of (`keepRuns`) while the same runs are running. */
  buildRunTable() {
    const kept = this.runsNow, done = this.data.doneVer;
    if (kept?.tab && kept.tabDone === done) return void this.useRunTable(kept.tab);
    const tab = this.runTableNow(), sig = hashArrays([tab.group, tab.column, tab.members, tab.starts]);
    // a table seen lately comes back as it was, version and all, so what was made for it serves again
    if (sig !== this.runTab?.sig) this.useRunTable(this.runTabs?.get(sig) ?? { ...tab, sig, ver: (this.runTabVer = (this.runTabVer ?? 0) + 1) });
    if (kept) (kept.tab = this.runTab), (kept.tabDone = done);
  }

  /** What the run table holds (`buildRunTable`), of the runs as they stand. */
  runTableNow() {
    const n = this.data.byIdx.length, group = new Int32Array(n).fill(-1), column = new Int32Array(n), members = [], starts = [], running = [];
    if (this.grouped) {
      let gi = 0;
      for (const g of this.groups.values()) {
        const st = members.length;
        for (const r of g.runs) if (r.shown) (group[r.idx] = gi), members.push(r.idx);
        starts.push(st, members.length - st);
        gi++;
      }
    } else for (const r of this.shown) group[r.idx] = 0;
    for (const r of this.data.runningOf(this.data.everyRun())) (column[r.idx] = 1), running.push(r.idx);
    return { n, group, column, columns: hashArrays([column]), members: Int32Array.from(members), starts: Int32Array.from(starts),
             running: Int32Array.from(running) };
  }

  /** Make `tab` the run table, the latest of those kept. */
  useRunTable(tab) {
    if (tab === this.runTab) return;
    this.runTab = tab;
    (this.runTabs ||= new Map()).delete(tab.sig);
    this.runTabs.set(tab.sig, tab);
    if (this.runTabs.size > KEPT_VIEWS) this.runTabs.delete(this.runTabs.keys().next().value);
  }

  /** The lines of metric `key`'s group statistics as the GPU bins them: one per group with runs logging it ({group,
   * label, color, gi (its index in the run table)}), kept for the runs that log it (the same list for every metric
   * they log) while neither the run table nor the groups change. */
  gpuGroupLines(key) {
    const tab = this.runTab, runs = this.data.runsWith(this.shown, key), sig = `${tab.ver}|${this.drawnSig}`;
    const kept = (this.gpuGroupsFor ||= new WeakMap()).get(runs);
    if (kept?.sig === sig) return kept.lines;
    const n = new Int32Array(tab.starts.length / 2), lines = [];
    for (const r of runs) if (tab.group[r.idx] >= 0) n[tab.group[r.idx]]++;
    let gi = 0;
    for (const g of this.groups.values()) {
      if (n[gi]) lines.push({ group: g.key, label: `${g.name} (${n[gi]})`, color: g.color, gi });
      gi++;
    }
    this.gpuGroupsFor.set(runs, { sig, lines });
    return lines;
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
      : k === "step" ? s._step : k === "runtime" ? s._runtime : k.startsWith(METRIC_SORT) ? s[k.slice(METRIC_SORT.length)] : m.created;
    if (typeof v === "string" && k.startsWith(METRIC_SORT)) return asNumber(v) ?? NaN;
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
    const vals = runs.map((r) => r.sortBy).filter((v) => v != null && !(typeof v === "number" && Number.isNaN(v)));
    if (!vals.length) return undefined;
    vals.sort((a, b) => (typeof a === "number" ? a - b : cmpNames(String(a), String(b))));
    return vals[(vals.length - 1) >> 1];
  }

  /** The sort fields (base ones, then each metric's last value) and the sort box showing the current one. */
  renderSortOptions() {
    const base = [["created", "created"], ["name", "name"], ["state", "state"], ["step", "steps"], ["runtime", "runtime"],
                  ["size", "group size"]];
    const metrics = [...this.data.keys.keys()].sort(cmpNames);
    this.sortFields = [...base.map(([value, label]) => ({ value, label, metric: false })),
                       ...metrics.map((k) => ({ value: METRIC_SORT + k, label: k, metric: true }))];
    if (!this.sortFields.some((f) => f.value === this.opts.sort)) this.opts.sort = "created";
    const box = $("#sortBy");
    if (document.activeElement !== box) box.value = this.sortLabel();
    $("#sortDir").textContent = this.opts.dir === "asc" ? "↑" : "↓";
    $("#sortDir").title = this.opts.dir === "asc" ? "ascending" : "descending";
  }

  sortLabel() {
    return this.sortFields?.find((f) => f.value === this.opts.sort)?.label ?? this.opts.sort;
  }

  /** The sort box: focusing it lists every sort field, typing narrows the list (matches anywhere in the name), and a
   * click, or Enter on the highlighted match (↑/↓ move it), picks one; Escape or leaving the box keeps the current. */
  bindSortBox() {
    const box = $("#sortBy"), st = { hl: 0, typed: false, shown: [] };
    const restore = () => {
      if (menu.anchor === box) menu.close();
      box.value = this.sortLabel();
    };
    const pick = (f) => {
      restore();
      box.blur();
      if (f.value !== this.opts.sort) this.setOpt("sort", f.value), this.onRuns();
      box.value = this.sortLabel();
    };
    box.addEventListener("focus", () => {
      Object.assign(st, { typed: false, hl: Math.max(0, this.sortFields.findIndex((f) => f.value === this.opts.sort)) });
      box.select();
      this.sortList(box, st, pick);
    });
    box.addEventListener("input", () => {
      Object.assign(st, { typed: true, hl: 0 });
      this.sortList(box, st, pick);
    });
    box.addEventListener("keydown", (e) => {
      const step = { ArrowDown: 1, ArrowUp: -1 }[e.key];
      if (step) (st.hl += step), this.sortList(box, st, pick);
      else if (e.key === "Enter" && st.shown[st.hl]) pick(st.shown[st.hl]);
      else if (e.key === "Escape") restore(), box.blur();
      else return;
      e.preventDefault();
      e.stopPropagation();
    });
    box.addEventListener("blur", restore);
  }

  /** The group-by box: an expression "a, b / c" applied as it is typed, with the fields completed at the caret (↑/↓
   * choose, Tab or Enter take one, Escape closes them); Enter without completions, or leaving, shows the expression in
   * use in its canonical form. */
  bindGroupBox() {
    const box = $("#groupBy"), apply = throttle(() => document.activeElement === box && this.setGroup(box.value), 150);
    this.bindCompletions(box, { context: groupContext, candidates: () => this.groupFieldItems() }, {
      input: apply,
      key: (e) => (e.key === "Enter" || e.key === "Escape") && (box.blur(), true),
      blur: () => (this.setGroup(box.value), (box.value = this.opts.group)),
    });
  }

  /** The filter box's completions: field names, operators, a field's values (with run counts) or and / or, as the
   * text before the caret calls for. */
  bindFilterBox() {
    this.bindCompletions($("#runFilter"), {});
  }

  /** Completions in box (`filterList`, with its st options): shown on focus and click, and after input once the frame
   * showing what the input changed has painted; ↑/↓ choose, Tab or Enter take one (of the completions of the text as
   * it then is), Escape closes them. `on.input` runs after input, `on.key` takes other keys (true when it handled one),
   * `on.blur` runs on leaving the box. */
  bindCompletions(box, opts, on = {}) {
    const st = { hl: 0, items: [], ctx: null, ...opts }, show = () => this.filterList(box, st);
    let due = false; // completions of input await the frame
    const showDue = () => due && ((due = false), show());
    box.addEventListener("focus", show);
    box.addEventListener("click", show);
    box.addEventListener("input", () => {
      st.hl = 0;
      if (!due) (due = true), requestAnimationFrame(() => setTimeout(showDue, 0));
      on.input?.();
    });
    box.addEventListener("keydown", (e) => {
      showDue();
      const listed = menu.anchor === box && st.items.length, step = listed && { ArrowDown: 1, ArrowUp: -1 }[e.key];
      if (step) (st.hl = (st.hl + step + st.items.length) % st.items.length), this.filterList(box, st, false);
      else if (listed && (e.key === "Tab" || e.key === "Enter")) this.takeCompletion(box, st, st.items[st.hl]);
      else if (listed && e.key === "Escape") menu.close();
      else if (!on.key?.(e)) return;
      e.preventDefault();
      e.stopPropagation();
    });
    box.addEventListener("blur", () => {
      if (menu.anchor === box) menu.close();
      on.blur?.();
    });
  }

  /** Show a box's completions for the word at the caret (recomputed unless `fresh` is false), or close them when there
   * are none: st.context finds the word (the filter language's by default) and st.candidates lists them (the
   * filter's by default). */
  filterList(box, st, fresh = true) {
    if (fresh) {
      st.ctx = (st.context ?? completionContext)(box.value, box.selectionStart ?? box.value.length);
      const q = st.ctx.prefix.toLowerCase(), all = st.candidates ? st.candidates(st.ctx) : this.completions(st.ctx);
      const rank = (it) => (it.label.toLowerCase().startsWith(q) ? 0 : 1); // matches from the start come first
      st.items = all.filter((it) => !q || it.label.toLowerCase().includes(q)).sort((a, b) => rank(a) - rank(b)).slice(0, 200);
      st.hl = Math.min(st.hl, Math.max(0, st.items.length - 1));
    }
    if (!st.items.length) return menu.anchor === box && menu.close();
    const list = h("div", { className: "mlist" }, ...st.items.map((it, i) =>
      mitem(it, i === st.hl ? " hl" : "", { onmousedown: (e) => e.preventDefault(), onclick: () => this.takeCompletion(box, st, it) })));
    menu.open(box, list);
    list.querySelector(".hl")?.scrollIntoView({ block: "nearest" });
  }

  /** Replace the caret's word with completion `it` and what follows it (`it.tail`, else a space), and apply the
   * result. */
  takeCompletion(box, st, it) {
    const { from, to } = st.ctx, after = box.value.slice(to);
    const text = it.insert + (it.tail ?? (it.insert.endsWith("(") || after.startsWith(" ") ? "" : " "));
    box.value = box.value.slice(0, from) + text + after;
    box.setSelectionRange(from + text.length, from + text.length);
    box.dispatchEvent(new Event("input", { bubbles: true }));
    box.focus();
  }

  /** Candidates for a completion context: [{label, insert, sub}]. */
  completions(ctx) {
    if (ctx.kind === "field") return this.filterFields().map((f) => ({ label: f, insert: fieldText(f) }));
    if (ctx.kind === "joiner") return ["and", "or"].map((w) => ({ label: w, insert: w }));
    if (ctx.kind === "operator") {
      const ops = /\bnot\s*$/i.test($("#runFilter").value.slice(0, ctx.from)) ? ["in (", "like"]
        : ["=", "!=", "<", "<=", ">", ">=", "in (", "not in (", "like", "~", "is null", "is not null"];
      return ops.map((o) => ({ label: o, insert: o }));
    }
    if (ctx.kind === "value" && ctx.field) return this.fieldValues(ctx.field);
    return [];
  }

  /** Fields the filter can name: the plain ones, then config keys, then each metric's last value as summary.KEY. */
  filterFields() {
    if (this.fieldsFor === this.data.runs.size + "#" + this.data.keys.size) return this.fieldList;
    const cfg = new Set();
    for (const r of this.data.runs.values()) for (const k of Object.keys(r.meta.config || {})) cfg.add(k);
    this.fieldsFor = this.data.runs.size + "#" + this.data.keys.size;
    this.fieldList = ["name", "path", "parent", "state", "visible", "tags", "dir", "step", "runtime", "created",
                      ...[...cfg].sort(cmpNames), ...[...this.data.keys.keys()].sort(cmpNames).map((k) => `summary.${k}`)];
    return this.fieldList;
  }

  /** The values `field` takes across the runs in scope, most common first, with their run counts. */
  fieldValues(field) {
    const seen = new Map();
    for (const r of this.data.runs.values()) {
      const v = runField(r, field);
      for (const x of Array.isArray(v) ? v : [v]) {
        if (x == null || (typeof x === "number" && !Number.isFinite(x))) continue;
        const k = textOf(x), e = seen.get(k);
        e ? e.n++ : seen.set(k, { v: x, n: 1 });
      }
    }
    return [...seen.values()].sort((a, b) => b.n - a.n || cmpNames(textOf(a.v), textOf(b.v))).slice(0, 500)
      .map(({ v, n }) => ({ label: textOf(v), insert: literal(v), sub: `${n} run${n === 1 ? "" : "s"}` }));
  }

  /** The sort box's suggestions: every field, or those matching what was typed, with metrics under a heading. */
  sortList(box, st, pick) {
    const q = st.typed ? box.value.trim().toLowerCase() : "";
    st.shown = this.sortFields.filter((f) => !q || f.label.toLowerCase().includes(q));
    st.hl = Math.max(0, Math.min(st.hl, st.shown.length - 1));
    const items = [];
    st.shown.forEach((f, i) => {
      if (f.metric && !st.shown[i - 1]?.metric) items.push(h("div", { className: "mtitle", textContent: "metric (last value)" }));
      items.push(mitem(f, (f.value === this.opts.sort ? " active" : "") + (i === st.hl ? " hl" : ""),
        { onmousedown: (e) => e.preventDefault(), onclick: () => pick(f) }));
    });
    const list = h("div", { className: "mlist" }, ...(items.length ? items : [h("div", { className: "mhint", textContent: "no matching field" })]));
    menu.open(box, list);
    list.querySelector(".hl")?.scrollIntoView({ block: "nearest" });
  }

  renderRunTable() {
    this.renderSortOptions();
    const rows = []; // row factories in display order
    const metric = this.opts.sort.startsWith(METRIC_SORT) ? this.opts.sort.slice(METRIC_SORT.length) : null;
    const fmtVal = (v) => (v == null ? "" : fmtAny(v));
    const saveHidden = () => store.set(`hidden:${this.data.rootKey}`, [...this.hidden]);
    const setHidden = (runs, hide) => {
      for (const r of runs) hide ? this.hidden.add(r.id) : this.hidden.delete(r.id);
      this.hiddenVer++;
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
        h("td", {}, h("input", { type: "checkbox", checked: r.visible, onchange: (e) => setHidden([r], !e.target.checked) })),
        h("td", {}, h("span", { className: "sw", style: `background:${r.color}` })),
        h("td", { className: "name", title: r.id, style: `padding-left:${4 + depth * 14}px` },
          h("a", { href: "#", textContent: r.meta.name ?? "", onclick: (e) => {
            e.preventDefault();
            this.setPath(r.id);
          } })),
        h("td", { className: "stc" }, h("span", { className: `dot ${r.meta.state}`, title: r.meta.state || "" })),
        metric ? h("td", { className: "num val", textContent: fmtVal(this.sortValue(r)) })
          : h("td", { className: "num", textContent: s._step != null ? fmtSI(s._step) : "" }),
        h("td", { className: "num", textContent: s._runtime != null ? fmtDur(s._runtime) : "" }),
      );
    };
    /** A collapsible header row for a node of the group tree; a group (one line) shows its color. */
    const headRow = ({ key, label, members, color, depth, open, openTitle, onOpen }) => {
      const vis = members.filter((r) => r.visible).length;
      return h("tr", { className: "grp" + (open ? " open" : "") + (this.sideMark === key ? " mark" : ""), title: open ? "click to collapse" : "click to expand",
        onclick: (e) => !e.target.closest("input, button") && toggle(key) },
        h("td", { className: "tw" }, h("span", { className: "caret", textContent: "▸" })),
        h("td", {}, h("input", { type: "checkbox", checked: vis > 0, indeterminate: vis > 0 && vis < members.length,
          onchange: (e) => setHidden(members, !e.target.checked) })),
        h("td", {}, color ? h("span", { className: "sw", style: `background:${color}` }) : h("span", { className: "folder", textContent: "▤" })),
        h("td", { className: "gname", title: label, style: `padding-left:${4 + depth * 14}px` }, label,
          h("span", { className: "gcount", textContent: ` ${members.length}` })),
        h("td", { className: "stc" }, h("span", { className: `dot ${groupState(members)}`, title: stateCounts(members) })),
        h("td", { className: "num val", textContent: metric ? fmtVal(this.groupSortValue(members, label)) : "" }),
        h("td", { className: "num" }, h("button", { className: "gfocus", textContent: "open ›", title: openTitle, onclick: onOpen })));
    };
    const inScope = this.runList.filter((r) => r.inFocus);
    const heads = [];
    const walk = (node, depth) => {
      for (const c of node.children) {
        const key = `node:${c.key}`, open = !this.collapsed.has(key);
        heads.push(key);
        rows.push(Object.assign(() => headRow({ key, label: c.label, members: c.runs, color: this.groups.get(c.key)?.color, depth, open,
          openTitle: "open this group", onOpen: () => this.openGroup(c.key) }), { id: key }));
        if (open) walk(c, depth + 1);
      }
      if (!node.children.length) for (const r of node.runs) rows.push(Object.assign(() => runRow(r, depth), { id: r.id }));
    };
    walk(this.groupTree, 0);
    this.sideRows = rows;
    this.renderSideWindow();
    const shown = this.shown.length;
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
    const anyVisible = inScope.some((r) => r.visible);
    const ha = $("#hideAll");
    ha.textContent = anyVisible ? "hide all" : "show all";
    ha.onclick = () => setHidden(inScope, anyVisible);
  }

  /** Offset of the run table within the sidebar's scrolled content. */
  sideTop() {
    const aside = $("aside");
    return $("#runTable").getBoundingClientRect().top - aside.getBoundingClientRect().top + aside.scrollTop;
  }

  /** Build only the sidebar rows near the scroll position; spacer rows stand in for the rest. */
  renderSideWindow() {
    const aside = $("aside"), rows = this.sideRows || [];
    const top = this.sideTop();
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
      this.savePanelCfg(key);
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
      row("group line", sel("center", o.center, [["median", "median"], ["mean", "mean"], ["iqm", "IQM (mean of the middle half)"]])),
      row("group band", sel("band", BAND_LABEL[o.band], Object.entries(BAND_LABEL))),
      row("lines", sel("render", "auto", [["lines", "lines"], ["density", "density"]])),
      h("div", { className: "actions" },
        h("button", { textContent: "reset chart", onclick: () => {
          delete this.panelCfg[key];
          this.savePanelCfg(key);
          this.panelSettings(chart, anchor);
        } }),
        h("button", { textContent: "close", onclick: () => menu.close() })),
      h("div", { className: "muted hint", textContent: "Outlier rejection scales the y axis to the chosen quantiles of the visible values; it does not drop data." }),
      h("div", { className: "muted hint", textContent: `Density draws a heatmap of all runs (WebGL only; ungrouped charts). Auto switches to it above ${DENSITY_AUTO} lines.` }),
    );
    menu.open(anchor, form);
  }

  /** Persist the per-chart settings after a change to `key`'s, and redraw its charts. */
  savePanelCfg(key) {
    store.set(`panels:${this.data.rootKey}`, this.panelCfg);
    for (const c of this.chartsOf(key)) c.dirty = true;
    this.schedule(true);
  }

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

  /** Show the panels' sections (`panelSections`). A section shown before keeps its elements, and only what differs is
   * put into the page, so the panels that stay keep their layout. */
  renderPanels() {
    const sections = this.panelSections(), root = $("#panels"), closed = store.get("closedSections", {}), old = this.sectionEls;
    this.sectionEls = new Map();
    let nsec = 0;
    const walk = (secs) => secs.forEach((x) => ((nsec += 1), walk(x.children)));
    walk(sections);
    const count = sections.filter((x) => x.id !== PINNED).reduce((n, x) => n + x.n, 0);
    this.panelBar ||= this.panelBarEl();
    this.panelCountText = `${nsec} sections · ${count} panels`;
    this.renderPanelCount();
    setChildren(root, [$("#infoPanel"), this.panelBar, ...sections.map((x) => this.sectionEl(x, closed, old))]);
    if (this.opts.chart && this.data.keys.has(this.opts.chart)) this.panelEl(this.opts.chart, "metric", false, true);
    this.placeAlone();
    this.updateSectionsToggle();
    this.schedule(true); // charts made here draw; the others show what they did
    this.renderMedia();
  }

  /** The bar above the sections: the button folding or unfolding them all, and the count of panels. */
  panelBarEl() {
    this.panelCount = h("span", { className: "muted" });
    return h("div", { className: "panelbar" }, h("button", { className: "sectionsToggle", onclick: () => foldAll($("#panels")) }), this.panelCount);
  }

  /** A foldable section: its shown panels, then its subsections; whether it is folded is remembered by its path. Its
   * header stays in view while scrolling, opens the section's menu, links to its subsections and hidden panels, and
   * folds or unfolds all of its subsections. The one of `old` (sections by id) with its id is kept, folded as it is. */
  sectionEl(sec, closed, old, depth = 0) {
    const subs = sec.children.map((x) => this.sectionEl(x, closed, old, depth + 1));
    const s = old?.get(sec.id) ?? this.newSection(sec.id, closed, depth);
    const items = sec.items.map(([key, kind, pinned]) => this.panelEl(key, kind, pinned));
    if (items.length) setChildren((s.grid ||= h("div", { className: "grid" })), items);
    else s.grid = null;
    Object.assign(s, { sec, subs });
    setChildren(s.el, [s.head, s.grid, ...subs].filter(Boolean));
    this.renderSectionHead(s);
    this.sectionEls.set(sec.id, s);
    return s.el;
  }

  /** An empty section of path `id`, `depth` sections deep: {el, head, grid, ...}. */
  newSection(id, closed, depth) {
    const head = h("summary", {});
    const el = h("details", { className: "section", style: `--depth:${depth}`, open: !closed[id], ontoggle: (e) => {
      store.patch("closedSections", {}, { [id]: !e.target.open });
      this.updateSectionsToggle();
    } }, head);
    return { sec: null, subs: [], el, head, grid: null, headOf: "" };
  }

  /** A section's header: its title, its menu button, and the links and fold button of `sectionControls`; made anew
   * only when what it says changed. */
  renderSectionHead(s) {
    const { sec, subs, head } = s, of = JSON.stringify([sec.title, sec.n, sec.children.map((x) => [x.id, x.title, x.n]), sec.hidden.map(([key]) => key)]);
    if (s.headOf === of) return;
    s.headOf = of;
    head.replaceChildren(h("span", { className: "stitle", textContent: `${sec.title} (${sec.n})` }),
      iconButton("secmenu", "show or hide this section's panels, fold or unfold its subsections", MENU_ICON, stop(() => this.sectionMenu(sec.id))),
      ...sectionControls(sec, subs, (key) => this.togglePanel(key)));
  }

  /** "N sections · M panels", with how many are hidden. */
  renderPanelCount() {
    const hidden = [...this.hiddenPanels].filter((k) => this.data.keys.has(k) || this.data.media.has(k)).length;
    const text = this.panelCountText + (hidden ? ` · ${hidden} hidden` : "");
    if (this.panelCount.textContent !== text) this.panelCount.textContent = text;
  }

  /** The menu of a section: its panels (checked when shown; picking one hides or shows it) and its subsections
   * (checked when unfolded; picking one folds or unfolds it). */
  sectionMenu(id) {
    const s = this.sectionEls?.get(id), anchor = s?.el.querySelector(":scope > summary .secmenu");
    if (!anchor) return menu.close();
    const { sec } = s, again = () => this.sectionMenu(id), sub = (x) => this.sectionEls.get(x.id)?.el; // looked up when used: a re-render replaces them
    const panels = [...sec.items, ...sec.hidden].sort((a, b) => cmpNames(a[0], b[0]));
    const items = [...panels.map(([key, kind]) => ({ label: shortName(key, sec), sub: kind === "media" ? "media" : "chart",
                     icon: this.hiddenPanels.has(key) ? " " : "✓", onpick: () => (this.togglePanel(key), again()) })),
                   ...sec.children.map((x) => ({ label: `${x.title}/`, sub: `${x.n} panels`, icon: sub(x)?.open ? "✓" : " ",
                     onpick: () => (sub(x) && (sub(x).open = !sub(x).open), again()) }))];
    menu.list(anchor, { title: `${sec.title}: shown panels, open subsections`, items, search: items.length > 12 });
  }

  /** Hide a panel (its section's header then links to it) or show it again, scrolling to it. Only the panel and the
   * headers of the sections holding it change, so the rest of the view stays where it is. */
  togglePanel(key) {
    const show = this.hiddenPanels.delete(key);
    if (!show) this.hiddenPanels.add(key);
    store.set(`hiddenPanels:${this.data.rootKey}`, [...this.hiddenPanels]);
    for (const s of this.sectionEls.values()) {
      const from = show ? s.sec.hidden : s.sec.items, i = from.findIndex(([k]) => k === key);
      if (i < 0) continue;
      const [item] = from.splice(i, 1);
      (show ? s.sec.items : s.sec.hidden).push(item);
      for (const list of [s.sec.items, s.sec.hidden]) list.sort((a, b) => cmpNames(a[0], b[0]));
      show ? this.insertPanel(s, item) : this.removePanel(s, item);
      this.renderSectionHead(s);
    }
    this.renderPanelCount();
    if (show) requestAnimationFrame(() => (this.charts.get(key) ?? this.mediaPanels.get(key))?.el.scrollIntoView({ block: "nearest" }));
  }

  /** Put a panel back into its section's grid, in name order (making the grid if the section had none shown). */
  insertPanel(s, [key, kind, pinned]) {
    const el = this.panelEl(key, kind, pinned), i = s.sec.items.findIndex(([k]) => k === key), next = s.sec.items[i + 1];
    if (!s.grid) s.head.after((s.grid = h("div", { className: "grid" })));
    s.grid.insertBefore(el, next ? this.panelEl(...next) : null);
  }

  /** Take a panel out of its section's grid (and the grid out of the section once empty). */
  removePanel(s, [key, kind, pinned]) {
    this.panelEl(key, kind, pinned).remove();
    if (s.grid && !s.grid.childElementCount) s.grid.remove(), (s.grid = null);
  }

  /** Sections of the shown runs' panels passing the chart filter, in display order: pinned charts (in pin order; each also
   * stays in its own section), "charts" (keys without a slash), then a section per key prefix, nested by path,
   * media-only sections last at each level. A section is {id (its path), title, items: [[key, kind, pinned]] shown,
   * hidden: the same for its hidden panels, children, n (panels in it and below, hidden ones too)}. */
  panelSections() {
    const kf = this.keyFilterFn(), top = new Map(), pins = [];
    const node = (id, title) => ({ id, title, items: [], hidden: [], children: new Map(), n: 0 });
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
      (this.hiddenPanels.has(key) ? n.hidden : n.items).push([key, kind, false]);
    };
    for (const k of this.data.keys.keys()) if (!this.scopeKeys || this.scopeKeys.has(k)) add(k, "metric");
    for (const k of this.data.media.keys()) if (!this.scopeKeys || this.scopeKeys.has(k)) add(k, "media");
    const pinOrder = new Map(this.pins.map((k, i) => [k, i]));
    pins.sort((a, b) => pinOrder.get(a[0]) - pinOrder.get(b[0]));
    const sorted = [...top.values()].map(orderSection).sort((a, b) => (a.id !== "charts") - (b.id !== "charts") || sectionCmp(a, b));
    const pinned = { id: PINNED, title: "pinned", items: pins.filter(([k]) => !this.hiddenPanels.has(k)),
                     hidden: pins.filter(([k]) => this.hiddenPanels.has(k)), children: [], n: pins.length };
    return pins.length ? [pinned, ...sorted] : sorted;
  }

  /** The chart or media panel of `key` (the pinned section's own copy when `pinned`, the chart shown alone when
   * `alone`), created on first use. */
  panelEl(key, kind, pinned = false, alone = false) {
    const [map, Make] = kind === "metric" ? [this.charts, Chart] : [this.mediaPanels, MediaPanel];
    const id = alone ? ALONE + key : pinned ? PINNED + key : key;
    let p = map.get(id);
    if (!p) {
      map.set(id, (p = new Make(this, key)));
      p.alone = alone;
      this.io.observe(p.el);
      if (kind === "metric") this.ioView.observe(p.el);
    }
    if (kind === "metric") p.setPinned(this.pins.includes(key));
    return p.el;
  }

  /** The charts of metric `key`: its own and, when pinned, its copy in the pinned section. */
  chartsOf(key) {
    return [this.charts.get(key), this.charts.get(PINNED + key), this.charts.get(ALONE + key)].filter(Boolean);
  }

  /** Pin a chart to the pinned section at the top (in pin order; it stays in its own section too), or unpin it. */
  togglePin(key) {
    const i = this.pins.indexOf(key);
    if (i >= 0) {
      this.pins.splice(i, 1);
      const copy = this.charts.get(PINNED + key);
      if (copy) this.io.unobserve(copy.el), this.ioView.unobserve(copy.el), copy.el.remove(), copy.dispose(), this.charts.delete(PINNED + key);
    } else this.pins.push(key);
    store.set(`pins:${this.data.rootKey}`, this.pins);
    this.renderPanels();
  }

  /** Show one chart filling the chart pane, as a level of the path bar ("" = all charts). */
  showChartAlone(key) {
    if ((this.opts.chart || "") === key) return;
    renderer()?.wake();
    this.opts.chart = key;
    this.saveHash(true);
    if (key && this.data.keys.has(key)) this.panelEl(key, "metric", false, true);
    this.placeAlone();
    this.charts.get(ALONE + key)?.resize(); // its size now, so that its blocks are planned for it before it draws
    this.planNow();
    this.schedule(true);
    this.renderCrumbs();
  }

  /** Show the chart shown alone (opts.chart), if any, over the grid, which keeps its layout, scroll position and
   * drawings underneath. */
  placeAlone() {
    const alone = this.opts.chart && this.charts.get(ALONE + this.opts.chart), pane = $("#alone");
    $("#panels").classList.toggle("alone", !!alone);
    pane.hidden = !alone;
    if (!alone) pane.replaceChildren();
    else if (alone.el.parentNode !== pane) pane.replaceChildren(alone.el);
  }

  /** "collapse all" while any chart section is open, else "expand all". */
  updateSectionsToggle() {
    const btn = $("#panels .sectionsToggle");
    if (!btn) return;
    btn.textContent = anyOpen($("#panels")) ? "collapse all sections" : "expand all sections";
    btn.hidden = !$("#panels details.section");
    for (const b of document.querySelectorAll("#panels .subfold")) {
      const open = anyOpen(b.closest("details"));
      b.textContent = open ? "fold inside" : "unfold inside";
      b.title = open ? "fold every subsection of this section" : "unfold every subsection of this section";
    }
  }

  /** Mark the charts of `keys` dirty: drawn on the next frame for work the view asked for (the data layer tells of a
   * metric once its columns are rebuilt, so a chart never draws some of them); paced when `streamed`. */
  onData(keys, streamed = false) {
    if (!keys || !this.runList) return this.redrawAll();
    const first = this.markDirty(keys); // a chart still showing nothing draws on the next frame
    if (!streamed) this.schedule(true);
    else if (first) this.nextFrame();
    else this.schedule(false);
  }

  /** Mark the charts of `keys` dirty; whether a visible one among them shows nothing yet. */
  markDirty(keys) {
    let first = false;
    for (const k of keys) {
      for (const c of this.chartsOf(k)) {
        c.dirty = true;
        if ((c.visible || c.full) && !c.view?.lines.length) first = true;
      }
    }
    return first;
  }

  redrawAll() {
    for (const c of this.charts.values()) c.dirty = true;
    this.schedule(true);
  }

  /** Draw dirty visible charts: `now` (for the view, which then also draws the charts dirtied while drawing) as soon
   * as the task asking for it ends; streamed updates are coalesced to 4 Hz, or less often when drawing the visible
   * charts takes more than a sixteenth of that. */
  schedule(now) {
    if (now) (this.urgent = true), this.drawSoon();
    else if (!this.slow) {
      let cost = 0;
      for (const c of this.charts.values()) if (c.visible || c.full) cost += c.drawMs || 0;
      this.slow = setTimeout(() => {
        this.slow = null;
        this.nextFrame();
      }, Math.min(2000, Math.max(250, 16 * cost)));
    }
  }

  nextFrame() {
    if (!this.raf) this.raf = requestAnimationFrame(() => this.drawDirty());
  }

  /** Draw the dirty visible charts once the current task's work is done, without waiting for a frame (on the next
   * frame when asked while drawing). */
  drawSoon() {
    if (this.drawing) return this.nextFrame();
    if (this.soon) return;
    this.soon = true;
    queueMicrotask(() => {
      this.soon = false;
      if (this.raf) cancelAnimationFrame(this.raf), (this.raf = null);
      this.drawing = true;
      try {
        this.drawDirty();
      } finally {
        this.drawing = false;
      }
    });
  }

  /** Draw the dirty visible charts all in one frame: a round prepares the charts dirty when it began,
   * FRAME_BUDGET_MS of them per frame, then draws them together. Charts dirtied meanwhile draw what they have,
   * and are prepared again in the next round. */
  drawDirty() {
    this.raf = null;
    if (!this.runList) return;
    if (!this.round) this.beginRound();
    const t0 = performance.now();
    let n = this.prepareRound();
    if (gpuQueued()) runGpuJobs(), (n = this.prepareRound()); // the GPU bins the round's charts together, then they draw
    if (n < 0) return this.nextFrame();
    if (this.round.some((c) => c.waiting)) return; // the round draws together once the workers answer
    if (n && this.paced && performance.now() - t0 > FRAME_BUDGET_MS / 2) return this.nextFrame(); // drawing gets a frame of its own
    const drew = this.round.length > 0;
    drawCharts(this.round);
    this.round = null;
    this.endRound(drew);
  }

  /** A round of the charts due: those in view (as last laid out, `inView`), or when none of them is due, the
   * NEAR_ROUND nearest the view. `later`: more are left of those an interaction made due; they are drawn in rounds
   * after this one, as for the interaction (not paced), each once the GPU has drawn the one before and a frame has
   * painted (`afterFrame`). */
  beginRound() {
    const due = [...this.charts.values()].filter((c) => this.due(c)), seen = due.filter((c) => c.full || c.inView);
    const fresh = this.urgent || this.later;
    this.round = seen.length ? seen : this.nearest(due);
    this.later = fresh && this.round.length < due.length;
    (this.paced = !fresh), (this.urgent = false); // paced: drawing streamed rows only
    this.budget = this.paced ? FRAME_BUDGET_MS : URGENT_BUDGET_MS;
  }

  /** The charts in the order they are in, those of `of` nearest the ones in view first. */
  byNearness(of) {
    const all = [...this.charts.values()], at = new Map(all.map((c, i) => [c, i])), seen = all.flatMap((c, i) => (c.full || c.inView ? [i] : []));
    const first = seen[0], last = seen.at(-1), far = new Map();
    for (const c of of) far.set(c, seen.length ? Math.min(Math.abs(at.get(c) - first), Math.abs(at.get(c) - last)) : at.get(c));
    return [...of].sort((a, b) => far.get(a) - far.get(b));
  }

  /** The NEAR_ROUND charts of `due` nearest the view. */
  nearest(due) {
    return due.length <= NEAR_ROUND ? due : this.byNearness(due).slice(0, NEAR_ROUND);
  }

  /** In idle tasks, WARM_BYTES at a time, copy to the GPU the bucket arrays a zoom into a chart that bins its runs
   * there would want (`Data.zoomArrays`), of the WARM_CHARTS charts nearest the view: the zoom then finds them there. */
  warmSoon() {
    if (this.warming || !this.runTab) return;
    this.warming = true;
    (window.requestIdleCallback ?? setTimeout)(() => {
      this.warming = false;
      const charts = this.byNearness([...this.charts.values()].filter((c) => (c.visible || c.full) && c.view?.gpu)).slice(0, WARM_CHARTS);
      if (!warmArrays(charts.flatMap((c) => this.data.zoomArrays(c.key)), this.runTab.n, WARM_BYTES)) this.warmSoon();
    }, { timeout: 500 });
  }

  /** After a round: canvases trimmed, blocks planned, and the charts still due drawn next: what an interaction left
   * (`later`) once the GPU has drawn this round, so that binning them does not wait for it. */
  endRound(drew) {
    if (drew) this.trimCanvases(), this.warmSoon();
    if (drew || this.planSoon) this.replan(this.planSoon), (this.planSoon = false);
    if (this.later) renderer().whenDone(() => this.afterFrame());
    else if ([...this.charts.values()].some((c) => this.due(c))) this.schedule(this.urgent);
  }

  /** Draw the dirty visible charts in a task once the next frame has painted: the rounds an interaction left, which
   * would otherwise delay the frame showing its charts in view. */
  afterFrame() {
    if (this.afterDue) return;
    this.afterDue = true;
    requestAnimationFrame(() => setTimeout(() => {
      this.afterDue = false;
      if (this.partPlanned) this.planNow(); // the charts these rounds draw, planned only in view so far
      this.drawDirty();
    }, 0));
  }

  /** Free the canvases of the charts out of view seen least recently, while the drawn charts' canvases hold more
   * than CANVAS_BYTES: a chart scrolled back to shows its drawing at once while nothing it shows changed. */
  trimCanvases() {
    let bytes = 0;
    const out = [];
    for (const c of this.charts.values()) {
      if (!c.canvas.width) continue;
      bytes += c.canvasBytes;
      if (!c.visible && !c.full) out.push(c);
    }
    if (bytes <= CANVAS_BYTES) return;
    out.sort((a, b) => (a.seen || 0) - (b.seen || 0));
    for (const c of out) {
      if (bytes <= CANVAS_BYTES) break;
      bytes -= c.canvasBytes;
      c.releaseCanvases();
    }
  }

  /** Whether chart c is to be drawn: shown and dirty, and not waiting for columns being rebuilt for a view set less
   * than HOLD_MS ago (it then draws once, with them; one showing nothing draws at once). */
  due(c) {
    if (!(c.visible || c.full) || !c.dirty) return false;
    return !(c.view?.lines.length && performance.now() < this.holdUntil && this.data.pending(c.key));
  }

  /** Let charts wait for their columns (`due`) from now, drawing whatever they have once HOLD_MS have passed. */
  hold() {
    this.holdUntil = performance.now() + HOLD_MS;
    clearTimeout(this.holdTimer);
    this.holdTimer = setTimeout(() => this.nextFrame(), HOLD_MS);
  }

  /** Prepare the round's charts not yet prepared or computed by a worker, until the round's budget of preparing here
   * (FRAME_BUDGET_MS, or URGENT_BUDGET_MS for an interaction; handing charts to workers aside): how many, or -1 when
   * some are left for the next frame. */
  prepareRound() {
    let n = 0, spent = 0;
    for (const c of this.round) {
      if (c.prepared || c.waiting) continue;
      if (n && spent > this.budget) return -1;
      const t = performance.now();
      c.prepare();
      c.drawMs = performance.now() - t;
      if (!c.waiting) (spent += c.drawMs), n++;
    }
    return n;
  }

  /** State, beside the status, when the server speaks another protocol than this page (another trex): a reload
   * fetches the page that matches it. */
  showProtocol(n) {
    const b = $("#mismatch");
    b.hidden = n === PROTOCOL;
    b.textContent = "server updated · reload";
    b.title = `this page speaks trex protocol ${PROTOCOL}, the server ${n ?? "an older one"}; reload to match it`;
    b.onclick = () => location.reload();
  }

  /** What to fetch ahead (`Data.nextAhead`): each chart's demand over the shown runs, the visible first, then the nearest
   * the view. */
  aheadOf() {
    if (!this.runList) return [];
    const box = $("#panels").getBoundingClientRect(), mid = (box.top + box.bottom) / 2, pw = this.gridPw(), seen = new Set();
    const far = (c) => {
      if (c.visible || c.full) return -1;
      const b = c.el.getBoundingClientRect();
      return b.height ? Math.abs((b.top + b.bottom) / 2 - mid) : Infinity; // folded away: last
    };
    return [...this.charts.values()].filter((c) => c.el.isConnected).map((c) => [far(c), c]).sort((a, b) => a[0] - b[0])
      .filter(([, c]) => !seen.has(c.key) && seen.add(c.key)).map(([, c]) => this.demand(c, this.shown, pw));
  }

  /** The plot width of the grid's charts, from one laid out (600 before any is). */
  gridPw() {
    for (const c of this.charts.values()) if (c.w && !c.full) return c.pw;
    return 600;
  }

  /** What chart c shows, for `Data.plan`, were the x zoom `range`; `pw` its plot width until it is laid out. */
  demand(c, runs, pw = 600, range = this.xrange) {
    const o = this.panelOpts(c.key), zoom = range && range[2] === o.xmode ? range : null;
    const x0 = o.xmin ?? zoom?.[0] ?? null, x1 = o.xmax ?? zoom?.[1] ?? null;
    return { key: c.key, runs, xmode: o.xmode, zoomed: x0 !== null || x1 !== null, x0: x0 ?? -Infinity, x1: x1 ?? Infinity,
             pw: c.w ? c.pw : pw, many: this.data.runsWith(runs, c.key).length > this.coarseAbove(o) };
  }

  /** Runs above which a chart with options `o` draws its runs from bins of their buckets: group statistics, or a
   * density heatmap. */
  coarseAbove(o) {
    if (this.grouped) return DENSITY_AUTO;
    return o.render === "density" ? 0 : o.render === "auto" ? DENSITY_AUTO : Infinity;
  }

  /** Tell the data layer what the visible charts show: `wait` ms from now if `now`, else within PLAN_IDLE_MS. */
  replan(now = false, wait = 30) {
    const due = performance.now() + (now ? wait : Math.max(30, PLAN_IDLE_MS - (performance.now() - (this.plannedAt || 0))));
    if (this.planTimer && this.planDue <= due) return;
    clearTimeout(this.planTimer);
    this.planDue = due;
    this.planTimer = setTimeout(() => this.plan(), due - performance.now());
  }

  /** Tell the data layer what the visible charts show (or those in view, when not `all`), now. */
  plan(all = true) {
    if (all) clearTimeout(this.planTimer), (this.planTimer = null), (this.plannedAt = performance.now());
    this.partPlanned = !all;
    if (this.runList) this.data.plan(this.demands(this.xrange, all));
  }

  /** What the visible charts (or those in view, when not `all`) show, were the x zoom `range`: one demand per metric,
   * from its widest chart, the metric of the chart last pressed (`lead`) first. */
  demands(range = this.xrange, all = true) {
    const out = new Map();
    for (const c of this.charts.values()) {
      if (!(all ? c.visible || c.full : c.inView || c.full)) continue;
      const d = { ...this.demand(c, this.shown, 600, range), runsSig: this.shownSig }, had = out.get(d.key);
      if (!had || d.pw > had.pw) out.set(d.key, d);
    }
    const lead = out.get(this.lead);
    return lead ? [lead, ...[...out.values()].filter((d) => d !== lead)] : [...out.values()];
  }

  /** While a zoom to x range r ([x0, x1, xmode]) is being dragged: have the blocks the charts would then want fetched,
   * for where the drag is every AIM_MS, and once it rests there for AIM_REST_MS, the charts in view binned as they
   * would then be (`binAhead`). */
  aimZoom(r) {
    this.aim = r;
    this.aimTimer ||= setTimeout(() => {
      this.aimTimer = null;
      if (this.runList && this.aim) this.data.fetchFor(this.demands(this.aim));
    }, AIM_MS);
    clearTimeout(this.restTimer);
    this.restTimer = setTimeout(() => this.runList && this.aim && this.binAhead(this.aim), AIM_REST_MS);
  }

  /** Have the GPU bin the charts in view as a zoom to x range r would show them at once, from the blocks here, and
   * keep the binnings (`Chart.binAhead`): a zoom released where its drag rested finds them made, and waits for no
   * binning nor reads anything back. */
  binAhead(r) {
    if (gpuQueued()) return; // a round's binnings are queued: they run with it
    for (const c of this.charts.values()) if (c.inView || c.full) c.binAhead(r, this.data.layersIf(this.demand(c, this.shown, 600, r)));
    if (gpuQueued()) runGpuJobs();
  }

  /** The drag that aimed a zoom ended: nothing more is fetched or binned for it. */
  endAim() {
    this.aim = null;
    clearTimeout(this.restTimer);
  }

  /** Shared x zoom [x0, x1, xmode], or null. */
  setXRange(r) {
    renderer()?.wake();
    this.xrange = r;
    this.endAim();
    this.plan(false); // before the charts in view redraw: their blocks are asked for, and those here wait for their columns
    this.hold();
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

  /** Value tooltip: every row by value, in a list TIP_ROWS tall centered on `near` (the line nearest the pointer). */
  tip(e, chart, heading, rows, near = -1) {
    if (!e) return ($("#tip").hidden = true);
    this.tipView = { e, chart, heading, rows, near };
    this.renderTip();
  }

  /** The tooltip beside the pointer: a heading, the scrolling list of rows, and a hint. */
  renderTip() {
    const t = $("#tip"), v = this.tipView, { e, chart, heading, rows, near } = v;
    v.body = h("div", {});
    v.list = h("div", { className: "tlist", style: `height:${Math.min(rows.length, TIP_ROWS) * TIP_ROW_PX}px`, onscroll: () => this.tipWindow() }, v.body);
    v.a = v.b = -1;
    t.replaceChildren(h("div", { className: "th", textContent: `${chart?.key} · ${heading}` }), v.list, h("div", { className: "tf" }));
    this.tipFooter();
    t.hidden = false;
    v.body.style.paddingBottom = `${rows.length * TIP_ROW_PX}px`; // the list's full height, so it can scroll to `near`
    v.list.scrollTop = Math.max(0, (near - (TIP_ROWS >> 1)) * TIP_ROW_PX);
    this.tipWindow();
    v.list.style.minWidth = `${v.body.offsetWidth}px`;
    const W = t.offsetWidth, H = t.offsetHeight;
    let x = e.clientX + 16, y = e.clientY + 12;
    if (x + W > innerWidth) x = e.clientX - W - 16;
    if (y + H > innerHeight) y = Math.max(0, innerHeight - H - 4);
    t.style.transform = `translate(${x}px, ${y}px)`;
  }

  /** The rows in view of the tooltip's list (and a few beyond), between spacers standing for the rest. */
  tipWindow() {
    const v = this.tipView, { list, body, rows, near, chart } = v, top = list.scrollTop;
    list.classList.toggle("up", top > 0);
    list.classList.toggle("down", top + list.clientHeight < list.scrollHeight - 1);
    const a = Math.max(0, Math.floor(top / TIP_ROW_PX) - 3), b = Math.min(rows.length, a + TIP_ROWS + 6);
    if (a === v.a && b === v.b) return;
    [v.a, v.b] = [a, b];
    body.style.paddingTop = `${a * TIP_ROW_PX}px`;
    body.style.paddingBottom = `${(rows.length - b) * TIP_ROW_PX}px`;
    body.replaceChildren(...rows.slice(a, b).map(({ ln, val, extra }, i) =>
      h("div", { className: "trow" + (a + i === near ? " near" : ""), onmouseenter: () => this.tipRowEnter(chart, ln),
                 onclick: () => this.tipRowOpen(ln) },
        h("span", { className: "sw", style: `background:${ln.color}` }),
        h("span", { className: "tl", textContent: ln.label }), h("b", { textContent: fmt(val) }),
        h("span", { className: "muted", textContent: extra }))));
  }

  /** While the tooltip is pinned, the wheel (vertical, or horizontal as Shift makes it) scrolls its list:
   * smoothly by a wheel notch, directly by a touchpad's small steps. */
  scrollTip(e) {
    if (!this.tipPinned || !this.tipView?.list || $("#tip").hidden) return;
    e.preventDefault();
    const d = (e.deltaY || e.deltaX) * (e.deltaMode === 1 ? TIP_ROW_PX : e.deltaMode === 2 ? TIP_ROWS * TIP_ROW_PX : 1);
    if (d) this.tipView.list.scrollBy({ top: d, behavior: Math.abs(d) >= 40 ? "smooth" : "instant" });
  }

  tipFooter() {
    const f = $("#tip .tf");
    const scroll = this.tipView?.rows.length > TIP_ROWS ? " · scroll for more" : "";
    if (f) f.textContent = this.tipPinned ? `hover a row to find it in the list · click to open it${scroll}`
      : "hold shift to pin · drag: zoom x · drag a box: zoom x and y · click: reset";
  }

  /** Freeze the value tooltip where it is, so the pointer can move into it. */
  pinTip() {
    if (this.tipPinned || !this.hovered || $("#tip").hidden) return;
    this.tipPinned = true;
    this.pinned = this.hovered;
    $("#tip").classList.add("pinned");
    document.addEventListener("wheel", this.tipWheel, { passive: false });
    this.tipFooter();
  }

  unpinTip() {
    if (!this.tipPinned) return;
    this.tipPinned = false;
    $("#tip").classList.remove("pinned");
    document.removeEventListener("wheel", this.tipWheel);
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
    else if (ln.group != null) this.revealSide(`node:${ln.group}`);
  }

  tipRowOpen(ln) {
    if (!this.tipPinned) return;
    this.unpinTip();
    if (ln.run) this.setPath(ln.run.id);
    else if (ln.group != null) this.openGroup(ln.group);
  }

  /** Scroll the sidebar to a run, opening the groups that hold it, and mark it. */
  revealRun(id) {
    const r = this.data.runs.get(id);
    if (!r) return;
    let opened = false;
    for (const k of r.nodes || []) if (this.collapsed.delete(`node:${k}`)) opened = true;
    if (opened) this.saveCollapsed();
    this.revealSide(id, opened);
  }

  /** Scroll the sidebar so the row for `id` (a run id or a "node:" key) is centered and marked. */
  revealSide(id, rebuild = false) {
    if (this.sideMark === id && !rebuild) return;
    this.sideMark = id;
    if (rebuild || !this.sideRows) this.renderRunTable();
    const i = (this.sideRows || []).findIndex((f) => f.id === id);
    if (i < 0) return;
    const aside = $("aside");
    aside.scrollTop = this.sideTop() + i * SIDE_ROW - aside.clientHeight / 2 + SIDE_ROW / 2;
    this.renderSideWindow();
  }

  // ---- media ----

  onMedia(key) {
    const m = this.mediaPanels.get(key);
    if (m) m.dirty = true;
    else this.panelsSoon();
    this.mediaThrottle();
  }

  renderMedia() {
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
    this.visible = false;
    this.follow = true;
    this.figs = new Map();
    this.slider = h("input", { type: "range", min: 0, max: 0, value: 0, oninput: () => {
      this.follow = +this.slider.value === +this.slider.max;
      this.render();
    } });
    this.stepLabel = h("span", { className: "muted" });
    this.grid = h("div", { className: "mgrid" });
    this.el = h("div", { className: "panel media" },
      h("div", { className: "ptitle" }, h("span", { textContent: key }), this.slider, this.stepLabel,
        iconButton("hide", "hide (its section's header links to it)", HIDE_ICON, () => app.togglePanel(key))), this.grid);
    this.el._media = this;
  }

  render() {
    this.dirty = false;
    const m = this.app.data.media.get(this.key) || new Map();
    const runs = (this.app.shown || []).filter((r) => m.has(r.id)).slice(0, MAX_MEDIA_RUNS);
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
      f.el.replaceChildren(cap, this.mediaEl(pick));
    }
    for (const [id, f] of this.figs) if (!keep.has(id)) (f.el.remove(), this.figs.delete(id));
  }

  /** An image, video, or sandboxed HTML frame for media item `pick`. */
  mediaEl(pick) {
    if (pick.kind === "image") return h("img", { src: mediaURL(pick), loading: "lazy", decoding: "async" });
    if (pick.kind === "video") return h("video", { src: mediaURL(pick), controls: true, muted: true, loop: true, preload: "metadata" });
    return h("iframe", { src: mediaURL(pick), sandbox: "allow-scripts", loading: "lazy" });
  }
}

addEventListener("popstate", () => app.restoreFromHash());
/** Test a filter over `runs` once, and drop the answers: the first filter typed then runs code already compiled,
 * rather than interpreting it for each of thousands of runs. */
function warmFilter(runs) {
  const w = compileWhere("name like 'x%' or created < 0 or state = 'x' or seed in (0, 1) or lr >= 0 and not tags is null");
  let run = null;
  const get = (f) => runField(run, f);
  for (const r of runs) (run = r), w.test(get);
}

/** Lay out, unseen in #fonts, the page's text styles: each size and weight it uses, with the symbols it shows. */
function warmFonts() {
  const symbols = "Ag09·…×±↓↑›▾▸–✓∅✎⛶⚙▪▤";
  for (const size of [10, 11, 12, 13, 14, 15]) {
    for (const weight of [400, 600, 700]) $("#fonts").append(h("span", { style: `font-size:${size}px;font-weight:${weight}`, textContent: symbols }));
  }
}

warmFonts();
const app = (window.app = new App());
app.start().catch((e) => {
  console.error(e);
  $("#status").textContent = `error: ${e.message}`;
});
