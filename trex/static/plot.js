// Line charts. Canvas 2D: each draw asks the kernel for a smoothed, pixel-decimated polyline (or
// group aggregate) of just the visible x-range. WebGL (gl.js): every run's line stays on the GPU
// and zoom is a transform; axes, labels and the hover overlay stay Canvas 2D.
import { LOGX, LOGY, NSTAT, RAW, STATS, agg as kagg, medianCiCoverage, nearest, prep as kprep, yrange } from "./kernel.js";
import { BREAK, Points, Table, pointBuffer, renderer, rgba } from "./gl.js";

/** Renderer choice: WebGL where available; `?gl=0` selects Canvas 2D. */
const GL_PARAM = typeof location === "undefined" ? null : new URLSearchParams(location.search).get("gl");
export const USE_GL = GL_PARAM !== "0";
const GPU_POINTS = 8e6; // points per line set kept at full resolution; larger sets are decimated
export const DENSITY_AUTO = 300; // "auto" draws a density heatmap above this many lines
const DENSITY_TIP = 8; // runs listed by the density tooltip

const M = { l: 52, r: 10, t: 6, b: 20 };
export const BAND_LABEL = { ci: "95% CI", iqr: "IQR", minmax: "min/max", std: "±std", stderr: "±stderr", none: "none" };
const MAX_LEGEND = 12;
const CLICK_MAX = 5; // px of movement below which a press is a click
const BOX_MIN = 8; // px of vertical drag that turns an x zoom into a box zoom

// Two-sided 95% Student t critical values for df = 1..30; normal beyond.
const T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.16, 2.145, 2.131,
             2.12, 2.11, 2.101, 2.093, 2.086, 2.08, 2.074, 2.069, 2.064, 2.06, 2.056, 2.052, 2.048, 2.045, 2.042];

let outArr = new Float64Array(2 << 14);
/** Reusable decimation output with room for `pairs` points. */
function outBuf(pairs) {
  if (2 * pairs > outArr.length) outArr = new Float64Array(2 * pairs);
  return outArr.subarray(0, 2 * pairs);
}

export function fmt(v) {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(2);
  return String(+v.toPrecision(4));
}

export function fmtSI(v) {
  const a = Math.abs(v);
  if (a >= 1e9) return +(v / 1e9).toPrecision(3) + "G";
  if (a >= 1e6) return +(v / 1e6).toPrecision(3) + "M";
  if (a >= 1e3) return +(v / 1e3).toPrecision(3) + "k";
  return String(+v.toPrecision(3));
}

export function fmtDur(s) {
  if (s < 60) return `${+s.toFixed(1)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h${Math.round((s % 3600) / 60)}m`;
}

function niceTicks(lo, hi, n) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

/** Log-axis ticks (log10 space) in [lo, hi]: decades, plus 2× and 5× within narrow ranges; at most ~n. */
function logTicks(lo, hi, n) {
  const mults = hi - lo < 2 ? [1, 2, 5] : [1];
  const out = [];
  for (let d = Math.floor(lo); d <= Math.ceil(hi); d++)
    for (const m of mults) {
      const t = d + Math.log10(m);
      if (t >= lo && t <= hi) out.push(t);
    }
  if (out.length < 2) return niceTicks(10 ** lo, 10 ** hi, n).filter((v) => v > 0).map(Math.log10);
  const every = Math.max(1, Math.ceil(out.length / n));
  return out.filter((_, i) => i % every === 0);
}

function durTicks(lo, hi, n) {
  const steps = [1, 5, 10, 30, 60, 300, 600, 1800, 3600, 7200, 21600, 43200, 86400];
  const step = steps.find((s) => (hi - lo) / s <= n) || 86400 * Math.ceil((hi - lo) / n / 86400);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(v);
  return out;
}

/** Smoothing reference width (x units per EMA step), quantized so streaming growth rarely invalidates caches. */
function smoothScale(span) {
  const s = span > 0 ? span / 1000 : 1;
  return 2 ** (Math.round(Math.log2(s) * 4) / 4);
}

/** [lo, hi] per bin: order-statistic CI for the median, Student t for the mean; none for one run. */
function bandOf(st, center, band, bins) {
  const lo = new Float64Array(bins), hi = new Float64Array(bins);
  const c = center === "mean" ? st.mean : st.median;
  for (let i = 0; i < bins; i++) {
    const n = st.n[i], m = c[i], se = st.std[i] / Math.sqrt(n);
    let a = m, b = m;
    if (n > 1) {
      if (band === "ci" && center === "mean") {
        const t = n - 1 <= 30 ? T95[n - 2] : 1.96;
        (a = m - t * se), (b = m + t * se);
      } else if (band === "ci") (a = st.medlo[i]), (b = st.medhi[i]);
      else if (band === "iqr") (a = st.q25[i]), (b = st.q75[i]);
      else if (band === "minmax") (a = st.min[i]), (b = st.max[i]);
      else if (band === "std") (a = m - st.std[i]), (b = m + st.std[i]);
      else if (band === "stderr") (a = m - se), (b = m + se);
    }
    (lo[i] = a), (hi[i] = b);
  }
  return [lo, hi];
}

/** Band name for the tooltip; the median CI states its exact coverage when 95% is unreachable. */
function bandLabel(band, center, n) {
  if (band !== "ci" || center === "mean") return BAND_LABEL[band];
  const cov = medianCiCoverage(n);
  return cov >= 0.95 ? "95% CI" : `${(cov * 100).toFixed(1)}% CI (min–max)`;
}

/** Tooltip heading for x: a runtime or a step. */
const xLabel = (v, x) => (v.xmode === 1 ? fmtDur(x) : `step ${fmtSI(Math.round(x))}`);

function dot(ctx, x, y, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, 7);
  ctx.fill();
}

/** Tooltip order: highest value first. */
const byValue = (a, b) => (b.val > a.val ? 1 : b.val < a.val ? -1 : 0);

function lowerBound(a, n, x) {
  let lo = 0, hi = n;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (a[m] < x) lo = m + 1;
    else hi = m;
  }
  return lo;
}

function upperBound(a, n, x) {
  let lo = 0, hi = n;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (a[m] <= x) lo = m + 1;
    else hi = m;
  }
  return lo;
}

/** Index range of column c that can touch [x0, x1] (transformed x), plus one neighbor per side. */
function visibleRange(c, xmode, x0, x1, logx) {
  if (!c.sorted[xmode === 0 ? 0 : 1]) return [0, c.n];
  const xs = c.xs(xmode), a = logx ? 10 ** x0 : x0, b = logx ? 10 ** x1 : x1;
  return [Math.max(0, lowerBound(xs, c.n, a) - 1), Math.min(c.n, upperBound(xs, c.n, b) + 1)];
}

/** Calls grow(y) with the min and max of valid y over points with transformed x in [x0, x1]. */
function visibleY(c, xmode, ys, x0, x1, logx, logy, grow) {
  const xs = c.xs(xmode), [lo, hi] = visibleRange(c, xmode, x0, x1, logx);
  let mn = Infinity, mx = -Infinity;
  for (let i = lo; i < hi; i++) {
    const x = xs[i], xt = logx ? (x > 0 ? Math.log10(x) : NaN) : x, y = ys[i];
    if (xt >= x0 && xt <= x1 && Number.isFinite(y) && (!logy || y > 0)) {
      if (y < mn) mn = y;
      if (y > mx) mx = y;
    }
  }
  if (mn <= mx) grow(mn), grow(mx);
}

const extents = new WeakMap(); // Col -> {key: ext}
/** [xmin, xmax (transformed), ymin, ymax] over a column's valid points, extended incrementally. */
function colExtent(c, xmode, ys, logx, logy, key) {
  let m = extents.get(c);
  if (!m) extents.set(c, (m = {}));
  let e = m[key];
  if (!e || e.n > c.n) e = m[key] = { n: 0, x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity };
  const xs = c.xs(xmode);
  for (let i = e.n; i < c.n; i++) {
    const x = xs[i], xt = logx ? (x > 0 ? Math.log10(x) : NaN) : x, y = ys[i];
    if (!Number.isFinite(xt) || !Number.isFinite(y) || (logy && !(y > 0))) continue;
    if (xt < e.x0) e.x0 = xt;
    if (xt > e.x1) e.x1 = xt;
    if (y < e.y0) e.y0 = y;
    if (y > e.y1) e.y1 = y;
  }
  e.n = c.n;
  return e;
}

/** Writes points (xs[i], ys[i]) for i in [from, to) as f32 relative to (ox, oy) in transformed
 * space at dst point `at`; invalid points become line breaks. */
function fillPoints(xs, xstep, ys, ystep, from, to, dst, at, ox, oy, logx, logy) {
  let k = 2 * at;
  for (let i = from; i < to; i++) {
    const x = xs[i * xstep], y = ys[i * ystep];
    const xt = logx ? (x > 0 ? Math.log10(x) : NaN) : x;
    if (Number.isFinite(xt) && Number.isFinite(y) && (!logy || y > 0)) {
      dst[k] = xt - ox;
      dst[k + 1] = (logy ? Math.log10(y) : y) - oy;
    } else dst[k] = dst[k + 1] = BREAK;
    k += 2;
  }
}

/** Running y range of the values a view shows (positive ones only on a log axis). */
class YRange {
  constructor(logy) {
    this.logy = logy;
    this.lo = Infinity;
    this.hi = -Infinity;
  }

  add(v) {
    if (Number.isFinite(v) && (!this.logy || v > 0)) (this.lo = Math.min(this.lo, v)), (this.hi = Math.max(this.hi, v));
  }
}

/** [x0, x1] in data units: the settings, else the zoom, else the extent (from its smallest positive x on log axes). */
function xRange(o, zoom, e0, e1, epos) {
  let x0 = o.xmin ?? zoom?.[0] ?? e0, x1 = o.xmax ?? zoom?.[1] ?? e1;
  if (o.logx && !(x0 > 0)) x0 = epos;
  if (o.logx && !(x1 > x0)) x1 = x0 * 10;
  if (x0 === x1 && Number.isFinite(x0)) (x0 -= 0.5), (x1 += 0.5);
  return [x0, x1];
}

/** [min, max, smallest positive] x of the columns. */
function xExtent(cols, xmode) {
  let e0 = Infinity, e1 = -Infinity, epos = Infinity;
  for (const c of cols) {
    const r = c.extent(xmode);
    if (r) (e0 = Math.min(e0, r[0])), (e1 = Math.max(e1, r[1])), (epos = Math.min(epos, r[2]));
  }
  return [e0, e1, epos];
}

/** Grow `yr` by column c's values (raw or smoothed) inside the view's x range. */
function growVisible(c, v, raw, yr) {
  const ys = c.ys(v.alpha, raw);
  const e = colExtent(c, v.xmode, ys, v.logx, v.logy, `${v.xmode}|${v.logx}|${v.logy}|${raw ? 0 : v.alpha}|${raw ? 0 : v.scale}`);
  if (e.x0 >= v.x0 && e.x1 <= v.x1) {
    if (e.y0 <= e.y1) yr.add(e.y0), yr.add(e.y1);
  } else visibleY(c, v.xmode, ys, v.x0, v.x1, v.logx, v.logy, (y) => yr.add(y));
}

let staging = null, scratchPts = null;

/** A chart's run lines (raw or smoothed) in a GPU texture, uploaded only when columns or the transform change. */
class LineSet {
  constructor(r, raw) {
    this.r = r;
    this.raw = raw;
    this.pts = new Points(r);
    this.slots = new Map(); // Col -> {off, cap, n, ver}
    this.key = null;
    this.win = null; // [w0, w1] transformed x when decimated
    this.table = new Table(64);
    this.uploadMs = 0;
  }

  /** Bring every column up to date. p: {xmode, logx, logy, alpha, scale, ex0, ex1, vx0, vx1, pw}.
   * False when the GPU cannot hold the set. */
  sync(cols, p) {
    const t0 = performance.now();
    const alpha = this.raw ? 0 : p.alpha;
    const key = `${p.xmode}|${p.logx}|${p.logy}|${alpha}|${alpha > 0 ? p.scale : 0}`;
    const fresh = this.pts.live && key === this.key && this.inWindow(p) && this.updateStale(cols, p, alpha);
    const ok = fresh || this.build(cols, p, alpha, key);
    const dt = performance.now() - t0;
    this.uploadMs = dt > 0.05 ? dt : 0;
    return ok;
  }

  /** Whether the view stays inside the decimation window (if any) at enough resolution. */
  inWindow(p) {
    if (!this.win) return true;
    const [w0, w1] = this.win;
    return p.vx0 >= w0 && p.vx1 <= w1 && (this.R * (p.vx1 - p.vx0)) / (w1 - w0) >= p.pw;
  }

  /** Rewrite the columns that changed in place; false if a rebuild is needed instead. */
  updateStale(cols, p, alpha) {
    const stale = cols.filter((c) => this.slots.get(c)?.ver !== c.n);
    return stale.length <= 256 && stale.every((c) => this.update(c, p, alpha));
  }

  build(cols, p, alpha, key) {
    this.key = key;
    this.slots.clear();
    this.garbage = 0;
    let total = 0;
    for (const c of cols) total += c.n;
    this.win = null;
    if (total > GPU_POINTS) {
      this.R = Math.min(16384, Math.max(Math.ceil(2 * p.pw), Math.floor(GPU_POINTS / (4 * cols.length))));
      const span = p.vx1 - p.vx0, head = (p.ex1 - p.ex0) / 8;
      this.win = [Math.max(p.ex0, p.vx0 - span / 2), Math.min(p.ex1 + head, p.vx1 + span / 2)];
      if (!(this.win[1] > this.win[0])) this.win = [p.vx0, p.vx1];
    }
    this.ox = this.win ? this.win[0] : p.ex0;
    this.oy = NaN;
    for (const c of cols) {
      if (this.oy === this.oy) break;
      c.ensureSmooth(alpha, p.scale, p.xmode);
      const ys = c.ys(alpha, false);
      for (let i = 0; i < c.n; i++) if (Number.isFinite(ys[i]) && (!p.logy || ys[i] > 0)) {
        this.oy = p.logy ? Math.log10(ys[i]) : ys[i];
        break;
      }
    }
    if (!(this.oy === this.oy)) this.oy = 0;
    let bound = 0;
    for (const c of cols) bound += this.bound(c) + this.slack(c.n);
    staging = pointBuffer(staging, bound);
    let at = 0;
    for (const c of cols) {
      const n = this.convert(c, p, alpha, staging, at);
      const cap = n + this.slack(n);
      this.slots.set(c, { off: at, cap, n, ver: c.n });
      at += cap;
    }
    this.next = at;
    return this.pts.upload(staging, at, Math.ceil(at * 1.25) + 4096);
  }

  slack(n) {
    return this.win ? 16 + (n >> 4) : 16 + (n >> 3);
  }

  /** Upper bound on points written by convert. */
  bound(c) {
    return this.win ? Math.min(2 * c.n + 2, 4 * this.R + 1024) : c.n;
  }

  /** Writes column c's points (all, or decimated over the window) at `at`; returns the count. */
  convert(c, p, alpha, dst, at, from = 0) {
    c.ensureSmooth(alpha, p.scale, p.xmode);
    if (!this.win) {
      fillPoints(c.xs(p.xmode), 1, c.ys(alpha, false), 1, from, c.n, dst, at, this.ox, this.oy, p.logx, p.logy);
      return c.n - from;
    }
    const out = outBuf(4 * this.R + 1024);
    const flags = (p.logy ? LOGY : 0) | (p.logx ? LOGX : 0) | (alpha > 0 ? 0 : RAW);
    const r = kprep(c, p.xmode, this.win[0], this.win[1], this.R, flags, alpha, p.scale, out);
    fillPoints(out, 2, out.subarray(1), 2, 0, r.n, dst, at, this.ox, this.oy, p.logx, p.logy);
    return r.n;
  }

  /** Rewrite one column in place (or in a new slot at the end); false if a rebuild is needed. */
  update(c, p, alpha) {
    const s = this.slots.get(c);
    if (!this.win && s && c.n > s.ver && c.n <= s.cap) {
      const k = c.n - s.ver;
      scratchPts = pointBuffer(scratchPts, k);
      this.convert(c, p, alpha, scratchPts, 0, s.ver);
      this.pts.write(s.off + s.ver, scratchPts, k);
      s.n = s.ver = c.n;
      return true;
    }
    scratchPts = pointBuffer(scratchPts, this.bound(c));
    const n = this.convert(c, p, alpha, scratchPts, 0);
    if (s && n <= s.cap) {
      this.pts.write(s.off, scratchPts, n);
      s.n = n;
      s.ver = c.n;
      return true;
    }
    const cap = n + this.slack(n);
    if (this.next + cap > this.pts.cap) return false;
    if (s) this.garbage += s.cap;
    if (this.garbage > this.next / 2) return false;
    this.pts.write(this.next, scratchPts, n);
    this.slots.set(c, { off: this.next, cap, n, ver: c.n });
    this.next += cap;
    return true;
  }

  /** Line table for `lines` (each with cols[0] and color) in draw order. */
  tableFor(lines) {
    const t = this.table;
    t.clear();
    for (const ln of lines) {
      const s = this.slots.get(ln.cols[0]);
      t.push(s ? s.off : 0, s ? s.n : 0, ln.color);
    }
    return t;
  }

  release() {
    this.pts.release();
    this.slots.clear();
    this.key = null;
  }
}

/** View transform for points stored relative to (ox, oy). */
function glView(chart, v, ox, oy) {
  const dpr = devicePixelRatio || 1;
  return {
    off: [v.x0 - ox, v.y0 - oy],
    scale: [(chart.pw * dpr) / (v.x1 - v.x0), (chart.ph * dpr) / (v.y1 - v.y0)],
    org: [M.l * dpr, (M.t + chart.ph) * dpr],
  };
}

function isDark() {
  const c = rgba(getComputedStyle(document.documentElement).getPropertyValue("--bg") || "#fff");
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] < 128;
}

export class Chart {
  constructor(app, key) {
    this.app = app;
    this.key = key;
    this.dirty = true;
    this.visible = false;
    this.el = document.createElement("div");
    this.el.className = "panel";
    this.el.innerHTML = `<div class="ptitle"><span class="pname"></span><button class="pin" title="pin to the top">📌</button><button class="full" title="show this chart large (Esc to go back)">⛶</button><button class="gear" title="chart settings">⚙</button></div>
      <div class="pbody"><canvas></canvas><canvas class="ov"></canvas></div><div class="legend"></div>`;
    this.el.querySelector(".pname").textContent = key;
    this.gear = this.el.querySelector(".gear");
    this.gear.addEventListener("click", (e) => app.panelSettings(this, e.currentTarget));
    this.el.querySelector(".full").addEventListener("click", () => this.toggleFullscreen());
    this.pinBtn = this.el.querySelector(".pin");
    this.pinBtn.addEventListener("click", () => app.togglePin(this.key));
    this.body = this.el.querySelector(".pbody");
    this.legendEl = this.el.querySelector(".legend");
    [this.cv, this.ov] = this.el.querySelectorAll("canvas");
    this.el._chart = this;
    this.drag = null;
    this.yzoom = null; // [y0, y1] in data space from a box drag on this chart
    this.ov.addEventListener("mousemove", (e) => {
      this.hoverEvent = e;
      this.hoverRaf ||= requestAnimationFrame(() => {
        this.hoverRaf = 0;
        if (this.hoverEvent) this.hover(this.hoverEvent);
      });
    });
    this.ov.addEventListener("mouseleave", () => {
      this.hoverEvent = null;
      this.unhover();
    });
    this.ov.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      this.drag = { x: e.offsetX, y: e.offsetY };
      const up = (ev) => {
        window.removeEventListener("mouseup", up);
        this.endDrag(ev);
      };
      window.addEventListener("mouseup", up);
    });
    new ResizeObserver(() => this.resize()).observe(this.body);
  }

  setPinned(on) {
    this.pinBtn.classList.toggle("on", on);
    this.pinBtn.title = on ? "unpin" : "pin to the top";
  }

  /** Whether this chart is the focused chart (and so draws regardless of scroll visibility). */
  get full() {
    return this.app.opts.chart === this.key;
  }

  /** Focus this chart (or, when focused, go back to all charts). */
  toggleFullscreen() {
    this.app.focusChart(this.full ? "" : this.key);
  }

  resize() {
    const w = this.body.clientWidth, h = this.body.clientHeight;
    if (!w || !h) return;
    this.w = w;
    this.h = h;
    this.dirty = true;
    this.app.schedule(true);
  }

  /** Give the canvases backing stores of the chart's size (allocated only for drawn charts). */
  fitCanvases() {
    const dpr = devicePixelRatio || 1, W = Math.round(this.w * dpr), H = Math.round(this.h * dpr);
    for (const c of [this.cv, this.ov]) if (c.width !== W || c.height !== H) (c.width = W), (c.height = H);
  }

  /** Free the canvas backing stores of a chart scrolled out of view; it redraws when back. */
  releaseCanvases() {
    for (const c of [this.cv, this.ov]) if (c.width) (c.width = 0), (c.height = 0);
    this.dirty = true;
  }

  get pw() {
    return Math.max(10, this.w - M.l - M.r);
  }
  get ph() {
    return Math.max(10, this.h - M.t - M.b);
  }

  /** Query the kernel for everything this chart draws. */
  compute() {
    const app = this.app, o = app.panelOpts(this.key);
    const groups = app.linesFor(this.key), allCols = groups.flatMap((g) => g.cols);
    const v = this.xView(o, allCols);
    if (!v) return null;
    const yr = new YRange(o.logy), r = this.glRenderer();
    const gl = !app.grouped && r ? this.linesGL(r, groups, v, o, yr) : null;
    const lines = gl ? gl.lines : app.grouped ? this.linesGrouped(groups, allCols, v, o, yr) : this.linesCanvas(groups, v, yr);
    const y = this.yView(o, yr, allCols, v);
    if (!y) return null;
    return { ...v, ...y, lines, o, gl: !!r && (!!gl || app.grouped), gpu: !!gl, density: !!gl?.density };
  }

  /** x range (transformed) and smoothing of the view: the data's extent, or the zoom; null if empty. */
  xView(o, allCols) {
    const { xmode, logx, logy } = o;
    const [e0, e1, epos] = xExtent(allCols, xmode);
    if (!(e1 >= e0)) return null;
    const zoom = this.app.xrange && this.app.xrange[2] === xmode ? this.app.xrange : null;
    const [x0, x1] = xRange(o, zoom, e0, e1, epos);
    if (!(x1 > x0) || !Number.isFinite(x0)) return null;
    const t = (x) => (logx ? Math.log10(x) : x);
    return { x0: t(x0), x1: t(x1), ex0: t(logx ? epos : e0), ex1: t(e1), xmode, logx, logy, alpha: o.smooth,
             scale: smoothScale(e1 - e0), flags: (logy ? LOGY : 0) | (logx ? LOGX : 0) };
  }

  /** Lines drawn from the GPU line sets ({lines, density}), or null when the GPU cannot hold them. */
  linesGL(r, groups, v, o, yr) {
    const faint = v.alpha > 0;
    const density = r.density && (o.render === "density" || (o.render === "auto" && groups.length > DENSITY_AUTO));
    const p = { ...v, pw: this.pw, vx0: v.x0, vx1: v.x1 };
    const g = (this.glState ||= { main: new LineSet(r, false), faint: new LineSet(r, true), tmp: new Points(r) });
    const cols = groups.map((ln) => ln.cols[0]);
    const ok = g.main.sync(cols, p) && (!faint || density || g.faint.sync(cols, p));
    this.uploadMs = g.main.uploadMs + (faint && !density ? g.faint.uploadMs : 0);
    if (!ok) return null;
    for (const c of cols) for (const raw of faint ? [false, true] : [false]) growVisible(c, v, raw, yr);
    return { lines: groups.map((ln) => ({ ...ln })), density };
  }

  /** Lines decimated per pixel for Canvas 2D. */
  linesCanvas(groups, v, yr) {
    const out = outBuf(Math.ceil(this.pw) * 4 + 1024);
    const prep = (c, raw) => {
      const r = kprep(c, v.xmode, v.x0, v.x1, this.pw, v.flags | (raw ? RAW : 0), v.alpha, v.scale, out);
      if (r.ymin <= r.ymax) yr.add(r.ymin), yr.add(r.ymax);
      return out.slice(0, 2 * r.n);
    };
    return groups.map((g) => ({ ...g, raw: v.alpha > 0 ? prep(g.cols[0], true) : null, xy: prep(g.cols[0], false) }));
  }

  /** One center line with its band per group, from per-bin group statistics. */
  linesGrouped(groups, allCols, v, o, yr) {
    let dens = 0;
    for (const c of allCols) dens = Math.max(dens, c.len);
    const bins = Math.max(8, Math.min(600, Math.floor(this.pw / 2), dens)), dx = (v.x1 - v.x0) / bins;
    const stats = (cols, raw) => {
      const a = kagg(cols, v.xmode, v.x0, v.x1, bins, (v.logx ? LOGX : 0) | (raw ? RAW : 0), v.alpha, v.scale);
      return Object.fromEntries(STATS.map((k, i) => [k, a.subarray(i * bins, (i + 1) * bins)]));
    };
    const centerXY = (center) => {
      const xy = new Float64Array(2 * bins);
      for (let i = 0; i < bins; i++) {
        const kx = v.x0 + (i + 0.5) * dx;
        xy[2 * i] = v.logx ? 10 ** kx : kx;
        xy[2 * i + 1] = v.logy && !(center[i] > 0) ? NaN : center[i];
      }
      return xy;
    };
    return groups.map((g) => {
      const st = stats(g.cols, false), center = o.center === "mean" ? st.mean : st.median;
      const [lo, hi] = bandOf(st, o.center, o.band, bins);
      for (let i = 0; i < bins; i++) {
        if (v.logy && !(lo[i] > 0)) lo[i] = center[i];
        yr.add(center[i]);
        if (o.band !== "none") yr.add(lo[i]), yr.add(hi[i]);
      }
      let raw = null;
      if (v.alpha > 0) {
        const rs = stats(g.cols, true), rc = o.center === "mean" ? rs.mean : rs.median;
        rc.forEach((y) => yr.add(y));
        raw = centerXY(rc);
      }
      return { ...g, xy: centerXY(center), raw, lo, hi, center, cnt: st.n, dx };
    });
  }

  /** [y0, y1] in data units: the lines' range or outlier quantiles, overridden by settings, then the box zoom. */
  yBounds(o, yr, allCols, v) {
    let { lo: y0, hi: y1 } = yr;
    const q = o.outliers > 0 && allCols.length ? yrange(allCols, v.xmode, v.x0, v.x1, v.flags, v.alpha, v.scale, o.outliers, 1 - o.outliers) : null;
    if (q) [y0, y1] = q;
    return this.yzoom || [o.ymin ?? y0, o.ymax ?? y1];
  }

  /** y range (transformed) of the view; null if empty. */
  yView(o, yr, allCols, v) {
    let [y0, y1] = this.yBounds(o, yr, allCols, v);
    if (!(y1 >= y0) || (o.logy && !(y0 > 0))) {
      if (!(o.logy && y1 > 0)) return null;
      y0 = y1 / 10;
    }
    if (o.logy) (y0 = Math.log10(y0)), (y1 = Math.log10(y1));
    if (y1 === y0) (y0 -= 0.5), (y1 += 0.5);
    const pad = (y1 - y0) * 0.04;
    return { y0: this.yzoom || o.ymin != null ? y0 : y0 - pad, y1: this.yzoom || o.ymax != null ? y1 : y1 + pad };
  }


  /** The WebGL renderer when selected and available, else null (Canvas 2D). */
  glRenderer() {
    return USE_GL ? renderer() : null;
  }

  /** Free this chart's GPU data. */
  dispose() {
    const g = this.glState;
    if (g) g.main.release(), g.faint.release(), g.tmp.release();
    this.glState = null;
  }

  /** Draw bands and lines of `view` through WebGL onto ctx (device px, clipped to the plot). */
  drawGL(ctx, view) {
    const r = renderer(), dpr = devicePixelRatio || 1;
    r.begin(this.cv.width, this.cv.height, [M.l * dpr, M.t * dpr, this.pw * dpr, this.ph * dpr]);
    const g = (this.glState ||= { main: new LineSet(r, false), faint: new LineSet(r, true), tmp: new Points(r) });
    if (view.gpu) {
      const { main, faint } = g;
      r.touch(main.pts);
      if (view.density) r.density(main.pts, main.tableFor(view.lines), glView(this, view, main.ox, main.oy), dpr, isDark());
      else {
        if (view.alpha > 0) {
          r.touch(faint.pts);
          r.lines(faint.pts, faint.tableFor(view.lines), glView(this, view, faint.ox, faint.oy), dpr, 0.22);
        }
        r.lines(main.pts, main.tableFor(view.lines), glView(this, view, main.ox, main.oy), 1.25 * dpr, 1);
      }
      r.trim(new Set([main.pts, faint.pts]));
    } else this.drawGroupsGL(r, g.tmp, view, dpr);
    r.copyTo(ctx);
  }

  /** Group bands, faint raw centers and center lines, uploaded per draw. */
  drawGroupsGL(r, pts, view, dpr) {
    const { logx, logy } = view, ox = view.x0, oy = view.y0, band = view.o.band !== "none";
    let n = 0;
    for (const ln of view.lines) n += (ln.xy.length >> 1) * (band && ln.lo ? 4 : 2);
    staging = pointBuffer(staging, n);
    const nl = view.lines.length, lines = new Table(nl), raws = new Table(nl), bands = new Table(2 * nl);
    let at = 0;
    const put = (xs, xstep, ys, ystep, len) => {
      fillPoints(xs, xstep, ys, ystep, 0, len, staging, at, ox, oy, logx, logy);
      at += len;
      return at - len;
    };
    for (const ln of view.lines) {
      const k = ln.xy.length >> 1;
      lines.push(put(ln.xy, 2, ln.xy.subarray(1), 2, k), k, ln.color);
      if (ln.raw) raws.push(put(ln.raw, 2, ln.raw.subarray(1), 2, k), k, ln.color);
      if (band && ln.lo) {
        bands.push(put(ln.xy, 2, ln.hi, 1, k), k, ln.color);
        bands.push(put(ln.xy, 2, ln.lo, 1, k), k, ln.color);
      }
    }
    if (!pts.upload(staging, at, at)) return;
    r.touch(pts);
    const tv = glView(this, view, ox, oy);
    if (bands.n) r.bands(pts, bands, tv, 0.18);
    if (raws.n) r.lines(pts, raws, tv, dpr, 0.22);
    r.lines(pts, lines, tv, 2 * dpr, 1);
  }

  px(x) {
    const v = this.view;
    const fx = v.logx ? Math.log10(x) : x;
    return M.l + ((fx - v.x0) / (v.x1 - v.x0)) * this.pw;
  }
  py(y) {
    const v = this.view;
    const fy = v.logy ? Math.log10(y) : y;
    return M.t + this.ph - ((fy - v.y0) / (v.y1 - v.y0)) * this.ph;
  }
  /** Data-space y at a canvas pixel. */
  yAt(py) {
    const v = this.view;
    const fy = v.y0 + ((M.t + this.ph - py) / this.ph) * (v.y1 - v.y0);
    return v.logy ? 10 ** fy : fy;
  }

  /** Data-space x at a canvas pixel. */
  xAt(px) {
    const v = this.view;
    const fx = v.x0 + ((px - M.l) / this.pw) * (v.x1 - v.x0);
    return v.logx ? 10 ** fx : fx;
  }

  draw() {
    this.dirty = false;
    this.uploadMs = 0;
    this.gear.classList.toggle("on", this.app.hasPanelOverrides(this.key));
    if (!this.w) return;
    this.fitCanvases();
    const dpr = devicePixelRatio || 1, ctx = this.cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    const view = (this.view = this.compute());
    this.renderLegend(view);
    const css = getComputedStyle(document.documentElement);
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillStyle = css.getPropertyValue("--muted");
    if (!view) return ctx.fillText("no data", M.l + 4, M.t + 14);
    this.drawAxes(ctx, view, css.getPropertyValue("--grid"));
    if (view.gl) return this.drawGL(ctx, view);
    ctx.save();
    ctx.beginPath();
    ctx.rect(M.l, M.t, this.pw, this.ph);
    ctx.clip();
    if (view.o.band !== "none") for (const ln of view.lines) if (ln.lo) this.drawBand(ctx, ln);
    ctx.lineJoin = "round";
    for (const ln of view.lines) if (ln.raw) this.stroke(ctx, ln.raw, ln.color, 1, 0.22);
    for (const ln of view.lines) this.stroke(ctx, ln.xy, ln.color, this.app.grouped ? 2 : 1.25, 1);
    ctx.restore();
  }

  /** Grid lines and tick labels. */
  drawAxes(ctx, view, grid) {
    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const ny = Math.max(2, Math.floor(this.ph / 28));
    for (const t of view.logy ? logTicks(view.y0, view.y1, ny) : niceTicks(view.y0, view.y1, ny)) {
      const y = Math.round(this.py(view.logy ? 10 ** t : t)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(M.l, y);
      ctx.lineTo(M.l + this.pw, y);
      ctx.stroke();
      ctx.fillText(fmt(view.logy ? +(10 ** t).toPrecision(6) : t), M.l - 4, y);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const nx = Math.max(2, Math.floor(this.pw / 80));
    const xt = view.logx ? logTicks(view.x0, view.x1, nx).map((d) => +(10 ** d).toPrecision(6))
      : view.xmode === 1 ? durTicks(view.x0, view.x1, nx) : niceTicks(view.x0, view.x1, nx);
    for (const t of xt) ctx.fillText(view.xmode === 1 ? fmtDur(t) : fmtSI(t), this.px(t), M.t + this.ph + 4);
  }

  stroke(ctx, xy, color, width, alpha) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.globalAlpha = alpha;
    this.polyline(ctx, xy, xy.length >> 1);
    ctx.globalAlpha = 1;
  }

  /** Stroke the first n (x, y) pairs of xy (data units); NaN y breaks the line. */
  polyline(ctx, xy, n) {
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < n; i++) {
      const y = xy[2 * i + 1];
      if (y !== y) {
        pen = false;
        continue;
      }
      const X = this.px(xy[2 * i]), Y = this.py(y);
      if (pen) ctx.lineTo(X, Y);
      else ctx.moveTo(X, Y), (pen = true);
    }
    ctx.stroke();
  }


  renderLegend(view) {
    const lines = view ? view.lines : [];
    const density = !!view?.density;
    const sig = density + "#" + lines.map((l) => l.color + l.label).join("|");
    if (sig === this.legendSig) return;
    this.legendSig = sig;
    if (density || lines.length > (this.app.grouped ? 2 * MAX_LEGEND : MAX_LEGEND)) {
      const text = density ? `${lines.length} lines · density (log scale) · hover for nearest runs` : `${lines.length} lines · hover for values`;
      this.legendEl.replaceChildren(Object.assign(document.createElement("span"), { textContent: text }));
      return;
    }
    const items = lines.map((l) => {
      const s = document.createElement("span");
      s.innerHTML = `<i style="background:${l.color}"></i>`;
      s.append(l.label);
      return s;
    });
    this.legendEl.replaceChildren(...items);
  }

  drawBand(ctx, ln) {
    ctx.fillStyle = ln.color;
    ctx.globalAlpha = 0.18;
    const n = ln.lo.length, xy = ln.xy;
    let i = 0;
    while (i < n) {
      while (i < n && !(Number.isFinite(ln.lo[i]) && Number.isFinite(ln.hi[i]))) i++;
      const s = i;
      while (i < n && Number.isFinite(ln.lo[i]) && Number.isFinite(ln.hi[i])) i++;
      if (i - s < 1) continue;
      ctx.beginPath();
      for (let j = s; j < i; j++) ctx.lineTo(this.px(xy[2 * j]), this.py(ln.hi[j]));
      for (let j = i - 1; j >= s; j--) ctx.lineTo(this.px(xy[2 * j]), this.py(ln.lo[j]));
      ctx.closePath();
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  hover(e) {
    const v = this.view;
    if (!v || this.app.tipPinned) return;
    this.app.hovered = this;
    this.lastX = e.offsetX;
    const dpr = devicePixelRatio || 1, ctx = this.ov.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    if (this.drag) return this.drawDragBox(ctx, e);
    if (e.offsetX < M.l || e.offsetX > M.l + this.pw) return this.unhover();
    const x = this.xAt(e.offsetX);
    this.crosshair(ctx, e.offsetX);
    if (v.density) return this.hoverDensity(e, ctx, x);
    const rows = [];
    for (const ln of v.lines) {
      const row = ln.lo ? this.groupRow(ln, x) : this.lineRow(ln, x, e.offsetX);
      if (!row) continue;
      row.py = Number.isFinite(row.val) && (!v.logy || row.val > 0) ? this.py(row.val) : NaN;
      if (row.py === row.py) dot(ctx, row.px, row.py, ln.color);
      rows.push(row);
    }
    const near = this.nearestRow(rows, e.offsetY);
    rows.sort(byValue);
    this.app.tip(e, this.key, xLabel(v, x), rows, rows.indexOf(near));
  }

  crosshair(ctx, px) {
    ctx.strokeStyle = "rgba(127,127,127,0.6)";
    ctx.beginPath();
    ctx.moveTo(px + 0.5, M.t);
    ctx.lineTo(px + 0.5, M.t + this.ph);
    ctx.stroke();
  }

  drawDragBox(ctx, e) {
    const d = this.drag, w = Math.abs(e.offsetX - d.x), h = Math.abs(e.offsetY - d.y);
    ctx.fillStyle = "rgba(127,127,127,0.2)";
    if (h >= BOX_MIN) ctx.fillRect(Math.min(d.x, e.offsetX), Math.min(d.y, e.offsetY), w, h);
    else ctx.fillRect(Math.min(d.x, e.offsetX), M.t, w, this.ph);
  }

  /** Tooltip row of a group line at x: its center, band and count in that bin. */
  groupRow(ln, x) {
    const v = this.view, fx = v.logx ? Math.log10(x) : x;
    const i = Math.min(ln.center.length - 1, Math.max(0, Math.floor((fx - v.x0) / ln.dx))), val = ln.center[i];
    if (!Number.isFinite(val)) return null;
    let extra = v.o.band !== "none" ? ` ${bandLabel(v.o.band, v.o.center, ln.cnt[i])} [${fmt(ln.lo[i])}, ${fmt(ln.hi[i])}]` : "";
    extra += ` n=${ln.cnt[i]}`;
    if (ln.raw && Number.isFinite(ln.raw[2 * i + 1])) extra += ` (raw ${fmt(ln.raw[2 * i + 1])})`;
    return { ln, val, extra, px: this.px(ln.xy[2 * i]) };
  }

  /** Tooltip row of a run line: its point nearest x, if within 40 px of the pointer. */
  lineRow(ln, x, offsetX) {
    const v = this.view, r = nearest(ln.cols[0], v.xmode, x, v.alpha, v.scale);
    if (!r || Math.abs(this.px(r.x) - offsetX) > 40) return null;
    return { ln, val: r.y, extra: v.alpha > 0 ? ` (raw ${fmt(r.raw)})` : "", px: this.px(r.x) };
  }


  /** The row whose point is vertically nearest the pointer, ringed on the overlay. */
  nearestRow(rows, y) {
    let best = null;
    for (const r of rows) if (Math.abs(r.py - y) < Math.abs((best?.py ?? Infinity) - y)) best = r;
    if (best) {
      const ctx = this.ov.getContext("2d");
      ctx.strokeStyle = best.ln.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(best.px, best.py, 6, 0, 7);
      ctx.stroke();
    }
    return best;
  }

  /** Density tooltip: the runs whose value at x is nearest the pointer, the nearest one traced. */
  hoverDensity(e, ctx, x) {
    const v = this.view, near = [];
    for (const ln of v.lines) {
      const r = nearest(ln.cols[0], v.xmode, x, v.alpha, v.scale);
      if (!r || !Number.isFinite(r.y) || (v.logy && !(r.y > 0)) || Math.abs(this.px(r.x) - e.offsetX) > 40) continue;
      const d = Math.abs(this.py(r.y) - e.offsetY);
      if (near.length === DENSITY_TIP && d >= near[DENSITY_TIP - 1].d) continue;
      let i = near.length === DENSITY_TIP ? DENSITY_TIP - 1 : near.length;
      while (i > 0 && near[i - 1].d > d) (near[i] = near[i - 1]), i--;
      near[i] = { d, ln, r };
    }
    if (near.length) this.traceLine(ctx, near[0].ln, 1.5);
    for (const { ln, r } of near) dot(ctx, this.px(r.x), this.py(r.y), ln.color);
    const rows = near.map(({ ln, r }) => ({ ln, val: r.y, extra: v.alpha > 0 ? ` (raw ${fmt(r.raw)})` : "",
                                           px: this.px(r.x), py: this.py(r.y) }));
    const closest = this.nearestRow(rows, e.offsetY);
    rows.sort(byValue);
    this.app.tip(e, this.key, `${xLabel(v, x)} · ${near.length} nearest of ${v.lines.length}`, rows, rows.indexOf(closest));
  }

  /** Stroke one line of the current view on the overlay: a run's own line, or a group's center. */
  traceLine(ctx, ln, width) {
    const v = this.view;
    let xy, n;
    if (ln.lo) (xy = ln.xy), (n = ln.xy.length >> 1);
    else {
      xy = outBuf(Math.ceil(this.pw) * 4 + 1024);
      const flags = (v.logy ? LOGY : 0) | (v.logx ? LOGX : 0);
      n = kprep(ln.cols[0], v.xmode, v.x0, v.x1, this.pw, flags, v.alpha, v.scale, xy).n;
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(M.l, M.t, this.pw, this.ph);
    ctx.clip();
    ctx.strokeStyle = ln.color;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    this.polyline(ctx, xy, n);
    ctx.restore();
  }

  /** Overlay showing the pinned crosshair with one line emphasized (null: none). */
  highlight(ln) {
    if (!this.view || !this.ov.width) return;
    const dpr = devicePixelRatio || 1, ctx = this.ov.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    ctx.strokeStyle = "rgba(127,127,127,0.6)";
    ctx.beginPath();
    ctx.moveTo(this.lastX + 0.5, M.t);
    ctx.lineTo(this.lastX + 0.5, M.t + this.ph);
    ctx.stroke();
    if (ln) this.traceLine(ctx, ln, 3);
  }

  unhover() {
    if (this.app.hovered === this) this.app.hovered = null;
    if (this.app.tipPinned) return;
    if (this.ov.width) this.ov.getContext("2d").clearRect(0, 0, this.ov.width, this.ov.height);
    this.app.tip(null);
  }

  /** A click resets axes; a drag zooms x (all charts); a box also zooms this chart's y. */
  endDrag(e) {
    const d = this.drag;
    this.drag = null;
    if (!d || !this.view) return;
    const r = this.ov.getBoundingClientRect();
    const ox = Math.min(Math.max(e.clientX - r.left, M.l), M.l + this.pw);
    const oy = Math.min(Math.max(e.clientY - r.top, M.t), M.t + this.ph);
    const dx = Math.abs(ox - d.x), dy = Math.abs(oy - d.y);
    this.unhover();
    if (dx < CLICK_MAX && dy < CLICK_MAX) return this.resetAxes();
    if (dy >= BOX_MIN) this.yzoom = [this.yAt(Math.max(d.y, oy)), this.yAt(Math.min(d.y, oy))];
    if (dx >= CLICK_MAX) this.app.setXRange([this.xAt(Math.min(d.x, ox)), this.xAt(Math.max(d.x, ox)), this.view.xmode]);
    else {
      this.dirty = true;
      this.app.schedule(true);
    }
    this.app.updateZoomButton();
  }

  /** Back to automatic axes: this chart's y, and the shared x zoom. */
  resetAxes() {
    this.yzoom = null;
    if (this.app.xrange) this.app.setXRange(null);
    else {
      this.dirty = true;
      this.app.schedule(true);
    }
    this.app.updateZoomButton();
  }
}
